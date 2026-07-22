#!/usr/bin/env python3
"""
analyze_java_template_repositories.py

Select the unique Java template repositories associated with eligible generated
repositories, clone each template, compile it, run SonarQube analysis, collect
project metrics and vulnerability details, and save failures separately.

Expected input:
    sample/persistence_sample_balanced/Java/eligible_candidates.csv

Required input column:
    target_template_repository

Example value:
    adempiere/adempiere

The script also accepts complete GitHub URLs, such as:
    https://github.com/adempiere/adempiere
    https://github.com/adempiere/adempiere.git

The generated-repository rows are filtered using:
    commit_count >= 2
    is_template == False
    data_update >= 2023-10-18

After filtering, the script extracts and deduplicates the associated template
repositories.

Required environment variable:
    SONAR_TOKEN

Optional environment variables:
    SONAR_URL
    SONAR_SCANNER

Example:
    export SONAR_TOKEN="your-token"
    export SONAR_URL="http://localhost:9002"

    python analyze_java_template_repositories.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse

import pandas as pd
import requests

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROGRAMMING_LANGUAGE = "Java"

SONAR_URL = os.getenv(
    "SONAR_URL",
    "http://localhost:9002",
).rstrip("/")

SONAR_TOKEN = "squ_f75043d923b1b7855a7c6e44ba031a9997df5881"

SONAR_SCANNER = Path(
    os.getenv(
        "SONAR_SCANNER",
        "/opt/sonar-scanner/bin/sonar-scanner",
    )
)

TEMPLATE_COLUMN = "target_template_repository"

# Keep template clones separate from generated-repository clones.
REPO_DIR = Path("temp_template_repos")

SAMPLE_ROOT = Path("sample/persistence_sample_balanced")
INPUT_CSV = (
    SAMPLE_ROOT
    / PROGRAMMING_LANGUAGE
    / "eligible_candidates.csv"
)

# Keep template results separate from generated-repository results.
OUTPUT_ROOT = (
    Path("vulnerability_issues")
    / PROGRAMMING_LANGUAGE
)

OUTPUT_CSV = (
    OUTPUT_ROOT
    / f"vulnerability_results_templates_{PROGRAMMING_LANGUAGE}.csv"
)

FAILURES_CSV = (
    OUTPUT_ROOT
    / f"analysis_failures_templates_{PROGRAMMING_LANGUAGE}.csv"
)

JSON_OUTPUT_DIR = OUTPUT_ROOT / "vulnerability_details"
LOG_DIR = OUTPUT_ROOT / "logs"

START_DATE = pd.Timestamp("2023-10-18", tz="UTC")

# Equivalent to commit_count > 1.
MIN_COMMIT_COUNT = 2

METRICS = [
    "ncloc",
    "files",
    "functions",
    "vulnerabilities",
    "security_hotspots",
    "code_smells",
    "bugs",
]

SEVERITIES = [
    "BLOCKER",
    "CRITICAL",
    "MAJOR",
    "MINOR",
    "INFO",
]

COMMAND_TIMEOUT_SECONDS = 20 * 60
CE_TIMEOUT_SECONDS = 5 * 60
CE_POLL_INTERVAL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 60

# Initial SonarQube execution plus this number of retries.
MAX_SONAR_RETRIES = 1

SKIP_WALK_DIRS = {
    ".git",
    ".scannerwork",
    ".gradle",
    ".idea",
    ".vscode",
    "node_modules",
    "target",
    "build",
    "dist",
    "coverage",
}

BASE_SONAR_EXCLUSIONS = [
    ".git/**",
    ".scannerwork/**",
    ".github/workflows/**",
    "**/node_modules/**",
    "**/target/**",
    "**/build/**",
    "**/dist/**",
    "**/coverage/**",
]

for directory in [
    REPO_DIR,
    OUTPUT_ROOT,
    JSON_OUTPUT_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()

if SONAR_TOKEN:
    SESSION.auth = (SONAR_TOKEN, "")

# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class CommandResult:
    ok: bool
    returncode: int
    output: str
    timed_out: bool = False


@dataclass
class TemplateRepository:
    """
    Normalized representation of a template repository.

    associated_generated_repositories indicates how many filtered rows in the
    input CSV reference this template.
    """

    owner: str
    repo_name: str
    slug: str
    canonical_slug: str
    clone_url: str
    associated_generated_repositories: int


@dataclass
class BuildResult:
    ok: bool
    build_system: str
    build_root: Optional[Path]
    binaries: list[Path]
    output: str
    reason: str


@dataclass
class ScanResult:
    ok: bool
    output: str
    excluded_files: list[str]
    reason: str


# -----------------------------------------------------------------------------
# Command execution
# -----------------------------------------------------------------------------


def _safe_command_text(command: Iterable[str]) -> str:
    """Return a printable command with secrets masked."""
    safe_parts: list[str] = []

    for part in command:
        text = str(part)

        text = re.sub(
            r"(?i)(sonar\.(?:token|login)=)[^\s]+",
            r"\1***",
            text,
        )

        safe_parts.append(text)

    return " ".join(safe_parts)


def run_cmd(
    command: list[str],
    *,
    cwd: Optional[Path] = None,
    timeout: int = COMMAND_TIMEOUT_SECONDS,
    env: Optional[dict[str, str]] = None,
) -> CommandResult:
    """
    Run a command, capture stdout and stderr together, and never raise when the
    command returns a nonzero exit status.
    """
    print(f"Running: {_safe_command_text(command)}")

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )

        output = completed.stdout or ""
        print(output)

        return CommandResult(
            ok=completed.returncode == 0,
            returncode=completed.returncode,
            output=output,
        )

    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""

        if isinstance(output, bytes):
            output = output.decode(
                "utf-8",
                errors="replace",
            )

        output += (
            f"\nERROR: Command timed out after "
            f"{timeout} seconds.\n"
        )

        print(output)

        return CommandResult(
            ok=False,
            returncode=124,
            output=output,
            timed_out=True,
        )

    except FileNotFoundError as exc:
        output = f"ERROR: Command not found: {exc}\n"
        print(output)

        return CommandResult(
            ok=False,
            returncode=127,
            output=output,
        )


def write_log(
    repo_id: str,
    stage: str,
    content: str,
) -> Path:
    """Write a build, clone, or SonarQube log."""
    log_path = LOG_DIR / f"{repo_id}_{stage}.log"

    log_path.write_text(
        content or "",
        encoding="utf-8",
        errors="replace",
    )

    return log_path


# -----------------------------------------------------------------------------
# Repository-reference handling
# -----------------------------------------------------------------------------


def parse_repository_reference(
    repository_reference: str,
) -> tuple[str, str]:
    """
    Extract the owner and repository name from a repository reference.

    Supported formats:

        owner/repository

        https://github.com/owner/repository
        https://github.com/owner/repository.git

        http://github.com/owner/repository

        github.com/owner/repository

        git@github.com:owner/repository.git

        ssh://git@github.com/owner/repository.git
    """
    value = str(repository_reference).strip()

    if not value:
        raise ValueError("Empty repository reference.")

    repository_path: str

    # GitHub SSH shorthand:
    # git@github.com:owner/repository.git
    if value.lower().startswith("git@github.com:"):
        repository_path = value.split(":", 1)[1]

    # Normal URL, including ssh:// URLs.
    elif "://" in value:
        parsed = urlparse(value)

        hostname = (parsed.hostname or "").lower()

        if hostname not in {
            "github.com",
            "www.github.com",
        }:
            raise ValueError(
                "Only GitHub repositories are supported: "
                f"{repository_reference}"
            )

        repository_path = parsed.path.strip("/")

    else:
        repository_path = value.strip("/")

        if repository_path.lower().startswith("github.com/"):
            repository_path = repository_path[len("github.com/") :]

        elif repository_path.lower().startswith(
            "www.github.com/"
        ):
            repository_path = repository_path[len("www.github.com/") :]

    # Remove possible query parameters and fragments.
    repository_path = repository_path.split("?", 1)[0]
    repository_path = repository_path.split("#", 1)[0]

    # Remove a trailing .git suffix.
    repository_path = re.sub(
        r"\.git$",
        "",
        repository_path,
        flags=re.IGNORECASE,
    )

    parts = [
        part.strip()
        for part in repository_path.split("/")
        if part.strip()
    ]

    if len(parts) != 2:
        raise ValueError(
            "Repository reference must have the format "
            f"'owner/repository': {repository_reference}"
        )

    owner, repo_name = parts

    if not owner or not repo_name:
        raise ValueError(
            f"Invalid repository reference: {repository_reference}"
        )

    return owner, repo_name


def build_github_clone_url(
    owner: str,
    repo_name: str,
) -> str:
    """Create a canonical HTTPS clone URL."""
    return f"https://github.com/{owner}/{repo_name}.git"


def safe_identifier(value: str) -> str:
    """Create a filesystem-safe identifier."""
    return re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    ).strip("_")


def compute_project_key(
    owner: str,
    repo_name: str,
) -> str:
    """
    Compute the SonarQube project key.

    Lowercasing ensures that GitHub capitalization differences do not produce
    separate SonarQube projects.
    """
    return f"{owner}__{repo_name}".lower()


def compute_project_key_from_reference(
    repository_reference: str,
) -> str:
    owner, repo_name = parse_repository_reference(
        repository_reference
    )

    return compute_project_key(
        owner,
        repo_name,
    )


def canonical_repository_slug(
    repository_reference: str,
) -> str:
    """Return a lowercase owner/repository identifier."""
    owner, repo_name = parse_repository_reference(
        repository_reference
    )

    return f"{owner}/{repo_name}".lower()


def _parse_boolean(value) -> Optional[bool]:
    """Parse common textual and numeric Boolean representations."""
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "y",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "n",
    }:
        return False

    return None


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------


def load_template_repositories() -> list[TemplateRepository]:
    """
    Select eligible generated-repository rows and extract their associated
    template repositories.

    Multiple generated repositories can reference the same template. Templates
    are therefore deduplicated case-insensitively.

    For example:

        target_template_repository = adempiere/adempiere

    becomes:

        https://github.com/adempiere/adempiere.git
    """
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(
        INPUT_CSV,
        on_bad_lines="skip",
    )

    required_columns = {
        TEMPLATE_COLUMN,
        "commit_count",
        "is_template",
        "data_update",
    }

    missing = sorted(
        required_columns - set(df.columns)
    )

    if missing:
        raise KeyError(
            f"Missing required input columns: {missing}"
        )

    df["commit_count"] = pd.to_numeric(
        df["commit_count"],
        errors="coerce",
    )

    df["data_update"] = pd.to_datetime(
        df["data_update"],
        errors="coerce",
        utc=True,
    )

    parsed_is_template = df["is_template"].map(
        _parse_boolean
    )

    # These conditions apply to the generated repository rows.
    mask = (
        (df["commit_count"] >= MIN_COMMIT_COUNT)
        & (
            parsed_is_template == False  # noqa: E712
        )
        & (df["data_update"] >= START_DATE)
    )

    selected_df = df.loc[
        mask,
        [TEMPLATE_COLUMN],
    ].copy()

    selected_df[TEMPLATE_COLUMN] = (
        selected_df[TEMPLATE_COLUMN]
        .astype("string")
        .str.strip()
    )

    selected_df = selected_df.loc[
        selected_df[TEMPLATE_COLUMN].notna()
        & (
            selected_df[TEMPLATE_COLUMN]
            != ""
        )
    ]

    templates_by_canonical_slug: dict[
        str,
        TemplateRepository,
    ] = {}

    invalid_references: list[str] = []

    for raw_reference in selected_df[TEMPLATE_COLUMN]:
        repository_reference = str(raw_reference).strip()

        try:
            owner, repo_name = parse_repository_reference(
                repository_reference
            )

        except ValueError as exc:
            invalid_references.append(repository_reference)

            print(
                "⚠️ Ignoring invalid "
                f"{TEMPLATE_COLUMN} value "
                f"'{repository_reference}': {exc}"
            )

            continue

        slug = f"{owner}/{repo_name}"
        canonical_slug = slug.lower()

        existing = templates_by_canonical_slug.get(
            canonical_slug
        )

        if existing is not None:
            existing.associated_generated_repositories += 1
            continue

        templates_by_canonical_slug[canonical_slug] = (
            TemplateRepository(
                owner=owner,
                repo_name=repo_name,
                slug=slug,
                canonical_slug=canonical_slug,
                clone_url=build_github_clone_url(
                    owner,
                    repo_name,
                ),
                associated_generated_repositories=1,
            )
        )

    templates = list(
        templates_by_canonical_slug.values()
    )

    print(
        "📥 Total input rows: "
        f"{len(df)}"
    )

    print(
        "📥 Eligible generated-repository rows: "
        f"{int(mask.sum())}"
    )

    print(
        "🔗 Eligible rows with a template reference: "
        f"{len(selected_df)}"
    )

    print(
        "🧩 Unique associated template repositories: "
        f"{len(templates)}"
    )

    if invalid_references:
        print(
            "⚠️ Invalid template repository references: "
            f"{len(invalid_references)}"
        )

    return templates


def project_contains_java(
    repo_path: Path,
) -> bool:
    """Return True when the cloned repository contains a Java source file."""
    for _, dirs, files in os.walk(repo_path):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in SKIP_WALK_DIRS
        ]

        if any(
            filename.lower().endswith(".java")
            for filename in files
        ):
            return True

    return False


# -----------------------------------------------------------------------------
# Java build handling
# -----------------------------------------------------------------------------


def _path_depth(
    path: Path,
    base: Path,
) -> int:
    return len(
        path.resolve()
        .relative_to(base.resolve())
        .parts
    )


def find_build_candidates(
    repo_path: Path,
) -> list[tuple[str, Path]]:
    """
    Return Maven and Gradle build roots.

    Root-level build definitions are preferred. When there is no root-level
    Maven or Gradle build, nested build definitions up to four directory levels
    below the repository root are considered.
    """
    candidates: list[tuple[str, Path]] = []

    root_has_maven = (
        repo_path / "pom.xml"
    ).is_file()

    root_has_gradle = any(
        (repo_path / filename).is_file()
        for filename in [
            "gradlew",
            "build.gradle",
            "build.gradle.kts",
            "settings.gradle",
            "settings.gradle.kts",
        ]
    )

    if root_has_maven:
        candidates.append(
            ("maven", repo_path)
        )

    if root_has_gradle:
        candidates.append(
            ("gradle", repo_path)
        )

    # Prefer the root build when one exists.
    if candidates:
        return candidates

    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)
        depth = _path_depth(
            root_path,
            repo_path,
        )

        dirs[:] = [
            directory
            for directory in dirs
            if (
                directory not in SKIP_WALK_DIRS
                and depth < 4
            )
        ]

        file_set = set(files)

        if "pom.xml" in file_set:
            candidates.append(
                ("maven", root_path)
            )

        if file_set.intersection(
            {
                "gradlew",
                "build.gradle",
                "build.gradle.kts",
                "settings.gradle",
                "settings.gradle.kts",
            }
        ):
            candidates.append(
                ("gradle", root_path)
            )

    # Deduplicate build roots.
    unique: dict[
        tuple[str, str],
        tuple[str, Path],
    ] = {}

    for system, root in candidates:
        key = (
            system,
            str(root.resolve()),
        )

        unique[key] = (
            system,
            root,
        )

    return sorted(
        unique.values(),
        key=lambda item: (
            _path_depth(
                item[1],
                repo_path,
            ),
            item[0],
        ),
    )


def _command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def run_maven_compile(
    build_root: Path,
) -> CommandResult:
    """Compile a Maven project without executing tests."""
    wrapper = build_root / "mvnw"

    if wrapper.is_file():
        wrapper.chmod(
            wrapper.stat().st_mode | 0o111
        )
        executable = "./mvnw"

    elif _command_exists("mvn"):
        executable = "mvn"

    else:
        return CommandResult(
            ok=False,
            returncode=127,
            output=(
                "Maven wrapper not found and system "
                "'mvn' is unavailable.\n"
            ),
        )

    command = [
        executable,
        "-B",
        "-DskipTests",
        "-DskipITs",
        "-Dmaven.javadoc.skip=true",
        "-Dcheckstyle.skip=true",
        "-Dpmd.skip=true",
        "-Dspotbugs.skip=true",
        "compile",
    ]

    return run_cmd(
        command,
        cwd=build_root,
    )


def run_gradle_compile(
    build_root: Path,
) -> CommandResult:
    """Compile a Gradle project without executing tests."""
    wrapper = build_root / "gradlew"

    if wrapper.is_file():
        wrapper.chmod(
            wrapper.stat().st_mode | 0o111
        )
        executable = "./gradlew"

    elif _command_exists("gradle"):
        executable = "gradle"

    else:
        return CommandResult(
            ok=False,
            returncode=127,
            output=(
                "Gradle wrapper not found and system "
                "'gradle' is unavailable.\n"
            ),
        )

    first = run_cmd(
        [
            executable,
            "classes",
            "-x",
            "test",
            "--no-daemon",
        ],
        cwd=build_root,
    )

    if first.ok:
        return first

    print(
        "🔁 Gradle 'classes' failed. "
        "Trying 'assemble -x test'..."
    )

    second = run_cmd(
        [
            executable,
            "assemble",
            "-x",
            "test",
            "--no-daemon",
        ],
        cwd=build_root,
    )

    second.output = (
        first.output
        + "\n\n--- GRADLE FALLBACK ---\n\n"
        + second.output
    )

    return second


def _find_subsequence(
    parts: tuple[str, ...],
    sequence: tuple[str, ...],
) -> int:
    if len(sequence) > len(parts):
        return -1

    for index in range(
        len(parts) - len(sequence) + 1
    ):
        if (
            parts[index : index + len(sequence)]
            == sequence
        ):
            return index

    return -1


def _binary_root_for_class_file(
    class_file: Path,
) -> Optional[Path]:
    """
    Infer the Java binary root from the location of a compiled .class file.
    """
    parts = class_file.parts

    patterns = [
        (
            "build",
            "classes",
            "java",
            "main",
        ),
        (
            "build",
            "classes",
            "kotlin",
            "main",
        ),
        (
            "build",
            "classes",
            "groovy",
            "main",
        ),
        (
            "build",
            "classes",
            "main",
        ),
        (
            "build",
            "classes",
        ),
        (
            "target",
            "classes",
        ),
        (
            "out",
            "production",
        ),
    ]

    for pattern in patterns:
        index = _find_subsequence(
            parts,
            pattern,
        )

        if index >= 0:
            return Path(
                *parts[
                    : index + len(pattern)
                ]
            )

    # Android/Gradle commonly stores bytecode under:
    # build/intermediates/javac/<variant>/classes
    build_index = _find_subsequence(
        parts,
        (
            "build",
            "intermediates",
            "javac",
        ),
    )

    if build_index >= 0:
        for index in range(
            build_index + 3,
            len(parts),
        ):
            if parts[index] == "classes":
                return Path(
                    *parts[: index + 1]
                )

    # Conservative fallback for classic bin/ output.
    for index, part in enumerate(
        parts[:-1]
    ):
        if part == "bin":
            return Path(
                *parts[: index + 1]
            )

    return None


def find_java_binaries_dirs(
    repo_path: Path,
) -> list[Path]:
    """Find directories that contain compiled Java bytecode."""
    roots: set[Path] = set()

    for class_file in repo_path.rglob("*.class"):
        if any(
            part
            in {
                ".git",
                ".gradle",
                ".scannerwork",
            }
            for part in class_file.parts
        ):
            continue

        root = _binary_root_for_class_file(
            class_file.resolve()
        )

        if root and root.is_dir():
            roots.add(root)

    return sorted(
        roots,
        key=lambda path: str(path),
    )


def compile_java_project(
    repo_path: Path,
) -> BuildResult:
    """
    Try the available Maven or Gradle build roots until compilation succeeds
    and Java bytecode is found.
    """
    candidates = find_build_candidates(
        repo_path
    )

    if not candidates:
        return BuildResult(
            ok=False,
            build_system="",
            build_root=None,
            binaries=[],
            output="",
            reason=(
                "No supported Maven or Gradle "
                "build file was found."
            ),
        )

    combined_logs: list[str] = []

    for build_system, build_root in candidates:
        print(
            f"☕ Trying {build_system} build "
            f"in {build_root}..."
        )

        if build_system == "maven":
            result = run_maven_compile(
                build_root
            )
        else:
            result = run_gradle_compile(
                build_root
            )

        combined_logs.append(
            f"=== {build_system.upper()} "
            f"@ {build_root} ===\n"
            f"{result.output}"
        )

        binaries = find_java_binaries_dirs(
            repo_path
        )

        if result.ok and binaries:
            return BuildResult(
                ok=True,
                build_system=build_system,
                build_root=build_root,
                binaries=binaries,
                output="\n\n".join(
                    combined_logs
                ),
                reason="",
            )

        if result.ok and not binaries:
            print(
                "⚠️ Build command succeeded, but "
                "no compiled .class output was found."
            )

    return BuildResult(
        ok=False,
        build_system=",".join(
            dict.fromkeys(
                system
                for system, _ in candidates
            )
        ),
        build_root=None,
        binaries=find_java_binaries_dirs(
            repo_path
        ),
        output="\n\n".join(
            combined_logs
        ),
        reason=(
            "All supported build attempts failed "
            "or produced no Java bytecode."
        ),
    )


# -----------------------------------------------------------------------------
# SonarQube analysis
# -----------------------------------------------------------------------------


LINE_RANGE_ERROR = re.compile(
    r"Line\s+\d+\s+is out of range for file\s+(.+?)\.\s+"
    r"File has\s+\d+\s+lines\.",
    flags=re.IGNORECASE,
)


def normalize_reported_source_path(
    raw_path: str,
    repo_path: Path,
) -> Optional[str]:
    """
    Convert a path reported by SonarQube into a repository-relative path.
    """
    normalized = (
        raw_path
        .strip()
        .replace("\\", "/")
    )

    repo_abs = (
        repo_path
        .resolve()
        .as_posix()
        .rstrip("/")
    )

    if normalized.startswith(
        repo_abs + "/"
    ):
        normalized = normalized[
            len(repo_abs) + 1 :
        ]

    # Compatibility with paths beginning with the clone directory.
    marker = f"/{repo_path.name}/"

    if marker in "/" + normalized:
        normalized = (
            "/" + normalized
        ).split(
            marker,
            1,
        )[1]

    normalized = normalized.lstrip("./")

    candidate = repo_path / normalized

    if candidate.is_file():
        return (
            candidate
            .relative_to(repo_path)
            .as_posix()
        )

    return None


def extract_line_range_file(
    scanner_output: str,
    repo_path: Path,
) -> Optional[str]:
    """Extract the file responsible for a SonarQube line-range error."""
    match = LINE_RANGE_ERROR.search(
        scanner_output
    )

    if not match:
        return None

    return normalize_reported_source_path(
        match.group(1),
        repo_path,
    )


def remove_scanner_workdir(
    repo_path: Path,
) -> None:
    shutil.rmtree(
        repo_path / ".scannerwork",
        ignore_errors=True,
    )


def build_sonar_command(
    *,
    project_key: str,
    project_name: str,
    java_binaries: list[Path],
    exclusions: list[str],
) -> list[str]:
    """Build the SonarScanner command."""
    binaries_value = ",".join(
        binary.resolve().as_posix()
        for binary in java_binaries
    )

    return [
        str(SONAR_SCANNER),
        f"-Dsonar.projectKey={project_key}",
        f"-Dsonar.projectName={project_name}",
        "-Dsonar.sources=.",
        "-Dsonar.sourceEncoding=UTF-8",
        "-Dsonar.scm.provider=git",
        "-Dsonar.scm.exclusions.disabled=true",
        (
            "-Dsonar.exclusions="
            f"{','.join(exclusions)}"
        ),
        (
            "-Dsonar.java.binaries="
            f"{binaries_value}"
        ),
        f"-Dsonar.host.url={SONAR_URL}",
    ]


def run_sonar_analysis(
    repo_path: Path,
    project_key: str,
    project_name: str,
    java_binaries: list[Path],
) -> ScanResult:
    """
    Run SonarQube analysis.

    If SonarQube reports that a source line is outside the file's actual line
    range, retry once while excluding only the problematic file.
    """
    exclusions = list(
        BASE_SONAR_EXCLUSIONS
    )

    excluded_files: list[str] = []
    outputs: list[str] = []

    scanner_env = os.environ.copy()
    scanner_env["SONAR_TOKEN"] = SONAR_TOKEN

    for attempt in range(
        MAX_SONAR_RETRIES + 1
    ):
        remove_scanner_workdir(
            repo_path
        )

        command = build_sonar_command(
            project_key=project_key,
            project_name=project_name,
            java_binaries=java_binaries,
            exclusions=exclusions,
        )

        print(
            "🔍 SonarQube attempt "
            f"{attempt + 1}/"
            f"{MAX_SONAR_RETRIES + 1}..."
        )

        result = run_cmd(
            command,
            cwd=repo_path,
            env=scanner_env,
        )

        outputs.append(
            result.output
        )

        if result.ok:
            return ScanResult(
                ok=True,
                output=(
                    "\n\n--- SONAR RETRY ---\n\n"
                ).join(outputs),
                excluded_files=excluded_files,
                reason="",
            )

        problematic_file = (
            extract_line_range_file(
                result.output,
                repo_path,
            )
        )

        can_retry = (
            attempt < MAX_SONAR_RETRIES
        )

        if (
            problematic_file
            and can_retry
            and problematic_file
            not in exclusions
        ):
            print(
                "⚠️ Sonar analyzer reported an "
                "out-of-range line for "
                f"{problematic_file}. Retrying "
                "with only that file excluded."
            )

            exclusions.append(
                problematic_file
            )

            excluded_files.append(
                problematic_file
            )

            continue

        reason = (
            "SonarScanner execution failed."
        )

        if problematic_file:
            reason += (
                " Problematic file: "
                f"{problematic_file}."
            )

        if result.timed_out:
            reason += (
                " Scanner command timed out."
            )

        return ScanResult(
            ok=False,
            output=(
                "\n\n--- SONAR RETRY ---\n\n"
            ).join(outputs),
            excluded_files=excluded_files,
            reason=reason,
        )

    return ScanResult(
        ok=False,
        output=(
            "\n\n--- SONAR RETRY ---\n\n"
        ).join(outputs),
        excluded_files=excluded_files,
        reason=(
            "SonarScanner failed after "
            "all retries."
        ),
    )


def read_report_task(
    repo_path: Path,
) -> dict[str, str]:
    """Read SonarScanner's report-task.txt file."""
    report_file = (
        repo_path
        / ".scannerwork"
        / "report-task.txt"
    )

    if not report_file.is_file():
        return {}

    values: dict[str, str] = {}

    for line in report_file.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if "=" not in line:
            continue

        key, value = line.split(
            "=",
            1,
        )

        values[key.strip()] = (
            value.strip()
        )

    return values


def wait_for_compute_engine(
    repo_path: Path,
) -> tuple[bool, str]:
    """Wait for SonarQube's Compute Engine task to complete."""
    report = read_report_task(
        repo_path
    )

    task_url = report.get(
        "ceTaskUrl"
    )

    if not task_url:
        return (
            False,
            (
                "SonarQube report-task.txt did "
                "not contain ceTaskUrl."
            ),
        )

    deadline = (
        time.monotonic()
        + CE_TIMEOUT_SECONDS
    )

    while time.monotonic() < deadline:
        try:
            response = SESSION.get(
                task_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        except requests.RequestException as exc:
            return (
                False,
                (
                    "Failed to query SonarQube "
                    f"Compute Engine: {exc}"
                ),
            )

        if response.status_code != 200:
            return (
                False,
                (
                    "Failed to query SonarQube "
                    "Compute Engine: "
                    f"HTTP {response.status_code} - "
                    f"{response.text}"
                ),
            )

        task = (
            response
            .json()
            .get("task", {})
        )

        status = str(
            task.get("status", "")
        ).upper()

        if status == "SUCCESS":
            return True, ""

        if status in {
            "FAILED",
            "CANCELED",
        }:
            error_message = task.get(
                "errorMessage",
                "",
            )

            return (
                False,
                (
                    "Compute Engine status "
                    f"{status}: {error_message}"
                ),
            )

        time.sleep(
            CE_POLL_INTERVAL_SECONDS
        )

    return (
        False,
        (
            "Compute Engine did not finish "
            f"within {CE_TIMEOUT_SECONDS} "
            "seconds."
        ),
    )


# -----------------------------------------------------------------------------
# SonarQube API collection
# -----------------------------------------------------------------------------


def fetch_metrics(
    project_key: str,
) -> tuple[
    Optional[dict[str, str]],
    str,
]:
    """Fetch project-level SonarQube metrics."""
    endpoint = (
        f"{SONAR_URL}"
        "/api/measures/component"
    )

    params = {
        "component": project_key,
        "metricKeys": ",".join(
            METRICS
        ),
    }

    try:
        response = SESSION.get(
            endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    except requests.RequestException as exc:
        return (
            None,
            f"Metrics request failed: {exc}",
        )

    if response.status_code != 200:
        return (
            None,
            (
                "Metrics request failed: "
                f"HTTP {response.status_code} - "
                f"{response.text}"
            ),
        )

    measures = {
        item["metric"]: item.get(
            "value",
            "0",
        )
        for item in (
            response
            .json()
            .get("component", {})
            .get("measures", [])
        )
    }

    # Preserve a stable CSV schema even when SonarQube omits zero metrics.
    return (
        {
            metric: measures.get(
                metric,
                "0",
            )
            for metric in METRICS
        },
        "",
    )


def get_vulnerability_details(
    project_key: str,
    repo_id: str,
) -> tuple[dict[str, int], str]:
    """
    Retrieve all vulnerability issues for a project and save them as JSON.
    """
    endpoint = (
        f"{SONAR_URL}"
        "/api/issues/search"
    )

    page_size = 500
    page = 1
    all_issues: list[dict] = []

    while True:
        params = {
            "componentKeys": project_key,
            "types": "VULNERABILITY",
            "ps": page_size,
            "p": page,
        }

        try:
            response = SESSION.get(
                endpoint,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

        except requests.RequestException as exc:
            return (
                {},
                f"Issues request failed: {exc}",
            )

        if response.status_code != 200:
            return (
                {},
                (
                    "Issues request failed: "
                    f"HTTP {response.status_code} - "
                    f"{response.text}"
                ),
            )

        payload = response.json()
        issues = payload.get(
            "issues",
            [],
        )

        all_issues.extend(
            issues
        )

        total = int(
            payload
            .get("paging", {})
            .get(
                "total",
                len(all_issues),
            )
        )

        if (
            not issues
            or len(all_issues) >= total
        ):
            break

        page += 1

    json_path = (
        JSON_OUTPUT_DIR
        / f"{repo_id}_vulnerabilities.json"
    )

    json_path.write_text(
        json.dumps(
            all_issues,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"💾 Saved {len(all_issues)} "
        f"vulnerabilities to {json_path}"
    )

    counts = Counter(
        str(
            issue.get(
                "severity",
                "UNKNOWN",
            )
        ).upper()
        for issue in all_issues
    )

    return dict(counts), ""


# -----------------------------------------------------------------------------
# CSV and failure handling
# -----------------------------------------------------------------------------


def append_csv_row(
    path: Path,
    row: dict,
) -> None:
    """Append one result to a CSV file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [row]
    ).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def record_failure(
    *,
    template: TemplateRepository,
    stage: str,
    reason: str,
    build_system: str = "",
    log_path: Optional[Path] = None,
    excluded_files: Optional[list[str]] = None,
) -> None:
    """Append a template-analysis failure to the failure CSV."""
    row = {
        "timestamp_utc": (
            datetime
            .now(timezone.utc)
            .isoformat()
        ),
        "target_template_repository": template.slug,
        "canonical_template_repository": (
            template.canonical_slug
        ),
        "repo_url": template.clone_url,
        "owner": template.owner,
        "repo": template.repo_name,
        "associated_generated_repositories": (
            template.associated_generated_repositories
        ),
        "stage": stage,
        "reason": reason,
        "build_system": build_system,
        "log_path": (
            str(log_path)
            if log_path
            else ""
        ),
        "excluded_files": ";".join(
            excluded_files or []
        ),
    }

    append_csv_row(
        FAILURES_CSV,
        row,
    )


def load_processed_templates() -> set[str]:
    """
    Load canonical owner/repository identifiers for templates already analyzed
    successfully.
    """
    if not OUTPUT_CSV.exists():
        return set()

    existing = pd.read_csv(
        OUTPUT_CSV,
        on_bad_lines="skip",
    )

    processed: set[str] = set()

    if (
        "canonical_template_repository"
        in existing.columns
    ):
        values = (
            existing[
                "canonical_template_repository"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        processed.update(
            value
            for value in values
            if value
        )

    elif (
        "target_template_repository"
        in existing.columns
    ):
        for value in (
            existing[
                "target_template_repository"
            ]
            .dropna()
            .astype(str)
        ):
            try:
                processed.add(
                    canonical_repository_slug(
                        value
                    )
                )
            except ValueError:
                continue

    elif "repo_url" in existing.columns:
        for value in (
            existing["repo_url"]
            .dropna()
            .astype(str)
        ):
            try:
                processed.add(
                    canonical_repository_slug(
                        value
                    )
                )
            except ValueError:
                continue

    return processed


# -----------------------------------------------------------------------------
# Template analysis
# -----------------------------------------------------------------------------


def analyze_template(
    template: TemplateRepository,
) -> Optional[dict]:
    """Clone, compile, analyze, and collect metrics for one template."""
    repo_path: Optional[Path] = None

    repo_id = safe_identifier(
        f"{template.owner}__"
        f"{template.repo_name}"
    )

    try:
        repo_path = (
            REPO_DIR
            / repo_id
        )

        project_key = compute_project_key(
            template.owner,
            template.repo_name,
        )

        if repo_path.exists():
            shutil.rmtree(
                repo_path,
                ignore_errors=True,
            )

        print(
            "📦 Template: "
            f"{template.slug}"
        )

        print(
            "🔗 Referenced by "
            f"{template.associated_generated_repositories} "
            "eligible generated-repository row(s)."
        )

        clone = run_cmd(
            [
                "git",
                "clone",
                "--depth",
                "1",
                template.clone_url,
                str(repo_path),
            ]
        )

        if not clone.ok:
            log_path = write_log(
                repo_id,
                "clone",
                clone.output,
            )

            record_failure(
                template=template,
                stage="clone",
                reason="Git clone failed.",
                log_path=log_path,
            )

            return None

        if not project_contains_java(
            repo_path
        ):
            record_failure(
                template=template,
                stage="precheck",
                reason=(
                    "No Java source file was found."
                ),
            )

            return None

        build = compile_java_project(
            repo_path
        )

        build_log_path = write_log(
            repo_id,
            "build",
            build.output,
        )

        if not build.ok:
            record_failure(
                template=template,
                stage="build",
                reason=build.reason,
                build_system=(
                    build.build_system
                ),
                log_path=build_log_path,
            )

            return None

        print(
            "✅ Java compilation succeeded "
            f"using {build.build_system}. "
            f"Found {len(build.binaries)} "
            "binary directories."
        )

        scan = run_sonar_analysis(
            repo_path=repo_path,
            project_key=project_key,
            project_name=template.slug,
            java_binaries=build.binaries,
        )

        sonar_log_path = write_log(
            repo_id,
            "sonar",
            scan.output,
        )

        if not scan.ok:
            record_failure(
                template=template,
                stage="sonar_scan",
                reason=scan.reason,
                build_system=(
                    build.build_system
                ),
                log_path=sonar_log_path,
                excluded_files=(
                    scan.excluded_files
                ),
            )

            return None

        ce_ok, ce_error = (
            wait_for_compute_engine(
                repo_path
            )
        )

        if not ce_ok:
            record_failure(
                template=template,
                stage="compute_engine",
                reason=ce_error,
                build_system=(
                    build.build_system
                ),
                log_path=sonar_log_path,
                excluded_files=(
                    scan.excluded_files
                ),
            )

            return None

        measures, metrics_error = (
            fetch_metrics(
                project_key
            )
        )

        if measures is None:
            record_failure(
                template=template,
                stage="metrics_api",
                reason=metrics_error,
                build_system=(
                    build.build_system
                ),
                log_path=sonar_log_path,
                excluded_files=(
                    scan.excluded_files
                ),
            )

            return None

        severity_counts, issues_error = (
            get_vulnerability_details(
                project_key,
                repo_id,
            )
        )

        if issues_error:
            record_failure(
                template=template,
                stage="issues_api",
                reason=issues_error,
                build_system=(
                    build.build_system
                ),
                log_path=sonar_log_path,
                excluded_files=(
                    scan.excluded_files
                ),
            )

            return None

        if (
            build.build_root
            and build.build_root != repo_path
        ):
            try:
                build_root_value = (
                    build.build_root
                    .resolve()
                    .relative_to(
                        repo_path.resolve()
                    )
                    .as_posix()
                )
            except ValueError:
                build_root_value = str(
                    build.build_root
                )
        else:
            build_root_value = "."

        result = {
            "target_template_repository": (
                template.slug
            ),
            "canonical_template_repository": (
                template.canonical_slug
            ),
            "repo_url": template.clone_url,
            "owner": template.owner,
            "repo": template.repo_name,
            "project_key": project_key,
            "associated_generated_repositories": (
                template
                .associated_generated_repositories
            ),
            "analysis_status": (
                "success_with_file_exclusion"
                if scan.excluded_files
                else "success"
            ),
            "build_system": (
                build.build_system
            ),
            "build_root": (
                build_root_value
            ),
            "java_binary_dirs": len(
                build.binaries
            ),
            "sonar_excluded_files": ";".join(
                scan.excluded_files
            ),
            **measures,
        }

        for severity in SEVERITIES:
            result[
                f"vuln_{severity.lower()}"
            ] = severity_counts.get(
                severity,
                0,
            )

        return result

    except Exception as exc:
        log_path = write_log(
            repo_id,
            "unexpected",
            repr(exc),
        )

        record_failure(
            template=template,
            stage="unexpected",
            reason=(
                f"Unexpected error: {exc}"
            ),
            log_path=log_path,
        )

        return None

    finally:
        if repo_path is not None:
            shutil.rmtree(
                repo_path,
                ignore_errors=True,
            )

            print(
                "🧹 Deleted local clone of "
                f"{template.slug}"
            )


# -----------------------------------------------------------------------------
# Optional maintenance utilities
# -----------------------------------------------------------------------------


def delete_all_sonar_projects() -> None:
    """Delete every project from the configured SonarQube instance."""
    print(
        "🗑️ Fetching all SonarQube projects..."
    )

    page = 1
    page_size = 500
    keys: list[str] = []

    while True:
        response = SESSION.get(
            f"{SONAR_URL}/api/projects/search",
            params={
                "p": page,
                "ps": page_size,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            print(
                "❌ Failed to fetch projects: "
                f"{response.text}"
            )
            return

        payload = response.json()

        components = payload.get(
            "components",
            [],
        )

        keys.extend(
            component["key"]
            for component in components
        )

        total = int(
            payload
            .get("paging", {})
            .get(
                "total",
                len(keys),
            )
        )

        if len(keys) >= total:
            break

        page += 1

    print(
        f"📌 Found {len(keys)} "
        "projects to delete."
    )

    for key in keys:
        response = SESSION.post(
            f"{SONAR_URL}/api/projects/delete",
            params={
                "project": key,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 204:
            print(
                f"✔ Deleted {key}"
            )
        else:
            print(
                f"❌ Failed to delete {key}: "
                f"HTTP {response.status_code} - "
                f"{response.text}"
            )


def _to_int_or_none(
    value,
) -> Optional[int]:
    try:
        if pd.isna(value):
            return None

        text = str(value).strip()

        if not text:
            return None

        return int(float(text))

    except (
        TypeError,
        ValueError,
    ):
        return None


def repair_incomplete_results(
    output_csv: Path = OUTPUT_CSV,
) -> None:
    """
    Refresh rows whose stored SonarQube metrics are all empty or zero.

    This utility assumes the corresponding SonarQube project is still available
    on the configured server.
    """
    if not output_csv.exists():
        print(
            f"❌ Output CSV not found: "
            f"{output_csv}"
        )
        return

    df = pd.read_csv(
        output_csv,
        on_bad_lines="skip",
    )

    repaired = 0

    for index, row in df.iterrows():
        repository_reference = row.get(
            "target_template_repository"
        )

        if (
            not isinstance(
                repository_reference,
                str,
            )
            or not repository_reference.strip()
        ):
            repository_reference = row.get(
                "repo_url"
            )

        if (
            not isinstance(
                repository_reference,
                str,
            )
            or not repository_reference.strip()
        ):
            continue

        current = [
            _to_int_or_none(
                row.get(metric)
            )
            for metric in METRICS
        ]

        if not all(
            value in (
                None,
                0,
            )
            for value in current
        ):
            continue

        try:
            project_key = (
                compute_project_key_from_reference(
                    repository_reference
                )
            )

            owner, repo_name = (
                parse_repository_reference(
                    repository_reference
                )
            )

        except ValueError as exc:
            print(
                "⚠️ Could not interpret repository "
                f"reference '{repository_reference}': "
                f"{exc}"
            )
            continue

        measures, error = fetch_metrics(
            project_key
        )

        if measures is None:
            print(
                f"⚠️ Could not repair "
                f"{repository_reference}: {error}"
            )
            continue

        sonar_values = [
            _to_int_or_none(
                measures.get(metric)
            )
            for metric in METRICS
        ]

        if not any(
            value not in (
                None,
                0,
            )
            for value in sonar_values
        ):
            continue

        for metric in METRICS:
            df.at[
                index,
                metric,
            ] = measures.get(
                metric,
                "0",
            )

        repo_id = safe_identifier(
            f"{owner}__{repo_name}"
        )

        severity_counts, issues_error = (
            get_vulnerability_details(
                project_key,
                repo_id,
            )
        )

        if not issues_error:
            for severity in SEVERITIES:
                column = (
                    f"vuln_"
                    f"{severity.lower()}"
                )

                if column in df.columns:
                    df.at[
                        index,
                        column,
                    ] = severity_counts.get(
                        severity,
                        0,
                    )

        repaired += 1

    df.to_csv(
        output_csv,
        index=False,
    )

    print(
        "💾 Repair complete. Updated "
        f"{repaired} row(s)."
    )


# -----------------------------------------------------------------------------
# Configuration validation
# -----------------------------------------------------------------------------


def validate_configuration() -> None:
    if not SONAR_TOKEN:
        raise RuntimeError(
            "SONAR_TOKEN is not set.\n\n"
            "Run:\n"
            "    export SONAR_TOKEN='your-token'"
        )

    if not SONAR_SCANNER.is_file():
        raise FileNotFoundError(
            "SonarScanner not found: "
            f"{SONAR_SCANNER}"
        )

    if not _command_exists("git"):
        raise FileNotFoundError(
            "The 'git' command is not installed "
            "or is not available on PATH."
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    validate_configuration()

    templates = (
        load_template_repositories()
    )

    processed_templates = (
        load_processed_templates()
    )

    print(
        f"📥 Selected unique templates: "
        f"{len(templates)}"
    )

    print(
        "📄 Already processed successfully: "
        f"{len(processed_templates)}"
    )

    print(
        f"📤 Output CSV: {OUTPUT_CSV}"
    )

    print(
        f"⚠️ Failure CSV: {FAILURES_CSV}"
    )

    failures_this_run: list[str] = []

    for index, template in enumerate(
        templates,
        start=1,
    ):
        print(
            "\n"
            f"=== ({index}/{len(templates)}) "
            f"{template.slug} ==="
        )

        if (
            template.canonical_slug
            in processed_templates
        ):
            print(
                "⏩ Skipping: template already "
                "processed successfully."
            )
            continue

        result = analyze_template(
            template
        )

        if result is None:
            failures_this_run.append(
                template.slug
            )
            continue

        append_csv_row(
            OUTPUT_CSV,
            result,
        )

        processed_templates.add(
            template.canonical_slug
        )

        print(
            "💾 Saved template result."
        )

    print(
        "\n✅ Successful output: "
        f"{OUTPUT_CSV}"
    )

    print(
        "⚠️ Failed in this run: "
        f"{len(failures_this_run)}"
    )

    if failures_this_run:
        print(
            f"See details in: {FAILURES_CSV}"
        )


if __name__ == "__main__":
    # Optional maintenance operations:
    #
    # delete_all_sonar_projects()
    #
    # repair_incomplete_results()
    #
    # Comment out main() when running only a maintenance operation.

    main()