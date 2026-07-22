#!/usr/bin/env python3
"""
analyze_java_repositories.py

Clone selected Java repositories, compile them, run SonarQube analysis,
collect project metrics and vulnerability details, and save failures separately.

Expected input:
    sample/persistence_sample_balanced/Java/selected_sample.csv

Supported input URL columns:
    original_link
    original_link_repository

Required environment variable:
    SONAR_TOKEN

Example:
    export SONAR_TOKEN="your-token"
    python analyze_java_repositories.py
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

SONAR_URL = os.getenv("SONAR_URL", "http://localhost:9002").rstrip("/")
SONAR_TOKEN = "squ_f75043d923b1b7855a7c6e44ba031a9997df5881"
SONAR_SCANNER = Path(
    os.getenv("SONAR_SCANNER", "/opt/sonar-scanner/bin/sonar-scanner")
)

REPO_DIR = Path("temp_repos")
SAMPLE_ROOT = Path("sample/persistence_sample_balanced")
INPUT_CSV = SAMPLE_ROOT / PROGRAMMING_LANGUAGE / "eligible_candidates.csv"

OUTPUT_ROOT = Path("vulnerability_issues") / PROGRAMMING_LANGUAGE
OUTPUT_CSV = OUTPUT_ROOT / f"vulnerability_results_{PROGRAMMING_LANGUAGE}.csv"
FAILURES_CSV = OUTPUT_ROOT / f"analysis_failures_{PROGRAMMING_LANGUAGE}.csv"
JSON_OUTPUT_DIR = OUTPUT_ROOT / "vulnerability_details"
LOG_DIR = OUTPUT_ROOT / "logs"

START_DATE = pd.Timestamp("2023-10-18", tz="UTC")
MIN_COMMIT_COUNT = 2  # Equivalent to commit_count > 1

METRICS = [
    "ncloc",
    "files",
    "functions",
    "vulnerabilities",
    "security_hotspots",
    "code_smells",
    "bugs",
]

SEVERITIES = ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]

COMMAND_TIMEOUT_SECONDS = 20 * 60
CE_TIMEOUT_SECONDS = 5 * 60
CE_POLL_INTERVAL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 60
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

for directory in [REPO_DIR, OUTPUT_ROOT, JSON_OUTPUT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

SESSION = requests.Session()
if SONAR_TOKEN:
    SESSION.auth = (SONAR_TOKEN, "")

# -----------------------------------------------------------------------------
# Command execution
# -----------------------------------------------------------------------------


@dataclass
class CommandResult:
    ok: bool
    returncode: int
    output: str
    timed_out: bool = False


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
    """Run a command, capture combined output, and never raise on failure."""
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
            output = output.decode("utf-8", errors="replace")

        message = (
            f"\nERROR: Command timed out after {timeout} seconds.\n"
        )
        output += message
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
        return CommandResult(ok=False, returncode=127, output=output)


def write_log(repo_id: str, stage: str, content: str) -> Path:
    log_path = LOG_DIR / f"{repo_id}_{stage}.log"
    log_path.write_text(content or "", encoding="utf-8", errors="replace")
    return log_path

# -----------------------------------------------------------------------------
# Repository and input helpers
# -----------------------------------------------------------------------------


def parse_repository_url(git_url: str) -> tuple[str, str]:
    """Extract owner and repository name from a GitHub URL."""
    parsed = urlparse(git_url.strip())
    parts = [part for part in parsed.path.strip("/").split("/") if part]

    if len(parts) < 2:
        raise ValueError(f"Unsupported repository URL: {git_url}")

    owner = parts[-2]
    repo_name = re.sub(r"\.git$", "", parts[-1], flags=re.IGNORECASE)
    return owner, repo_name


def safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def compute_project_key(owner: str, repo_name: str) -> str:
    return f"{owner}__{repo_name}".lower()


def compute_project_key_from_url(repo_url: str) -> str:
    owner, repo_name = parse_repository_url(repo_url)
    return compute_project_key(owner, repo_name)


def _parse_boolean(value) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def load_repository_urls() -> list[str]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, on_bad_lines="skip")

    url_column = next(
        (
            column
            for column in ["original_link", "original_link_repository"]
            if column in df.columns
        ),
        None,
    )

    if url_column is None:
        raise KeyError(
            "The input CSV must contain 'original_link' or "
            "'original_link_repository'."
        )

    required_columns = {"commit_count", "is_template", "data_update"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Missing required input columns: {missing}")

    df["commit_count"] = pd.to_numeric(df["commit_count"], errors="coerce")
    df["data_update"] = pd.to_datetime(
        df["data_update"],
        errors="coerce",
        utc=True,
    )
    is_template = df["is_template"].map(_parse_boolean)

    mask = (
        (df["commit_count"] >= MIN_COMMIT_COUNT)
        & (is_template == False)  # noqa: E712 - intentional pandas comparison
        & (df["data_update"] >= START_DATE)
    )

    urls = (
        df.loc[mask, url_column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    # Deduplicate while preserving CSV order.
    return list(dict.fromkeys(url for url in urls if url))


def project_contains_java(repo_path: Path) -> bool:
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [directory for directory in dirs if directory not in SKIP_WALK_DIRS]
        if any(filename.lower().endswith(".java") for filename in files):
            return True
    return False

# -----------------------------------------------------------------------------
# Java build handling
# -----------------------------------------------------------------------------


def _path_depth(path: Path, base: Path) -> int:
    return len(path.resolve().relative_to(base.resolve()).parts)


def find_build_candidates(repo_path: Path) -> list[tuple[str, Path]]:
    """
    Return Maven/Gradle build roots.

    A root-level build is preferred. If none exists, inspect nested build files
    up to four directory levels below the repository root.
    """
    candidates: list[tuple[str, Path]] = []

    root_has_maven = (repo_path / "pom.xml").is_file()
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
        candidates.append(("maven", repo_path))
    if root_has_gradle:
        candidates.append(("gradle", repo_path))

    if candidates:
        return candidates

    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)
        depth = _path_depth(root_path, repo_path)

        dirs[:] = [
            directory
            for directory in dirs
            if directory not in SKIP_WALK_DIRS and depth < 4
        ]

        file_set = set(files)
        if "pom.xml" in file_set:
            candidates.append(("maven", root_path))

        if file_set.intersection(
            {
                "gradlew",
                "build.gradle",
                "build.gradle.kts",
                "settings.gradle",
                "settings.gradle.kts",
            }
        ):
            candidates.append(("gradle", root_path))

    # Deduplicate and prefer shallower roots.
    unique: dict[tuple[str, str], tuple[str, Path]] = {}
    for system, root in candidates:
        key = (system, str(root.resolve()))
        unique[key] = (system, root)

    return sorted(
        unique.values(),
        key=lambda item: (_path_depth(item[1], repo_path), item[0]),
    )


def _command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def run_maven_compile(build_root: Path) -> CommandResult:
    wrapper = build_root / "mvnw"

    if wrapper.is_file():
        wrapper.chmod(wrapper.stat().st_mode | 0o111)
        executable = "./mvnw"
    elif _command_exists("mvn"):
        executable = "mvn"
    else:
        return CommandResult(
            ok=False,
            returncode=127,
            output="Maven wrapper not found and system 'mvn' is unavailable.\n",
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
    return run_cmd(command, cwd=build_root)


def run_gradle_compile(build_root: Path) -> CommandResult:
    wrapper = build_root / "gradlew"

    if wrapper.is_file():
        wrapper.chmod(wrapper.stat().st_mode | 0o111)
        executable = "./gradlew"
    elif _command_exists("gradle"):
        executable = "gradle"
    else:
        return CommandResult(
            ok=False,
            returncode=127,
            output="Gradle wrapper not found and system 'gradle' is unavailable.\n",
        )

    first = run_cmd(
        [executable, "classes", "-x", "test", "--no-daemon"],
        cwd=build_root,
    )
    if first.ok:
        return first

    print("🔁 Gradle 'classes' failed. Trying 'assemble -x test'...")
    second = run_cmd(
        [executable, "assemble", "-x", "test", "--no-daemon"],
        cwd=build_root,
    )
    second.output = first.output + "\n\n--- GRADLE FALLBACK ---\n\n" + second.output
    return second


def _find_subsequence(parts: tuple[str, ...], sequence: tuple[str, ...]) -> int:
    if len(sequence) > len(parts):
        return -1

    for index in range(len(parts) - len(sequence) + 1):
        if parts[index : index + len(sequence)] == sequence:
            return index
    return -1


def _binary_root_for_class_file(class_file: Path) -> Optional[Path]:
    parts = class_file.parts

    patterns = [
        ("build", "classes", "java", "main"),
        ("build", "classes", "kotlin", "main"),
        ("build", "classes", "groovy", "main"),
        ("build", "classes", "main"),
        ("build", "classes"),
        ("target", "classes"),
        ("out", "production"),
    ]

    for pattern in patterns:
        index = _find_subsequence(parts, pattern)
        if index >= 0:
            return Path(*parts[: index + len(pattern)])

    # Android/Gradle commonly stores bytecode below:
    # build/intermediates/javac/<variant>/classes
    build_index = _find_subsequence(parts, ("build", "intermediates", "javac"))
    if build_index >= 0:
        for index in range(build_index + 3, len(parts)):
            if parts[index] == "classes":
                return Path(*parts[: index + 1])

    # Conservative fallback for classic bin/ output.
    for index, part in enumerate(parts[:-1]):
        if part == "bin":
            return Path(*parts[: index + 1])

    return None


def find_java_binaries_dirs(repo_path: Path) -> list[Path]:
    roots: set[Path] = set()

    for class_file in repo_path.rglob("*.class"):
        if any(part in {".git", ".gradle", ".scannerwork"} for part in class_file.parts):
            continue

        root = _binary_root_for_class_file(class_file.resolve())
        if root and root.is_dir():
            roots.add(root)

    return sorted(roots, key=lambda path: str(path))


@dataclass
class BuildResult:
    ok: bool
    build_system: str
    build_root: Optional[Path]
    binaries: list[Path]
    output: str
    reason: str


def compile_java_project(repo_path: Path) -> BuildResult:
    candidates = find_build_candidates(repo_path)

    if not candidates:
        return BuildResult(
            ok=False,
            build_system="",
            build_root=None,
            binaries=[],
            output="",
            reason="No supported Maven or Gradle build file was found.",
        )

    combined_logs: list[str] = []

    for build_system, build_root in candidates:
        print(f"☕ Trying {build_system} build in {build_root}...")

        if build_system == "maven":
            result = run_maven_compile(build_root)
        else:
            result = run_gradle_compile(build_root)

        combined_logs.append(
            f"=== {build_system.upper()} @ {build_root} ===\n{result.output}"
        )

        binaries = find_java_binaries_dirs(repo_path)
        if result.ok and binaries:
            return BuildResult(
                ok=True,
                build_system=build_system,
                build_root=build_root,
                binaries=binaries,
                output="\n\n".join(combined_logs),
                reason="",
            )

        if result.ok and not binaries:
            print("⚠️ Build command succeeded, but no compiled .class output was found.")

    return BuildResult(
        ok=False,
        build_system=",".join(dict.fromkeys(system for system, _ in candidates)),
        build_root=None,
        binaries=find_java_binaries_dirs(repo_path),
        output="\n\n".join(combined_logs),
        reason="All supported build attempts failed or produced no Java bytecode.",
    )

# -----------------------------------------------------------------------------
# SonarQube analysis
# -----------------------------------------------------------------------------


LINE_RANGE_ERROR = re.compile(
    r"Line\s+\d+\s+is out of range for file\s+(.+?)\.\s+"
    r"File has\s+\d+\s+lines\.",
    flags=re.IGNORECASE,
)


def normalize_reported_source_path(raw_path: str, repo_path: Path) -> Optional[str]:
    normalized = raw_path.strip().replace("\\", "/")
    repo_abs = repo_path.resolve().as_posix().rstrip("/")

    if normalized.startswith(repo_abs + "/"):
        normalized = normalized[len(repo_abs) + 1 :]

    # Compatibility with scans that reported paths beginning at temp_repos/.
    marker = f"/{repo_path.name}/"
    if marker in "/" + normalized:
        normalized = ("/" + normalized).split(marker, 1)[1]

    normalized = normalized.lstrip("./")

    candidate = repo_path / normalized
    if candidate.is_file():
        return candidate.relative_to(repo_path).as_posix()

    return None


def extract_line_range_file(scanner_output: str, repo_path: Path) -> Optional[str]:
    match = LINE_RANGE_ERROR.search(scanner_output)
    if not match:
        return None
    return normalize_reported_source_path(match.group(1), repo_path)


def remove_scanner_workdir(repo_path: Path) -> None:
    shutil.rmtree(repo_path / ".scannerwork", ignore_errors=True)


@dataclass
class ScanResult:
    ok: bool
    output: str
    excluded_files: list[str]
    reason: str


def build_sonar_command(
    *,
    project_key: str,
    project_name: str,
    java_binaries: list[Path],
    exclusions: list[str],
) -> list[str]:
    binaries_value = ",".join(
        binary.resolve().as_posix() for binary in java_binaries
    )

    return [
        str(SONAR_SCANNER),
        f"-Dsonar.projectKey={project_key}",
        f"-Dsonar.projectName={project_name}",
        "-Dsonar.sources=.",
        "-Dsonar.sourceEncoding=UTF-8",
        "-Dsonar.scm.provider=git",
        "-Dsonar.scm.exclusions.disabled=true",
        f"-Dsonar.exclusions={','.join(exclusions)}",
        f"-Dsonar.java.binaries={binaries_value}",
        f"-Dsonar.host.url={SONAR_URL}",
    ]


def run_sonar_analysis(
    repo_path: Path,
    project_key: str,
    project_name: str,
    java_binaries: list[Path],
) -> ScanResult:
    exclusions = list(BASE_SONAR_EXCLUSIONS)
    excluded_files: list[str] = []
    outputs: list[str] = []

    scanner_env = os.environ.copy()
    scanner_env["SONAR_TOKEN"] = SONAR_TOKEN

    for attempt in range(MAX_SONAR_RETRIES + 1):
        remove_scanner_workdir(repo_path)

        command = build_sonar_command(
            project_key=project_key,
            project_name=project_name,
            java_binaries=java_binaries,
            exclusions=exclusions,
        )

        print(f"🔍 SonarQube attempt {attempt + 1}/{MAX_SONAR_RETRIES + 1}...")
        result = run_cmd(command, cwd=repo_path, env=scanner_env)
        outputs.append(result.output)

        if result.ok:
            return ScanResult(
                ok=True,
                output="\n\n--- SONAR RETRY ---\n\n".join(outputs),
                excluded_files=excluded_files,
                reason="",
            )

        problematic_file = extract_line_range_file(result.output, repo_path)
        can_retry = attempt < MAX_SONAR_RETRIES

        if problematic_file and can_retry and problematic_file not in exclusions:
            print(
                "⚠️ Sonar analyzer reported an out-of-range line for "
                f"{problematic_file}. Retrying with only that file excluded."
            )
            exclusions.append(problematic_file)
            excluded_files.append(problematic_file)
            continue

        reason = "SonarScanner execution failed."
        if problematic_file:
            reason += f" Problematic file: {problematic_file}."
        if result.timed_out:
            reason += " Scanner command timed out."

        return ScanResult(
            ok=False,
            output="\n\n--- SONAR RETRY ---\n\n".join(outputs),
            excluded_files=excluded_files,
            reason=reason,
        )

    return ScanResult(
        ok=False,
        output="\n\n--- SONAR RETRY ---\n\n".join(outputs),
        excluded_files=excluded_files,
        reason="SonarScanner failed after all retries.",
    )


def read_report_task(repo_path: Path) -> dict[str, str]:
    report_file = repo_path / ".scannerwork" / "report-task.txt"
    if not report_file.is_file():
        return {}

    values: dict[str, str] = {}
    for line in report_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def wait_for_compute_engine(repo_path: Path) -> tuple[bool, str]:
    report = read_report_task(repo_path)
    task_url = report.get("ceTaskUrl")

    if not task_url:
        return False, "SonarQube report-task.txt did not contain ceTaskUrl."

    deadline = time.monotonic() + CE_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        try:
            response = SESSION.get(task_url, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            return False, f"Failed to query SonarQube Compute Engine: {exc}"

        if response.status_code != 200:
            return False, (
                "Failed to query SonarQube Compute Engine: "
                f"HTTP {response.status_code} - {response.text}"
            )

        task = response.json().get("task", {})
        status = str(task.get("status", "")).upper()

        if status == "SUCCESS":
            return True, ""

        if status in {"FAILED", "CANCELED"}:
            error_message = task.get("errorMessage", "")
            return False, f"Compute Engine status {status}: {error_message}"

        time.sleep(CE_POLL_INTERVAL_SECONDS)

    return False, (
        f"Compute Engine did not finish within {CE_TIMEOUT_SECONDS} seconds."
    )

# -----------------------------------------------------------------------------
# SonarQube API collection
# -----------------------------------------------------------------------------


def fetch_metrics(project_key: str) -> tuple[Optional[dict[str, str]], str]:
    endpoint = f"{SONAR_URL}/api/measures/component"
    params = {
        "component": project_key,
        "metricKeys": ",".join(METRICS),
    }

    try:
        response = SESSION.get(
            endpoint,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return None, f"Metrics request failed: {exc}"

    if response.status_code != 200:
        return None, (
            f"Metrics request failed: HTTP {response.status_code} - "
            f"{response.text}"
        )

    measures = {
        item["metric"]: item.get("value", "0")
        for item in response.json().get("component", {}).get("measures", [])
    }

    # Keep a stable CSV schema even when Sonar omits zero-valued measures.
    return {metric: measures.get(metric, "0") for metric in METRICS}, ""


def get_vulnerability_details(
    project_key: str,
    repo_id: str,
) -> tuple[dict[str, int], str]:
    endpoint = f"{SONAR_URL}/api/issues/search"
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
            return {}, f"Issues request failed: {exc}"

        if response.status_code != 200:
            return {}, (
                f"Issues request failed: HTTP {response.status_code} - "
                f"{response.text}"
            )

        payload = response.json()
        issues = payload.get("issues", [])
        all_issues.extend(issues)

        total = int(payload.get("paging", {}).get("total", len(all_issues)))
        if not issues or len(all_issues) >= total:
            break

        page += 1

    json_path = JSON_OUTPUT_DIR / f"{repo_id}_vulnerabilities.json"
    json_path.write_text(
        json.dumps(all_issues, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"💾 Saved {len(all_issues)} vulnerabilities to {json_path}")

    counts = Counter(
        str(issue.get("severity", "UNKNOWN")).upper()
        for issue in all_issues
    )
    return dict(counts), ""

# -----------------------------------------------------------------------------
# CSV and failure handling
# -----------------------------------------------------------------------------


def append_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
    )


def record_failure(
    *,
    repo_url: str,
    owner: str,
    repo_name: str,
    stage: str,
    reason: str,
    build_system: str = "",
    log_path: Optional[Path] = None,
    excluded_files: Optional[list[str]] = None,
) -> None:
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_url": repo_url,
        "owner": owner,
        "repo": repo_name,
        "stage": stage,
        "reason": reason,
        "build_system": build_system,
        "log_path": str(log_path) if log_path else "",
        "excluded_files": ";".join(excluded_files or []),
    }
    append_csv_row(FAILURES_CSV, row)


def load_processed_urls() -> set[str]:
    if not OUTPUT_CSV.exists():
        return set()

    existing = pd.read_csv(OUTPUT_CSV, on_bad_lines="skip")
    if "repo_url" not in existing.columns:
        return set()

    return set(existing["repo_url"].dropna().astype(str).str.strip())

# -----------------------------------------------------------------------------
# Repository analysis
# -----------------------------------------------------------------------------


def analyze_repo(git_url: str) -> Optional[dict]:
    owner = ""
    repo_name = ""
    repo_path: Optional[Path] = None

    try:
        owner, repo_name = parse_repository_url(git_url)
        repo_id = safe_identifier(f"{owner}__{repo_name}")
        repo_path = REPO_DIR / repo_id
        project_key = compute_project_key(owner, repo_name)

        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

        clone = run_cmd(
            ["git", "clone", "--depth", "1", git_url, str(repo_path)]
        )
        if not clone.ok:
            log_path = write_log(repo_id, "clone", clone.output)
            record_failure(
                repo_url=git_url,
                owner=owner,
                repo_name=repo_name,
                stage="clone",
                reason="Git clone failed.",
                log_path=log_path,
            )
            return None

        if not project_contains_java(repo_path):
            record_failure(
                repo_url=git_url,
                owner=owner,
                repo_name=repo_name,
                stage="precheck",
                reason="No Java source file was found.",
            )
            return None

        build = compile_java_project(repo_path)
        build_log_path = write_log(repo_id, "build", build.output)

        if not build.ok:
            record_failure(
                repo_url=git_url,
                owner=owner,
                repo_name=repo_name,
                stage="build",
                reason=build.reason,
                build_system=build.build_system,
                log_path=build_log_path,
            )
            return None

        print(
            f"✅ Java compilation succeeded using {build.build_system}. "
            f"Found {len(build.binaries)} binary directories."
        )

        scan = run_sonar_analysis(
            repo_path=repo_path,
            project_key=project_key,
            project_name=f"{owner}/{repo_name}",
            java_binaries=build.binaries,
        )
        sonar_log_path = write_log(repo_id, "sonar", scan.output)

        if not scan.ok:
            record_failure(
                repo_url=git_url,
                owner=owner,
                repo_name=repo_name,
                stage="sonar_scan",
                reason=scan.reason,
                build_system=build.build_system,
                log_path=sonar_log_path,
                excluded_files=scan.excluded_files,
            )
            return None

        ce_ok, ce_error = wait_for_compute_engine(repo_path)
        if not ce_ok:
            record_failure(
                repo_url=git_url,
                owner=owner,
                repo_name=repo_name,
                stage="compute_engine",
                reason=ce_error,
                build_system=build.build_system,
                log_path=sonar_log_path,
                excluded_files=scan.excluded_files,
            )
            return None

        measures, metrics_error = fetch_metrics(project_key)
        if measures is None:
            record_failure(
                repo_url=git_url,
                owner=owner,
                repo_name=repo_name,
                stage="metrics_api",
                reason=metrics_error,
                build_system=build.build_system,
                log_path=sonar_log_path,
                excluded_files=scan.excluded_files,
            )
            return None

        severity_counts, issues_error = get_vulnerability_details(
            project_key,
            repo_id,
        )
        if issues_error:
            record_failure(
                repo_url=git_url,
                owner=owner,
                repo_name=repo_name,
                stage="issues_api",
                reason=issues_error,
                build_system=build.build_system,
                log_path=sonar_log_path,
                excluded_files=scan.excluded_files,
            )
            return None

        result = {
            "repo_url": git_url,
            "owner": owner,
            "repo": repo_name,
            "project_key": project_key,
            "analysis_status": (
                "success_with_file_exclusion"
                if scan.excluded_files
                else "success"
            ),
            "build_system": build.build_system,
            "build_root": (
                build.build_root.relative_to(repo_path).as_posix()
                if build.build_root and build.build_root != repo_path
                else "."
            ),
            "java_binary_dirs": len(build.binaries),
            "sonar_excluded_files": ";".join(scan.excluded_files),
            **measures,
        }

        for severity in SEVERITIES:
            result[f"vuln_{severity.lower()}"] = severity_counts.get(severity, 0)

        return result

    except Exception as exc:
        repo_id = safe_identifier(f"{owner}__{repo_name}" or "unknown_repo")
        log_path = write_log(repo_id, "unexpected", repr(exc))
        record_failure(
            repo_url=git_url,
            owner=owner,
            repo_name=repo_name,
            stage="unexpected",
            reason=f"Unexpected error: {exc}",
            log_path=log_path,
        )
        return None

    finally:
        if repo_path is not None:
            shutil.rmtree(repo_path, ignore_errors=True)
            print(f"🧹 Deleted local clone of {repo_name or repo_path.name}")

# -----------------------------------------------------------------------------
# Optional maintenance utilities
# -----------------------------------------------------------------------------


def delete_all_sonar_projects() -> None:
    """Delete all projects from this SonarQube instance."""
    print("🗑️ Fetching all SonarQube projects...")

    page = 1
    page_size = 500
    keys: list[str] = []

    while True:
        response = SESSION.get(
            f"{SONAR_URL}/api/projects/search",
            params={"p": page, "ps": page_size},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code != 200:
            print(f"❌ Failed to fetch projects: {response.text}")
            return

        payload = response.json()
        components = payload.get("components", [])
        keys.extend(component["key"] for component in components)

        total = int(payload.get("paging", {}).get("total", len(keys)))
        if len(keys) >= total:
            break
        page += 1

    print(f"📌 Found {len(keys)} projects to delete.")

    for key in keys:
        response = SESSION.post(
            f"{SONAR_URL}/api/projects/delete",
            params={"project": key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code == 204:
            print(f"✔ Deleted {key}")
        else:
            print(
                f"❌ Failed to delete {key}: "
                f"HTTP {response.status_code} - {response.text}"
            )


def _to_int_or_none(value) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def repair_incomplete_results(output_csv: Path = OUTPUT_CSV) -> None:
    """Refresh rows whose stored Sonar metrics are all empty or zero."""
    if not output_csv.exists():
        print(f"❌ Output CSV not found: {output_csv}")
        return

    df = pd.read_csv(output_csv, on_bad_lines="skip")
    repaired = 0

    for index, row in df.iterrows():
        repo_url = row.get("repo_url")
        if not isinstance(repo_url, str) or not repo_url.startswith("http"):
            continue

        current = [_to_int_or_none(row.get(metric)) for metric in METRICS]
        if not all(value in (None, 0) for value in current):
            continue

        project_key = compute_project_key_from_url(repo_url)
        measures, error = fetch_metrics(project_key)
        if measures is None:
            print(f"⚠️ Could not repair {repo_url}: {error}")
            continue

        sonar_values = [_to_int_or_none(measures.get(metric)) for metric in METRICS]
        if not any(value not in (None, 0) for value in sonar_values):
            continue

        for metric in METRICS:
            df.at[index, metric] = measures.get(metric, "0")

        owner, repo_name = parse_repository_url(repo_url)
        repo_id = safe_identifier(f"{owner}__{repo_name}")
        severity_counts, issues_error = get_vulnerability_details(
            project_key,
            repo_id,
        )
        if not issues_error:
            for severity in SEVERITIES:
                column = f"vuln_{severity.lower()}"
                if column in df.columns:
                    df.at[index, column] = severity_counts.get(severity, 0)

        repaired += 1

    df.to_csv(output_csv, index=False)
    print(f"💾 Repair complete. Updated {repaired} row(s).")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def validate_configuration() -> None:
    if not SONAR_TOKEN:
        raise RuntimeError(
            "SONAR_TOKEN is not set. Run: export SONAR_TOKEN='your-token'"
        )

    if not SONAR_SCANNER.is_file():
        raise FileNotFoundError(f"SonarScanner not found: {SONAR_SCANNER}")

    if not _command_exists("git"):
        raise FileNotFoundError("The 'git' command is not installed or not on PATH.")


def main() -> None:
    validate_configuration()

    repo_urls = load_repository_urls()
    processed_urls = load_processed_urls()

    print(f"📥 Selected repositories: {len(repo_urls)}")
    print(f"📄 Already processed successfully: {len(processed_urls)}")
    print(f"📤 Output CSV: {OUTPUT_CSV}")
    print(f"⚠️ Failure CSV: {FAILURES_CSV}")

    failures_this_run: list[str] = []

    for index, git_url in enumerate(repo_urls, start=1):
        print(f"\n=== ({index}/{len(repo_urls)}) {git_url} ===")

        if git_url in processed_urls:
            print("⏩ Skipping: already processed successfully.")
            continue

        result = analyze_repo(git_url)

        if result is None:
            failures_this_run.append(git_url)
            continue

        append_csv_row(OUTPUT_CSV, result)
        processed_urls.add(git_url)
        print("💾 Saved result.")

    print(f"\n✅ Successful output: {OUTPUT_CSV}")
    print(f"⚠️ Failed in this run: {len(failures_this_run)}")
    if failures_this_run:
        print(f"See details in: {FAILURES_CSV}")


if __name__ == "__main__":
    # Optional maintenance actions:
    # delete_all_sonar_projects()
    # repair_incomplete_results()

    main()