#!/usr/bin/env python3
"""
smells_analysis_csharp_regular_repositories.py

Analyze the regular/generated C# repositories listed in the ``original_link``
column of:

    sample/persistence_sample_balanced/C#/eligible_candidates.csv

Example input row:

    original_link:
        https://github.com/andremorata/dotnet-clean-architecture-model

    target_template_repository:
        ardalis/CleanArchitecture

The script clones and analyzes ``original_link``. The
``target_template_repository`` and ``template_slug`` columns are retained only
as metadata linking the regular repository to its source template.

Repositories are filtered using:

    commit_count >= 2
    is_template == False
    data_update >= 2023-10-18

When an ``eligible`` column is present, only rows with eligible == True are
selected.

Required environment variable:

    SONAR_TOKEN

Optional environment variables:

    SONAR_URL
    DOTNET_ROOT
    DOTNET_COMMAND
    DOTNET_TOOLS_DIR
    DOTNET_SONAR_SCANNER

Example:

    export SONAR_TOKEN="your-token"
    export SONAR_URL="http://localhost:9002"
    export DOTNET_ROOT="$HOME/.dotnet"
    export DOTNET_COMMAND="$HOME/.dotnet/dotnet"
    export DOTNET_SONAR_SCANNER="$HOME/.dotnet/tools/dotnet-sonarscanner"

    python smells_analysis_csharp_regular_repositories.py
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

PROGRAMMING_LANGUAGE = "Csharp"
OUTPUT_LANGUAGE_NAME = "CSharp"

SONAR_URL = os.getenv("SONAR_URL", "http://localhost:9002").rstrip("/")
SONAR_TOKEN = "squ_f75043d923b1b7855a7c6e44ba031a9997df5881"

DOTNET_ROOT = Path(
    os.getenv("DOTNET_ROOT", str(Path.home() / ".dotnet"))
).expanduser().resolve()

DOTNET_COMMAND = os.getenv(
    "DOTNET_COMMAND",
    str(DOTNET_ROOT / "dotnet"),
)

DOTNET_TOOLS_DIR = Path(
    os.getenv("DOTNET_TOOLS_DIR", str(DOTNET_ROOT / "tools"))
).expanduser().resolve()

DOTNET_SONAR_SCANNER = os.getenv(
    "DOTNET_SONAR_SCANNER",
    str(DOTNET_TOOLS_DIR / "dotnet-sonarscanner"),
)

# Ensure that every subprocess sees the same private .NET installation,
# including when the script is launched from PyCharm.
SUBPROCESS_ENV = os.environ.copy()
SUBPROCESS_ENV["DOTNET_ROOT"] = str(DOTNET_ROOT)
SUBPROCESS_ENV["DOTNET_ROOT_X64"] = str(DOTNET_ROOT)

_existing_path = SUBPROCESS_ENV.get("PATH", "")
_required_path_entries = [
    str(DOTNET_ROOT),
    str(DOTNET_TOOLS_DIR),
]
SUBPROCESS_ENV["PATH"] = os.pathsep.join(
    _required_path_entries
    + ([_existing_path] if _existing_path else [])
)

REPOSITORY_COLUMN = "original_link"
TEMPLATE_COLUMN = "target_template_repository"

REPO_DIR = Path("temp_repos_csharp")

SAMPLE_ROOT = Path("sample/persistence_sample_balanced")
INPUT_CSV = SAMPLE_ROOT / PROGRAMMING_LANGUAGE / "eligible_candidates.csv"

OUTPUT_ROOT = Path("vulnerability_issues") / OUTPUT_LANGUAGE_NAME
OUTPUT_CSV = OUTPUT_ROOT / f"vulnerability_results_{OUTPUT_LANGUAGE_NAME}.csv"
FAILURES_CSV = OUTPUT_ROOT / f"analysis_failures_{OUTPUT_LANGUAGE_NAME}.csv"
JSON_OUTPUT_DIR = OUTPUT_ROOT / "vulnerability_details"
LOG_DIR = OUTPUT_ROOT / "logs"

START_DATE = pd.Timestamp("2023-10-18", tz="UTC")
MIN_COMMIT_COUNT = 2

METRICS = [
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

COMMAND_TIMEOUT_SECONDS = 30 * 60
CE_TIMEOUT_SECONDS = 5 * 60
CE_POLL_INTERVAL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 60

SKIP_WALK_DIRS = {
    ".git",
    ".github",
    ".scannerwork",
    ".sonarqube",
    ".idea",
    ".vscode",
    "node_modules",
    "bin",
    "obj",
    "packages",
    "artifacts",
    "TestResults",
}

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
class Repository:
    owner: str
    repo_name: str
    slug: str
    canonical_slug: str
    clone_url: str
    target_template_repository: str
    template_slug: str
    source_row_number: str
    duplicate_input_rows: int = 1


@dataclass
class BuildResult:
    ok: bool
    target: Optional[str]
    output: str
    reason: str


@dataclass
class SonarResult:
    ok: bool
    output: str
    build_target: Optional[str]
    reason: str


# -----------------------------------------------------------------------------
# Command execution
# -----------------------------------------------------------------------------


def _safe_command_text(command: Iterable[str]) -> str:
    """Return a printable command with SonarQube credentials masked."""
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
        command_env = SUBPROCESS_ENV.copy()
        if env:
            command_env.update(env)

        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=command_env,
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

        output += f"\nERROR: Command timed out after {timeout} seconds.\n"
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


def write_log(repo_id: str, stage: str, content: str) -> Path:
    """Save a command or analysis log."""
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
    Extract owner and repository name from a GitHub URL or owner/repository slug.
    """
    value = str(repository_reference).strip()

    if not value:
        raise ValueError("Empty repository reference.")

    if value.lower().startswith("git@github.com:"):
        repository_path = value.split(":", 1)[1]

    elif "://" in value:
        parsed = urlparse(value)
        hostname = (parsed.hostname or "").lower()

        if hostname not in {"github.com", "www.github.com"}:
            raise ValueError(
                "Only GitHub repositories are supported: "
                f"{repository_reference}"
            )

        repository_path = parsed.path.strip("/")

    else:
        repository_path = value.strip("/")

        if repository_path.lower().startswith("github.com/"):
            repository_path = repository_path[len("github.com/") :]
        elif repository_path.lower().startswith("www.github.com/"):
            repository_path = repository_path[len("www.github.com/") :]

    repository_path = repository_path.split("?", 1)[0]
    repository_path = repository_path.split("#", 1)[0]
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
            "Repository reference must use the format "
            f"'owner/repository': {repository_reference}"
        )

    owner, repo_name = parts
    return owner, repo_name


def build_github_clone_url(owner: str, repo_name: str) -> str:
    return f"https://github.com/{owner}/{repo_name}.git"


def canonical_repository_slug(repository_reference: str) -> str:
    owner, repo_name = parse_repository_reference(repository_reference)
    return f"{owner}/{repo_name}".lower()


def safe_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def compute_project_key(owner: str, repo_name: str) -> str:
    return f"{owner}__{repo_name}".lower()


def compute_project_key_from_reference(repository_reference: str) -> str:
    owner, repo_name = parse_repository_reference(repository_reference)
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


# -----------------------------------------------------------------------------
# Input loading
# -----------------------------------------------------------------------------


def _optional_text(row: pd.Series, column: str) -> str:
    """Return a stripped optional CSV value."""
    if column not in row.index:
        return ""

    value = row.get(column)

    if pd.isna(value):
        return ""

    return str(value).strip()


def load_repositories() -> list[Repository]:
    """
    Load, filter, and deduplicate regular repositories from ``original_link``.

    ``target_template_repository`` is not cloned. It is retained as metadata so
    the resulting project can later be paired with its associated template.
    """
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV, on_bad_lines="skip")

    required_columns = {
        REPOSITORY_COLUMN,
        "commit_count",
        "is_template",
        "data_update",
    }

    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise KeyError(f"Missing required input columns: {missing}")

    df["commit_count"] = pd.to_numeric(
        df["commit_count"],
        errors="coerce",
    )

    df["data_update"] = pd.to_datetime(
        df["data_update"],
        errors="coerce",
        utc=True,
    )

    parsed_is_template = df["is_template"].map(_parse_boolean)

    mask = (
        (df["commit_count"] >= MIN_COMMIT_COUNT)
        & (parsed_is_template == False)  # noqa: E712
        & (df["data_update"] >= START_DATE)
    )

    if "eligible" in df.columns:
        parsed_eligible = df["eligible"].map(_parse_boolean)
        mask &= parsed_eligible == True  # noqa: E712

    selected_df = df.loc[mask].copy()

    repositories_by_slug: dict[str, Repository] = {}
    invalid_references: list[str] = []
    conflicting_template_mappings: list[str] = []

    for _, row in selected_df.iterrows():
        repository_reference = _optional_text(row, REPOSITORY_COLUMN)

        if not repository_reference:
            continue

        try:
            owner, repo_name = parse_repository_reference(
                repository_reference
            )
        except ValueError as exc:
            invalid_references.append(repository_reference)
            print(
                f"⚠️ Ignoring invalid {REPOSITORY_COLUMN} value "
                f"'{repository_reference}': {exc}"
            )
            continue

        slug = f"{owner}/{repo_name}"
        canonical_slug = slug.lower()

        target_template_repository = _optional_text(
            row,
            TEMPLATE_COLUMN,
        )
        template_slug = _optional_text(
            row,
            "template_slug",
        )

        if not template_slug and target_template_repository:
            try:
                template_owner, template_repo = parse_repository_reference(
                    target_template_repository
                )
                template_slug = f"{template_owner}/{template_repo}".lower()
            except ValueError:
                template_slug = target_template_repository.lower()

        source_row_number = _optional_text(
            row,
            "_source_row_number",
        )

        existing = repositories_by_slug.get(canonical_slug)

        if existing is not None:
            existing.duplicate_input_rows += 1

            existing_template = (
                existing.template_slug
                or existing.target_template_repository.lower()
            )
            new_template = (
                template_slug
                or target_template_repository.lower()
            )

            if (
                existing_template
                and new_template
                and existing_template != new_template
            ):
                conflicting_template_mappings.append(
                    f"{canonical_slug}: "
                    f"{existing_template} != {new_template}"
                )

            continue

        repositories_by_slug[canonical_slug] = Repository(
            owner=owner,
            repo_name=repo_name,
            slug=slug,
            canonical_slug=canonical_slug,
            clone_url=build_github_clone_url(owner, repo_name),
            target_template_repository=target_template_repository,
            template_slug=template_slug,
            source_row_number=source_row_number,
        )

    repositories = list(repositories_by_slug.values())

    print(f"📥 Total input rows: {len(df)}")
    print(f"📥 Eligible regular-repository rows: {int(mask.sum())}")
    print(f"📦 Unique regular repositories: {len(repositories)}")

    if invalid_references:
        print(
            "⚠️ Invalid original_link values: "
            f"{len(invalid_references)}"
        )

    if conflicting_template_mappings:
        print(
            "⚠️ Duplicate repositories with conflicting template mappings: "
            f"{len(conflicting_template_mappings)}"
        )
        for conflict in conflicting_template_mappings[:10]:
            print(f"   - {conflict}")

    return repositories


# -----------------------------------------------------------------------------
# C# project discovery and build
# -----------------------------------------------------------------------------


def project_contains_csharp(repo_path: Path) -> bool:
    """Return True when the repository contains at least one .cs file."""
    for _, dirs, files in os.walk(repo_path):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in SKIP_WALK_DIRS
        ]

        if any(filename.lower().endswith(".cs") for filename in files):
            return True

    return False


def _relative_depth(path: Path, base: Path) -> int:
    return len(path.resolve().relative_to(base.resolve()).parts)


def find_dotnet_files(
    repo_path: Path,
) -> tuple[list[Path], list[Path]]:
    """Find solution and C# project files, preferring shallower paths."""
    solution_files: list[Path] = []
    project_files: list[Path] = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [
            directory
            for directory in dirs
            if directory not in SKIP_WALK_DIRS
        ]

        root_path = Path(root)

        for filename in files:
            lower = filename.lower()

            if lower.endswith(".sln"):
                solution_files.append(root_path / filename)
            elif lower.endswith(".csproj"):
                project_files.append(root_path / filename)

    solution_files.sort(
        key=lambda path: (
            _relative_depth(path.parent, repo_path),
            path.as_posix().lower(),
        )
    )

    project_files.sort(
        key=lambda path: (
            _relative_depth(path.parent, repo_path),
            path.as_posix().lower(),
        )
    )

    return solution_files, project_files


def create_temporary_solution(
    repo_path: Path,
    project_files: list[Path],
) -> tuple[Optional[str], str]:
    """
    Create sonar_temp.sln and add all discovered .csproj files.

    Returns:
        (relative solution path or None, combined command output)
    """
    temp_solution_name = "sonar_temp.sln"
    temp_solution_path = repo_path / temp_solution_name

    if temp_solution_path.exists():
        temp_solution_path.unlink()

    outputs: list[str] = []

    print("🧩 No .sln found. Creating temporary solution sonar_temp.sln...")

    create_result = run_cmd(
        [DOTNET_COMMAND, "new", "sln", "-n", "sonar_temp"],
        cwd=repo_path,
    )
    outputs.append(create_result.output)

    if not create_result.ok:
        return None, "\n\n".join(outputs)

    added_projects = 0

    for project_path in project_files:
        relative_project = project_path.relative_to(repo_path).as_posix()

        add_result = run_cmd(
            [
                DOTNET_COMMAND,
                "sln",
                temp_solution_name,
                "add",
                relative_project,
            ],
            cwd=repo_path,
        )

        outputs.append(
            f"=== ADD {relative_project} ===\n{add_result.output}"
        )

        if add_result.ok:
            added_projects += 1
        else:
            print(
                "⚠️ Could not add project to temporary solution: "
                f"{relative_project}"
            )

    if added_projects == 0:
        return None, "\n\n".join(outputs)

    return temp_solution_name, "\n\n".join(outputs)


def choose_dotnet_build_target(
    repo_path: Path,
) -> tuple[Optional[str], str]:
    """
    Select a .sln or .csproj build target.

    Existing solutions are preferred. If no solution exists, the script tries
    to create a temporary solution containing all discovered .csproj files. If
    that fails, it falls back to the shallowest .csproj file.
    """
    solution_files, project_files = find_dotnet_files(repo_path)

    if solution_files:
        chosen = solution_files[0].relative_to(repo_path).as_posix()
        print(f"🧩 Using existing solution: {chosen}")
        return chosen, ""

    if not project_files:
        return None, ""

    temporary_solution, creation_output = create_temporary_solution(
        repo_path,
        project_files,
    )

    if temporary_solution:
        print(f"🧩 Using temporary solution: {temporary_solution}")
        return temporary_solution, creation_output

    chosen = project_files[0].relative_to(repo_path).as_posix()
    print(
        "⚠️ Temporary solution creation failed. "
        f"Using project file directly: {chosen}"
    )

    return chosen, creation_output


def build_csharp_project(repo_path: Path) -> BuildResult:
    """Restore and build the selected C# solution or project."""
    target, preparation_output = choose_dotnet_build_target(repo_path)

    if target is None:
        return BuildResult(
            ok=False,
            target=None,
            output=preparation_output,
            reason="No .sln or .csproj file was found.",
        )

    outputs: list[str] = []

    if preparation_output:
        outputs.append(
            "=== BUILD TARGET PREPARATION ===\n"
            f"{preparation_output}"
        )

    print(f"📦 Restoring NuGet packages for {target}...")

    restore = run_cmd(
        [DOTNET_COMMAND, "restore", target],
        cwd=repo_path,
    )
    outputs.append(f"=== DOTNET RESTORE ===\n{restore.output}")

    if not restore.ok:
        print(
            "⚠️ 'dotnet restore' failed. "
            "The script will still attempt the build."
        )

    print(f"🏗️ Building C# target {target}...")

    build = run_cmd(
        [
            DOTNET_COMMAND,
            "build",
            target,
            "-c",
            "Release",
            "--no-incremental",
            "/p:SkipTests=true",
        ],
        cwd=repo_path,
    )
    outputs.append(f"=== DOTNET BUILD ===\n{build.output}")

    if not build.ok:
        reason = "'dotnet build' failed."
        if build.timed_out:
            reason += " The build command timed out."

        return BuildResult(
            ok=False,
            target=target,
            output="\n\n".join(outputs),
            reason=reason,
        )

    print("✅ C# build succeeded.")

    return BuildResult(
        ok=True,
        target=target,
        output="\n\n".join(outputs),
        reason="",
    )


# -----------------------------------------------------------------------------
# SonarScanner for .NET
# -----------------------------------------------------------------------------


def _command_exists(command: str) -> bool:
    path = Path(command).expanduser()

    if os.path.sep in command:
        return path.is_file() and os.access(path, os.X_OK)

    return shutil.which(command) is not None


def remove_sonar_workdirs(repo_path: Path) -> None:
    """Remove scanner state that may remain after a failed attempt."""
    shutil.rmtree(repo_path / ".sonarqube", ignore_errors=True)
    shutil.rmtree(repo_path / ".scannerwork", ignore_errors=True)


def run_sonar_for_csharp(
    *,
    project_key: str,
    project_name: str,
    repo_path: Path,
) -> SonarResult:
    """
    Run the SonarScanner for .NET pipeline:

        begin -> restore/build -> end
    """
    print(f"🔧 Using Sonar scanner: {DOTNET_SONAR_SCANNER}")

    if not _command_exists(DOTNET_SONAR_SCANNER):
        return SonarResult(
            ok=False,
            output="",
            build_target=None,
            reason=(
                f"SonarScanner for .NET was not found or is not executable: "
                f"{DOTNET_SONAR_SCANNER}"
            ),
        )

    remove_sonar_workdirs(repo_path)

    outputs: list[str] = []

    begin_command = [
        DOTNET_SONAR_SCANNER,
        "begin",
        f"/k:{project_key}",
        f"/n:{project_name}",
        f"/d:sonar.host.url={SONAR_URL}",
        f"/d:sonar.login={SONAR_TOKEN}",
    ]

    begin_result = run_cmd(
        begin_command,
        cwd=repo_path,
    )
    outputs.append(f"=== SONAR BEGIN ===\n{begin_result.output}")

    if not begin_result.ok:
        reason = "SonarScanner 'begin' step failed."
        if begin_result.timed_out:
            reason += " The begin command timed out."

        return SonarResult(
            ok=False,
            output="\n\n".join(outputs),
            build_target=None,
            reason=reason,
        )

    build_result = build_csharp_project(repo_path)
    outputs.append(
        f"=== BUILD INSIDE SONAR CONTEXT ===\n{build_result.output}"
    )

    if not build_result.ok:
        return SonarResult(
            ok=False,
            output="\n\n".join(outputs),
            build_target=build_result.target,
            reason=(
                "Build inside the SonarScanner context failed. "
                f"{build_result.reason}"
            ),
        )

    end_command = [
        DOTNET_SONAR_SCANNER,
        "end",
        f"/d:sonar.login={SONAR_TOKEN}",
    ]

    end_result = run_cmd(
        end_command,
        cwd=repo_path,
    )
    outputs.append(f"=== SONAR END ===\n{end_result.output}")

    if not end_result.ok:
        reason = "SonarScanner 'end' step failed."
        if end_result.timed_out:
            reason += " The end command timed out."

        return SonarResult(
            ok=False,
            output="\n\n".join(outputs),
            build_target=build_result.target,
            reason=reason,
        )

    return SonarResult(
        ok=True,
        output="\n\n".join(outputs),
        build_target=build_result.target,
        reason="",
    )


def locate_report_task(repo_path: Path) -> Optional[Path]:
    """
    Locate report-task.txt produced by SonarScanner for .NET.

    Common locations include:
        .sonarqube/out/.sonar/report-task.txt
        .scannerwork/report-task.txt
    """
    direct_candidates = [
        repo_path / ".sonarqube" / "out" / ".sonar" / "report-task.txt",
        repo_path / ".scannerwork" / "report-task.txt",
    ]

    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    candidates = list(repo_path.rglob("report-task.txt"))

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def read_report_task(repo_path: Path) -> dict[str, str]:
    report_file = locate_report_task(repo_path)

    if report_file is None:
        return {}

    values: dict[str, str] = {}

    for line in report_file.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    return values


def wait_for_compute_engine(repo_path: Path) -> tuple[bool, str]:
    """Wait for the SonarQube Compute Engine task to finish."""
    report = read_report_task(repo_path)
    task_url = report.get("ceTaskUrl")

    if not task_url:
        return (
            False,
            "SonarQube report-task.txt did not contain ceTaskUrl.",
        )

    deadline = time.monotonic() + CE_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        try:
            response = SESSION.get(
                task_url,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            return (
                False,
                f"Failed to query SonarQube Compute Engine: {exc}",
            )

        if response.status_code != 200:
            return (
                False,
                "Failed to query SonarQube Compute Engine: "
                f"HTTP {response.status_code} - {response.text}",
            )

        task = response.json().get("task", {})
        status = str(task.get("status", "")).upper()

        if status == "SUCCESS":
            return True, ""

        if status in {"FAILED", "CANCELED"}:
            error_message = task.get("errorMessage", "")
            return (
                False,
                f"Compute Engine status {status}: {error_message}",
            )

        time.sleep(CE_POLL_INTERVAL_SECONDS)

    return (
        False,
        f"Compute Engine did not finish within {CE_TIMEOUT_SECONDS} seconds.",
    )


# -----------------------------------------------------------------------------
# SonarQube API collection
# -----------------------------------------------------------------------------


def fetch_metrics(
    project_key: str,
) -> tuple[Optional[dict[str, str]], str]:
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
        return (
            None,
            "Metrics request failed: "
            f"HTTP {response.status_code} - {response.text}",
        )

    measures = {
        measure["metric"]: measure.get("value", "0")
        for measure in (
            response.json()
            .get("component", {})
            .get("measures", [])
        )
    }

    return (
        {
            metric: measures.get(metric, "0")
            for metric in METRICS
        },
        "",
    )


def get_vulnerability_details(
    project_key: str,
    repo_id: str,
) -> tuple[dict[str, int], str]:
    """Fetch all vulnerability issues and save their raw JSON representation."""
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
            return (
                {},
                "Issues request failed: "
                f"HTTP {response.status_code} - {response.text}",
            )

        payload = response.json()
        issues = payload.get("issues", [])
        all_issues.extend(issues)

        total = int(
            payload.get("paging", {}).get(
                "total",
                len(all_issues),
            )
        )

        if not issues or len(all_issues) >= total:
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
        f"💾 Saved {len(all_issues)} vulnerability issues to {json_path}"
    )

    severity_counts = Counter(
        str(issue.get("severity", "UNKNOWN")).upper()
        for issue in all_issues
    )

    return dict(severity_counts), ""


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
    repository: Repository,
    stage: str,
    reason: str,
    build_target: str = "",
    log_path: Optional[Path] = None,
) -> None:
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_url": repository.clone_url,
        "original_link": repository.clone_url.removesuffix(".git"),
        "canonical_repository": repository.canonical_slug,
        "owner": repository.owner,
        "repo": repository.repo_name,
        "target_template_repository": (
            repository.target_template_repository
        ),
        "template_slug": repository.template_slug,
        "source_row_number": repository.source_row_number,
        "duplicate_input_rows": repository.duplicate_input_rows,
        "stage": stage,
        "reason": reason,
        "build_target": build_target,
        "log_path": str(log_path) if log_path else "",
    }

    append_csv_row(FAILURES_CSV, row)


def load_processed_repositories() -> set[str]:
    """Load regular repositories already analyzed successfully."""
    if not OUTPUT_CSV.exists():
        return set()

    existing = pd.read_csv(OUTPUT_CSV, on_bad_lines="skip")
    processed: set[str] = set()

    if "canonical_repository" in existing.columns:
        values = (
            existing["canonical_repository"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        processed.update(value for value in values if value)
        return processed

    for column in ["original_link", "repo_url"]:
        if column not in existing.columns:
            continue

        for value in existing[column].dropna().astype(str):
            try:
                processed.add(canonical_repository_slug(value))
            except ValueError:
                continue

        if processed:
            break

    return processed


# -----------------------------------------------------------------------------
# Repository analysis
# -----------------------------------------------------------------------------


def analyze_repository(
    repository: Repository,
) -> Optional[dict]:
    """Clone and analyze one regular C# repository."""
    repo_id = safe_identifier(
        f"{repository.owner}__{repository.repo_name}"
    )
    repo_path = REPO_DIR / repo_id

    try:
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)

        print(f"📦 Regular repository: {repository.slug}")

        if repository.target_template_repository:
            print(
                "🧩 Associated template: "
                f"{repository.target_template_repository}"
            )

        clone_result = run_cmd(
            [
                "git",
                "clone",
                "--depth",
                "1",
                repository.clone_url,
                str(repo_path),
            ]
        )

        if not clone_result.ok:
            log_path = write_log(
                repo_id,
                "clone",
                clone_result.output,
            )

            record_failure(
                repository=repository,
                stage="clone",
                reason="Git clone failed.",
                log_path=log_path,
            )
            return None

        if not project_contains_csharp(repo_path):
            record_failure(
                repository=repository,
                stage="precheck",
                reason="No C# source file was found.",
            )
            return None

        project_key = compute_project_key(
            repository.owner,
            repository.repo_name,
        )

        sonar_result = run_sonar_for_csharp(
            project_key=project_key,
            project_name=repository.slug,
            repo_path=repo_path,
        )

        sonar_log_path = write_log(
            repo_id,
            "sonar",
            sonar_result.output,
        )

        if not sonar_result.ok:
            record_failure(
                repository=repository,
                stage="sonar_analysis",
                reason=sonar_result.reason,
                build_target=sonar_result.build_target or "",
                log_path=sonar_log_path,
            )
            return None

        ce_ok, ce_error = wait_for_compute_engine(repo_path)

        if not ce_ok:
            record_failure(
                repository=repository,
                stage="compute_engine",
                reason=ce_error,
                build_target=sonar_result.build_target or "",
                log_path=sonar_log_path,
            )
            return None

        measures, metrics_error = fetch_metrics(project_key)

        if measures is None:
            record_failure(
                repository=repository,
                stage="metrics_api",
                reason=metrics_error,
                build_target=sonar_result.build_target or "",
                log_path=sonar_log_path,
            )
            return None

        severity_counts, issues_error = get_vulnerability_details(
            project_key,
            repo_id,
        )

        if issues_error:
            record_failure(
                repository=repository,
                stage="issues_api",
                reason=issues_error,
                build_target=sonar_result.build_target or "",
                log_path=sonar_log_path,
            )
            return None

        result = {
            "repo_url": repository.clone_url,
            "original_link": repository.clone_url.removesuffix(".git"),
            "canonical_repository": repository.canonical_slug,
            "owner": repository.owner,
            "repo": repository.repo_name,
            "project_key": project_key,
            "target_template_repository": (
                repository.target_template_repository
            ),
            "template_slug": repository.template_slug,
            "source_row_number": repository.source_row_number,
            "duplicate_input_rows": repository.duplicate_input_rows,
            "analysis_status": "success",
            "build_target": sonar_result.build_target or "",
            **measures,
        }

        for severity in SEVERITIES:
            result[f"vuln_{severity.lower()}"] = severity_counts.get(
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
            repository=repository,
            stage="unexpected",
            reason=f"Unexpected error: {exc}",
            log_path=log_path,
        )
        return None

    finally:
        shutil.rmtree(repo_path, ignore_errors=True)
        print(f"🧹 Deleted local clone of {repository.slug}")


# -----------------------------------------------------------------------------
# Optional maintenance utilities
# -----------------------------------------------------------------------------


def delete_all_sonar_projects() -> None:
    """Delete every project from the configured SonarQube instance."""
    print("🗑️ Fetching all SonarQube projects...")

    page = 1
    page_size = 500
    project_keys: list[str] = []

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
            print(f"❌ Failed to fetch projects: {response.text}")
            return

        payload = response.json()
        components = payload.get("components", [])

        project_keys.extend(
            component["key"]
            for component in components
        )

        total = int(
            payload.get("paging", {}).get(
                "total",
                len(project_keys),
            )
        )

        if len(project_keys) >= total:
            break

        page += 1

    print(f"📌 Found {len(project_keys)} projects to delete.")

    for project_key in project_keys:
        response = SESSION.post(
            f"{SONAR_URL}/api/projects/delete",
            params={"project": project_key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        if response.status_code == 204:
            print(f"✔ Deleted {project_key}")
        else:
            print(
                f"❌ Failed to delete {project_key}: "
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


def repair_incomplete_results(
    output_csv: Path = OUTPUT_CSV,
) -> None:
    """
    Refresh rows whose stored metrics are all empty or zero, using SonarQube as
    the source of truth.
    """
    if not output_csv.exists():
        print(f"❌ Output CSV not found: {output_csv}")
        return

    df = pd.read_csv(output_csv, on_bad_lines="skip")
    repaired = 0

    for index, row in df.iterrows():
        repository_reference = row.get("original_link")

        if (
            not isinstance(repository_reference, str)
            or not repository_reference.strip()
        ):
            repository_reference = row.get("repo_url")

        if (
            not isinstance(repository_reference, str)
            or not repository_reference.strip()
        ):
            continue

        current_values = [
            _to_int_or_none(row.get(metric))
            for metric in METRICS
        ]

        if not all(
            value in (None, 0)
            for value in current_values
        ):
            continue

        try:
            owner, repo_name = parse_repository_reference(
                repository_reference
            )
        except ValueError as exc:
            print(
                "⚠️ Could not interpret repository reference "
                f"'{repository_reference}': {exc}"
            )
            continue

        project_key = compute_project_key(owner, repo_name)
        measures, error = fetch_metrics(project_key)

        if measures is None:
            print(
                f"⚠️ Could not repair {repository_reference}: {error}"
            )
            continue

        sonar_values = [
            _to_int_or_none(measures.get(metric))
            for metric in METRICS
        ]

        if not any(
            value not in (None, 0)
            for value in sonar_values
        ):
            continue

        for metric in METRICS:
            df.at[index, metric] = measures.get(metric, "0")

        repo_id = safe_identifier(f"{owner}__{repo_name}")

        severity_counts, issues_error = get_vulnerability_details(
            project_key,
            repo_id,
        )

        if not issues_error:
            for severity in SEVERITIES:
                column = f"vuln_{severity.lower()}"

                if column in df.columns:
                    df.at[index, column] = severity_counts.get(
                        severity,
                        0,
                    )

        repaired += 1

    df.to_csv(output_csv, index=False)

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

    if not _command_exists("git"):
        raise FileNotFoundError(
            "The 'git' command is not installed or not available on PATH."
        )

    if not _command_exists(DOTNET_COMMAND):
        raise FileNotFoundError(
            "The .NET SDK executable was not found.\n"
            f"Configured DOTNET_COMMAND: {DOTNET_COMMAND}\n"
            f"Configured DOTNET_ROOT: {DOTNET_ROOT}"
        )

    if not _command_exists(DOTNET_SONAR_SCANNER):
        raise FileNotFoundError(
            "SonarScanner for .NET was not found or is not executable.\n"
            f"Configured scanner: {DOTNET_SONAR_SCANNER}\n\n"
            "Install it with:\n"
            f"    {DOTNET_COMMAND} tool install --global "
            "dotnet-sonarscanner"
        )

    print(f"🔧 DOTNET_ROOT: {DOTNET_ROOT}")
    print(f"🔧 DOTNET_COMMAND: {DOTNET_COMMAND}")
    print(f"🔧 DOTNET_TOOLS_DIR: {DOTNET_TOOLS_DIR}")
    print(f"🔧 DOTNET_SONAR_SCANNER: {DOTNET_SONAR_SCANNER}")

    dotnet_check = run_cmd(
        [DOTNET_COMMAND, "--info"],
        timeout=60,
    )

    if not dotnet_check.ok:
        raise RuntimeError(
            "The configured dotnet executable could not start correctly. "
            "See the output above."
        )

    scanner_check = run_cmd(
        [DOTNET_SONAR_SCANNER, "--version"],
        timeout=60,
    )

    # SonarScanner for .NET 11 prints its version successfully but may still
    # return a nonzero exit code because the executable expects either the
    # 'begin' or 'end' command. Therefore, validate the output instead of
    # relying only on the return code.
    scanner_output_lower = scanner_check.output.lower()

    runtime_failure_markers = [
        "you must install .net to run this application",
        ".net location: not found",
        "failed to resolve libhostfxr.so",
        "app-launch-failed",
        "missing_runtime=true",
    ]

    if any(
        marker in scanner_output_lower
        for marker in runtime_failure_markers
    ):
        raise RuntimeError(
            "SonarScanner for .NET could not locate the .NET runtime. "
            f"DOTNET_ROOT was set to {DOTNET_ROOT}. "
            "See the scanner output above."
        )

    if "sonarscanner for .net" not in scanner_output_lower:
        raise RuntimeError(
            "SonarScanner for .NET did not start as expected. "
            f"Configured scanner: {DOTNET_SONAR_SCANNER}. "
            "See the scanner output above."
        )

    print("✅ SonarScanner for .NET is available.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    validate_configuration()

    repositories = load_repositories()
    processed_repositories = load_processed_repositories()

    print(
        "📥 Selected unique regular repositories: "
        f"{len(repositories)}"
    )
    print(
        "📄 Already processed successfully: "
        f"{len(processed_repositories)}"
    )
    print(f"📤 Output CSV: {OUTPUT_CSV}")
    print(f"⚠️ Failure CSV: {FAILURES_CSV}")

    failures_this_run: list[str] = []

    for index, repository in enumerate(repositories, start=1):
        print(
            f"\n=== ({index}/{len(repositories)}) "
            f"{repository.slug} ==="
        )

        if repository.canonical_slug in processed_repositories:
            print(
                "⏩ Skipping: repository already processed successfully."
            )
            continue

        result = analyze_repository(repository)

        if result is None:
            failures_this_run.append(repository.slug)
            continue

        append_csv_row(OUTPUT_CSV, result)
        processed_repositories.add(repository.canonical_slug)

        print("💾 Saved repository result.")

    print(f"\n✅ Successful output: {OUTPUT_CSV}")
    print(f"⚠️ Failed in this run: {len(failures_this_run)}")

    if failures_this_run:
        print(f"See details in: {FAILURES_CSV}")


if __name__ == "__main__":
    # Optional maintenance operations:
    #
    # delete_all_sonar_projects()
    #
    # repair_incomplete_results()
    #
    # Comment out main() when running only a maintenance operation.

    main()
