#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

DEFAULT_SONAR_URL = "http://localhost:9002"

SONAR_SCANNER = Path("/opt/sonar-scanner/bin/sonar-scanner")

REPO_DIR = Path("./temp_template_repos")
JSON_OUTPUT_DIR = Path("vulnerability_issues/TypeScript/template_vulnerability_details")

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

TARGET_COLUMN = "target_template_repository"


# ----------------------------------------------------------------------
# GENERAL HELPERS
# ----------------------------------------------------------------------

def parse_boolean(value: Any) -> bool:
    """Convert common CSV boolean representations to bool."""
    if value is None or pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


def sanitize_filename(value: str) -> str:
    """Replace characters that are unsafe in file and directory names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def metric_to_number(value: Any) -> int | float:
    """Convert a SonarQube metric value to int or float."""
    if value is None:
        return 0

    try:
        number = float(value)

        if number.is_integer():
            return int(number)

        return number

    except (TypeError, ValueError):
        return 0


def run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
) -> bool:
    """Run a command and stream its combined output."""
    printable_command = " ".join(str(item) for item in cmd)
    print(f"Running: {printable_command}")

    result = subprocess.run(
        [str(item) for item in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    print(result.stdout)

    return result.returncode == 0


# ----------------------------------------------------------------------
# GITHUB REPOSITORY NORMALIZATION
# ----------------------------------------------------------------------

def normalize_github_repository(value: Any) -> str | None:
    """
    Convert a GitHub repository slug or URL to a canonical HTTPS URL.

    Supported examples:

        9d8dev/pay
        github.com/9d8dev/pay
        https://github.com/9d8dev/pay
        https://github.com/9d8dev/pay.git
        git@github.com:9d8dev/pay.git

    Result:

        https://github.com/9d8dev/pay
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    # Handle SSH format:
    # git@github.com:owner/repository.git
    if text.startswith("git@github.com:"):
        text = text.removeprefix("git@github.com:")

    # Handle URLs without protocol:
    # github.com/owner/repository
    elif text.startswith("github.com/"):
        text = text.removeprefix("github.com/")

    # Handle complete HTTP/HTTPS URLs.
    elif text.startswith(("http://", "https://")):
        parsed = urlparse(text)

        hostname = parsed.hostname or ""

        if hostname.lower() not in {
            "github.com",
            "www.github.com",
        }:
            print(
                f"⚠️ Ignoring non-GitHub repository value: {text}"
            )
            return None

        text = parsed.path

    text = text.strip("/")

    # Remove query parameters or fragments from malformed URL-like values.
    text = text.split("?", maxsplit=1)[0]
    text = text.split("#", maxsplit=1)[0]

    parts = [
        part.strip()
        for part in text.split("/")
        if part.strip()
    ]

    if len(parts) < 2:
        print(
            f"⚠️ Could not identify owner/repository from: {value}"
        )
        return None

    owner = parts[0]
    repository = parts[1]

    if repository.endswith(".git"):
        repository = repository[:-4]

    if not owner or not repository:
        return None

    return f"https://github.com/{owner}/{repository}"


def get_owner_and_repo(git_url: str) -> tuple[str, str]:
    """Extract the repository owner and name from a normalized GitHub URL."""
    normalized_url = normalize_github_repository(git_url)

    if normalized_url is None:
        raise ValueError(
            f"Invalid GitHub repository URL: {git_url}"
        )

    parsed = urlparse(normalized_url)
    parts = parsed.path.strip("/").split("/")

    if len(parts) != 2:
        raise ValueError(
            f"Could not extract owner and repository from: {git_url}"
        )

    owner, repository = parts

    return owner, repository


def compute_project_key(git_url: str) -> str:
    """
    Create the SonarQube project key used during scanning.

    Example:

        https://github.com/9d8dev/pay
        -> 9d8dev__pay
    """
    owner, repository = get_owner_and_repo(git_url)

    return f"{owner}__{repository}".lower()


# ----------------------------------------------------------------------
# CSV PROCESSING
# ----------------------------------------------------------------------

def collect_template_repositories(
    dataframe: pd.DataFrame,
    target_column: str,
    eligible_only: bool,
) -> dict[str, int]:
    """
    Collect unique template repositories from the target column.

    The returned dictionary maps each normalized GitHub URL to the number
    of sampled repositories that reference that template.
    """
    if target_column not in dataframe.columns:
        raise ValueError(
            f"Column '{target_column}' was not found in the CSV. "
            f"Available columns: {list(dataframe.columns)}"
        )

    selected_dataframe = dataframe.copy()

    if eligible_only:
        if "eligible" not in selected_dataframe.columns:
            print(
                "⚠️ --eligible-only was requested, but the CSV does not "
                "contain an 'eligible' column. All rows will be considered."
            )
        else:
            eligible_mask = selected_dataframe["eligible"].apply(
                parse_boolean
            )

            selected_dataframe = selected_dataframe.loc[
                eligible_mask
            ].copy()

            print(
                f"📌 Rows with eligible=True: "
                f"{len(selected_dataframe)}"
            )

    template_counts: dict[str, int] = {}
    invalid_values = 0

    for value in selected_dataframe[target_column]:
        normalized_url = normalize_github_repository(value)

        if normalized_url is None:
            invalid_values += 1
            continue

        # Use lowercase only for duplicate detection, while preserving the
        # first URL representation in the output.
        normalized_key = normalized_url.lower()

        existing_url = next(
            (
                existing
                for existing in template_counts
                if existing.lower() == normalized_key
            ),
            None,
        )

        if existing_url is None:
            template_counts[normalized_url] = 1
        else:
            template_counts[existing_url] += 1

    print(
        f"📄 Valid target-template references: "
        f"{sum(template_counts.values())}"
    )
    print(
        f"📦 Unique template repositories: "
        f"{len(template_counts)}"
    )

    if invalid_values:
        print(
            f"⚠️ Empty or invalid target-template values: "
            f"{invalid_values}"
        )

    return template_counts


# ----------------------------------------------------------------------
# SONARQUBE API
# ----------------------------------------------------------------------

def get_vulnerability_details(
    session: requests.Session,
    project_key: str,
    owner: str,
    repository: str,
    sonar_url: str,
    json_output_dir: Path,
) -> dict[str, int]:
    """Fetch vulnerability issues and return their severity counts."""
    endpoint = f"{sonar_url.rstrip('/')}/api/issues/search"

    page = 1
    page_size = 500
    all_issues: list[dict[str, Any]] = []

    while True:
        response = session.get(
            endpoint,
            params={
                "componentKeys": project_key,
                "types": "VULNERABILITY",
                "p": page,
                "ps": page_size,
            },
            timeout=60,
        )

        if response.status_code != 200:
            print(
                f"⚠️ Failed to get vulnerability issues for "
                f"{owner}/{repository}: "
                f"{response.status_code} — {response.text}"
            )
            break

        payload = response.json()
        current_issues = payload.get("issues", [])

        if not isinstance(current_issues, list):
            print(
                f"⚠️ Unexpected issues response for "
                f"{owner}/{repository}"
            )
            break

        all_issues.extend(current_issues)

        paging = payload.get("paging", {})
        total = int(paging.get("total", len(all_issues)))

        print(
            f"    Vulnerability page {page}: "
            f"{len(current_issues)} issues "
            f"({len(all_issues)}/{total})"
        )

        if len(all_issues) >= total:
            break

        if not current_issues:
            break

        page += 1

    json_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_name = sanitize_filename(
        f"{owner}__{repository}_vulnerabilities.json"
    )

    json_path = json_output_dir / json_name

    json_path.write_text(
        json.dumps(
            all_issues,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"💾 Saved {len(all_issues)} vulnerability issues "
        f"to {json_path}"
    )

    severities = [
        issue.get("severity", "UNKNOWN")
        for issue in all_issues
    ]

    return dict(Counter(severities))


def get_project_metrics(
    session: requests.Session,
    project_key: str,
    sonar_url: str,
) -> dict[str, int | float]:
    """Fetch the configured SonarQube metrics for one project."""
    endpoint = f"{sonar_url.rstrip('/')}/api/measures/component"

    response = session.get(
        endpoint,
        params={
            "component": project_key,
            "metricKeys": ",".join(METRICS),
        },
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    raw_measures = {
        measure["metric"]: measure.get("value", 0)
        for measure in (
            payload
            .get("component", {})
            .get("measures", [])
        )
    }

    # Ensure every metric is present in the output CSV.
    return {
        metric: metric_to_number(
            raw_measures.get(metric, 0)
        )
        for metric in METRICS
    }


# ----------------------------------------------------------------------
# REPOSITORY ANALYSIS
# ----------------------------------------------------------------------

def analyze_repo(
    git_url: str,
    template_reference_count: int,
    session: requests.Session,
    sonar_url: str,
    sonar_token: str,
    repo_dir: Path,
    json_output_dir: Path,
    wait_seconds: int,
) -> dict[str, Any] | None:
    """
    Clone, scan and collect SonarQube results for one template repository.
    """
    owner, repository = get_owner_and_repo(git_url)

    project_key = compute_project_key(git_url)

    local_directory_name = sanitize_filename(
        f"{owner}__{repository}"
    )

    repo_path = repo_dir / local_directory_name

    if repo_path.exists():
        print(
            f"🧹 Removing an existing local directory: {repo_path}"
        )
        shutil.rmtree(repo_path, ignore_errors=True)

    repo_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        print(f"📥 Cloning template {owner}/{repository}")

        clone_succeeded = run_cmd(
            [
                "git",
                "clone",
                "--depth",
                "1",
                git_url,
                str(repo_path),
            ]
        )

        if not clone_succeeded:
            print(f"⚠️ Failed to clone {git_url}")
            return None

        sonar_command = [
            str(SONAR_SCANNER),
            f"-Dsonar.projectKey={project_key}",
            f"-Dsonar.projectName={owner}/{repository}",
            "-Dsonar.sources=.",
            "-Dsonar.scm.exclusions.disabled=true",
            "-Dsonar.inclusions=**/*",
            "-Dsonar.exclusions=**/*.java",
            f"-Dsonar.host.url={sonar_url}",
            f"-Dsonar.token={sonar_token}",
        ]

        print(
            f"🔍 Analyzing template repository "
            f"{owner}/{repository}"
        )

        scan_succeeded = run_cmd(
            sonar_command,
            cwd=repo_path,
        )

        if not scan_succeeded:
            print(
                f"⚠️ SonarScanner failed for "
                f"{owner}/{repository}"
            )
            return None

        print(
            f"⏳ Waiting {wait_seconds} seconds for SonarQube "
            f"to finalize the analysis"
        )

        time.sleep(wait_seconds)

        try:
            measures = get_project_metrics(
                session=session,
                project_key=project_key,
                sonar_url=sonar_url,
            )

        except requests.HTTPError as error:
            response = error.response

            if response is not None:
                print(
                    f"⚠️ Failed to retrieve metrics for "
                    f"{owner}/{repository}: "
                    f"{response.status_code} — {response.text}"
                )
            else:
                print(
                    f"⚠️ Failed to retrieve metrics for "
                    f"{owner}/{repository}: {error}"
                )

            return None

        result: dict[str, Any] = {
            "repo_url": git_url,
            "target_template_repository": f"{owner}/{repository}",
            "project_key": project_key,
            "owner": owner,
            "repo": repository,
            "template_reference_count": template_reference_count,
            **measures,
        }

        severity_counts = get_vulnerability_details(
            session=session,
            project_key=project_key,
            owner=owner,
            repository=repository,
            sonar_url=sonar_url,
            json_output_dir=json_output_dir,
        )

        for severity in SEVERITIES:
            result[f"vuln_{severity.lower()}"] = (
                severity_counts.get(severity, 0)
            )

        return result

    finally:
        shutil.rmtree(
            repo_path,
            ignore_errors=True,
        )

        print(
            f"🧹 Deleted local clone of "
            f"{owner}/{repository}"
        )


# ----------------------------------------------------------------------
# OUTPUT AND RESUME SUPPORT
# ----------------------------------------------------------------------

def load_processed_project_keys(
    output_csv: Path,
) -> set[str]:
    """Read project keys that have already been saved."""
    if not output_csv.exists():
        return set()

    try:
        existing = pd.read_csv(
            output_csv,
            on_bad_lines="skip",
        )

    except pd.errors.EmptyDataError:
        return set()

    if "project_key" in existing.columns:
        return {
            str(value).strip().lower()
            for value in existing["project_key"].dropna()
            if str(value).strip()
        }

    if "repo_url" in existing.columns:
        processed: set[str] = set()

        for value in existing["repo_url"].dropna():
            normalized_url = normalize_github_repository(value)

            if normalized_url:
                processed.add(
                    compute_project_key(normalized_url)
                )

        return processed

    return set()


def save_failure_report(
    failures: list[dict[str, Any]],
    failure_csv: Path,
) -> None:
    """Save repositories that could not be analyzed."""
    if not failures:
        if failure_csv.exists():
            failure_csv.unlink()
        return

    pd.DataFrame(failures).to_csv(
        failure_csv,
        index=False,
    )

    print(f"💾 Failure report saved to {failure_csv}")


# ----------------------------------------------------------------------
# COMMAND-LINE ARGUMENTS
# ----------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the unique GitHub template repositories referenced "
            "by the target_template_repository column of a CSV file."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV containing the target_template_repository column.",
    )

    parser.add_argument(
        "output_csv",
        type=Path,
        help="CSV in which SonarQube metrics will be saved.",
    )

    parser.add_argument(
        "--target-column",
        default=TARGET_COLUMN,
        help=(
            "Column containing template repository slugs or URLs. "
            f"Default: {TARGET_COLUMN}"
        ),
    )

    parser.add_argument(
        "--sonar-url",
        default=DEFAULT_SONAR_URL,
        help=(
            "SonarQube server URL. "
            f"Default: {DEFAULT_SONAR_URL}"
        ),
    )

    parser.add_argument(
        "--token",
        default=os.environ.get("SONAR_TOKEN"),
        help=(
            "SonarQube token. By default, SONAR_TOKEN is read "
            "from the environment."
        ),
    )

    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=REPO_DIR,
        help=f"Temporary clone directory. Default: {REPO_DIR}",
    )

    parser.add_argument(
        "--json-output-dir",
        type=Path,
        default=JSON_OUTPUT_DIR,
        help=(
            "Directory for vulnerability JSON files. "
            f"Default: {JSON_OUTPUT_DIR}"
        ),
    )

    parser.add_argument(
        "--eligible-only",
        action="store_true",
        help=(
            "Use template repositories referenced only by rows where "
            "eligible=True."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Analyze projects even when they already exist in the output CSV.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of unique template repositories to analyze.",
    )

    parser.add_argument(
        "--wait-seconds",
        type=int,
        default=10,
        help=(
            "Seconds to wait after SonarScanner finishes. "
            "Default: 10"
        ),
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    if not args.input_csv.exists():
        raise SystemExit(
            f"Input CSV not found: {args.input_csv}"
        )

    if not SONAR_SCANNER.exists():
        raise SystemExit(
            f"SonarScanner not found: {SONAR_SCANNER}"
        )

    if not args.token:
        raise SystemExit(
            "SonarQube token not provided. Define SONAR_TOKEN or use "
            "--token TOKEN."
        )

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero.")

    if args.wait_seconds < 0:
        raise SystemExit("--wait-seconds cannot be negative.")

    dataframe = pd.read_csv(
        args.input_csv,
        on_bad_lines="skip",
    )

    print(
        f"📄 Loaded {len(dataframe)} rows from "
        f"{args.input_csv}"
    )

    try:
        template_counts = collect_template_repositories(
            dataframe=dataframe,
            target_column=args.target_column,
            eligible_only=args.eligible_only,
        )

    except ValueError as error:
        raise SystemExit(str(error)) from error

    template_items = list(template_counts.items())

    if args.limit is not None:
        template_items = template_items[:args.limit]

    processed_project_keys = (
        set()
        if args.overwrite
        else load_processed_project_keys(args.output_csv)
    )

    print(
        f"📌 Previously processed templates: "
        f"{len(processed_project_keys)}"
    )

    session = requests.Session()
    session.auth = (args.token, "")
    session.headers.update(
        {
            "Accept": "application/json",
        }
    )

    successful = 0
    failed = 0
    skipped = 0

    failures: list[dict[str, Any]] = []

    try:
        for index, (
            git_url,
            template_reference_count,
        ) in enumerate(template_items, start=1):

            project_key = compute_project_key(git_url)

            print(
                f"\n{'=' * 70}\n"
                f"({index}/{len(template_items)}) {git_url}\n"
                f"Referenced by {template_reference_count} sampled "
                f"repositories\n"
                f"{'=' * 70}"
            )

            if (
                not args.overwrite
                and project_key in processed_project_keys
            ):
                print("⏩ Skipping template already processed")
                skipped += 1
                continue

            try:
                result = analyze_repo(
                    git_url=git_url,
                    template_reference_count=(
                        template_reference_count
                    ),
                    session=session,
                    sonar_url=args.sonar_url,
                    sonar_token=args.token,
                    repo_dir=args.repo_dir,
                    json_output_dir=args.json_output_dir,
                    wait_seconds=args.wait_seconds,
                )

            except requests.RequestException as error:
                print(
                    f"⚠️ SonarQube connection error for "
                    f"{git_url}: {error}"
                )
                result = None

            except Exception as error:
                print(
                    f"⚠️ Unexpected error for {git_url}: "
                    f"{type(error).__name__}: {error}"
                )
                result = None

            if result is None:
                failed += 1

                failures.append(
                    {
                        "repo_url": git_url,
                        "target_template_repository": (
                            git_url.removeprefix(
                                "https://github.com/"
                            )
                        ),
                        "project_key": project_key,
                        "template_reference_count": (
                            template_reference_count
                        ),
                        "error": "Repository analysis failed.",
                    }
                )

                continue

            args.output_csv.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            pd.DataFrame([result]).to_csv(
                args.output_csv,
                mode="a",
                header=not args.output_csv.exists(),
                index=False,
            )

            processed_project_keys.add(project_key)
            successful += 1

            print(
                f"💾 Saved analysis results for "
                f"{result['target_template_repository']}"
            )

    finally:
        session.close()

    failure_csv = args.output_csv.with_name(
        f"{args.output_csv.stem}_failures.csv"
    )

    save_failure_report(
        failures=failures,
        failure_csv=failure_csv,
    )

    print(f"\n{'=' * 70}")
    print("Analysis summary")
    print(f"{'=' * 70}")
    print(f"Unique templates selected: {len(template_items)}")
    print(f"Successfully analyzed:     {successful}")
    print(f"Previously processed:      {skipped}")
    print(f"Failed:                    {failed}")
    print(f"Output CSV:                {args.output_csv}")
    print(f"Vulnerability JSON files:  {args.json_output_dir}")


if __name__ == "__main__":
    main()