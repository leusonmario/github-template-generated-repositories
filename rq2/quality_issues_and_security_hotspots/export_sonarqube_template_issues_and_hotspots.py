#!/usr/bin/env python3
"""
Export SonarQube standard issues and Security Hotspots for projects listed in a CSV.

For each project this script writes:
    <project_key>_issues.json
    <project_key>_hotspots.json

Existing issue JSON files are preserved. Therefore, if you already exported the
issues, rerunning this script will only collect the missing hotspot files unless
--overwrite is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


SONAR_URL = "http://localhost:9002"
PAGE_SIZE = 500
HOTSPOT_PAGE_SIZE = 500
MAX_SEARCH_RESULTS = 10_000
COMPONENT_PAGE_SIZE = 500


def parse_boolean(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def parse_repository_reference(value: str) -> tuple[str, str] | None:
    value = str(value or "").strip()
    if not value:
        return None

    if "://" in value:
        parsed_url = urlparse(value)
        if parsed_url.netloc.lower() not in {"github.com", "www.github.com"}:
            return None
        path = parsed_url.path.strip("/")
    else:
        path = value.strip("/")

    parts = [part.strip() for part in path.split("/") if part.strip()]
    if len(parts) < 2:
        return None

    owner, repository = parts[0], parts[1]
    if repository.endswith(".git"):
        repository = repository[:-4]

    if not owner or not repository:
        return None

    return owner, repository


def build_template_project_key(row: dict[str, str]) -> str:
    """
    Determine the SonarQube project key for the SOURCE TEMPLATE referenced by
    one row of the sampling CSV.

    Supported template columns, in priority order:

    1. template_project_key
    2. template_slug
    3. target_template_repository
    4. template_repository

    The repository-reference columns may contain either:
        owner/repository
        https://github.com/owner/repository
        https://github.com/owner/repository.git
    """
    explicit_project_key = str(
        row.get("template_project_key", "") or ""
    ).strip()

    if explicit_project_key:
        return explicit_project_key

    for field_name in (
        "template_slug",
        "target_template_repository",
        "template_repository",
    ):
        ref = parse_repository_reference(
            row.get(field_name, "")
        )

        if ref is not None:
            owner, repository = ref
            return f"{owner}__{repository}"

    raise ValueError(
        "Could not determine the source-template SonarQube project key. "
        "The row must contain one of: template_project_key, template_slug, "
        "target_template_repository, or template_repository."
    )


def template_reference_from_row(row: dict[str, str]) -> str:
    """Return the best human-readable source-template reference in the row."""
    for field_name in (
        "template_slug",
        "target_template_repository",
        "template_repository",
        "template_project_key",
    ):
        value = str(row.get(field_name, "") or "").strip()
        if value:
            return value
    return ""


class TooManyIssuesError(RuntimeError):
    def __init__(self, component_key: str, total: int) -> None:
        self.component_key = component_key
        self.total = total
        super().__init__(
            f"Component {component_key} has {total} issues, which exceeds "
            f"SonarQube's {MAX_SEARCH_RESULTS}-result limit."
        )


# ---------------------------------------------------------------------------
# Standard issues
# ---------------------------------------------------------------------------

def fetch_issue_partition(
    session: requests.Session,
    component_key: str,
    sonar_url: str,
    *,
    on_component_only: bool,
    display_name: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    page = 1

    while True:
        params: dict[str, str | int] = {
            "componentKeys": component_key,
            "p": page,
            "ps": PAGE_SIZE,
        }
        if on_component_only:
            params["onComponentOnly"] = "true"

        response = session.get(
            f"{sonar_url.rstrip('/')}/api/issues/search",
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        current_issues = payload.get("issues", [])
        if not isinstance(current_issues, list):
            raise RuntimeError(
                f"Unexpected SonarQube response for {component_key}: "
                "'issues' is not a list."
            )

        paging = payload.get("paging", {})
        try:
            total = int(paging.get("total", len(current_issues)))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid issue total returned for {component_key}."
            ) from error

        if total > MAX_SEARCH_RESULTS:
            raise TooManyIssuesError(component_key, total)

        issues.extend(current_issues)
        if total > 0:
            print(
                f"        {display_name}: page {page}, "
                f"{len(current_issues)} issues ({len(issues)}/{total})"
            )

        if len(issues) >= total:
            break
        if not current_issues:
            raise RuntimeError(
                f"SonarQube returned an empty issue page for {component_key}, "
                f"but only {len(issues)} of {total} issues were retrieved."
            )
        page += 1

    return issues


def fetch_descendant_component_keys(
    session: requests.Session,
    project_key: str,
    sonar_url: str,
) -> list[str]:
    component_keys: list[str] = []
    seen_component_keys: set[str] = set()
    page = 1

    while True:
        response = session.get(
            f"{sonar_url.rstrip('/')}/api/components/tree",
            params={
                "component": project_key,
                "strategy": "all",
                "p": page,
                "ps": COMPONENT_PAGE_SIZE,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        components = payload.get("components", [])
        if not isinstance(components, list):
            raise RuntimeError(
                f"Unexpected component-tree response for {project_key}: "
                "'components' is not a list."
            )

        for component in components:
            if not isinstance(component, dict):
                continue
            key = str(component.get("key", "")).strip()
            if key and key not in seen_component_keys:
                seen_component_keys.add(key)
                component_keys.append(key)

        paging = payload.get("paging", {})
        try:
            total = int(paging.get("total", len(component_keys)))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid component total returned for {project_key}."
            ) from error

        print(
            f"    Component page {page}: {len(components)} components "
            f"({len(component_keys)}/{total})"
        )

        if len(component_keys) >= total:
            break
        if not components:
            raise RuntimeError(
                f"SonarQube returned an empty component page for {project_key}, "
                f"but only {len(component_keys)} of {total} components were retrieved."
            )
        page += 1

    return component_keys


def fetch_issues_by_component(
    session: requests.Session,
    project_key: str,
    sonar_url: str,
    expected_total: int,
) -> list[dict[str, Any]]:
    descendant_keys = fetch_descendant_component_keys(session, project_key, sonar_url)
    component_keys = [project_key, *descendant_keys]

    print(f"    Retrieving issues directly from {len(component_keys)} components.")
    issues_by_key: dict[str, dict[str, Any]] = {}

    for component_number, component_key in enumerate(component_keys, start=1):
        if (
            component_number == 1
            or component_number % 100 == 0
            or component_number == len(component_keys)
        ):
            print(
                f"    Component {component_number}/{len(component_keys)}: "
                f"{component_key}"
            )

        try:
            component_issues = fetch_issue_partition(
                session,
                component_key,
                sonar_url,
                on_component_only=True,
                display_name=component_key,
            )
        except TooManyIssuesError as error:
            raise RuntimeError(
                f"The individual component {error.component_key} contains "
                f"{error.total} issues and cannot be exported through the "
                "normal issues/search pagination endpoint."
            ) from error

        for issue in component_issues:
            issue_key = str(issue.get("key", "")).strip()
            if not issue_key:
                raise RuntimeError(
                    f"An issue returned for {component_key} does not contain an issue key."
                )
            issues_by_key[issue_key] = issue

    retrieved_total = len(issues_by_key)
    if retrieved_total != expected_total:
        raise RuntimeError(
            f"Issue-count validation failed for {project_key}. The initial query "
            f"reported {expected_total} issues, but component-based export "
            f"retrieved {retrieved_total}."
        )

    return [issues_by_key[key] for key in sorted(issues_by_key)]


def fetch_issues(
    session: requests.Session,
    project_key: str,
    sonar_url: str,
) -> list[dict[str, Any]]:
    try:
        return fetch_issue_partition(
            session,
            project_key,
            sonar_url,
            on_component_only=False,
            display_name=project_key,
        )
    except TooManyIssuesError as error:
        print(f"    Project contains {error.total} issues.")
        print("    Switching to component-based issue retrieval.")
        return fetch_issues_by_component(
            session, project_key, sonar_url, expected_total=error.total
        )


def save_issues(
    session: requests.Session,
    project_key: str,
    output_path: Path,
    sonar_url: str,
) -> int:
    issues = fetch_issues(session, project_key, sonar_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(issues, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(issues)


# ---------------------------------------------------------------------------
# Security Hotspots
# ---------------------------------------------------------------------------

def fetch_security_hotspots(
    session: requests.Session,
    project_key: str,
    sonar_url: str,
) -> list[dict[str, Any]]:
    """
    Fetch all Security Hotspots for one project through /api/hotspots/search.

    Hotspots are stored separately from standard issues by SonarQube, so they
    must be exported through this endpoint rather than /api/issues/search.
    """
    hotspots: list[dict[str, Any]] = []
    page = 1

    while True:
        response = session.get(
            f"{sonar_url.rstrip('/')}/api/hotspots/search",
            params={
                "projectKey": project_key,
                "p": page,
                "ps": HOTSPOT_PAGE_SIZE,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        current_hotspots = payload.get("hotspots", [])
        if not isinstance(current_hotspots, list):
            raise RuntimeError(
                f"Unexpected SonarQube hotspot response for {project_key}: "
                "'hotspots' is not a list."
            )

        paging = payload.get("paging", {})
        try:
            total = int(paging.get("total", len(current_hotspots)))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid hotspot total returned for {project_key}."
            ) from error

        hotspots.extend(current_hotspots)

        if total > 0:
            print(
                f"        Hotspots: page {page}, {len(current_hotspots)} hotspots "
                f"({len(hotspots)}/{total})"
            )

        if len(hotspots) >= total:
            break
        if not current_hotspots:
            raise RuntimeError(
                f"SonarQube returned an empty hotspot page for {project_key}, "
                f"but only {len(hotspots)} of {total} hotspots were retrieved."
            )
        page += 1

    # Basic duplicate-key validation when keys are present.
    keys = [
        str(h.get("key", "")).strip()
        for h in hotspots
        if str(h.get("key", "")).strip()
    ]
    if keys and len(keys) != len(set(keys)):
        raise RuntimeError(
            f"Duplicate Security Hotspot keys were returned for {project_key}."
        )

    return hotspots


def save_security_hotspots(
    session: requests.Session,
    project_key: str,
    output_path: Path,
    sonar_url: str,
) -> int:
    hotspots = fetch_security_hotspots(session, project_key, sonar_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(hotspots, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return len(hotspots)


# ---------------------------------------------------------------------------
# CSV and project-key helpers
# ---------------------------------------------------------------------------

def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("The CSV file does not contain a header.")
        return list(reader)


def write_failures(failures: list[dict[str, str]], output_path: Path) -> None:
    if not failures:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["csv_row", "project_key", "repo_slug", "stage", "error"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def fetch_sonar_project_keys(
    session: requests.Session,
    sonar_url: str,
) -> set[str]:
    project_keys: set[str] = set()
    page = 1

    while True:
        response = session.get(
            f"{sonar_url.rstrip('/')}/api/projects/search",
            params={"p": page, "ps": 500},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()

        components = payload.get("components", [])
        if not isinstance(components, list):
            raise RuntimeError(
                "Unexpected SonarQube response: 'components' is not a list."
            )

        for component in components:
            if not isinstance(component, dict):
                continue
            project_key = str(component.get("key", "")).strip()
            if project_key:
                project_keys.add(project_key)

        paging = payload.get("paging", {})
        try:
            total = int(paging.get("total", len(project_keys)))
        except (TypeError, ValueError) as error:
            raise RuntimeError("SonarQube returned an invalid project count.") from error

        print(
            f"    Project inventory page {page}: {len(components)} projects "
            f"({len(project_keys)}/{total})"
        )

        if len(project_keys) >= total:
            break
        if not components:
            raise RuntimeError(
                "SonarQube returned an empty project inventory page, but only "
                f"{len(project_keys)} of {total} projects were retrieved."
            )
        page += 1

    return project_keys


def split_project_key(project_key: str) -> tuple[str, str] | None:
    project_key = str(project_key or "").strip()
    if "__" in project_key:
        owner, repository = project_key.split("__", maxsplit=1)
    elif "/" in project_key:
        owner, repository = project_key.split("/", maxsplit=1)
    else:
        return None

    owner, repository = owner.strip(), repository.strip()
    if not owner or not repository:
        return None
    return owner, repository


def normalize_project_identity(
    project_key: str,
    *,
    legacy_git_removal: bool = False,
) -> str:
    parsed_key = split_project_key(project_key)
    if parsed_key is None:
        return project_key.strip().casefold()

    owner, repository = parsed_key
    owner = owner.casefold()
    repository = repository.casefold()

    if legacy_git_removal:
        repository = repository.replace(".git", "")
    elif repository.endswith(".git"):
        repository = repository[:-4]

    return f"{owner}/{repository}"


def resolve_project_key(candidate_key: str, sonar_project_keys: set[str]) -> str:
    if candidate_key in sonar_project_keys:
        return candidate_key

    candidate_casefold = candidate_key.casefold()
    case_insensitive_matches = [
        key for key in sonar_project_keys if key.casefold() == candidate_casefold
    ]

    if len(case_insensitive_matches) == 1:
        resolved = case_insensitive_matches[0]
        print(
            "    Resolved project-key capitalization:\n"
            f"        CSV key:       {candidate_key}\n"
            f"        SonarQube key: {resolved}"
        )
        return resolved
    if len(case_insensitive_matches) > 1:
        raise ValueError(
            f"Ambiguous case-insensitive project key for {candidate_key}: "
            + ", ".join(sorted(case_insensitive_matches))
        )

    normalized_candidate = normalize_project_identity(candidate_key)
    normalized_matches = [
        key
        for key in sonar_project_keys
        if normalize_project_identity(key) == normalized_candidate
    ]

    if len(normalized_matches) == 1:
        resolved = normalized_matches[0]
        print(
            "    Resolved normalized project key:\n"
            f"        CSV key:       {candidate_key}\n"
            f"        SonarQube key: {resolved}"
        )
        return resolved
    if len(normalized_matches) > 1:
        raise ValueError(
            f"Ambiguous normalized project key for {candidate_key}: "
            + ", ".join(sorted(normalized_matches))
        )

    legacy_candidate = normalize_project_identity(
        candidate_key, legacy_git_removal=True
    )
    legacy_matches = [
        key
        for key in sonar_project_keys
        if normalize_project_identity(key, legacy_git_removal=True)
        == legacy_candidate
    ]

    if len(legacy_matches) == 1:
        resolved = legacy_matches[0]
        print(
            "    Resolved project using legacy '.git' removal:\n"
            f"        CSV key:       {candidate_key}\n"
            f"        Legacy form:   {legacy_candidate}\n"
            f"        SonarQube key: {resolved}"
        )
        return resolved
    if len(legacy_matches) > 1:
        raise ValueError(
            f"Ambiguous legacy project key for {candidate_key}: "
            + ", ".join(sorted(legacy_matches))
        )

    raise ValueError(
        f"Project not found in SonarQube: {candidate_key}. "
        f"Normalized identity: {normalized_candidate}. "
        f"Legacy identity: {legacy_candidate}."
    )


# ---------------------------------------------------------------------------
# Export workflow
# ---------------------------------------------------------------------------

def export_projects(
    csv_path: Path,
    output_dir: Path,
    sonar_url: str,
    sonar_token: str,
    eligible_only: bool,
    overwrite: bool,
    limit: int | None,
) -> None:
    rows = load_csv_rows(csv_path)
    print(f"Loaded {len(rows)} sampling rows from {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {sonar_token}",
            "Accept": "application/json",
        }
    )

    print("\nLoading SonarQube project inventory...")
    sonar_project_keys = fetch_sonar_project_keys(session, sonar_url)
    print(f"Found {len(sonar_project_keys)} projects in SonarQube.")

    processed = 0
    successful = 0
    failed = 0
    skipped_ineligible = 0
    skipped_existing = 0
    skipped_duplicate = 0
    total_issues_exported = 0
    total_hotspots_exported = 0

    seen_project_keys: set[str] = set()
    failures: list[dict[str, str]] = []

    try:
        for csv_row_number, row in enumerate(rows, start=2):
            if eligible_only and not parse_boolean(row.get("eligible")):
                skipped_ineligible += 1
                continue
            if limit is not None and processed >= limit:
                break

            try:
                candidate_key = build_template_project_key(row)
                project_key = resolve_project_key(candidate_key, sonar_project_keys)
            except ValueError as error:
                failed += 1
                failures.append(
                    {
                        "csv_row": str(csv_row_number),
                        "project_key": "",
                        "repo_slug": template_reference_from_row(row),
                        "stage": "project_key_resolution",
                        "error": str(error),
                    }
                )
                print(f"[CSV row {csv_row_number}] ERROR: {error}")
                continue

            if project_key in seen_project_keys:
                skipped_duplicate += 1
                print(
                    f"[CSV row {csv_row_number}] Skipping duplicate template: "
                    f"{project_key}"
                )
                continue

            seen_project_keys.add(project_key)
            processed += 1

            safe_project_key = sanitize_filename(project_key)
            issues_output_path = output_dir / f"{safe_project_key}_issues.json"
            hotspots_output_path = output_dir / f"{safe_project_key}_hotspots.json"

            print(
                f"\n[{processed}] Source template: {project_key}\n"
                f"    Issues:   {issues_output_path}\n"
                f"    Hotspots: {hotspots_output_path}"
            )

            issues_exist = issues_output_path.exists()
            hotspots_exist = hotspots_output_path.exists()

            # Important for your current dataset: if the issue JSON already
            # exists but the hotspot JSON does not, the project is NOT skipped.
            if issues_exist and hotspots_exist and not overwrite:
                skipped_existing += 1
                print("    Both outputs already exist. Skipping project.")
                continue

            project_failed = False

            if overwrite or not issues_exist:
                try:
                    issue_count = save_issues(
                        session, project_key, issues_output_path, sonar_url
                    )
                    total_issues_exported += issue_count
                    print(f"    Saved {issue_count} issues.")
                except (
                    requests.RequestException,
                    RuntimeError,
                    ValueError,
                    TypeError,
                    json.JSONDecodeError,
                ) as error:
                    project_failed = True
                    failures.append(
                        {
                            "csv_row": str(csv_row_number),
                            "project_key": project_key,
                            "repo_slug": template_reference_from_row(row),
                            "stage": "issues",
                            "error": str(error),
                        }
                    )
                    print(f"    ISSUE EXPORT ERROR: {error}")
            else:
                print("    Issue JSON already exists. Keeping it.")

            if overwrite or not hotspots_exist:
                try:
                    hotspot_count = save_security_hotspots(
                        session, project_key, hotspots_output_path, sonar_url
                    )
                    total_hotspots_exported += hotspot_count
                    print(f"    Saved {hotspot_count} Security Hotspots.")
                except (
                    requests.RequestException,
                    RuntimeError,
                    ValueError,
                    TypeError,
                    json.JSONDecodeError,
                ) as error:
                    project_failed = True
                    failures.append(
                        {
                            "csv_row": str(csv_row_number),
                            "project_key": project_key,
                            "repo_slug": template_reference_from_row(row),
                            "stage": "security_hotspots",
                            "error": str(error),
                        }
                    )
                    print(f"    SECURITY HOTSPOT EXPORT ERROR: {error}")
            else:
                print("    Hotspot JSON already exists. Keeping it.")

            if project_failed:
                failed += 1
            else:
                successful += 1

    finally:
        session.close()

    failure_path = output_dir / "finding_export_failures.csv"
    write_failures(failures, failure_path)

    print("\n" + "=" * 60)
    print("Export summary")
    print("=" * 60)
    print(f"CSV rows:                    {len(rows)}")
    print(f"Unique templates considered: {processed}")
    print(f"Successful templates:        {successful}")
    print(f"Templates with failures:     {failed}")
    print(f"Both outputs already exist:  {skipped_existing}")
    print(f"Duplicate template rows:      {skipped_duplicate}")
    print(f"Ineligible rows skipped:     {skipped_ineligible}")
    print(f"Issues exported this run:    {total_issues_exported}")
    print(f"Hotspots exported this run:  {total_hotspots_exported}")

    if failures:
        print(f"Failure report:              {failure_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the source templates referenced by a sampling CSV and export "
            "their SonarQube standard issues and Security Hotspots."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "Sampling CSV containing source-template references, e.g. "
            "target_template_repository."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Directory in which template _issues.json and _hotspots.json "
            "files will be saved."
        ),
    )
    parser.add_argument("--sonar-url", default=SONAR_URL)
    parser.add_argument("--token", default=os.environ.get("SONAR_TOKEN"))
    parser.add_argument("--eligible-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of UNIQUE source templates to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.input_csv.exists():
        raise SystemExit(f"Input CSV file not found: {args.input_csv}")
    if not args.input_csv.is_file():
        raise SystemExit(f"The input path is not a file: {args.input_csv}")
    if not args.token:
        raise SystemExit(
            "SonarQube token not provided. Use --token TOKEN or define "
            "SONAR_TOKEN."
        )
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero.")

    try:
        export_projects(
            csv_path=args.input_csv,
            output_dir=args.output_dir,
            sonar_url=args.sonar_url,
            sonar_token=args.token,
            eligible_only=args.eligible_only,
            overwrite=args.overwrite,
            limit=args.limit,
        )
    except (OSError, ValueError, requests.RequestException) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
