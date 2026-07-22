#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

import requests

from urllib.parse import urlparse


SONAR_URL = "http://localhost:9002"
PAGE_SIZE = 500
MAX_SEARCH_RESULTS = 10_000
COMPONENT_PAGE_SIZE = 500

def parse_boolean(value: Any) -> bool:
    """Convert common CSV boolean representations to bool."""
    if value is None:
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
        "t",
    }


def sanitize_filename(value: str) -> str:
    """Replace characters that are unsafe in filenames."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def parse_repository_reference(
    value: str,
) -> tuple[str, str] | None:
    """
    Extract owner and repository from either:

    - owner/repository
    - https://github.com/owner/repository
    - https://github.com/owner/repository.git
    """
    value = str(value or "").strip()

    if not value:
        return None

    # Handle GitHub URLs.
    if "://" in value:
        parsed_url = urlparse(value)

        if parsed_url.netloc.lower() not in {
            "github.com",
            "www.github.com",
        }:
            return None

        path = parsed_url.path.strip("/")

    else:
        # Handle owner/repository directly.
        path = value.strip("/")

    parts = [
        part.strip()
        for part in path.split("/")
        if part.strip()
    ]

    if len(parts) < 2:
        return None

    owner = parts[0]
    repository = parts[1]

    if repository.endswith(".git"):
        repository = repository[:-4]

    if not owner or not repository:
        return None

    return owner, repository

def build_project_key(row: dict[str, str]) -> str:
    """
    Determine the SonarQube project key.

    Supported CSV formats, in priority order:

    1. project_key
    2. username + project_name
    3. owner + repo
    4. repo_slug
    5. repo_url
    6. original_link
    """
    explicit_project_key = str(
        row.get("project_key", "") or ""
    ).strip()

    if explicit_project_key:
        return explicit_project_key

    # Previous CSV format.
    username = str(
        row.get("username", "") or ""
    ).strip()

    project_name = str(
        row.get("project_name", "") or ""
    ).strip()

    if username and project_name:
        return f"{username}__{project_name}"

    # New CSV format.
    owner = str(
        row.get("owner", "") or ""
    ).strip()

    repository = str(
        row.get("repo", "") or ""
    ).strip()

    if owner and repository:
        if repository.endswith(".git"):
            repository = repository[:-4]

        return f"{owner}__{repository}"

    # Slug and URL fallbacks.
    repository_fields = [
        "repo_slug",
        "repo_url",
        "original_link",
    ]

    for field_name in repository_fields:
        repository_reference = parse_repository_reference(
            row.get(field_name, "")
        )

        if repository_reference is None:
            continue

        owner, repository = repository_reference
        return f"{owner}__{repository}"

    raise ValueError(
        "Could not determine the SonarQube project key. The row "
        "must contain one of the following: 'project_key'; "
        "'username' and 'project_name'; 'owner' and 'repo'; "
        "'repo_slug'; 'repo_url'; or 'original_link'."
    )

class TooManyIssuesError(RuntimeError):
    """Raised when one SonarQube query exceeds the 10,000-result window."""

    def __init__(
        self,
        component_key: str,
        total: int,
    ) -> None:
        self.component_key = component_key
        self.total = total

        super().__init__(
            f"Component {component_key} has {total} issues, which "
            f"exceeds SonarQube's {MAX_SEARCH_RESULTS}-result limit."
        )


def fetch_issue_partition(
    session: requests.Session,
    component_key: str,
    sonar_url: str,
    *,
    on_component_only: bool,
    display_name: str,
) -> list[dict[str, Any]]:
    """
    Fetch issues for one SonarQube component.

    When on_component_only is False, SonarQube includes issues from
    descendant components.

    When on_component_only is True, only issues assigned directly to
    the specified component are returned.
    """
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

        # Detect the limitation before requesting page 21.
        if total > MAX_SEARCH_RESULTS:
            raise TooManyIssuesError(
                component_key=component_key,
                total=total,
            )

        issues.extend(current_issues)

        if total > 0:
            print(
                f"        {display_name}: page {page}, "
                f"{len(current_issues)} issues "
                f"({len(issues)}/{total})"
            )

        if len(issues) >= total:
            break

        if not current_issues:
            raise RuntimeError(
                f"SonarQube returned an empty page for "
                f"{component_key}, but only {len(issues)} of "
                f"{total} issues were retrieved."
            )

        page += 1

    return issues


def fetch_descendant_component_keys(
    session: requests.Session,
    project_key: str,
    sonar_url: str,
) -> list[str]:
    """
    Retrieve all descendant components of a SonarQube project.

    This includes files, test files, directories, and modules. Querying
    each component directly prevents the issues/search request from
    exceeding the 10,000-result limit.
    """
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
                f"Unexpected component-tree response for "
                f"{project_key}: 'components' is not a list."
            )

        for component in components:
            if not isinstance(component, dict):
                continue

            component_key = str(
                component.get("key", "")
            ).strip()

            if (
                component_key
                and component_key not in seen_component_keys
            ):
                seen_component_keys.add(component_key)
                component_keys.append(component_key)

        paging = payload.get("paging", {})

        try:
            total = int(paging.get("total", len(component_keys)))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid component total returned for "
                f"{project_key}."
            ) from error

        print(
            f"    Component page {page}: "
            f"{len(components)} components "
            f"({len(component_keys)}/{total})"
        )

        if len(component_keys) >= total:
            break

        if not components:
            raise RuntimeError(
                f"SonarQube returned an empty component page for "
                f"{project_key}, but only {len(component_keys)} of "
                f"{total} components were retrieved."
            )

        page += 1

    return component_keys


def fetch_issues_by_component(
    session: requests.Session,
    project_key: str,
    sonar_url: str,
    expected_total: int,
) -> list[dict[str, Any]]:
    """
    Retrieve issues component by component.

    The project itself is queried first to capture issues assigned
    directly to the project. Its descendants are then queried
    individually.
    """
    descendant_keys = fetch_descendant_component_keys(
        session=session,
        project_key=project_key,
        sonar_url=sonar_url,
    )

    component_keys = [project_key, *descendant_keys]

    print(
        f"    Retrieving issues directly from "
        f"{len(component_keys)} components."
    )

    issues_by_key: dict[str, dict[str, Any]] = {}

    for component_number, component_key in enumerate(
        component_keys,
        start=1,
    ):
        if (
            component_number == 1
            or component_number % 100 == 0
            or component_number == len(component_keys)
        ):
            print(
                f"    Component {component_number}/"
                f"{len(component_keys)}: {component_key}"
            )

        try:
            component_issues = fetch_issue_partition(
                session=session,
                component_key=component_key,
                sonar_url=sonar_url,
                on_component_only=True,
                display_name=component_key,
            )

        except TooManyIssuesError as error:
            raise RuntimeError(
                f"The individual component "
                f"{error.component_key} contains {error.total} "
                "issues. It cannot be exported through the normal "
                "issues/search pagination endpoint."
            ) from error

        for issue in component_issues:
            issue_key = str(issue.get("key", "")).strip()

            if not issue_key:
                raise RuntimeError(
                    f"An issue returned for {component_key} does "
                    "not contain an issue key."
                )

            issues_by_key[issue_key] = issue

    retrieved_total = len(issues_by_key)

    if retrieved_total != expected_total:
        raise RuntimeError(
            f"Issue-count validation failed for {project_key}. "
            f"The initial query reported {expected_total} issues, "
            f"but the component-based export retrieved "
            f"{retrieved_total}. The JSON file was not saved to "
            "avoid producing an incomplete dataset."
        )

    # Sort by issue key to make the output deterministic.
    return [
        issues_by_key[issue_key]
        for issue_key in sorted(issues_by_key)
    ]


def fetch_issues(
    session: requests.Session,
    project_key: str,
    sonar_url: str,
) -> list[dict[str, Any]]:
    """
    Fetch every issue associated with one SonarQube project.

    The normal project-level query is used when the project contains
    at most 10,000 issues. Larger projects are exported component by
    component.
    """
    try:
        return fetch_issue_partition(
            session=session,
            component_key=project_key,
            sonar_url=sonar_url,
            on_component_only=False,
            display_name=project_key,
        )

    except TooManyIssuesError as error:
        print(
            f"    Project contains {error.total} issues."
        )
        print(
            "    SonarQube limits a single search to the first "
            f"{MAX_SEARCH_RESULTS} results."
        )
        print(
            "    Switching to component-based issue retrieval."
        )

        return fetch_issues_by_component(
            session=session,
            project_key=project_key,
            sonar_url=sonar_url,
            expected_total=error.total,
        )

def save_issues(
    session: requests.Session,
    project_key: str,
    output_path: Path,
    sonar_url: str,
) -> int:
    """Fetch and save all issues for one project."""
    issues = fetch_issues(
        session=session,
        project_key=project_key,
        sonar_url=sonar_url,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            issues,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return len(issues)


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load project rows from the input CSV file."""
    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError("The CSV file does not contain a header.")

        rows = list(reader)

    return rows


def write_failures(
    failures: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Save information about projects that could not be exported."""
    if not failures:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = [
            "csv_row",
            "project_key",
            "repo_slug",
            "error",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(failures)


def export_projects(
    csv_path: Path,
    output_dir: Path,
    sonar_url: str,
    sonar_token: str,
    eligible_only: bool,
    overwrite: bool,
    limit: int | None,
) -> None:
    """Export SonarQube issues for all selected projects in the CSV."""
    rows = load_csv_rows(csv_path)

    print(f"Loaded {len(rows)} rows from {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {sonar_token}",
            "Accept": "application/json",
        }
    )

    print("\nLoading SonarQube project inventory...")

    sonar_project_keys = fetch_sonar_project_keys(
        session=session,
        sonar_url=sonar_url,
    )

    print(
        f"Found {len(sonar_project_keys)} projects in SonarQube."
    )

    processed = 0
    successful = 0
    skipped_ineligible = 0
    skipped_existing = 0
    skipped_duplicate = 0
    failed = 0
    total_issues = 0

    seen_project_keys: set[str] = set()
    failures: list[dict[str, str]] = []

    try:
        for csv_row_number, row in enumerate(rows, start=2):
            if eligible_only and not parse_boolean(row.get("eligible")):
                skipped_ineligible += 1
                continue

            if limit is not None and processed >= limit:
                break

            project_key = ""

            try:
                candidate_project_key = build_project_key(row)

                project_key = resolve_project_key(
                    candidate_key=candidate_project_key,
                    sonar_project_keys=sonar_project_keys,
                )

            except ValueError as error:
                failed += 1

                failures.append(
                    {
                        "csv_row": str(csv_row_number),
                        "project_key": "",
                        "repo_slug": row.get("repo_slug", ""),
                        "error": str(error),
                    }
                )

                print(
                    f"[CSV row {csv_row_number}] ERROR: {error}"
                )
                continue

            if project_key in seen_project_keys:
                skipped_duplicate += 1
                print(
                    f"[CSV row {csv_row_number}] "
                    f"Skipping duplicate project: {project_key}"
                )
                continue

            seen_project_keys.add(project_key)
            processed += 1

            safe_project_key = sanitize_filename(project_key)
            output_path = (
                output_dir / f"{safe_project_key}_issues.json"
            )

            print(
                f"\n[{processed}] Project: {project_key}\n"
                f"    Output: {output_path}"
            )

            if output_path.exists() and not overwrite:
                skipped_existing += 1
                print(
                    "    Output already exists. Skipping. "
                    "Use --overwrite to export it again."
                )
                continue

            try:
                issue_count = save_issues(
                    session=session,
                    project_key=project_key,
                    output_path=output_path,
                    sonar_url=sonar_url,
                )

                successful += 1
                total_issues += issue_count

                print(
                    f"    Saved {issue_count} issues to {output_path}"
                )

            except requests.HTTPError as error:
                failed += 1
                response = error.response

                if response is not None:
                    error_message = (
                        f"HTTP {response.status_code}: "
                        f"{response.text.strip()}"
                    )
                else:
                    error_message = str(error)

                failures.append(
                    {
                        "csv_row": str(csv_row_number),
                        "project_key": project_key,
                        "repo_slug": row.get("repo_slug", ""),
                        "error": error_message,
                    }
                )

                print(f"    ERROR: {error_message}")

            except requests.RequestException as error:
                failed += 1
                error_message = (
                    f"Could not connect to SonarQube: {error}"
                )

                failures.append(
                    {
                        "csv_row": str(csv_row_number),
                        "project_key": project_key,
                        "repo_slug": row.get("repo_slug", ""),
                        "error": error_message,
                    }
                )

                print(f"    ERROR: {error_message}")

            except (
                RuntimeError,
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ) as error:
                failed += 1

                failures.append(
                    {
                        "csv_row": str(csv_row_number),
                        "project_key": project_key,
                        "repo_slug": row.get("repo_slug", ""),
                        "error": str(error),
                    }
                )

                print(f"    ERROR: {error}")

    finally:
        session.close()

    failure_path = output_dir / "issue_export_failures.csv"
    write_failures(failures, failure_path)

    print("\n" + "=" * 60)
    print("Export summary")
    print("=" * 60)
    print(f"CSV rows:                  {len(rows)}")
    print(f"Projects considered:       {processed}")
    print(f"Successful exports:        {successful}")
    print(f"Failed exports:            {failed}")
    print(f"Existing outputs skipped:  {skipped_existing}")
    print(f"Duplicate projects skipped:{skipped_duplicate}")
    print(f"Ineligible rows skipped:   {skipped_ineligible}")
    print(f"Total issues exported:     {total_issues}")

    if failures:
        print(f"Failure report:            {failure_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read projects from a CSV file and export all SonarQube "
            "issues associated with each project."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV file containing the project information.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory in which the JSON files will be saved.",
    )

    parser.add_argument(
        "--sonar-url",
        default=SONAR_URL,
        help=f"SonarQube server URL. Default: {SONAR_URL}",
    )

    parser.add_argument(
        "--token",
        default=os.environ.get("SONAR_TOKEN"),
        help=(
            "SonarQube token. By default, the script reads the "
            "SONAR_TOKEN environment variable."
        ),
    )

    parser.add_argument(
        "--eligible-only",
        action="store_true",
        help="Process only rows where the eligible column is True.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite JSON files that already exist.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of projects to process. This is useful "
            "for testing the script."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if not args.input_csv.exists():
        raise SystemExit(
            f"Input CSV file not found: {args.input_csv}"
        )

    if not args.input_csv.is_file():
        raise SystemExit(
            f"The input path is not a file: {args.input_csv}"
        )

    if not args.token:
        raise SystemExit(
            "SonarQube token not provided. Either use --token TOKEN "
            "or define the SONAR_TOKEN environment variable."
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

    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


def fetch_sonar_project_keys(
    session: requests.Session,
    sonar_url: str,
) -> set[str]:
    """
    Retrieve all project keys currently registered in SonarQube.

    The exact capitalization stored by SonarQube is preserved.
    """
    project_keys: set[str] = set()
    page = 1

    while True:
        response = session.get(
            f"{sonar_url.rstrip('/')}/api/projects/search",
            params={
                "p": page,
                "ps": 500,
            },
            timeout=60,
        )

        response.raise_for_status()
        payload = response.json()

        components = payload.get("components", [])

        if not isinstance(components, list):
            raise RuntimeError(
                "Unexpected SonarQube response: "
                "'components' is not a list."
            )

        for component in components:
            if not isinstance(component, dict):
                continue

            project_key = str(
                component.get("key", "")
            ).strip()

            if project_key:
                project_keys.add(project_key)

        paging = payload.get("paging", {})

        try:
            total = int(
                paging.get("total", len(project_keys))
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "SonarQube returned an invalid project count."
            ) from error

        print(
            f"    Project inventory page {page}: "
            f"{len(components)} projects "
            f"({len(project_keys)}/{total})"
        )

        if len(project_keys) >= total:
            break

        if not components:
            raise RuntimeError(
                "SonarQube returned an empty project inventory "
                f"page, but only {len(project_keys)} of {total} "
                "projects were retrieved."
            )

        page += 1

    return project_keys

def split_project_key(
    project_key: str,
) -> tuple[str, str] | None:
    """
    Split a project key into owner and repository.

    Supported formats:
    - owner__repository
    - owner/repository
    """
    project_key = str(project_key or "").strip()

    if "__" in project_key:
        owner, repository = project_key.split("__", maxsplit=1)
    elif "/" in project_key:
        owner, repository = project_key.split("/", maxsplit=1)
    else:
        return None

    owner = owner.strip()
    repository = repository.strip()

    if not owner or not repository:
        return None

    return owner, repository


def normalize_project_identity(
    project_key: str,
    *,
    legacy_git_removal: bool = False,
) -> str:
    """
    Normalize a SonarQube project identity for matching.

    Normalization includes:
    - case-insensitive owner and repository comparison
    - support for owner__repository and owner/repository
    - removal of a normal trailing .git suffix
    - optional support for the historical bug that removed '.git'
      from anywhere in a repository name

    Example of the historical transformation:
        kinsaeng.github.io -> kinsaenghub.io
    """
    parsed_key = split_project_key(project_key)

    if parsed_key is None:
        return project_key.strip().casefold()

    owner, repository = parsed_key

    owner = owner.casefold()
    repository = repository.casefold()

    if legacy_git_removal:
        # Reproduce the previous project-key generation bug.
        repository = repository.replace(".git", "")
    elif repository.endswith(".git"):
        # Correct behavior: remove only the clone suffix.
        repository = repository[:-4]

    return f"{owner}/{repository}"

def resolve_project_key(
    candidate_key: str,
    sonar_project_keys: set[str],
) -> str:
    """
    Resolve a CSV-generated project key against SonarQube.

    Resolution order:
    1. Exact key
    2. Case-insensitive key
    3. Normalized owner/repository identity
    4. Historical '.git' removal transformation

    A match is accepted only when it uniquely identifies one
    SonarQube project.
    """
    # ---------------------------------------------------------
    # 1. Exact match
    # ---------------------------------------------------------
    if candidate_key in sonar_project_keys:
        return candidate_key

    # ---------------------------------------------------------
    # 2. Case-insensitive exact match
    # ---------------------------------------------------------
    candidate_casefold = candidate_key.casefold()

    case_insensitive_matches = [
        sonar_key
        for sonar_key in sonar_project_keys
        if sonar_key.casefold() == candidate_casefold
    ]

    if len(case_insensitive_matches) == 1:
        resolved_key = case_insensitive_matches[0]

        print(
            "    Resolved project-key capitalization:\n"
            f"        CSV key:       {candidate_key}\n"
            f"        SonarQube key: {resolved_key}"
        )

        return resolved_key

    if len(case_insensitive_matches) > 1:
        matches = ", ".join(sorted(case_insensitive_matches))

        raise ValueError(
            f"Ambiguous case-insensitive SonarQube project key for "
            f"{candidate_key}. Matching keys: {matches}"
        )

    # ---------------------------------------------------------
    # 3. Normal owner/repository normalization
    # ---------------------------------------------------------
    normalized_candidate = normalize_project_identity(
        candidate_key,
        legacy_git_removal=False,
    )

    normalized_matches = [
        sonar_key
        for sonar_key in sonar_project_keys
        if normalize_project_identity(
            sonar_key,
            legacy_git_removal=False,
        ) == normalized_candidate
    ]

    if len(normalized_matches) == 1:
        resolved_key = normalized_matches[0]

        print(
            "    Resolved normalized project key:\n"
            f"        CSV key:       {candidate_key}\n"
            f"        SonarQube key: {resolved_key}"
        )

        return resolved_key

    if len(normalized_matches) > 1:
        matches = ", ".join(sorted(normalized_matches))

        raise ValueError(
            f"Ambiguous normalized SonarQube project key for "
            f"{candidate_key}. Matching keys: {matches}"
        )

    # ---------------------------------------------------------
    # 4. Historical bug: remove '.git' anywhere in repo name
    # ---------------------------------------------------------
    legacy_candidate = normalize_project_identity(
        candidate_key,
        legacy_git_removal=True,
    )

    legacy_matches = [
        sonar_key
        for sonar_key in sonar_project_keys
        if normalize_project_identity(
            sonar_key,
            legacy_git_removal=True,
        ) == legacy_candidate
    ]

    if len(legacy_matches) == 1:
        resolved_key = legacy_matches[0]

        print(
            "    Resolved project using legacy '.git' removal:\n"
            f"        CSV key:       {candidate_key}\n"
            f"        Legacy form:   {legacy_candidate}\n"
            f"        SonarQube key: {resolved_key}"
        )

        return resolved_key

    if len(legacy_matches) > 1:
        matches = ", ".join(sorted(legacy_matches))

        raise ValueError(
            f"Ambiguous legacy SonarQube project key for "
            f"{candidate_key}. Matching keys: {matches}"
        )

    raise ValueError(
        f"Project not found in SonarQube: {candidate_key}. "
        f"Normalized identity: {normalized_candidate}. "
        f"Legacy identity: {legacy_candidate}."
    )

if __name__ == "__main__":
    main()