#!/usr/bin/env python3
"""
compare_sonar_issue_pairs_from_sample.py

Build template/generated-repository pairs directly from the sampling CSV and
compare their exported SonarQube issue JSON files.

Expected issue filenames:
    issues_templates/<template_owner>__<template_repo>_issues.json
    issues_projects/<project_owner>__<project_repo>_issues.json

Example:
    target_template_repository = 9d8dev/pay
    repo_slug                 = brijr/rileigh

    issues_templates/9d8dev__pay_issues.json
    issues_projects/brijr__rileigh_issues.json

Usage:
    python compare_sonar_issue_pairs_from_sample.py \
        sample/persistence_sample_balanced/TypeScript/selected_sample.csv \
        issues_templates \
        issues_projects \
        sonar_issue_comparisons \
        --eligible-only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict, deque
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Callable, Iterable
from urllib.parse import urlparse
import unicodedata


Issue = dict[str, Any]
PairResult = dict[str, Any]

SEVERITIES = ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]
SCRIPT_VERSION = "2.2-legacy-dot-git-filename-resolution"


def parse_boolean(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def parse_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def normalize_github_slug(value: Any) -> str | None:
    """Convert a GitHub slug or URL to owner/repository."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    if text.startswith("git@github.com:"):
        text = text.removeprefix("git@github.com:")
    elif text.startswith("github.com/"):
        text = text.removeprefix("github.com/")
    elif text.startswith(("http://", "https://")):
        parsed = urlparse(text)
        if (parsed.hostname or "").lower() not in {"github.com", "www.github.com"}:
            return None
        text = parsed.path

    text = text.split("?", 1)[0].split("#", 1)[0].strip("/")
    parts = [part for part in text.split("/") if part]

    if len(parts) < 2:
        return None

    owner = parts[0].strip()
    repository = parts[1].strip()
    if repository.endswith(".git"):
        repository = repository[:-4]

    if not owner or not repository:
        return None

    return f"{owner}/{repository}"


def project_slug_from_row(row: dict[str, str]) -> str | None:
    """Resolve the generated repository slug from the CSV row."""
    for column in ("repo_slug", "original_link"):
        slug = normalize_github_slug(row.get(column))
        if slug:
            return slug

    username = str(row.get("username") or "").strip()
    project_name = str(row.get("project_name") or "").strip()
    if username and project_name:
        return f"{username}/{project_name}"

    return None


def template_slug_from_row(row: dict[str, str]) -> str | None:
    """Resolve the template repository slug from the CSV row."""
    for column in ("template_slug", "target_template_repository"):
        slug = normalize_github_slug(row.get(column))
        if slug:
            return slug
    return None


def sonar_project_key(slug: str) -> str:
    owner, repository = slug.split("/", 1)
    return f"{owner}__{repository}"


def safe_identifier(value: str) -> str:
    """Create a directory-safe identifier for output artifacts."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def issue_filename_identifier(value: str) -> str:
    """
    Create an issue-filename identifier without removing meaningful hyphens.

    A repository may legitimately end in a hyphen, for example
    ``spookyynate/Music-Bot-``.  The previous implementation called
    ``safe_identifier()``, which stripped that final hyphen and produced the
    wrong expected filename.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def canonical_slug_key(value: str) -> str:
    """Return a Unicode-normalized, case-insensitive repository identity."""
    normalized = normalize_github_slug(value)
    if normalized is None:
        raise ValueError(f"Invalid GitHub repository identifier: {value}")
    return unicodedata.normalize("NFKC", normalized).casefold()

def legacy_git_removed_slug_key(value: str) -> str:
    """
    Reproduce the historical SonarQube filename transformation that
    removed '.git' from anywhere in the repository name.

    This is different from correctly removing only a trailing '.git'
    clone suffix.

    Examples:
        kodiar/kodiar.github.io
            -> kodiar/kodiarhub.io

        kinsaeng/kinsaeng.github.io
            -> kinsaeng/kinsaenghub.io
    """
    normalized = normalize_github_slug(value)

    if normalized is None:
        raise ValueError(
            f"Invalid GitHub repository identifier: {value}"
        )

    owner, repository = normalized.split("/", 1)

    owner = unicodedata.normalize(
        "NFKC",
        owner,
    ).casefold()

    repository = unicodedata.normalize(
        "NFKC",
        repository,
    ).casefold()

    # Reproduce the previous erroneous transformation.
    repository = repository.replace(".git", "")

    return f"{owner}/{repository}"

def expected_issue_filename(slug: str) -> str:
    return (
        f"{issue_filename_identifier(sonar_project_key(slug))}"
        "_issues.json"
    )


class IssueFileIndex:
    """
    Case-insensitive index of issue JSON files.

    Supported layouts include:

        issues_projects/owner__repository_issues.json
        issues_projects/owner/repository_issues.json
        issues_projects/owner/repository/issues.json

    Repository identity comparisons use ``casefold()``, so a CSV value such
    as ``spookyynate/music-bot-`` matches a file representing
    ``spookyynate/Music-Bot-``.
    """

    ISSUE_SUFFIXES = (
        "_issues",
        "_sonar_issues",
        "_vulnerabilities",
    )

    GENERIC_ISSUE_STEMS = {
        "issues",
        "sonar_issues",
        "vulnerabilities",
    }

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        if not self.directory.exists():
            raise ValueError(f"Issue directory not found: {self.directory}")
        if not self.directory.is_dir():
            raise ValueError(f"Issue path is not a directory: {self.directory}")

        self.by_name: dict[str, list[Path]] = defaultdict(list)
        self.by_slug: dict[str, list[Path]] = defaultdict(list)
        self.all_files: list[Path] = []

        for path in sorted(self.directory.rglob("*.json")):
            if not path.is_file():
                continue

            resolved = path.resolve()
            self.all_files.append(resolved)
            self.by_name[path.name.casefold()].append(resolved)

            for slug_key in self._infer_slug_keys(path):
                if resolved not in self.by_slug[slug_key]:
                    self.by_slug[slug_key].append(resolved)

    @staticmethod
    def _strip_issue_suffix(stem: str) -> str:
        """Remove a known issue-export suffix from a filename stem."""
        folded = stem.casefold()

        for suffix in IssueFileIndex.ISSUE_SUFFIXES:
            if folded.endswith(suffix):
                return stem[: -len(suffix)]

        return stem

    @staticmethod
    def _slug_key(owner: str, repository: str) -> str | None:
        owner = unicodedata.normalize(
            "NFKC", owner.strip().strip("/\\")
        )
        repository = unicodedata.normalize(
            "NFKC", repository.strip().strip("/\\")
        )

        if repository.casefold().endswith(".git"):
            repository = repository[:-4]

        if not owner or not repository:
            return None

        return f"{owner}/{repository}".casefold()

    def _infer_slug_keys(self, path: Path) -> set[str]:
        """Infer possible owner/repository identities from a JSON path."""
        keys: set[str] = set()
        relative = path.relative_to(self.directory)
        parts = relative.parts
        stripped_stem = self._strip_issue_suffix(path.stem)

        # Flat naming convention: owner__repository_issues.json
        if "__" in stripped_stem:
            owner, repository = stripped_stem.split("__", 1)
            key = self._slug_key(owner, repository)
            if key:
                keys.add(key)

        # Nested convention: owner/repository_issues.json
        if len(parts) >= 2 and stripped_stem.casefold() not in self.GENERIC_ISSUE_STEMS:
            owner = parts[-2]
            repository = stripped_stem
            key = self._slug_key(owner, repository)
            if key:
                keys.add(key)

        # More deeply nested convention: owner/repository/issues.json
        if len(parts) >= 3 and path.stem.casefold() in self.GENERIC_ISSUE_STEMS:
            owner = parts[-3]
            repository = parts[-2]
            key = self._slug_key(owner, repository)
            if key:
                keys.add(key)

        return keys

    @staticmethod
    def _single_match(
        matches: list[Path],
        description: str,
    ) -> Path | None:
        unique_matches = list(dict.fromkeys(matches))

        if len(unique_matches) == 1:
            return unique_matches[0]

        if len(unique_matches) > 1:
            raise ValueError(
                f"Ambiguous issue files for {description}: "
                + ", ".join(str(path) for path in unique_matches)
            )

        return None

    def resolve(self, slug: str) -> Path:
        normalized_slug = normalize_github_slug(slug)
        if normalized_slug is None:
            raise ValueError(f"Invalid GitHub repository identifier: {slug}")

        # First try the standard flat filename, case-insensitively.
        expected = expected_issue_filename(normalized_slug)
        match = self._single_match(
            self.by_name.get(expected.casefold(), []),
            f"{normalized_slug} using filename {expected}",
        )
        if match is not None:
            return match

        # Then use the canonical owner/repository identity inferred from paths.
        slug_key = canonical_slug_key(normalized_slug)
        match = self._single_match(
            self.by_slug.get(slug_key, []),
            normalized_slug,
        )
        if match is not None:
            return match

        # Final canonical scan. This deliberately recomputes identities from
        # every path so unusual capitalization and nested layouts cannot be
        # missed because of a filename-indexing assumption.
        canonical_matches: list[Path] = []
        for candidate in self.all_files:
            candidate_keys = self._infer_slug_keys(candidate)
            if slug_key in candidate_keys:
                canonical_matches.append(candidate)

        match = self._single_match(
            canonical_matches,
            f"canonical case-insensitive identity {normalized_slug}",
        )
        if match is not None:
            return match

        # -------------------------------------------------------------
        # Historical compatibility:
        #
        # Some SonarQube project keys were generated by removing ".git"
        # from anywhere in the repository name instead of removing only
        # a trailing clone suffix.
        #
        # For example:
        #     kodiar/kodiar.github.io
        # became:
        #     Kodiar__kodiarhub.io_issues.json
        #
        # This fallback affects only file resolution. The original GitHub
        # slug and pair ID remain unchanged.
        # -------------------------------------------------------------
        legacy_slug_key = legacy_git_removed_slug_key(
            normalized_slug
        )

        if legacy_slug_key != slug_key:
            legacy_matches = list(
                self.by_slug.get(legacy_slug_key, [])
            )

            # Perform a complete scan as an additional safeguard for
            # unusual nested directory layouts.
            for candidate in self.all_files:
                candidate_keys = self._infer_slug_keys(candidate)

                if (
                        legacy_slug_key in candidate_keys
                        and candidate not in legacy_matches
                ):
                    legacy_matches.append(candidate)

            match = self._single_match(
                legacy_matches,
                (
                    f"legacy '.git' removal identity "
                    f"{normalized_slug} -> {legacy_slug_key}"
                ),
            )

            if match is not None:
                print(
                    "  Resolved issue file using historical '.git' "
                    "removal:\n"
                    f"    GitHub repository: {normalized_slug}\n"
                    f"    Legacy identity:   {legacy_slug_key}\n"
                    f"    Issue file:        {match}"
                )
                return match

        # Compatibility fallback for flat files saved without the owner prefix.
        repository = normalized_slug.split("/", 1)[1]
        legacy_names = [
            f"{safe_identifier(repository)}_issues.json",
            f"{safe_identifier(repository)}_vulnerabilities.json",
        ]

        legacy_matches: list[Path] = []
        for legacy_name in legacy_names:
            legacy_matches.extend(self.by_name.get(legacy_name.casefold(), []))

        match = self._single_match(
            legacy_matches,
            f"legacy repository-only filename for {normalized_slug}",
        )
        if match is not None:
            return match

        raise ValueError(
            f"Issue file not found for {normalized_slug}. Matching is "
            f"case-insensitive. Expected a layout such as "
            f"'{expected}', 'owner/repository_issues.json', or "
            f"'owner/repository/issues.json' under {self.directory}."
        )


def load_issues(path: Path) -> list[Issue]:
    """Load either a JSON issue array or a SonarQube response object."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    if isinstance(data, dict) and isinstance(data.get("issues"), list):
        data = data["issues"]

    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain a JSON array or an object with an 'issues' array."
        )

    invalid_indexes = [
        index for index, item in enumerate(data) if not isinstance(item, dict)
    ]
    if invalid_indexes:
        raise ValueError(
            f"{path} contains non-object entries at indexes {invalid_indexes[:5]}."
        )

    return data


def normalize_component(component: str) -> str:
    """Convert a SonarQube component into a repository-relative path."""
    if not component:
        return ""

    if ":" in component:
        component = component.split(":", 1)[1]

    component = component.replace("\\", "/")
    parts = list(PurePosixPath(component).parts)

    for marker in ("temp_repos", "temp_template_repos"):
        if marker in parts:
            marker_index = parts.index(marker)
            relative_start = marker_index + 2
            if relative_start < len(parts):
                parts = parts[relative_start:]
            break

    return "/".join(part for part in parts if part not in {"/", ""})


def get_issue_hash(issue: Issue) -> str:
    return str(issue.get("hash") or "").strip()


def exact_signature(issue: Issue) -> tuple[str, str, str]:
    return (
        str(issue.get("rule") or ""),
        normalize_component(str(issue.get("component") or "")),
        get_issue_hash(issue),
    )


def moved_signature(issue: Issue) -> tuple[str, str, str, str]:
    return (
        str(issue.get("rule") or ""),
        get_issue_hash(issue),
        str(issue.get("message") or ""),
        str(issue.get("type") or ""),
    )


def fallback_signature(issue: Issue) -> tuple[str, str, str, int | None]:
    return (
        str(issue.get("rule") or ""),
        normalize_component(str(issue.get("component") or "")),
        str(issue.get("message") or ""),
        issue.get("line"),
    )


def template_issue_identity(issue: Issue) -> str:
    """Stable identifier used to aggregate one template issue across descendants."""
    issue_key = str(issue.get("key") or "").strip()
    if issue_key:
        return issue_key

    parts = [
        str(issue.get("rule") or ""),
        normalize_component(str(issue.get("component") or "")),
        get_issue_hash(issue),
        str(issue.get("message") or ""),
        str(issue.get("line") if issue.get("line") is not None else ""),
    ]
    return "|".join(parts)


def software_qualities(issue: Issue) -> list[str]:
    impacts = issue.get("impacts") or []
    return sorted(
        {
            str(impact.get("softwareQuality"))
            for impact in impacts
            if isinstance(impact, dict) and impact.get("softwareQuality")
        }
    )


def issue_summary(issue: Issue) -> dict[str, Any]:
    return {
        "key": issue.get("key"),
        "rule": issue.get("rule"),
        "type": issue.get("type"),
        "severity": issue.get("severity"),
        "software_qualities": software_qualities(issue),
        "component": issue.get("component"),
        "relative_path": normalize_component(str(issue.get("component") or "")),
        "line": issue.get("line"),
        "hash": issue.get("hash"),
        "message": issue.get("message"),
        "effort": issue.get("effort"),
        "debt": issue.get("debt"),
        "status": issue.get("status"),
        "issue_status": issue.get("issueStatus"),
    }


def compare_issues(
    template_issues: list[Issue],
    project_issues: list[Issue],
) -> PairResult:
    """Match each template issue to at most one generated-project issue."""
    matched: list[dict[str, Any]] = []
    removed: list[Issue] = []

    matched_project_indexes: set[int] = set()
    unmatched_template_indexes: list[int] = []

    exact_project_index: dict[tuple[str, str, str], deque[int]] = defaultdict(deque)
    fallback_project_index: dict[
        tuple[str, str, str, int | None], deque[int]
    ] = defaultdict(deque)

    for project_index, project_issue in enumerate(project_issues):
        if get_issue_hash(project_issue):
            exact_project_index[exact_signature(project_issue)].append(project_index)
        else:
            fallback_project_index[fallback_signature(project_issue)].append(
                project_index
            )

    for template_index, template_issue in enumerate(template_issues):
        project_index: int | None = None
        match_type: str | None = None

        if get_issue_hash(template_issue):
            candidates = exact_project_index[exact_signature(template_issue)]
            while candidates:
                candidate = candidates.popleft()
                if candidate not in matched_project_indexes:
                    project_index = candidate
                    match_type = "same_path_and_hash"
                    break
        else:
            candidates = fallback_project_index[fallback_signature(template_issue)]
            while candidates:
                candidate = candidates.popleft()
                if candidate not in matched_project_indexes:
                    project_index = candidate
                    match_type = "fallback_without_hash"
                    break

        if project_index is None:
            unmatched_template_indexes.append(template_index)
            continue

        matched_project_indexes.add(project_index)
        matched.append(
            {
                "match_type": match_type,
                "template_issue": template_issue,
                "project_issue": project_issues[project_index],
            }
        )

    moved_project_index: dict[
        tuple[str, str, str, str], deque[int]
    ] = defaultdict(deque)

    for project_index, project_issue in enumerate(project_issues):
        if project_index in matched_project_indexes or not get_issue_hash(project_issue):
            continue
        moved_project_index[moved_signature(project_issue)].append(project_index)

    for template_index in unmatched_template_indexes:
        template_issue = template_issues[template_index]

        if not get_issue_hash(template_issue):
            removed.append(template_issue)
            continue

        candidates = moved_project_index[moved_signature(template_issue)]
        project_index = None

        while candidates:
            candidate = candidates.popleft()
            if candidate not in matched_project_indexes:
                project_index = candidate
                break

        if project_index is None:
            removed.append(template_issue)
            continue

        matched_project_indexes.add(project_index)
        matched.append(
            {
                "match_type": "same_hash_in_moved_or_renamed_file",
                "template_issue": template_issue,
                "project_issue": project_issues[project_index],
            }
        )

    new_project_issues = [
        issue
        for index, issue in enumerate(project_issues)
        if index not in matched_project_indexes
    ]

    return {"matched": matched, "removed": removed, "new": new_project_issues}


def rate(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 2)


def count_by(
    issues: Iterable[Issue],
    key_function: Callable[[Issue], Iterable[str]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for issue in issues:
        labels = list(key_function(issue)) or ["UNKNOWN"]
        for label in labels:
            counts[str(label or "UNKNOWN")] += 1
    return dict(sorted(counts.items()))


def build_breakdown(
    template_issues: list[Issue],
    project_issues: list[Issue],
    matched: list[dict[str, Any]],
    removed: list[Issue],
    new_issues: list[Issue],
    key_function: Callable[[Issue], Iterable[str]],
) -> dict[str, dict[str, Any]]:
    template_counts = count_by(template_issues, key_function)
    project_counts = count_by(project_issues, key_function)
    persistent_counts = count_by(
        (item["template_issue"] for item in matched), key_function
    )
    removed_counts = count_by(removed, key_function)
    new_counts = count_by(new_issues, key_function)

    labels = sorted(
        set(template_counts)
        | set(project_counts)
        | set(persistent_counts)
        | set(removed_counts)
        | set(new_counts)
    )

    result: dict[str, dict[str, Any]] = {}
    for label in labels:
        template_count = template_counts.get(label, 0)
        persistent_count = persistent_counts.get(label, 0)
        persistence_rate = rate(persistent_count, template_count)

        result[label] = {
            "template_issue_count": template_count,
            "project_issue_count": project_counts.get(label, 0),
            "persistent_issue_count": persistent_count,
            "no_longer_present_issue_count": removed_counts.get(label, 0),
            "new_issue_count": new_counts.get(label, 0),
            "persistence_rate": persistence_rate,
            "persistence_percentage": percent(persistence_rate),
        }

    return result


def build_pair_summary(
    pair: dict[str, Any],
    template_file: Path,
    project_file: Path,
    template_issues: list[Issue],
    project_issues: list[Issue],
    result: PairResult,
) -> dict[str, Any]:
    matched = result["matched"]
    removed = result["removed"]
    new_issues = result["new"]

    template_count = len(template_issues)
    project_count = len(project_issues)
    persistent_count = len(matched)
    removed_count = len(removed)
    new_count = len(new_issues)

    persistence_rate = rate(persistent_count, template_count)
    removal_rate = rate(removed_count, template_count)
    new_issue_share = rate(new_count, project_count)

    match_type_counts: dict[str, int] = defaultdict(int)
    for item in matched:
        match_type_counts[str(item["match_type"])] += 1

    return {
        "pair_id": pair["pair_id"],
        "source_csv_row": pair["source_csv_row"],
        "template_id": pair["template_slug"],
        "project_id": pair["project_slug"],
        "template_repository": pair["template_slug"],
        "generated_repository": pair["project_slug"],
        "template_repository_issue_file": str(template_file),
        "generated_repository_issue_file": str(project_file),
        "original_link": pair.get("original_link", ""),
        "evolution_band": pair.get("evolution_band", ""),
        "sampling_method": pair.get("sampling_method", ""),
        "selection_probability": pair.get("selection_probability"),
        "sampling_weight": pair.get("sampling_weight"),
        "template_issue_count": template_count,
        "project_issue_count": project_count,
        "persistent_issue_count": persistent_count,
        "no_longer_present_issue_count": removed_count,
        "new_issue_count": new_count,
        "same_path_and_hash_match_count": match_type_counts.get(
            "same_path_and_hash", 0
        ),
        "moved_or_renamed_match_count": match_type_counts.get(
            "same_hash_in_moved_or_renamed_file", 0
        ),
        "fallback_without_hash_match_count": match_type_counts.get(
            "fallback_without_hash", 0
        ),
        "persistence_rate": persistence_rate,
        "persistence_percentage": percent(persistence_rate),
        "removal_rate": removal_rate,
        "removal_percentage": percent(removal_rate),
        "new_issue_share_of_project_issues": new_issue_share,
        "new_issue_percentage_of_project_issues": percent(new_issue_share),
        "breakdown_by_type": build_breakdown(
            template_issues,
            project_issues,
            matched,
            removed,
            new_issues,
            lambda issue: [str(issue.get("type") or "UNKNOWN")],
        ),
        "breakdown_by_severity": build_breakdown(
            template_issues,
            project_issues,
            matched,
            removed,
            new_issues,
            lambda issue: [str(issue.get("severity") or "UNKNOWN")],
        ),
        "breakdown_by_software_quality": build_breakdown(
            template_issues,
            project_issues,
            matched,
            removed,
            new_issues,
            lambda issue: software_qualities(issue) or ["UNKNOWN"],
        ),
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_pair_comparison_csv(
    path: Path,
    pair_summary: dict[str, Any],
    result: PairResult,
) -> None:
    fields = [
        "pair_id",
        "template_repository",
        "generated_repository",
        "classification",
        "match_type",
        "rule",
        "type",
        "severity",
        "software_qualities",
        "template_issue_key",
        "project_issue_key",
        "template_path",
        "template_line",
        "project_path",
        "project_line",
        "hash",
        "message",
    ]

    rows: list[dict[str, Any]] = []
    common = {
        "pair_id": pair_summary["pair_id"],
        "template_repository": pair_summary["template_repository"],
        "generated_repository": pair_summary["generated_repository"],
    }

    for match in result["matched"]:
        template = match["template_issue"]
        project = match["project_issue"]
        rows.append(
            {
                **common,
                "classification": "persistent_in_generated_repository",
                "match_type": match["match_type"],
                "rule": template.get("rule"),
                "type": template.get("type"),
                "severity": template.get("severity"),
                "software_qualities": "|".join(software_qualities(template)),
                "template_issue_key": template.get("key"),
                "project_issue_key": project.get("key"),
                "template_path": normalize_component(
                    str(template.get("component") or "")
                ),
                "template_line": template.get("line"),
                "project_path": normalize_component(
                    str(project.get("component") or "")
                ),
                "project_line": project.get("line"),
                "hash": template.get("hash"),
                "message": template.get("message"),
            }
        )

    for template in result["removed"]:
        rows.append(
            {
                **common,
                "classification": "not_persistent_in_generated_repository",
                "match_type": "",
                "rule": template.get("rule"),
                "type": template.get("type"),
                "severity": template.get("severity"),
                "software_qualities": "|".join(software_qualities(template)),
                "template_issue_key": template.get("key"),
                "project_issue_key": "",
                "template_path": normalize_component(
                    str(template.get("component") or "")
                ),
                "template_line": template.get("line"),
                "project_path": "",
                "project_line": "",
                "hash": template.get("hash"),
                "message": template.get("message"),
            }
        )

    for project in result["new"]:
        rows.append(
            {
                **common,
                "classification": "new_in_generated_repository",
                "match_type": "",
                "rule": project.get("rule"),
                "type": project.get("type"),
                "severity": project.get("severity"),
                "software_qualities": "|".join(software_qualities(project)),
                "template_issue_key": "",
                "project_issue_key": project.get("key"),
                "template_path": "",
                "template_line": "",
                "project_path": normalize_component(
                    str(project.get("component") or "")
                ),
                "project_line": project.get("line"),
                "hash": project.get("hash"),
                "message": project.get("message"),
            }
        )

    save_csv(path, rows, fields)


def summary_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = [
        "pair_id",
        "source_csv_row",
        "template_repository",
        "generated_repository",
        "template_repository_issue_file",
        "generated_repository_issue_file",
        "original_link",
        "evolution_band",
        "sampling_method",
        "selection_probability",
        "sampling_weight",
        "template_issue_count",
        "project_issue_count",
        "persistent_issue_count",
        "no_longer_present_issue_count",
        "new_issue_count",
        "same_path_and_hash_match_count",
        "moved_or_renamed_match_count",
        "fallback_without_hash_match_count",
        "persistence_rate",
        "persistence_percentage",
        "removal_rate",
        "removal_percentage",
        "new_issue_share_of_project_issues",
        "new_issue_percentage_of_project_issues",
    ]
    return {field: summary.get(field) for field in scalar_fields}


def build_aggregate_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total_template = sum(item["template_issue_count"] for item in summaries)
    total_project = sum(item["project_issue_count"] for item in summaries)
    total_persistent = sum(item["persistent_issue_count"] for item in summaries)
    total_removed = sum(
        item["no_longer_present_issue_count"] for item in summaries
    )
    total_new = sum(item["new_issue_count"] for item in summaries)

    persistence_values = [
        item["persistence_rate"]
        for item in summaries
        if item["persistence_rate"] is not None
    ]
    removal_values = [
        item["removal_rate"]
        for item in summaries
        if item["removal_rate"] is not None
    ]

    micro_persistence = rate(total_persistent, total_template)
    micro_removal = rate(total_removed, total_template)
    macro_persistence = mean(persistence_values) if persistence_values else None
    macro_removal = mean(removal_values) if removal_values else None

    weighted_persistent = 0.0
    weighted_template = 0.0
    weighted_pair_rates: list[tuple[float, float]] = []

    for summary in summaries:
        weight = parse_float(summary.get("sampling_weight"))
        if weight is None or weight <= 0:
            continue
        weighted_persistent += summary["persistent_issue_count"] * weight
        weighted_template += summary["template_issue_count"] * weight
        if summary["persistence_rate"] is not None:
            weighted_pair_rates.append((summary["persistence_rate"], weight))

    weighted_micro = rate(weighted_persistent, weighted_template)
    weighted_macro = None
    if weighted_pair_rates:
        weighted_macro = sum(value * weight for value, weight in weighted_pair_rates) / sum(
            weight for _, weight in weighted_pair_rates
        )

    return {
        "analyzed_pair_count": len(summaries),
        "unique_template_count": len(
            {item["template_repository"] for item in summaries}
        ),
        "unique_generated_repository_count": len(
            {item["generated_repository"] for item in summaries}
        ),
        "template_issue_observation_count": total_template,
        "project_issue_observation_count": total_project,
        "persistent_issue_observation_count": total_persistent,
        "no_longer_present_issue_observation_count": total_removed,
        "new_issue_observation_count": total_new,
        "micro_persistence_rate": micro_persistence,
        "micro_persistence_percentage": percent(micro_persistence),
        "macro_average_pair_persistence_rate": macro_persistence,
        "macro_average_pair_persistence_percentage": percent(macro_persistence),
        "sampling_weighted_micro_persistence_rate": weighted_micro,
        "sampling_weighted_micro_persistence_percentage": percent(weighted_micro),
        "sampling_weighted_macro_pair_persistence_rate": weighted_macro,
        "sampling_weighted_macro_pair_persistence_percentage": percent(weighted_macro),
        "micro_removal_rate": micro_removal,
        "micro_removal_percentage": percent(micro_removal),
        "macro_average_pair_removal_rate": macro_removal,
        "macro_average_pair_removal_percentage": percent(macro_removal),
        "interpretation_note": (
            "Counts are issue observations across template/generated-repository "
            "pairs. A template issue is counted once for every generated "
            "repository compared with that template."
        ),
    }


def build_template_summary(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[summary["template_repository"]].append(summary)

    rows: list[dict[str, Any]] = []
    for template, items in sorted(grouped.items()):
        total_template_observations = sum(
            item["template_issue_count"] for item in items
        )
        total_persistent = sum(item["persistent_issue_count"] for item in items)
        total_removed = sum(
            item["no_longer_present_issue_count"] for item in items
        )
        pair_rates = [
            item["persistence_rate"]
            for item in items
            if item["persistence_rate"] is not None
        ]
        micro = rate(total_persistent, total_template_observations)
        macro = mean(pair_rates) if pair_rates else None

        rows.append(
            {
                "template_repository": template,
                "generated_repository_count": len(items),
                "template_issue_count_per_repository": items[0][
                    "template_issue_count"
                ],
                "template_issue_observation_count": total_template_observations,
                "persistent_issue_observation_count": total_persistent,
                "not_persistent_issue_observation_count": total_removed,
                "micro_persistence_rate": micro,
                "micro_persistence_percentage": percent(micro),
                "macro_average_pair_persistence_rate": macro,
                "macro_average_pair_persistence_percentage": percent(macro),
            }
        )

    return rows


def add_template_issue_observations(
    storage: dict[tuple[str, str], dict[str, Any]],
    pair: dict[str, Any],
    result: PairResult,
) -> None:
    weight = parse_float(pair.get("sampling_weight"))
    if weight is None or weight <= 0:
        weight = 1.0

    for match in result["matched"]:
        issue = match["template_issue"]
        key = (pair["template_slug"], template_issue_identity(issue))
        record = storage.setdefault(
            key,
            {
                "template_repository": pair["template_slug"],
                "template_issue_identity": key[1],
                **issue_summary(issue),
                "compared_project_count": 0,
                "persistent_project_count": 0,
                "not_persistent_project_count": 0,
                "weighted_compared_project_count": 0.0,
                "weighted_persistent_project_count": 0.0,
                "same_path_and_hash_match_count": 0,
                "moved_or_renamed_match_count": 0,
                "fallback_without_hash_match_count": 0,
            },
        )
        record["compared_project_count"] += 1
        record["persistent_project_count"] += 1
        record["weighted_compared_project_count"] += weight
        record["weighted_persistent_project_count"] += weight

        if match["match_type"] == "same_path_and_hash":
            record["same_path_and_hash_match_count"] += 1
        elif match["match_type"] == "same_hash_in_moved_or_renamed_file":
            record["moved_or_renamed_match_count"] += 1
        elif match["match_type"] == "fallback_without_hash":
            record["fallback_without_hash_match_count"] += 1

    for issue in result["removed"]:
        key = (pair["template_slug"], template_issue_identity(issue))
        record = storage.setdefault(
            key,
            {
                "template_repository": pair["template_slug"],
                "template_issue_identity": key[1],
                **issue_summary(issue),
                "compared_project_count": 0,
                "persistent_project_count": 0,
                "not_persistent_project_count": 0,
                "weighted_compared_project_count": 0.0,
                "weighted_persistent_project_count": 0.0,
                "same_path_and_hash_match_count": 0,
                "moved_or_renamed_match_count": 0,
                "fallback_without_hash_match_count": 0,
            },
        )
        record["compared_project_count"] += 1
        record["not_persistent_project_count"] += 1
        record["weighted_compared_project_count"] += weight


def finalize_template_issue_observations(
    storage: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, record in sorted(storage.items(), key=lambda item: item[0]):
        persistence = rate(
            record["persistent_project_count"], record["compared_project_count"]
        )
        weighted_persistence = rate(
            record["weighted_persistent_project_count"],
            record["weighted_compared_project_count"],
        )

        row = dict(record)
        row["software_qualities"] = "|".join(record.get("software_qualities") or [])
        row["persistence_rate_across_generated_repositories"] = persistence
        row["persistence_percentage_across_generated_repositories"] = percent(
            persistence
        )
        row[
            "weighted_persistence_rate_across_generated_repositories"
        ] = weighted_persistence
        row[
            "weighted_persistence_percentage_across_generated_repositories"
        ] = percent(weighted_persistence)
        rows.append(row)

    return rows


def read_sample_pairs(
    csv_path: Path,
    eligible_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    try:
        file = csv_path.open("r", encoding="utf-8-sig", newline="")
    except FileNotFoundError as error:
        raise ValueError(f"Sample CSV not found: {csv_path}") from error

    pairs: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    with file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("The sample CSV has no header row.")
        if "target_template_repository" not in reader.fieldnames and "template_slug" not in reader.fieldnames:
            raise ValueError(
                "The sample CSV must contain target_template_repository or template_slug."
            )

        for row_number, raw_row in enumerate(reader, start=2):
            row = {
                str(key).strip(): str(value or "").strip()
                for key, value in raw_row.items()
                if key is not None
            }

            if not any(row.values()):
                continue
            if eligible_only and not parse_boolean(row.get("eligible")):
                continue

            template_slug = template_slug_from_row(row)
            project_slug = project_slug_from_row(row)

            if not template_slug or not project_slug:
                errors.append(
                    {
                        "source_csv_row": str(row_number),
                        "pair_id": "",
                        "template_repository": template_slug or "",
                        "generated_repository": project_slug or "",
                        "error": (
                            "Could not resolve template or generated repository slug "
                            "from the CSV row."
                        ),
                    }
                )
                continue

            dedup_key = (template_slug.casefold(), project_slug.casefold())
            if dedup_key in seen_pairs:
                continue
            seen_pairs.add(dedup_key)

            pair_id = safe_identifier(
                f"{sonar_project_key(template_slug)}__TO__{sonar_project_key(project_slug)}"
            )

            pairs.append(
                {
                    "source_csv_row": row_number,
                    "pair_id": pair_id,
                    "template_slug": template_slug,
                    "project_slug": project_slug,
                    "original_link": row.get("original_link", ""),
                    "evolution_band": row.get("evolution_band", ""),
                    "sampling_method": row.get("sampling_method", ""),
                    "selection_probability": parse_float(
                        row.get("selection_probability")
                    ),
                    "sampling_weight": parse_float(row.get("sampling_weight")),
                }
            )

    if not pairs:
        raise ValueError("No valid template/generated-repository pairs were found.")

    return pairs, errors


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare template and generated-repository SonarQube issue files "
            "using pairs derived directly from a sampling CSV."
        )
    )
    parser.add_argument("sample_csv", type=Path)
    parser.add_argument("template_issue_directory", type=Path)
    parser.add_argument("project_issue_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--eligible-only",
        action="store_true",
        help="Process only rows whose eligible column is true.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first missing file or invalid pair.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many unique pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print(f"compare_sonar_issue_pairs_from_sample.py version: {SCRIPT_VERSION}")

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero.")

    sample_csv = args.sample_csv.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()
    pairs_directory = output_directory / "pairs"
    output_directory.mkdir(parents=True, exist_ok=True)
    pairs_directory.mkdir(parents=True, exist_ok=True)

    try:
        pairs, input_errors = read_sample_pairs(sample_csv, args.eligible_only)
        template_index = IssueFileIndex(args.template_issue_directory.expanduser())
        project_index = IssueFileIndex(args.project_issue_directory.expanduser())
        print(
            f"Indexed {len(template_index.all_files)} template issue files "
            f"and {len(project_index.all_files)} generated-project issue files."
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.limit is not None:
        pairs = pairs[: args.limit]

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = list(input_errors)
    resolved_pair_rows: list[dict[str, Any]] = []
    issue_cache: dict[Path, list[Issue]] = {}
    template_issue_observations: dict[tuple[str, str], dict[str, Any]] = {}

    for index, pair in enumerate(pairs, start=1):
        print(
            f"[{index}/{len(pairs)}] Template={pair['template_slug']} | "
            f"Generated={pair['project_slug']}"
        )

        try:
            template_file = template_index.resolve(pair["template_slug"])
            project_file = project_index.resolve(pair["project_slug"])

            print(f"  template issue file:  {template_file}")
            print(f"  generated issue file: {project_file}")

            resolved_pair_rows.append(
                {
                    "source_csv_row": pair["source_csv_row"],
                    "pair_id": pair["pair_id"],
                    "template_repository": pair["template_slug"],
                    "generated_repository": pair["project_slug"],
                    "template_repository_issue_file": str(template_file),
                    "generated_repository_issue_file": str(project_file),
                    "status": "resolved",
                }
            )

            if template_file not in issue_cache:
                issue_cache[template_file] = load_issues(template_file)
            if project_file not in issue_cache:
                issue_cache[project_file] = load_issues(project_file)

            template_issues = issue_cache[template_file]
            project_issues = issue_cache[project_file]
            result = compare_issues(template_issues, project_issues)

            summary = build_pair_summary(
                pair,
                template_file,
                project_file,
                template_issues,
                project_issues,
                result,
            )

            pair_directory = pairs_directory / pair["pair_id"]
            pair_directory.mkdir(parents=True, exist_ok=True)

            save_json(pair_directory / "summary.json", summary)
            save_pair_comparison_csv(
                pair_directory / "issue_comparison.csv", summary, result
            )
            save_json(
                pair_directory / "matched_issues.json",
                [
                    {
                        "pair_id": pair["pair_id"],
                        "template_repository": pair["template_slug"],
                        "generated_repository": pair["project_slug"],
                        "match_type": item["match_type"],
                        "template_issue": issue_summary(item["template_issue"]),
                        "project_issue": issue_summary(item["project_issue"]),
                    }
                    for item in result["matched"]
                ],
            )
            save_json(
                pair_directory / "not_persistent_template_issues.json",
                [
                    {
                        "pair_id": pair["pair_id"],
                        "template_repository": pair["template_slug"],
                        "generated_repository": pair["project_slug"],
                        "template_issue": issue_summary(issue),
                    }
                    for issue in result["removed"]
                ],
            )
            save_json(
                pair_directory / "new_generated_repository_issues.json",
                [
                    {
                        "pair_id": pair["pair_id"],
                        "template_repository": pair["template_slug"],
                        "generated_repository": pair["project_slug"],
                        "project_issue": issue_summary(issue),
                    }
                    for issue in result["new"]
                ],
            )

            add_template_issue_observations(
                template_issue_observations, pair, result
            )
            summaries.append(summary)

            print(
                "  "
                f"persistent={summary['persistent_issue_count']}, "
                f"not_persistent={summary['no_longer_present_issue_count']}, "
                f"new={summary['new_issue_count']}, "
                f"persistence={summary['persistence_percentage']}%"
            )

        except (OSError, ValueError) as error:
            errors.append(
                {
                    "source_csv_row": str(pair["source_csv_row"]),
                    "pair_id": pair["pair_id"],
                    "template_repository": pair["template_slug"],
                    "generated_repository": pair["project_slug"],
                    "error": str(error),
                }
            )
            resolved_pair_rows.append(
                {
                    "source_csv_row": pair["source_csv_row"],
                    "pair_id": pair["pair_id"],
                    "template_repository": pair["template_slug"],
                    "generated_repository": pair["project_slug"],
                    "template_repository_issue_file": "",
                    "generated_repository_issue_file": "",
                    "status": f"error: {error}",
                }
            )
            print(f"  ERROR: {error}")
            if args.fail_fast:
                raise SystemExit(str(error)) from error

    save_csv(output_directory / "resolved_pairs.csv", resolved_pair_rows)

    if summaries:
        summary_rows = [summary_csv_row(summary) for summary in summaries]
        save_csv(output_directory / "pair_summary.csv", summary_rows)
        save_json(output_directory / "pair_summary.json", summaries)
        save_json(
            output_directory / "aggregate_summary.json",
            build_aggregate_summary(summaries),
        )

        template_rows = build_template_summary(summaries)
        save_csv(output_directory / "template_summary.csv", template_rows)
        save_json(output_directory / "template_summary.json", template_rows)

        issue_rows = finalize_template_issue_observations(
            template_issue_observations
        )
        save_csv(output_directory / "template_issue_persistence.csv", issue_rows)
        save_json(output_directory / "template_issue_persistence.json", issue_rows)

    if errors:
        save_csv(
            output_directory / "errors.csv",
            errors,
            [
                "source_csv_row",
                "pair_id",
                "template_repository",
                "generated_repository",
                "error",
            ],
        )

    print("\nCompleted")
    print(f"Pairs found in CSV: {len(pairs)}")
    print(f"Successful pairs: {len(summaries)}")
    print(f"Failed pairs: {len(errors)}")
    print(f"Output directory: {output_directory}")

    if not summaries:
        raise SystemExit(
            "No pair was successfully analyzed. See errors.csv and "
            "resolved_pairs.csv for details."
        )


if __name__ == "__main__":
    main()
