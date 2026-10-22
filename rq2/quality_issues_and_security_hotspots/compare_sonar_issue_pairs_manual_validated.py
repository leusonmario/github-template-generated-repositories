#!/usr/bin/env python3
"""
compare_sonar_issue_pairs_from_sample.py

Build template/generated-repository pairs directly from a sampling CSV and
compare both SonarQube issues and Security Hotspots between each template and
its generated repository.

The original issue-persistence outputs are preserved. Security Hotspots are
analyzed separately because SonarQube models them separately from ordinary
issues/vulnerabilities.

Default layout
--------------
The same directories may contain both issue and hotspot exports. Common names
are recognized automatically, including:

    owner__repo_issues.json
    owner__repo_vulnerabilities.json
    owner__repo_hotspots.json
    owner__repo_security_hotspots.json
    owner/repo/issues.json
    owner/repo/hotspots.json
    owner__repo_all_findings.json   # object containing issues/hotspots arrays

If hotspot exports live elsewhere, pass:

    --template-hotspot-directory PATH
    --project-hotspot-directory PATH

Example
-------
    python compare_sonar_issue_pairs_from_sample.py \
        ../sample/persistence_sample_balanced/Java/eligible_candidates.csv \
        Java/issues_projects_templates \
        Java/issues_projects \
        Java/issue_persistence_results \
        --eligible-only

The command above assumes hotspot JSON files are somewhere under the same two
input directories as the issue JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Callable, Iterable
from urllib.parse import urlparse


Finding = dict[str, Any]
ComparisonResult = dict[str, Any]
SCRIPT_VERSION = "3.2-artifact-exclusion-manual-validation-hotspot-policy"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

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
    for column in ("template_slug", "target_template_repository"):
        slug = normalize_github_slug(row.get(column))
        if slug:
            return slug
    return None


def sonar_project_key(slug: str) -> str:
    owner, repository = slug.split("/", 1)
    return f"{owner}__{repository}"


def safe_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def finding_filename_identifier(value: str) -> str:
    """Preserve meaningful trailing hyphens in repository names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unknown"


def canonical_slug_key(value: str) -> str:
    normalized = normalize_github_slug(value)
    if normalized is None:
        raise ValueError(f"Invalid GitHub repository identifier: {value}")
    return unicodedata.normalize("NFKC", normalized).casefold()


def legacy_git_removed_slug_key(value: str) -> str:
    """Reproduce the historical transformation that removed '.git' anywhere."""
    normalized = normalize_github_slug(value)
    if normalized is None:
        raise ValueError(f"Invalid GitHub repository identifier: {value}")

    owner, repository = normalized.split("/", 1)
    owner = unicodedata.normalize("NFKC", owner).casefold()
    repository = unicodedata.normalize("NFKC", repository).casefold()
    repository = repository.replace(".git", "")
    return f"{owner}/{repository}"


def rate(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 2)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Finding-file resolution
# ---------------------------------------------------------------------------

class FindingFileIndex:
    """Case-insensitive repository index for issue or hotspot JSON files."""

    ISSUE_SUFFIXES = (
        "_issues",
        "_sonar_issues",
        "_vulnerabilities",
        "_all_findings",
    )
    ISSUE_GENERIC_STEMS = {
        "issues",
        "sonar_issues",
        "vulnerabilities",
        "all_findings",
    }

    HOTSPOT_SUFFIXES = (
        "_security_hotspots",
        "_hotspots",
        "_sonar_hotspots",
        "_securityhotspots",
        "_all_findings",
    )
    HOTSPOT_GENERIC_STEMS = {
        "security_hotspots",
        "hotspots",
        "sonar_hotspots",
        "securityhotspots",
        "all_findings",
    }

    def __init__(self, directory: Path, kind: str) -> None:
        if kind not in {"issues", "hotspots"}:
            raise ValueError(f"Unknown finding kind: {kind}")

        self.kind = kind
        self.directory = directory.expanduser().resolve()
        if not self.directory.exists():
            raise ValueError(f"Finding directory not found: {self.directory}")
        if not self.directory.is_dir():
            raise ValueError(f"Finding path is not a directory: {self.directory}")

        self.by_name: dict[str, list[Path]] = defaultdict(list)
        self.by_slug: dict[str, list[Path]] = defaultdict(list)
        self.all_files: list[Path] = []

        for path in sorted(self.directory.rglob("*.json")):
            if not path.is_file() or not self._looks_like_kind(path):
                continue

            resolved = path.resolve()
            self.all_files.append(resolved)
            self.by_name[path.name.casefold()].append(resolved)

            for slug_key in self._infer_slug_keys(path):
                if resolved not in self.by_slug[slug_key]:
                    self.by_slug[slug_key].append(resolved)

    @property
    def suffixes(self) -> tuple[str, ...]:
        return self.ISSUE_SUFFIXES if self.kind == "issues" else self.HOTSPOT_SUFFIXES

    @property
    def generic_stems(self) -> set[str]:
        return (
            self.ISSUE_GENERIC_STEMS
            if self.kind == "issues"
            else self.HOTSPOT_GENERIC_STEMS
        )

    def _looks_like_kind(self, path: Path) -> bool:
        stem = path.stem.casefold()

        # Explicit hotspot names must never be indexed as ordinary issue files.
        hotspot_named = (
            any(stem.endswith(suffix) for suffix in self.HOTSPOT_SUFFIXES[:-1])
            or stem in self.HOTSPOT_GENERIC_STEMS - {"all_findings"}
        )

        if self.kind == "issues" and hotspot_named:
            return False

        return (
            any(stem.endswith(suffix) for suffix in self.suffixes)
            or stem in self.generic_stems
        )

    def _strip_suffix(self, stem: str) -> str:
        folded = stem.casefold()
        # Longest first prevents _hotspots from stripping a longer suffix badly.
        for suffix in sorted(self.suffixes, key=len, reverse=True):
            if folded.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    @staticmethod
    def _slug_key(owner: str, repository: str) -> str | None:
        owner = unicodedata.normalize("NFKC", owner.strip().strip("/\\"))
        repository = unicodedata.normalize(
            "NFKC", repository.strip().strip("/\\")
        )

        if repository.casefold().endswith(".git"):
            repository = repository[:-4]

        if not owner or not repository:
            return None

        return f"{owner}/{repository}".casefold()

    def _infer_slug_keys(self, path: Path) -> set[str]:
        keys: set[str] = set()
        relative = path.relative_to(self.directory)
        parts = relative.parts
        stripped_stem = self._strip_suffix(path.stem)

        # owner__repository_<kind>.json
        if "__" in stripped_stem:
            owner, repository = stripped_stem.split("__", 1)
            key = self._slug_key(owner, repository)
            if key:
                keys.add(key)

        # owner/repository_<kind>.json
        if len(parts) >= 2 and stripped_stem.casefold() not in self.generic_stems:
            owner = parts[-2]
            repository = stripped_stem
            key = self._slug_key(owner, repository)
            if key:
                keys.add(key)

        # owner/repository/issues.json or owner/repository/hotspots.json
        if len(parts) >= 3 and path.stem.casefold() in self.generic_stems:
            owner = parts[-3]
            repository = parts[-2]
            key = self._slug_key(owner, repository)
            if key:
                keys.add(key)

        return keys

    @staticmethod
    def _single_match(matches: list[Path], description: str) -> Path | None:
        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) == 1:
            return unique_matches[0]
        if len(unique_matches) > 1:
            raise ValueError(
                f"Ambiguous finding files for {description}: "
                + ", ".join(str(path) for path in unique_matches)
            )
        return None

    def expected_filenames(self, slug: str) -> list[str]:
        base = finding_filename_identifier(sonar_project_key(slug))
        if self.kind == "issues":
            suffixes = ["_issues.json", "_vulnerabilities.json", "_all_findings.json"]
        else:
            suffixes = [
                "_security_hotspots.json",
                "_hotspots.json",
                "_sonar_hotspots.json",
                "_all_findings.json",
            ]
        return [f"{base}{suffix}" for suffix in suffixes]

    def resolve(self, slug: str) -> Path:
        normalized_slug = normalize_github_slug(slug)
        if normalized_slug is None:
            raise ValueError(f"Invalid GitHub repository identifier: {slug}")

        for expected in self.expected_filenames(normalized_slug):
            match = self._single_match(
                self.by_name.get(expected.casefold(), []),
                f"{normalized_slug} using filename {expected}",
            )
            if match is not None:
                return match

        slug_key = canonical_slug_key(normalized_slug)
        match = self._single_match(
            self.by_slug.get(slug_key, []),
            f"{self.kind} for {normalized_slug}",
        )
        if match is not None:
            return match

        canonical_matches: list[Path] = []
        for candidate in self.all_files:
            if slug_key in self._infer_slug_keys(candidate):
                canonical_matches.append(candidate)

        match = self._single_match(
            canonical_matches,
            f"canonical case-insensitive {self.kind} identity {normalized_slug}",
        )
        if match is not None:
            return match

        # Historical '.git' removal fallback.
        legacy_slug_key = legacy_git_removed_slug_key(normalized_slug)
        if legacy_slug_key != slug_key:
            legacy_matches = list(self.by_slug.get(legacy_slug_key, []))
            for candidate in self.all_files:
                if (
                    legacy_slug_key in self._infer_slug_keys(candidate)
                    and candidate not in legacy_matches
                ):
                    legacy_matches.append(candidate)

            match = self._single_match(
                legacy_matches,
                f"legacy '.git' removal {self.kind} identity "
                f"{normalized_slug} -> {legacy_slug_key}",
            )
            if match is not None:
                print(
                    f"  Resolved {self.kind} file using historical '.git' removal:\n"
                    f"    GitHub repository: {normalized_slug}\n"
                    f"    Legacy identity:   {legacy_slug_key}\n"
                    f"    Finding file:      {match}"
                )
                return match

        # Repository-only flat fallback.
        repository = normalized_slug.split("/", 1)[1]
        if self.kind == "issues":
            legacy_names = [
                f"{safe_identifier(repository)}_issues.json",
                f"{safe_identifier(repository)}_vulnerabilities.json",
            ]
        else:
            legacy_names = [
                f"{safe_identifier(repository)}_hotspots.json",
                f"{safe_identifier(repository)}_security_hotspots.json",
            ]

        legacy_matches: list[Path] = []
        for name in legacy_names:
            legacy_matches.extend(self.by_name.get(name.casefold(), []))

        match = self._single_match(
            legacy_matches,
            f"legacy repository-only {self.kind} filename for {normalized_slug}",
        )
        if match is not None:
            return match

        expected = ", ".join(self.expected_filenames(normalized_slug))
        raise ValueError(
            f"{self.kind.capitalize()} file not found for {normalized_slug}. "
            f"Expected a recognizable layout under {self.directory}, e.g. {expected}."
        )


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def is_hotspot_record(item: Finding) -> bool:
    record_type = str(item.get("type") or "").upper()
    if record_type == "SECURITY_HOTSPOT":
        return True
    return any(
        key in item
        for key in (
            "securityCategory",
            "vulnerabilityProbability",
            "ruleKey",
        )
    ) and record_type not in {"BUG", "CODE_SMELL", "VULNERABILITY"}


def load_findings(path: Path, kind: str) -> list[Finding]:
    """Load issues or hotspots from list, API response, or combined export."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"File not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error

    selected: Any = None

    if isinstance(data, dict):
        if kind == "issues":
            for key in ("issues", "vulnerabilities"):
                if isinstance(data.get(key), list):
                    selected = data[key]
                    break
        else:
            for key in ("hotspots", "security_hotspots", "securityHotspots"):
                if isinstance(data.get(key), list):
                    selected = data[key]
                    break

        if selected is None and isinstance(data.get("findings"), list):
            findings = data["findings"]
            selected = [
                item
                for item in findings
                if isinstance(item, dict)
                and (is_hotspot_record(item) if kind == "hotspots" else not is_hotspot_record(item))
            ]

    elif isinstance(data, list):
        # For explicitly named issue/hotspot files, the list itself is assumed to
        # contain the corresponding finding kind. For all_findings, filter it.
        if path.stem.casefold().endswith("_all_findings") or path.stem.casefold() == "all_findings":
            selected = [
                item
                for item in data
                if isinstance(item, dict)
                and (is_hotspot_record(item) if kind == "hotspots" else not is_hotspot_record(item))
            ]
        else:
            selected = data

    if not isinstance(selected, list):
        expected = "issues" if kind == "issues" else "hotspots"
        raise ValueError(
            f"{path} does not contain a recognizable {expected} array/list."
        )

    invalid_indexes = [
        index for index, item in enumerate(selected) if not isinstance(item, dict)
    ]
    if invalid_indexes:
        raise ValueError(
            f"{path} contains non-object entries at indexes {invalid_indexes[:5]}."
        )

    return selected


# ---------------------------------------------------------------------------
# Shared path normalization
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Reproducible generated-artifact exclusions
# ---------------------------------------------------------------------------

ARTIFACT_POLICIES = ("none", "gradle-report", "java-build-outputs")


def excluded_artifact_reason(finding: Finding, policy: str) -> str | None:
    """Return the deterministic exclusion reason for a SonarQube finding.

    Uses the same repository-relative component normalization as the matcher.
    This policy is intended for Java analyses; do not apply the broad policy
    indiscriminately to other languages without reviewing their repositories.
    """
    if policy == "none":
        return None
    path = normalize_component(str(finding.get("component") or ""))
    parts = tuple(piece.casefold() for piece in path.replace("\\", "/").split("/") if piece)
    if len(parts) >= 4 and parts[-4:] == (
        "build", "reports", "problems", "problems-report.html"
    ):
        return "gradle_problems_report"
    if policy == "java-build-outputs":
        for directory in ("build", "target", ".gradle"):
            if directory in parts[:-1]:
                return f"generated_directory:{directory}"
    return None


def exclude_artifact_findings(
    findings: list[Finding], policy: str
) -> tuple[list[Finding], list[dict[str, str]]]:
    if policy == "none":
        return findings, []
    kept: list[Finding] = []
    excluded: list[dict[str, str]] = []
    for finding in findings:
        reason = excluded_artifact_reason(finding, policy)
        if reason is None:
            kept.append(finding)
        else:
            excluded.append({
                "component": str(finding.get("component") or ""),
                "normalized_path": normalize_component(str(finding.get("component") or "")),
                "rule": str(finding.get("rule") or finding.get("ruleKey") or ""),
                "reason": reason,
            })
    return kept, excluded


# ---------------------------------------------------------------------------
# Ordinary SonarQube issue comparison
# ---------------------------------------------------------------------------

def get_issue_hash(issue: Finding) -> str:
    return str(issue.get("hash") or "").strip()


def exact_issue_signature(issue: Finding) -> tuple[str, str, str]:
    return (
        str(issue.get("rule") or ""),
        normalize_component(str(issue.get("component") or "")),
        get_issue_hash(issue),
    )


def moved_issue_signature(issue: Finding) -> tuple[str, str, str, str]:
    return (
        str(issue.get("rule") or ""),
        get_issue_hash(issue),
        str(issue.get("message") or ""),
        str(issue.get("type") or ""),
    )


def fallback_issue_signature(issue: Finding) -> tuple[str, str, str, int | None]:
    return (
        str(issue.get("rule") or ""),
        normalize_component(str(issue.get("component") or "")),
        str(issue.get("message") or ""),
        issue.get("line"),
    )


def template_issue_identity(issue: Finding) -> str:
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


def software_qualities(issue: Finding) -> list[str]:
    impacts = issue.get("impacts") or []
    return sorted(
        {
            str(impact.get("softwareQuality"))
            for impact in impacts
            if isinstance(impact, dict) and impact.get("softwareQuality")
        }
    )


def issue_summary(issue: Finding) -> dict[str, Any]:
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
    template_issues: list[Finding],
    project_issues: list[Finding],
) -> ComparisonResult:
    """Match every template issue to at most one generated-project issue."""
    matched: list[dict[str, Any]] = []
    removed: list[Finding] = []
    matched_project_indexes: set[int] = set()
    unmatched_template_indexes: list[int] = []

    exact_project_index: dict[tuple[str, str, str], deque[int]] = defaultdict(deque)
    fallback_project_index: dict[
        tuple[str, str, str, int | None], deque[int]
    ] = defaultdict(deque)

    for project_index, project_issue in enumerate(project_issues):
        if get_issue_hash(project_issue):
            exact_project_index[exact_issue_signature(project_issue)].append(project_index)
        else:
            fallback_project_index[fallback_issue_signature(project_issue)].append(
                project_index
            )

    for template_index, template_issue in enumerate(template_issues):
        project_index: int | None = None
        match_type: str | None = None

        if get_issue_hash(template_issue):
            candidates = exact_project_index[exact_issue_signature(template_issue)]
            while candidates:
                candidate = candidates.popleft()
                if candidate not in matched_project_indexes:
                    project_index = candidate
                    match_type = "same_path_and_hash"
                    break
        else:
            candidates = fallback_project_index[fallback_issue_signature(template_issue)]
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

    moved_project_index: dict[tuple[str, str, str, str], deque[int]] = defaultdict(deque)
    for project_index, project_issue in enumerate(project_issues):
        if project_index in matched_project_indexes or not get_issue_hash(project_issue):
            continue
        moved_project_index[moved_issue_signature(project_issue)].append(project_index)

    for template_index in unmatched_template_indexes:
        template_issue = template_issues[template_index]

        if not get_issue_hash(template_issue):
            removed.append(template_issue)
            continue

        candidates = moved_project_index[moved_issue_signature(template_issue)]
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


# ---------------------------------------------------------------------------
# Security Hotspot comparison
# ---------------------------------------------------------------------------

def hotspot_rule_key(hotspot: Finding) -> str:
    value = hotspot.get("ruleKey")
    if value:
        return str(value)

    rule = hotspot.get("rule")
    if isinstance(rule, dict):
        return str(rule.get("key") or "")
    return str(rule or "")


def hotspot_text_range(hotspot: Finding) -> dict[str, Any]:
    value = hotspot.get("textRange")
    return value if isinstance(value, dict) else {}


def hotspot_start_line(hotspot: Finding) -> int | None:
    value = hotspot.get("line")
    if isinstance(value, int):
        return value
    text_range = hotspot_text_range(hotspot)
    value = text_range.get("startLine")
    return value if isinstance(value, int) else None


def hotspot_location_signature(hotspot: Finding) -> tuple[Any, ...]:
    text_range = hotspot_text_range(hotspot)
    return (
        hotspot_start_line(hotspot),
        text_range.get("startOffset"),
        text_range.get("endLine"),
        text_range.get("endOffset"),
    )


def exact_hotspot_signature(hotspot: Finding) -> tuple[Any, ...]:
    return (
        hotspot_rule_key(hotspot),
        normalize_component(str(hotspot.get("component") or "")),
        str(hotspot.get("message") or ""),
        *hotspot_location_signature(hotspot),
    )


def shifted_hotspot_signature(hotspot: Finding) -> tuple[str, str, str]:
    """Same hotspot in same file while tolerating inserted/deleted lines."""
    return (
        hotspot_rule_key(hotspot),
        normalize_component(str(hotspot.get("component") or "")),
        str(hotspot.get("message") or ""),
    )


def moved_hotspot_signature(hotspot: Finding) -> tuple[str, str, str, str]:
    """Conservative path-independent signature for moved/renamed files."""
    return (
        hotspot_rule_key(hotspot),
        str(hotspot.get("message") or ""),
        str(hotspot.get("securityCategory") or ""),
        str(hotspot.get("vulnerabilityProbability") or ""),
    )


def template_hotspot_identity(hotspot: Finding) -> str:
    hotspot_key = str(hotspot.get("key") or "").strip()
    if hotspot_key:
        return hotspot_key

    parts = [
        hotspot_rule_key(hotspot),
        normalize_component(str(hotspot.get("component") or "")),
        str(hotspot.get("message") or ""),
        str(hotspot_start_line(hotspot) or ""),
    ]
    return "|".join(parts)


def hotspot_summary(hotspot: Finding) -> dict[str, Any]:
    return {
        "key": hotspot.get("key"),
        "rule_key": hotspot_rule_key(hotspot),
        "security_category": hotspot.get("securityCategory"),
        "vulnerability_probability": hotspot.get("vulnerabilityProbability"),
        "component": hotspot.get("component"),
        "relative_path": normalize_component(str(hotspot.get("component") or "")),
        "line": hotspot_start_line(hotspot),
        "text_range": hotspot.get("textRange"),
        "message": hotspot.get("message"),
        "status": hotspot.get("status"),
        "resolution": hotspot.get("resolution"),
        "creation_date": hotspot.get("creationDate"),
        "update_date": hotspot.get("updateDate"),
    }


def compare_hotspots(
    template_hotspots: list[Finding],
    project_hotspots: list[Finding],
    *,
    allow_moved_matches: bool = True,
) -> ComparisonResult:
    """
    Match Security Hotspots conservatively.

    Pass 1: rule + path + message + exact location
    Pass 2: rule + path + message, allowing line shifts
    Pass 3: rule + message + category + probability for moved/renamed files,
            but only when the remaining signature is unique on BOTH sides.
    """
    matched: list[dict[str, Any]] = []
    matched_template_indexes: set[int] = set()
    matched_project_indexes: set[int] = set()

    # Pass 1: exact path/location.
    exact_index: dict[tuple[Any, ...], deque[int]] = defaultdict(deque)
    for project_index, hotspot in enumerate(project_hotspots):
        exact_index[exact_hotspot_signature(hotspot)].append(project_index)

    for template_index, hotspot in enumerate(template_hotspots):
        candidates = exact_index[exact_hotspot_signature(hotspot)]
        project_index = None
        while candidates:
            candidate = candidates.popleft()
            if candidate not in matched_project_indexes:
                project_index = candidate
                break
        if project_index is None:
            continue

        matched_template_indexes.add(template_index)
        matched_project_indexes.add(project_index)
        matched.append(
            {
                "match_type": "same_path_message_and_location",
                "template_hotspot": hotspot,
                "project_hotspot": project_hotspots[project_index],
            }
        )

    # Pass 2: same path/rule/message, location may shift.
    shifted_index: dict[tuple[str, str, str], deque[int]] = defaultdict(deque)
    for project_index, hotspot in enumerate(project_hotspots):
        if project_index in matched_project_indexes:
            continue
        shifted_index[shifted_hotspot_signature(hotspot)].append(project_index)

    for template_index, hotspot in enumerate(template_hotspots):
        if template_index in matched_template_indexes:
            continue
        candidates = shifted_index[shifted_hotspot_signature(hotspot)]
        project_index = None
        while candidates:
            candidate = candidates.popleft()
            if candidate not in matched_project_indexes:
                project_index = candidate
                break
        if project_index is None:
            continue

        matched_template_indexes.add(template_index)
        matched_project_indexes.add(project_index)
        matched.append(
            {
                "match_type": "same_path_rule_and_message_line_shifted",
                "template_hotspot": hotspot,
                "project_hotspot": project_hotspots[project_index],
            }
        )

    # Optional restriction motivated by manual validation: the generic
    # rule/message/category/probability signature can match unrelated code.
    # Keep unmatched hotspots as not-persistent/new when this stage is disabled.
    if not allow_moved_matches:
        removed = [
            hotspot for index, hotspot in enumerate(template_hotspots)
            if index not in matched_template_indexes
        ]
        new = [
            hotspot for index, hotspot in enumerate(project_hotspots)
            if index not in matched_project_indexes
        ]
        return {"matched": matched, "removed": removed, "new": new}

    # Pass 3: moved/renamed file. Require uniqueness among remaining records on
    # both sides to avoid falsely pairing repeated hotspot occurrences.
    remaining_template_by_signature: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    remaining_project_by_signature: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)

    for template_index, hotspot in enumerate(template_hotspots):
        if template_index not in matched_template_indexes:
            remaining_template_by_signature[moved_hotspot_signature(hotspot)].append(
                template_index
            )

    for project_index, hotspot in enumerate(project_hotspots):
        if project_index not in matched_project_indexes:
            remaining_project_by_signature[moved_hotspot_signature(hotspot)].append(
                project_index
            )

    for signature, template_indexes in remaining_template_by_signature.items():
        project_indexes = remaining_project_by_signature.get(signature, [])
        if len(template_indexes) != 1 or len(project_indexes) != 1:
            continue

        template_index = template_indexes[0]
        project_index = project_indexes[0]
        if (
            template_index in matched_template_indexes
            or project_index in matched_project_indexes
        ):
            continue

        template_hotspot = template_hotspots[template_index]
        project_hotspot = project_hotspots[project_index]

        # Only call this a move/rename if the normalized paths differ.
        template_path = normalize_component(str(template_hotspot.get("component") or ""))
        project_path = normalize_component(str(project_hotspot.get("component") or ""))
        if template_path == project_path:
            # Same file but unusual location/message representation; leave it
            # unmatched rather than making a weaker same-file claim.
            continue

        matched_template_indexes.add(template_index)
        matched_project_indexes.add(project_index)
        matched.append(
            {
                "match_type": "unique_rule_and_message_in_moved_or_renamed_file",
                "template_hotspot": template_hotspot,
                "project_hotspot": project_hotspot,
            }
        )

    removed = [
        hotspot
        for index, hotspot in enumerate(template_hotspots)
        if index not in matched_template_indexes
    ]
    new = [
        hotspot
        for index, hotspot in enumerate(project_hotspots)
        if index not in matched_project_indexes
    ]

    return {"matched": matched, "removed": removed, "new": new}


# ---------------------------------------------------------------------------
# Breakdowns and pair-level summary
# ---------------------------------------------------------------------------

def count_by(
    findings: Iterable[Finding],
    key_function: Callable[[Finding], Iterable[str]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for finding in findings:
        labels = list(key_function(finding)) or ["UNKNOWN"]
        for label in labels:
            counts[str(label or "UNKNOWN")] += 1
    return dict(sorted(counts.items()))


def build_breakdown(
    template_findings: list[Finding],
    project_findings: list[Finding],
    matched: list[dict[str, Any]],
    removed: list[Finding],
    new_findings: list[Finding],
    template_key: str,
    key_function: Callable[[Finding], Iterable[str]],
    noun: str,
) -> dict[str, dict[str, Any]]:
    template_counts = count_by(template_findings, key_function)
    project_counts = count_by(project_findings, key_function)
    persistent_counts = count_by(
        (item[template_key] for item in matched), key_function
    )
    removed_counts = count_by(removed, key_function)
    new_counts = count_by(new_findings, key_function)

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
            f"template_{noun}_count": template_count,
            f"project_{noun}_count": project_counts.get(label, 0),
            f"persistent_{noun}_count": persistent_count,
            f"no_longer_present_{noun}_count": removed_counts.get(label, 0),
            f"new_{noun}_count": new_counts.get(label, 0),
            "persistence_rate": persistence_rate,
            "persistence_percentage": percent(persistence_rate),
        }

    return result


def build_pair_summary(
    pair: dict[str, Any],
    template_issue_file: Path,
    project_issue_file: Path,
    template_hotspot_file: Path | None,
    project_hotspot_file: Path | None,
    template_issues: list[Finding],
    project_issues: list[Finding],
    issue_result: ComparisonResult,
    template_hotspots: list[Finding],
    project_hotspots: list[Finding],
    hotspot_result: ComparisonResult,
) -> dict[str, Any]:
    issue_matched = issue_result["matched"]
    issue_removed = issue_result["removed"]
    issue_new = issue_result["new"]

    hotspot_matched = hotspot_result["matched"]
    hotspot_removed = hotspot_result["removed"]
    hotspot_new = hotspot_result["new"]

    template_issue_count = len(template_issues)
    project_issue_count = len(project_issues)
    persistent_issue_count = len(issue_matched)
    removed_issue_count = len(issue_removed)
    new_issue_count = len(issue_new)

    template_hotspot_count = len(template_hotspots)
    project_hotspot_count = len(project_hotspots)
    persistent_hotspot_count = len(hotspot_matched)
    removed_hotspot_count = len(hotspot_removed)
    new_hotspot_count = len(hotspot_new)

    issue_match_counts = Counter(str(item["match_type"]) for item in issue_matched)
    hotspot_match_counts = Counter(str(item["match_type"]) for item in hotspot_matched)

    issue_persistence = rate(persistent_issue_count, template_issue_count)
    issue_removal = rate(removed_issue_count, template_issue_count)
    issue_new_share = rate(new_issue_count, project_issue_count)

    hotspot_persistence = rate(persistent_hotspot_count, template_hotspot_count)
    hotspot_removal = rate(removed_hotspot_count, template_hotspot_count)
    hotspot_new_share = rate(new_hotspot_count, project_hotspot_count)

    return {
        "pair_id": pair["pair_id"],
        "source_csv_row": pair["source_csv_row"],
        "template_id": pair["template_slug"],
        "project_id": pair["project_slug"],
        "template_repository": pair["template_slug"],
        "generated_repository": pair["project_slug"],
        "template_repository_issue_file": str(template_issue_file),
        "generated_repository_issue_file": str(project_issue_file),
        "template_repository_hotspot_file": str(template_hotspot_file or ""),
        "generated_repository_hotspot_file": str(project_hotspot_file or ""),
        "original_link": pair.get("original_link", ""),
        "evolution_band": pair.get("evolution_band", ""),
        "sampling_method": pair.get("sampling_method", ""),
        "selection_probability": pair.get("selection_probability"),
        "sampling_weight": pair.get("sampling_weight"),

        # Existing issue metrics retained for backward compatibility.
        "template_issue_count": template_issue_count,
        "project_issue_count": project_issue_count,
        "persistent_issue_count": persistent_issue_count,
        "no_longer_present_issue_count": removed_issue_count,
        "new_issue_count": new_issue_count,
        "same_path_and_hash_match_count": issue_match_counts.get("same_path_and_hash", 0),
        "moved_or_renamed_match_count": issue_match_counts.get(
            "same_hash_in_moved_or_renamed_file", 0
        ),
        "fallback_without_hash_match_count": issue_match_counts.get(
            "fallback_without_hash", 0
        ),
        "persistence_rate": issue_persistence,
        "persistence_percentage": percent(issue_persistence),
        "removal_rate": issue_removal,
        "removal_percentage": percent(issue_removal),
        "new_issue_share_of_project_issues": issue_new_share,
        "new_issue_percentage_of_project_issues": percent(issue_new_share),

        # Explicit aliases make it clear these refer to ordinary issues.
        "issue_persistence_rate": issue_persistence,
        "issue_persistence_percentage": percent(issue_persistence),
        "issue_removal_rate": issue_removal,
        "issue_removal_percentage": percent(issue_removal),

        # Hotspot metrics are deliberately separate.
        "template_hotspot_count": template_hotspot_count,
        "project_hotspot_count": project_hotspot_count,
        "persistent_hotspot_count": persistent_hotspot_count,
        "no_longer_present_hotspot_count": removed_hotspot_count,
        "new_hotspot_count": new_hotspot_count,
        "same_path_message_and_location_hotspot_match_count": hotspot_match_counts.get(
            "same_path_message_and_location", 0
        ),
        "same_path_line_shifted_hotspot_match_count": hotspot_match_counts.get(
            "same_path_rule_and_message_line_shifted", 0
        ),
        "moved_or_renamed_hotspot_match_count": hotspot_match_counts.get(
            "unique_rule_and_message_in_moved_or_renamed_file", 0
        ),
        "hotspot_persistence_rate": hotspot_persistence,
        "hotspot_persistence_percentage": percent(hotspot_persistence),
        "hotspot_removal_rate": hotspot_removal,
        "hotspot_removal_percentage": percent(hotspot_removal),
        "new_hotspot_share_of_project_hotspots": hotspot_new_share,
        "new_hotspot_percentage_of_project_hotspots": percent(hotspot_new_share),

        # Combined counts are called findings, not issues.
        "template_finding_count": template_issue_count + template_hotspot_count,
        "project_finding_count": project_issue_count + project_hotspot_count,
        "persistent_finding_count": persistent_issue_count + persistent_hotspot_count,
        "no_longer_present_finding_count": removed_issue_count + removed_hotspot_count,
        "new_finding_count": new_issue_count + new_hotspot_count,

        "breakdown_by_type": build_breakdown(
            template_issues,
            project_issues,
            issue_matched,
            issue_removed,
            issue_new,
            "template_issue",
            lambda issue: [str(issue.get("type") or "UNKNOWN")],
            "issue",
        ),
        "breakdown_by_severity": build_breakdown(
            template_issues,
            project_issues,
            issue_matched,
            issue_removed,
            issue_new,
            "template_issue",
            lambda issue: [str(issue.get("severity") or "UNKNOWN")],
            "issue",
        ),
        "breakdown_by_software_quality": build_breakdown(
            template_issues,
            project_issues,
            issue_matched,
            issue_removed,
            issue_new,
            "template_issue",
            lambda issue: software_qualities(issue) or ["UNKNOWN"],
            "issue",
        ),
        "hotspot_breakdown_by_security_category": build_breakdown(
            template_hotspots,
            project_hotspots,
            hotspot_matched,
            hotspot_removed,
            hotspot_new,
            "template_hotspot",
            lambda hotspot: [str(hotspot.get("securityCategory") or "UNKNOWN")],
            "hotspot",
        ),
        "hotspot_breakdown_by_vulnerability_probability": build_breakdown(
            template_hotspots,
            project_hotspots,
            hotspot_matched,
            hotspot_removed,
            hotspot_new,
            "template_hotspot",
            lambda hotspot: [
                str(hotspot.get("vulnerabilityProbability") or "UNKNOWN")
            ],
            "hotspot",
        ),
        "hotspot_breakdown_by_status": build_breakdown(
            template_hotspots,
            project_hotspots,
            hotspot_matched,
            hotspot_removed,
            hotspot_new,
            "template_hotspot",
            lambda hotspot: [str(hotspot.get("status") or "UNKNOWN")],
            "hotspot",
        ),
    }


# ---------------------------------------------------------------------------
# Per-pair CSVs
# ---------------------------------------------------------------------------

def save_pair_issue_comparison_csv(
    path: Path,
    pair_summary: dict[str, Any],
    result: ComparisonResult,
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
                "template_path": normalize_component(str(template.get("component") or "")),
                "template_line": template.get("line"),
                "project_path": normalize_component(str(project.get("component") or "")),
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
                "template_path": normalize_component(str(template.get("component") or "")),
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
                "project_path": normalize_component(str(project.get("component") or "")),
                "project_line": project.get("line"),
                "hash": project.get("hash"),
                "message": project.get("message"),
            }
        )

    save_csv(path, rows, fields)


def save_pair_hotspot_comparison_csv(
    path: Path,
    pair_summary: dict[str, Any],
    result: ComparisonResult,
) -> None:
    fields = [
        "pair_id",
        "template_repository",
        "generated_repository",
        "classification",
        "match_type",
        "rule_key",
        "security_category",
        "vulnerability_probability",
        "template_hotspot_key",
        "project_hotspot_key",
        "template_path",
        "template_line",
        "project_path",
        "project_line",
        "message",
        "template_status",
        "project_status",
    ]

    rows: list[dict[str, Any]] = []
    common = {
        "pair_id": pair_summary["pair_id"],
        "template_repository": pair_summary["template_repository"],
        "generated_repository": pair_summary["generated_repository"],
    }

    for match in result["matched"]:
        template = match["template_hotspot"]
        project = match["project_hotspot"]
        rows.append(
            {
                **common,
                "classification": "persistent_in_generated_repository",
                "match_type": match["match_type"],
                "rule_key": hotspot_rule_key(template),
                "security_category": template.get("securityCategory"),
                "vulnerability_probability": template.get("vulnerabilityProbability"),
                "template_hotspot_key": template.get("key"),
                "project_hotspot_key": project.get("key"),
                "template_path": normalize_component(str(template.get("component") or "")),
                "template_line": hotspot_start_line(template),
                "project_path": normalize_component(str(project.get("component") or "")),
                "project_line": hotspot_start_line(project),
                "message": template.get("message"),
                "template_status": template.get("status"),
                "project_status": project.get("status"),
            }
        )

    for template in result["removed"]:
        rows.append(
            {
                **common,
                "classification": "not_persistent_in_generated_repository",
                "match_type": "",
                "rule_key": hotspot_rule_key(template),
                "security_category": template.get("securityCategory"),
                "vulnerability_probability": template.get("vulnerabilityProbability"),
                "template_hotspot_key": template.get("key"),
                "project_hotspot_key": "",
                "template_path": normalize_component(str(template.get("component") or "")),
                "template_line": hotspot_start_line(template),
                "project_path": "",
                "project_line": "",
                "message": template.get("message"),
                "template_status": template.get("status"),
                "project_status": "",
            }
        )

    for project in result["new"]:
        rows.append(
            {
                **common,
                "classification": "new_in_generated_repository",
                "match_type": "",
                "rule_key": hotspot_rule_key(project),
                "security_category": project.get("securityCategory"),
                "vulnerability_probability": project.get("vulnerabilityProbability"),
                "template_hotspot_key": "",
                "project_hotspot_key": project.get("key"),
                "template_path": "",
                "template_line": "",
                "project_path": normalize_component(str(project.get("component") or "")),
                "project_line": hotspot_start_line(project),
                "message": project.get("message"),
                "template_status": "",
                "project_status": project.get("status"),
            }
        )

    save_csv(path, rows, fields)


# ---------------------------------------------------------------------------
# Aggregate summaries
# ---------------------------------------------------------------------------

def summary_csv_row(summary: dict[str, Any]) -> dict[str, Any]:
    scalar_fields = [
        "pair_id",
        "source_csv_row",
        "template_repository",
        "generated_repository",
        "template_repository_issue_file",
        "generated_repository_issue_file",
        "template_repository_hotspot_file",
        "generated_repository_hotspot_file",
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
        "template_hotspot_count",
        "project_hotspot_count",
        "persistent_hotspot_count",
        "no_longer_present_hotspot_count",
        "new_hotspot_count",
        "same_path_message_and_location_hotspot_match_count",
        "same_path_line_shifted_hotspot_match_count",
        "moved_or_renamed_hotspot_match_count",
        "hotspot_persistence_rate",
        "hotspot_persistence_percentage",
        "hotspot_removal_rate",
        "hotspot_removal_percentage",
        "new_hotspot_share_of_project_hotspots",
        "new_hotspot_percentage_of_project_hotspots",
        "template_finding_count",
        "project_finding_count",
        "persistent_finding_count",
        "no_longer_present_finding_count",
        "new_finding_count",
    ]
    return {field: summary.get(field) for field in scalar_fields}


def aggregate_family(
    summaries: list[dict[str, Any]],
    template_count_key: str,
    project_count_key: str,
    persistent_count_key: str,
    removed_count_key: str,
    new_count_key: str,
    persistence_rate_key: str,
    removal_rate_key: str,
) -> dict[str, Any]:
    total_template = sum(item[template_count_key] for item in summaries)
    total_project = sum(item[project_count_key] for item in summaries)
    total_persistent = sum(item[persistent_count_key] for item in summaries)
    total_removed = sum(item[removed_count_key] for item in summaries)
    total_new = sum(item[new_count_key] for item in summaries)

    persistence_values = [
        item[persistence_rate_key]
        for item in summaries
        if item[persistence_rate_key] is not None
    ]
    removal_values = [
        item[removal_rate_key]
        for item in summaries
        if item[removal_rate_key] is not None
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
        weighted_persistent += summary[persistent_count_key] * weight
        weighted_template += summary[template_count_key] * weight
        if summary[persistence_rate_key] is not None:
            weighted_pair_rates.append((summary[persistence_rate_key], weight))

    weighted_micro = rate(weighted_persistent, weighted_template)
    weighted_macro = None
    if weighted_pair_rates:
        weighted_macro = sum(value * weight for value, weight in weighted_pair_rates) / sum(
            weight for _, weight in weighted_pair_rates
        )

    return {
        "template_observation_count": total_template,
        "project_observation_count": total_project,
        "persistent_observation_count": total_persistent,
        "no_longer_present_observation_count": total_removed,
        "new_observation_count": total_new,
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
    }


def build_aggregate_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    issue_aggregate = aggregate_family(
        summaries,
        "template_issue_count",
        "project_issue_count",
        "persistent_issue_count",
        "no_longer_present_issue_count",
        "new_issue_count",
        "persistence_rate",
        "removal_rate",
    )
    hotspot_aggregate = aggregate_family(
        summaries,
        "template_hotspot_count",
        "project_hotspot_count",
        "persistent_hotspot_count",
        "no_longer_present_hotspot_count",
        "new_hotspot_count",
        "hotspot_persistence_rate",
        "hotspot_removal_rate",
    )

    # Preserve original top-level issue aggregate keys for compatibility.
    return {
        "analyzed_pair_count": len(summaries),
        "unique_template_count": len({item["template_repository"] for item in summaries}),
        "unique_generated_repository_count": len(
            {item["generated_repository"] for item in summaries}
        ),
        "template_issue_observation_count": issue_aggregate["template_observation_count"],
        "project_issue_observation_count": issue_aggregate["project_observation_count"],
        "persistent_issue_observation_count": issue_aggregate["persistent_observation_count"],
        "no_longer_present_issue_observation_count": issue_aggregate[
            "no_longer_present_observation_count"
        ],
        "new_issue_observation_count": issue_aggregate["new_observation_count"],
        "micro_persistence_rate": issue_aggregate["micro_persistence_rate"],
        "micro_persistence_percentage": issue_aggregate["micro_persistence_percentage"],
        "macro_average_pair_persistence_rate": issue_aggregate[
            "macro_average_pair_persistence_rate"
        ],
        "macro_average_pair_persistence_percentage": issue_aggregate[
            "macro_average_pair_persistence_percentage"
        ],
        "sampling_weighted_micro_persistence_rate": issue_aggregate[
            "sampling_weighted_micro_persistence_rate"
        ],
        "sampling_weighted_micro_persistence_percentage": issue_aggregate[
            "sampling_weighted_micro_persistence_percentage"
        ],
        "sampling_weighted_macro_pair_persistence_rate": issue_aggregate[
            "sampling_weighted_macro_pair_persistence_rate"
        ],
        "sampling_weighted_macro_pair_persistence_percentage": issue_aggregate[
            "sampling_weighted_macro_pair_persistence_percentage"
        ],
        "micro_removal_rate": issue_aggregate["micro_removal_rate"],
        "micro_removal_percentage": issue_aggregate["micro_removal_percentage"],
        "macro_average_pair_removal_rate": issue_aggregate[
            "macro_average_pair_removal_rate"
        ],
        "macro_average_pair_removal_percentage": issue_aggregate[
            "macro_average_pair_removal_percentage"
        ],
        "issues": issue_aggregate,
        "security_hotspots": hotspot_aggregate,
        "interpretation_note": (
            "Issues and Security Hotspots are aggregated separately. Counts are "
            "observations across template/generated-repository pairs, so one "
            "template finding is counted once per generated repository compared "
            "with that template."
        ),
    }


def build_template_summary(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[summary["template_repository"]].append(summary)

    rows: list[dict[str, Any]] = []
    for template, items in sorted(grouped.items()):
        issue_total = sum(item["template_issue_count"] for item in items)
        issue_persistent = sum(item["persistent_issue_count"] for item in items)
        issue_removed = sum(item["no_longer_present_issue_count"] for item in items)
        issue_rates = [
            item["persistence_rate"]
            for item in items
            if item["persistence_rate"] is not None
        ]

        hotspot_total = sum(item["template_hotspot_count"] for item in items)
        hotspot_persistent = sum(item["persistent_hotspot_count"] for item in items)
        hotspot_removed = sum(item["no_longer_present_hotspot_count"] for item in items)
        hotspot_rates = [
            item["hotspot_persistence_rate"]
            for item in items
            if item["hotspot_persistence_rate"] is not None
        ]

        rows.append(
            {
                "template_repository": template,
                "generated_repository_count": len(items),
                "template_issue_count_per_repository": items[0]["template_issue_count"],
                "template_issue_observation_count": issue_total,
                "persistent_issue_observation_count": issue_persistent,
                "not_persistent_issue_observation_count": issue_removed,
                # Original issue-summary names retained for compatibility.
                "micro_persistence_rate": rate(issue_persistent, issue_total),
                "micro_persistence_percentage": percent(
                    rate(issue_persistent, issue_total)
                ),
                "macro_average_pair_persistence_rate": (
                    mean(issue_rates) if issue_rates else None
                ),
                "macro_average_pair_persistence_percentage": percent(
                    mean(issue_rates) if issue_rates else None
                ),
                # Explicit issue aliases.
                "micro_issue_persistence_rate": rate(issue_persistent, issue_total),
                "micro_issue_persistence_percentage": percent(
                    rate(issue_persistent, issue_total)
                ),
                "macro_average_pair_issue_persistence_rate": (
                    mean(issue_rates) if issue_rates else None
                ),
                "macro_average_pair_issue_persistence_percentage": percent(
                    mean(issue_rates) if issue_rates else None
                ),
                "template_hotspot_count_per_repository": items[0]["template_hotspot_count"],
                "template_hotspot_observation_count": hotspot_total,
                "persistent_hotspot_observation_count": hotspot_persistent,
                "not_persistent_hotspot_observation_count": hotspot_removed,
                "micro_hotspot_persistence_rate": rate(
                    hotspot_persistent, hotspot_total
                ),
                "micro_hotspot_persistence_percentage": percent(
                    rate(hotspot_persistent, hotspot_total)
                ),
                "macro_average_pair_hotspot_persistence_rate": (
                    mean(hotspot_rates) if hotspot_rates else None
                ),
                "macro_average_pair_hotspot_persistence_percentage": percent(
                    mean(hotspot_rates) if hotspot_rates else None
                ),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Per-template finding persistence across descendants
# ---------------------------------------------------------------------------

def add_template_issue_observations(
    storage: dict[tuple[str, str], dict[str, Any]],
    pair: dict[str, Any],
    result: ComparisonResult,
) -> None:
    weight = parse_float(pair.get("sampling_weight"))
    if weight is None or weight <= 0:
        weight = 1.0

    def record_for(issue: Finding) -> dict[str, Any]:
        key = (pair["template_slug"], template_issue_identity(issue))
        return storage.setdefault(
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

    for match in result["matched"]:
        issue = match["template_issue"]
        record = record_for(issue)
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
        record = record_for(issue)
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
        row["weighted_persistence_rate_across_generated_repositories"] = weighted_persistence
        row["weighted_persistence_percentage_across_generated_repositories"] = percent(
            weighted_persistence
        )
        rows.append(row)

    return rows


def add_template_hotspot_observations(
    storage: dict[tuple[str, str], dict[str, Any]],
    pair: dict[str, Any],
    result: ComparisonResult,
) -> None:
    weight = parse_float(pair.get("sampling_weight"))
    if weight is None or weight <= 0:
        weight = 1.0

    def record_for(hotspot: Finding) -> dict[str, Any]:
        key = (pair["template_slug"], template_hotspot_identity(hotspot))
        return storage.setdefault(
            key,
            {
                "template_repository": pair["template_slug"],
                "template_hotspot_identity": key[1],
                **hotspot_summary(hotspot),
                "compared_project_count": 0,
                "persistent_project_count": 0,
                "not_persistent_project_count": 0,
                "weighted_compared_project_count": 0.0,
                "weighted_persistent_project_count": 0.0,
                "same_path_message_and_location_match_count": 0,
                "same_path_line_shifted_match_count": 0,
                "moved_or_renamed_match_count": 0,
            },
        )

    for match in result["matched"]:
        hotspot = match["template_hotspot"]
        record = record_for(hotspot)
        record["compared_project_count"] += 1
        record["persistent_project_count"] += 1
        record["weighted_compared_project_count"] += weight
        record["weighted_persistent_project_count"] += weight

        if match["match_type"] == "same_path_message_and_location":
            record["same_path_message_and_location_match_count"] += 1
        elif match["match_type"] == "same_path_rule_and_message_line_shifted":
            record["same_path_line_shifted_match_count"] += 1
        elif match["match_type"] == "unique_rule_and_message_in_moved_or_renamed_file":
            record["moved_or_renamed_match_count"] += 1

    for hotspot in result["removed"]:
        record = record_for(hotspot)
        record["compared_project_count"] += 1
        record["not_persistent_project_count"] += 1
        record["weighted_compared_project_count"] += weight


def finalize_template_hotspot_observations(
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
        row["text_range"] = json.dumps(record.get("text_range"), ensure_ascii=False)
        row["persistence_rate_across_generated_repositories"] = persistence
        row["persistence_percentage_across_generated_repositories"] = percent(
            persistence
        )
        row["weighted_persistence_rate_across_generated_repositories"] = weighted_persistence
        row["weighted_persistence_percentage_across_generated_repositories"] = percent(
            weighted_persistence
        )
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# CSV pair input
# ---------------------------------------------------------------------------

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
        if (
            "target_template_repository" not in reader.fieldnames
            and "template_slug" not in reader.fieldnames
        ):
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
                    "selection_probability": parse_float(row.get("selection_probability")),
                    "sampling_weight": parse_float(row.get("sampling_weight")),
                }
            )

    if not pairs:
        raise ValueError("No valid template/generated-repository pairs were found.")

    return pairs, errors


# ---------------------------------------------------------------------------
# CLI and execution
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare template/generated-repository SonarQube issues and Security "
            "Hotspots using pairs derived directly from a sampling CSV."
        )
    )
    parser.add_argument("sample_csv", type=Path)
    parser.add_argument("template_issue_directory", type=Path)
    parser.add_argument("project_issue_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--template-hotspot-directory",
        type=Path,
        default=None,
        help=(
            "Directory containing template Security Hotspot JSON files. "
            "Defaults to template_issue_directory."
        ),
    )
    parser.add_argument(
        "--project-hotspot-directory",
        type=Path,
        default=None,
        help=(
            "Directory containing generated-project Security Hotspot JSON files. "
            "Defaults to project_issue_directory."
        ),
    )
    parser.add_argument(
        "--allow-missing-hotspots",
        action="store_true",
        help=(
            "Treat a missing hotspot file as an empty hotspot set. Use only when "
            "you know missing files mean that the repository had zero hotspots."
        ),
    )
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
    parser.add_argument(
        "--artifact-policy",
        choices=ARTIFACT_POLICIES,
        default="none",
        help=(
            "Exclude generated findings from BOTH sides before comparing: "
            "none (original behavior), gradle-report (narrow), "
            "java-build-outputs (build/, target/, .gradle/). "
            "Use an entirely new output directory for each policy."
        ),
    )
    parser.add_argument(
        "--disable-moved-hotspot-match",
        action="store_true",
        help=(
            "Skip cross-file hotspot matches based only on rule, generic message, "
            "security category, and vulnerability probability. Retain exact "
            "location and same-file line-shift hotspot strategies."
        ),
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help=(
            "Explicitly allow replacing files in a nonempty output directory. "
            "By default existing results are protected."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    print(f"compare_sonar_issue_pairs_from_sample.py version: {SCRIPT_VERSION}")

    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero.")

    sample_csv = args.sample_csv.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()
    if (
        output_directory.exists()
        and any(output_directory.iterdir())
        and not args.overwrite_output
    ):
        raise SystemExit(
            f"Refusing to overwrite nonempty output directory: {output_directory}. "
            "Use a new directory for each exclusion policy (recommended), or "
            "pass --overwrite-output explicitly."
        )
    pairs_directory = output_directory / "pairs"
    output_directory.mkdir(parents=True, exist_ok=True)
    pairs_directory.mkdir(parents=True, exist_ok=True)

    template_hotspot_directory = (
        args.template_hotspot_directory
        if args.template_hotspot_directory is not None
        else args.template_issue_directory
    )
    project_hotspot_directory = (
        args.project_hotspot_directory
        if args.project_hotspot_directory is not None
        else args.project_issue_directory
    )

    try:
        pairs, input_errors = read_sample_pairs(sample_csv, args.eligible_only)
        template_issue_index = FindingFileIndex(args.template_issue_directory, "issues")
        project_issue_index = FindingFileIndex(args.project_issue_directory, "issues")
        template_hotspot_index = FindingFileIndex(template_hotspot_directory, "hotspots")
        project_hotspot_index = FindingFileIndex(project_hotspot_directory, "hotspots")

        print(
            f"Indexed {len(template_issue_index.all_files)} template issue files, "
            f"{len(project_issue_index.all_files)} generated-project issue files, "
            f"{len(template_hotspot_index.all_files)} template hotspot files, and "
            f"{len(project_hotspot_index.all_files)} generated-project hotspot files."
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if args.limit is not None:
        pairs = pairs[: args.limit]

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = list(input_errors)
    resolved_pair_rows: list[dict[str, Any]] = []
    finding_cache: dict[tuple[Path, str], list[Finding]] = {}
    excluded_by_file: dict[tuple[Path, str], list[dict[str, str]]] = {}
    original_counts: dict[tuple[Path, str], int] = {}
    exclusion_pair_rows: list[dict[str, Any]] = []
    template_issue_observations: dict[tuple[str, str], dict[str, Any]] = {}
    template_hotspot_observations: dict[tuple[str, str], dict[str, Any]] = {}

    for index, pair in enumerate(pairs, start=1):
        print(
            f"[{index}/{len(pairs)}] Template={pair['template_slug']} | "
            f"Generated={pair['project_slug']}"
        )

        try:
            template_issue_file = template_issue_index.resolve(pair["template_slug"])
            project_issue_file = project_issue_index.resolve(pair["project_slug"])

            template_hotspot_file: Path | None
            project_hotspot_file: Path | None

            try:
                template_hotspot_file = template_hotspot_index.resolve(
                    pair["template_slug"]
                )
            except ValueError:
                if not args.allow_missing_hotspots:
                    raise
                template_hotspot_file = None

            try:
                project_hotspot_file = project_hotspot_index.resolve(
                    pair["project_slug"]
                )
            except ValueError:
                if not args.allow_missing_hotspots:
                    raise
                project_hotspot_file = None

            print(f"  template issue file:    {template_issue_file}")
            print(f"  generated issue file:   {project_issue_file}")
            print(
                f"  template hotspot file:  "
                f"{template_hotspot_file or '[missing -> empty]'}"
            )
            print(
                f"  generated hotspot file: "
                f"{project_hotspot_file or '[missing -> empty]'}"
            )

            resolved_pair_rows.append(
                {
                    "source_csv_row": pair["source_csv_row"],
                    "pair_id": pair["pair_id"],
                    "template_repository": pair["template_slug"],
                    "generated_repository": pair["project_slug"],
                    "template_repository_issue_file": str(template_issue_file),
                    "generated_repository_issue_file": str(project_issue_file),
                    "template_repository_hotspot_file": str(template_hotspot_file or ""),
                    "generated_repository_hotspot_file": str(project_hotspot_file or ""),
                    "status": "resolved",
                }
            )

            def findings_for(path: Path | None, kind: str) -> list[Finding]:
                if path is None:
                    return []
                cache_key = (path, kind)
                if cache_key not in finding_cache:
                    raw_findings = load_findings(path, kind)
                    original_counts[cache_key] = len(raw_findings)
                    filtered, excluded = exclude_artifact_findings(
                        raw_findings, args.artifact_policy
                    )
                    finding_cache[cache_key] = filtered
                    excluded_by_file[cache_key] = excluded
                return finding_cache[cache_key]

            template_issues = findings_for(template_issue_file, "issues")
            project_issues = findings_for(project_issue_file, "issues")
            template_hotspots = findings_for(template_hotspot_file, "hotspots")
            project_hotspots = findings_for(project_hotspot_file, "hotspots")

            audit_row: dict[str, Any] = {
                "pair_id": pair["pair_id"],
                "template_repository": pair["template_slug"],
                "generated_repository": pair["project_slug"],
                "artifact_policy": args.artifact_policy,
            }
            for side, kind, path in (
                ("template", "issues", template_issue_file),
                ("generated", "issues", project_issue_file),
                ("template", "hotspots", template_hotspot_file),
                ("generated", "hotspots", project_hotspot_file),
            ):
                cache_key = (path, kind) if path is not None else None
                raw_count = original_counts.get(cache_key, 0)
                excluded_count = len(excluded_by_file.get(cache_key, []))
                audit_row[f"{side}_{kind}_original"] = raw_count
                audit_row[f"{side}_{kind}_excluded"] = excluded_count
                audit_row[f"{side}_{kind}_retained"] = raw_count - excluded_count
            exclusion_pair_rows.append(audit_row)

            issue_result = compare_issues(template_issues, project_issues)
            hotspot_result = compare_hotspots(
                template_hotspots,
                project_hotspots,
                allow_moved_matches=not args.disable_moved_hotspot_match,
            )

            summary = build_pair_summary(
                pair,
                template_issue_file,
                project_issue_file,
                template_hotspot_file,
                project_hotspot_file,
                template_issues,
                project_issues,
                issue_result,
                template_hotspots,
                project_hotspots,
                hotspot_result,
            )

            pair_directory = pairs_directory / pair["pair_id"]
            pair_directory.mkdir(parents=True, exist_ok=True)

            save_json(pair_directory / "summary.json", summary)
            save_pair_issue_comparison_csv(
                pair_directory / "issue_comparison.csv", summary, issue_result
            )
            save_pair_hotspot_comparison_csv(
                pair_directory / "hotspot_comparison.csv", summary, hotspot_result
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
                    for item in issue_result["matched"]
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
                    for issue in issue_result["removed"]
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
                    for issue in issue_result["new"]
                ],
            )

            save_json(
                pair_directory / "matched_hotspots.json",
                [
                    {
                        "pair_id": pair["pair_id"],
                        "template_repository": pair["template_slug"],
                        "generated_repository": pair["project_slug"],
                        "match_type": item["match_type"],
                        "template_hotspot": hotspot_summary(item["template_hotspot"]),
                        "project_hotspot": hotspot_summary(item["project_hotspot"]),
                    }
                    for item in hotspot_result["matched"]
                ],
            )
            save_json(
                pair_directory / "not_persistent_template_hotspots.json",
                [
                    {
                        "pair_id": pair["pair_id"],
                        "template_repository": pair["template_slug"],
                        "generated_repository": pair["project_slug"],
                        "template_hotspot": hotspot_summary(hotspot),
                    }
                    for hotspot in hotspot_result["removed"]
                ],
            )
            save_json(
                pair_directory / "new_generated_repository_hotspots.json",
                [
                    {
                        "pair_id": pair["pair_id"],
                        "template_repository": pair["template_slug"],
                        "generated_repository": pair["project_slug"],
                        "project_hotspot": hotspot_summary(hotspot),
                    }
                    for hotspot in hotspot_result["new"]
                ],
            )

            add_template_issue_observations(
                template_issue_observations, pair, issue_result
            )
            add_template_hotspot_observations(
                template_hotspot_observations, pair, hotspot_result
            )
            summaries.append(summary)

            print(
                "  issues:   "
                f"persistent={summary['persistent_issue_count']}, "
                f"not_persistent={summary['no_longer_present_issue_count']}, "
                f"new={summary['new_issue_count']}, "
                f"persistence={summary['persistence_percentage']}%"
            )
            print(
                "  hotspots: "
                f"persistent={summary['persistent_hotspot_count']}, "
                f"not_persistent={summary['no_longer_present_hotspot_count']}, "
                f"new={summary['new_hotspot_count']}, "
                f"persistence={summary['hotspot_persistence_percentage']}%"
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
                    "template_repository_hotspot_file": "",
                    "generated_repository_hotspot_file": "",
                    "status": f"error: {error}",
                }
            )
            print(f"  ERROR: {error}")
            if args.fail_fast:
                raise SystemExit(str(error)) from error

    save_csv(output_directory / "resolved_pairs.csv", resolved_pair_rows)
    save_json(output_directory / "artifact_filter_config.json", {
        "script_version": SCRIPT_VERSION,
        "artifact_policy": args.artifact_policy,
        "hotspot_matching_policy": (
            "same_file_only" if args.disable_moved_hotspot_match else "all_stages"
        ),
        "applied_to": "both source-template and generated-repository findings before matching",
        "input_files_modified": False,
        "successful_pairs": len(summaries),
        "scope_warning": (
            "This is an analysis of the existing SonarQube exports. "
            "It is not a fresh SonarQube scan with source exclusions."
        ),
    })
    save_csv(output_directory / "artifact_exclusions_by_pair.csv", exclusion_pair_rows)
    excluded_file_rows: list[dict[str, Any]] = []
    for (file_path, kind), excluded in sorted(
        excluded_by_file.items(), key=lambda pair: (str(pair[0][0]), pair[0][1])
    ):
        counts = Counter(
            (row["normalized_path"], row["reason"], row["rule"])
            for row in excluded
        )
        for (normalized_path, reason, rule), count in sorted(counts.items()):
            excluded_file_rows.append({
                "finding_file": str(file_path),
                "kind": kind,
                "normalized_path": normalized_path,
                "exclusion_reason": reason,
                "rule": rule,
                "excluded_findings": count,
            })
    # Keep a CSV header even when no exclusions were needed.
    exclusion_file_fields = [
        "finding_file", "kind", "normalized_path", "exclusion_reason",
        "rule", "excluded_findings",
    ]
    if excluded_file_rows:
        save_csv(output_directory / "artifact_exclusions_by_input_file.csv", excluded_file_rows)
    else:
        (output_directory / "artifact_exclusions_by_input_file.csv").write_text(
            ",".join(exclusion_file_fields) + "\n", encoding="utf-8"
        )
    total_distinct_file_findings_excluded = sum(
        len(items) for items in excluded_by_file.values()
    )
    print(
        f"Artifact policy {args.artifact_policy}: excluded "
        f"{total_distinct_file_findings_excluded:,} findings across "
        f"{len(excluded_by_file):,} unique input files."
    )

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

        hotspot_rows = finalize_template_hotspot_observations(
            template_hotspot_observations
        )
        save_csv(output_directory / "template_hotspot_persistence.csv", hotspot_rows)
        save_json(output_directory / "template_hotspot_persistence.json", hotspot_rows)

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
