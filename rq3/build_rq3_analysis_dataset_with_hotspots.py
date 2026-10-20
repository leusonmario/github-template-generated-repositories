#!/usr/bin/env python3
"""
build_rq3_analysis_dataset_with_hotspots.py

Build a regression-ready dataset for:

RQ3: Which template and generated-repository characteristics are associated
     with the persistence of SonarQube findings?

Unit of analysis
----------------
One row = one source-template / generated-repository pair from RQ2.

Outcome families
----------------
Quality Issues:
    persistent_issue_count
    no_longer_present_issue_count
    template_issue_observations
    issue_persistence_rate_recomputed

Security Hotspots:
    persistent_hotspot_count
    no_longer_present_hotspot_count
    template_hotspot_observations
    hotspot_persistence_rate_recomputed

Issues and Security Hotspots are retained as separate outcome families.
They are NOT merged into a single persistence denominator.

Predictors assembled here
-------------------------
- language
- source-template ownership
- template domain
- generated-repository commit count
- RQ1 evolution level
- template family size (# unique generated repos associated with a template)
- template issue count
- template Security Hotspot count
- generated-repository creation date and age from
  rq1/reports/generated_GHRep_from_templates_<LANGUAGE>_with_commits.csv,
  using data_creation/created_at as the creation timestamp and data_update
  as the observation timestamp when available

Sampling
--------
Exact sampling metadata can be supplied via repeated --sampling-file
LANGUAGE selected_sample.csv arguments. These files are the authoritative
source for selection_probability and sampling_weight. If they are omitted,
pair_summary.csv values are used and missing/invalid weights fall back to 1.0.

Outputs
-------
rq3_analysis_dataset.csv
rq3_analysis_dataset_long.csv
rq3_analysis_dataset_complete_cases.csv                  # legacy alias: issues
rq3_issue_analysis_dataset_complete_cases.csv
rq3_hotspot_analysis_dataset_complete_cases.csv
rq3_variable_summary.csv
rq3_data_quality.csv
rq3_domain_distribution.csv
rq3_correlation_candidates.csv
rq3_unmatched_domain_templates.csv

Recommended initial models
--------------------------
Issue persistence:
    cbind(persistent_issue_count, no_longer_present_issue_count)
      ~ log_commit_count + owner_type + log_template_family_size
        + language + domain

Security-Hotspot persistence:
    cbind(persistent_hotspot_count, no_longer_present_hotspot_count)
      ~ log_commit_count + owner_type + log_template_family_size
        + language + domain

Because Security Hotspots are typically much sparser than quality issues,
model adequacy and category sparsity should be checked before fitting the same
specification to both outcome families.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


LANGUAGE_ORDER = ["C#", "Java", "Python", "TypeScript", "JavaScript"]

LANGUAGE_CONFIG = {
    "C#": {
        "aliases": ["C#", "CSharp", "Csharp", "csharp"],
        "rq2_dirs": ["CSharp", "Csharp", "C#", "csharp"],
        "template_tokens": ["C#", "CSharp", "Csharp", "csharp"],
        "domain_tokens": ["C#", "CSharp", "Csharp", "csharp"],
        "generated_report_file": "generated_GHRep_from_templates_Csharp_with_commits.csv",
    },
    "Java": {
        "aliases": ["Java", "java"],
        "rq2_dirs": ["Java", "java"],
        "template_tokens": ["Java", "java"],
        "domain_tokens": ["Java", "java"],
        "generated_report_file": "generated_GHRep_from_templates_Java_with_commits.csv",
    },
    "Python": {
        "aliases": ["Python", "python"],
        "rq2_dirs": ["Python", "python"],
        "template_tokens": ["Python", "python"],
        "domain_tokens": ["Python", "python"],
        "generated_report_file": "generated_GHRep_from_templates_Python_with_commits.csv",
    },
    "TypeScript": {
        "aliases": ["TypeScript", "Typescript", "typescript"],
        "rq2_dirs": ["TypeScript", "Typescript", "typescript"],
        "template_tokens": ["TypeScript", "Typescript", "typescript"],
        "domain_tokens": ["TypeScript", "Typescript", "typescript"],
        "generated_report_file": "generated_GHRep_from_templates_TypeScript_with_commits.csv",
    },
    "JavaScript": {
        "aliases": ["JavaScript", "Javascript", "javascript"],
        "rq2_dirs": ["JavaScript", "Javascript", "javascript"],
        "template_tokens": ["JavaScript", "Javascript", "javascript"],
        "domain_tokens": ["JavaScript", "Javascript", "javascript"],
        "generated_report_file": "generated_GHRep_from_templates_JavaScript_with_commits.csv",
    },
}

PAIR_TEMPLATE_CANDIDATES = [
    "template_repository", "template_repo", "source_template", "template"
]
PAIR_GENERATED_CANDIDATES = [
    "generated_repository", "generated_repo", "repository", "repo"
]

DOMAIN_REPO_CANDIDATES = [
    "template_repository", "template_repo", "repository", "repo",
    "full_name", "repository_full_name", "original_link_repository", "repo_url"
]

GENERATED_METADATA_REPO_CANDIDATES = [
    "generated_repository", "generated_repo", "generated_repo_full_name",
    "generated_repository_full_name", "generated_repo_url",
    "generated_repository_url", "repository", "repo", "full_name",
    "name_with_owner", "repository_full_name", "repo_url",
    "repository_url", "repo_link", "repository_link", "html_url",
    "original_link", "original_link_repository", "url",
]

CREATED_AT_CANDIDATES = [
    "created_at", "created", "creation_date", "created_date", "data_creation"
]

OBSERVATION_DATE_CANDIDATES = [
    "data_update", "collected_at", "collection_date", "snapshot_date",
    "data_collected_at"
]

# RQ2 pair_summary aliases.
ISSUE_REQUIRED = {
    "persistent_issue_count": ["persistent_issue_count"],
    "no_longer_present_issue_count": [
        "no_longer_present_issue_count",
        "not_persistent_issue_count",
        "removed_issue_count",
    ],
    "template_issue_count": ["template_issue_count"],
}

ISSUE_OPTIONAL = {
    "new_issue_count": ["new_issue_count"],
    "issue_persistence_rate": [
        "issue_persistence_rate",
        "persistence_rate",
    ],
    "issue_persistence_percentage": [
        "issue_persistence_percentage",
        "persistence_percentage",
    ],
}

HOTSPOT_REQUIRED = {
    "persistent_hotspot_count": [
        "persistent_hotspot_count",
        "persistent_security_hotspot_count",
    ],
    "no_longer_present_hotspot_count": [
        "no_longer_present_hotspot_count",
        "not_persistent_hotspot_count",
        "removed_hotspot_count",
        "no_longer_present_security_hotspot_count",
    ],
    "template_hotspot_count": [
        "template_hotspot_count",
        "template_security_hotspot_count",
    ],
}

HOTSPOT_OPTIONAL = {
    "new_hotspot_count": [
        "new_hotspot_count",
        "new_security_hotspot_count",
    ],
    "hotspot_persistence_rate": [
        "hotspot_persistence_rate",
        "security_hotspot_persistence_rate",
    ],
    "hotspot_persistence_percentage": [
        "hotspot_persistence_percentage",
        "security_hotspot_persistence_percentage",
    ],
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build an RQ3 analysis dataset with separate Quality-Issue and "
            "Security-Hotspot persistence outcomes."
        )
    )
    p.add_argument("--rq1-detailed", required=True, type=Path)
    p.add_argument("--rq2-root", required=True, type=Path)
    p.add_argument("--templates-dir", required=True, type=Path)

    p.add_argument("--domains-dir", type=Path, default=None)
    p.add_argument("--domain-pattern", default="*{domain_token}*.csv")
    p.add_argument(
        "--domain-file",
        action="append",
        nargs=2,
        default=[],
        metavar=("LANGUAGE", "PATH"),
    )

    p.add_argument(
        "--generated-metadata",
        action="append",
        nargs=2,
        default=[],
        metavar=("LANGUAGE", "PATH"),
    )

    p.add_argument(
        "--sampling-file",
        action="append",
        nargs=2,
        default=[],
        metavar=("LANGUAGE", "PATH"),
        help=(
            "Exact selected_sample.csv produced by define_persistence_sample.py "
            "for one language. May be repeated. When provided, its "
            "selection_probability, sampling_weight, sampling_method, "
            "evolution_band, family/stratum population counts, and seed are "
            "merged into the RQ3 pair dataset by template/repository identity."
        ),
    )

    p.add_argument(
        "--generated-reports-dir",
        type=Path,
        default=Path("rq1/reports"),
        help=(
            "Directory containing RQ1 generated-repository report CSVs. "
            "The script uses the configured per-language filename."
        ),
    )

    p.add_argument(
        "--generated-reports-pattern",
        default=None,
        help=(
            "Optional override glob pattern for generated-repository metadata."
        ),
    )

    p.add_argument(
        "--data-location",
        type=Path,
        default=None,
        help=(
            "Optional fallback directory containing generated-repository "
            "metadata CSVs."
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rq3_results"),
    )

    p.add_argument(
        "--rq2-pair-pattern",
        default="{language_dir}/issue_persistence_results/pair_summary.csv",
    )

    p.add_argument(
        "--templates-pattern",
        default="all_templates_{template_token}.csv",
    )

    p.add_argument(
        "--age-reference-date",
        default=None,
        help=(
            "Optional fixed YYYY-MM-DD observation date for repository age. "
            "If omitted, per-row data_update/collection dates are used when "
            "available. Today's date is never used implicitly."
        ),
    )

    p.add_argument(
        "--allow-missing-hotspot-columns",
        action="store_true",
        help=(
            "Compatibility option for old pair_summary.csv files. If hotspot "
            "columns are absent, create zero hotspot counts instead of failing. "
            "Do not use this for the final RQ3 dataset if hotspot extraction "
            "has been completed."
        ),
    )

    return p.parse_args()


def canonicalize_language(value):
    value = str(value).strip()
    for canonical, config in LANGUAGE_CONFIG.items():
        if any(value.casefold() == x.casefold() for x in config["aliases"]):
            return canonical
    raise ValueError(f"Unsupported language: {value}")


def build_override_map(items):
    result = {}
    for language, path in items:
        result[canonicalize_language(language)] = (
            Path(path).expanduser().resolve()
        )
    return result


def find_column(df, candidates, required=True):
    normalized = {
        str(c).strip().casefold(): c
        for c in df.columns
    }
    for candidate in candidates:
        if candidate.casefold() in normalized:
            return normalized[candidate.casefold()]
    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Available: {list(df.columns)}"
        )
    return None


def normalize_repo_reference(value):
    if pd.isna(value):
        return None

    text = (
        str(value)
        .strip()
        .replace("\\", "/")
        .lower()
        .rstrip("/")
    )

    if not text:
        return None

    for prefix in [
        "https://github.com/",
        "http://github.com/",
        "https://www.github.com/",
        "http://www.github.com/",
        "git@github.com:",
        "github.com/",
        "www.github.com/",
    ]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    if text.endswith(".git"):
        text = text[:-4]

    # Tolerate accidental terminal punctuation in exported repository URLs/slugs
    # (for example, "owner/repo."). This keeps exact pair matching strict while
    # normalizing obvious CSV/text-export artifacts.
    text = text.rstrip(" .,")

    parts = [x for x in text.split("/") if x]

    if len(parts) < 2:
        return None

    return f"{parts[0]}/{parts[1]}"


def normalize_repo_from_owner_name(owner, name):
    if pd.isna(owner) or pd.isna(name):
        return None
    return normalize_repo_reference(f"{owner}/{name}")


def normalize_owner(value):
    if pd.isna(value):
        return None

    x = str(value).strip().casefold()

    if x == "user":
        return "User"

    if x in {"organization", "organisation", "org"}:
        return "Organization"

    return None


def clean_domain(value):
    if pd.isna(value):
        return None

    text = re.sub(r"\s+", " ", str(value).strip())
    return text if text else None


def extract_domain_from_llm_output(value):
    """
    Robustly extract `category` from a domain-classification LLM output.
    """
    if pd.isna(value):
        return None

    raw = str(value)

    if not raw.strip():
        return None

    text = (
        raw.replace("\u00a0", " ")
        .replace("\u2007", " ")
        .replace("\u202f", " ")
        .replace("\ufeff", "")
        .strip()
    )

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    candidates = [text]

    deescaped = text.replace('""', '"')
    if deescaped != text:
        candidates.append(deescaped)

    expanded = []
    for candidate in candidates:
        expanded.append(candidate)

        start = candidate.find("{")
        end = candidate.rfind("}")

        if start != -1 and end != -1 and end > start:
            expanded.append(candidate[start:end + 1])

    seen = set()
    candidates = []

    for candidate in expanded:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                category = clean_domain(parsed.get("category"))
                if category:
                    return category
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                category = clean_domain(parsed.get("category"))
                if category:
                    return category
        except (ValueError, SyntaxError, TypeError):
            pass

    match = re.search(
        r'(?is)["\']?\s*category\s*["\']?\s*:\s*'
        r'["\']\s*([^"\']+?)\s*["\']',
        deescaped,
    )
    if match:
        return clean_domain(match.group(1))

    match = re.search(
        r'(?is)["\']?\s*category\s*["\']?\s*:\s*'
        r'([^,\}\r\n]+)',
        deescaped,
    )
    if match:
        return clean_domain(
            match.group(1).strip().strip('"\'')
        )

    return None


def resolve_pair_file(root, pattern, language):
    tried = []

    for language_dir in LANGUAGE_CONFIG[language]["rq2_dirs"]:
        path = root / pattern.format(
            language=language,
            language_dir=language_dir,
        )
        tried.append(path)

        if path.exists():
            return path

    raise FileNotFoundError(
        f"No pair_summary.csv for {language}. Tried:\n"
        + "\n".join(map(str, tried))
    )


def resolve_template_file(root, pattern, language):
    tried = []

    for token in LANGUAGE_CONFIG[language]["template_tokens"]:
        path = root / pattern.format(
            language=language,
            template_token=token,
        )
        tried.append(path)

        if path.exists():
            return path

    raise FileNotFoundError(
        f"No all_templates file for {language}. Tried:\n"
        + "\n".join(map(str, tried))
    )


def resolve_domain_file(
    language,
    explicit,
    domains_dir,
    pattern,
):
    if language in explicit:
        return explicit[language]

    if domains_dir is None:
        return None

    candidates = []

    for token in LANGUAGE_CONFIG[language]["domain_tokens"]:
        candidates.extend(
            sorted(
                domains_dir.glob(
                    pattern.format(
                        language=language,
                        domain_token=token,
                    )
                )
            )
        )

    candidates = [
        p
        for p in dict.fromkeys(candidates)
        if (
            p.is_file()
            and "all_templates" not in p.name.lower()
            and "pair_summary" not in p.name.lower()
        )
    ]

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        print(
            f"[WARN] Multiple domain files for {language}; "
            "use --domain-file explicitly:"
        )
        for p in candidates:
            print("      ", p)

    return None


def build_template_owner_map(template_file):
    df = pd.read_csv(template_file)

    owner_col = find_column(df, ["owner_type"])
    username_col = find_column(
        df,
        ["username"],
        required=False,
    )
    project_col = find_column(
        df,
        ["project_name"],
        required=False,
    )
    repo_col = find_column(
        df,
        DOMAIN_REPO_CANDIDATES,
        required=False,
    )

    if username_col is not None and project_col is not None:
        repos = df.apply(
            lambda r: normalize_repo_from_owner_name(
                r[username_col],
                r[project_col],
            ),
            axis=1,
        )
    elif repo_col is not None:
        repos = df[repo_col].apply(normalize_repo_reference)
    else:
        raise ValueError(
            f"Cannot construct template identity from {template_file}"
        )

    out = pd.DataFrame(
        {
            "template_repository": repos,
            "owner_type": df[owner_col].apply(normalize_owner),
        }
    ).dropna(
        subset=["template_repository"]
    )

    conflicts = (
        out.dropna(subset=["owner_type"])
        .groupby("template_repository")["owner_type"]
        .nunique()
    )

    if (conflicts > 1).any():
        raise ValueError(
            f"Conflicting owner values in {template_file}: "
            f"{conflicts[conflicts > 1].index.tolist()[:10]}"
        )

    return out.drop_duplicates("template_repository")


def build_domain_map(domain_file):
    df = pd.read_csv(
        domain_file,
        low_memory=False,
    )

    username_col = find_column(
        df,
        ["username"],
        required=False,
    )
    project_col = find_column(
        df,
        ["project_name"],
        required=False,
    )
    repo_col = find_column(
        df,
        DOMAIN_REPO_CANDIDATES,
        required=False,
    )

    if username_col is not None and project_col is not None:
        repos = df.apply(
            lambda r: normalize_repo_from_owner_name(
                r[username_col],
                r[project_col],
            ),
            axis=1,
        )
    elif repo_col is not None:
        repos = df[repo_col].apply(normalize_repo_reference)
    else:
        raise ValueError(
            f"Cannot construct repository identity from {domain_file}. "
            "Expected repo/full-name or username/project_name."
        )

    if "llm_output" in df.columns:
        domains = df["llm_output"].apply(
            extract_domain_from_llm_output
        )
        domain_source = "llm_output.category"
    else:
        domain_col = find_column(
            df,
            [
                "domain",
                "category",
                "classification",
                "label",
                "predicted_domain",
                "final_domain",
                "final_category",
            ],
        )
        domains = df[domain_col].apply(clean_domain)
        domain_source = domain_col

    out = pd.DataFrame(
        {
            "template_repository": repos,
            "domain": domains,
        }
    )

    invalid_repo = int(out["template_repository"].isna().sum())

    out = out.dropna(
        subset=["template_repository"]
    ).copy()

    conflicts = (
        out.dropna(subset=["domain"])
        .groupby("template_repository")["domain"]
        .nunique()
    )

    if (conflicts > 1).any():
        raise ValueError(
            f"Conflicting domain labels in {domain_file}: "
            f"{conflicts[conflicts > 1].index.tolist()[:10]}"
        )

    missing_domain = int(out["domain"].isna().sum())
    parsed_domain = int(out["domain"].notna().sum())

    print(
        f"[INFO] Domain file {Path(domain_file).name}: "
        f"rows={len(df):,}, valid_repo={len(out):,}, "
        f"parsed_domain={parsed_domain:,}, "
        f"missing_domain={missing_domain:,}, "
        f"invalid_repo={invalid_repo:,}, "
        f"source={domain_source}"
    )

    return out.drop_duplicates("template_repository")


def load_rq1_detailed(path):
    df = pd.read_csv(path)

    for col in [
        "language",
        "generated_repo",
        "template_repo",
        "commit_count",
    ]:
        if col not in df.columns:
            raise ValueError(
                f"{path} is missing '{col}'. "
                f"Available: {list(df.columns)}"
            )

    work = df.copy()

    work["language"] = work["language"].apply(
        canonicalize_language
    )

    work["generated_repository"] = work[
        "generated_repo"
    ].apply(
        normalize_repo_reference
    )

    work["template_repository"] = work[
        "template_repo"
    ].apply(
        normalize_repo_reference
    )

    work["commit_count"] = pd.to_numeric(
        work["commit_count"],
        errors="coerce",
    )

    if "evolution_level" not in work.columns:

        def classify(n):
            if pd.isna(n):
                return None
            if n == 1:
                return "Not evolved"
            if 2 <= n <= 6:
                return "Slightly evolved"
            if 7 <= n <= 20:
                return "Moderately evolved"
            if n > 20:
                return "Substantially evolved"
            return None

        work["evolution_level"] = work[
            "commit_count"
        ].apply(classify)

    family = (
        work.dropna(
            subset=[
                "template_repository",
                "generated_repository",
            ]
        )
        .groupby(
            [
                "language",
                "template_repository",
            ]
        )["generated_repository"]
        .nunique()
        .rename("template_family_size")
        .reset_index()
    )

    pair_vars = (
        work[
            [
                "language",
                "template_repository",
                "generated_repository",
                "commit_count",
                "evolution_level",
            ]
        ]
        .dropna(
            subset=[
                "template_repository",
                "generated_repository",
            ]
        )
        .sort_values(
            "commit_count",
            na_position="last",
        )
        .drop_duplicates(
            [
                "language",
                "template_repository",
                "generated_repository",
            ],
            keep="last",
        )
        .merge(
            family,
            on=[
                "language",
                "template_repository",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    return pair_vars


def _parse_reference_date(reference_date):
    if reference_date is None:
        return None

    return pd.to_datetime(
        reference_date,
        errors="raise",
        utc=True,
    )


def _metadata_identity_columns(df):
    repo_col = find_column(
        df,
        GENERATED_METADATA_REPO_CANDIDATES,
        required=False,
    )

    username_col = find_column(
        df,
        ["username", "owner", "repo_owner"],
        required=False,
    )

    project_col = find_column(
        df,
        ["project_name", "name", "repo_name"],
        required=False,
    )

    return repo_col, username_col, project_col


def _build_repository_series(df):
    (
        repo_col,
        username_col,
        project_col,
    ) = _metadata_identity_columns(df)

    if repo_col is not None:
        return df[repo_col].apply(
            normalize_repo_reference
        )

    if (
        username_col is not None
        and project_col is not None
    ):
        return df.apply(
            lambda r: normalize_repo_from_owner_name(
                r[username_col],
                r[project_col],
            ),
            axis=1,
        )

    raise ValueError(
        "Could not identify a repository column. Expected a URL/full-name "
        "column or owner/name columns."
    )


def _build_age_frame(
    df,
    language,
    source_path,
    reference_date=None,
    target_repositories=None,
):
    created_col = find_column(
        df,
        CREATED_AT_CANDIDATES,
    )

    observation_col = find_column(
        df,
        OBSERVATION_DATE_CANDIDATES,
        required=False,
    )

    repos = _build_repository_series(df)

    created = pd.to_datetime(
        df[created_col],
        errors="coerce",
        utc=True,
    )

    fixed_ref = _parse_reference_date(
        reference_date
    )

    if fixed_ref is not None:
        observed = pd.Series(
            fixed_ref,
            index=df.index,
            dtype="datetime64[ns, UTC]",
        )
        reference_source = "fixed_age_reference_date"

    elif observation_col is not None:
        observed = pd.to_datetime(
            df[observation_col],
            errors="coerce",
            utc=True,
        )
        reference_source = observation_col

    else:
        observed = pd.Series(
            pd.NaT,
            index=df.index,
            dtype="datetime64[ns, UTC]",
        )
        reference_source = "missing"

    age_days = (
        observed - created
    ).dt.total_seconds() / 86400.0

    age_days = age_days.where(
        age_days >= 0
    )

    out = pd.DataFrame(
        {
            "language": language,
            "generated_repository": repos,
            "generated_created_at": created,
            "age_reference_at": observed,
            "age_reference_source": reference_source,
            "repository_age_days": age_days,
            "repository_age_years": age_days / 365.25,
            "generated_metadata_source": str(source_path),
        }
    )

    out = out.dropna(
        subset=["generated_repository"]
    ).copy()

    if target_repositories is not None:
        out = out[
            out["generated_repository"].isin(
                target_repositories
            )
        ].copy()

    return out


def _deduplicate_generated_metadata(metadata):
    if metadata.empty:
        return metadata

    work = metadata.copy()

    work["_created_ok"] = (
        work["generated_created_at"]
        .notna()
        .astype(int)
    )

    work["_age_ok"] = (
        work["repository_age_years"]
        .notna()
        .astype(int)
    )

    work = work.sort_values(
        [
            "generated_repository",
            "_age_ok",
            "_created_ok",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    conflicts = (
        work.dropna(
            subset=["generated_created_at"]
        )
        .groupby(
            "generated_repository"
        )["generated_created_at"]
        .nunique()
    )

    conflicts = conflicts[
        conflicts > 1
    ]

    if not conflicts.empty:
        print(
            f"[WARN] {len(conflicts):,} generated repositories have "
            "conflicting created_at values across metadata files. "
            "The first highest-quality record is retained."
        )

    work = work.drop_duplicates(
        [
            "language",
            "generated_repository",
        ],
        keep="first",
    )

    return work.drop(
        columns=[
            "_created_ok",
            "_age_ok",
        ]
    )


def load_generated_metadata(
    path,
    language,
    reference_date=None,
    target_repositories=None,
):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    out = _build_age_frame(
        df,
        language=language,
        source_path=path,
        reference_date=reference_date,
        target_repositories=target_repositories,
    )

    if (
        not out.empty
        and reference_date is None
        and (out["age_reference_source"] == "missing").all()
    ):
        print(
            f"[WARN] {path}: creation date was found, but no "
            "data_update/collection-date column was found. "
            "generated_created_at is populated, but repository age "
            "remains missing. Supply --age-reference-date if needed."
        )

    return _deduplicate_generated_metadata(
        out
    )


def _csv_has_generated_metadata_columns(path):
    try:
        header = pd.read_csv(
            path,
            nrows=0,
        )
    except Exception:
        return False

    created_col = find_column(
        header,
        CREATED_AT_CANDIDATES,
        required=False,
    )

    if created_col is None:
        return False

    (
        repo_col,
        username_col,
        project_col,
    ) = _metadata_identity_columns(header)

    return (
        repo_col is not None
        or (
            username_col is not None
            and project_col is not None
        )
    )


def resolve_generated_report_file(
    generated_reports_dir,
    language,
    override_pattern=None,
):
    root = (
        Path(generated_reports_dir)
        .expanduser()
        .resolve()
    )

    if not root.exists():
        raise FileNotFoundError(root)

    if override_pattern:
        matches = sorted(
            root.rglob(
                override_pattern.format(
                    language=language,
                )
            )
        )

        if len(matches) == 1:
            return matches[0]

        if len(matches) == 0:
            raise FileNotFoundError(
                f"No file matching override pattern "
                f"'{override_pattern}' under {root}"
            )

        raise ValueError(
            f"Override pattern '{override_pattern}' matched "
            f"multiple files under {root}: {matches}"
        )

    filename = LANGUAGE_CONFIG[
        language
    ]["generated_report_file"]

    direct = root / filename

    if direct.exists():
        return direct

    matches = sorted(
        root.rglob(filename)
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Generated-report file for {language} was not found.\n"
            f"Expected: {filename}\n"
            f"Under: {root}"
        )

    raise ValueError(
        f"Multiple generated-report files found for {language}: "
        f"{matches}"
    )


def discover_generated_metadata(
    data_location,
    language,
    target_repositories,
    reference_date=None,
    file_pattern="*.csv",
    source_label="metadata",
):
    data_location = (
        Path(data_location)
        .expanduser()
        .resolve()
    )

    if not data_location.exists():
        raise FileNotFoundError(
            data_location
        )

    csv_files = sorted(
        data_location.rglob(
            file_pattern
        )
    )

    candidates = [
        path
        for path in csv_files
        if _csv_has_generated_metadata_columns(path)
    ]

    empty_columns = [
        "language",
        "generated_repository",
        "generated_created_at",
        "age_reference_at",
        "age_reference_source",
        "repository_age_days",
        "repository_age_years",
        "generated_metadata_source",
    ]

    if not candidates:
        print(
            f"[WARN] No {source_label} CSV matching '{file_pattern}' "
            f"with repository identity + creation date was found under "
            f"{data_location}."
        )
        return pd.DataFrame(
            columns=empty_columns
        )

    frames = []
    matched_files = []

    for path in candidates:
        try:
            df = pd.read_csv(
                path,
                low_memory=False,
            )

            frame = _build_age_frame(
                df,
                language=language,
                source_path=path,
                reference_date=reference_date,
                target_repositories=target_repositories,
            )

            if not frame.empty:
                frames.append(frame)
                matched_files.append(path)

        except Exception as exc:
            print(
                f"[WARN] Skipping {source_label} candidate "
                f"{path}: {exc}"
            )

    if not frames:
        print(
            f"[WARN] No RQ2 generated repositories for {language} "
            f"were matched to {source_label} under {data_location}."
        )
        return pd.DataFrame(
            columns=empty_columns
        )

    out = _deduplicate_generated_metadata(
        pd.concat(
            frames,
            ignore_index=True,
        )
    )

    age_count = int(
        out["repository_age_years"]
        .notna()
        .sum()
    )

    created_count = int(
        out["generated_created_at"]
        .notna()
        .sum()
    )

    print(
        f"[INFO] {language}: {source_label} matched {len(out):,}/"
        f"{len(target_repositories):,} generated repositories from "
        f"{len(matched_files):,} CSV file(s); "
        f"created_at={created_count:,}, age={age_count:,}."
    )

    if (
        age_count == 0
        and reference_date is None
    ):
        print(
            "[WARN] Creation timestamps were matched, but repository "
            "ages could not be calculated because no observation/data_update "
            "date was found. Re-run with --age-reference-date YYYY-MM-DD "
            "if age is required."
        )

    return out


def _copy_numeric_alias(
    source,
    target_name,
    candidates,
    required,
    path,
):
    column = find_column(
        source,
        candidates,
        required=False,
    )

    if column is None:
        if required:
            raise ValueError(
                f"{path} is missing a required field for '{target_name}'. "
                f"Accepted aliases: {candidates}. "
                f"Available: {list(source.columns)}"
            )
        return pd.Series(
            np.nan,
            index=source.index,
            dtype=float,
        )

    return pd.to_numeric(
        source[column],
        errors="coerce",
    )



def load_sampling_file(path, language):
    """
    Load the exact selected_sample.csv written by define_persistence_sample.py.

    Expected identifiers:
      repo_slug
      template_slug

    Expected sampling fields include:
      selection_probability
      sampling_weight
      sampling_method
      evolution_band
      family_population_n
      family_sample_n
      stratum_population_n
      stratum_sample_n
      sampling_seed
    """
    df = pd.read_csv(path, low_memory=False)

    repo_col = find_column(
        df,
        ["repo_slug", "generated_repository", "generated_repo", "original_link"],
        required=False,
    )
    template_col = find_column(
        df,
        ["template_slug", "template_repository", "template_repo",
         "target_template_repository"],
        required=False,
    )

    if repo_col is None or template_col is None:
        raise ValueError(
            f"{path} does not contain usable sampling identifiers. "
            f"Available columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "language": language,
        "template_repository": df[template_col].apply(normalize_repo_reference),
        "generated_repository": df[repo_col].apply(normalize_repo_reference),
    })

    fields = [
        "selection_probability",
        "sampling_weight",
        "sampling_method",
        "evolution_band",
        "family_population_n",
        "family_sample_n",
        "stratum_population_n",
        "stratum_sample_n",
        "sampling_seed",
    ]

    for field in fields:
        if field in df.columns:
            out[f"sample_{field}"] = df[field]

    for field in [
        "sample_selection_probability",
        "sample_sampling_weight",
        "sample_family_population_n",
        "sample_family_sample_n",
        "sample_stratum_population_n",
        "sample_stratum_sample_n",
        "sample_sampling_seed",
    ]:
        if field in out.columns:
            out[field] = pd.to_numeric(out[field], errors="coerce")

    required_sampling = [
        "sample_selection_probability",
        "sample_sampling_weight",
    ]
    missing = [c for c in required_sampling if c not in out.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required sampling field(s): {missing}. "
            "Use the selected_sample.csv generated by define_persistence_sample.py."
        )

    bad = out[
        out["sample_selection_probability"].isna()
        | (out["sample_selection_probability"] <= 0)
        | (out["sample_selection_probability"] > 1)
        | out["sample_sampling_weight"].isna()
        | (out["sample_sampling_weight"] <= 0)
    ]
    if not bad.empty:
        raise ValueError(
            f"{path} contains {len(bad):,} invalid sampling rows."
        )

    dup = out.duplicated(
        ["language", "template_repository", "generated_repository"],
        keep=False,
    )
    if dup.any():
        raise ValueError(
            f"{path} contains duplicate repository pairs after normalization: "
            f"{int(dup.sum()):,} rows."
        )

    return out.dropna(
        subset=["template_repository", "generated_repository"]
    )


def merge_exact_sampling_metadata(data, sampling, language, sampling_path):
    """
    Merge exact RQ2 sampling metadata. Exact selected_sample.csv values take
    precedence over blank/defaulted values from pair_summary.csv.
    """
    work = data.merge(
        sampling,
        on=["language", "template_repository", "generated_repository"],
        how="left",
        validate="one_to_one",
    )

    matched = int(work["sample_sampling_weight"].notna().sum())
    total = len(work)

    if matched != total:
        missing = work.loc[
            work["sample_sampling_weight"].isna(),
            ["template_repository", "generated_repository"],
        ].head(10).to_dict("records")
        raise ValueError(
            f"{language}: exact sampling file matched {matched:,}/{total:,} "
            f"RQ2 pairs. The RQ3 dataset should use the same selected sample. "
            f"Examples of unmatched pairs: {missing}"
        )

    work["sampling_weight"] = work["sample_sampling_weight"]
    work["selection_probability"] = work["sample_selection_probability"]
    work["sampling_weight_source"] = "selected_sample.csv"

    mapping = {
        "sample_sampling_method": "sampling_method",
        "sample_evolution_band": "sampling_evolution_band",
        "sample_family_population_n": "sampling_family_population_n",
        "sample_family_sample_n": "sampling_family_sample_n",
        "sample_stratum_population_n": "sampling_stratum_population_n",
        "sample_stratum_sample_n": "sampling_stratum_sample_n",
        "sample_sampling_seed": "sampling_seed",
    }

    for source, target in mapping.items():
        if source in work.columns:
            work[target] = work[source]

    # Verify w = 1 / p.
    expected_weight = 1.0 / work["selection_probability"]
    inconsistent = ~np.isclose(
        pd.to_numeric(work["sampling_weight"], errors="coerce"),
        pd.to_numeric(expected_weight, errors="coerce"),
        rtol=1e-9,
        atol=1e-12,
        equal_nan=False,
    )
    if inconsistent.any():
        raise ValueError(
            f"{language}: {int(inconsistent.sum()):,} rows in {sampling_path} "
            "do not satisfy sampling_weight = 1 / selection_probability."
        )

    print(
        f"[INFO] {language}: exact sampling metadata matched all {total:,} pairs; "
        f"non-unit weights="
        f"{int((~np.isclose(work['sampling_weight'], 1.0)).sum()):,}; "
        f"weight range={work['sampling_weight'].min():.3f}--"
        f"{work['sampling_weight'].max():.3f}."
    )

    drop_cols = [c for c in work.columns if c.startswith("sample_")]
    return work.drop(columns=drop_cols)

def load_pair_summary(
    path,
    language,
    allow_missing_hotspots=False,
):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    template_col = find_column(
        df,
        PAIR_TEMPLATE_CANDIDATES,
    )

    generated_col = find_column(
        df,
        PAIR_GENERATED_CANDIDATES,
    )

    out = pd.DataFrame(
        index=df.index
    )

    out["language"] = language

    out["template_repository"] = df[
        template_col
    ].apply(
        normalize_repo_reference
    )

    out["generated_repository"] = df[
        generated_col
    ].apply(
        normalize_repo_reference
    )

    # ---------------------------
    # Quality Issues
    # ---------------------------
    for target, aliases in ISSUE_REQUIRED.items():
        out[target] = _copy_numeric_alias(
            df,
            target,
            aliases,
            required=True,
            path=path,
        )

    for target, aliases in ISSUE_OPTIONAL.items():
        out[target] = _copy_numeric_alias(
            df,
            target,
            aliases,
            required=False,
            path=path,
        )

    out["template_issue_observations"] = (
        out["persistent_issue_count"]
        + out["no_longer_present_issue_count"]
    )

    out["issue_count_consistent"] = (
        out["template_issue_count"].isna()
        | np.isclose(
            out["template_issue_count"],
            out["template_issue_observations"],
            equal_nan=False,
        )
    )

    out["issue_persistence_rate_recomputed"] = np.where(
        out["template_issue_observations"] > 0,
        (
            out["persistent_issue_count"]
            / out["template_issue_observations"]
        ),
        np.nan,
    )

    # Backward-compatibility aliases.
    # NOTE: despite the historical name, these are all quality issues,
    # not vulnerabilities only.
    out["template_vulnerability_observations"] = (
        out["template_issue_observations"]
    )
    out["persistence_rate_recomputed"] = (
        out["issue_persistence_rate_recomputed"]
    )

    # ---------------------------
    # Security Hotspots
    # ---------------------------
    hotspot_columns_present = any(
        find_column(
            df,
            aliases,
            required=False,
        )
        is not None
        for aliases in HOTSPOT_REQUIRED.values()
    )

    if not hotspot_columns_present:
        if not allow_missing_hotspots:
            raise ValueError(
                f"{path} contains no Security-Hotspot persistence columns. "
                "Run the updated RQ2 comparison first, or use "
                "--allow-missing-hotspot-columns only for compatibility."
            )

        print(
            f"[WARN] {language}: hotspot columns are missing from {path}; "
            "zero counts are being created because "
            "--allow-missing-hotspot-columns was supplied."
        )

        for target in HOTSPOT_REQUIRED:
            out[target] = 0.0

        for target in HOTSPOT_OPTIONAL:
            out[target] = np.nan

    else:
        for target, aliases in HOTSPOT_REQUIRED.items():
            out[target] = _copy_numeric_alias(
                df,
                target,
                aliases,
                required=True,
                path=path,
            )

        for target, aliases in HOTSPOT_OPTIONAL.items():
            out[target] = _copy_numeric_alias(
                df,
                target,
                aliases,
                required=False,
                path=path,
            )

    out["template_hotspot_observations"] = (
        out["persistent_hotspot_count"]
        + out["no_longer_present_hotspot_count"]
    )

    out["hotspot_count_consistent"] = (
        out["template_hotspot_count"].isna()
        | np.isclose(
            out["template_hotspot_count"],
            out["template_hotspot_observations"],
            equal_nan=False,
        )
    )

    out["hotspot_persistence_rate_recomputed"] = np.where(
        out["template_hotspot_observations"] > 0,
        (
            out["persistent_hotspot_count"]
            / out["template_hotspot_observations"]
        ),
        np.nan,
    )

    # ---------------------------
    # Sampling metadata
    # ---------------------------
    sampling_weight_col = find_column(
        df,
        ["sampling_weight", "weight"],
        required=False,
    )

    if sampling_weight_col is None:
        out["sampling_weight"] = 1.0
        out["sampling_weight_source"] = "default_1"
    else:
        raw_weight = pd.to_numeric(
            df[sampling_weight_col],
            errors="coerce",
        )

        invalid = (
            raw_weight.isna()
            | (raw_weight <= 0)
        )

        out["sampling_weight"] = raw_weight.where(
            ~invalid,
            1.0,
        )

        out["sampling_weight_source"] = np.where(
            invalid,
            "default_1_invalid_or_missing",
            "pair_summary",
        )

    for col in [
        "selection_probability",
        "sampling_method",
        "evolution_band",
        "pair_id",
        "source_csv_row",
        "sampling_stratum",
        "sampling_strategy",
    ]:
        if col in df.columns:
            out[col] = df[col]

    # Preserve any explicit issue/hotspot persistence percentages if present.
    for source_col in [
        "persistence_rate",
        "persistence_percentage",
        "issue_persistence_rate",
        "issue_persistence_percentage",
        "hotspot_persistence_rate",
        "hotspot_persistence_percentage",
    ]:
        if source_col in df.columns and source_col not in out.columns:
            out[source_col] = df[source_col]

    duplicates = out.duplicated(
        [
            "language",
            "template_repository",
            "generated_repository",
        ],
        keep=False,
    )

    if duplicates.any():
        print(
            f"[WARN] {language}: {int(duplicates.sum()):,} rows belong to "
            "duplicate template/generated-repository pair keys. "
            "The first row per pair is retained."
        )

    return (
        out.dropna(
            subset=[
                "template_repository",
                "generated_repository",
            ]
        )
        .drop_duplicates(
            [
                "language",
                "template_repository",
                "generated_repository",
            ],
            keep="first",
        )
    )


def safe_log1p(series):
    x = pd.to_numeric(
        series,
        errors="coerce",
    )

    return np.log1p(
        x.where(x >= 0)
    )


def make_long_outcome_dataset(dataset):
    """
    Optional long representation:
      two rows per pair (Issue, Security Hotspot).

    This is useful for descriptive checks or a future model with finding_type
    interactions, but the primary RQ3 analyses should remain separate unless
    a pooled model is explicitly justified.
    """
    identity_predictors = [
        col
        for col in [
            "language",
            "template_repository",
            "generated_repository",
            "owner_type",
            "domain",
            "commit_count",
            "log_commit_count",
            "evolution_level",
            "template_family_size",
            "log_template_family_size",
            "generated_created_at",
            "age_reference_at",
            "age_reference_source",
            "repository_age_days",
            "repository_age_years",
            "sampling_weight",
            "sampling_weight_source",
            "selection_probability",
            "sampling_method",
            "sampling_strategy",
            "sampling_stratum",
            "evolution_band",
            "pair_id",
        ]
        if col in dataset.columns
    ]

    issue = dataset[
        identity_predictors
    ].copy()

    issue["finding_type"] = "Issue"
    issue["persistent_count"] = dataset[
        "persistent_issue_count"
    ].to_numpy()
    issue["not_persistent_count"] = dataset[
        "no_longer_present_issue_count"
    ].to_numpy()
    issue["template_finding_count"] = dataset[
        "template_issue_count"
    ].to_numpy()
    issue["template_finding_observations"] = dataset[
        "template_issue_observations"
    ].to_numpy()
    issue["new_finding_count"] = dataset[
        "new_issue_count"
    ].to_numpy()
    issue["persistence_rate_recomputed"] = dataset[
        "issue_persistence_rate_recomputed"
    ].to_numpy()
    issue["has_template_findings"] = (
        dataset["template_issue_observations"] > 0
    ).to_numpy()

    hotspot = dataset[
        identity_predictors
    ].copy()

    hotspot["finding_type"] = "Security Hotspot"
    hotspot["persistent_count"] = dataset[
        "persistent_hotspot_count"
    ].to_numpy()
    hotspot["not_persistent_count"] = dataset[
        "no_longer_present_hotspot_count"
    ].to_numpy()
    hotspot["template_finding_count"] = dataset[
        "template_hotspot_count"
    ].to_numpy()
    hotspot["template_finding_observations"] = dataset[
        "template_hotspot_observations"
    ].to_numpy()
    hotspot["new_finding_count"] = dataset[
        "new_hotspot_count"
    ].to_numpy()
    hotspot["persistence_rate_recomputed"] = dataset[
        "hotspot_persistence_rate_recomputed"
    ].to_numpy()
    hotspot["has_template_findings"] = (
        dataset["template_hotspot_observations"] > 0
    ).to_numpy()

    return pd.concat(
        [
            issue,
            hotspot,
        ],
        ignore_index=True,
    )


def main():
    args = parse_args()

    rq1_path = (
        args.rq1_detailed
        .expanduser()
        .resolve()
    )

    rq2_root = (
        args.rq2_root
        .expanduser()
        .resolve()
    )

    templates_dir = (
        args.templates_dir
        .expanduser()
        .resolve()
    )

    domains_dir = (
        args.domains_dir
        .expanduser()
        .resolve()
        if args.domains_dir
        else None
    )

    data_location = (
        args.data_location
        .expanduser()
        .resolve()
        if args.data_location
        else None
    )

    generated_reports_dir = (
        args.generated_reports_dir
        .expanduser()
        .resolve()
        if args.generated_reports_dir
        else None
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    explicit_domain_files = build_override_map(
        args.domain_file
    )

    generated_metadata_files = build_override_map(
        args.generated_metadata
    )

    sampling_files = build_override_map(
        args.sampling_file
    )

    rq1_pairs = load_rq1_detailed(
        rq1_path
    )

    all_rows = []
    quality_rows = []
    unmatched_domain_rows = []

    for language in LANGUAGE_ORDER:
        pair_file = resolve_pair_file(
            rq2_root,
            args.rq2_pair_pattern,
            language,
        )

        template_file = resolve_template_file(
            templates_dir,
            args.templates_pattern,
            language,
        )

        domain_file = resolve_domain_file(
            language,
            explicit_domain_files,
            domains_dir,
            args.domain_pattern,
        )

        pairs = load_pair_summary(
            pair_file,
            language,
            allow_missing_hotspots=args.allow_missing_hotspot_columns,
        )

        if language in sampling_files:
            exact_sampling = load_sampling_file(
                sampling_files[language],
                language,
            )
            pairs = merge_exact_sampling_metadata(
                pairs,
                exact_sampling,
                language,
                sampling_files[language],
            )
        else:
            print(
                f"[WARN] {language}: no --sampling-file supplied; "
                "sampling weights will rely on pair_summary.csv/defaults."
            )

        owners = build_template_owner_map(
            template_file
        )

        data = pairs.merge(
            owners,
            on="template_repository",
            how="left",
            validate="many_to_one",
        )

        rq1_language = rq1_pairs[
            rq1_pairs["language"] == language
        ].copy()

        data = data.merge(
            rq1_language[
                [
                    "template_repository",
                    "generated_repository",
                    "commit_count",
                    "evolution_level",
                    "template_family_size",
                ]
            ],
            on=[
                "template_repository",
                "generated_repository",
            ],
            how="left",
            validate="one_to_one",
        )

        if domain_file is not None:
            domains = build_domain_map(
                domain_file
            )

            data = data.merge(
                domains,
                on="template_repository",
                how="left",
                validate="many_to_one",
            )
        else:
            data["domain"] = pd.NA

        if domain_file is not None:
            missing_domain_pairs = data[
                data["domain"].isna()
            ].copy()

            if not missing_domain_pairs.empty:
                unmatched_summary = (
                    missing_domain_pairs
                    .groupby(
                        [
                            "language",
                            "template_repository",
                        ],
                        dropna=False,
                    )
                    .agg(
                        rq2_pairs=(
                            "generated_repository",
                            "nunique",
                        ),
                        pairs_with_template_issues=(
                            "template_issue_observations",
                            lambda s: int(
                                (s > 0).sum()
                            ),
                        ),
                        pairs_with_template_hotspots=(
                            "template_hotspot_observations",
                            lambda s: int(
                                (s > 0).sum()
                            ),
                        ),
                    )
                    .reset_index()
                )

                unmatched_summary[
                    "domain_file"
                ] = str(domain_file)

                unmatched_domain_rows.append(
                    unmatched_summary
                )

        target_generated_repositories = set(
            data["generated_repository"]
            .dropna()
            .unique()
        )

        # Metadata priority:
        # 1) explicit --generated-metadata
        # 2) rq1/reports per-language generated report
        # 3) optional --data-location fallback
        if language in generated_metadata_files:
            metadata = load_generated_metadata(
                generated_metadata_files[
                    language
                ],
                language,
                args.age_reference_date,
                target_repositories=target_generated_repositories,
            )

        elif (
            generated_reports_dir is not None
            and generated_reports_dir.exists()
        ):
            try:
                generated_report_file = resolve_generated_report_file(
                    generated_reports_dir,
                    language,
                    args.generated_reports_pattern,
                )

                metadata = load_generated_metadata(
                    generated_report_file,
                    language,
                    args.age_reference_date,
                    target_repositories=target_generated_repositories,
                )

                print(
                    f"[INFO] {language}: using generated-report file "
                    f"{generated_report_file.name}; matched "
                    f"{len(metadata):,}/"
                    f"{len(target_generated_repositories):,} "
                    "RQ2 generated repositories."
                )

            except (
                FileNotFoundError,
                ValueError,
            ) as exc:
                print(
                    f"[WARN] {exc}"
                )
                metadata = pd.DataFrame()

        elif data_location is not None:
            metadata = discover_generated_metadata(
                data_location,
                language,
                target_generated_repositories,
                args.age_reference_date,
                file_pattern="*.csv",
                source_label="data-location",
            )

        else:
            metadata = pd.DataFrame()

        if not metadata.empty:
            data = data.merge(
                metadata,
                on=[
                    "language",
                    "generated_repository",
                ],
                how="left",
                validate="one_to_one",
            )
        else:
            data["generated_created_at"] = pd.NaT
            data["age_reference_at"] = pd.NaT
            data["age_reference_source"] = pd.NA
            data["repository_age_days"] = np.nan
            data["repository_age_years"] = np.nan
            data["generated_metadata_source"] = pd.NA

        # Transformations / candidate predictors.
        data["log_commit_count"] = safe_log1p(
            data["commit_count"]
        )

        data[
            "log_template_family_size"
        ] = safe_log1p(
            data["template_family_size"]
        )

        data[
            "log_template_issue_count"
        ] = safe_log1p(
            data["template_issue_count"]
        )

        data[
            "log_template_hotspot_count"
        ] = safe_log1p(
            data["template_hotspot_count"]
        )

        # Outcome eligibility.
        data["has_template_issues"] = (
            data["template_issue_observations"] > 0
        )

        data["has_template_hotspots"] = (
            data["template_hotspot_observations"] > 0
        )

        # Legacy field kept so older downstream notebooks do not break.
        data["has_template_vulnerabilities"] = (
            data["has_template_issues"]
        )

        # Covariate availability.
        data["ownership_available"] = (
            data["owner_type"].notna()
        )
        data["domain_available"] = (
            data["domain"].notna()
        )
        data["commit_count_available"] = (
            data["commit_count"].notna()
        )
        data["age_available"] = (
            data["repository_age_years"].notna()
        )

        non_unit_weights = int(
            (
                ~np.isclose(
                    data["sampling_weight"],
                    1.0,
                    equal_nan=False,
                )
            ).sum()
        )

        issue_inconsistency = int(
            (~data["issue_count_consistent"]).sum()
        )

        hotspot_inconsistency = int(
            (~data["hotspot_count_consistent"]).sum()
        )

        quality_rows.append(
            {
                "Language": language,
                "Pair summary": str(pair_file),
                "Template metadata": str(template_file),
                "Domain file": (
                    str(domain_file)
                    if domain_file
                    else ""
                ),
                "Pairs": len(data),
                "Pairs with template issues": int(
                    data[
                        "has_template_issues"
                    ].sum()
                ),
                "Pairs with template Security Hotspots": int(
                    data[
                        "has_template_hotspots"
                    ].sum()
                ),
                "Issue count inconsistencies": issue_inconsistency,
                "Hotspot count inconsistencies": hotspot_inconsistency,
                "Owner matched": int(
                    data["owner_type"].notna().sum()
                ),
                "Domain matched": int(
                    data["domain"].notna().sum()
                ),
                "Commit count matched": int(
                    data["commit_count"].notna().sum()
                ),
                "Created at matched": int(
                    data[
                        "generated_created_at"
                    ].notna().sum()
                ),
                "Generated metadata sources": int(
                    data[
                        "generated_metadata_source"
                    ].nunique(
                        dropna=True
                    )
                ),
                "Age matched": int(
                    data[
                        "repository_age_years"
                    ].notna().sum()
                ),
                "Unique templates": int(
                    data[
                        "template_repository"
                    ].nunique()
                ),
                "Unique generated repositories": int(
                    data[
                        "generated_repository"
                    ].nunique()
                ),
                "Non-unit sampling weights": non_unit_weights,
                "Sampling weight min": (
                    data["sampling_weight"].min()
                ),
                "Sampling weight median": (
                    data["sampling_weight"].median()
                ),
                "Sampling weight max": (
                    data["sampling_weight"].max()
                ),
            }
        )

        all_rows.append(
            data
        )

    dataset = pd.concat(
        all_rows,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Complete-case datasets
    # ------------------------------------------------------------------

    predictor_columns = [
        "sampling_weight",
        "log_commit_count",
        "owner_type",
        "log_template_family_size",
        "language",
        "domain",
    ]

    issue_complete_columns = [
        "persistent_issue_count",
        "no_longer_present_issue_count",
        *predictor_columns,
    ]

    hotspot_complete_columns = [
        "persistent_hotspot_count",
        "no_longer_present_hotspot_count",
        *predictor_columns,
    ]

    issue_complete = dataset.loc[
        dataset["has_template_issues"]
        & dataset[
            issue_complete_columns
        ].notna().all(axis=1)
    ].copy()

    hotspot_complete = dataset.loc[
        dataset["has_template_hotspots"]
        & dataset[
            hotspot_complete_columns
        ].notna().all(axis=1)
    ].copy()

    # Long representation for optional finding_type comparisons.
    long_dataset = make_long_outcome_dataset(
        dataset
    )

    # ------------------------------------------------------------------
    # Variable summary
    # ------------------------------------------------------------------

    summary_rows = []

    for language, group in dataset.groupby(
        "language",
        sort=False,
    ):
        summary_rows.append(
            {
                "Language": language,
                "Pairs": len(group),
                "Pairs with template issues": int(
                    group[
                        "has_template_issues"
                    ].sum()
                ),
                "Pairs with template Security Hotspots": int(
                    group[
                        "has_template_hotspots"
                    ].sum()
                ),
                "Templates": int(
                    group[
                        "template_repository"
                    ].nunique()
                ),
                "Generated repositories": int(
                    group[
                        "generated_repository"
                    ].nunique()
                ),
                "Median template issue count": (
                    group[
                        "template_issue_count"
                    ].median()
                ),
                "Mean template issue count": (
                    group[
                        "template_issue_count"
                    ].mean()
                ),
                "Median template Security Hotspot count": (
                    group[
                        "template_hotspot_count"
                    ].median()
                ),
                "Mean template Security Hotspot count": (
                    group[
                        "template_hotspot_count"
                    ].mean()
                ),
                "Median issue persistence rate": (
                    group[
                        "issue_persistence_rate_recomputed"
                    ].median()
                ),
                "Median hotspot persistence rate": (
                    group[
                        "hotspot_persistence_rate_recomputed"
                    ].median()
                ),
                "Median commit count": (
                    group["commit_count"].median()
                ),
                "Mean commit count": (
                    group["commit_count"].mean()
                ),
                "Median template family size": (
                    group[
                        "template_family_size"
                    ].median()
                ),
                "Mean template family size": (
                    group[
                        "template_family_size"
                    ].mean()
                ),
                "Domains": int(
                    group[
                        "domain"
                    ].nunique(
                        dropna=True
                    )
                ),
                "Issue complete cases (initial model)": int(
                    issue_complete[
                        "language"
                    ].eq(language).sum()
                ),
                "Hotspot complete cases (initial model)": int(
                    hotspot_complete[
                        "language"
                    ].eq(language).sum()
                ),
                "Non-unit sampling weights": int(
                    (
                        ~np.isclose(
                            group[
                                "sampling_weight"
                            ],
                            1.0,
                            equal_nan=False,
                        )
                    ).sum()
                ),
            }
        )

    variable_summary = pd.DataFrame(
        summary_rows
    )

    # ------------------------------------------------------------------
    # Domain distribution
    # ------------------------------------------------------------------

    domain_distribution = (
        dataset.dropna(
            subset=["domain"]
        )
        .groupby(
            [
                "language",
                "domain",
            ]
        )
        .agg(
            templates=(
                "template_repository",
                "nunique",
            ),
            pairs=(
                "generated_repository",
                "size",
            ),
            pairs_with_template_issues=(
                "has_template_issues",
                "sum",
            ),
            pairs_with_template_hotspots=(
                "has_template_hotspots",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "language",
                "templates",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    # ------------------------------------------------------------------
    # Candidate correlations
    # ------------------------------------------------------------------

    numeric_candidates = [
        "commit_count",
        "log_commit_count",
        "template_family_size",
        "log_template_family_size",
        "template_issue_count",
        "log_template_issue_count",
        "template_hotspot_count",
        "log_template_hotspot_count",
        "repository_age_years",
        "issue_persistence_rate_recomputed",
        "hotspot_persistence_rate_recomputed",
        "new_issue_count",
        "new_hotspot_count",
    ]

    numeric_candidates = [
        col
        for col in numeric_candidates
        if col in dataset.columns
    ]

    corr = dataset[
        numeric_candidates
    ].corr(
        method="spearman",
        min_periods=20,
    )

    correlations = (
        corr.stack(
            dropna=True
        )
        .rename(
            "spearman_rho"
        )
        .reset_index()
        .rename(
            columns={
                "level_0": "variable_1",
                "level_1": "variable_2",
            }
        )
    )

    correlations = correlations[
        correlations[
            "variable_1"
        ]
        < correlations[
            "variable_2"
        ]
    ].sort_values(
        "spearman_rho",
        key=lambda s: s.abs(),
        ascending=False,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    paths = {
        "dataset": (
            output_dir
            / "rq3_analysis_dataset.csv"
        ),
        "long": (
            output_dir
            / "rq3_analysis_dataset_long.csv"
        ),
        "complete_legacy": (
            output_dir
            / "rq3_analysis_dataset_complete_cases.csv"
        ),
        "complete_issue": (
            output_dir
            / "rq3_issue_analysis_dataset_complete_cases.csv"
        ),
        "complete_hotspot": (
            output_dir
            / "rq3_hotspot_analysis_dataset_complete_cases.csv"
        ),
        "summary": (
            output_dir
            / "rq3_variable_summary.csv"
        ),
        "quality": (
            output_dir
            / "rq3_data_quality.csv"
        ),
        "domain": (
            output_dir
            / "rq3_domain_distribution.csv"
        ),
        "corr": (
            output_dir
            / "rq3_correlation_candidates.csv"
        ),
        "unmatched_domains": (
            output_dir
            / "rq3_unmatched_domain_templates.csv"
        ),
    }

    dataset.to_csv(
        paths["dataset"],
        index=False,
    )

    long_dataset.to_csv(
        paths["long"],
        index=False,
    )

    # Legacy output remains an issue-model complete-case dataset.
    issue_complete.to_csv(
        paths["complete_legacy"],
        index=False,
    )

    issue_complete.to_csv(
        paths["complete_issue"],
        index=False,
    )

    hotspot_complete.to_csv(
        paths["complete_hotspot"],
        index=False,
    )

    variable_summary.to_csv(
        paths["summary"],
        index=False,
    )

    pd.DataFrame(
        quality_rows
    ).to_csv(
        paths["quality"],
        index=False,
    )

    domain_distribution.to_csv(
        paths["domain"],
        index=False,
    )

    correlations.to_csv(
        paths["corr"],
        index=False,
    )

    if unmatched_domain_rows:
        unmatched_domains = pd.concat(
            unmatched_domain_rows,
            ignore_index=True,
        ).sort_values(
            [
                "language",
                "rq2_pairs",
            ],
            ascending=[
                True,
                False,
            ],
        )
    else:
        unmatched_domains = pd.DataFrame(
            columns=[
                "language",
                "template_repository",
                "rq2_pairs",
                "pairs_with_template_issues",
                "pairs_with_template_hotspots",
                "domain_file",
            ]
        )

    unmatched_domains.to_csv(
        paths["unmatched_domains"],
        index=False,
    )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------

    print(
        "\nRQ3 DATASET"
    )
    print(
        "=" * 96
    )

    print(
        f"Rows (template/generated pairs): "
        f"{len(dataset):,}"
    )

    print(
        f"Templates: "
        f"{dataset['template_repository'].nunique():,}"
    )

    print(
        f"Generated repositories: "
        f"{dataset['generated_repository'].nunique():,}"
    )

    print(
        f"Pairs with >=1 template issue: "
        f"{int(dataset['has_template_issues'].sum()):,}"
    )

    print(
        f"Pairs with >=1 template Security Hotspot: "
        f"{int(dataset['has_template_hotspots'].sum()):,}"
    )

    print(
        f"Issue complete cases for initial model: "
        f"{len(issue_complete):,}"
    )

    print(
        f"Security-Hotspot complete cases for initial model: "
        f"{len(hotspot_complete):,}"
    )

    print(
        "\nMATCHING / DATA QUALITY"
    )
    print(
        "=" * 96
    )

    quality_df = pd.DataFrame(
        quality_rows
    )

    print(
        quality_df.to_string(
            index=False
        )
    )

    if (
        quality_df[
            "Issue count inconsistencies"
        ].sum()
        > 0
    ):
        print(
            "\n[WARN] At least one pair has template_issue_count != "
            "persistent_issue_count + no_longer_present_issue_count."
        )

    if (
        quality_df[
            "Hotspot count inconsistencies"
        ].sum()
        > 0
    ):
        print(
            "\n[WARN] At least one pair has template_hotspot_count != "
            "persistent_hotspot_count + no_longer_present_hotspot_count."
        )

    sampled_languages = {
        "Python",
        "TypeScript",
        "JavaScript",
    }

    sampled_quality = quality_df[
        quality_df[
            "Language"
        ].isin(
            sampled_languages
        )
    ]

    if (
        not sampled_quality.empty
        and (
            sampled_quality[
                "Non-unit sampling weights"
            ]
            == 0
        ).any()
    ):
        affected = sampled_quality.loc[
            sampled_quality[
                "Non-unit sampling weights"
            ]
            == 0,
            "Language",
        ].tolist()

        print(
            "\n[WARN] Sampled language(s) with no non-unit sampling "
            f"weights detected: {affected}. "
            "If inverse-probability weighting is intended for RQ3, verify "
            "that sampling_weight was propagated into pair_summary.csv."
        )

    print(
        "\nSuggested initial issue-persistence model:"
    )
    print(
        "cbind(persistent_issue_count, no_longer_present_issue_count) "
        "~ log_commit_count + owner_type + log_template_family_size "
        "+ language + domain"
    )

    print(
        "\nSuggested initial Security-Hotspot-persistence model:"
    )
    print(
        "cbind(persistent_hotspot_count, no_longer_present_hotspot_count) "
        "~ log_commit_count + owner_type + log_template_family_size "
        "+ language + domain"
    )

    print(
        "\nRepository age uses creation metadata plus either a "
        "data_update/collection date or --age-reference-date."
    )

    print(
        "\nSaved:"
    )

    for p in paths.values():
        print(
            f"- {p}"
        )


if __name__ == "__main__":
    main()
