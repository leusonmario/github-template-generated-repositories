#!/usr/bin/env python3
"""
run_rq3_quality_issue_analysis.py

Comprehensive quality-dimension analysis for:

RQ3: Which template and generated-repository characteristics are associated
     with the persistence of quality issues?

This script extends the RQ3 analysis along three SonarQube dimensions already
stored in RQ2 `pair_summary.json` files:

1. Issue type
      BUG, CODE_SMELL, VULNERABILITY, ...
   Source: `breakdown_by_type`

2. Severity
      BLOCKER, CRITICAL, MAJOR, MINOR, INFO, ...
   Source: `breakdown_by_severity`

3. Software quality
      MAINTAINABILITY, RELIABILITY, SECURITY, ...
   Source: `breakdown_by_software_quality`

Important distinction
---------------------
`SECURITY` is a SonarQube software-quality classification. It is not renamed
to `SECURITY_HOTSPOT`. If a repository-level `security_hotspots` metric exists
elsewhere in the collection outputs, this script does not treat that aggregate
metric as issue-level persistence unless it is explicitly represented in the
RQ2 pair-comparison breakdowns.

Statistical design
------------------
Primary RQ3 analysis:
    one row = template / generated repository / issue type

    persistence ~ log2(commit_count + 1)
                + repository_age_years
                + source-template ownership
                + programming language
                + issue type

The response is the type-specific persistence proportion. The primary case
weight is:

    inherited issues of that type * RQ2 sampling weight

Repositories generated from the same source template are clustered using a
binomial GEE with independence working correlation and robust sandwich
covariance.

Parallel analyses
-----------------
A. Issue type
   - adjusted primary model
   - no-sampling-weight robustness
   - RQ1 evolution-level robustness
   - C# exclusion
   - corrected pair-level weighting
   - evolution x issue-type interaction
   - secondary domain model
   - type-stratified models

B. Severity
   Severity is mutually exclusive at the issue level, so the script runs an
   analogous adjusted model, interaction model, and severity-stratified models.

C. Software quality
   SonarQube software qualities can overlap: one issue can contribute to more
   than one quality. Therefore, MAINTAINABILITY, RELIABILITY, SECURITY, etc.
   are NOT treated as mutually exclusive levels of one categorical predictor.
   Instead, the script fits separate stratified models for each software
   quality. This avoids double-counting overlapping quality labels inside one
   categorical regression.

Corrected pair-level weighting
------------------------------
For the long-format issue-type and severity sensitivity models, a repository
pair can contribute multiple category rows. To ensure that the total pair
weight remains its RQ2 sampling weight, the pair-level sensitivity uses:

    row weight =
        RQ2 sampling weight
        / number of represented category rows for that pair

Thus, a pair with three issue types does not receive three times the influence
of a pair with one issue type.

Inputs
------
--dataset
    rq3_analysis_dataset.csv produced by the RQ3 dataset builder.

--rq2-root
    Root directory containing RQ2 outputs. The script recursively discovers
    `pair_summary.json` files.

--age-reference-date
    Fixed study snapshot date, e.g. 2026-08-17.

Outputs
-------
rq3_quality_dimension_dataset.csv
rq3_quality_dimension_support.csv
rq3_quality_dimension_persistence_summary.csv
rq3_quality_dimension_reconciliation.csv

rq3_quality_model_results.csv
rq3_quality_model_summary.csv
rq3_quality_omnibus_wald_tests.csv
rq3_quality_evolution_stratified.csv
rq3_quality_main_results.csv

rq3_quality_analysis_metadata.txt
rq3_quality_model_<MODEL>.txt
"""

from __future__ import annotations

import argparse
import json
import platform
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf


LANGUAGE_REFERENCE = "TypeScript"
OWNER_REFERENCE = "Organization"
EVOLUTION_REFERENCE = "Slightly evolved"
DOMAIN_REFERENCE_PREFERRED = "Web Development"

CATEGORY_REFERENCE_PREFERRED = {
    "issue_type": "CODE_SMELL",
    "severity": "MAJOR",
}

DIMENSION_CONFIG = {
    "issue_type": {
        "json_key": "breakdown_by_type",
        "mutually_exclusive": True,
    },
    "severity": {
        "json_key": "breakdown_by_severity",
        "mutually_exclusive": True,
    },
    "software_quality": {
        "json_key": "breakdown_by_software_quality",
        "mutually_exclusive": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run comprehensive RQ3 quality-issue analyses across issue type, "
            "severity, and SonarQube software quality."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to rq3_analysis_dataset.csv.",
    )
    parser.add_argument(
        "--rq2-root",
        type=Path,
        default=None,
        help=(
            "Root containing RQ2 outputs. Recursively discovers "
            "pair_summary.json files."
        ),
    )
    parser.add_argument(
        "--pair-summary-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit pair_summary.json path. May be repeated and combined "
            "with --rq2-root."
        ),
    )
    parser.add_argument(
        "--age-reference-date",
        required=True,
        help="Fixed study snapshot date, e.g. 2026-08-17.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rq3_quality_results"),
    )
    parser.add_argument(
        "--domain-min-templates",
        type=int,
        default=10,
        help="Minimum templates required to retain a domain separately.",
    )
    parser.add_argument(
        "--min-stratum-templates",
        type=int,
        default=10,
        help=(
            "Minimum source templates needed to fit a category-stratified "
            "model. Default: 10."
        ),
    )
    parser.add_argument(
        "--allow-partial-breakdowns",
        action="store_true",
        help=(
            "Allow RQ3 pairs with source-template issues to be missing from "
            "the supplied pair_summary.json files. Intended only for "
            "diagnostic runs."
        ),
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text if text else None


def normalize_repo(value: Any) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None

    text = text.replace("\\", "/")
    lower = text.lower()

    if "github.com/" in lower:
        pos = lower.find("github.com/")
        text = text[pos + len("github.com/"):]

    text = text.split("?", 1)[0].split("#", 1)[0].strip("/")

    if text.endswith(".git"):
        text = text[:-4]

    parts = [p for p in text.split("/") if p]
    if len(parts) < 2:
        return None

    return f"{parts[0]}/{parts[1]}".casefold()


def normalize_pair_id(value: Any) -> str | None:
    text = normalize_text(value)
    return text.casefold() if text else None


def parse_reference_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(value, utc=True, errors="raise"))


def prepare_base_dataset(
    raw: pd.DataFrame,
    age_reference_date: pd.Timestamp,
) -> pd.DataFrame:
    required = [
        "language",
        "template_repository",
        "generated_repository",
        "persistent_issue_count",
        "no_longer_present_issue_count",
        "sampling_weight",
        "owner_type",
        "commit_count",
        "evolution_level",
        "generated_created_at",
        "domain",
    ]
    require_columns(raw, required, "RQ3 dataset")

    df = raw.copy()

    numeric = [
        "persistent_issue_count",
        "no_longer_present_issue_count",
        "sampling_weight",
        "commit_count",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "template_issue_count" in df.columns:
        df["template_issue_count"] = pd.to_numeric(
            df["template_issue_count"],
            errors="coerce",
        )
    else:
        df["template_issue_count"] = (
            df["persistent_issue_count"]
            + df["no_longer_present_issue_count"]
        )

    if "new_issue_count" in df.columns:
        df["new_issue_count"] = pd.to_numeric(
            df["new_issue_count"],
            errors="coerce",
        )
    else:
        df["new_issue_count"] = np.nan

    df["sampling_weight"] = df["sampling_weight"].fillna(1.0)
    df.loc[df["sampling_weight"] <= 0, "sampling_weight"] = 1.0

    created = pd.to_datetime(
        df["generated_created_at"],
        utc=True,
        errors="coerce",
    )
    age_days = (
        age_reference_date - created
    ).dt.total_seconds() / 86400.0
    age_days = age_days.where(age_days >= 0)

    df["generated_created_at_fixed"] = created
    df["age_reference_at_fixed"] = age_reference_date
    df["repository_age_years_fixed"] = age_days / 365.25

    df["log2_commit_count"] = np.log2(df["commit_count"] + 1.0)

    df["pair_id_norm"] = (
        df["pair_id"].map(normalize_pair_id)
        if "pair_id" in df.columns
        else None
    )
    df["template_repository_norm"] = df[
        "template_repository"
    ].map(normalize_repo)
    df["generated_repository_norm"] = df[
        "generated_repository"
    ].map(normalize_repo)

    df["aggregate_trials"] = (
        df["persistent_issue_count"]
        + df["no_longer_present_issue_count"]
    )
    df["has_template_issues_fixed"] = df["aggregate_trials"] > 0

    return df


def discover_pair_summary_jsons(
    rq2_root: Path | None,
    explicit: list[Path],
) -> list[Path]:
    paths: list[Path] = []

    for item in explicit:
        path = item.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)

    if rq2_root is not None:
        root = rq2_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        paths.extend(root.rglob("pair_summary.json"))

    unique: list[Path] = []
    seen = set()

    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)

    if not unique:
        raise FileNotFoundError(
            "No pair_summary.json files were supplied or discovered."
        )

    return unique


def load_pair_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = None
        for key in ("pairs", "results", "data"):
            if isinstance(payload.get(key), list):
                rows = payload[key]
                break
        if rows is None:
            raise ValueError(
                f"{path}: expected a JSON list or pairs/results/data wrapper."
            )
    else:
        raise ValueError(f"{path}: unsupported JSON structure.")

    return [row for row in rows if isinstance(row, dict)]


def pair_key_from_record(
    record: dict[str, Any],
) -> tuple[str | None, str | None]:
    template = (
        record.get("template_repository")
        or record.get("template_id")
        or record.get("template_repo")
    )
    generated = (
        record.get("generated_repository")
        or record.get("project_id")
        or record.get("generated_repo")
    )
    return normalize_repo(template), normalize_repo(generated)


def _to_nonnegative_int(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    return max(0, int(numeric))


def build_dimension_dataset(
    base: pd.DataFrame,
    json_paths: list[Path],
    allow_partial: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct one combined long-format table with:
        dimension in {issue_type, severity, software_quality}
        category  = the dimension-specific value

    Only rows with category_template_issue_count > 0 are retained for
    persistence modeling.
    """
    by_pair_id: dict[str, int] = {}
    by_repo_pair: dict[tuple[str, str], int] = {}

    for idx, row in base.iterrows():
        pair_id = row.get("pair_id_norm")
        if pair_id:
            if pair_id in by_pair_id:
                raise ValueError(f"Duplicate pair_id: {pair_id}")
            by_pair_id[pair_id] = idx

        key = (
            row.get("template_repository_norm"),
            row.get("generated_repository_norm"),
        )
        if all(key):
            if key in by_repo_pair and by_repo_pair[key] != idx:
                raise ValueError(
                    "Duplicate normalized template/generated pair: "
                    f"{key}"
                )
            by_repo_pair[key] = idx

    matched_records: dict[int, tuple[dict[str, Any], Path]] = {}
    unmatched_json = []

    for path in json_paths:
        for record in load_pair_json(path):
            idx = None

            pair_id = normalize_pair_id(record.get("pair_id"))
            if pair_id:
                idx = by_pair_id.get(pair_id)

            if idx is None:
                key = pair_key_from_record(record)
                if all(key):
                    idx = by_repo_pair.get(key)

            if idx is None:
                unmatched_json.append({
                    "json_file": str(path),
                    "pair_id": record.get("pair_id"),
                    "template_repository": record.get(
                        "template_repository",
                        record.get("template_id"),
                    ),
                    "generated_repository": record.get(
                        "generated_repository",
                        record.get("project_id"),
                    ),
                    "note": "JSON record not matched to RQ3 dataset",
                })
                continue

            if idx in matched_records:
                previous, previous_path = matched_records[idx]

                # Require identical breakdowns if duplicate pair summaries exist.
                previous_signature = {
                    dim: previous.get(cfg["json_key"]) or {}
                    for dim, cfg in DIMENSION_CONFIG.items()
                }
                current_signature = {
                    dim: record.get(cfg["json_key"]) or {}
                    for dim, cfg in DIMENSION_CONFIG.items()
                }
                if previous_signature != current_signature:
                    raise ValueError(
                        "Conflicting pair_summary breakdowns for RQ3 row "
                        f"{idx}: {previous_path} vs {path}"
                    )
                continue

            matched_records[idx] = (record, path)

    expected_indices = base.index[
        base["has_template_issues_fixed"]
    ].tolist()

    missing_expected = [
        idx for idx in expected_indices
        if idx not in matched_records
    ]

    if missing_expected and not allow_partial:
        preview = base.loc[
            missing_expected[:10],
            ["language", "template_repository", "generated_repository"],
        ]
        raise ValueError(
            f"Missing pair_summary.json data for "
            f"{len(missing_expected):,} RQ3 pairs with source-template "
            "issues. Supply all RQ2 outputs or use "
            "--allow-partial-breakdowns for diagnostics only.\n"
            f"Examples:\n{preview.to_string(index=False)}"
        )

    long_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []

    for idx, (record, source_path) in matched_records.items():
        base_row = base.loc[idx]

        aggregate_counts = {
            "template": _to_nonnegative_int(
                base_row["template_issue_count"]
            ),
            "persistent": _to_nonnegative_int(
                base_row["persistent_issue_count"]
            ),
            "removed": _to_nonnegative_int(
                base_row["no_longer_present_issue_count"]
            ),
            "new": (
                None
                if pd.isna(base_row.get("new_issue_count", np.nan))
                else _to_nonnegative_int(base_row["new_issue_count"])
            ),
        }

        for dimension, cfg in DIMENSION_CONFIG.items():
            breakdown = record.get(cfg["json_key"]) or {}

            sums = {
                "template": 0,
                "persistent": 0,
                "removed": 0,
                "new": 0,
            }

            for category_raw, values in breakdown.items():
                if not isinstance(values, dict):
                    continue

                category = str(category_raw).strip().upper()
                if not category:
                    category = "UNKNOWN"

                template_count = _to_nonnegative_int(
                    values.get("template_issue_count", 0)
                )
                project_count = _to_nonnegative_int(
                    values.get("project_issue_count", 0)
                )
                persistent = _to_nonnegative_int(
                    values.get("persistent_issue_count", 0)
                )
                removed = _to_nonnegative_int(
                    values.get(
                        "no_longer_present_issue_count",
                        0,
                    )
                )
                new_count = _to_nonnegative_int(
                    values.get("new_issue_count", 0)
                )

                sums["template"] += template_count
                sums["persistent"] += persistent
                sums["removed"] += removed
                sums["new"] += new_count

                if persistent + removed != template_count:
                    raise ValueError(
                        f"{source_path}: {record.get('pair_id')} "
                        f"{dimension}={category}: persistent + removed "
                        "does not equal template_issue_count."
                    )

                # New-only categories cannot define persistence of an inherited
                # source-template issue, so they are omitted from regression.
                if template_count <= 0:
                    continue

                out = base_row.to_dict()
                out.update({
                    "dimension": dimension,
                    "category": category,
                    "category_template_issue_count": template_count,
                    "category_project_issue_count": project_count,
                    "category_persistent_issue_count": persistent,
                    "category_no_longer_present_issue_count": removed,
                    "category_new_issue_count": new_count,
                    "category_binomial_trials": template_count,
                    "category_persistence_proportion": (
                        persistent / template_count
                    ),
                    "breakdown_source": str(source_path),
                    "dimension_is_mutually_exclusive": bool(
                        cfg["mutually_exclusive"]
                    ),
                })
                long_rows.append(out)

            reconciliation_expected = cfg["mutually_exclusive"]

            reconciliation_rows.append({
                "language": base_row["language"],
                "template_repository": base_row["template_repository"],
                "generated_repository": base_row["generated_repository"],
                "pair_id": base_row.get("pair_id"),
                "dimension": dimension,
                "json_source": str(source_path),
                "mutually_exclusive_dimension": reconciliation_expected,
                "aggregate_template_issue_count": aggregate_counts["template"],
                "dimension_sum_template_issue_count": sums["template"],
                "aggregate_persistent_issue_count": aggregate_counts["persistent"],
                "dimension_sum_persistent_issue_count": sums["persistent"],
                "aggregate_removed_issue_count": aggregate_counts["removed"],
                "dimension_sum_removed_issue_count": sums["removed"],
                "aggregate_new_issue_count": aggregate_counts["new"],
                "dimension_sum_new_issue_count": sums["new"],
                "template_count_matches": (
                    aggregate_counts["template"] == sums["template"]
                    if reconciliation_expected else np.nan
                ),
                "persistent_count_matches": (
                    aggregate_counts["persistent"] == sums["persistent"]
                    if reconciliation_expected else np.nan
                ),
                "removed_count_matches": (
                    aggregate_counts["removed"] == sums["removed"]
                    if reconciliation_expected else np.nan
                ),
                "new_count_matches": (
                    (
                        True
                        if aggregate_counts["new"] is None
                        else aggregate_counts["new"] == sums["new"]
                    )
                    if reconciliation_expected else np.nan
                ),
                "note": (
                    ""
                    if reconciliation_expected
                    else (
                        "Software-quality labels may overlap; sums are not "
                        "expected to equal aggregate issue counts."
                    )
                ),
            })

    long_df = pd.DataFrame(long_rows)
    reconciliation = pd.DataFrame(reconciliation_rows)

    if long_df.empty:
        raise ValueError("No quality-dimension persistence rows were built.")

    if unmatched_json:
        reconciliation = pd.concat(
            [
                reconciliation,
                pd.DataFrame(unmatched_json),
            ],
            ignore_index=True,
            sort=False,
        )

    # Hard reconciliation checks only for mutually exclusive dimensions.
    exclusive = reconciliation[
        reconciliation["mutually_exclusive_dimension"].eq(True)
    ].copy()

    for col in [
        "template_count_matches",
        "persistent_count_matches",
        "removed_count_matches",
        "new_count_matches",
    ]:
        bad = exclusive[col].eq(False).sum()
        if bad:
            raise ValueError(
                f"Reconciliation failed for {bad} pair/dimension rows "
                f"in {col}. Inspect rq3_quality_dimension_reconciliation.csv."
            )

    return long_df, reconciliation


def create_domain_grouping(
    df: pd.DataFrame,
    min_templates: int,
) -> tuple[pd.DataFrame, pd.Series, str | None]:
    work = df.copy()
    work["domain_clean"] = work["domain"].map(normalize_text)

    support = (
        work.dropna(subset=["domain_clean"])
        .groupby("domain_clean")["template_repository"]
        .nunique()
        .sort_values(ascending=False)
    )

    keep = set(support[support >= min_templates].index)

    def grouped(value):
        if value is None or pd.isna(value):
            return np.nan
        return value if value in keep else "Other"

    work["domain_grouped"] = work["domain_clean"].map(grouped)

    grouped_support = (
        work.dropna(subset=["domain_grouped"])
        .groupby("domain_grouped")["template_repository"]
        .nunique()
        .sort_values(ascending=False)
    )

    if grouped_support.empty:
        return work, grouped_support, None

    reference = (
        DOMAIN_REFERENCE_PREFERRED
        if DOMAIN_REFERENCE_PREFERRED in grouped_support.index
        else str(grouped_support.index[0])
    )

    return work, grouped_support, reference


def support_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dimension, category), group in df.groupby(
        ["dimension", "category"],
        observed=False,
    ):
        raw_trials = group["category_binomial_trials"].sum()
        weighted_trials = (
            group["category_binomial_trials"]
            * group["sampling_weight"]
        ).sum()

        raw_persistent = group[
            "category_persistent_issue_count"
        ].sum()
        weighted_persistent = (
            group["category_persistent_issue_count"]
            * group["sampling_weight"]
        ).sum()

        rows.append({
            "dimension": dimension,
            "category": category,
            "pair_category_rows": len(group),
            "source_templates": group[
                "template_repository"
            ].nunique(),
            "generated_repositories": group[
                "generated_repository"
            ].nunique(),
            "languages": group["language"].nunique(),
            "raw_binomial_trials": raw_trials,
            "weighted_binomial_trials": weighted_trials,
            "raw_persistent": raw_persistent,
            "weighted_persistent": weighted_persistent,
            "weighted_persistence_percentage": (
                100.0 * weighted_persistent / weighted_trials
                if weighted_trials > 0
                else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values(
        [
            "dimension",
            "weighted_binomial_trials",
            "source_templates",
        ],
        ascending=[True, False, False],
    )


def persistence_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for scope, keys in [
        ("All", ["dimension", "category"]),
        ("By language", ["dimension", "category", "language"]),
    ]:
        for values, group in df.groupby(keys, observed=False):
            if not isinstance(values, tuple):
                values = (values,)

            info = dict(zip(keys, values))

            weighted_trials = (
                group["category_binomial_trials"]
                * group["sampling_weight"]
            ).sum()

            weighted_persistent = (
                group["category_persistent_issue_count"]
                * group["sampling_weight"]
            ).sum()

            rows.append({
                "scope": scope,
                "dimension": info["dimension"],
                "category": info["category"],
                "language": info.get("language", "All"),
                "pair_category_rows": len(group),
                "source_templates": group[
                    "template_repository"
                ].nunique(),
                "raw_trials": group[
                    "category_binomial_trials"
                ].sum(),
                "weighted_trials": weighted_trials,
                "weighted_persistent": weighted_persistent,
                "weighted_persistence_percentage": (
                    100.0
                    * weighted_persistent
                    / weighted_trials
                    if weighted_trials > 0
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


def escape_reference(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def categorical_term(column: str, reference: str) -> str:
    return (
        f'C({column}, Treatment(reference="{escape_reference(reference)}"))'
    )


def choose_category_reference(
    data: pd.DataFrame,
    dimension: str,
) -> str:
    levels = [
        str(x)
        for x in data["category"].dropna().unique()
    ]

    if not levels:
        raise ValueError(
            f"No categories available for dimension {dimension}."
        )

    preferred = CATEGORY_REFERENCE_PREFERRED.get(dimension)
    if preferred in levels:
        return preferred

    support = (
        data.groupby("category")["category_binomial_trials"]
        .sum()
        .sort_values(ascending=False)
    )
    return str(support.index[0])


def build_formula(
    data: pd.DataFrame,
    *,
    include_category: bool,
    category_reference: str | None,
    use_evolution_levels: bool,
    include_domain: bool,
    domain_reference: str | None,
    interaction: bool,
) -> tuple[str, dict[str, str]]:
    terms: list[str] = []
    references: dict[str, str] = {}

    if use_evolution_levels:
        levels = [
            str(x)
            for x in data["evolution_level"].dropna().unique()
        ]
        if not levels:
            raise ValueError("No evolution levels available.")
        evolution_ref = (
            EVOLUTION_REFERENCE
            if EVOLUTION_REFERENCE in levels
            else sorted(levels)[0]
        )
        references["evolution_level"] = evolution_ref
        evolution_term = categorical_term(
            "evolution_level",
            evolution_ref,
        )
    else:
        evolution_term = "log2_commit_count"

    # Ownership.
    owner_levels = [
        str(x)
        for x in data["owner_type"].dropna().unique()
    ]
    if len(owner_levels) > 1:
        owner_ref = (
            OWNER_REFERENCE
            if OWNER_REFERENCE in owner_levels
            else sorted(owner_levels)[0]
        )
        references["owner_type"] = owner_ref
        terms.append(
            categorical_term("owner_type", owner_ref)
        )

    # Language.
    language_levels = [
        str(x)
        for x in data["language"].dropna().unique()
    ]
    if len(language_levels) > 1:
        language_ref = (
            LANGUAGE_REFERENCE
            if LANGUAGE_REFERENCE in language_levels
            else str(
                data.groupby("language")[
                    "category_binomial_trials"
                ].sum().idxmax()
            )
        )
        references["language"] = language_ref
        terms.append(
            categorical_term("language", language_ref)
        )

    # Main evolution term and category.
    if include_category:
        if category_reference is None:
            raise ValueError("Category reference is required.")
        references["category"] = category_reference
        category_term = categorical_term(
            "category",
            category_reference,
        )

        if interaction:
            terms.insert(
                0,
                f"{evolution_term} * {category_term}",
            )
        else:
            terms.insert(0, evolution_term)
            terms.append(category_term)
    else:
        terms.insert(0, evolution_term)

    # Age immediately after evolution for readability.
    terms.insert(1, "repository_age_years_fixed")

    if include_domain:
        if domain_reference is None:
            raise ValueError("Domain reference is required.")
        references["domain_grouped"] = domain_reference
        terms.append(
            categorical_term(
                "domain_grouped",
                domain_reference,
            )
        )

    return (
        "category_persistence_proportion ~ "
        + " + ".join(terms),
        references,
    )


def required_columns(
    *,
    include_category: bool,
    use_evolution_levels: bool,
    include_domain: bool,
) -> list[str]:
    columns = [
        "category_persistence_proportion",
        "category_binomial_trials",
        "sampling_weight",
        "repository_age_years_fixed",
        "owner_type",
        "language",
        "template_repository",
        "generated_repository",
    ]

    columns.append(
        "evolution_level"
        if use_evolution_levels
        else "log2_commit_count"
    )

    if include_category:
        columns.append("category")

    if include_domain:
        columns.append("domain_grouped")

    return columns


def fit_gee(
    source: pd.DataFrame,
    *,
    model_name: str,
    include_category: bool,
    category_reference: str | None = None,
    use_sampling_weights: bool = True,
    pair_level_weighting: bool = False,
    use_evolution_levels: bool = False,
    include_domain: bool = False,
    domain_reference: str | None = None,
    exclude_csharp: bool = False,
    interaction: bool = False,
) -> tuple[Any, pd.DataFrame, str, dict[str, str], list[str]]:
    data = source.copy()

    if exclude_csharp:
        data = data[data["language"] != "C#"].copy()

    data = data.dropna(
        subset=required_columns(
            include_category=include_category,
            use_evolution_levels=use_evolution_levels,
            include_domain=include_domain,
        )
    ).copy()

    if data.empty:
        raise ValueError(f"{model_name}: no complete observations.")

    if pair_level_weighting:
        # Correct pair-level sensitivity:
        # split each pair's sampling weight across the category rows that
        # actually enter this model.
        data["_pair_key_for_weight"] = (
            data["template_repository"].astype(str)
            + "||"
            + data["generated_repository"].astype(str)
        )
        data["_represented_category_rows"] = (
            data.groupby("_pair_key_for_weight")[
                "category"
            ].transform("size")
            if include_category
            else 1
        )

        data["analysis_weight"] = (
            data["sampling_weight"]
            / data["_represented_category_rows"]
        )
    elif use_sampling_weights:
        data["analysis_weight"] = (
            data["category_binomial_trials"]
            * data["sampling_weight"]
        )
    else:
        data["analysis_weight"] = data[
            "category_binomial_trials"
        ]

    data = data[
        np.isfinite(data["analysis_weight"])
        & (data["analysis_weight"] > 0)
    ].copy()

    formula, references = build_formula(
        data,
        include_category=include_category,
        category_reference=category_reference,
        use_evolution_levels=use_evolution_levels,
        include_domain=include_domain,
        domain_reference=domain_reference,
        interaction=interaction,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        model = smf.gee(
            formula=formula,
            groups="template_repository",
            data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
            weights=data["analysis_weight"],
        )
        result = model.fit()

    warning_messages = [
        f"{type(w.message).__name__}: {w.message}"
        for w in caught
    ]

    return (
        result,
        data,
        formula,
        references,
        warning_messages,
    )


def extract_results(
    model_name: str,
    result: Any,
    *,
    dimension: str,
    category_stratum: str | None = None,
) -> pd.DataFrame:
    conf = result.conf_int()

    rows = []
    for term in result.params.index:
        coef = float(result.params.loc[term])
        low = float(conf.loc[term, 0])
        high = float(conf.loc[term, 1])

        rows.append({
            "model": model_name,
            "dimension": dimension,
            "category_stratum": category_stratum,
            "term": term,
            "coefficient_log_odds": coef,
            "std_error": float(result.bse.loc[term]),
            "odds_ratio": float(np.exp(coef)),
            "ci_95_low": float(np.exp(low)),
            "ci_95_high": float(np.exp(high)),
            "p_value": float(result.pvalues.loc[term]),
        })

    return pd.DataFrame(rows)


def joint_wald(
    result: Any,
    terms: list[str],
) -> dict[str, Any] | None:
    terms = [
        term
        for term in terms
        if term in result.params.index
    ]

    if not terms:
        return None

    positions = {
        name: i
        for i, name in enumerate(result.params.index)
    }

    R = np.zeros(
        (len(terms), len(result.params)),
        dtype=float,
    )

    for row, term in enumerate(terms):
        R[row, positions[term]] = 1.0

    test = result.wald_test(R, scalar=True)

    return {
        "df": int(np.linalg.matrix_rank(R)),
        "wald_chi2": float(
            np.asarray(test.statistic).squeeze()
        ),
        "p_value": float(
            np.asarray(test.pvalue).squeeze()
        ),
        "terms_tested": " | ".join(terms),
    }


def omnibus_tests(
    model_name: str,
    result: Any,
    *,
    dimension: str,
) -> pd.DataFrame:
    names = list(result.params.index)

    groups = {
        "programming_language": [
            t
            for t in names
            if t.startswith("C(language,")
        ],
        "source_template_ownership": [
            t
            for t in names
            if t.startswith("C(owner_type,")
        ],
        "evolution_level": [
            t
            for t in names
            if t.startswith("C(evolution_level,")
        ],
        dimension: [
            t
            for t in names
            if t.startswith("C(category,")
            and ":" not in t
        ],
        "domain": [
            t
            for t in names
            if t.startswith("C(domain_grouped,")
        ],
        f"evolution_by_{dimension}_interaction": [
            t
            for t in names
            if (
                "log2_commit_count:C(category," in t
                or (
                    "C(category," in t
                    and ":log2_commit_count" in t
                )
            )
        ],
    }

    rows = []

    for characteristic, terms in groups.items():
        tested = joint_wald(result, terms)
        if tested is None:
            continue

        rows.append({
            "model": model_name,
            "dimension": dimension,
            "characteristic": characteristic,
            **tested,
        })

    return pd.DataFrame(rows)


def write_text_result(
    path: Path,
    result: Any,
    formula: str,
    data: pd.DataFrame,
    references: dict[str, str],
    warning_messages: list[str],
) -> None:
    lines = [
        f"Formula: {formula}",
        f"Rows: {len(data)}",
        (
            "Source templates: "
            f"{data['template_repository'].nunique()}"
        ),
        (
            "Generated repositories: "
            f"{data['generated_repository'].nunique()}"
        ),
        f"Languages: {data['language'].nunique()}",
        (
            "Categories: "
            f"{data['category'].nunique()}"
            if "category" in data.columns
            else "Categories: stratified model"
        ),
        f"References: {references}",
        (
            "Converged: "
            f"{getattr(result, 'converged', True)}"
        ),
        "",
        str(result.summary()),
    ]

    if warning_messages:
        lines.extend(["", "Warnings:"])
        lines.extend(warning_messages)

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def summary_row(
    model_name: str,
    result: Any,
    data: pd.DataFrame,
    formula: str,
    references: dict[str, str],
    warnings_list: list[str],
    *,
    dimension: str,
    weighting_scheme: str,
    category_stratum: str | None = None,
    status: str = "fit",
) -> dict[str, Any]:
    return {
        "model": model_name,
        "dimension": dimension,
        "category_stratum": category_stratum,
        "formula": formula,
        "rows": len(data),
        "source_templates": data[
            "template_repository"
        ].nunique(),
        "generated_repositories": data[
            "generated_repository"
        ].nunique(),
        "languages": data["language"].nunique(),
        "categories": (
            data["category"].nunique()
            if "category" in data.columns
            else 1
        ),
        "weighting_scheme": weighting_scheme,
        "references": json.dumps(
            references,
            sort_keys=True,
        ),
        "converged": bool(
            getattr(result, "converged", True)
        ),
        "warnings": " || ".join(warnings_list),
        "status": status,
    }


def fit_adjusted_dimension_models(
    dimension_df: pd.DataFrame,
    *,
    dimension: str,
    output_dir: Path,
    domain_reference: str | None,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict[str, Any]]]:
    """
    Run adjusted models for mutually exclusive dimensions:
      issue_type and severity.
    """
    category_reference = choose_category_reference(
        dimension_df,
        dimension,
    )

    model_specs = [
        (
            f"{dimension}_adjusted_primary_weighted",
            dict(
                use_sampling_weights=True,
                pair_level_weighting=False,
                use_evolution_levels=False,
                include_domain=False,
                exclude_csharp=False,
                interaction=False,
            ),
            "category_trials_x_sampling_weight",
        ),
        (
            f"{dimension}_adjusted_no_sampling_weights",
            dict(
                use_sampling_weights=False,
                pair_level_weighting=False,
                use_evolution_levels=False,
                include_domain=False,
                exclude_csharp=False,
                interaction=False,
            ),
            "category_trials_only",
        ),
        (
            f"{dimension}_adjusted_evolution_levels",
            dict(
                use_sampling_weights=True,
                pair_level_weighting=False,
                use_evolution_levels=True,
                include_domain=False,
                exclude_csharp=False,
                interaction=False,
            ),
            "category_trials_x_sampling_weight",
        ),
        (
            f"{dimension}_adjusted_no_csharp",
            dict(
                use_sampling_weights=True,
                pair_level_weighting=False,
                use_evolution_levels=False,
                include_domain=False,
                exclude_csharp=True,
                interaction=False,
            ),
            "category_trials_x_sampling_weight",
        ),
        (
            f"{dimension}_adjusted_pair_level_weighted",
            dict(
                use_sampling_weights=True,
                pair_level_weighting=True,
                use_evolution_levels=False,
                include_domain=False,
                exclude_csharp=False,
                interaction=False,
            ),
            "sampling_weight_split_across_pair_category_rows",
        ),
        (
            f"{dimension}_evolution_x_category",
            dict(
                use_sampling_weights=True,
                pair_level_weighting=False,
                use_evolution_levels=False,
                include_domain=False,
                exclude_csharp=False,
                interaction=True,
            ),
            "category_trials_x_sampling_weight",
        ),
    ]

    # Domain is secondary and is attached to the issue-type model only.
    if dimension == "issue_type" and domain_reference is not None:
        model_specs.append(
            (
                f"{dimension}_adjusted_secondary_domain",
                dict(
                    use_sampling_weights=True,
                    pair_level_weighting=False,
                    use_evolution_levels=False,
                    include_domain=True,
                    exclude_csharp=False,
                    interaction=False,
                ),
                "category_trials_x_sampling_weight",
            )
        )

    result_frames = []
    omnibus_frames = []
    summaries = []

    for model_name, spec, weighting_scheme in model_specs:
        print(f"Fitting {model_name}...")

        result, data, formula, references, warning_messages = fit_gee(
            dimension_df,
            model_name=model_name,
            include_category=True,
            category_reference=category_reference,
            domain_reference=domain_reference,
            **spec,
        )

        result_frames.append(
            extract_results(
                model_name,
                result,
                dimension=dimension,
            )
        )

        omnibus = omnibus_tests(
            model_name,
            result,
            dimension=dimension,
        )
        if not omnibus.empty:
            omnibus_frames.append(omnibus)

        summaries.append(
            summary_row(
                model_name,
                result,
                data,
                formula,
                references,
                warning_messages,
                dimension=dimension,
                weighting_scheme=weighting_scheme,
            )
        )

        write_text_result(
            output_dir
            / f"rq3_quality_model_{model_name}.txt",
            result,
            formula,
            data,
            references,
            warning_messages,
        )

    return result_frames, omnibus_frames, summaries


def fit_stratified_models(
    full_df: pd.DataFrame,
    support: pd.DataFrame,
    *,
    dimension: str,
    min_templates: int,
    output_dir: Path,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    result_frames = []
    summaries = []

    dimension_support = support[
        support["dimension"] == dimension
    ].copy()

    for _, support_row in dimension_support.iterrows():
        category = str(support_row["category"])
        n_templates = int(
            support_row["source_templates"]
        )

        model_name = (
            f"{dimension}_stratified_{category}"
        )

        if n_templates < min_templates:
            summaries.append({
                "model": model_name,
                "dimension": dimension,
                "category_stratum": category,
                "status": "skipped_low_template_support",
                "source_templates": n_templates,
                "minimum_required_templates": min_templates,
            })
            continue

        subset = full_df[
            (full_df["dimension"] == dimension)
            & (full_df["category"] == category)
        ].copy()

        try:
            result, data, formula, references, warning_messages = fit_gee(
                subset,
                model_name=model_name,
                include_category=False,
                category_reference=None,
                use_sampling_weights=True,
                pair_level_weighting=False,
                use_evolution_levels=False,
                include_domain=False,
                domain_reference=None,
                exclude_csharp=False,
                interaction=False,
            )
        except Exception as exc:
            summaries.append({
                "model": model_name,
                "dimension": dimension,
                "category_stratum": category,
                "status": "fit_failed",
                "source_templates": n_templates,
                "minimum_required_templates": min_templates,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        result_frames.append(
            extract_results(
                model_name,
                result,
                dimension=dimension,
                category_stratum=category,
            )
        )

        summaries.append(
            summary_row(
                model_name,
                result,
                data,
                formula,
                references,
                warning_messages,
                dimension=dimension,
                weighting_scheme=(
                    "category_trials_x_sampling_weight"
                ),
                category_stratum=category,
            )
        )

        write_text_result(
            output_dir
            / f"rq3_quality_model_{model_name}.txt",
            result,
            formula,
            data,
            references,
            warning_messages,
        )

    return result_frames, summaries


def build_evolution_stratified_table(
    model_results: pd.DataFrame,
) -> pd.DataFrame:
    subset = model_results[
        model_results["category_stratum"].notna()
        & model_results["term"].eq("log2_commit_count")
    ].copy()

    if subset.empty:
        return pd.DataFrame(columns=[
            "dimension",
            "category",
            "odds_ratio",
            "ci_95_low",
            "ci_95_high",
            "p_value",
        ])

    return subset[
        [
            "dimension",
            "category_stratum",
            "odds_ratio",
            "ci_95_low",
            "ci_95_high",
            "p_value",
        ]
    ].rename(
        columns={"category_stratum": "category"}
    ).sort_values(
        ["dimension", "category"]
    )


def build_main_results(
    model_results: pd.DataFrame,
    omnibus: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compact table for the main paper interpretation.
    """
    rows = []

    main_models = [
        "issue_type_adjusted_primary_weighted",
        "issue_type_adjusted_no_sampling_weights",
        "issue_type_adjusted_pair_level_weighted",
        "issue_type_adjusted_no_csharp",
        "severity_adjusted_primary_weighted",
    ]

    for model_name in main_models:
        subset = model_results[
            model_results["model"] == model_name
        ]

        for term, label in [
            (
                "log2_commit_count",
                "Commit count (doubling)",
            ),
            (
                "repository_age_years_fixed",
                "Repository age (+1 year)",
            ),
        ]:
            hit = subset[subset["term"] == term]
            if hit.empty:
                continue
            row = hit.iloc[0]

            rows.append({
                "model": model_name,
                "result_kind": "coefficient",
                "characteristic": label,
                "odds_ratio": row["odds_ratio"],
                "ci_95_low": row["ci_95_low"],
                "ci_95_high": row["ci_95_high"],
                "p_value": row["p_value"],
                "wald_chi2": np.nan,
                "df": np.nan,
            })

    if not omnibus.empty:
        wanted = omnibus[
            (
                omnibus["model"].eq(
                    "issue_type_adjusted_primary_weighted"
                )
                & omnibus["characteristic"].isin([
                    "programming_language",
                    "source_template_ownership",
                    "issue_type",
                ])
            )
            | (
                omnibus["model"].eq(
                    "issue_type_adjusted_evolution_levels"
                )
                & omnibus["characteristic"].eq(
                    "evolution_level"
                )
            )
            | (
                omnibus["model"].eq(
                    "issue_type_evolution_x_category"
                )
                & omnibus["characteristic"].eq(
                    "evolution_by_issue_type_interaction"
                )
            )
            | (
                omnibus["model"].eq(
                    "severity_adjusted_primary_weighted"
                )
                & omnibus["characteristic"].eq(
                    "severity"
                )
            )
            | (
                omnibus["model"].eq(
                    "severity_evolution_x_category"
                )
                & omnibus["characteristic"].eq(
                    "evolution_by_severity_interaction"
                )
            )
            | (
                omnibus["model"].eq(
                    "issue_type_adjusted_secondary_domain"
                )
                & omnibus["characteristic"].eq(
                    "domain"
                )
            )
        ]

        for _, row in wanted.iterrows():
            rows.append({
                "model": row["model"],
                "result_kind": "omnibus_wald",
                "characteristic": row[
                    "characteristic"
                ],
                "odds_ratio": np.nan,
                "ci_95_low": np.nan,
                "ci_95_high": np.nan,
                "p_value": row["p_value"],
                "wald_chi2": row["wald_chi2"],
                "df": row["df"],
            })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    if args.domain_min_templates < 1:
        raise ValueError(
            "--domain-min-templates must be >= 1."
        )
    if args.min_stratum_templates < 1:
        raise ValueError(
            "--min-stratum-templates must be >= 1."
        )

    dataset_path = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    age_reference_date = parse_reference_date(
        args.age_reference_date
    )

    raw = pd.read_csv(
        dataset_path,
        low_memory=False,
    )
    base = prepare_base_dataset(
        raw,
        age_reference_date,
    )

    json_paths = discover_pair_summary_jsons(
        args.rq2_root,
        args.pair_summary_json,
    )

    print("Discovered pair_summary.json files:")
    for path in json_paths:
        print(f"  - {path}")

    long_df, reconciliation = build_dimension_dataset(
        base,
        json_paths,
        args.allow_partial_breakdowns,
    )

    long_df, domain_support, domain_reference = (
        create_domain_grouping(
            long_df,
            args.domain_min_templates,
        )
    )

    support = support_table(long_df)

    # Save derived data and diagnostics first.
    long_df.to_csv(
        output_dir
        / "rq3_quality_dimension_dataset.csv",
        index=False,
    )
    support.to_csv(
        output_dir
        / "rq3_quality_dimension_support.csv",
        index=False,
    )
    persistence_summary(long_df).to_csv(
        output_dir
        / "rq3_quality_dimension_persistence_summary.csv",
        index=False,
    )
    reconciliation.to_csv(
        output_dir
        / "rq3_quality_dimension_reconciliation.csv",
        index=False,
    )

    print("\nQUALITY-DIMENSION SUPPORT")
    print("=" * 100)
    print(
        support[
            [
                "dimension",
                "category",
                "pair_category_rows",
                "source_templates",
                "raw_binomial_trials",
                "weighted_persistence_percentage",
            ]
        ].to_string(index=False)
    )

    all_result_frames: list[pd.DataFrame] = []
    all_omnibus_frames: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # A. Adjusted models for mutually exclusive dimensions.
    # ------------------------------------------------------------
    for dimension in ["issue_type", "severity"]:
        dimension_df = long_df[
            long_df["dimension"] == dimension
        ].copy()

        (
            result_frames,
            omnibus_frames,
            summaries,
        ) = fit_adjusted_dimension_models(
            dimension_df,
            dimension=dimension,
            output_dir=output_dir,
            domain_reference=domain_reference,
        )

        all_result_frames.extend(result_frames)
        all_omnibus_frames.extend(omnibus_frames)
        all_summaries.extend(summaries)

    # ------------------------------------------------------------
    # B. Stratified models for all three dimensions.
    #
    # For software_quality this is the inferential strategy because
    # categories may overlap.
    # ------------------------------------------------------------
    for dimension in [
        "issue_type",
        "severity",
        "software_quality",
    ]:
        result_frames, summaries = fit_stratified_models(
            long_df,
            support,
            dimension=dimension,
            min_templates=args.min_stratum_templates,
            output_dir=output_dir,
        )

        all_result_frames.extend(result_frames)
        all_summaries.extend(summaries)

    model_results = (
        pd.concat(
            all_result_frames,
            ignore_index=True,
            sort=False,
        )
        if all_result_frames
        else pd.DataFrame()
    )

    omnibus = (
        pd.concat(
            all_omnibus_frames,
            ignore_index=True,
            sort=False,
        )
        if all_omnibus_frames
        else pd.DataFrame()
    )

    model_summary = pd.DataFrame(all_summaries)

    model_results.to_csv(
        output_dir
        / "rq3_quality_model_results.csv",
        index=False,
    )
    model_summary.to_csv(
        output_dir
        / "rq3_quality_model_summary.csv",
        index=False,
    )
    omnibus.to_csv(
        output_dir
        / "rq3_quality_omnibus_wald_tests.csv",
        index=False,
    )

    evolution_table = (
        build_evolution_stratified_table(
            model_results
        )
    )
    evolution_table.to_csv(
        output_dir
        / "rq3_quality_evolution_stratified.csv",
        index=False,
    )

    build_main_results(
        model_results,
        omnibus,
    ).to_csv(
        output_dir
        / "rq3_quality_main_results.csv",
        index=False,
    )

    # Metadata / reproducibility notes.
    discovered_categories = (
        support.groupby("dimension")["category"]
        .apply(lambda s: ", ".join(map(str, s)))
        .to_dict()
    )

    metadata_lines = [
        f"Input RQ3 dataset: {dataset_path}",
        (
            "Age reference date: "
            f"{age_reference_date.isoformat()}"
        ),
        (
            "Pair-summary JSON files: "
            f"{len(json_paths)}"
        ),
        *[
            f"  - {path}"
            for path in json_paths
        ],
        "",
        "Dimensions discovered:",
        *[
            f"  {dimension}: "
            f"{discovered_categories.get(dimension, '')}"
            for dimension in DIMENSION_CONFIG
        ],
        "",
        (
            "Issue type and severity are treated as mutually "
            "exclusive categorical dimensions."
        ),
        (
            "Software quality is treated only through separate "
            "stratified models because SonarQube software-quality "
            "labels can overlap."
        ),
        (
            "SECURITY is preserved as the SonarQube software-quality "
            "label and is not renamed SECURITY_HOTSPOT."
        ),
        "",
        "Primary issue-type estimation:",
        (
            "Binomial GEE on pair x issue-type observations, "
            "clustered by source template, independence working "
            "correlation, robust sandwich covariance."
        ),
        (
            "Primary weight = type-specific inherited issue count "
            "* RQ2 sampling weight."
        ),
        "",
        "Corrected pair-level sensitivity:",
        (
            "For mutually exclusive long-format dimensions, each "
            "pair's sampling weight is divided by the number of "
            "category rows represented by that pair in the fitted "
            "model. Therefore, the category rows sum back to the "
            "pair's original sampling weight."
        ),
        "",
        "Software:",
        f"Python: {platform.python_version()}",
        f"pandas: {pd.__version__}",
        f"numpy: {np.__version__}",
        f"scipy: {scipy.__version__}",
        f"statsmodels: {statsmodels.__version__}",
    ]

    (
        output_dir
        / "rq3_quality_analysis_metadata.txt"
    ).write_text(
        "\n".join(metadata_lines) + "\n",
        encoding="utf-8",
    )

    print("\nRQ3 QUALITY-DIMENSION ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"Base RQ3 pairs: {len(base):,}")
    print(
        "Base pairs with >=1 source-template issue: "
        f"{int(base['has_template_issues_fixed'].sum()):,}"
    )
    print(
        "Long-format dimension rows: "
        f"{len(long_df):,}"
    )
    print(
        "Source templates represented: "
        f"{long_df['template_repository'].nunique():,}"
    )
    print(
        "Generated repositories represented: "
        f"{long_df['generated_repository'].nunique():,}"
    )
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
