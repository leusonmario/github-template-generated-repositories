#!/usr/bin/env python3
"""
define_persistence_sample.py

Define a reproducible, template-stratified sample of repositories generated
from GitHub templates.

Recommended use for issue-persistence analysis:
  1. Keep only genuine generated repositories.
  2. Require at least two commits.
  3. Restrict repository creation to the original observation window.
  4. Represent every eligible template family.
  5. For large families, sample repositories across evolution bands defined
     by commit count.
  6. Record selection probabilities and sampling weights.

Default evolution bands:
  - low:    2-5 commits
  - medium: 6-20 commits
  - high:   21+ commits

Default sampling:
  - Include every eligible repository when a template has <= 20 descendants.
  - Otherwise, sample at most 20 descendants from that template.
  - Allocate the 20 positions proportionally across evolution bands while
    guaranteeing at least one repository from every non-empty band.
  - Use a fixed seed for reproducibility.

Example:
    python define_persistence_sample.py \
        generated_repositories.csv \
        sampling_outputs

Alternative balanced allocation:
    python define_persistence_sample.py \
        generated_repositories.csv \
        sampling_outputs_balanced \
        --allocation balanced

Outputs:
    sampling_outputs/
        selected_sample.csv
        eligible_candidates.csv
        template_sampling_report.csv
        sampling_summary.json
        exclusion_reason_counts.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "original_link",
    "target_template_repository",
    "is_template",
    "commit_count",
    "data_creation",
}

OPTIONAL_COLUMNS = {
    "data_update",
    "commit_info_status",
    "stars",
    "forks",
}


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Parse common boolean representations without treating non-empty strings as True."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = (
        series.astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }

    return normalized.map(mapping).astype("boolean")


def github_slug(value: Any) -> str | None:
    """
    Normalize a GitHub URL or owner/repository identifier.

    Examples:
      https://github.com/Owner/Repo.git -> owner/repo
      Owner/Repo                       -> owner/repo
    """
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.rstrip("/")
    if text.lower().endswith(".git"):
        text = text[:-4]

    github_marker = "github.com/"
    marker_index = text.lower().find(github_marker)

    if marker_index >= 0:
        text = text[marker_index + len(github_marker):]

    text = text.strip("/")

    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        return None

    return f"{parts[-2]}/{parts[-1]}".lower()


def stable_random_state(seed: int, *parts: str) -> int:
    """Create a deterministic pandas random_state from the seed and identifiers."""
    raw = "::".join([str(seed), *map(str, parts)])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def evolution_band(commit_count: float) -> str:
    """Classify a generated repository by degree of evolution."""
    if commit_count <= 6:
        return "low_2_to_6"
    if commit_count <= 20:
        return "medium_7_to_20"
    return "high_21_plus"


def allocate_proportionally(
    stratum_sizes: dict[str, int],
    sample_size: int,
) -> dict[str, int]:
    """
    Allocate sample positions proportionally across non-empty strata.

    At least one position is assigned to every non-empty stratum when the
    total sample size permits it. Remaining positions are distributed using
    a largest-remainder approach, respecting each stratum's capacity.
    """
    non_empty = {
        stratum: size
        for stratum, size in stratum_sizes.items()
        if size > 0
    }

    total_population = sum(non_empty.values())

    if sample_size >= total_population:
        return dict(non_empty)

    allocations = {stratum: 0 for stratum in non_empty}

    # Guarantee representation of each occupied band when possible.
    if sample_size >= len(non_empty):
        for stratum in non_empty:
            allocations[stratum] = 1
    else:
        # This is unlikely with the default cap, but handle it deterministically:
        # select the largest strata first.
        largest = sorted(
            non_empty,
            key=lambda key: (-non_empty[key], key),
        )[:sample_size]
        for stratum in largest:
            allocations[stratum] = 1
        return allocations

    remaining = sample_size - sum(allocations.values())

    while remaining > 0:
        capacities = {
            stratum: non_empty[stratum] - allocations[stratum]
            for stratum in non_empty
            if allocations[stratum] < non_empty[stratum]
        }

        if not capacities:
            break

        capacity_total = sum(capacities.values())
        ideals = {
            stratum: remaining * (capacity / capacity_total)
            for stratum, capacity in capacities.items()
        }

        floor_added = 0
        for stratum, ideal in ideals.items():
            addition = min(capacities[stratum], math.floor(ideal))
            allocations[stratum] += addition
            floor_added += addition

        remaining -= floor_added

        if remaining <= 0:
            break

        # Assign leftover positions by largest fractional remainder.
        remainder_order = sorted(
            capacities,
            key=lambda stratum: (
                -(ideals[stratum] - math.floor(ideals[stratum])),
                -capacities[stratum],
                stratum,
            ),
        )

        assigned_this_round = 0
        for stratum in remainder_order:
            if remaining <= 0:
                break
            if allocations[stratum] >= non_empty[stratum]:
                continue

            allocations[stratum] += 1
            remaining -= 1
            assigned_this_round += 1

        if assigned_this_round == 0:
            break

    return allocations


def allocate_balanced(
    stratum_sizes: dict[str, int],
    sample_size: int,
) -> dict[str, int]:
    """
    Allocate sample positions as evenly as possible across occupied strata.

    Use this mode when the objective is explicitly to compare low-, medium-,
    and high-evolution repositories rather than estimate their natural
    population distribution.
    """
    non_empty = {
        stratum: size
        for stratum, size in stratum_sizes.items()
        if size > 0
    }

    total_population = sum(non_empty.values())

    if sample_size >= total_population:
        return dict(non_empty)

    allocations = {stratum: 0 for stratum in non_empty}
    order = sorted(non_empty)

    while sum(allocations.values()) < sample_size:
        made_progress = False

        for stratum in order:
            if sum(allocations.values()) >= sample_size:
                break

            if allocations[stratum] < non_empty[stratum]:
                allocations[stratum] += 1
                made_progress = True

        if not made_progress:
            break

    return allocations


def build_eligibility(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_commits: int,
    require_commit_status_success: bool,
    require_updated_in_window: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str]]:
    """Apply eligibility rules and return eligible rows plus a full audit table."""
    work = df.copy()

    work["_source_row_number"] = np.arange(2, len(work) + 2)

    work["repo_slug"] = work["original_link"].map(github_slug)
    work["template_slug"] = work["target_template_repository"].map(
        github_slug
    )

    work["is_template_parsed"] = parse_boolean_series(work["is_template"])
    work["commit_count"] = pd.to_numeric(
        work["commit_count"],
        errors="coerce",
    )

    for column in ("stars", "forks"):
        if column in work.columns:
            work[column] = pd.to_numeric(
                work[column],
                errors="coerce",
            ).fillna(0)

    work["data_creation_parsed"] = pd.to_datetime(
        work["data_creation"],
        errors="coerce",
        utc=True,
    )

    if "data_update" in work.columns:
        work["data_update_parsed"] = pd.to_datetime(
            work["data_update"],
            errors="coerce",
            utc=True,
        )
    else:
        work["data_update_parsed"] = pd.NaT

    reason_lists: list[list[str]] = []

    for row in work.itertuples(index=False):
        reasons: list[str] = []

        if pd.isna(row.is_template_parsed):
            reasons.append("invalid_is_template")
        elif bool(row.is_template_parsed):
            reasons.append("row_is_template")

        if not row.repo_slug:
            reasons.append("invalid_repository_url")

        if not row.template_slug:
            reasons.append("missing_target_template")

        if (
            row.repo_slug
            and row.template_slug
            and row.repo_slug == row.template_slug
        ):
            reasons.append("repository_equals_template")

        if pd.isna(row.commit_count):
            reasons.append("missing_commit_count")
        elif row.commit_count < min_commits:
            reasons.append(f"fewer_than_{min_commits}_commits")

        if pd.isna(row.data_creation_parsed):
            reasons.append("invalid_creation_date")
        else:
            if row.data_creation_parsed < start_date:
                reasons.append("created_before_window")
            #if row.data_creation_parsed > end_date:
             #   reasons.append("created_after_window")

        if require_commit_status_success and "commit_info_status" in work.columns:
            status = str(getattr(row, "commit_info_status", "") or "").strip().lower()
            if status != "success":
                reasons.append("commit_info_not_success")

        if require_updated_in_window:
            update_date = row.data_update_parsed
            if pd.isna(update_date):
                reasons.append("invalid_update_date")
            else:
                if update_date < start_date:
                    reasons.append("updated_before_window")
                if update_date > end_date:
                    reasons.append("updated_after_window")

        reason_lists.append(reasons)

    work["exclusion_reasons"] = [
        "|".join(reasons) for reasons in reason_lists
    ]
    work["eligible"] = work["exclusion_reasons"].eq("")

    # De-duplicate repository URLs after the substantive filters.
    duplicate_mask = (
        work["eligible"]
        & work.duplicated(subset=["repo_slug"], keep="first")
    )

    work.loc[duplicate_mask, "eligible"] = False
    work.loc[duplicate_mask, "exclusion_reasons"] = (
        work.loc[duplicate_mask, "exclusion_reasons"]
        .replace("", "duplicate_repository_url")
    )

    reason_counts: Counter[str] = Counter()
    for reasons in work.loc[~work["eligible"], "exclusion_reasons"]:
        for reason in str(reasons).split("|"):
            if reason:
                reason_counts[reason] += 1

    eligible = work.loc[work["eligible"]].copy()

    eligible["evolution_band"] = eligible["commit_count"].map(
        evolution_band
    )

    return eligible, work, reason_counts


def sample_template_family(
    family: pd.DataFrame,
    template_slug: str,
    max_per_template: int,
    allocation: str,
    seed: int,
) -> pd.DataFrame:
    """Select repositories from one template family."""
    family = family.copy()
    family_population = len(family)

    band_population = (
        family["evolution_band"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    target_size = min(family_population, max_per_template)

    if family_population <= max_per_template:
        selected = family.copy()
        allocation_counts = band_population
        selection_method = "all_eligible_in_family"
    else:
        if allocation == "proportional":
            allocation_counts = allocate_proportionally(
                band_population,
                target_size,
            )
        elif allocation == "balanced":
            allocation_counts = allocate_balanced(
                band_population,
                target_size,
            )
        else:
            raise ValueError(f"Unsupported allocation mode: {allocation}")

        selected_parts: list[pd.DataFrame] = []

        for band, number_to_select in allocation_counts.items():
            band_rows = family.loc[
                family["evolution_band"] == band
            ]

            random_state = stable_random_state(
                seed,
                template_slug,
                band,
            )

            selected_parts.append(
                band_rows.sample(
                    n=number_to_select,
                    replace=False,
                    random_state=random_state,
                )
            )

        selected = pd.concat(
            selected_parts,
            ignore_index=False,
        )
        selection_method = f"stratified_random_{allocation}"

    selected = selected.copy()

    selected["family_population_n"] = family_population
    selected["family_sample_n"] = len(selected)
    selected["sampling_method"] = selection_method
    selected["sampling_seed"] = seed

    selected["stratum_population_n"] = selected[
        "evolution_band"
    ].map(band_population)

    stratum_sample_counts = (
        selected["evolution_band"]
        .value_counts()
        .to_dict()
    )

    selected["stratum_sample_n"] = selected[
        "evolution_band"
    ].map(stratum_sample_counts)

    selected["selection_probability"] = (
        selected["stratum_sample_n"]
        / selected["stratum_population_n"]
    )

    selected["sampling_weight"] = (
        1.0 / selected["selection_probability"]
    )

    return selected


def create_template_report(
    eligible: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Produce one summary row per template family."""
    bands = [
        "low_2_to_6",
        "medium_7_to_20",
        "high_21_plus",
    ]

    eligible_counts = (
        eligible.groupby(["template_slug", "evolution_band"])
        .size()
        .unstack(fill_value=0)
    )

    selected_counts = (
        selected.groupby(["template_slug", "evolution_band"])
        .size()
        .unstack(fill_value=0)
    )

    report = pd.DataFrame(
        {
            "eligible_repository_count": (
                eligible.groupby("template_slug").size()
            ),
            "selected_repository_count": (
                selected.groupby("template_slug").size()
            ),
        }
    ).fillna(0)

    for band in bands:
        report[f"eligible_{band}"] = eligible_counts.get(band, 0)
        report[f"selected_{band}"] = selected_counts.get(band, 0)

    report["sampling_fraction"] = (
        report["selected_repository_count"]
        / report["eligible_repository_count"]
    )

    report = report.reset_index().sort_values(
        ["eligible_repository_count", "template_slug"],
        ascending=[False, True],
    )

    integer_columns = [
        column
        for column in report.columns
        if column.endswith("_count")
        or column.startswith("eligible_")
        or column.startswith("selected_")
    ]

    for column in integer_columns:
        if column in report.columns:
            report[column] = report[column].astype(int)

    return report


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""
    safe = df.copy()

    for column in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[column]):
            safe[column] = safe[column].dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )

    safe = safe.replace({np.nan: None})
    return safe.to_dict(orient="records")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible, template-stratified sample for "
            "SonarQube issue-persistence analysis."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="CSV containing generated repositories.",
    )

    parser.add_argument(
        "output_directory",
        type=Path,
        help="Directory in which sampling outputs will be saved.",
    )

    parser.add_argument(
        "--start-date",
        default="2023-10-18",
        help=(
            "Beginning of the repository-creation window, inclusive. "
            "Default: 2023-10-18"
        ),
    )

    parser.add_argument(
        "--end-date",
        default="2025-10-18",
        help=(
            "End of the repository-creation window, inclusive. "
            "Default: 2025-10-18"
        ),
    )

    parser.add_argument(
        "--min-commits",
        type=int,
        default=2,
        help="Minimum commit count. Default: 2",
    )

    parser.add_argument(
        "--max-per-template",
        type=int,
        default=20,
        help=(
            "Maximum generated repositories selected per template. "
            "Default: 20"
        ),
    )

    parser.add_argument(
        "--allocation",
        choices=["proportional", "balanced"],
        default="proportional",
        help=(
            "How capped template families are allocated across commit-count "
            "bands. 'proportional' estimates the natural population; "
            "'balanced' emphasizes comparisons among evolution levels. "
            "Default: proportional"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20251018,
        help="Random seed used for reproducible sampling.",
    )

    parser.add_argument(
        "--allow-unsuccessful-commit-info",
        action="store_true",
        help=(
            "Do not require commit_info_status == success when that column "
            "is available."
        ),
    )

    parser.add_argument(
        "--require-updated-in-window",
        action="store_true",
        help=(
            "Additionally require data_update to fall inside the observation "
            "window. This is not recommended for the primary persistence "
            "sample because it removes quickly abandoned repositories."
        ),
    )

    parser.add_argument(
        "--write-eligibility-audit",
        action="store_true",
        help=(
            "Write eligibility_audit.csv containing every input row and its "
            "exclusion reasons."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.min_commits < 1:
        raise SystemExit("--min-commits must be at least 1.")

    if args.max_per_template < 1:
        raise SystemExit("--max-per-template must be at least 1.")

    input_path = args.input_csv.expanduser().resolve()
    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError as error:
        raise SystemExit(f"Input CSV not found: {input_path}") from error
    except Exception as error:
        raise SystemExit(f"Could not read {input_path}: {error}") from error

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise SystemExit(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    try:
        start_date = pd.Timestamp(args.start_date, tz="UTC")
        end_date = pd.Timestamp(args.end_date, tz="UTC")
    except Exception as error:
        raise SystemExit(f"Invalid date argument: {error}") from error

    # Include the complete end date when a date without time is supplied.
    if len(str(args.end_date).strip()) <= 10:
        end_date = end_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)

    if start_date > end_date:
        raise SystemExit("--start-date must not be after --end-date.")

    eligible, audit, exclusion_counts = build_eligibility(
        df=df,
        start_date=start_date,
        end_date=end_date,
        min_commits=args.min_commits,
        require_commit_status_success=(
            not args.allow_unsuccessful_commit_info
        ),
        require_updated_in_window=args.require_updated_in_window,
    )

    if eligible.empty:
        raise SystemExit(
            "No repositories satisfy the eligibility criteria."
        )

    selected_families: list[pd.DataFrame] = []

    for template_slug, family in eligible.groupby(
        "template_slug",
        sort=True,
    ):
        selected_families.append(
            sample_template_family(
                family=family,
                template_slug=template_slug,
                max_per_template=args.max_per_template,
                allocation=args.allocation,
                seed=args.seed,
            )
        )

    selected = pd.concat(
        selected_families,
        ignore_index=True,
    )

    selected = selected.sort_values(
        ["template_slug", "evolution_band", "repo_slug"],
        kind="stable",
    ).reset_index(drop=True)

    eligible = eligible.sort_values(
        ["template_slug", "evolution_band", "repo_slug"],
        kind="stable",
    ).reset_index(drop=True)

    template_report = create_template_report(
        eligible=eligible,
        selected=selected,
    )

    sample_output = output_directory / "selected_sample.csv"
    eligible_output = output_directory / "eligible_candidates.csv"
    template_report_output = (
        output_directory / "template_sampling_report.csv"
    )
    summary_output = output_directory / "sampling_summary.json"
    exclusions_output = (
        output_directory / "exclusion_reason_counts.csv"
    )

    # Keep original columns and append audit/sampling fields.
    selected.to_csv(sample_output, index=False)
    eligible.to_csv(eligible_output, index=False)
    template_report.to_csv(template_report_output, index=False)

    pd.DataFrame(
        [
            {"exclusion_reason": reason, "row_count": count}
            for reason, count in sorted(exclusion_counts.items())
        ]
    ).to_csv(exclusions_output, index=False)

    if args.write_eligibility_audit:
        audit.to_csv(
            output_directory / "eligibility_audit.csv",
            index=False,
        )

    capped_family_count = int(
        (
            template_report["eligible_repository_count"]
            > args.max_per_template
        ).sum()
    )

    summary = {
        "input_file": str(input_path),
        "input_row_count": int(len(df)),
        "observation_window": {
            "start_inclusive": start_date.isoformat(),
            "end_inclusive": end_date.isoformat(),
        },
        "eligibility_rules": {
            "exclude_template_rows": True,
            "minimum_commits": args.min_commits,
            "require_valid_repository_and_template": True,
            "exclude_repository_equal_to_template": True,
            "require_creation_inside_window": True,
            "require_commit_info_status_success": (
                not args.allow_unsuccessful_commit_info
                and "commit_info_status" in df.columns
            ),
            "require_update_inside_window": (
                args.require_updated_in_window
            ),
            "remove_duplicate_repository_urls": True,
        },
        "sampling_rules": {
            "all_template_families_represented": True,
            "maximum_per_template": args.max_per_template,
            "allocation": args.allocation,
            "evolution_bands": {
                "low_2_to_6": "2-6 commits",
                "medium_7_to_20": "7-20 commits",
                "high_21_plus": "21 or more commits",
            },
            "random_seed": args.seed,
        },
        "eligible_repository_count": int(len(eligible)),
        "eligible_template_family_count": int(
            eligible["template_slug"].nunique()
        ),
        "selected_repository_count": int(len(selected)),
        "selected_template_family_count": int(
            selected["template_slug"].nunique()
        ),
        "fully_enumerated_family_count": int(
            len(template_report) - capped_family_count
        ),
        "capped_family_count": capped_family_count,
        "overall_sampling_fraction": float(
            len(selected) / len(eligible)
        ),
        "eligible_by_evolution_band": {
            str(key): int(value)
            for key, value in eligible[
                "evolution_band"
            ].value_counts().sort_index().items()
        },
        "selected_by_evolution_band": {
            str(key): int(value)
            for key, value in selected[
                "evolution_band"
            ].value_counts().sort_index().items()
        },
        "exclusion_reason_counts": dict(
            sorted(exclusion_counts.items())
        ),
        "outputs": {
            "selected_sample": str(sample_output),
            "eligible_candidates": str(eligible_output),
            "template_sampling_report": str(
                template_report_output
            ),
            "exclusion_reason_counts": str(exclusions_output),
        },
    }

    summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSampling completed")
    print("=" * 60)
    print(f"Input rows:                 {len(df)}")
    print(f"Eligible repositories:      {len(eligible)}")
    print(
        "Eligible template families: "
        f"{eligible['template_slug'].nunique()}"
    )
    print(f"Selected repositories:      {len(selected)}")
    print(
        "Selected template families: "
        f"{selected['template_slug'].nunique()}"
    )
    print(f"Families capped:             {capped_family_count}")
    print(
        "Overall sampling fraction:  "
        f"{len(selected) / len(eligible):.3f}"
    )
    print("\nOutputs")
    print(f"  {sample_output}")
    print(f"  {eligible_output}")
    print(f"  {template_report_output}")
    print(f"  {summary_output}")
    print(f"  {exclusions_output}")


if __name__ == "__main__":
    main()