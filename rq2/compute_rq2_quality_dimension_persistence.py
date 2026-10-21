#!/usr/bin/env python3
"""
compute_rq2_quality_dimension_persistence.py

Compute descriptive RQ2 persistence rates directly from RQ2 pair_summary.json
files.

Dimensions:
- issue_type <- breakdown_by_type
- severity <- breakdown_by_severity
- software_quality <- breakdown_by_software_quality

Weighted micro persistence:

    sum(persistent_issue_count * sampling_weight)
    ------------------------------------------------ * 100
    sum(template_issue_count   * sampling_weight)

Missing/null/non-positive sampling weights are treated as 1.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DIMENSION_KEYS = {
    "issue_type": "breakdown_by_type",
    "severity": "breakdown_by_severity",
    "software_quality": "breakdown_by_software_quality",
}

KNOWN_LANGUAGE_NAMES = {
    "c#": "C#",
    "csharp": "C#",
    "c-sharp": "C#",
    "java": "Java",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "python": "Python",
    "typescript": "TypeScript",
    "ts": "TypeScript",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute descriptive RQ2 quality-issue persistence by issue type, "
            "severity, and software quality from pair_summary.json files."
        )
    )
    parser.add_argument(
        "--rq2-root",
        type=Path,
        default=None,
        help="Root directory containing RQ2 outputs; searched recursively.",
    )
    parser.add_argument(
        "--pair-summary-json",
        action="append",
        type=Path,
        default=[],
        help="Explicit pair_summary.json path. May be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rq2_quality_dimension_results"),
    )
    return parser.parse_args()


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

    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)

    if not unique:
        raise FileNotFoundError(
            "No pair_summary.json files found. Use --rq2-root and/or "
            "--pair-summary-json."
        )

    return unique


def load_pair_summary(path: Path) -> list[dict[str, Any]]:
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
                f"{path}: expected a JSON list or a wrapper containing "
                "'pairs', 'results', or 'data'."
            )
    else:
        raise ValueError(f"{path}: unsupported JSON structure.")

    return [row for row in rows if isinstance(row, dict)]


def infer_language_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        key = part.strip().lower()
        if key in KNOWN_LANGUAGE_NAMES:
            return KNOWN_LANGUAGE_NAMES[key]
    return None


def get_language(record: dict[str, Any], source_path: Path) -> str:
    for key in ("language", "programming_language", "repository_language"):
        value = normalize_text(record.get(key))
        if value:
            return KNOWN_LANGUAGE_NAMES.get(value.lower(), value)

    inferred = infer_language_from_path(source_path)
    return inferred if inferred else "Unknown"


def to_nonnegative_int(value: Any) -> int:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return 0
    return max(0, int(numeric))


def get_sampling_weight(record: dict[str, Any]) -> float:
    value = pd.to_numeric(record.get("sampling_weight"), errors="coerce")
    if pd.isna(value) or value <= 0:
        return 1.0
    return float(value)


def build_long_dataset(json_paths: list[Path]) -> pd.DataFrame:
    rows = []
    seen_pairs: dict[str, dict[str, Any]] = {}

    for path in json_paths:
        for record in load_pair_summary(path):
            pair_id = normalize_text(record.get("pair_id"))
            template_repo = (
                record.get("template_repository")
                or record.get("template_id")
            )
            generated_repo = (
                record.get("generated_repository")
                or record.get("project_id")
            )

            template_norm = normalize_repo(template_repo)
            generated_norm = normalize_repo(generated_repo)

            if pair_id:
                pair_key = pair_id.casefold()
            elif template_norm and generated_norm:
                pair_key = f"{template_norm}__to__{generated_norm}"
            else:
                raise ValueError(f"{path}: could not identify one pair.")

            weight = get_sampling_weight(record)
            language = get_language(record, path)

            signature = {
                "template_repository": template_norm,
                "generated_repository": generated_norm,
                "sampling_weight": weight,
                "breakdown_by_type": record.get("breakdown_by_type") or {},
                "breakdown_by_severity": record.get("breakdown_by_severity") or {},
                "breakdown_by_software_quality": record.get(
                    "breakdown_by_software_quality"
                ) or {},
            }

            if pair_key in seen_pairs:
                if seen_pairs[pair_key] != signature:
                    raise ValueError(
                        f"Conflicting duplicate pair summary for {pair_key}"
                    )
                continue

            seen_pairs[pair_key] = signature

            for dimension, json_key in DIMENSION_KEYS.items():
                breakdown = record.get(json_key) or {}

                for category_raw, values in breakdown.items():
                    if not isinstance(values, dict):
                        continue

                    category = str(category_raw).strip().upper() or "UNKNOWN"

                    template_count = to_nonnegative_int(
                        values.get("template_issue_count", 0)
                    )
                    project_count = to_nonnegative_int(
                        values.get("project_issue_count", 0)
                    )
                    persistent = to_nonnegative_int(
                        values.get("persistent_issue_count", 0)
                    )
                    not_persistent = to_nonnegative_int(
                        values.get("no_longer_present_issue_count", 0)
                    )
                    new_count = to_nonnegative_int(
                        values.get("new_issue_count", 0)
                    )

                    if persistent + not_persistent != template_count:
                        raise ValueError(
                            f"{path}: pair={pair_key}, {dimension}={category}: "
                            "persistent + not_persistent != template_issue_count"
                        )

                    if template_count <= 0:
                        continue

                    rows.append({
                        "pair_id": pair_key,
                        "language": language,
                        "template_repository": template_repo,
                        "generated_repository": generated_repo,
                        "dimension": dimension,
                        "category": category,
                        "sampling_weight": weight,
                        "template_issue_count": template_count,
                        "project_issue_count": project_count,
                        "persistent_issue_count": persistent,
                        "not_persistent_issue_count": not_persistent,
                        "new_issue_count": new_count,
                        "pair_persistence_rate": persistent / template_count,
                        "weighted_template_issue_count": template_count * weight,
                        "weighted_persistent_issue_count": persistent * weight,
                    })

    if not rows:
        raise ValueError("No persistence observations found.")

    return pd.DataFrame(rows)


def aggregate_micro(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    output_rows = []

    for keys, group in df.groupby(
        group_cols,
        observed=False,
        dropna=False,
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_values = dict(zip(group_cols, keys))

        raw_trials = group["template_issue_count"].sum()
        raw_persistent = group["persistent_issue_count"].sum()

        weighted_trials = group["weighted_template_issue_count"].sum()
        weighted_persistent = group[
            "weighted_persistent_issue_count"
        ].sum()

        output_rows.append({
            **key_values,
            "pair_category_rows": len(group),
            "source_templates": group["template_repository"].nunique(),
            "generated_repositories": group["generated_repository"].nunique(),
            "raw_template_issue_count": int(raw_trials),
            "raw_persistent_issue_count": int(raw_persistent),
            "raw_persistence_percentage": (
                100.0 * raw_persistent / raw_trials
                if raw_trials > 0 else np.nan
            ),
            "weighted_template_issue_count": weighted_trials,
            "weighted_persistent_issue_count": weighted_persistent,
            "weighted_persistence_percentage": (
                100.0 * weighted_persistent / weighted_trials
                if weighted_trials > 0 else np.nan
            ),
        })

    return pd.DataFrame(output_rows)


def pair_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (dimension, category), group in df.groupby(
        ["dimension", "category"],
        observed=False,
    ):
        rates = group["pair_persistence_rate"].dropna()

        rows.append({
            "dimension": dimension,
            "category": category,
            "pair_category_rows": len(group),
            "source_templates": group["template_repository"].nunique(),
            "mean_pair_persistence_percentage": (
                100.0 * rates.mean() if not rates.empty else np.nan
            ),
            "median_pair_persistence_percentage": (
                100.0 * rates.median() if not rates.empty else np.nan
            ),
            "q1_pair_persistence_percentage": (
                100.0 * rates.quantile(0.25) if not rates.empty else np.nan
            ),
            "q3_pair_persistence_percentage": (
                100.0 * rates.quantile(0.75) if not rates.empty else np.nan
            ),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths = discover_pair_summary_jsons(
        args.rq2_root,
        args.pair_summary_json,
    )

    print("Discovered pair_summary.json files:")
    for path in json_paths:
        print(f"  - {path}")

    long_df = build_long_dataset(json_paths)

    overall = aggregate_micro(
        long_df,
        ["dimension", "category"],
    ).sort_values(["dimension", "category"])

    by_language = aggregate_micro(
        long_df,
        ["language", "dimension", "category"],
    ).sort_values(["language", "dimension", "category"])

    pair_summary = pair_level_summary(long_df).sort_values(
        ["dimension", "category"]
    )

    long_df.to_csv(
        output_dir / "rq2_quality_dimension_long.csv",
        index=False,
    )
    overall.to_csv(
        output_dir / "rq2_quality_dimension_persistence_overall.csv",
        index=False,
    )
    by_language.to_csv(
        output_dir / "rq2_quality_dimension_persistence_by_language.csv",
        index=False,
    )
    pair_summary.to_csv(
        output_dir / "rq2_quality_dimension_pair_level_summary.csv",
        index=False,
    )

    metadata = [
        "RQ2 quality-dimension persistence aggregation",
        "",
        f"pair_summary.json files: {len(json_paths)}",
        f"unique pairs: {long_df['pair_id'].nunique()}",
        f"unique source templates: {long_df['template_repository'].nunique()}",
        (
            "unique generated repositories: "
            f"{long_df['generated_repository'].nunique()}"
        ),
        "",
        "Dimensions:",
        "  issue_type <- breakdown_by_type",
        "  severity <- breakdown_by_severity",
        "  software_quality <- breakdown_by_software_quality",
        "",
        "Weighted micro persistence:",
        (
            "  sum(persistent_issue_count * sampling_weight) / "
            "sum(template_issue_count * sampling_weight) * 100"
        ),
        "",
        "Missing/null/non-positive sampling weights are treated as 1.0.",
        (
            "Software-quality classifications may overlap, so their counts "
            "should not be summed to reconstruct unique issue totals."
        ),
        (
            "SECURITY is retained as the SonarQube software-quality "
            "classification and is not renamed SECURITY_HOTSPOT."
        ),
    ]

    (output_dir / "rq2_quality_dimension_metadata.txt").write_text(
        "\n".join(metadata) + "\n",
        encoding="utf-8",
    )

    print("\nOVERALL RQ2 QUALITY-DIMENSION PERSISTENCE")
    print("=" * 100)
    print(
        overall[
            [
                "dimension",
                "category",
                "pair_category_rows",
                "source_templates",
                "raw_template_issue_count",
                "weighted_persistence_percentage",
            ]
        ].to_string(index=False)
    )

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
