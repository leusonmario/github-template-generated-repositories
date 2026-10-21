#!/usr/bin/env python3
"""Analyze template-generated repositories that have themselves become templates.

Only repositories with valid commit_count (positive, finite integers) are
included in prevalence and activity statistics. Duplicate records are resolved
with preference for valid commit_count values. Source-template reference links
may come from any record with a valid repository identity, even when its own
commit_count is missing: such a record still documents a repository reference.

Reads generated_GHRep_from_templates_*.csv recursively. Does not modify inputs.

Usage:
    python -u analyze_generated_templates.py reports \
      --output_dir rq1_generated_template_reuse

Requires: pandas, numpy
"""
import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

PATTERN = "generated_GHRep_from_templates_*.csv"
PREFIX = "generated_GHRep_from_templates_"


def slug(value):
    """Normalize a GitHub URL or owner/repo to a lowercase owner/repo key."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "none", "null"}:
        return None
    if text.startswith("git@github.com:"):
        text = text[len("git@github.com:"):]
    elif text.lower().startswith(("http://", "https://")):
        parsed = urlparse(text)
        if (parsed.hostname or "").lower() not in {"github.com", "www.github.com"}:
            return None
        text = parsed.path
    elif text.lower().startswith(("github.com/", "www.github.com/")):
        text = text.split("/", 1)[1]
    text = text.strip("/").split("?", 1)[0].split("#", 1)[0]
    parts = text.split("/")
    if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
        return None
    owner, repo = parts[0].strip(), parts[1].strip()
    if repo.casefold().endswith(".git"):
        repo = repo[:-4]
    if not repo:
        return None
    return f"{owner}/{repo}".casefold()


def flag(value):
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False
    return None


def language_from_filename(path):
    suffix = path.stem[len(PREFIX):]
    if suffix.endswith("_with_commits"):
        suffix = suffix[:-len("_with_commits")]
    aliases = {
        "csharp": "C#", "c#": "C#", "java": "Java", "javascript": "JavaScript",
        "typescript": "TypeScript", "python": "Python",
    }
    return aliases.get(suffix.casefold(), suffix)


def load_rows(folder):
    paths = sorted(p for p in folder.rglob(PATTERN) if p.is_file())
    if not paths:
        raise SystemExit(f"No files matched {PATTERN!r} under {folder}")
    data = []
    for path in paths:
        print(f"Reading {path}", flush=True)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"username", "project_name", "is_template", "target_template_repository", "commit_count"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        df["_language"] = language_from_filename(path)
        df["_source_file"] = str(path.relative_to(folder))
        df["_source_row"] = df.index + 2

        names = df["username"].str.strip() + "/" + df["project_name"].str.strip()
        df["_name_key"] = names.map(slug)
        df["_url_key"] = df["original_link"].map(slug) if "original_link" in df.columns else None
        df["_repo_key"] = df["_url_key"].combine_first(df["_name_key"])
        df["_template_key"] = df["target_template_repository"].map(slug)
        df["_flag"] = df["is_template"].map(flag)

        # In particular, 0, negative, decimal, missing and non-finite counts
        # cannot be used in the RQ1 post-creation activity classification.
        numeric = pd.to_numeric(df["commit_count"].str.strip(), errors="coerce")
        df["_commit_count_numeric"] = numeric
        df["_valid_commit_count"] = (
            numeric.notna() & np.isfinite(numeric)
            & (numeric >= 1) & (numeric % 1 == 0)
        )

        df["_updated"] = pd.to_datetime(
            df["data_update"] if "data_update" in df.columns
            else pd.Series("", index=df.index), errors="coerce", utc=True,
        )
        completeness_fields = [
            c for c in ["original_link", "commit_count", "data_creation", "data_update",
                      "last_commit_datetime", "target_template_repository"] if c in df.columns
        ]
        df["_completeness"] = (
            df[completeness_fields].apply(lambda col: col.astype(str).str.strip().ne("")).sum(axis=1)
            if completeness_fields else 0
        )
        data.append(df)
    return pd.concat(data, ignore_index=True), paths


def deduplicate(df, group_cols):
    """Keep best row per identity, propagating contradictory True/False flags.

    Prefer a row with valid commit_count, then latest update and completeness.
    A conflict in either deduplication pass remains unknown thereafter.
    """
    if df.empty:
        return df.copy(), df.copy()
    working = df.copy()
    if "_flag_conflict" not in working.columns:
        working["_flag_conflict"] = False
    if "_occurrences" not in working.columns:
        working["_occurrences"] = 1

    working = working.sort_values(
        ["_valid_commit_count", "_updated", "_completeness"],
        ascending=[False, False, False], na_position="last", kind="stable",
    )
    working["_seen_true"] = working["_flag"].eq(True)
    working["_seen_false"] = working["_flag"].eq(False)

    grouped = working.groupby(group_cols, sort=False, dropna=False)
    aggregate = grouped.agg(
        _occurrences=("_occurrences", "sum"),
        _seen_true=("_seen_true", "any"),
        _seen_false=("_seen_false", "any"),
        _previous_conflict=("_flag_conflict", "any"),
    ).reset_index()
    aggregate["_flag_conflict"] = (
        (aggregate["_seen_true"] & aggregate["_seen_false"])
        | aggregate["_previous_conflict"]
    )
    unique = working.drop_duplicates(subset=group_cols, keep="first").copy()
    unique = unique.drop(columns=["_occurrences", "_flag_conflict", "_seen_true", "_seen_false"])
    unique = unique.merge(
        aggregate[group_cols + ["_occurrences", "_flag_conflict"]],
        on=group_cols, how="left", validate="one_to_one", sort=False,
    )
    unique["_flag"] = unique["_flag"].astype(object)
    unique.loc[unique["_flag_conflict"], "_flag"] = None

    # Audit rows in newly detected conflicts. Conflicts inherited from the
    # previous pass remain marked unknown in the chosen row.
    conflicting_keys = aggregate.loc[
        aggregate["_seen_true"] & aggregate["_seen_false"], group_cols
    ]
    conflict_rows = (
        working.merge(conflicting_keys, on=group_cols, how="inner")
        if not conflicting_keys.empty else df.iloc[0:0].copy()
    )
    conflict_rows = conflict_rows.drop(columns=["_seen_true", "_seen_false"], errors="ignore")
    return unique, conflict_rows


def summarize(df, label, downstream, invalid_count=0):
    true_mask = df["_flag"].eq(True)
    false_mask = df["_flag"].eq(False)
    observed = df["_repo_key"].map(lambda key: len(downstream.get(key, set())))
    total = len(df)
    true_count = int(true_mask.sum())
    false_count = int(false_mask.sum())
    known = true_count + false_count
    flagged = df.loc[true_mask]
    commits = flagged["_commit_count_numeric"]

    def count_in(mask):
        return int(mask.sum())

    return {
        "language": label,
        "unique_generated_repositories": total,
        "excluded_unique_repositories_invalid_commit_count": invalid_count,
        "is_template_true": true_count,
        "is_template_false": false_count,
        "is_template_unknown_or_conflicting": total - known,
        "true_percentage_of_all": round(true_count / total * 100, 2) if total else 0,
        "true_percentage_of_known": round(true_count / known * 100, 2) if known else None,
        "marked_templates_one_commit": count_in(commits.eq(1)),
        "marked_templates_2_to_6_commits": count_in(commits.between(2, 6)),
        "marked_templates_7_to_20_commits": count_in(commits.between(7, 20)),
        "marked_templates_over_20_commits": count_in(commits.gt(20)),
        "marked_templates_same_source_owner": int(flagged["_same_owner"].eq(True).sum()),
        "marked_templates_different_source_owner": int(flagged["_same_owner"].eq(False).sum()),
        "marked_templates_referenced_as_source": int((true_mask & observed.gt(0)).sum()),
        "marked_templates_not_observed_as_source": int((true_mask & observed.eq(0)).sum()),
        "all_generated_repos_referenced_as_source": int(observed.gt(0).sum()),
        "extra_duplicate_rows_removed": int((df["_occurrences"] - 1).clip(lower=0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Folder containing generated CSV files")
    parser.add_argument("--output_dir", type=Path, default=Path("rq1_generated_template_reuse"))
    args = parser.parse_args()
    folder = args.input_dir.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"Input directory does not exist: {folder}")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw, paths = load_rows(folder)
    invalid_ids = raw[raw["_repo_key"].isna()].copy()
    mismatches = raw[
        raw["_url_key"].notna() & raw["_name_key"].notna()
        & raw["_url_key"].ne(raw["_name_key"])
    ].copy()
    valid_ids = raw.dropna(subset=["_repo_key"]).copy()

    # Deduplicate before applying the commit-count restriction, so when a
    # duplicate has a valid count and another has a missing one we keep the
    # usable record. Conflicting template flags remain conservatively unknown.
    by_language_all, flag_conflicts = deduplicate(valid_ids, ["_language", "_repo_key"])
    overall_all, cross_language_conflicts = deduplicate(by_language_all, ["_repo_key"])

    invalid_by_language = by_language_all[~by_language_all["_valid_commit_count"]].copy()
    by_language = by_language_all[by_language_all["_valid_commit_count"]].copy()
    invalid_overall = overall_all[~overall_all["_valid_commit_count"]].copy()
    overall = overall_all[overall_all["_valid_commit_count"]].copy()

    for data in [by_language, overall]:
        source_owner = data["_template_key"].map(
            lambda value: value.split("/", 1)[0] if isinstance(value, str) else None
        )
        repo_owner = data["_repo_key"].map(
            lambda value: value.split("/", 1)[0] if isinstance(value, str) else None
        )
        data["_same_owner"] = (source_owner == repo_owner).where(source_owner.notna(), None)

    # A child with unknown commit_count can still be an observed generated
    # repository. Use all identified rows to detect source-template references;
    # the intermediate template itself must have a valid commit_count.
    edges = by_language_all[["_template_key", "_repo_key"]].dropna().drop_duplicates()
    downstream = defaultdict(set)
    for source_template, generated_repo in edges.itertuples(index=False, name=None):
        if source_template != generated_repo:
            downstream[source_template].add(generated_repo)

    invalid_counts = invalid_by_language.groupby("_language").size().to_dict()
    summary_rows = [
        summarize(part, lang, downstream, invalid_counts.get(lang, 0))
        for lang, part in by_language.groupby("_language", sort=True)
    ]
    # Languages having only invalid counts should still be shown.
    analyzed_langs = {row["language"] for row in summary_rows}
    for lang, group in invalid_by_language.groupby("_language", sort=True):
        if lang not in analyzed_langs:
            summary_rows.append(summarize(by_language.iloc[0:0], lang, downstream, len(group)))
    summary_rows.append(summarize(overall, "Overall (unique GitHub repositories)", downstream, len(invalid_overall)))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "generated_template_occurrence_summary.csv", index=False)

    flagged = by_language.loc[by_language["_flag"].eq(True)].copy()
    flagged.insert(0, "observed_as_source_in_dataset",
                   flagged["_repo_key"].map(lambda k: bool(downstream.get(k))))
    flagged.insert(1, "observed_downstream_repository_count",
                   flagged["_repo_key"].map(lambda k: len(downstream.get(k, set()))))
    flagged.insert(2, "same_owner_as_source_template", flagged["_same_owner"])
    flagged.insert(3, "activity_level", flagged["_commit_count_numeric"].map(
        lambda n: "No post-creation activity" if n == 1 else
                  "Low activity" if n <= 6 else
                  "Moderate activity" if n <= 20 else "High activity"
    ))
    flagged = flagged.rename(columns={
        "_language": "language", "_repo_key": "normalized_repository",
        "_template_key": "normalized_source_template", "_source_file": "source_csv",
    })
    flagged.drop(columns=[c for c in flagged.columns if c.startswith("_")]).to_csv(
        out / "generated_repositories_marked_as_templates.csv", index=False,
    )

    chains = []
    for _, row in flagged.iterrows():
        for child in sorted(downstream.get(row["normalized_repository"], set())):
            chains.append({
                "language_of_intermediate": row["language"],
                "original_template_A": row["normalized_source_template"],
                "generated_template_B": row["normalized_repository"],
                "downstream_generated_repository_C": child,
            })
    pd.DataFrame(chains, columns=[
        "language_of_intermediate", "original_template_A", "generated_template_B",
        "downstream_generated_repository_C",
    ]).to_csv(out / "observed_second_generation_links.csv", index=False)

    audit_columns = [c for c in [
        "_language", "_source_file", "_source_row", "username", "project_name",
        "original_link", "is_template", "target_template_repository", "commit_count",
        "_repo_key", "_url_key", "_name_key",
    ] if c in raw.columns]
    flag_conflicts.reindex(columns=audit_columns).to_csv(
        out / "conflicting_duplicate_template_flags.csv", index=False)
    cross_language_conflicts.reindex(columns=audit_columns).to_csv(
        out / "conflicting_cross_language_flags.csv", index=False)
    mismatches[audit_columns].to_csv(out / "repository_identity_mismatches.csv", index=False)
    invalid_ids[audit_columns].to_csv(out / "missing_repository_identity.csv", index=False)
    invalid_by_language[audit_columns].to_csv(
        out / "excluded_repositories_invalid_commit_count.csv", index=False)

    print("\nRESULTS (repositories with valid commit_count only)")
    print(summary.to_string(index=False))
    print(f"\nFiles read: {len(paths)}; input rows: {len(raw):,}")
    print(f"Rows without usable repository identity: {len(invalid_ids):,}")
    print(f"Unique repositories excluded (invalid commit_count, overall): {len(invalid_overall):,}")
    print(f"Repository name / URL mismatches: {len(mismatches):,}")
    print(f"Rows involved in conflicting flags within language: {len(flag_conflicts):,}")
    print(f"Observed second-generation links: {len(chains):,}")
    print(f"Outputs saved to: {out}")


if __name__ == "__main__":
    main()
