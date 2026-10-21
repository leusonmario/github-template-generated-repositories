#!/usr/bin/env python3
"""RQ1: rank source-template families by post-creation commit activity.

Reads only generated_GHRep_from_templates_*.csv (recursively), optionally
joins all_templates_*.csv to identify source-template ownership.

Definitions:
  one_commit  = exactly 1 commit (no observed subsequent commit)
  low         = 2--6 commits
  moderate    = 7--20 commits
  high        = >20 commits

One observation per generated repository *within each language*. Valid commit
counts are strictly positive integers. Duplicate observations are deduplicated
before ranking. Outputs include full family metrics, rankings by count/rate,
stratified summaries, and repository URLs for a qualitative inspection.

Example, from the rq1/ directory:
  python -u analyze_rq1_template_families.py reports \\
      --templates_dir ../data_collection \\
      --output_dir rq1_family_activity_results \\
      --min_family_size 20 --top 20

Requires: pandas
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

GENERATED_GLOB = "generated_GHRep_from_templates_*.csv"
TEMPLATES_GLOB = "all_templates_*.csv"
GENERATED_PREFIX = "generated_GHRep_from_templates_"
TEMPLATE_PREFIX = "all_templates_"
LANGUAGE_MAP = {
    "c#": "C#", "csharp": "C#", "java": "Java", "python": "Python",
    "typescript": "TypeScript", "javascript": "JavaScript",
}
LANGUAGE_ORDER = ["C#", "Java", "Python", "TypeScript", "JavaScript"]
CATEGORIES = ("one_commit", "low_2_6", "moderate_7_20", "high_gt_20")


def normalize_repo(value):
    """Return lowercase GitHub owner/repository; no suffix or query string."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "null", "none"}:
        return None
    if s.lower().startswith("git@github.com:"):
        s = s[len("git@github.com:"):]
    elif s.lower().startswith(("http://", "https://")):
        parsed = urlparse(s)
        if (parsed.hostname or "").lower() not in {"github.com", "www.github.com"}:
            return None
        s = parsed.path
    elif s.lower().startswith(("github.com/", "www.github.com/")):
        s = s.split("/", 1)[1]
    s = s.split("?", 1)[0].split("#", 1)[0].strip("/")
    parts = s.split("/")
    if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
        return None
    owner, repo = parts[0].strip(), parts[1].strip()
    if repo.lower().endswith(".git"):
        repo = repo[:-4]
    return f"{owner}/{repo}".lower() if repo else None


def language_from_path(path, prefix):
    suffix = path.stem[len(prefix):]
    if suffix.endswith("_with_commits"):
        suffix = suffix[:-len("_with_commits")]
    return LANGUAGE_MAP.get(suffix.casefold(), suffix)


def read_generated(folder):
    paths = sorted(p for p in folder.rglob(GENERATED_GLOB) if p.is_file())
    if not paths:
        raise SystemExit(f"No input files matching {GENERATED_GLOB!r} under {folder}")
    frames = []
    for path in paths:
        print(f"Reading {path}", flush=True)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"username", "project_name", "target_template_repository", "commit_count"}
        if missing := required.difference(df.columns):
            raise ValueError(f"Missing columns {sorted(missing)} in {path}")
        df["language"] = language_from_path(path, GENERATED_PREFIX)
        df["source_csv"] = str(path.relative_to(folder))
        df["source_row"] = df.index + 2
        names = df["username"].str.strip() + "/" + df["project_name"].str.strip()
        df["name_repo"] = names.map(normalize_repo)
        df["url_repo"] = df["original_link"].map(normalize_repo) if "original_link" in df else None
        df["generated_repo"] = df["url_repo"].combine_first(df["name_repo"])
        df["source_template"] = df["target_template_repository"].map(normalize_repo)
        commits = pd.to_numeric(df["commit_count"], errors="coerce")
        df["valid_commits"] = commits.notna() & commits.ge(1) & commits.eq(commits.round(0))
        df["commits_num"] = commits
        df["updated_dt"] = pd.to_datetime(df["data_update"], errors="coerce", utc=True) if "data_update" in df else pd.NaT
        df["creation_dt"] = pd.to_datetime(df["data_creation"], errors="coerce", utc=True) if "data_creation" in df else pd.NaT
        df["last_commit_dt"] = pd.to_datetime(df["last_commit_datetime"], errors="coerce", utc=True) if "last_commit_datetime" in df else pd.NaT
        df["info_success"] = (
            df["commit_info_status"].str.strip().str.lower().isin({"success", "ok", "completed", "complete"})
            if "commit_info_status" in df else False
        )
        df["metadata_filled"] = sum(
            df[col].str.strip().ne("").astype(int)
            for col in ("original_link", "data_update", "data_creation", "last_commit_datetime", "commit_info_status")
            if col in df
        )
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(raw):,} rows from {len(paths)} files", flush=True)
    return raw


def deduplicate_generated(raw):
    """Choose best observed row per (language, repository) before applying filters.

    A conflicting source_template between duplicate rows is exposed for audit.
    """
    missing_ids = raw[raw["generated_repo"].isna()].copy()
    known = raw.dropna(subset=["generated_repo"]).copy()
    known = known.sort_values(
        ["valid_commits", "info_success", "metadata_filled", "updated_dt"],
        ascending=[False, False, False, False], kind="stable", na_position="last",
    )
    keys = ["language", "generated_repo"]
    grouped = known.groupby(keys, dropna=False, sort=False)
    counts = grouped.size().rename("duplicate_occurrences")
    distinct_template_count = grouped["source_template"].nunique(dropna=True).rename("distinct_source_templates")
    distinct_commit_count = grouped["commits_num"].nunique(dropna=True).rename("distinct_commit_counts")
    selected = known.drop_duplicates(keys, keep="first").copy()
    selected = selected.merge(counts.reset_index(), on=keys, how="left", validate="one_to_one")
    selected = selected.merge(distinct_template_count.reset_index(), on=keys, how="left", validate="one_to_one")
    selected = selected.merge(distinct_commit_count.reset_index(), on=keys, how="left", validate="one_to_one")
    conflicts = selected[
        (selected["distinct_source_templates"] > 1) | (selected["distinct_commit_counts"] > 1)
    ].copy()
    # Raw rows on excluded side are deliberately not returned to the analysis.
    return selected, missing_ids, conflicts


def read_template_owner_map(folder):
    """Join source-template owner_type from template datasets, if available."""
    if folder is None:
        return pd.DataFrame(columns=["language", "source_template", "source_owner_type"]), pd.DataFrame()
    files = sorted(p for p in folder.rglob(TEMPLATES_GLOB) if p.is_file())
    if not files:
        raise SystemExit(f"No template metadata matched {TEMPLATES_GLOB!r} under {folder}")
    frames = []
    for path in files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"username", "project_name", "owner_type"}
        if missing := required.difference(df.columns):
            print(f"WARNING: ignoring template metadata {path}: missing {sorted(missing)}", flush=True)
            continue
        print(f"Reading template owner metadata: {path}", flush=True)
        slim = pd.DataFrame({
            "language": language_from_path(path, TEMPLATE_PREFIX),
            "source_template": (df["username"].str.strip() + "/" + df["project_name"].str.strip()).map(normalize_repo),
            "source_owner_type": df["owner_type"].str.strip().str.lower().map({
                "user": "User", "organization": "Organization", "organisation": "Organization", "org": "Organization"
            }),
        }).dropna(subset=["source_template"])
        frames.append(slim)
    if not frames:
        print("WARNING: no usable template ownership metadata; ownership will be Unknown", flush=True)
        return pd.DataFrame(columns=["language", "source_template", "source_owner_type"]), pd.DataFrame()
    owners = pd.concat(frames, ignore_index=True).drop_duplicates()
    grouped = owners.groupby(["language", "source_template"], as_index=False, dropna=False)
    flags = grouped["source_owner_type"].agg(lambda x: x.dropna().nunique()).rename(columns={"source_owner_type": "owner_type_count"})
    conflicts = flags[flags["owner_type_count"] > 1].copy()
    owners = owners.merge(flags, on=["language", "source_template"], how="left", validate="many_to_one")
    owners.loc[owners["owner_type_count"] > 1, "source_owner_type"] = None
    owners = (owners.sort_values("source_owner_type", na_position="last")
              .drop_duplicates(["language", "source_template"])
              [["language", "source_template", "source_owner_type"]])
    return owners, conflicts


def assign_category(commits):
    if commits == 1:
        return "one_commit"
    if commits <= 6:
        return "low_2_6"
    if commits <= 20:
        return "moderate_7_20"
    return "high_gt_20"


def percent(n, denominator):
    return round(100 * n / denominator, 2) if denominator else 0.0


def family_table(details):
    group_cols = ["language", "source_template", "source_owner_type"]
    agg = details.groupby(group_cols, dropna=False).agg(
        generated_repositories=("generated_repo", "nunique"),
        one_commit=("category", lambda v: int((v == "one_commit").sum())),
        low_2_6=("category", lambda v: int((v == "low_2_6").sum())),
        moderate_7_20=("category", lambda v: int((v == "moderate_7_20").sum())),
        high_gt_20=("category", lambda v: int((v == "high_gt_20").sum())),
        median_commits=("commits_num", "median"),
        mean_commits=("commits_num", "mean"),
        max_commits=("commits_num", "max"),
    ).reset_index()
    agg["source_owner_type"] = agg["source_owner_type"].fillna("Unknown")
    agg["mean_commits"] = agg["mean_commits"].round(2)
    for cat in CATEGORIES:
        agg[f"{cat}_percentage"] = (
            100 * agg[cat] / agg["generated_repositories"]
        ).round(2)
    # A template's share of language-level activity, not just its within-family rate.
    totals = details.groupby("language")["category"].agg(
        language_one_commit=lambda x: int((x == "one_commit").sum()),
        language_high_gt_20=lambda x: int((x == "high_gt_20").sum()),
    ).reset_index()
    agg = agg.merge(totals, on="language", how="left", validate="many_to_one")
    agg["share_of_language_one_commit_percentage"] = (
        100 * agg["one_commit"] / agg["language_one_commit"].replace(0, float("nan"))
    ).round(2)
    agg["share_of_language_high_gt_20_percentage"] = (
        100 * agg["high_gt_20"] / agg["language_high_gt_20"].replace(0, float("nan"))
    ).round(2)
    agg["source_template_url"] = "https://github.com/" + agg["source_template"]
    return agg


def group_summary(details, keys):
    out = details.groupby(keys, dropna=False).agg(
        repositories=("generated_repo", "size"),
        source_template_families=("source_template", "nunique"),
        one_commit=("category", lambda x: int((x == "one_commit").sum())),
        low_2_6=("category", lambda x: int((x == "low_2_6").sum())),
        moderate_7_20=("category", lambda x: int((x == "moderate_7_20").sum())),
        high_gt_20=("category", lambda x: int((x == "high_gt_20").sum())),
    ).reset_index()
    for cat in CATEGORIES:
        out[f"{cat}_percentage"] = (100 * out[cat] / out["repositories"]).round(2)
    return out


def inspection_candidates(details, families, top):
    """Deterministic cases for later manual inspection; not a random sample."""
    focus = set(map(tuple, families[["language", "source_template"]].itertuples(index=False, name=None)))
    subset = details[details.apply(lambda r: (r["language"], r["source_template"]) in focus, axis=1)].copy()
    output = []
    for (lang, template), frame in subset.groupby(["language", "source_template"]):
        for cat, ascending in [("one_commit", True), ("high_gt_20", False)]:
            cases = frame[frame["category"] == cat].sort_values(
                ["commits_num", "generated_repo"], ascending=[ascending, True]
            ).head(3)
            for _, row in cases.iterrows():
                output.append({
                    "language": lang,
                    "source_template": template,
                    "source_owner_type": row["source_owner_type"],
                    "activity_group": cat,
                    "generated_repo": row["generated_repo"],
                    "generated_repo_url": "https://github.com/" + row["generated_repo"],
                    "commit_count": int(row["commits_num"]),
                    "data_creation": row.get("data_creation", ""),
                    "last_commit_datetime": row.get("last_commit_datetime", ""),
                    "source_csv": row["source_csv"],
                })
    return pd.DataFrame(output, columns=[
        "language", "source_template", "source_owner_type", "activity_group",
        "generated_repo", "generated_repo_url", "commit_count", "data_creation",
        "last_commit_datetime", "source_csv",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="Directory with generated_GHRep_from_templates_*.csv")
    ap.add_argument("--templates_dir", type=Path, default=None,
                    help="Optional directory with all_templates_*.csv (for source owner_type)")
    ap.add_argument("--output_dir", type=Path, default=Path("rq1_family_activity_results"))
    ap.add_argument("--min_family_size", type=int, default=20,
                    help="Minimum generated repositories per family for rate-based rankings (default: 20)")
    ap.add_argument("--top", type=int, default=20,
                    help="Number of leading families per language and ranking (default: 20)")
    args = ap.parse_args()
    if not args.input_dir.is_dir():
        ap.error(f"Input directory not found: {args.input_dir}")
    if args.templates_dir is not None and not args.templates_dir.is_dir():
        ap.error(f"Template directory not found: {args.templates_dir}")
    if args.min_family_size < 1 or args.top < 1:
        ap.error("--min_family_size and --top must both be >= 1")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    raw = read_generated(args.input_dir)
    selected, missing_ids, dup_conflicts = deduplicate_generated(raw)
    no_valid_commit = selected[~selected["valid_commits"]].copy()
    no_source = selected[selected["valid_commits"] & selected["source_template"].isna()].copy()
    details = selected[selected["valid_commits"] & selected["source_template"].notna()].copy()
    owner_map, owner_conflicts = read_template_owner_map(args.templates_dir)
    details = details.merge(owner_map, on=["language", "source_template"], how="left", validate="many_to_one")
    details["source_owner_type"] = details["source_owner_type"].fillna("Unknown")
    details["commits_num"] = details["commits_num"].astype(int)
    details["category"] = details["commits_num"].map(assign_category)
    details["generated_repo_url"] = "https://github.com/" + details["generated_repo"]
    details["source_template_url"] = "https://github.com/" + details["source_template"]

    fields = ["language", "generated_repo", "generated_repo_url", "source_template",
              "source_template_url", "source_owner_type", "commits_num", "category",
              "data_creation", "data_update", "last_commit_datetime", "source_csv",
              "source_row", "duplicate_occurrences"]
    details[[x for x in fields if x in details.columns]].to_csv(
        out / "generated_repository_activity_details.csv", index=False
    )
    fam = family_table(details)
    fam.sort_values(["language", "generated_repositories", "source_template"],
                    ascending=[True, False, True]).to_csv(out / "source_template_family_metrics.csv", index=False)

    rank_specs = [
        ("top_one_commit_by_count.csv", "one_commit", "one_commit_percentage", False),
        ("top_one_commit_by_rate.csv", "one_commit_percentage", "one_commit", True),
        ("top_high_activity_by_count.csv", "high_gt_20", "high_gt_20_percentage", False),
        ("top_high_activity_by_rate.csv", "high_gt_20_percentage", "high_gt_20", True),
    ]
    all_focus = []
    for filename, primary, secondary, rate_rank in rank_specs:
        tmp = fam[fam["generated_repositories"] >= args.min_family_size].copy() if rate_rank else fam.copy()
        focus_count = "one_commit" if "one_commit" in filename else "high_gt_20"
        tmp = tmp[tmp[focus_count] > 0].copy()  # exclude families with no examples
        tmp = tmp.sort_values(
            ["language", primary, secondary, "generated_repositories", "source_template"],
            ascending=[True, False, False, False, True]
        ).groupby("language", sort=False).head(args.top)
        tmp.insert(0, "rank_within_language", tmp.groupby("language").cumcount() + 1)
        tmp.to_csv(out / filename, index=False)
        all_focus.append(tmp[["language", "source_template"]])

    language_summary = group_summary(details, ["language"])
    language_summary.to_csv(out / "language_activity_summary.csv", index=False)
    ownership_summary = group_summary(details, ["language", "source_owner_type"])
    overall_by_owner = group_summary(details, ["source_owner_type"])
    overall_by_owner.insert(0, "language", "Overall")
    pd.concat([ownership_summary, overall_by_owner], ignore_index=True).to_csv(
        out / "ownership_activity_summary.csv", index=False
    )

    focus = pd.concat(all_focus, ignore_index=True).drop_duplicates() if all_focus else fam.iloc[0:0]
    inspection_candidates(details, focus, args.top).to_csv(out / "qualitative_inspection_candidates.csv", index=False)
    # Audits: no input changes. These are separate populations, so not all
    # invalid rows necessarily affect the valid-repository denominator.
    missing_ids.to_csv(out / "audit_missing_repository_identity.csv", index=False)
    no_valid_commit.to_csv(out / "audit_invalid_commit_count.csv", index=False)
    no_source.to_csv(out / "audit_missing_source_template.csv", index=False)
    dup_conflicts.to_csv(out / "audit_conflicting_duplicate_rows.csv", index=False)
    owner_conflicts.to_csv(out / "audit_conflicting_source_owner_type.csv", index=False)

    print("\nRQ1 SOURCE-TEMPLATE FAMILY ANALYSIS", flush=True)
    print(f"Raw rows: {len(raw):,}")
    print(f"Unique (language, repository) observations: {len(selected):,}")
    print(f"Excess duplicate rows: {len(raw) - len(missing_ids) - len(selected):,}")
    print(f"Missing repository identity: {len(missing_ids):,}")
    print(f"Invalid commit count: {len(no_valid_commit):,}")
    print(f"Missing source template (with valid commits): {len(no_source):,}")
    print(f"Valid analyzed rows: {len(details):,}")
    print(f"Distinct (language, source-template) families: {len(fam):,}")
    print(f"Conflicting duplicate observations: {len(dup_conflicts):,}")
    print("\nBy language:")
    print(language_summary.to_string(index=False))
    print("\nTop 5 families by one-commit count:")
    print(fam.sort_values("one_commit", ascending=False)[
        ["language", "source_template", "source_owner_type", "generated_repositories",
         "one_commit", "one_commit_percentage"]].head(5).to_string(index=False))
    print("\nTop 5 families by >20-commit count:")
    print(fam.sort_values("high_gt_20", ascending=False)[
        ["language", "source_template", "source_owner_type", "generated_repositories",
         "high_gt_20", "high_gt_20_percentage"]].head(5).to_string(index=False))
    print(f"\nResults saved in {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
