#!/usr/bin/env python3
"""RQ1: repository-age sensitivity analysis for GitHub template-generated projects.

Uses only the Python standard library. Tested with Python 3.10+.

Example:
  python rq1_repository_age_analysis.py "path/to/csv_folder" --as-of 2026-09-15

IMPORTANT: --as-of must be the date on which commit counts were observed
(or a suitable common collection cutoff), NOT automatically today's date.

Inputs: CSV files matched by --pattern, e.g.
    generated_GHRep_from_templates_JavaScript_commits.csv
    generated_GHRep_from_templates_TypeScript_commits.csv
Required columns: username, project_name, original_link, data_creation,
                  commit_count
Optional: target_template_repository, last_commit_datetime

Outputs (in --output directory):
    rq1_age_by_language.csv       Percentages per language and age band
    rq1_age_robustness.csv        All repos vs >=6, >=12, >=24 months
    rq1_template_weighting.csv   Repository- vs equal-template-weighted results
    rq1_data_quality.csv         Input rows and exclusion reasons
    rq1_duplicate_conflicts.csv  Conflicting records excluded from analysis
    rq1_summary.txt              Human-readable results to share
"""

import argparse
import csv
import calendar
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median
from urllib.parse import urlparse, unquote

AGE_BANDS = ["<6 months", "6-<12 months", "12-<24 months", ">=24 months"]
ACTIVITY = ["1 commit", "2-6 commits", "7-20 commits", ">20 commits"]
FILTERS = ["All ages", ">=6 months", ">=12 months", ">=24 months"]


def read_date(value):
    """Parse an ISO/GitHub date; do not silently guess ambiguous numeric dates."""
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"nan", "none", "null", "nat"}:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def subtract_months(day, months):
    year = day.year
    month = day.month - months
    while month <= 0:
        month += 12
        year -= 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def get_language(filename):
    name = filename.lower()
    if "javascript" in name:
        return "JavaScript"
    if "typescript" in name:
        return "TypeScript"
    if "python" in name:
        return "Python"
    if any(t in name for t in ("csharp", "c_sharp", "c-sharp", "c#", "c%23")):
        return "C#"
    if re.search(r"(?:^|[_\-. ])java(?:$|[_\-. ])", name):
        return "Java"
    return "Unknown: " + filename


def normalize_repo(row):
    url = str(row.get("original_link", "") or "").strip()
    if url:
        try:
            parsed = urlparse(url if "://" in url else "https://" + url)
            if parsed.hostname and parsed.hostname.lower() in ("github.com", "www.github.com"):
                bits = [unquote(x) for x in parsed.path.strip("/").split("/")]
                if len(bits) >= 2 and bits[0] and bits[1]:
                    return (bits[0] + "/" + re.sub(r"\.git$", "", bits[1], flags=re.I)).lower()
        except (ValueError, AttributeError):
            pass
    owner = str(row.get("username", "") or "").strip()
    repo = str(row.get("project_name", "") or "").strip()
    if owner and repo:
        return (owner + "/" + repo).lower()
    return None


def normalize_template(value):
    name = str(value or "").strip().strip("/").lower()
    if name.startswith("https://github.com/"):
        name = name.removeprefix("https://github.com/")
    if name.endswith(".git"):
        name = name[:-4]
    return name or None


def activity(count):
    if count == 1:
        return ACTIVITY[0]
    if count <= 6:
        return ACTIVITY[1]
    if count <= 20:
        return ACTIVITY[2]
    return ACTIVITY[3]


def age_band(created, asof):
    if created > subtract_months(asof, 6):
        return AGE_BANDS[0]
    if created > subtract_months(asof, 12):
        return AGE_BANDS[1]
    if created > subtract_months(asof, 24):
        return AGE_BANDS[2]
    return AGE_BANDS[3]


def qualifies(r, restriction):
    if restriction == "All ages":
        return True
    if restriction == ">=6 months":
        return r["band"] != AGE_BANDS[0]
    if restriction == ">=12 months":
        return r["band"] in AGE_BANDS[2:]
    return r["band"] == AGE_BANDS[3]


def percent(num, denom):
    return round(100 * num / denom, 2) if denom else ""


def save_csv(path, records, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def activity_stats(rows, label, group):
    total = len(rows)
    counts = Counter(r["activity"] for r in rows)
    out = {"language": label, "group": group, "repositories": total}
    for index, cat in enumerate(ACTIVITY):
        out[f"n_{index}"] = counts[cat]
        out[f"pct_{index}"] = percent(counts[cat], total)
    out["pct_at_most_6"] = percent(counts[ACTIVITY[0]] + counts[ACTIVITY[1]], total)
    out["pct_with_extra_commit"] = percent(total - counts[ACTIVITY[0]], total)
    out["distinct_templates"] = len({r["template"] for r in rows if r["template"]})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder containing per-language commits CSV files")
    parser.add_argument("--as-of", required=True, help="Common commit-count collection date, YYYY-MM-DD")
    parser.add_argument("--pattern", default="*commits*.csv", help="Filename glob (default: *commits*.csv)")
    parser.add_argument("--output", type=Path, default=Path("rq1_age_results"), help="Output directory")
    args = parser.parse_args()
    try:
        asof = date.fromisoformat(args.as_of)
    except ValueError:
        parser.error("--as-of must be YYYY-MM-DD")
    if not args.folder.is_dir():
        parser.error("Input folder does not exist: " + str(args.folder))

    paths = sorted(p for p in args.folder.glob(args.pattern) if p.is_file())
    if not paths:
        parser.error(f"No matching CSV files in {args.folder}; check --pattern")

    counts = defaultdict(Counter)
    records = {}
    conflict_ids = set()
    conflict_details = []
    languages = set()
    for path in paths:
        file = path.name
        language = get_language(file)
        languages.add(language)
        print(f"Reading {file} -> {language}")
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"username", "project_name", "original_link", "data_creation", "commit_count"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                parser.error(f"{file} is missing columns: {', '.join(sorted(missing))}")
            for row in reader:
                counts[file]["input_rows"] += 1
                rid = normalize_repo(row)
                if not rid:
                    counts[file]["missing_identity"] += 1
                    continue
                created = read_date(row.get("data_creation"))
                if created is None:
                    counts[file]["invalid_creation_date"] += 1
                    continue
                if created > asof:
                    counts[file]["created_after_asof"] += 1
                    continue
                try:
                    raw_count = str(row.get("commit_count") or "").strip()
                    number = float(raw_count)
                    if not number.is_integer() or number < 1:
                        raise ValueError("not a positive integer")
                    ncommits = int(number)
                except (ValueError, OverflowError):
                    counts[file]["invalid_commit_count"] += 1
                    continue

                last_commit = read_date(row.get("last_commit_datetime"))
                if last_commit and last_commit > asof:
                    counts[file]["last_commit_after_asof"] += 1

                template = normalize_template(row.get("target_template_repository"))
                data = {"identity": rid, "language": language, "created": created,
                        "commits": ncommits, "template": template, "file": file,
                        "band": age_band(created, asof), "activity": activity(ncommits)}
                if rid in conflict_ids:
                    counts[file]["conflicting_duplicate_rows"] += 1
                    conflict_details.append({"repo": rid, "filename": file, "language": language,
                                             "creation": str(created), "commit_count": ncommits})
                    continue
                if rid in records:
                    prior = records[rid]
                    fields = ("language", "created", "commits", "template")
                    if all(prior[k] == data[k] for k in fields):
                        counts[file]["identical_duplicates_removed"] += 1
                    else:
                        counts[file]["conflicting_duplicate_rows"] += 1
                        counts[prior["file"]]["conflicting_duplicate_rows"] += 1
                        conflict_ids.add(rid)
                        conflict_details.extend([
                            {"repo": rid, "filename": x["file"], "language": x["language"],
                             "creation": str(x["created"]), "commit_count": x["commits"]}
                            for x in (prior, data)
                        ])
                        del records[rid]
                    continue
                records[rid] = data
                counts[file]["valid_first_appearances"] += 1

    all_rows = list(records.values())
    if not all_rows:
        parser.error("No analyzable repositories after checks; verify dates and column values")
    args.output.mkdir(parents=True, exist_ok=True)
    language_names = [x for x in ("C#", "Java", "Python", "TypeScript", "JavaScript") if x in languages]
    language_names.extend(sorted(languages - set(language_names)))
    all_groups = ["Overall"] + language_names

    age_table = []
    sensitivity = []
    weighting = []
    for lang in all_groups:
        subset = all_rows if lang == "Overall" else [r for r in all_rows if r["language"] == lang]
        for band in AGE_BANDS:
            age_table.append(activity_stats([r for r in subset if r["band"] == band], lang, band))
        for restriction in FILTERS:
            filtered = [r for r in subset if qualifies(r, restriction)]
            sensitivity.append(activity_stats(filtered, lang, restriction))
            families = defaultdict(list)
            for r in filtered:
                if r["template"]:
                    # Pair source-template identity with language, for the pooled overall analysis
                    families[(r["language"], r["template"])].append(r)
            for cat in ACTIVITY:
                num = sum(r["activity"] == cat for r in filtered)
                per_template = [sum(r["activity"] == cat for r in rs) / len(rs)
                                for rs in families.values()]
                weighting.append({"language": lang, "min_age": restriction,
                                  "activity": cat, "repos": len(filtered),
                                  "repos_with_template": sum(len(x) for x in families.values()),
                                  "templates": len(families),
                                  "repo_weighted_pct": percent(num, len(filtered)),
                                  "equal_template_weighted_pct": round(100 * mean(per_template), 2)
                                  if per_template else ""})

    cols = ["language", "group", "repositories", "distinct_templates"]
    cols += [x for i in range(4) for x in (f"n_{i}", f"pct_{i}")]
    cols += ["pct_at_most_6", "pct_with_extra_commit"]
    save_csv(args.output / "rq1_age_by_language.csv", age_table, cols)
    save_csv(args.output / "rq1_age_robustness.csv", sensitivity, cols)
    save_csv(args.output / "rq1_template_weighting.csv", weighting,
             ["language", "min_age", "activity", "repos", "repos_with_template", "templates",
              "repo_weighted_pct", "equal_template_weighted_pct"])

    qc_fields = ["filename", "input_rows", "valid_first_appearances", "missing_identity",
                 "invalid_creation_date", "created_after_asof", "invalid_commit_count",
                 "last_commit_after_asof", "identical_duplicates_removed", "conflicting_duplicate_rows"]
    qc = [{"filename": file, **{key: v.get(key, 0) for key in qc_fields[1:]}}
          for file, v in sorted(counts.items())]
    save_csv(args.output / "rq1_data_quality.csv", qc, qc_fields)
    save_csv(args.output / "rq1_duplicate_conflicts.csv", conflict_details,
             ["repo", "filename", "language", "creation", "commit_count"])

    def display_pct(value):
        return f"{value:.2f}%" if isinstance(value, (int, float)) else "--"

    lines = [f"RQ1 age sensitivity analysis | As-of date: {asof}",
             f"Input files: {len(paths)} | Input rows: {sum(c['input_rows'] for c in counts.values()):,}",
             f"Unique eligible repositories: {len(all_rows):,}",
             f"Conflicting project identifiers excluded: {len(conflict_ids):,}",
             "", "Age-by-activity table: n, % no extra commits, % 2-6, % 7-20, % >20", ""]
    for r in age_table:
        lines.append(f"{r['language']:12} | {r['group']:14} | n={r['repositories']:7,} | "
                     f"{display_pct(r['pct_0']):>7} | {display_pct(r['pct_1']):>7} | "
                     f"{display_pct(r['pct_2']):>7} | {display_pct(r['pct_3']):>7}")
    lines += ["", "Sensitivity to minimum repository age:", ""]
    for r in sensitivity:
        lines.append(f"{r['language']:12} | {r['group']:12} | n={r['repositories']:7,} | "
                     f"<=6 commits {display_pct(r['pct_at_most_6']):>7} | "
                     f">20 commits {display_pct(r['pct_3']):>7}")
    lines += ["", "CONSIDERATIONS:",
              "- Age is creation date to as-of collection date, not age at last commit.",
              "- Percentages are calculated within each language x age band (or filter).",
              "- The >=12 month group is a subset of 'All ages'; not a separate population.",
              "- No causal age effect can be inferred from cross-sectional comparisons.",
              "- Template weighting excludes records without target_template_repository.",
              "- Commit counts should be collected at/about the --as-of date.",
              "- If any last_commit_after_asof values are nonzero, inspect collection-date alignment."]
    (args.output / "rq1_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:6]))
    print(f"\nWrote 6 files to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
