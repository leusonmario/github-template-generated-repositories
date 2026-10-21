#!/usr/bin/env python3
"""Plot horizontal violin distributions of commit counts for template-generated repositories.

Usage (from ~/MSR/template_based_analysis/rq1):

    python -u plot_rq1_commit_violin.py reports --output_dir rq1_violin

Optional second, zoomed-in figure (KDE still uses the FULL population):

    python -u plot_rq1_commit_violin.py reports --output_dir rq1_violin --zoom_max 100

Reads generated_GHRep_from_templates_*.csv recursively. Assumes the same
activity thresholds as the RQ1 table: 1, 2--6, 7--20, and >20 commits.

Exactly one observation is selected per (language, generated repository).
Like analyze_rq1_template_families.py, duplicate rows are prioritized by:
valid positive-integer commit count, successful commit-info status, quantity
of nonempty metadata, and most recent data_update. Repositories lacking a
source-template identifier are excluded for consistency with family analyses.

Outputs:
    commit_violin.pdf / commit_violin.png / commit_violin.svg
    commit_violin_zoom_100.* (only with --zoom_max 100)
    commit_activity_summary.csv
    violin_input_audit.csv

Requirements: pandas, numpy, matplotlib, scipy.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")   # works on headless machines / remote servers
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

INPUT_GLOB = "generated_GHRep_from_templates_*.csv"
PREFIX = "generated_GHRep_from_templates_"
LANG_ORDER = ["C#", "Java", "Python", "TypeScript", "JavaScript"]
LANG_MAP = {"c#": "C#", "csharp": "C#", "java": "Java", "python": "Python",
            "typescript": "TypeScript", "javascript": "JavaScript"}


def normalize_repo(value):
    """Normalize repo URL or owner/repo to a lowercase 'owner/repo'."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() in {"nan", "null", "none"}:
        return None
    if s.lower().startswith("git@github.com:"):
        s = s[len("git@github.com:"):]
    elif s.lower().startswith(("https://", "http://")):
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
    if not repo:
        return None
    return f"{owner}/{repo}".lower()


def language_from_file(path):
    suffix = path.stem[len(PREFIX):]
    if suffix.endswith("_with_commits"):
        suffix = suffix[:-len("_with_commits")]
    return LANG_MAP.get(suffix.casefold(), suffix)


def read_data(input_dir):
    paths = sorted(p for p in input_dir.rglob(INPUT_GLOB) if p.is_file())
    if not paths:
        raise SystemExit(f"ERROR: no files matching {INPUT_GLOB} inside {input_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"username", "project_name", "target_template_repository", "commit_count"}
        missing = required - set(frame.columns)
        if missing:
            raise SystemExit(f"ERROR: missing {sorted(missing)} in {path}")

        # Matches the normalization and selection logic from the family ranking.
        name_repo = (frame["username"].str.strip() + "/" +
                     frame["project_name"].str.strip()).map(normalize_repo)
        if "original_link" in frame:
            url_repo = frame["original_link"].map(normalize_repo)
            repo = url_repo.combine_first(name_repo)
        else:
            repo = name_repo

        commit_count = pd.to_numeric(frame["commit_count"], errors="coerce")
        valid = commit_count.notna() & commit_count.ge(1) & commit_count.eq(commit_count.round())
        info = (frame["commit_info_status"].str.strip().str.lower().isin(
            {"success", "ok", "completed", "complete"})
            if "commit_info_status" in frame else pd.Series(False, index=frame.index))
        metadata = pd.Series(0, index=frame.index)
        for col in ("original_link", "data_update", "data_creation",
                    "last_commit_datetime", "commit_info_status"):
            if col in frame:
                metadata += frame[col].str.strip().ne("").astype(int)
        updated = (pd.to_datetime(frame["data_update"], errors="coerce", utc=True)
                   if "data_update" in frame else pd.Series(pd.NaT, index=frame.index))
        slim = pd.DataFrame({
            "language": language_from_file(path),
            "repository": repo,
            "source_template": frame["target_template_repository"].map(normalize_repo),
            "commit_count": commit_count,
            "valid_commit_count": valid,
            "commit_info_success": info,
            "metadata_filled": metadata,
            "updated_dt": updated,
        })
        print(f"Read {len(slim):,} rows: {path}", flush=True)
        frames.append(slim)

    raw = pd.concat(frames, ignore_index=True)
    no_identity = int(raw["repository"].isna().sum())
    recognized = raw.dropna(subset=["repository"]).copy()
    recognized = recognized.sort_values(
        ["valid_commit_count", "commit_info_success", "metadata_filled", "updated_dt"],
        ascending=[False, False, False, False],
        kind="stable", na_position="last")
    selected = recognized.drop_duplicates(["language", "repository"], keep="first").copy()
    duplicates = len(recognized) - len(selected)
    invalid = int((~selected["valid_commit_count"]).sum())
    missing_template = int((selected["valid_commit_count"] & selected["source_template"].isna()).sum())
    data = selected[selected["valid_commit_count"] & selected["source_template"].notna()].copy()
    data["commit_count"] = data["commit_count"].astype(np.int64)

    audit = pd.DataFrame([
        {"metric": "CSV files read", "count": len(paths)},
        {"metric": "Raw rows", "count": len(raw)},
        {"metric": "Rows missing generated-repository identity", "count": no_identity},
        {"metric": "Excess duplicate rows within language", "count": duplicates},
        {"metric": "Unique (language, repository) observations", "count": len(selected)},
        {"metric": "Selected rows with invalid commit counts", "count": invalid},
        {"metric": "Selected valid rows missing source template", "count": missing_template},
        {"metric": "Final analyzed repositories", "count": len(data)},
    ])
    return data, audit


def activity_summary(data):
    rows = []
    languages = [x for x in LANG_ORDER if x in set(data["language"])]
    languages += sorted(set(data["language"]) - set(languages))
    for lang in languages + ["Overall"]:
        arr = data["commit_count"].to_numpy() if lang == "Overall" else (
            data.loc[data["language"] == lang, "commit_count"].to_numpy())
        if not len(arr):
            continue
        n = len(arr)
        counts = [int((arr == 1).sum()), int(((arr >= 2) & (arr <= 6)).sum()),
                  int(((arr >= 7) & (arr <= 20)).sum()), int((arr > 20).sum())]
        rows.append({
            "language": lang, "repositories": n,
            "one_commit_pct": 100 * counts[0] / n,
            "low_2_6_pct": 100 * counts[1] / n,
            "moderate_7_20_pct": 100 * counts[2] / n,
            "high_gt_20_pct": 100 * counts[3] / n,
            "median_commits": float(np.median(arr)),
            "p90_commits": float(np.percentile(arr, 90)),
            "p95_commits": float(np.percentile(arr, 95)),
            "p99_commits": float(np.percentile(arr, 99)),
            "max_commits": int(arr.max()),
        })
    return pd.DataFrame(rows)



# Accessible, muted colors that remain distinguishable in the legend.
PALETTE = {
    "C#": "#4477AA", "Java": "#CC6677", "Python": "#228833",
    "TypeScript": "#AA4499", "JavaScript": "#DD9922",
}


def violin_density(log_values: np.ndarray, positions: np.ndarray, bw_adjust: float):
    """Estimate smoothed density *in log10(commit_count) units*.

    KDE does not model discrete commits precisely. We restrict the display to
    the actual data range, and avoid drawing a misleading violin for groups
    with no variation. Every KDE is normalized to equal maximum width; this
    displays shape, not the relative number of generated repositories.
    """
    if len(log_values) < 2 or np.ptp(log_values) <= 1e-10:
        return None
    try:
        kde = gaussian_kde(log_values, bw_method=lambda obj: obj.scotts_factor() * bw_adjust)
        y = kde(positions)
    except np.linalg.LinAlgError:
        return None
    y[(positions < log_values.min()) | (positions > log_values.max())] = 0
    if not np.isfinite(y).all() or np.max(y) <= 0:
        return None
    return y / np.max(y)


def make_violin(data, outdir, stem, zoom_max=None, dpi=300,
                width=7.0, height=3.6, bw_adjust=1.0, grayscale=False):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 12,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.7,
    })
    present = set(data["language"])
    languages = [x for x in LANG_ORDER if x in present]
    languages += sorted(present - set(languages))
    true_max = int(data["commit_count"].max())
    display_max = zoom_max if zoom_max else true_max
    display_max = max(2, display_max)
    log_display_max = np.log10(display_max)

    fig, ax = plt.subplots(figsize=(width, height), layout="constrained")
    xgrid = np.linspace(0, np.log10(true_max), 450)
    if true_max == 1:
        xgrid = np.linspace(0, np.log10(2), 50)

    for row, lang in enumerate(languages):
        commits = data.loc[data["language"].eq(lang), "commit_count"].to_numpy(dtype=float)
        log_values = np.log10(commits)
        d = violin_density(log_values, xgrid, bw_adjust)
        color = "0.68" if grayscale else PALETTE.get(lang, "#777777")
        ycenter = row
        if d is not None:
            radius = 0.38 * d
            ax.fill_between(xgrid, ycenter - radius, ycenter + radius,
                            facecolor=color, edgecolor="0.28" if grayscale else color,
                            linewidth=0.75, alpha=0.72, zorder=2)
        else:
            # With identical commit counts the empirical distribution is a
            # single point, not a continuous violin.
            ax.plot([np.log10(commits[0])], [ycenter], marker="o", ms=5,
                    color=color, markeredgecolor="0.25", zorder=4)

        q25, median, q75 = np.quantile(log_values, [0.25, 0.5, 0.75])
        ax.plot([q25, q75], [ycenter, ycenter], lw=3.1,
                color="0.12", solid_capstyle="round", zorder=5)
        ax.plot([median], [ycenter], marker="o", ms=4.1, color="white",
                markeredgecolor="0.12", markeredgewidth=0.8, zorder=6)
        if zoom_max is None or commits.max() <= zoom_max:
            ax.plot([np.log10(commits.max())], [ycenter], marker="x", ms=5.0,
                    markeredgewidth=0.9, linestyle="None", color="0.22", zorder=6)

    # Show the 6/20 boundaries explicitly: they correspond to the RQ1 bins.
    for threshold in (6, 20):
        if threshold <= display_max:
            ax.axvline(np.log10(threshold), ls=(0, (2.8, 2.8)), color="0.55",
                       lw=0.8, zorder=1)
    ax.set_yticks(range(len(languages)), languages)
    ax.invert_yaxis()
    ax.set_ylim(len(languages) - 0.5, -0.5)
    ax.set_xlim(-0.04, log_display_max + max(0.05, 0.03 * log_display_max))

    # Labels show actual commit counts, even though positions use log10.
    tick_candidates = [1, 2, 6, 20, 100, 1000, 10000, 100000, 1000000]
    ticks = [v for v in tick_candidates if v <= display_max]
    ax.xaxis.set_major_locator(FixedLocator([np.log10(v) for v in ticks]))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda x, pos: f"{10**x:,.0f}" if any(abs(x - np.log10(v)) < 1e-8 for v in ticks) else ""))
    ax.set_xlabel("Number of commits (log scale)")
    ax.set_ylabel("")
    ax.grid(axis="x", color="0.90", linewidth=0.55, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color="0.50")

    key = [
        Line2D([0], [0], color="0.15", lw=3.1, label="Interquartile range"),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="0.15",
               markersize=5, lw=0, label="Median"),
    ]
    if zoom_max is None:
        key += [Line2D([0], [0], marker="x", color="0.22", markersize=6,
                       lw=0, label="Maximum")]
    ax.legend(handles=key, ncol=len(key), loc="upper center",
              bbox_to_anchor=(0.50, -0.18), fontsize=7.2, frameon=False,
              handlelength=1.5, columnspacing=1.2)

    # A 'violin' is a smoothed representation: preserve exact percentages in
    # the companion summary, not via interpreting violin density area.
    for ext in ("pdf", "png", "svg"):
        opt = {"dpi": dpi} if ext == "png" else {}
        fig.savefig(outdir / f"{stem}.{ext}", bbox_inches="tight", **opt)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Directory with source CSVs, e.g. reports")
    parser.add_argument("--output_dir", type=Path, default=Path("rq1_violin"),
                        help="Output directory (default: rq1_violin)")
    parser.add_argument("--zoom_max", type=int, default=None,
                        help="Also save a version zoomed to at most this many commits, e.g. 100")
    parser.add_argument("--width", type=float, default=7.0,
                        help="Figure width in inches (e.g. 3.5 for an IEEE column)")
    parser.add_argument("--height", type=float, default=3.6,
                        help="Figure height in inches (default: 3.6)")
    parser.add_argument("--bw_adjust", type=float, default=1.0,
                        help="KDE bandwidth multiplier (default: 1; >1 smoother, <1 more detailed)")
    parser.add_argument("--grayscale", action="store_true",
                        help="Use gray violins for black-and-white publication")
    parser.add_argument("--dpi", type=int, default=300, help="PNG DPI (default: 300)")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error(f"Not a directory: {args.input_dir}")
    if args.zoom_max is not None and args.zoom_max < 21:
        parser.error("--zoom_max must be at least 21 to show both 6 and 20 thresholds")
    if not (3.0 <= args.width <= 16.0) or not (2.0 <= args.height <= 10.0):
        parser.error("--width must be 3..16 and --height 2..10 inches")
    if not (0.2 <= args.bw_adjust <= 5.0):
        parser.error("--bw_adjust must be 0.2..5.0")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, audit = read_data(args.input_dir)
    if data.empty:
        raise SystemExit("ERROR: no valid repository commit counts were found")
    summary = activity_summary(data)
    summary.to_csv(args.output_dir / "commit_activity_summary.csv", index=False,
                   float_format="%.4f")
    audit.to_csv(args.output_dir / "violin_input_audit.csv", index=False)
    make_violin(data, args.output_dir, "commit_violin", dpi=args.dpi,
                width=args.width, height=args.height, grayscale=args.grayscale,
                bw_adjust=args.bw_adjust)
    if args.zoom_max is not None:
        make_violin(data, args.output_dir, f"commit_violin_zoom_{args.zoom_max}",
                    dpi=args.dpi, width=args.width, height=args.height,
                    zoom_max=args.zoom_max, grayscale=args.grayscale,
                    bw_adjust=args.bw_adjust)

    print("\nAnalysis summary (repository-weighted):", flush=True)
    print(summary[["language", "repositories", "one_commit_pct", "low_2_6_pct",
                   "moderate_7_20_pct", "high_gt_20_pct", "median_commits", "max_commits"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"), flush=True)
    print(f"\nOutputs: {args.output_dir.resolve()}", flush=True)
    print("IMPORTANT: Match counts/percentages against your current RQ1 table before publication.", flush=True)
    print("Violin curves are smoothed densities of log10(commits); do not infer exact percentages",
          "from their widths or interpret a long violin tail as an outlier count.", flush=True)


if __name__ == "__main__":
    main()
