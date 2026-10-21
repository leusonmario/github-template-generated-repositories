#!/usr/bin/env python3
"""
generate_issue_persistence_charts.py

Generate publication-ready charts from:

    pair_summary.csv
    template_summary.csv
    template_issue_persistence.csv

The script is tolerant of small differences in column names. Charts whose
required columns are unavailable are skipped and recorded in
chart_generation_report.csv.

Example:
    python generate_issue_persistence_charts.py \
        pair_summary.csv \
        template_summary.csv \
        template_issue_persistence.csv \
        charts

Optional:
    python generate_issue_persistence_charts.py \
        pair_summary.csv \
        template_summary.csv \
        template_issue_persistence.csv \
        charts \
        --top-n 20 \
        --min-projects-per-template 2 \
        --min-rule-observations 20 \
        --dpi 300
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------------

PAIR_TEMPLATE_COLUMN = [
    "template_repository",
    "template_id",
]

PAIR_PROJECT_COLUMN = [
    "generated_repository",
    "project_id",
]

PAIR_PERSISTENCE_PERCENTAGE = [
    "persistence_percentage",
    "persistence_rate",
]

PAIR_TEMPLATE_ISSUE_COUNT = [
    "template_issue_count",
]

PAIR_PROJECT_ISSUE_COUNT = [
    "project_issue_count",
]

PAIR_EVOLUTION_BAND = [
    "evolution_band",
]


TEMPLATE_REPOSITORY_COLUMN = [
    "template_repository",
    "template_id",
]

TEMPLATE_PROJECT_COUNT = [
    "generated_repository_count",
    "compared_project_count",
]

TEMPLATE_PERSISTENCE_PERCENTAGE = [
    "micro_persistence_percentage",
    "macro_average_pair_persistence_percentage",
    "micro_persistence_rate",
    "macro_average_pair_persistence_rate",
]

TEMPLATE_ISSUE_COUNT = [
    "template_issue_count_per_repository",
    "template_issue_count",
]


ISSUE_TEMPLATE_COLUMN = [
    "template_repository",
    "template_id",
]

ISSUE_RULE_COLUMN = [
    "rule",
]

ISSUE_TYPE_COLUMN = [
    "type",
]

ISSUE_SEVERITY_COLUMN = [
    "severity",
]

ISSUE_PERSISTENCE_PERCENTAGE = [
    "persistence_percentage_across_generated_repositories",
    "weighted_persistence_percentage_across_generated_repositories",
    "persistence_rate_across_generated_repositories",
    "weighted_persistence_rate_across_generated_repositories",
]

ISSUE_COMPARED_PROJECT_COUNT = [
    "compared_project_count",
]

ISSUE_PERSISTENT_PROJECT_COUNT = [
    "persistent_project_count",
]


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate charts from pair_summary.csv, template_summary.csv, "
            "and template_issue_persistence.csv."
        )
    )

    parser.add_argument(
        "pair_summary",
        type=Path,
        help="Path to pair_summary.csv.",
    )
    parser.add_argument(
        "template_summary",
        type=Path,
        help="Path to template_summary.csv.",
    )
    parser.add_argument(
        "template_issue_persistence",
        type=Path,
        help="Path to template_issue_persistence.csv.",
    )
    parser.add_argument(
        "output_directory",
        type=Path,
        help="Directory in which charts and derived tables will be saved.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Maximum number of templates or rules in ranking charts. Default: 20.",
    )
    parser.add_argument(
        "--min-projects-per-template",
        type=int,
        default=1,
        help=(
            "Minimum number of generated repositories required for a template "
            "to appear in template-ranking charts. Default: 1."
        ),
    )
    parser.add_argument(
        "--min-rule-observations",
        type=int,
        default=10,
        help=(
            "Minimum number of template-issue/project comparisons required "
            "for a rule to appear in the rule-ranking chart. Default: 10."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution of PNG files. Default: 300.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display charts interactively in addition to saving them.",
    )

    return parser.parse_args()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{label} not found: {path}")

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError as error:
        raise SystemExit(f"{label} is empty: {path}") from error

    if dataframe.empty:
        raise SystemExit(f"{label} contains no rows: {path}")

    print(f"Loaded {len(dataframe):,} rows from {path}")
    return dataframe


def find_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """
    Find the first matching column, using case-insensitive comparison.
    """
    actual_by_lower = {
        str(column).strip().casefold(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        actual = actual_by_lower.get(candidate.casefold())
        if actual is not None:
            return actual

    return None


def numeric_series(dataframe: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(dataframe[column], errors="coerce")


def percentage_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return percentage-point values in the range normally interpreted as 0-100.

    If a rate column contains values between 0 and 1, it is multiplied by 100.
    Percentage columns already expressed as 0-100 are left unchanged.
    """
    values = numeric_series(dataframe, column)

    finite = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return values

    column_name = column.casefold()

    if "percentage" in column_name:
        return values

    if finite.abs().max() <= 1.000001:
        return values * 100.0

    return values


def clean_category(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "None": pd.NA,
            "<NA>": pd.NA,
        }
    )
    return cleaned.fillna("UNKNOWN")


def new_figure(
    width: float = 9.0,
    height: float = 5.5,
) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=(width, height))
    return figure, axis


def save_figure(
    figure: plt.Figure,
    output_directory: Path,
    filename_stem: str,
    dpi: int,
    show: bool,
) -> list[Path]:
    """
    Save both PNG and PDF versions.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    png_path = output_directory / f"{filename_stem}.png"
    pdf_path = output_directory / f"{filename_stem}.pdf"

    figure.tight_layout()
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(figure)
    return [png_path, pdf_path]


def add_percentage_axis(axis: plt.Axes) -> None:
    axis.set_ylim(0, 100)
    axis.set_ylabel("Persistence (%)")
    axis.grid(axis="y", alpha=0.25)


def shorten_label(value: object, max_length: int = 45) -> str:
    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 1] + "…"


def compatible_boxplot(
    axis: plt.Axes,
    values: list[np.ndarray],
    labels: list[str],
    **kwargs: object,
) -> dict[str, object]:
    """
    Support both newer Matplotlib versions (tick_labels) and older versions
    (labels).
    """
    try:
        return axis.boxplot(
            values,
            tick_labels=labels,
            **kwargs,
        )
    except TypeError:
        return axis.boxplot(
            values,
            labels=labels,
            **kwargs,
        )


def record(
    report: list[dict[str, str]],
    chart: str,
    status: str,
    details: str,
) -> None:
    report.append(
        {
            "chart": chart,
            "status": status,
            "details": details,
        }
    )


# ---------------------------------------------------------------------------
# Pair-level charts
# ---------------------------------------------------------------------------

def chart_pair_persistence_distribution(
    dataframe: pd.DataFrame,
    output_directory: Path,
    dpi: int,
    show: bool,
    report: list[dict[str, str]],
) -> None:
    chart_name = "01_pair_persistence_distribution"
    persistence_column = find_column(
        dataframe,
        PAIR_PERSISTENCE_PERCENTAGE,
    )

    if persistence_column is None:
        record(
            report,
            chart_name,
            "skipped",
            "No pair-level persistence column was found.",
        )
        return

    values = percentage_series(
        dataframe,
        persistence_column,
    ).dropna()

    if values.empty:
        record(report, chart_name, "skipped", "No numeric persistence values.")
        return

    figure, axis = new_figure()

    bins = min(20, max(5, int(math.sqrt(len(values)))))
    axis.hist(values, bins=bins, edgecolor="black", linewidth=0.6)

    mean_value = values.mean()
    median_value = values.median()

    axis.axvline(
        mean_value,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_value:.1f}%",
    )
    axis.axvline(
        median_value,
        linestyle=":",
        linewidth=1.8,
        label=f"Median: {median_value:.1f}%",
    )

    axis.set_title("Distribution of issue persistence across repository pairs")
    axis.set_xlabel("Persistence (%)")
    axis.set_ylabel("Number of template/generated-repository pairs")
    axis.set_xlim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    paths = save_figure(
        figure,
        output_directory,
        chart_name,
        dpi,
        show,
    )
    record(report, chart_name, "generated", "; ".join(map(str, paths)))


def chart_pair_persistence_by_evolution_band(
    dataframe: pd.DataFrame,
    output_directory: Path,
    dpi: int,
    show: bool,
    report: list[dict[str, str]],
) -> None:
    chart_name = "02_pair_persistence_by_evolution_band"

    persistence_column = find_column(
        dataframe,
        PAIR_PERSISTENCE_PERCENTAGE,
    )
    band_column = find_column(
        dataframe,
        PAIR_EVOLUTION_BAND,
    )

    if persistence_column is None or band_column is None:
        record(
            report,
            chart_name,
            "skipped",
            "Persistence or evolution_band column was not found.",
        )
        return

    chart_data = pd.DataFrame(
        {
            "band": clean_category(dataframe[band_column]),
            "persistence": percentage_series(
                dataframe,
                persistence_column,
            ),
        }
    ).dropna(subset=["persistence"])

    if chart_data.empty or chart_data["band"].nunique() < 2:
        record(
            report,
            chart_name,
            "skipped",
            "Fewer than two evolution-band groups contain numeric values.",
        )
        return

    group_order = (
        chart_data.groupby("band")["persistence"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    grouped_values = [
        chart_data.loc[
            chart_data["band"] == band,
            "persistence",
        ].to_numpy()
        for band in group_order
    ]

    figure, axis = new_figure(width=10.0, height=6.0)
    compatible_boxplot(
        axis,
        grouped_values,
        [
            shorten_label(band, 28)
            for band in group_order
        ],
        showmeans=True,
    )

    axis.set_title("Pair-level persistence by repository evolution band")
    axis.set_xlabel("Evolution band")
    add_percentage_axis(axis)
    axis.tick_params(axis="x", rotation=25)

    paths = save_figure(
        figure,
        output_directory,
        chart_name,
        dpi,
        show,
    )
    record(report, chart_name, "generated", "; ".join(map(str, paths)))


def chart_pair_template_issue_count_vs_persistence(
    dataframe: pd.DataFrame,
    output_directory: Path,
    dpi: int,
    show: bool,
    report: list[dict[str, str]],
) -> None:
    chart_name = "03_pair_template_issue_count_vs_persistence"

    persistence_column = find_column(
        dataframe,
        PAIR_PERSISTENCE_PERCENTAGE,
    )
    issue_count_column = find_column(
        dataframe,
        PAIR_TEMPLATE_ISSUE_COUNT,
    )

    if persistence_column is None or issue_count_column is None:
        record(
            report,
            chart_name,
            "skipped",
            "Pair persistence or template issue-count column was not found.",
        )
        return

    chart_data = pd.DataFrame(
        {
            "template_issues": numeric_series(
                dataframe,
                issue_count_column,
            ),
            "persistence": percentage_series(
                dataframe,
                persistence_column,
            ),
        }
    ).dropna()

    if chart_data.empty:
        record(report, chart_name, "skipped", "No complete numeric rows.")
        return

    figure, axis = new_figure()
    axis.scatter(
        chart_data["template_issues"],
        chart_data["persistence"],
        alpha=0.65,
    )

    if len(chart_data) >= 3 and chart_data["template_issues"].nunique() >= 2:
        coefficients = np.polyfit(
            chart_data["template_issues"],
            chart_data["persistence"],
            deg=1,
        )
        x_values = np.linspace(
            chart_data["template_issues"].min(),
            chart_data["template_issues"].max(),
            100,
        )
        y_values = np.polyval(coefficients, x_values)
        axis.plot(x_values, y_values, linestyle="--", linewidth=1.2)

    axis.set_title("Template issue count versus pair-level persistence")
    axis.set_xlabel("Issues in template repository")
    add_percentage_axis(axis)

    paths = save_figure(
        figure,
        output_directory,
        chart_name,
        dpi,
        show,
    )
    record(report, chart_name, "generated", "; ".join(map(str, paths)))


# ---------------------------------------------------------------------------
# Template-level charts
# ---------------------------------------------------------------------------

def chart_template_persistence_ranking(
    dataframe: pd.DataFrame,
    output_directory: Path,
    top_n: int,
    min_projects: int,
    dpi: int,
    show: bool,
    report: list[dict[str, str]],
) -> None:
    chart_name = "04_template_persistence_ranking"

    template_column = find_column(
        dataframe,
        TEMPLATE_REPOSITORY_COLUMN,
    )
    persistence_column = find_column(
        dataframe,
        TEMPLATE_PERSISTENCE_PERCENTAGE,
    )
    project_count_column = find_column(
        dataframe,
        TEMPLATE_PROJECT_COUNT,
    )

    if template_column is None or persistence_column is None:
        record(
            report,
            chart_name,
            "skipped",
            "Template repository or template persistence column was not found.",
        )
        return

    chart_data = pd.DataFrame(
        {
            "template": clean_category(dataframe[template_column]),
            "persistence": percentage_series(
                dataframe,
                persistence_column,
            ),
        }
    )

    if project_count_column is not None:
        chart_data["project_count"] = numeric_series(
            dataframe,
            project_count_column,
        )
        chart_data = chart_data.loc[
            chart_data["project_count"].fillna(0) >= min_projects
        ]

    chart_data = (
        chart_data.dropna(subset=["persistence"])
        .sort_values("persistence", ascending=False)
        .head(top_n)
        .sort_values("persistence", ascending=True)
    )

    if chart_data.empty:
        record(
            report,
            chart_name,
            "skipped",
            "No templates satisfy the filtering criteria.",
        )
        return

    figure_height = max(5.0, 0.38 * len(chart_data) + 1.5)
    figure, axis = new_figure(width=10.0, height=figure_height)

    axis.barh(
        [
            shorten_label(value, 48)
            for value in chart_data["template"]
        ],
        chart_data["persistence"],
    )

    axis.set_title(
        f"Top {len(chart_data)} template repositories by issue persistence"
    )
    axis.set_xlabel("Persistence (%)")
    axis.set_xlim(0, 100)
    axis.grid(axis="x", alpha=0.25)

    paths = save_figure(
        figure,
        output_directory,
        chart_name,
        dpi,
        show,
    )
    record(report, chart_name, "generated", "; ".join(map(str, paths)))


def chart_template_project_count_vs_persistence(
    dataframe: pd.DataFrame,
    output_directory: Path,
    min_projects: int,
    dpi: int,
    show: bool,
    report: list[dict[str, str]],
) -> None:
    chart_name = "05_template_project_count_vs_persistence"

    persistence_column = find_column(
        dataframe,
        TEMPLATE_PERSISTENCE_PERCENTAGE,
    )
    project_count_column = find_column(
        dataframe,
        TEMPLATE_PROJECT_COUNT,
    )

    if persistence_column is None or project_count_column is None:
        record(
            report,
            chart_name,
            "skipped",
            "Template persistence or generated-repository-count column missing.",
        )
        return

    chart_data = pd.DataFrame(
        {
            "project_count": numeric_series(
                dataframe,
                project_count_column,
            ),
            "persistence": percentage_series(
                dataframe,
                persistence_column,
            ),
        }
    ).dropna()

    chart_data = chart_data.loc[
        chart_data["project_count"] >= min_projects
    ]

    if chart_data.empty:
        record(report, chart_name, "skipped", "No complete numeric rows.")
        return

    figure, axis = new_figure()
    axis.scatter(
        chart_data["project_count"],
        chart_data["persistence"],
        alpha=0.7,
    )

    axis.set_title(
        "Number of generated repositories versus template persistence"
    )
    axis.set_xlabel("Generated repositories compared")
    add_percentage_axis(axis)

    paths = save_figure(
        figure,
        output_directory,
        chart_name,
        dpi,
        show,
    )
    record(report, chart_name, "generated", "; ".join(map(str, paths)))


# ---------------------------------------------------------------------------
# Issue-level charts
# ---------------------------------------------------------------------------

def chart_issue_persistence_distribution(
    dataframe: pd.DataFrame,
    output_directory: Path,
    dpi: int,
    show: bool,
    report: list[dict[str, str]],
) -> None:
    chart_name = "06_issue_persistence_distribution"

    persistence_column = find_column(
        dataframe,
        ISSUE_PERSISTENCE_PERCENTAGE,
    )

    if persistence_column is None:
        record(
            report,
            chart_name,
            "skipped",
            "No issue-level persistence column was found.",
        )
        return

    values = percentage_series(
        dataframe,
        persistence_column,
    ).dropna()

    if values.empty:
        record(report, chart_name, "skipped", "No numeric persistence values.")
        return

    figure, axis = new_figure()

    bins = min(20, max(5, int(math.sqrt(len(values)))))
    axis.hist(values, bins=bins, edgecolor="black", linewidth=0.6)

    axis.set_title(
        "Distribution of template-issue persistence across generated repositories"
    )
    axis.set_xlabel("Persistence across generated repositories (%)")
    axis.set_ylabel("Number of template issues")
    axis.set_xlim(0, 100)
    axis.grid(axis="y", alpha=0.25)

    paths = save_figure(
        figure,
        output_directory,
        chart_name,
        dpi,
        show,
    )
    record(report, chart_name, "generated", "; ".join(map(str, paths)))


def category_boxplot(
    dataframe: pd.DataFrame,
    category_candidates: list[str],
    title: str,
    x_label: str,
    filename_stem: str,
    output_directory: Path,
    dpi: int,
    show: bool,
    report: list[dict[str, str]],
    minimum_group_size: int = 2,
) -> None:
    persistence_column = find_column(
        dataframe,
        ISSUE_PERSISTENCE_PERCENTAGE,
    )
    category_column = find_column(
        dataframe,
        category_candidates,
    )

    if persistence_column is None or category_column is None:
        record(
            report,
            filename_stem,
            "skipped",
            "Persistence or category column was not found.",
        )
        return

    chart_data = pd.DataFrame(
        {
            "category": clean_category(dataframe[category_column]),
            "persistence": percentage_series(
                dataframe,
                persistence_column,
            ),
        }
    ).dropna(subset=["persistence"])

    group_sizes = chart_data.groupby("category").size()
    valid_categories = group_sizes.loc[
        group_sizes >= minimum_group_size
    ].index

    chart_data = chart_data.loc[
        chart_data["category"].isin(valid_categories)
    ]

    if chart_data.empty or chart_data["category"].nunique() < 2:
        record(
            report,
            filename_stem,
            "skipped",
            "Fewer than two categories meet the minimum group size.",
        )
        return

    category_order = (
        chart_data.groupby("category")["persistence"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    grouped_values = [
        chart_data.loc[
            chart_data["category"] == category,
            "persistence",
        ].to_numpy()
        for category in category_order
    ]

    figure_height = max(5.5, 0.45 * len(category_order) + 1.5)
    figure, axis = new_figure(width=10.0, height=figure_height)

    compatible_boxplot(
        axis,
        grouped_values,
        [
            shorten_label(category, 45)
            for category in category_order
        ],
        vert=False,
        showmeans=True,
    )

    axis.set_title(title)
    axis.set_xlabel("Persistence (%)")
    axis.set_ylabel(x_label)
    axis.set_xlim(0, 100)
    axis.grid(axis="x", alpha=0.25)

    paths = save_figure(
        figure,
        output_directory,
        filename_stem,
        dpi,
        show,
    )
    record(
        report,
        filename_stem,
        "generated",
        "; ".join(map(str, paths)),
    )


def chart_top_rules_by_persistence(
    dataframe: pd.DataFrame,
    output_directory: Path,
    top_n: int,
    min_observations: int,
    dpi: int,
    show: bool,
    report: list[dict[str, str]],
) -> None:
    chart_name = "09_top_rules_by_persistence"

    rule_column = find_column(
        dataframe,
        ISSUE_RULE_COLUMN,
    )
    compared_column = find_column(
        dataframe,
        ISSUE_COMPARED_PROJECT_COUNT,
    )
    persistent_column = find_column(
        dataframe,
        ISSUE_PERSISTENT_PROJECT_COUNT,
    )

    if (
        rule_column is None
        or compared_column is None
        or persistent_column is None
    ):
        record(
            report,
            chart_name,
            "skipped",
            "Rule, compared-project-count, or persistent-project-count missing.",
        )
        return

    working = pd.DataFrame(
        {
            "rule": clean_category(dataframe[rule_column]),
            "compared": numeric_series(dataframe, compared_column),
            "persistent": numeric_series(dataframe, persistent_column),
        }
    ).dropna(subset=["compared", "persistent"])

    if working.empty:
        record(report, chart_name, "skipped", "No complete rule observations.")
        return

    summary = (
        working.groupby("rule", as_index=False)
        .agg(
            issue_count=("rule", "size"),
            compared_project_observations=("compared", "sum"),
            persistent_project_observations=("persistent", "sum"),
        )
    )

    summary["persistence_percentage"] = np.where(
        summary["compared_project_observations"] > 0,
        (
            summary["persistent_project_observations"]
            / summary["compared_project_observations"]
            * 100.0
        ),
        np.nan,
    )

    summary = summary.loc[
        summary["compared_project_observations"] >= min_observations
    ]

    summary = (
        summary.dropna(subset=["persistence_percentage"])
        .sort_values(
            [
                "persistence_percentage",
                "compared_project_observations",
            ],
            ascending=[False, False],
        )
        .head(top_n)
        .sort_values("persistence_percentage", ascending=True)
    )

    summary_path = output_directory / "issue_rule_persistence_summary.csv"
    output_directory.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    if summary.empty:
        record(
            report,
            chart_name,
            "skipped",
            (
                "No rules satisfy --min-rule-observations="
                f"{min_observations}. Derived table saved to {summary_path}."
            ),
        )
        return

    figure_height = max(5.0, 0.4 * len(summary) + 1.5)
    figure, axis = new_figure(width=10.0, height=figure_height)

    axis.barh(
        [
            shorten_label(rule, 48)
            for rule in summary["rule"]
        ],
        summary["persistence_percentage"],
    )

    axis.set_title(
        f"Top {len(summary)} SonarQube rules by issue persistence"
    )
    axis.set_xlabel("Persistence (%)")
    axis.set_ylabel("SonarQube rule")
    axis.set_xlim(0, 100)
    axis.grid(axis="x", alpha=0.25)

    paths = save_figure(
        figure,
        output_directory,
        chart_name,
        dpi,
        show,
    )
    record(
        report,
        chart_name,
        "generated",
        "; ".join(map(str, paths)) + f"; {summary_path}",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    if args.top_n <= 0:
        raise SystemExit("--top-n must be greater than zero.")

    if args.min_projects_per_template <= 0:
        raise SystemExit(
            "--min-projects-per-template must be greater than zero."
        )

    if args.min_rule_observations <= 0:
        raise SystemExit(
            "--min-rule-observations must be greater than zero."
        )

    if args.dpi <= 0:
        raise SystemExit("--dpi must be greater than zero.")

    output_directory = args.output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    pair_dataframe = read_csv(
        args.pair_summary.expanduser().resolve(),
        "pair_summary.csv",
    )
    template_dataframe = read_csv(
        args.template_summary.expanduser().resolve(),
        "template_summary.csv",
    )
    issue_dataframe = read_csv(
        args.template_issue_persistence.expanduser().resolve(),
        "template_issue_persistence.csv",
    )

    report: list[dict[str, str]] = []

    chart_pair_persistence_distribution(
        pair_dataframe,
        output_directory,
        args.dpi,
        args.show,
        report,
    )

    chart_pair_persistence_by_evolution_band(
        pair_dataframe,
        output_directory,
        args.dpi,
        args.show,
        report,
    )

    chart_pair_template_issue_count_vs_persistence(
        pair_dataframe,
        output_directory,
        args.dpi,
        args.show,
        report,
    )

    chart_template_persistence_ranking(
        template_dataframe,
        output_directory,
        args.top_n,
        args.min_projects_per_template,
        args.dpi,
        args.show,
        report,
    )

    chart_template_project_count_vs_persistence(
        template_dataframe,
        output_directory,
        args.min_projects_per_template,
        args.dpi,
        args.show,
        report,
    )

    chart_issue_persistence_distribution(
        issue_dataframe,
        output_directory,
        args.dpi,
        args.show,
        report,
    )

    category_boxplot(
        issue_dataframe,
        ISSUE_SEVERITY_COLUMN,
        "Template-issue persistence by severity",
        "Severity",
        "07_issue_persistence_by_severity",
        output_directory,
        args.dpi,
        args.show,
        report,
    )

    category_boxplot(
        issue_dataframe,
        ISSUE_TYPE_COLUMN,
        "Template-issue persistence by issue type",
        "Issue type",
        "08_issue_persistence_by_type",
        output_directory,
        args.dpi,
        args.show,
        report,
    )

    chart_top_rules_by_persistence(
        issue_dataframe,
        output_directory,
        args.top_n,
        args.min_rule_observations,
        args.dpi,
        args.show,
        report,
    )

    report_dataframe = pd.DataFrame(report)
    report_path = output_directory / "chart_generation_report.csv"
    report_dataframe.to_csv(report_path, index=False)

    generated_count = int(
        (report_dataframe["status"] == "generated").sum()
    )
    skipped_count = int(
        (report_dataframe["status"] == "skipped").sum()
    )

    print("\nChart generation completed")
    print(f"Generated charts: {generated_count}")
    print(f"Skipped charts:   {skipped_count}")
    print(f"Output directory: {output_directory}")
    print(f"Generation report: {report_path}")

    if skipped_count:
        print(
            "Some charts were skipped because their required columns "
            "or sufficient category values were unavailable. "
            "See chart_generation_report.csv."
        )


if __name__ == "__main__":
    main()
