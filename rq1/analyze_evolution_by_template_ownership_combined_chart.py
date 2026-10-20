#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests


LANGUAGE_CONFIG = {
    "C#": {
        "aliases": ["C#", "Csharp", "CSharp", "csharp"],
        "template_tokens": ["C#", "Csharp", "CSharp", "csharp"],
        "generated_tokens": ["Csharp", "C#", "CSharp", "csharp"],
    },
    "Java": {
        "aliases": ["Java", "java"],
        "template_tokens": ["Java", "java"],
        "generated_tokens": ["Java", "java"],
    },
    "Python": {
        "aliases": ["Python", "python"],
        "template_tokens": ["Python", "python"],
        "generated_tokens": ["Python", "python"],
    },
    "TypeScript": {
        "aliases": ["TypeScript", "Typescript", "typescript"],
        "template_tokens": ["TypeScript", "Typescript", "typescript"],
        "generated_tokens": ["TypeScript", "Typescript", "typescript"],
    },
    "JavaScript": {
        "aliases": ["JavaScript", "Javascript", "javascript"],
        "template_tokens": ["JavaScript", "Javascript", "javascript"],
        "generated_tokens": ["JavaScript", "Javascript", "javascript"],
    },
}

LANGUAGE_ORDER = ["C#", "Java", "Python", "TypeScript", "JavaScript"]
OWNER_ORDER = ["User", "Organization"]

EVOLUTION_LEVEL_ORDER = [
    "Not evolved",
    "Slightly evolved",
    "Moderately evolved",
    "Substantially evolved",
]

EVOLUTION_COLORS = {
    "Not evolved": "#B8B8B8",
    "Slightly evolved": "#6FA8DC",
    "Moderately evolved": "#F6A04D",
    "Substantially evolved": "#D96B6B",
}


def percentage(part, total):
    return round(part / total * 100, 2) if total else 0.0


def validate_columns(df, required, file_name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in {file_name}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_repo_name(username, project_name):
    if pd.isna(username) or pd.isna(project_name):
        return None
    username = str(username).strip().lower()
    project_name = str(project_name).strip().lower()
    if not username or not project_name:
        return None
    return f"{username}/{project_name}"


def normalize_template_reference(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower().rstrip("/")
    if not value:
        return None

    if value.endswith(".git"):
        value = value[:-4]

    for prefix in [
        "https://github.com/",
        "http://github.com/",
        "git@github.com:",
        "github.com/",
    ]:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break

    parts = value.split("/")
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else value


def normalize_owner_type(value):
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value == "user":
        return "User"
    if value in {"organization", "organisation", "org"}:
        return "Organization"

    return None


def classify_evolution_level(commit_count):
    if pd.isna(commit_count):
        return None
    if commit_count <= 1:
        return "Not evolved"
    if commit_count <= 6:
        return "Slightly evolved"
    if commit_count <= 20:
        return "Moderately evolved"
    return "Substantially evolved"


def canonicalize_language(language):
    language = language.strip()

    for canonical, config in LANGUAGE_CONFIG.items():
        aliases = config["aliases"]
        if language in aliases:
            return canonical
        if language.lower() in [a.lower() for a in aliases]:
            return canonical

    raise ValueError(f"Unsupported language: {language}")


def parse_languages(value):
    result = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        canonical = canonicalize_language(raw)
        if canonical not in result:
            result.append(canonical)
    return result


def resolve_existing_file(base_dir, pattern, language, token_type):
    config = LANGUAGE_CONFIG[language]

    if token_type == "template":
        tokens = config["template_tokens"]
    else:
        tokens = config["generated_tokens"]

    candidates = []

    for token in tokens:
        filename = pattern.format(
            language=language,
            template_token=token,
            generated_token=token,
        )
        candidates.append(base_dir / filename)

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No input file found for {language}. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


def build_template_owner_map(templates_df, templates_csv):
    validate_columns(
        templates_df,
        ["username", "project_name", "owner_type"],
        templates_csv,
    )

    df = templates_df.copy()

    df["template_repo"] = df.apply(
        lambda row: normalize_repo_name(
            row["username"],
            row["project_name"],
        ),
        axis=1,
    )

    df["owner_type_normalized"] = df["owner_type"].apply(
        normalize_owner_type
    )

    df = df.dropna(subset=["template_repo"]).copy()

    conflicts = (
        df.dropna(subset=["owner_type_normalized"])
        .groupby("template_repo")["owner_type_normalized"]
        .nunique()
    )

    conflicts = conflicts[conflicts > 1]

    if not conflicts.empty:
        raise ValueError(
            "Conflicting owner_type values found for templates: "
            + ", ".join(conflicts.index.tolist()[:10])
        )

    return (
        df[["template_repo", "owner_type_normalized"]]
        .drop_duplicates("template_repo")
        .rename(columns={"owner_type_normalized": "owner_type"})
    )


def summarize_template_ownership(language, owner_map, generated_df):
    templates_with_generated = set(
        generated_df["template_repo"].dropna().unique()
    )

    rows = []

    for owner_type in OWNER_ORDER:
        owned = owner_map[
            owner_map["owner_type"] == owner_type
        ]["template_repo"]

        total_templates = owned.nunique()
        with_repos = owned[owned.isin(templates_with_generated)].nunique()

        rows.append({
            "language": language,
            "owner_type": owner_type,
            "total_templates": total_templates,
            "templates_with_generated_repositories": with_repos,
            "templates_without_generated_repositories":
                total_templates - with_repos,
            "templates_with_generated_repositories_percentage":
                percentage(with_repos, total_templates),
        })

    return pd.DataFrame(rows)


def summarize_evolution(df, language_label):
    rows = []

    for owner_type in OWNER_ORDER:
        group = df[df["owner_type"] == owner_type].copy()
        total = len(group)

        counts = (
            group["evolution_level"]
            .value_counts()
            .reindex(EVOLUTION_LEVEL_ORDER, fill_value=0)
        )

        row = {
            "language": language_label,
            "owner_type": owner_type,
            "generated_repositories_analyzed": total,
        }

        for level in EVOLUTION_LEVEL_ORDER:
            key = level.lower().replace(" ", "_")
            count = int(counts[level])
            row[f"{key}_count"] = count
            row[f"{key}_percentage"] = percentage(count, total)

        not_evolved = int(counts["Not evolved"])
        evolved = total - not_evolved

        row["evolved_repositories"] = evolved
        row["evolved_percentage"] = percentage(evolved, total)

        if total:
            row["median_commits"] = float(group["commit_count"].median())
            row["mean_commits"] = round(float(group["commit_count"].mean()), 2)
            row["min_commits"] = float(group["commit_count"].min())
            row["max_commits"] = float(group["commit_count"].max())
        else:
            row["median_commits"] = None
            row["mean_commits"] = None
            row["min_commits"] = None
            row["max_commits"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def analyze_language(
    language,
    templates_csv,
    generated_csv,
    min_stars,
):
    templates_df = pd.read_csv(templates_csv)
    generated_df = pd.read_csv(generated_csv)

    owner_map = build_template_owner_map(
        templates_df,
        str(templates_csv),
    )

    validate_columns(
        generated_df,
        [
            "username",
            "project_name",
            "target_template_repository",
            "commit_count",
            "stars",
        ],
        str(generated_csv),
    )

    generated_df = generated_df.copy()

    generated_df["generated_repo"] = generated_df.apply(
        lambda row: normalize_repo_name(
            row["username"],
            row["project_name"],
        ),
        axis=1,
    )

    generated_df["template_repo"] = (
        generated_df["target_template_repository"]
        .apply(normalize_template_reference)
    )

    generated_df["commit_count"] = pd.to_numeric(
        generated_df["commit_count"],
        errors="coerce",
    )

    generated_df["stars"] = pd.to_numeric(
        generated_df["stars"],
        errors="coerce",
    ).fillna(0)

    generated_df = generated_df.dropna(
        subset=["generated_repo", "template_repo"]
    ).copy()

    template_ownership = summarize_template_ownership(
        language,
        owner_map,
        generated_df,
    )

    total_input = len(generated_df)

    generated_df = generated_df[
        generated_df["stars"] >= min_stars
    ].copy()

    after_star_filter = len(generated_df)

    merged = generated_df.merge(
        owner_map,
        on="template_repo",
        how="left",
        validate="many_to_one",
    )

    missing_owner = int(merged["owner_type"].isna().sum())
    missing_commits = int(merged["commit_count"].isna().sum())

    valid = merged.dropna(
        subset=["owner_type", "commit_count"]
    ).copy()

    valid["evolution_level"] = valid["commit_count"].apply(
        classify_evolution_level
    )

    evolution_summary = summarize_evolution(
        valid,
        language,
    )

    quality = pd.DataFrame([{
        "language": language,
        "generated_repositories_in_input": total_input,
        "generated_repositories_after_star_filter": after_star_filter,
        "generated_repositories_missing_commit_count": missing_commits,
        "generated_repositories_without_template_owner_type": missing_owner,
        "generated_repositories_analyzed": len(valid),
    }])

    detailed = valid[
        [
            "generated_repo",
            "template_repo",
            "owner_type",
            "commit_count",
            "evolution_level",
        ]
    ].copy()

    detailed.insert(0, "language", language)

    return template_ownership, evolution_summary, quality, detailed



def build_owner_evolution_contingency_table(df):
    """
    Build the 2 x 4 contingency table used by the ownership test.

    Rows:
        User, Organization

    Columns:
        Not evolved, Slightly evolved, Moderately evolved,
        Substantially evolved
    """
    table = pd.crosstab(
        df["owner_type"],
        df["evolution_level"],
    )

    table = table.reindex(
        index=OWNER_ORDER,
        columns=EVOLUTION_LEVEL_ORDER,
        fill_value=0,
    )

    return table


def cramers_v_from_chi_square(chi_square, n, rows, columns):
    """Compute Cramer's V for a contingency table."""
    if n <= 0:
        return np.nan

    denominator = n * min(rows - 1, columns - 1)

    if denominator <= 0:
        return np.nan

    return float(np.sqrt(chi_square / denominator))


def run_ownership_chi_square_test(df, language_label):
    """
    Test whether the distribution of the four evolution levels differs
    between User- and Organization-owned source templates.

    Returns one row with the Chi-square statistic, p-value, Cramer's V,
    and diagnostics for the expected cell counts.
    """
    table = build_owner_evolution_contingency_table(df)

    # A 2 x 4 test is only meaningful if both owner types are represented
    # and at least two evolution categories have non-zero total counts.
    nonzero_rows = int((table.sum(axis=1) > 0).sum())
    nonzero_columns = int((table.sum(axis=0) > 0).sum())

    result = {
        "language": language_label,
        "user_repositories": int(table.loc["User"].sum()),
        "organization_repositories": int(table.loc["Organization"].sum()),
        "repositories_analyzed": int(table.to_numpy().sum()),
        "chi_square": np.nan,
        "degrees_of_freedom": np.nan,
        "p_value": np.nan,
        "cramers_v": np.nan,
        "minimum_expected_count": np.nan,
        "expected_cells_below_5": np.nan,
        "expected_cells_total": np.nan,
        "chi_square_expected_count_check": False,
        "test_performed": False,
    }

    if nonzero_rows < 2 or nonzero_columns < 2:
        return result

    # Remove evolution columns that are entirely empty. This avoids
    # zero expected-frequency errors while preserving the test meaning.
    test_table = table.loc[:, table.sum(axis=0) > 0]

    chi_square, p_value, dof, expected = chi2_contingency(
        test_table.to_numpy(),
        correction=False,
    )

    n = int(test_table.to_numpy().sum())
    rows, columns = test_table.shape

    cramers_v = cramers_v_from_chi_square(
        chi_square=chi_square,
        n=n,
        rows=rows,
        columns=columns,
    )

    expected_cells_below_5 = int((expected < 5).sum())
    expected_cells_total = int(expected.size)

    result.update({
        "chi_square": float(chi_square),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
        "minimum_expected_count": float(expected.min()),
        "expected_cells_below_5": expected_cells_below_5,
        "expected_cells_total": expected_cells_total,
        "chi_square_expected_count_check": expected_cells_below_5 == 0,
        "test_performed": True,
    })

    return result


def run_language_statistical_tests(detailed_df, alpha=0.05):
    """
    Run one 2 x 4 Chi-square test per language and correct the five
    language-level p-values using Benjamini-Hochberg FDR correction.
    """
    rows = []

    for language in LANGUAGE_ORDER:
        language_df = detailed_df[
            detailed_df["language"] == language
        ].copy()

        if language_df.empty:
            continue

        rows.append(
            run_ownership_chi_square_test(
                language_df,
                language,
            )
        )

    results = pd.DataFrame(rows)

    if results.empty:
        return results

    results["p_value_bh"] = np.nan
    results["significant_bh_0_05"] = False

    valid_mask = (
        results["test_performed"]
        & results["p_value"].notna()
    )

    if valid_mask.any():
        _, corrected_p_values, _, _ = multipletests(
            results.loc[valid_mask, "p_value"].to_numpy(dtype=float),
            alpha=alpha,
            method="fdr_bh",
        )

        results.loc[valid_mask, "p_value_bh"] = corrected_p_values
        results.loc[valid_mask, "significant_bh_0_05"] = (
            corrected_p_values < alpha
        )

    return results


def run_overall_statistical_test(detailed_df):
    """
    Run the same 2 x 4 Chi-square test after pooling all languages.

    This result is reported separately and is NOT included in the
    Benjamini-Hochberg correction across language-specific tests.
    """
    result = run_ownership_chi_square_test(
        detailed_df,
        "Overall",
    )

    return pd.DataFrame([result])


def format_p_value(value):
    """Format p-values compactly for console reporting."""
    if pd.isna(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def print_statistical_test_report(language_tests, overall_test):
    print("\nStatistical tests: owner type x evolution level")
    print("-" * 80)
    print(
        "Per-language tests use a 2 x 4 Chi-square test of independence; "
        "p-values are adjusted with Benjamini-Hochberg across languages."
    )

    for _, row in language_tests.iterrows():
        if not bool(row["test_performed"]):
            print(f"- {row['language']}: test not performed")
            continue

        significance = (
            "significant"
            if bool(row["significant_bh_0_05"])
            else "not significant"
        )

        print(
            f"- {row['language']}: "
            f"chi2={row['chi_square']:.3f}, "
            f"df={int(row['degrees_of_freedom'])}, "
            f"p={format_p_value(row['p_value'])}, "
            f"BH-adjusted p={format_p_value(row['p_value_bh'])}, "
            f"Cramer's V={row['cramers_v']:.3f} "
            f"({significance} at alpha=0.05)"
        )

        if not bool(row["chi_square_expected_count_check"]):
            print(
                "  Warning: at least one expected cell count is below 5 "
                f"(minimum={row['minimum_expected_count']:.3f})."
            )

    if not overall_test.empty:
        row = overall_test.iloc[0]
        if bool(row["test_performed"]):
            print(
                "- Overall (pooled; descriptive inferential check, not part "
                "of BH correction): "
                f"chi2={row['chi_square']:.3f}, "
                f"df={int(row['degrees_of_freedom'])}, "
                f"p={format_p_value(row['p_value'])}, "
                f"Cramer's V={row['cramers_v']:.3f}"
            )

    print("-" * 80)

def add_percentage_labels(
    ax,
    x,
    values_by_level,
    min_inside=5.0,
    fontsize=8,
):
    cumulative = np.zeros(len(x), dtype=float)

    for index, level in enumerate(EVOLUTION_LEVEL_ORDER):
        values = np.asarray(values_by_level[level], dtype=float)
        is_top = index == len(EVOLUTION_LEVEL_ORDER) - 1

        for i, value in enumerate(values):
            if value <= 0:
                continue

            if is_top and value < min_inside:
                ax.text(
                    x[i],
                    101.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    clip_on=False,
                )
            elif value >= min_inside:
                ax.text(
                    x[i],
                    cumulative[i] + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )

        cumulative += values


def save_figure(fig, output_dir, name):
    output_dir.mkdir(parents=True, exist_ok=True)

    png = output_dir / f"{name}.png"
    pdf = output_dir / f"{name}.pdf"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    return [png, pdf]


def generate_language_figure(summary_df, detailed_df, output_dir):
    """
    Generate one compact figure with three stacked bars per language:

        All | Org. | User

    "All" is computed from the same ownership-valid repositories used by
    the User/Organization analysis, so All = Organization + User.
    """
    rows = []
    x_positions = []
    labels = []
    language_group_centers = []
    current_x = 0.0
    within_group_gap = 0.82
    between_group_gap = 0.72

    for language in LANGUAGE_ORDER:
        language_summary = summary_df[summary_df["language"] == language].copy()
        language_detailed = detailed_df[detailed_df["language"] == language].copy()

        if language_summary.empty or language_detailed.empty:
            continue

        # Combined distribution from the exact same ownership-valid sample.
        total_all = len(language_detailed)
        combined_counts = (
            language_detailed["evolution_level"]
            .value_counts()
            .reindex(EVOLUTION_LEVEL_ORDER, fill_value=0)
        )

        combined_row = {
            "language": language,
            "owner_type": "All",
            "generated_repositories_analyzed": total_all,
        }
        for level in EVOLUTION_LEVEL_ORDER:
            key = level.lower().replace(" ", "_")
            count = int(combined_counts[level])
            combined_row[f"{key}_count"] = count
            combined_row[f"{key}_percentage"] = percentage(count, total_all)

        group_positions = []

        rows.append(pd.Series(combined_row))
        x_positions.append(current_x)
        group_positions.append(current_x)
        labels.append("All")
        current_x += within_group_gap

        organization_row = language_summary[
            language_summary["owner_type"] == "Organization"
        ]
        if not organization_row.empty:
            rows.append(organization_row.iloc[0])
            x_positions.append(current_x)
            group_positions.append(current_x)
            labels.append("Org.")
            current_x += within_group_gap

        user_row = language_summary[
            language_summary["owner_type"] == "User"
        ]
        if not user_row.empty:
            rows.append(user_row.iloc[0])
            x_positions.append(current_x)
            group_positions.append(current_x)
            labels.append("User")
            current_x += within_group_gap

        if group_positions:
            language_group_centers.append(
                (language, sum(group_positions) / len(group_positions))
            )

        current_x += between_group_gap

    if not rows:
        return []

    x = np.array(x_positions, dtype=float)
    values_by_level = {}

    for level in EVOLUTION_LEVEL_ORDER:
        key = level.lower().replace(" ", "_") + "_percentage"
        values_by_level[level] = np.array(
            [float(row[key]) for row in rows],
            dtype=float,
        )

    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    bottom = np.zeros(len(x), dtype=float)

    for level in EVOLUTION_LEVEL_ORDER:
        values = values_by_level[level]
        ax.bar(
            x,
            values,
            width=0.68,
            bottom=bottom,
            label=level,
            color=EVOLUTION_COLORS[level],
            edgecolor="white",
            linewidth=0.7,
        )
        bottom += values

    # Show labels for segments >= 5%; retain the existing treatment for
    # small top (Substantially evolved) segments.
    add_percentage_labels(
        ax,
        x,
        values_by_level,
        min_inside=5.0,
        fontsize=7.5,
    )

    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Repositories (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)

    # Show each language once, centered beneath its three ownership bars.
    # This prevents TypeScript/JavaScript from overlapping.
    for language, center in language_group_centers:
        ax.text(
            center,
            -0.115,
            language,
            ha="center",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
        )

    ax.grid(axis="y", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        frameon=True,
        title="Evolution level",
        fontsize=9,
        title_fontsize=10,
        fancybox=False,
        framealpha=1.0,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("0.55")

    fig.subplots_adjust(
        left=0.075,
        right=0.80,
        bottom=0.23,
        top=0.97,
    )

    return save_figure(
        fig,
        output_dir,
        "evolution_levels_combined_and_by_template_ownership",
    )

def generate_overall_figure(summary_df, output_dir):
    x = np.arange(2, dtype=float)

    values_by_level = {}

    for level in EVOLUTION_LEVEL_ORDER:
        key = level.lower().replace(" ", "_") + "_percentage"
        values_by_level[level] = np.array([
            float(
                summary_df[
                    summary_df["owner_type"] == owner_type
                ].iloc[0][key]
            )
            for owner_type in OWNER_ORDER
        ])

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    bottom = np.zeros(2, dtype=float)

    for level in EVOLUTION_LEVEL_ORDER:
        values = values_by_level[level]

        ax.bar(
            x,
            values,
            width=0.72,
            bottom=bottom,
            label=level,
            color=EVOLUTION_COLORS[level],
            edgecolor="white",
            linewidth=0.7,
        )

        bottom += values

    add_percentage_labels(
        ax,
        x,
        values_by_level,
        min_inside=4.0,
        fontsize=9,
    )

    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    ax.set_ylabel("Repositories (%)", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["User-owned templates", "Organization-owned templates"],
        fontsize=10,
    )

    ax.grid(axis="y", alpha=0.20, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        title="Evolution level",
        fontsize=9,
        title_fontsize=10,
        fancybox=False,
        framealpha=1.0,
    )

    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_edgecolor("0.55")

    fig.subplots_adjust(
        left=0.13,
        right=0.70,
        bottom=0.17,
        top=0.97,
    )

    return save_figure(
        fig,
        output_dir,
        "ownership_evolution_overall",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze generated-repository evolution according to "
            "the owner type of the source template."
        )
    )

    parser.add_argument(
        "--templates_dir",
        required=True,
        help="Directory containing all_templates_<LANGUAGE>.csv files.",
    )

    parser.add_argument(
        "--generated_dir",
        required=True,
        help=(
            "Directory containing "
            "generated_GHRep_from_templates_<LANGUAGE>_with_commits.csv files."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default="ownership_evolution_results",
        help="Output directory.",
    )

    parser.add_argument(
        "--languages",
        default="C#,Java,Python,TypeScript,JavaScript",
    )

    parser.add_argument(
        "--min_stars",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--templates_pattern",
        default="all_templates_{template_token}.csv",
    )

    parser.add_argument(
        "--generated_pattern",
        default="generated_GHRep_from_templates_{generated_token}_with_commits.csv",
    )

    parser.add_argument(
        "--no_charts",
        action="store_true",
    )

    parser.add_argument(
        "--skip_missing",
        action="store_true",
    )

    args = parser.parse_args()

    templates_dir = Path(args.templates_dir)
    generated_dir = Path(args.generated_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    languages = parse_languages(args.languages)

    ownership_frames = []
    evolution_frames = []
    quality_frames = []
    detailed_frames = []

    print()
    print("Template ownership x generated-repository evolution")
    print("-" * 80)

    for language in languages:
        try:
            templates_csv = resolve_existing_file(
                templates_dir,
                args.templates_pattern,
                language,
                "template",
            )

            generated_csv = resolve_existing_file(
                generated_dir,
                args.generated_pattern,
                language,
                "generated",
            )

            print(f"\nProcessing {language}")
            print(f"  Templates: {templates_csv}")
            print(f"  Generated: {generated_csv}")

            ownership, evolution, quality, detailed = analyze_language(
                language,
                templates_csv,
                generated_csv,
                args.min_stars,
            )

            ownership_frames.append(ownership)
            evolution_frames.append(evolution)
            quality_frames.append(quality)
            detailed_frames.append(detailed)

        except Exception as error:
            if not args.skip_missing:
                raise
            print(f"  Skipped: {error}")

    if not evolution_frames:
        raise RuntimeError("No language was successfully analyzed.")

    ownership_by_language = pd.concat(
        ownership_frames,
        ignore_index=True,
    )

    evolution_by_language = pd.concat(
        evolution_frames,
        ignore_index=True,
    )

    quality = pd.concat(
        quality_frames,
        ignore_index=True,
    )

    detailed = pd.concat(
        detailed_frames,
        ignore_index=True,
    )

    evolution_overall = summarize_evolution(
        detailed,
        "Overall",
    )

    language_statistical_tests = run_language_statistical_tests(
        detailed,
        alpha=0.05,
    )

    overall_statistical_test = run_overall_statistical_test(
        detailed,
    )

    outputs = {
        "template_ownership_by_language":
            output_dir / "template_ownership_by_language.csv",
        "ownership_evolution_by_language":
            output_dir / "ownership_evolution_by_language.csv",
        "ownership_evolution_overall":
            output_dir / "ownership_evolution_overall.csv",
        "ownership_analysis_data_quality":
            output_dir / "ownership_analysis_data_quality.csv",
        "ownership_evolution_detailed":
            output_dir / "ownership_evolution_detailed.csv",
        "ownership_evolution_statistical_tests_by_language":
            output_dir / "ownership_evolution_statistical_tests_by_language.csv",
        "ownership_evolution_statistical_test_overall":
            output_dir / "ownership_evolution_statistical_test_overall.csv",
    }

    ownership_by_language.to_csv(
        outputs["template_ownership_by_language"],
        index=False,
    )

    evolution_by_language.to_csv(
        outputs["ownership_evolution_by_language"],
        index=False,
    )

    evolution_overall.to_csv(
        outputs["ownership_evolution_overall"],
        index=False,
    )

    quality.to_csv(
        outputs["ownership_analysis_data_quality"],
        index=False,
    )

    detailed.to_csv(
        outputs["ownership_evolution_detailed"],
        index=False,
    )

    language_statistical_tests.to_csv(
        outputs["ownership_evolution_statistical_tests_by_language"],
        index=False,
    )

    overall_statistical_test.to_csv(
        outputs["ownership_evolution_statistical_test_overall"],
        index=False,
    )

    print("\nSaved CSV files:")
    for path in outputs.values():
        print(f"- {path}")

    print_statistical_test_report(
        language_statistical_tests,
        overall_statistical_test,
    )

    if not args.no_charts:
        charts_dir = output_dir / "charts"

        figure_paths = []

        figure_paths.extend(
            generate_language_figure(
                evolution_by_language,
                detailed,
                charts_dir,
            )
        )

        figure_paths.extend(
            generate_overall_figure(
                evolution_overall,
                charts_dir,
            )
        )

        print("\nSaved figures:")
        for path in figure_paths:
            print(f"- {path}")


if __name__ == "__main__":
    main()

# python -u analyze_evolution_by_template_ownership_combined_chart.py
# --templates_dir ../data_collection
# --generated_dir reports
# --output_dir ownership_evolution_results