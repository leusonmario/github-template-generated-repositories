import argparse
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LANGUAGE_CONFIG = {
    "C#": {
        "aliases": ["C#", "Csharp", "CSharp", "csharp"],
        "template_tokens": ["C#", "Csharp", "CSharp", "csharp"],
        "generated_tokens": ["Csharp", "C#", "CSharp", "csharp"],
        "slug": "csharp"
    },
    "Java": {
        "aliases": ["Java", "java"],
        "template_tokens": ["Java", "java"],
        "generated_tokens": ["Java", "java"],
        "slug": "java"
    },
    "Python": {
        "aliases": ["Python", "python"],
        "template_tokens": ["Python", "python"],
        "generated_tokens": ["Python", "python"],
        "slug": "python"
    },
    "TypeScript": {
        "aliases": ["TypeScript", "Typescript", "typescript"],
        "template_tokens": ["TypeScript", "Typescript", "typescript"],
        "generated_tokens": ["TypeScript", "Typescript", "typescript"],
        "slug": "typescript"
    },
    "JavaScript": {
        "aliases": ["JavaScript", "Javascript", "javascript"],
        "template_tokens": ["JavaScript", "Javascript", "javascript"],
        "generated_tokens": ["JavaScript", "Javascript", "javascript"],
        "slug": "javascript"
    }
}


EVOLUTION_LEVEL_ORDER = [
    "Not evolved",
    "Slightly evolved",
    "Moderately evolved",
    "Substantially evolved"
]


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

    value = str(value).strip().lower()

    if not value:
        return None

    value = value.rstrip("/")

    if value.endswith(".git"):
        value = value[:-4]

    prefixes = [
        "https://github.com/",
        "http://github.com/",
        "git@github.com:",
        "github.com/"
    ]

    for prefix in prefixes:
        if value.startswith(prefix):
            value = value[len(prefix):]
            break

    parts = value.split("/")

    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"

    return value


def classify_evolution_binary(commit_count):
    if pd.isna(commit_count):
        return "Unknown"

    if commit_count > 1:
        return "Evolved"

    return "Not evolved"


def classify_evolution_level(commit_count):
    if pd.isna(commit_count):
        return "Unknown"

    if commit_count <= 1:
        return "Not evolved"
    elif commit_count <= 6:
        return "Slightly evolved"
    elif commit_count <= 20:
        return "Moderately evolved"
    else:
        return "Substantially evolved"


def validate_columns(df, required_columns, csv_name):
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in {csv_name}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def percentage(part, total):
    if total == 0:
        return 0.0
    return round(part / total * 100, 2)


def analyze_language(
    language,
    templates_csv,
    generated_csv,
    output_dir,
    min_stars
):
    language_config = LANGUAGE_CONFIG[language]
    language_slug = language_config["slug"]

    language_output_dir = output_dir / language_slug
    language_output_dir.mkdir(parents=True, exist_ok=True)

    if not templates_csv.exists():
        raise FileNotFoundError(f"Template CSV not found for {language}: {templates_csv}")

    if not generated_csv.exists():
        raise FileNotFoundError(f"Generated CSV not found for {language}: {generated_csv}")

    templates_df = pd.read_csv(templates_csv)
    generated_df = pd.read_csv(generated_csv)

    validate_columns(
        templates_df,
        [
            "username",
            "project_name",
            "original_link_repository"
        ],
        str(templates_csv)
    )

    validate_columns(
        generated_df,
        [
            "username",
            "project_name",
            "stars",
            "target_template_repository",
            "commit_count"
        ],
        str(generated_csv)
    )

    templates_df["template_repo"] = templates_df.apply(
        lambda row: normalize_repo_name(row["username"], row["project_name"]),
        axis=1
    )

    generated_df["generated_repo"] = generated_df.apply(
        lambda row: normalize_repo_name(row["username"], row["project_name"]),
        axis=1
    )

    generated_df["template_repo"] = generated_df["target_template_repository"].apply(
        normalize_template_reference
    )

    templates_df = templates_df.dropna(subset=["template_repo"]).copy()
    generated_df = generated_df.dropna(subset=["generated_repo", "template_repo"]).copy()

    generated_df["stars"] = pd.to_numeric(
        generated_df["stars"],
        errors="coerce"
    ).fillna(0)

    generated_df["commit_count"] = pd.to_numeric(
        generated_df["commit_count"],
        errors="coerce"
    )

    # ------------------------------------------------------------
    # Template coverage
    # ------------------------------------------------------------

    all_templates = set(templates_df["template_repo"].unique())
    templates_with_generated = set(generated_df["template_repo"].unique())

    templates_with_generated_in_template_list = templates_with_generated & all_templates
    templates_without_generated = all_templates - templates_with_generated

    total_templates = len(all_templates)
    total_templates_with_generated = len(templates_with_generated_in_template_list)
    total_templates_without_generated = len(templates_without_generated)

    # ------------------------------------------------------------
    # Star filter
    # ------------------------------------------------------------

    total_generated_repositories_in_input = len(generated_df)

    filtered_df = generated_df[
        generated_df["stars"] >= min_stars
    ].copy()

    total_generated_after_star_filter = len(filtered_df)

    total_skipped_by_min_stars = (
        total_generated_repositories_in_input
        - total_generated_after_star_filter
    )

    # ------------------------------------------------------------
    # Evolution classification
    # ------------------------------------------------------------

    filtered_df["evolution_status"] = filtered_df["commit_count"].apply(
        classify_evolution_binary
    )

    filtered_df["evolution_level"] = filtered_df["commit_count"].apply(
        classify_evolution_level
    )

    valid_df = filtered_df[
        filtered_df["commit_count"].notna()
    ].copy()

    total_successfully_analyzed = len(valid_df)
    total_missing_commit_count = (
        total_generated_after_star_filter
        - total_successfully_analyzed
    )

    total_evolved = (
        valid_df["evolution_status"] == "Evolved"
    ).sum()

    total_not_evolved = (
        valid_df["evolution_status"] == "Not evolved"
    ).sum()

    evolved_percentage = percentage(total_evolved, total_successfully_analyzed)
    not_evolved_percentage = percentage(total_not_evolved, total_successfully_analyzed)

    # ------------------------------------------------------------
    # Evolution level distribution
    # ------------------------------------------------------------

    if valid_df.empty:
        evolution_distribution = pd.DataFrame({
            "language": [language] * len(EVOLUTION_LEVEL_ORDER),
            "evolution_level": EVOLUTION_LEVEL_ORDER,
            "count": [0] * len(EVOLUTION_LEVEL_ORDER),
            "percentage": [0.0] * len(EVOLUTION_LEVEL_ORDER)
        })
    else:
        level_counts = (
            valid_df["evolution_level"]
            .value_counts()
            .reindex(EVOLUTION_LEVEL_ORDER, fill_value=0)
            .reset_index()
        )

        level_counts.columns = ["evolution_level", "count"]

        level_counts["percentage"] = level_counts["count"].apply(
            lambda count: percentage(count, total_successfully_analyzed)
        )

        level_counts.insert(0, "language", language)

        evolution_distribution = level_counts

    # ------------------------------------------------------------
    # Per-template summary
    # ------------------------------------------------------------

    if valid_df.empty:
        per_template_summary = pd.DataFrame(columns=[
            "language",
            "template_repo",
            "total_generated_repos",
            "evolved_repos",
            "not_evolved_repos",
            "median_commits",
            "mean_commits",
            "min_commits",
            "max_commits",
            "evolved_percentage",
            "not_evolved_percentage"
        ])
    else:
        per_template_summary = (
            valid_df
            .groupby("template_repo")
            .agg(
                total_generated_repos=("generated_repo", "count"),
                evolved_repos=("evolution_status", lambda x: (x == "Evolved").sum()),
                not_evolved_repos=("evolution_status", lambda x: (x == "Not evolved").sum()),
                median_commits=("commit_count", "median"),
                mean_commits=("commit_count", "mean"),
                min_commits=("commit_count", "min"),
                max_commits=("commit_count", "max")
            )
            .reset_index()
        )

        per_template_summary["evolved_percentage"] = (
            per_template_summary["evolved_repos"]
            / per_template_summary["total_generated_repos"]
            * 100
        ).round(2)

        per_template_summary["not_evolved_percentage"] = (
            per_template_summary["not_evolved_repos"]
            / per_template_summary["total_generated_repos"]
            * 100
        ).round(2)

        per_template_summary.insert(0, "language", language)

        per_template_summary = per_template_summary.sort_values(
            by="total_generated_repos",
            ascending=False
        )

    # ------------------------------------------------------------
    # Global language summary
    # ------------------------------------------------------------

    if valid_df.empty:
        median_commits = None
        mean_commits = None
        min_commits = None
        max_commits = None
    else:
        median_commits = valid_df["commit_count"].median()
        mean_commits = round(valid_df["commit_count"].mean(), 2)
        min_commits = valid_df["commit_count"].min()
        max_commits = valid_df["commit_count"].max()

    level_count_map = dict(
        zip(
            evolution_distribution["evolution_level"],
            evolution_distribution["count"]
        )
    )

    level_percentage_map = dict(
        zip(
            evolution_distribution["evolution_level"],
            evolution_distribution["percentage"]
        )
    )

    global_summary = pd.DataFrame([
        {
            "language": language,
            "templates_csv": str(templates_csv),
            "generated_csv": str(generated_csv),
            "min_stars": min_stars,

            "total_templates": total_templates,
            "templates_with_generated_repositories": total_templates_with_generated,
            "templates_without_generated_repositories": total_templates_without_generated,

            "total_generated_repositories_in_input": total_generated_repositories_in_input,
            "generated_repositories_skipped_by_min_stars": total_skipped_by_min_stars,
            "generated_repositories_after_star_filter": total_generated_after_star_filter,
            "generated_repositories_with_valid_commit_count": total_successfully_analyzed,
            "generated_repositories_missing_commit_count": total_missing_commit_count,

            "evolved_generated_repositories": total_evolved,
            "not_evolved_generated_repositories": total_not_evolved,
            "evolved_generated_repositories_percentage": evolved_percentage,
            "not_evolved_generated_repositories_percentage": not_evolved_percentage,

            "not_evolved_count": level_count_map.get("Not evolved", 0),
            "slightly_evolved_count": level_count_map.get("Slightly evolved", 0),
            "moderately_evolved_count": level_count_map.get("Moderately evolved", 0),
            "highly_evolved_count": level_count_map.get("Substantially evolved", 0),

            "not_evolved_level_percentage": level_percentage_map.get("Not evolved", 0.0),
            "slightly_evolved_percentage": level_percentage_map.get("Slightly evolved", 0.0),
            "moderately_evolved_percentage": level_percentage_map.get("Moderately evolved", 0.0),
            "highly_evolved_percentage": level_percentage_map.get("Substantially evolved", 0.0),

            "median_commits": median_commits,
            "mean_commits": mean_commits,
            "min_commits": min_commits,
            "max_commits": max_commits
        }
    ])

    # ------------------------------------------------------------
    # Save files for this language
    # ------------------------------------------------------------

    global_summary_path = language_output_dir / "global_summary.csv"
    per_template_summary_path = language_output_dir / "per_template_summary.csv"
    evolution_distribution_path = language_output_dir / "evolution_level_distribution.csv"

    global_summary.to_csv(global_summary_path, index=False)
    per_template_summary.to_csv(per_template_summary_path, index=False)
    evolution_distribution.to_csv(evolution_distribution_path, index=False)

    return {
        "language": language,
        "global_summary": global_summary,
        "per_template_summary": per_template_summary,
        "evolution_distribution": evolution_distribution,
        "global_summary_path": global_summary_path,
        "per_template_summary_path": per_template_summary_path,
        "evolution_distribution_path": evolution_distribution_path
    }


def canonicalize_language(language):
    language = language.strip()

    for canonical_language, config in LANGUAGE_CONFIG.items():
        aliases = config["aliases"]

        if language in aliases:
            return canonical_language

        if language.lower() in [alias.lower() for alias in aliases]:
            return canonical_language

    raise ValueError(
        f"Unsupported language: {language}\n"
        f"Supported languages and aliases: "
        f"{ {lang: config['aliases'] for lang, config in LANGUAGE_CONFIG.items()} }"
    )


def parse_languages(languages_argument):
    raw_languages = [
        language.strip()
        for language in languages_argument.split(",")
        if language.strip()
    ]

    canonical_languages = []

    for language in raw_languages:
        canonical_language = canonicalize_language(language)

        if canonical_language not in canonical_languages:
            canonical_languages.append(canonical_language)

    return canonical_languages

def resolve_existing_file(base_dir, pattern, language, token_type):
    """
    Try different filename variants for the same language.

    Example for C#:
    all_templates_C#.csv
    all_templates_Csharp.csv
    all_templates_CSharp.csv
    all_templates_csharp.csv
    """
    config = LANGUAGE_CONFIG[language]

    if token_type == "template":
        tokens = config["template_tokens"]
    elif token_type == "generated":
        tokens = config["generated_tokens"]
    else:
        raise ValueError(f"Unknown token_type: {token_type}")

    candidates = []

    for token in tokens:
        filename = pattern.format(
            language=language,
            template_token=token,
            generated_token=token,
            slug=config["slug"]
        )

        candidates.append(base_dir / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_list = "\n".join(str(candidate) for candidate in candidates)

    raise FileNotFoundError(
        f"No matching file found for {language}.\n"
        f"Tried:\n{candidate_list}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute template evolution results for multiple programming languages "
            "using existing commit_count values. No cloning."
        )
    )

    parser.add_argument(
        "--templates_dir",
        required=True,
        help="Directory containing all_templates_<LANGUAGE>.csv files."
    )

    parser.add_argument(
        "--generated_dir",
        required=True,
        help="Directory containing generated_GHRep_from_templates_<LANGUAGE>_with_commits.csv files."
    )

    parser.add_argument(
        "--output_dir",
        default="multi_language_evolution_results",
        help="Directory where language-based result files will be saved."
    )

    parser.add_argument(
        "--languages",
        default="C#,Java,Python,TypeScript,JavaScript",
        help="Comma-separated languages to process. Aliases such as Csharp and C# are supported."
    )

    parser.add_argument(
        "--min_stars",
        type=int,
        default=0,
        help="Optional minimum number of stars for generated repositories."
    )

    parser.add_argument(
        "--templates_pattern",
        default="all_templates_{template_token}.csv",
        help="Template CSV filename pattern."
    )

    parser.add_argument(
        "--generated_pattern",
        default="generated_GHRep_from_templates_{generated_token}_with_commits.csv",
        help="Generated CSV filename pattern."
    )

    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip languages whose input files are missing instead of stopping."
    )

    parser.add_argument(
        "--no_charts",
        action="store_true",
        help="Disable chart generation."
    )

    args = parser.parse_args()

    templates_dir = Path(args.templates_dir)
    generated_dir = Path(args.generated_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    languages = parse_languages(args.languages)

    all_global_summaries = []
    all_per_template_summaries = []
    all_evolution_distributions = []

    print()
    print("Multi-language template evolution analysis")
    print("-" * 80)
    print(f"Languages: {', '.join(languages)}")
    print(f"Minimum stars filter: {args.min_stars}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)

    for language in languages:
        config = LANGUAGE_CONFIG[language]

        templates_csv = resolve_existing_file(
            base_dir=templates_dir,
            pattern=args.templates_pattern,
            language=language,
            token_type="template"
        )

        generated_csv = resolve_existing_file(
            base_dir=generated_dir,
            pattern=args.generated_pattern,
            language=language,
            token_type="generated"
        )

        print()
        print(f"Processing {language}")
        print(f"Templates CSV: {templates_csv}")
        print(f"Generated CSV: {generated_csv}")

        try:
            result = analyze_language(
                language=language,
                templates_csv=templates_csv,
                generated_csv=generated_csv,
                output_dir=output_dir,
                min_stars=args.min_stars
            )

            all_global_summaries.append(result["global_summary"])
            all_per_template_summaries.append(result["per_template_summary"])
            all_evolution_distributions.append(result["evolution_distribution"])

            summary_row = result["global_summary"].iloc[0]

            print(f"Done: {language}")
            print(f"  Total templates: {summary_row['total_templates']}")
            print(f"  Generated repositories: {summary_row['total_generated_repositories_in_input']}")
            print(f"  Valid commit_count: {summary_row['generated_repositories_with_valid_commit_count']}")
            print(
                f"  Evolved: {summary_row['evolved_generated_repositories']} "
                f"({summary_row['evolved_generated_repositories_percentage']:.2f}%)"
            )
            print(
                f"  Not evolved: {summary_row['not_evolved_generated_repositories']} "
                f"({summary_row['not_evolved_generated_repositories_percentage']:.2f}%)"
            )

        except Exception as error:
            if not args.skip_missing:
                raise

            print(f"Skipped {language}: {error}")

            failed_summary = pd.DataFrame([
                {
                    "language": language,
                    "templates_csv": str(templates_csv),
                    "generated_csv": str(generated_csv),
                    "min_stars": args.min_stars,
                    "error": str(error)
                }
            ])

            all_global_summaries.append(failed_summary)

    # ------------------------------------------------------------
    # Save combined files across all languages
    # ------------------------------------------------------------

    if all_global_summaries:
        combined_global_summary = pd.concat(
            all_global_summaries,
            ignore_index=True
        )
    else:
        combined_global_summary = pd.DataFrame()

    if all_per_template_summaries:
        combined_per_template_summary = pd.concat(
            all_per_template_summaries,
            ignore_index=True
        )
    else:
        combined_per_template_summary = pd.DataFrame()

    if all_evolution_distributions:
        combined_evolution_distribution = pd.concat(
            all_evolution_distributions,
            ignore_index=True
        )
    else:
        combined_evolution_distribution = pd.DataFrame()

    combined_global_summary_path = output_dir / "all_languages_global_summary.csv"
    combined_per_template_summary_path = output_dir / "all_languages_per_template_summary.csv"
    combined_evolution_distribution_path = output_dir / "all_languages_evolution_level_distribution.csv"

    combined_global_summary.to_csv(combined_global_summary_path, index=False)
    combined_per_template_summary.to_csv(combined_per_template_summary_path, index=False)
    combined_evolution_distribution.to_csv(combined_evolution_distribution_path, index=False)

    print()
    print("-" * 80)
    print("Combined output files:")
    print(f"1. {combined_global_summary_path}")
    print(f"2. {combined_per_template_summary_path}")
    print(f"3. {combined_evolution_distribution_path}")
    print("-" * 80)

    if not combined_global_summary.empty:
        print()
        print("Global summary by language")
        print("-" * 80)

        display_columns = [
            "language",
            "total_templates",
            "templates_with_generated_repositories",
            "templates_without_generated_repositories",
            "total_generated_repositories_in_input",
            "generated_repositories_with_valid_commit_count",
            "evolved_generated_repositories",
            "evolved_generated_repositories_percentage",
            "not_evolved_generated_repositories",
            "not_evolved_generated_repositories_percentage"
        ]

        available_display_columns = [
            col for col in display_columns
            if col in combined_global_summary.columns
        ]

        print(
            combined_global_summary[available_display_columns]
            .to_string(index=False)
        )

        print("-" * 80)

    combined_global_summary.to_csv(combined_global_summary_path, index=False)
    combined_per_template_summary.to_csv(combined_per_template_summary_path, index=False)
    combined_evolution_distribution.to_csv(combined_evolution_distribution_path, index=False)

    chart_paths = []

    if not args.no_charts:
        chart_paths = generate_charts(
            combined_global_summary=combined_global_summary,
            combined_evolution_distribution=combined_evolution_distribution,
            output_dir=output_dir
        )

    if chart_paths:
        print()
        print("Generated charts:")
        for chart_path in chart_paths:
            print(f"- {chart_path}")

def add_value_labels(ax, bars, value_format="{:.0f}", min_height_to_label=0):
    for bar in bars:
        height = bar.get_height()

        if height <= min_height_to_label:
            continue

        ax.annotate(
            value_format.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9
        )


def save_chart(fig, charts_dir, filename):
    png_path = charts_dir / f"{filename}.png"
    pdf_path = charts_dir / f"{filename}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def generate_charts(
    combined_global_summary,
    combined_evolution_distribution,
    output_dir
):
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = []

    if combined_global_summary.empty:
        print("No global summary available. Skipping charts.")
        return chart_paths

    global_df = combined_global_summary.copy()

    required_columns = [
        "language",
        "total_templates",
        "templates_with_generated_repositories",
        "templates_without_generated_repositories",
        "evolved_generated_repositories",
        "not_evolved_generated_repositories",
        "evolved_generated_repositories_percentage",
        "not_evolved_generated_repositories_percentage"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in global_df.columns
    ]

    if missing_columns:
        print(f"Skipping charts because required columns are missing: {missing_columns}")
        return chart_paths

    # Keep only rows that actually contain computed results
    global_df = global_df.dropna(subset=["language"]).copy()

    numeric_columns = [
        "total_templates",
        "templates_with_generated_repositories",
        "templates_without_generated_repositories",
        "evolved_generated_repositories",
        "not_evolved_generated_repositories",
        "evolved_generated_repositories_percentage",
        "not_evolved_generated_repositories_percentage"
    ]

    for col in numeric_columns:
        global_df[col] = pd.to_numeric(global_df[col], errors="coerce").fillna(0)

    if global_df.empty:
        print("No valid rows available for charts.")
        return chart_paths

    languages = global_df["language"].tolist()
    x = np.arange(len(languages))

    # ------------------------------------------------------------
    # Chart 1: Evolution counts by language
    # ------------------------------------------------------------

    evolved_counts = global_df["evolved_generated_repositories"].to_numpy()
    not_evolved_counts = global_df["not_evolved_generated_repositories"].to_numpy()

    width = 0.38

    fig, ax = plt.subplots(figsize=(max(9, len(languages) * 1.5), 6))

    bars_evolved = ax.bar(
        x - width / 2,
        evolved_counts,
        width,
        label="Evolved"
    )

    bars_not_evolved = ax.bar(
        x + width / 2,
        not_evolved_counts,
        width,
        label="Not evolved"
    )

    ax.set_title("Evolution of generated repositories by language")
    ax.set_ylabel("Number of generated repositories")
    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    add_value_labels(ax, bars_evolved)
    add_value_labels(ax, bars_not_evolved)

    fig.tight_layout()

    chart_paths.extend(
        save_chart(
            fig,
            charts_dir,
            "01_evolution_counts_by_language"
        )
    )

    # ------------------------------------------------------------
    # Chart 2: Evolution percentages by language
    # ------------------------------------------------------------

    evolved_percentages = global_df["evolved_generated_repositories_percentage"].to_numpy()
    not_evolved_percentages = global_df["not_evolved_generated_repositories_percentage"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(9, len(languages) * 1.5), 6))

    bars_evolved = ax.bar(
        languages,
        evolved_percentages,
        label="Evolved"
    )

    bars_not_evolved = ax.bar(
        languages,
        not_evolved_percentages,
        bottom=evolved_percentages,
        label="Not evolved"
    )

    ax.set_title("Evolution percentage of generated repositories by language")
    ax.set_ylabel("Percentage of analyzed generated repositories")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(languages, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, language in enumerate(languages):
        evolved_value = evolved_percentages[i]
        not_evolved_value = not_evolved_percentages[i]

        if evolved_value > 4:
            ax.text(
                i,
                evolved_value / 2,
                f"{evolved_value:.1f}%",
                ha="center",
                va="center",
                fontsize=9
            )

        if not_evolved_value > 4:
            ax.text(
                i,
                evolved_value + not_evolved_value / 2,
                f"{not_evolved_value:.1f}%",
                ha="center",
                va="center",
                fontsize=9
            )

    fig.tight_layout()

    chart_paths.extend(
        save_chart(
            fig,
            charts_dir,
            "02_evolution_percentages_by_language"
        )
    )

    # ------------------------------------------------------------
    # Chart 3: Evolution level distribution by language
    # ------------------------------------------------------------

    if not combined_evolution_distribution.empty:
        distribution_df = combined_evolution_distribution.copy()

        required_distribution_columns = [
            "language",
            "evolution_level",
            "percentage"
        ]

        missing_distribution_columns = [
            col for col in required_distribution_columns
            if col not in distribution_df.columns
        ]

        if not missing_distribution_columns:
            distribution_df["percentage"] = pd.to_numeric(
                distribution_df["percentage"],
                errors="coerce"
            ).fillna(0)

            pivot_df = (
                distribution_df
                .pivot_table(
                    index="language",
                    columns="evolution_level",
                    values="percentage",
                    aggfunc="sum",
                    fill_value=0
                )
                .reindex(languages)
            )

            for level in EVOLUTION_LEVEL_ORDER:
                if level not in pivot_df.columns:
                    pivot_df[level] = 0

            pivot_df = pivot_df[EVOLUTION_LEVEL_ORDER]

            fig, ax = plt.subplots(figsize=(max(9, len(languages) * 1.5), 6))

            bottom = np.zeros(len(pivot_df))

            for level in EVOLUTION_LEVEL_ORDER:
                values = pivot_df[level].to_numpy()

                ax.bar(
                    pivot_df.index,
                    values,
                    bottom=bottom,
                    label=level
                )

                for i, value in enumerate(values):
                    if value > 5:
                        ax.text(
                            i,
                            bottom[i] + value / 2,
                            f"{value:.1f}%",
                            ha="center",
                            va="center",
                            fontsize=8
                        )

                bottom += values

            ax.set_title("Evolution level distribution by language")
            ax.set_ylabel("Percentage of analyzed generated repositories")
            ax.set_ylim(0, 100)
            ax.set_xticklabels(pivot_df.index, rotation=30, ha="right")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()

            chart_paths.extend(
                save_chart(
                    fig,
                    charts_dir,
                    "03_evolution_levels_by_language"
                )
            )

    # ------------------------------------------------------------
    # Chart 4: Template implementation coverage by language
    # ------------------------------------------------------------

    global_df["templates_with_generated_percentage"] = global_df.apply(
        lambda row: percentage(
            row["templates_with_generated_repositories"],
            row["total_templates"]
        ),
        axis=1
    )

    global_df["templates_without_generated_percentage"] = global_df.apply(
        lambda row: percentage(
            row["templates_without_generated_repositories"],
            row["total_templates"]
        ),
        axis=1
    )

    with_generated = global_df["templates_with_generated_percentage"].to_numpy()
    without_generated = global_df["templates_without_generated_percentage"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(9, len(languages) * 1.5), 6))

    ax.bar(
        languages,
        with_generated,
        label="Templates with generated repositories"
    )

    ax.bar(
        languages,
        without_generated,
        bottom=with_generated,
        label="Templates without generated repositories"
    )

    ax.set_title("Template implementation coverage by language")
    ax.set_ylabel("Percentage of templates")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(languages, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, language in enumerate(languages):
        implemented_value = with_generated[i]
        not_implemented_value = without_generated[i]

        if implemented_value > 4:
            ax.text(
                i,
                implemented_value / 2,
                f"{implemented_value:.1f}%",
                ha="center",
                va="center",
                fontsize=9
            )

        if not_implemented_value > 4:
            ax.text(
                i,
                implemented_value + not_implemented_value / 2,
                f"{not_implemented_value:.1f}%",
                ha="center",
                va="center",
                fontsize=9
            )

    fig.tight_layout()

    chart_paths.extend(
        save_chart(
            fig,
            charts_dir,
            "04_template_implementation_coverage_by_language"
        )
    )

    return chart_paths

if __name__ == "__main__":
    main()