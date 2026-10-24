#!/usr/bin/env python3

import csv
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

INPUT_FILE = "survey_responses.csv"

OUTPUT_PDF = "guidelines_likert.pdf"
OUTPUT_PNG = "guidelines_likert.png"

OUTPUT_STATS_TXT = "survey_paper_statistics.txt"
OUTPUT_STATS_CSV = "survey_paper_statistics.csv"
OUTPUT_PROFILE_CSV = "survey_participant_profile.csv"

INVITED_PARTICIPANTS = 4538
DELIVERED_INVITATIONS = 4215  # Manuscript denominator: delivered invitations


# =============================================================================
# Colors -- same style as the previous study
# =============================================================================

BLUE = "#6FA8DC"
DARK_BLUE = "#4F81BD"

GRAY = "#B8B8B8"

ORANGE = "#F6A04D"

DARK_RED = "#C0504D"


LIKERT_ORDER = [
    "Strongly disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly agree",
]


LIKERT_COLORS = {
    "Strongly disagree": DARK_RED,
    "Disagree": ORANGE,
    "Neutral": GRAY,
    "Agree": BLUE,
    "Strongly agree": DARK_BLUE,
}


# =============================================================================
# Guideline-related survey statements
# =============================================================================

GUIDELINES = {
    "TC1": (
        "Inspect code-quality and security vulnerabilities before adopting "
        "a template"
    ),

    "TC2": (
        "Consider the programming language and its ecosystem when evaluating "
        "the template"
    ),

    "TC3a": (
        "Consider the effort required to maintain the reused code and "
        "incorporate future"
    ),

    "TC3b": (
        "Evaluate whether the template's predefined structure, dependencies, "
        "and configurations"
    ),

    "TC4a": (
        "Consider whether the template is maintained by an individual or "
        "an organization"
    ),

    "TC4b": (
        "Consider the ownership of the source template"
    ),
}


ELIGIBILITY_FRAGMENT = (
    "Have you ever used a GitHub template repository to create a new "
    "repository or contributed to a repository created from a GitHub template?"
)


# =============================================================================
# Helpers
# =============================================================================

def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip()).casefold()


def find_column(headers, fragment):
    matches = [
        header
        for header in headers
        if normalize(fragment) in normalize(header)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one column matching:\n{fragment}\n"
            f"Found: {matches}"
        )

    return matches[0]


def percentage(count, denominator):
    if denominator == 0:
        return 0.0

    return 100.0 * count / denominator


def contrast_text_color(color):
    """White text on darker segments, black on lighter ones."""
    return (
        "black"
        if color in {GRAY, ORANGE}
        else "white"
    )


# =============================================================================
# Read survey
# =============================================================================

with open(
    INPUT_FILE,
    "r",
    encoding="cp1252",
    newline="",
) as f:

    reader = csv.DictReader(f)

    headers = reader.fieldnames
    rows = list(reader)


eligibility_column = find_column(
    headers,
    ELIGIBILITY_FRAGMENT,
)


analysis_rows = [
    row
    for row in rows
    if row[eligibility_column].strip() == "Yes"
]


n = len(analysis_rows)

print(f"Eligible respondents: {n}")


# =============================================================================
# Resolve columns and compute Likert distributions
# =============================================================================

guideline_columns = {
    guideline: find_column(
        headers,
        fragment,
    )
    for guideline, fragment
    in GUIDELINES.items()
}


practices = list(
    GUIDELINES.keys()
)


values_by_category = {}

for category in LIKERT_ORDER:

    values = []

    for guideline in practices:

        column = guideline_columns[
            guideline
        ]

        counts = Counter(
            row[column].strip()
            for row in analysis_rows
            if row[column].strip()
        )

        denominator = sum(
            counts.values()
        )

        values.append(
            percentage(
                counts[category],
                denominator,
            )
        )

    values_by_category[
        category
    ] = np.array(
        values,
        dtype=float,
    )

# =============================================================================
# Statistics reported in the paper
# =============================================================================

def get_distribution(fragment):
    column = find_column(
        headers,
        fragment,
    )

    values = [
        row[column].strip()
        for row in analysis_rows
        if row[column].strip()
    ]

    return Counter(values), len(values)


paper_stats = []


def add_stat(
    section,
    metric,
    count,
    denominator,
):
    paper_stats.append({
        "section": section,
        "metric": metric,
        "count": count,
        "denominator": denominator,
        "percent": percentage(
            count,
            denominator,
        ),
    })


# =============================================================================
# Survey sample
# =============================================================================

add_stat(
    "Survey",
    "Responses",
    len(rows),
    DELIVERED_INVITATIONS,
)

add_stat(
    "Survey",
    "Eligible respondents",
    n,
    len(rows),
)


# =============================================================================
# Participant profile -- eligible participants only
# =============================================================================

# Report distributions with N=eligible respondents. Explicitly include missing
# values so readers can distinguish non-response from a zero-count category.
# Individual records, emails and free-text job titles are never exported.

def add_profile_distribution(label, fragment, category_order=None, group_other=False):
    column = find_column(headers, fragment)
    values = [row[column].strip() for row in analysis_rows]
    if group_other:
        known = set(category_order or [])
        values = [
            value if value in known else ("Other positions" if value else "No response")
            for value in values
        ]
    else:
        values = [value or "No response" for value in values]

    counts = Counter(values)
    order = list(category_order or [])
    for category in sorted(counts):
        if category not in order:
            order.append(category)
    for category in order:
        if counts[category]:
            add_stat("Participant profile", f"{label}: {category}", counts[category], n)
    return counts


positions = add_profile_distribution(
    "Current position",
    "What is your current position?",
    ["Software Engineer", "DevOps Engineer", "Student", "Researcher"],
    group_other=True,
)

experience = add_profile_distribution(
    "Software-development experience",
    "How many years of experience do you have in software development?",
    ["Less than 1 year", "1–3 years", "3–5 years", "5–10 years", "More than 10 years"],
)

position_tenure = add_profile_distribution(
    "Years in current position",
    "How many years have you been in this position?",
    ["Less than 1 year", "1–3 years", "3–5 years", "5–10 years", "More than 10 years"],
)

github_usage = add_profile_distribution(
    "GitHub usage",
    "How frequently do you use GitHub?",
    ["Daily", "Frequently", "Occasionally", "Rarely", "Never"],
)

template_usage = add_profile_distribution(
    "Times using a GitHub template",
    "Approximately how many times have you used a GitHub template",
    ["1", "2–5", "6-10", "11-20", "More than 20"],
)

organization_size = add_profile_distribution(
    "Organization size",
    "What is the size of the organization where you currently work?",
    ["1–10 employees", "11–50 employees", "51–200 employees", "201–1000 employees", "More than 1000 employees", "Not applicable"],
)

gender = add_profile_distribution(
    "Gender (optional)",
    "What is your gender?",
    ["Man", "Woman", "Non-binary", "Prefer not to say"],
)

add_stat(
    "Participant profile",
    "At least five years of software-development experience",
    sum(value for key, value in experience.items()
        if key in {"5–10 years", "5-10 years", "More than 10 years"}),
    n,
)

add_stat(
    "Participant profile",
    "Uses GitHub daily or frequently",
    github_usage["Daily"] + github_usage["Frequently"],
    n,
)

# The development-type question permits multiple selections. Percentages
# therefore use 32 eligible respondents as denominator and need not sum to 100%.
contexts_column = find_column(headers, "What type of development do you primarily work on")
known_contexts = (
    "Open-source projects",
    "Industry projects",
    "Academic research",
    "Personal projects",
    "Government",
)
context_counts = Counter()
context_other = 0
for row in analysis_rows:
    selections = {part.strip() for part in row[contexts_column].split(", ") if part.strip()}
    for context in known_contexts:
        if context in selections:
            context_counts[context] += 1
    if selections.difference(known_contexts):
        context_other += 1

for context in known_contexts:
    add_stat("Development contexts (multiple selections)", context,
             context_counts[context], n)
if context_other:
    add_stat("Development contexts (multiple selections)",
             "Other response", context_other, n)


# =============================================================================
# RQ1 -- Downstream development
# =============================================================================

# Initial modification of the template
modification_counts, modification_n = get_distribution(
    "How much do you typically modify the initial structure"
)

add_stat(
    "RQ1",
    "Slightly or moderately modify initial template structure",
    modification_counts["Slightly"]
    + modification_counts["Moderately"],
    modification_n,
)

add_stat(
    "RQ1",
    "No initial modification",
    modification_counts["Nothing"],
    modification_n,
)


# Continued repository development
commit_counts, commits_n = get_distribution(
    "After this initial adaptation, how extensively do you typically "
    "continue modifying"
)

for category in [
    "1 commit",
    "2-5 commits",
    "6-20 commits",
    "More than 20 commits",
]:
    add_stat(
        "RQ1",
        f"Typical downstream development: {category}",
        commit_counts[category],
        commits_n,
    )


# Local changes pushed to GitHub
push_counts, push_n = get_distribution(
    "After making changes locally in your repository, how often do you "
    "upload/push"
)

push_most_or_all = (
    push_counts[
        "I push most of my local changes to GitHub"
    ]
    + push_counts[
        "I push nearly all of my local changes to GitHub"
    ]
)

add_stat(
    "RQ1",
    "Push most or nearly all local changes to GitHub",
    push_most_or_all,
    push_n,
)


# =============================================================================
# RQ2 -- Quality and security concerns
# =============================================================================

# How often practitioners check concerns
check_counts, check_n = get_distribution(
    "How often do you check for quality or security issues"
)

add_stat(
    "RQ2",
    "Check quality/security issues often or always",
    check_counts["Often"]
    + check_counts["Always"],
    check_n,
)

add_stat(
    "RQ2",
    "Check quality/security issues rarely or never",
    check_counts["Rarely"]
    + check_counts["Never"],
    check_n,
)


# How often practitioners address identified concerns
fix_counts, fix_n = get_distribution(
    "When you identify such issues, how often do you address or fix them"
)

add_stat(
    "RQ2",
    "Address identified concerns often or always",
    fix_counts["Often"]
    + fix_counts["Always"],
    fix_n,
)


# Awareness of concern propagation
awareness_counts, awareness_n = get_distribution(
    "Before this survey, were you aware that quality or security issues present"
)

add_stat(
    "RQ2",
    "Aware template concerns may also appear downstream",
    awareness_counts["Yes"],
    awareness_n,
)


# Awareness that later source-template changes do not propagate automatically
upstream_counts, upstream_n = get_distribution(
    "Are you aware that changes made later to the original template repository"
)

add_stat(
    "RQ2",
    "Aware later template changes are not automatically propagated",
    upstream_counts["Yes"],
    upstream_n,
)


# Perceived impact of template quality
quality_impact_counts, quality_impact_n = get_distribution(
    "To what extent do you think the quality of a GitHub template can affect"
)

add_stat(
    "RQ2",
    "Template quality affects generated repository very much or extremely",
    quality_impact_counts["Very much"]
    + quality_impact_counts["Extremely"],
    quality_impact_n,
)

add_stat(
    "RQ2",
    "Template quality affects generated repository at least moderately",
    quality_impact_counts["Moderately"]
    + quality_impact_counts["Very much"]
    + quality_impact_counts["Extremely"],
    quality_impact_n,
)


# =============================================================================
# RQ3 -- Language and ownership perceptions
# =============================================================================

# Direct ownership preference
ownership_counts, ownership_n = get_distribution(
    "In your experience, does the ownership of a GitHub template repository "
    "influence your decision"
)

add_stat(
    "RQ3",
    "More likely to consider organization-owned templates",
    ownership_counts[
        "More likely to consider organization-owned templates"
    ],
    ownership_n,
)


# =============================================================================
# Guideline-related Likert responses
# =============================================================================

for guideline in GUIDELINES:

    column = guideline_columns[
        guideline
    ]

    responses = [
        row[column].strip()
        for row in analysis_rows
        if row[column].strip()
    ]

    counts = Counter(
        responses
    )

    denominator = len(
        responses
    )

    # Full five-point Likert distribution
    for category in LIKERT_ORDER:

        add_stat(
            "Guidelines",
            f"{guideline}: {category}",
            counts[category],
            denominator,
        )

    # Grouped agreement used in the manuscript
    agree = (
        counts["Agree"]
        + counts["Strongly agree"]
    )

    add_stat(
        "Guidelines",
        f"{guideline}: Agree + Strongly agree",
        agree,
        denominator,
    )


# =============================================================================
# Write machine-readable CSV
# =============================================================================

with open(
    OUTPUT_STATS_CSV,
    "w",
    encoding="utf-8",
    newline="",
) as f:

    fieldnames = [
        "section",
        "metric",
        "count",
        "denominator",
        "percent",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    for stat in paper_stats:

        writer.writerow({
            **stat,
            "percent": round(
                stat["percent"],
                1,
            ),
        })


# =============================================================================
# Separate participant-profile table for the manuscript
# =============================================================================

with open(OUTPUT_PROFILE_CSV, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "section", "metric", "count", "denominator", "percent"
    ])
    writer.writeheader()
    for stat in paper_stats:
        if stat["section"] in {
            "Participant profile", "Development contexts (multiple selections)"
        }:
            writer.writerow({**stat, "percent": round(stat["percent"], 1)})

# =============================================================================
# Write human-readable report
# =============================================================================

with open(
    OUTPUT_STATS_TXT,
    "w",
    encoding="utf-8",
) as f:

    current_section = None

    for stat in paper_stats:

        if stat["section"] != current_section:

            current_section = stat[
                "section"
            ]

            f.write(
                f"\n{current_section}\n"
            )

            f.write(
                "=" * len(current_section)
                + "\n"
            )

        # Response rate is normally reported with two decimals.
        if (
            stat["section"] == "Survey"
            and stat["metric"] == "Responses"
        ):
            percent_text = (
                f"{stat['percent']:.2f}%"
            )

        else:
            percent_text = (
                f"{stat['percent']:.1f}%"
            )

        f.write(
            f"{stat['metric']}: "
            f"{stat['count']}/"
            f"{stat['denominator']} "
            f"({percent_text})\n"
        )


# =============================================================================
# Also print manuscript-relevant statistics to console
# =============================================================================

print()
print("=" * 72)
print("SURVEY STATISTICS USED IN THE PAPER")
print("=" * 72)

current_section = None

for stat in paper_stats:

    if stat["section"] != current_section:

        current_section = stat[
            "section"
        ]

        print()
        print(
            current_section
        )
        print(
            "-" * len(current_section)
        )

    if (
        stat["section"] == "Survey"
        and stat["metric"] == "Responses"
    ):
        percent_text = (
            f"{stat['percent']:.2f}%"
        )

    else:
        percent_text = (
            f"{stat['percent']:.1f}%"
        )

    print(
        f"{stat['metric']}: "
        f"{stat['count']}/"
        f"{stat['denominator']} "
        f"({percent_text})"
    )

print()
print(
    f"Saved: {OUTPUT_STATS_TXT}"
)

print(
    f"Saved: {OUTPUT_STATS_CSV}"
)
print(f"Saved: {OUTPUT_PROFILE_CSV}")

# =============================================================================
# Generate figure
# =============================================================================

labels = practices

y = np.arange(
    len(practices)
)

# Generate close to ACM single-column width.
fig, ax = plt.subplots(
    figsize=(3.35, 2.65)
)

left = np.zeros(
    len(practices),
    dtype=float,
)

for category in LIKERT_ORDER:

    values = values_by_category[
        category
    ]

    ax.barh(
        y,
        values,
        height=0.48,
        left=left,
        label=category,
        color=LIKERT_COLORS[
            category
        ],
        edgecolor="white",
        linewidth=0.6,
    )

    text_color = (
        contrast_text_color(
            LIKERT_COLORS[
                category
            ]
        )
    )

    for i, value in enumerate(
        values
    ):

        # Show labels only when there is enough room.
        if value >= 9:

            ax.text(
                left[i] + value / 2,
                y[i],
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=7.0,   # slightly smaller
                color=text_color,
                fontweight="bold",
            )

    left += values


# =============================================================================
# Axis formatting
# =============================================================================

ax.set_xlim(
    0,
    100,
)

ax.set_xticks(
    [
        0,
        25,
        50,
        75,
        100,
    ]
)

ax.set_xticklabels(
    [
        "0%",
        "25%",
        "50%",
        "75%",
        "100%",
    ],
    fontsize=7.5,
)

ax.set_xlabel(
    "Participants (%)",
    fontsize=8,
    labelpad=3,
)

ax.set_yticks(
    y
)

ax.set_yticklabels(
    labels,
    fontsize=8.5,
    fontweight="bold",
)

ax.tick_params(
    axis="y",
    length=0,
)

ax.tick_params(
    axis="x",
    length=2.5,
    pad=2,
)

ax.invert_yaxis()


# =============================================================================
# Clean visual style
# =============================================================================

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

ax.spines["bottom"].set_linewidth(0.6)
ax.spines["bottom"].set_color("0.4")

ax.grid(
    axis="x",
    linestyle="-",
    linewidth=0.4,
    alpha=0.20,
)

ax.set_axisbelow(
    True
)


# =============================================================================
# Legend
# =============================================================================

legend = ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.24),   # a bit lower -> more space from chart
    ncol=3,
    frameon=False,
    fontsize=6.4,
    handlelength=1.1,
    handleheight=0.8,
    columnspacing=1.0,
    handletextpad=0.4,
    borderaxespad=0.0,
)


# =============================================================================
# Layout
# =============================================================================

fig.subplots_adjust(
    left=0.10,
    right=0.99,
    bottom=0.34,   # more room for legend
    top=0.98,
)


# =============================================================================
# Save
# =============================================================================

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PNG,
    dpi=600,
    bbox_inches="tight",
)

plt.close(
    fig
)

print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")