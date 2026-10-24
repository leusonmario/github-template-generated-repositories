# RQ1 — How do GitHub template-generated repositories evolve?

## Analysis and population

RQ1 characterizes commit activity in **123,360** generated repositories from **1,533** source templates across C#, Java, Python, TypeScript, and JavaScript. GitHub template-generated repositories begin with one initialization commit. Accordingly, RQ1 classifies total commit counts as:

| Total commits | Evolution class |
| --- | --- |
| 1 | No observable post-creation commit |
| 2–6 | Low activity |
| 7–20 | Moderate activity |
| >20 | High activity |

The measure describes observable GitHub commit activity. It does not quantify the amount of code changed or exclude local unpushed work.

## Inputs

The analysis uses the per-language CSV reports under `rq1/reports/`, conventionally named:

```text
rq1/reports/generated_GHRep_from_templates_Csharp_with_commits.csv
rq1/reports/generated_GHRep_from_templates_Java_with_commits.csv
rq1/reports/generated_GHRep_from_templates_Python_with_commits.csv
rq1/reports/generated_GHRep_from_templates_TypeScript_with_commits.csv
rq1/reports/generated_GHRep_from_templates_JavaScript_with_commits.csv
```

The age-sensitivity script expects `username`, `project_name`, `original_link`, `data_creation`, and `commit_count`. `target_template_repository` and `last_commit_datetime` are additional useful columns.

Confirm the input files exist before running:

```bash
find rq1/reports -maxdepth 1 -name '*with_commits.csv' -print
```

## Run the age and template-family robustness checks

Run from the repository root:

```bash
python rq1/rq1_repository_age_analysis.py rq1/reports \
  --pattern 'generated_GHRep_from_templates_*_with_commits.csv' \
  --as-of 2026-07-22 \
  --output rq1/reproduced_age_results
```

The `--as-of` parameter must refer to the observation date of the commit counts, **not the current date**. The manuscript reports July 22, 2026 as its RQ1 observation date; update the command only if reproducing a different frozen snapshot.

Expected outputs under `rq1/reproduced_age_results/`:

- `rq1_age_by_language.csv` — evolution distributions by repository age and language.
- `rq1_age_robustness.csv` — full sample versus repositories at least 6, 12, and 24 months old.
- `rq1_template_weighting.csv` — repository-weighted versus equal-source-template-weighted proportions.
- `rq1_data_quality.csv` — filtering and missing-data information.
- `rq1_duplicate_conflicts.csv` — inconsistent duplicate records excluded by the script.
- `rq1_summary.txt` — human-readable totals.

The script is written using only the Python standard library and indicates compatibility with Python 3.10+.

## Interpretation and validation

For the published RQ1 sample, there should be **123,360** analyzed repositories, of which **85,955** have more than one commit and **37,405** have only their initialization commit. Some intermediate CSVs may contain unavailable repositories, unparsed commit counts, or duplicate entries: reconcile any discrepancy against the filtering and duplicate reports rather than silently altering the denominator.

Template-family weighting gives each source template equal influence, in contrast to ordinary repository-level percentages where highly adopted templates contribute more observations.

## Optional domain exploration

Domain comparisons use *source-template* classifications from the previous template-repository study (for example, DeepSeek classifications with `llm_output.category`). Match the domain classification on the **source template's** `owner/repository` identity, not on the generated repository. This is exploratory and is not itself a change to the RQ1 population.

If no domain-classification CSVs are distributed, provide them as a documented external input or omit this optional analysis; do not silently label unknown templates as a known category.
