# RQ3 — Which characteristics are associated with concern persistence?

## Overview

RQ3 models whether source-template quality issues and security hotspots remain detectable in template-generated repositories. The primary models are **binomial generalized estimating equations (GEEs)** clustered by source template, with robust sandwich standard errors. Analyses for quality issues and security hotspots are separate.

Two verified scripts in `rq3/` are:

```text
rq3/build_rq3_analysis_dataset_with_hotspots.py
rq3/run_rq3_statistical_analysis_with_hotspots.py
```

Use Python with `numpy`, `pandas`, `scipy`, and `statsmodels`.

## Inputs

1. **RQ1 detailed pair-level CSV**, with `language`, `template_repo`, `generated_repo`, `commit_count` (and optionally `evolution_level`). This is passed using `--rq1-detailed`.
2. **RQ2 pair summaries** under a language-specific `issue_persistence_results/pair_summary.csv` path and the corresponding `pair_summary.json` files.
3. **Template metadata** under `--templates-dir`, conventionally `all_templates_<LANGUAGE>.csv` for the five languages; this supplies the source-template ownership labels.
4. **Generated-repository metadata** (the RQ1 report CSVs), used for creation dates and age where applicable.
5. **Domain classifications** from the earlier source-template study, using `llm_output.category` or an equivalent categorical column. These are joined to **source-template identities** and are optional for the base dataset but required for reproducing the reported domain secondary analyses.
6. **Exact RQ2 selected-sample files** per language. These are needed for the original inclusion probabilities and weighted inferences, especially Python, TypeScript, and JavaScript.

Check which files are available:

```bash
find data_collection -name 'all_templates_*.csv' | sort
find rq1 -name '*with_commits.csv' | sort
find rq2 -name 'pair_summary.csv' -o -name 'pair_summary.json' | sort
find rq2 -name 'selected_sample.csv' | sort
```

## Step 1 — Build the RQ3 dataset

From the repository root, substitute the real path to the existing RQ1 detailed dataset:

```bash
python rq3/build_rq3_analysis_dataset_with_hotspots.py \
  --rq1-detailed /path/to/rq1_detailed.csv \
  --rq2-root rq2 \
  --templates-dir data_collection \
  --generated-reports-dir rq1/reports \
  --domains-dir /path/to/earlier-studies/domain-classifications \
  --output-dir rq3/reproduced_dataset
```

For **exact sample-weight reproduction**, add one `--sampling-file LANGUAGE PATH` pair for each available original sampling CSV. For example:

```bash
  --sampling-file Python /path/to/Python/selected_sample.csv \
  --sampling-file TypeScript /path/to/TypeScript/selected_sample.csv \
  --sampling-file JavaScript /path/to/JavaScript/selected_sample.csv
```

Append these options **inside the builder command**, not as a separate shell command. Where no exact sampling file is supplied, the script falls back to weights in `pair_summary.csv` and ultimately to 1.0 for missing/invalid weights. This can change findings and must be reported.

If domain auto-discovery matches multiple files for a language, use repeated `--domain-file LANGUAGE PATH` options instead of assuming an arbitrary match:

```bash
  --domain-file TypeScript /path/to/deepseek_judger_TypeScript.csv
```

The builder outputs include:

- `rq3_analysis_dataset.csv` — main pair-level dataset.
- `rq3_analysis_dataset_long.csv` — separate issue/hotspot observations in long form.
- `rq3_issue_analysis_dataset_complete_cases.csv` and `rq3_hotspot_analysis_dataset_complete_cases.csv`.
- `rq3_variable_summary.csv`, `rq3_data_quality.csv`.
- `rq3_domain_distribution.csv` and `rq3_unmatched_domain_templates.csv`.

Inspect the data-quality and unmatched-domain files before modeling.

## Step 2 — Run the statistical analysis

```bash
python rq3/run_rq3_statistical_analysis_with_hotspots.py \
  --dataset rq3/reproduced_dataset/rq3_analysis_dataset.csv \
  --rq2-root rq2 \
  --age-reference-date YYYY-MM-DD \
  --age-mode exclude \
  --domain-min-templates 10 \
  --min-adjusted-hotspot-category-templates 10 \
  --output-dir rq3/reproduced_statistics
```

Use the *fixed reference date associated with the study's model dataset* in place of `YYYY-MM-DD`, never today's date. `--age-mode exclude` matches the reported RQ3 primary specification without repository age as a predictor; if you wish to test age-adjusted variants, `--age-mode auto` or `--age-mode include` are available but represent additional sensitivity analyses.

The script uses `pair_summary.json` to recover SonarQube finding breakdowns. Without complete JSON files, the ordinary RQ3 runs should not be described as exact reproduction. Avoid `--allow-partial-breakdowns` for the reported final results; it is intended for diagnostics.

The model step produces:

- `rq3_finding_dimension_dataset.csv`, `rq3_finding_dimension_support.csv`.
- `rq3_finding_model_results.csv`, `rq3_finding_model_summary.csv`.
- `rq3_finding_omnibus_wald_tests.csv`.
- `rq3_finding_evolution_stratified.csv` (including Benjamini–Hochberg-adjusted values).
- `rq3_finding_main_results.csv`.
- Per-model text output and analysis metadata.

## Interpretation and integrity checks

- Clustering is at the **source-template** level: generated repositories from the same template are not statistically independent.
- Sampling weights must be consistent with RQ2. Quality issues and security hotspots use separate denominators.
- Source-template ownership and programming-language effects are *associations*, not demonstrated causes.
- The source-template **domain** analysis is secondary and uses categories from the earlier template study; domains with fewer than ten source templates are pooled into `Other`.
- Sparse adjusted hotspot security categories are excluded when represented by fewer than ten source templates.
- The manuscript reports that commit count has no robust, general association with reduced persistence. Verify the reproduced tables and sample totals against the manuscript rather than assuming every rerun will match with different inputs or analyzer versions.
