# GitHub Template-Generated Repositories: An Exploratory Study

**Replication package** for the empirical study *GitHub Template-Generated Repositories: An Exploratory Study*.

GitHub template repositories provide reusable project scaffolding. Unlike forks, repositories generated from templates begin as independent projects and do not automatically receive subsequent changes from the source template. This artifact supports a consumer-side analysis of their subsequent development and of the quality and security concerns detectable downstream.

## Study Overview

| Component | Scope |
| --- | --- |
| Languages | C#, Java, Python, TypeScript, JavaScript |
| Source templates with identified generated repositories | **1,533** |
| Generated repositories studied for evolution (RQ1) | **123,360** |
| Generated repositories analyzed for concern correspondence (RQ2) | **4,956**, from **713** templates |
| Consumer survey | **32** eligible respondents |

The analyses address:

- **RQ1 — Repository evolution:** How do GitHub template-generated repositories evolve after creation?
- **RQ2 — Concern persistence:** To what extent do SonarQube quality issues and security hotspots detected in source templates remain detectable in their generated repositories?
- **RQ3 — Associated characteristics:** Which template, repository, and finding characteristics are associated with this measured persistence?

Survey responses complement the repository analyses; they are not treated as representative estimates of the GitHub developer population.

## Repository layout

The codebase is organized by the stages of the study:

```text
.
├── data_collection/    # Discovery and validation of template-generated repositories
├── docs/               # Step-by-step reproduction instructions (this package)
├── rq1/                # Commit activity and repository-evolution analyses
├── rq2/                # SonarQube extraction, matching, and persistence results
├── rq3/                # RQ3 dataset construction and statistical analysis
├── survey/             # Consumer-survey materials and/or aggregate results
└── README.md
```

Input datasets, generated results, and large external dependencies may not all be committed to Git. The commands below assume the corresponding input files are available locally. **Do not treat an example path as evidence that the dataset is included in the checkout.**

## Getting started

The commands below are intended for a **Linux/macOS terminal**. Run them from the **repository root** (the directory containing `data_collection/`, `rq1/`, `rq2/`, and `rq3/`). All example paths are relative to this directory. If using an IDE such as PyCharm, supply the script path, arguments, and working directory separately; do not paste shell continuation characters into the IDE's arguments field.

### 1. Set up Python

The RQ1 age-robustness script uses Python 3.10+ and the standard library. The RQ1 ownership analysis also uses `pandas`, `numpy`, `scipy`, `statsmodels`, and **`matplotlib`**. RQ3 uses `pandas`, `numpy`, `scipy`, and `statsmodels`. The RQ2 *aggregation* step reads already computed SonarQube matching results and does not rerun SonarQube. Fresh scans or re-matching require additional inputs and tools.

```bash
# Start in the repository root (after cloning the repository).
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy statsmodels matplotlib
```

On Windows, activate the environment with `.venv\Scripts\activate` (Command Prompt) or `.venv\Scripts\Activate.ps1` (PowerShell); paths in examples below use `/` as the separator for Linux/macOS.

Before proceeding, verify your working directory and script locations:

```bash
pwd
ls rq1/analyze_evolution_by_template_ownership_combined_chart.py
ls rq1/rq1_repository_age_analysis.py
ls data_collection
ls rq1/reports
```

If a path is missing, the corresponding script or input has not been included in the checkout. Check the supported options using:

```bash
python rq1/analyze_evolution_by_template_ownership_combined_chart.py --help
python rq1/rq1_repository_age_analysis.py --help
python rq3/build_rq3_analysis_dataset_with_hotspots.py --help
python rq3/run_rq3_statistical_analysis_with_hotspots.py --help
```

### 2. Reproduction guides

| Stage | Documentation | Main inputs and outputs |
| --- | --- | --- |
| Data collection | [Study inputs and provenance](docs/DATA_AND_PROVENANCE.md) | Source-template metadata, GitHub API observations, generated-repository identifiers |
| RQ1 | **[Reproduce RQ1](docs/RQ1.md)** | Generated-repository commit reports → activity, ownership, age robustness, template weighting |
| RQ2 | **[Reproduce RQ2](docs/RQ2.md)** | Validated `pair_summary` and per-template results → RQ2 tables, figures, and dimension summaries |
| RQ3 | **[Reproduce RQ3](docs/RQ3.md)** | RQ1 details + RQ2 results + template metadata → regression-ready dataset and GEE models |
| Survey | [Survey and ethics](docs/SURVEY.md) | Questionnaire, de-identified responses, aggregate counts |

### 3. RQ1: quick start

**A. Repository evolution by source-template ownership.** This script needs the `all_templates_*.csv` files under `data_collection/` and the `generated_GHRep_from_templates_*_with_commits.csv` files under `rq1/reports/`.

From the **repository root**, copy and execute this **single line**:

```bash
python rq1/analyze_evolution_by_template_ownership_combined_chart.py --templates_dir data_collection --generated_dir rq1/reports --output_dir rq1/reproduced_ownership_results
```

Its outputs include `ownership_evolution_by_language.csv`, `ownership_evolution_overall.csv`, `ownership_evolution_detailed.csv`, and figures in `rq1/reproduced_ownership_results/charts/`. The supported options are `--templates_dir`, `--generated_dir`, and `--output_dir` (with **underscores**). This script does **not** accept `--as-of`, `--pattern`, or `--output`.

**B. Repository age robustness and equal-template weighting.** This is a **different script**. It uses the observed commit counts and a fixed observation date; the manuscript reports **2026-07-22**.

Run from the **repository root**:

```bash
python rq1/rq1_repository_age_analysis.py rq1/reports --pattern 'generated_GHRep_from_templates_*_with_commits.csv' --as-of 2026-07-22 --output rq1/reproduced_age_results
```

This produces `rq1_age_robustness.csv`, `rq1_template_weighting.csv`, `rq1_age_by_language.csv`, and data-quality reports under `rq1/reproduced_age_results/`. The `--as-of` value must match the observation date for the input data, not the current date.

The ownership script requires template metadata as well as generated-repository reports; the age script requires only generated-repository reports. **All commands in this README use paths relative to the repository root.** For the complete methodology, see [RQ1 documentation](docs/RQ1.md).

### 4. RQ2: regenerate the reported results from validated matching outputs

**No new matching or SonarQube scanning is required for this step.** This is the analysis used to produce the final RQ2 summaries. It consumes the already computed files in:

```text
rq2/quality_issues_and_security_hotspots/vulnerability_issues/
  CSharp/issue_persistence_results_validated/
  Java/issue_persistence_results_validated/
  Python/issue_persistence_results_validated/
  TypeScript/issue_persistence_results_validated/
  JavaScript/issue_persistence_results_validated/
```

Each language directory must contain `pair_summary.csv`, `template_summary.csv`, `template_issue_persistence.csv`, `template_hotspot_persistence.csv`, and `pair_summary.json`. The commands below are written **for the repository root**. Do not switch into `rq2/` or type any of the displayed CSV filenames as commands.

**A. Reproduce the RQ2 tables and figures (Issues and Security Hotspots separately).**

```bash
BASE="rq2/quality_issues_and_security_hotspots/vulnerability_issues"
ARGS=()
MISSING=0

for entry in "C#|CSharp" "Java|Java" "Python|Python" "TypeScript|TypeScript" "JavaScript|JavaScript"; do
  LANG="${entry%%|*}"
  FOLDER="${entry#*|}"
  DIR="$BASE/$FOLDER/issue_persistence_results_validated"
  for file in pair_summary.csv template_summary.csv template_issue_persistence.csv template_hotspot_persistence.csv; do
    if [ ! -s "$DIR/$file" ]; then echo "MISSING: $DIR/$file"; MISSING=1; fi
  done
  ARGS+=(
    --dataset "$LANG"
    "$DIR/pair_summary.csv"
    "$DIR/template_summary.csv"
    "$DIR/template_issue_persistence.csv"
    "$DIR/template_hotspot_persistence.csv"
  )
done

if [ "$MISSING" -eq 0 ]; then
  python -u rq2/quality_issues_and_security_hotspots/generate_rq2_vulnerability_results_with_ownership_fixed.py \
    "${ARGS[@]}" \
    --ownership-directory data_collection \
    --output-directory rq2/quality_issues_and_security_hotspots/rq2_results_validated \
    --finding-aggregation separate
else
  echo 'RQ2 aggregation not started: one or more validated CSV inputs are missing.'
fi
```

Outputs are written to `rq2/quality_issues_and_security_hotspots/rq2_results_validated/`. The script aggregates **existing** validated observations; it does not create `pair_summary.csv` files. If any of the required CSVs is reported as missing, the corresponding input is unavailable and the aggregation cannot complete.

**B. Reproduce the RQ2 quality-dimension summaries (type, severity, software quality).** This is the descriptive script used in the original analysis:

```bash
python rq2/compute_rq2_quality_dimension_persistence.py \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/CSharp/issue_persistence_results_validated/pair_summary.json \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/Java/issue_persistence_results_validated/pair_summary.json \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/Python/issue_persistence_results_validated/pair_summary.json \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/TypeScript/issue_persistence_results_validated/pair_summary.json \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/JavaScript/issue_persistence_results_validated/pair_summary.json \
  --output-dir rq2/quality_issues_and_security_hotspots/rq2_quality_dimension_results_validated
```

**Artifact completeness note:** The `compute_rq2_quality_dimension_persistence.py` file was used in the original analysis, but could not be verified at this path in the published `submission` branch. It must be included in the released artifact for this second command to work; the main RQ2 aggregation above is independent of it. The validated `pair_summary.json` inputs are also used for the RQ3 finding-dimension analysis below.

To reproduce the **matching stage itself**, rather than only the final RQ2 summaries, use the stored `compare_sonar_issue_pairs_manual_validated.py` workflow with the original selected samples and SonarQube exports. That upstream step is not required merely to regenerate these summaries; see [the RQ2 guide](docs/RQ2.md).

### 5. RQ3: build the dataset and reproduce the GEE analyses

RQ3 consumes the RQ1 detailed output and the **same validated** RQ2 pair files used above. It does not consume the aggregated `rq2_results_validated` figures and tables.

**A. Verify prerequisite files.** Run these checks from the repository root:

```bash
ls -lh rq1/reproduced_ownership_results/ownership_evolution_detailed.csv
BASE="rq2/quality_issues_and_security_hotspots/vulnerability_issues"
for lang in CSharp Java Python TypeScript JavaScript; do
  for ext in csv json; do
    ls -lh "$BASE/$lang/issue_persistence_results_validated/pair_summary.$ext"
  done
done
```

**B. Download the external domain classifications (required for the full RQ3 results).** The source-template domain labels used in the secondary-domain models are **not generated by this repository**. They come from the earlier [template-repository-analysis repository](https://github.com/leusonmario/template-repository-analysis/tree/main/rq1/llm_evaluation), specifically the per-language `deepseek_judger_*.csv` files under `rq1/llm_evaluation/`.

From the **root of this replication repository**, clone that repository as a sibling (skip cloning if it is already available locally):

```bash
git clone --depth 1 https://github.com/leusonmario/template-repository-analysis.git ../template-repository-analysis
DOMAIN_DIR="../template-repository-analysis/rq1/llm_evaluation"
ls "$DOMAIN_DIR"/deepseek_judger_*.csv
```

If you already have those CSVs elsewhere, set `DOMAIN_DIR` to the directory containing them instead. The builder must receive both `--domains-dir "$DOMAIN_DIR"` and `--domain-pattern 'deepseek_judger_{domain_token}.csv'`. Without these inputs, the main models can still run, but the **`issue_type_adjusted_secondary_domain`** and **`hotspot_security_category_adjusted_secondary_domain`** models are omitted. For exact archival reproducibility, record the upstream repository commit used for the domain files; its `main` branch may change.

**C. Build the RQ3 dataset.** The `--rq2-pair-pattern` argument is essential: the builder otherwise searches for the older `issue_persistence_results` directory rather than the validated version.

```bash
python rq3/build_rq3_analysis_dataset_with_hotspots.py \
  --rq1-detailed rq1/reproduced_ownership_results/ownership_evolution_detailed.csv \
  --rq2-root rq2/quality_issues_and_security_hotspots/vulnerability_issues \
  --rq2-pair-pattern '{language_dir}/issue_persistence_results_validated/pair_summary.csv' \
  --templates-dir data_collection \
  --generated-reports-dir rq1/reports \
  --domains-dir "$DOMAIN_DIR" \
  --domain-pattern 'deepseek_judger_{domain_token}.csv' \
  --output-dir rq3/reproduced_dataset
```

Expected output: `rq3/reproduced_dataset/rq3_analysis_dataset.csv`, along with diagnostic files. **For exact dataset-level reproduction**, additionally pass the original selected-sample metadata through repeated `--sampling-file LANGUAGE FILE` arguments. Without those files, the builder uses weights already present in `pair_summary.csv` (falling back to 1.0 if absent) but does not include all sampling-metadata columns. The earlier RQ3 coefficient reproduction matched the validated model results after the domain files were supplied; matching coefficients alone does not establish identical dataset metadata.

**D. Run the RQ3 GEE models and robustness analyses.** Explicitly select the five validated JSON files. **Do not use `--rq2-root` for this script** when the RQ2 directory also contains old, filtered, or alternative matching versions: it recursively discovers every `pair_summary.json` under that root.

```bash
python rq3/run_rq3_statistical_analysis_with_hotspots.py \
  --dataset rq3/reproduced_dataset/rq3_analysis_dataset.csv \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/CSharp/issue_persistence_results_validated/pair_summary.json \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/Java/issue_persistence_results_validated/pair_summary.json \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/Python/issue_persistence_results_validated/pair_summary.json \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/TypeScript/issue_persistence_results_validated/pair_summary.json \
  --pair-summary-json rq2/quality_issues_and_security_hotspots/vulnerability_issues/JavaScript/issue_persistence_results_validated/pair_summary.json \
  --age-reference-date 2026-07-22 \
  --age-mode exclude \
  --domain-min-templates 10 \
  --min-adjusted-hotspot-category-templates 10 \
  --output-dir rq3/reproduced_statistics
```

The GEE step requires `rq3_analysis_dataset.csv` from the immediately preceding builder step. The reported RQ3 results depend on the original sampling weights and domain inputs being available and consistently applied. See [RQ3 documentation](docs/RQ3.md).

## License and contact

See the repository's `LICENSE` file **if present**. If none is included, please specify the intended code and dataset reuse terms before external redistribution. For double-blind review, use an anonymized issue or contact channel rather than identifying author information.
