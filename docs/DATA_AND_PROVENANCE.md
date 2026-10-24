# Data collection and provenance

## Source-template collection

The study begins from an earlier dataset of GitHub template repositories covering five programming languages. Candidates with at least **10 stars** are queried by their `owner/repository` identifier in GitHub repository names, descriptions, and README files.

Large GitHub Search API result sets are split into smaller creation-date intervals to avoid the API's 1,000-result limit. Candidate generated repositories are validated using the GitHub REST API's `template_repository` metadata: the template's `full_name` must agree with the target source template.

Repositories are restricted to the same programming language as their source template and must have been created after it. Forks of the source-template repositories are excluded. Additional filtering excludes unavailable GitHub repositories from the final RQ1 dataset.

## Repository organization

```text
data_collection/     Candidate discovery and input template metadata
rq1/                 Generated-repository metadata and commit-count analysis
rq2/                 Sample selection and SonarQube results
rq3/                 Merged pair-level dataset and statistical models
survey/              Questionnaire and aggregate survey evidence
```

The above is a *stage-level* map. Inspect the corresponding directory for the actual filenames and scripts supplied in the checkout.

## Reproduction levels

- **Level A: Reuse prepared outputs.** Start from the recorded RQ1 per-language reports, RQ2 pair summaries and selected-sample weights, and RQ3 input data. This is the most practical way to reproduce tables and statistical models.
- **Level B: Recompute GitHub metadata.** Rerun discovery or commit counting, using an authenticated API where required. GitHub state and availability may change; new results are not necessarily identical to the frozen study snapshot.
- **Level C: Rerun SonarQube on code snapshots.** Requires pinned repository revisions, the recorded SonarQube version and quality profiles, build environments, and sufficient computing resources.

If scripts, pinned versions, snapshot lists, or frozen result files needed for Levels B/C are not yet included, provide them before claiming end-to-end reproducibility.

## Dataset boundaries

RQ1: 123,360 generated repositories / 1,533 templates. RQ2: 4,956 sampled generated repositories / 713 templates; one-commit repositories are excluded. RQ3 uses a further restricted population of pair observations where the source template has at least one corresponding detected finding.

Finding correspondences are computed between observed states. They do not prove that a concern was inherited at template adoption or that an unmatched finding was fixed.
