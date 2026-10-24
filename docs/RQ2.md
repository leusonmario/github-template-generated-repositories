# RQ2 — To what extent do concerns remain detectable downstream?

## Scope

RQ2 compares static-analysis findings from source templates with findings from eligible generated repositories. It uses **4,956** generated repositories originating from **713** source templates. Unlike RQ1, repositories with only their initialization commit are excluded.

RQ2 treats **quality issues** (bugs, code smells, and vulnerabilities) and **security hotspots** as separate finding families. The analysis is based on the analyzed snapshots, not a reconstruction of the exact source-template version at adoption time.

## Inputs and prerequisites

To reproduce the analysis starting from source code, the following are needed:

1. The sampled template–repository pairs, including the source-template identity and generated-repository identity.
2. The source-template and generated-repository revisions used in the study.
3. A compatible SonarQube installation, with its version, language analyzers, and quality profiles documented.
4. SonarQube issue and security-hotspot reports for each analyzed repository.
5. The final sampling files (`selected_sample.csv`) containing inclusion probabilities or weights, where applicable.

*Note:* RQ2 is much more resource intensive than running descriptive analysis on existing CSVs. SonarQube findings can change with analyzer version, quality profile, and build configuration. Do not present a fresh scan as an exact replication without aligning these parameters.

## Examine the included artifacts

Run from the repository root:

```bash
# See available scripts and result files before choosing a stage
find rq2 -type f -name '*.py' | sort
find rq2 -type f -name 'selected_sample.csv' | sort
find rq2 -type f -name 'pair_summary.csv' | sort
find rq2 -type f -name 'pair_summary.json' | sort
```

The verified RQ3 builder expects pair summaries using the pattern:

```text
rq2/<language_directory>/issue_persistence_results/pair_summary.csv
```

It also expects the underlying `pair_summary.json` files for the finding-dimension analyses. Inspect your checkout for the corresponding paths; the language directory for C# may be `CSharp`.

## Reproduction sequence

### 1. Select generated repositories

For languages with large template families, apply the study's template-stratified sampling: analyze all eligible repositories from families with at most 20 eligible repositories, otherwise select at most 20 per template using proportional allocation across commit-count strata (2–5, 6–20, >20) and a fixed random seed. Preserve exact `selection_probability` and `sampling_weight` values for later analysis.

For C# and Java, the study analyzes eligible projects that could be compiled and scanned, rather than applying the same capped sample.

Locate the sampling script available in your checkout and consult its actual `--help` interface rather than guessing flags:

```bash
find rq2 -type f -iname '*sample*.py' -print
```

### 2. Run SonarQube and export findings

Analyze the source-template and generated-repository revisions independently, then export quality issues and security hotspots with file path, rule, message, source location, and issue hash (where available). Java generated build artifacts must be excluded consistently from both sides.

The repository's SonarQube export scripts and environment/configuration files should be documented with the exact command used for the submitted results. If those environment-specific commands are not included, the **existing JSON exports** are the practical starting point for re-running the comparison.

### 3. Compare findings

Apply one-to-one matching within each source-template / generated-repository pair:

**Quality issues**

1. Primary: rule + normalized path + issue hash.
2. Fallback for unavailable hash: rule + normalized path + message + line.
3. Additional check for moved/renamed files: rule + hash + message + issue type, without requiring the same file path.

**Security hotspots**

1. Primary: rule + normalized path + message + source location.
2. Line-shift fallback: rule + normalized path + message, ignoring the original line.
3. **Cross-file hotspot matching is excluded from the primary results** following manual validation.

The output should distinguish matched template findings, unmatched template findings, and generated-repository-exclusive findings. These are *detected correspondences*, not direct proofs of historical inheritance, correction, or introduction.

Use the comparison script **actually present in your checkout**. The available RQ2 directory could not be fully enumerated while preparing this documentation, so no unverified executable filename or flags are asserted here. Find the candidate scripts with:

```bash
find rq2 -type f \( -iname '*compare*.py' -o -iname '*match*.py' \) -print
```

Then run the relevant script with `--help` to see its required inputs. To make this a one-command, clean-room reproduction, add the tested invocation and the SonarQube version/configuration in this section.

### 4. Aggregate and validate

Produce pair-level summaries (`pair_summary.csv` and `pair_summary.json`) and language/ownership summaries using inverse-probability weights where applicable. Inspect the output files and compare finding totals with the study.

The manual-validation study sampled **260** observations: **180 matched** and **80 unmatched**. Of the matched examples, 171 were judged plausible, 2 inconclusive, and 7 incorrect; the incorrect matches all involved the excluded cross-file hotspot strategy. The 80 unmatched examples included 38 judged correctly unmatched and 42 inconclusive.

Do not interpret a stratified manual sample as an overall precision or recall estimate.

## Outputs and next stage

The `pair_summary.csv` and `pair_summary.json` files and the original sampling weights are the main bridge to [RQ3](RQ3.md). Preserve both the raw and filtered/sensitivity results separately so that cross-file hotspot matches cannot accidentally enter the primary model.
