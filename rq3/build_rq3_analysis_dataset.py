#!/usr/bin/env python3
"""
build_rq3_analysis_dataset.py

Build a regression-ready dataset for:

RQ3: Which template and generated-repository characteristics are associated
     with vulnerability persistence?

Unit of analysis
----------------
One row = one source-template / generated-repository pair from RQ2.

Primary outcome fields
----------------------
persistent_issue_count
no_longer_present_issue_count

Predictors assembled here
-------------------------
- language
- source-template ownership
- template domain
- generated-repository commit count
- RQ1 evolution level
- template family size (# unique generated repos associated with a template)
- template vulnerability count
- generated-repository creation date and age from rq1/reports/generated_GHRep_from_templates_<LANGUAGE>_with_commits.csv, using data_creation as the creation timestamp and data_update as the observation timestamp when available

Sampling
--------
The existing RQ2 sampling_weight is preserved. For C# and Java, missing
weights are normalized to 1.0.

Outputs
-------
rq3_analysis_dataset.csv
rq3_analysis_dataset_complete_cases.csv
rq3_variable_summary.csv
rq3_data_quality.csv
rq3_domain_distribution.csv
rq3_correlation_candidates.csv
"""

from __future__ import annotations
import argparse, re, json, ast
from pathlib import Path
import numpy as np
import pandas as pd

LANGUAGE_ORDER = ["C#", "Java", "Python", "TypeScript", "JavaScript"]

LANGUAGE_CONFIG = {
    "C#": {
        "aliases": ["C#", "CSharp", "Csharp", "csharp"],
        "rq2_dirs": ["CSharp", "Csharp", "C#", "csharp"],
        "template_tokens": ["C#", "CSharp", "Csharp", "csharp"],
        "domain_tokens": ["C#", "CSharp", "Csharp", "csharp"],
        "generated_report_file": "generated_GHRep_from_templates_Csharp_with_commits.csv",
    },
    "Java": {
        "aliases": ["Java", "java"],
        "rq2_dirs": ["Java", "java"],
        "template_tokens": ["Java", "java"],
        "domain_tokens": ["Java", "java"],
        "generated_report_file": "generated_GHRep_from_templates_Java_with_commits.csv",
    },
    "Python": {
        "aliases": ["Python", "python"],
        "rq2_dirs": ["Python", "python"],
        "template_tokens": ["Python", "python"],
        "domain_tokens": ["Python", "python"],
        "generated_report_file": "generated_GHRep_from_templates_Python_with_commits.csv",
    },
    "TypeScript": {
        "aliases": ["TypeScript", "Typescript", "typescript"],
        "rq2_dirs": ["TypeScript", "Typescript", "typescript"],
        "template_tokens": ["TypeScript", "Typescript", "typescript"],
        "domain_tokens": ["TypeScript", "Typescript", "typescript"],
        "generated_report_file": "generated_GHRep_from_templates_TypeScript_with_commits.csv",
    },
    "JavaScript": {
        "aliases": ["JavaScript", "Javascript", "javascript"],
        "rq2_dirs": ["JavaScript", "Javascript", "javascript"],
        "template_tokens": ["JavaScript", "Javascript", "javascript"],
        "domain_tokens": ["JavaScript", "Javascript", "javascript"],
        "generated_report_file": "generated_GHRep_from_templates_JavaScript_with_commits.csv",
    },
}

PAIR_TEMPLATE_CANDIDATES = ["template_repository", "template_repo", "source_template", "template"]
PAIR_GENERATED_CANDIDATES = ["generated_repository", "generated_repo", "repository", "repo"]
DOMAIN_VALUE_CANDIDATES = ["domain", "category", "classification", "label", "predicted_domain", "final_domain", "final_category", "llm_output"]
DOMAIN_REPO_CANDIDATES = ["template_repository", "template_repo", "repository", "repo", "full_name", "repository_full_name", "original_link_repository", "repo_url"]
GENERATED_METADATA_REPO_CANDIDATES = [
    "generated_repository", "generated_repo", "generated_repo_full_name",
    "generated_repository_full_name", "generated_repo_url",
    "generated_repository_url", "repository", "repo", "full_name",
    "name_with_owner", "repository_full_name", "repo_url",
    "repository_url", "repo_link", "repository_link", "html_url",
    "original_link", "original_link_repository", "url"
]
CREATED_AT_CANDIDATES = ["created_at", "created", "creation_date", "created_date", "data_creation"]
OBSERVATION_DATE_CANDIDATES = ["data_update", "collected_at", "collection_date", "snapshot_date", "data_collected_at"]
LANGUAGE_COLUMN_CANDIDATES = ["language", "programming_language", "primary_language"]


def parse_args():
    p = argparse.ArgumentParser(description="Build an RQ3 vulnerability-persistence analysis dataset.")
    p.add_argument("--rq1-detailed", required=True, type=Path)
    p.add_argument("--rq2-root", required=True, type=Path)
    p.add_argument("--templates-dir", required=True, type=Path)
    p.add_argument("--domains-dir", type=Path, default=None)
    p.add_argument("--domain-pattern", default="*{domain_token}*.csv")
    p.add_argument("--domain-file", action="append", nargs=2, default=[], metavar=("LANGUAGE", "PATH"))
    p.add_argument("--generated-metadata", action="append", nargs=2, default=[], metavar=("LANGUAGE", "PATH"))
    p.add_argument(
        "--generated-reports-dir",
        type=Path,
        default=Path("rq1/reports"),
        help=(
            "Directory containing the RQ1 generated-repository report CSVs. "
            "By default the script scans rq1/reports for files matching "
            "generated_GHRep_from_templates_<LANGUAGE>_with_commits.csv and uses their created_at metadata."
        ),
    )
    p.add_argument(
        "--generated-reports-pattern",
        default=None,
        help=(
            "Optional override glob pattern. By default, the script uses an "
            "explicit per-language filename such as "
            "generated_GHRep_from_templates_Csharp_with_commits.csv."
        ),
    )
    p.add_argument(
        "--data-location",
        type=Path,
        default=None,
        help=(
            "Directory containing CSV metadata for generated repositories. "
            "The script recursively discovers CSVs with repository identity + created_at "
            "and matches them to the RQ2 generated repositories. Explicit "
            "--generated-metadata files take precedence."
        ),
    )
    p.add_argument("--output-dir", type=Path, default=Path("rq3_results"))
    p.add_argument("--rq2-pair-pattern", default="{language_dir}/issue_persistence_results/pair_summary.csv")
    p.add_argument("--templates-pattern", default="all_templates_{template_token}.csv")
    p.add_argument(
        "--age-reference-date",
        default=None,
        help=(
            "Optional fixed YYYY-MM-DD observation date for repository age. "
            "If omitted, the script uses a per-row data_update/collection date when "
            "available in data_location. It does not use today's date."
        ),
    )
    return p.parse_args()


def canonicalize_language(value):
    value = str(value).strip()
    for canonical, config in LANGUAGE_CONFIG.items():
        if any(value.casefold() == x.casefold() for x in config["aliases"]):
            return canonical
    raise ValueError(f"Unsupported language: {value}")


def build_override_map(items):
    result = {}
    for language, path in items:
        result[canonicalize_language(language)] = Path(path).expanduser().resolve()
    return result


def find_column(df, candidates, required=True):
    normalized = {str(c).strip().casefold(): c for c in df.columns}
    for candidate in candidates:
        if candidate.casefold() in normalized:
            return normalized[candidate.casefold()]
    if required:
        raise ValueError(f"Could not find any of {candidates}. Available: {list(df.columns)}")
    return None


def normalize_repo_reference(value):
    if pd.isna(value):
        return None
    text = str(value).strip().replace("\\", "/").lower().rstrip("/")
    if not text:
        return None
    if text.endswith(".git"):
        text = text[:-4]
    for prefix in ["https://github.com/", "http://github.com/", "https://www.github.com/", "http://www.github.com/", "git@github.com:", "github.com/", "www.github.com/"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    parts = [x for x in text.split("/") if x]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def normalize_repo_from_owner_name(owner, name):
    if pd.isna(owner) or pd.isna(name):
        return None
    return normalize_repo_reference(f"{owner}/{name}")


def normalize_owner(value):
    if pd.isna(value):
        return None
    x = str(value).strip().casefold()
    if x == "user":
        return "User"
    if x in {"organization", "organisation", "org"}:
        return "Organization"
    return None


def clean_domain(value):
    if pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    return text if text else None


def extract_domain_from_llm_output(value):
    """
    Robustly extract `category` from the domain-classification LLM output.

    Handles:
      1) plain JSON objects
      2) Markdown fenced ```json ... ``` objects
      3) CSV-escaped doubled quotes (""category"")
      4) non-breaking spaces / unusual whitespace
      5) Python-dict-like payloads using single quotes
      6) minor malformed JSON, via a conservative category regex
    """
    if pd.isna(value):
        return None

    raw = str(value)
    if not raw.strip():
        return None

    text = (
        raw.replace("\u00a0", " ")
           .replace("\u2007", " ")
           .replace("\u202f", " ")
           .replace("\ufeff", "")
           .strip()
    )

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)

    candidates = [text]

    deescaped = text.replace('""', '"')
    if deescaped != text:
        candidates.append(deescaped)

    expanded_candidates = []
    for candidate in candidates:
        expanded_candidates.append(candidate)
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            expanded_candidates.append(candidate[start:end + 1])

    seen = set()
    candidates = []
    for candidate in expanded_candidates:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                category = clean_domain(parsed.get("category"))
                if category:
                    return category
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                category = clean_domain(parsed.get("category"))
                if category:
                    return category
        except (ValueError, SyntaxError, TypeError):
            pass

    match = re.search(
        r'(?is)["\']?\s*category\s*["\']?\s*:\s*["\']\s*([^"\']+?)\s*["\']',
        deescaped,
    )
    if match:
        return clean_domain(match.group(1))

    match = re.search(
        r'(?is)["\']?\s*category\s*["\']?\s*:\s*([^,\}\r\n]+)',
        deescaped,
    )
    if match:
        value = match.group(1).strip().strip('"\'')
        return clean_domain(value)

    return None


    text = str(value).strip()
    if not text:
        return None

    # Remove common Markdown code fences.
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # Some exports may preserve doubled quotes.
    candidates = [text]
    if '""category""' in text or '""reasoning""' in text:
        candidates.append(text.replace('""', '"'))

    for candidate in candidates:
        candidate = candidate.strip()

        # If prose accidentally surrounds the JSON, isolate the outer object.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_text = candidate[start:end + 1]
        else:
            json_text = candidate

        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                category = parsed.get("category")
                category = clean_domain(category)
                if category:
                    return category
        except (json.JSONDecodeError, TypeError):
            pass

    # Conservative fallback for malformed JSON.
    match = re.search(
        r'["\']?category["\']?\s*:\s*["\']([^"\']+)["\']',
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return clean_domain(match.group(1))

    return None


def resolve_pair_file(root, pattern, language):
    tried = []
    for language_dir in LANGUAGE_CONFIG[language]["rq2_dirs"]:
        path = root / pattern.format(language=language, language_dir=language_dir)
        tried.append(path)
        if path.exists():
            return path
    raise FileNotFoundError(f"No pair_summary.csv for {language}. Tried:\n" + "\n".join(map(str, tried)))


def resolve_template_file(root, pattern, language):
    tried = []
    for token in LANGUAGE_CONFIG[language]["template_tokens"]:
        path = root / pattern.format(language=language, template_token=token)
        tried.append(path)
        if path.exists():
            return path
    raise FileNotFoundError(f"No all_templates file for {language}. Tried:\n" + "\n".join(map(str, tried)))


def resolve_domain_file(language, explicit, domains_dir, pattern):
    if language in explicit:
        return explicit[language]
    if domains_dir is None:
        return None
    candidates = []
    for token in LANGUAGE_CONFIG[language]["domain_tokens"]:
        candidates.extend(sorted(domains_dir.glob(pattern.format(language=language, domain_token=token))))
    candidates = [p for p in dict.fromkeys(candidates) if p.is_file() and "all_templates" not in p.name.lower() and "pair_summary" not in p.name.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"[WARN] Multiple domain files for {language}; use --domain-file explicitly:")
        for p in candidates:
            print("      ", p)
    return None


def build_template_owner_map(template_file):
    df = pd.read_csv(template_file)
    owner_col = find_column(df, ["owner_type"])
    username_col = find_column(df, ["username"], required=False)
    project_col = find_column(df, ["project_name"], required=False)
    repo_col = find_column(df, DOMAIN_REPO_CANDIDATES, required=False)

    if username_col is not None and project_col is not None:
        repos = df.apply(lambda r: normalize_repo_from_owner_name(r[username_col], r[project_col]), axis=1)
    elif repo_col is not None:
        repos = df[repo_col].apply(normalize_repo_reference)
    else:
        raise ValueError(f"Cannot construct template identity from {template_file}")

    out = pd.DataFrame({
        "template_repository": repos,
        "owner_type": df[owner_col].apply(normalize_owner),
    }).dropna(subset=["template_repository"])

    conflicts = out.dropna(subset=["owner_type"]).groupby("template_repository")["owner_type"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting owner values in {template_file}: {conflicts[conflicts > 1].index.tolist()[:10]}")
    return out.drop_duplicates("template_repository")


def build_domain_map(domain_file):
    df = pd.read_csv(domain_file, low_memory=False)

    username_col = find_column(df, ["username"], required=False)
    project_col = find_column(df, ["project_name"], required=False)
    repo_col = find_column(df, DOMAIN_REPO_CANDIDATES, required=False)

    if username_col is not None and project_col is not None:
        repos = df.apply(
            lambda r: normalize_repo_from_owner_name(
                r[username_col], r[project_col]
            ),
            axis=1,
        )
    elif repo_col is not None:
        repos = df[repo_col].apply(normalize_repo_reference)
    else:
        raise ValueError(
            f"Cannot construct repository identity from {domain_file}. "
            "Expected repo_url, repository/full_name, or username/project_name."
        )

    if "llm_output" in df.columns:
        domains = df["llm_output"].apply(extract_domain_from_llm_output)
        domain_source = "llm_output.category"
    else:
        domain_col = find_column(
            df,
            [
                "domain",
                "category",
                "classification",
                "label",
                "predicted_domain",
                "final_domain",
                "final_category",
            ],
        )
        domains = df[domain_col].apply(clean_domain)
        domain_source = domain_col

    out = pd.DataFrame({
        "template_repository": repos,
        "domain": domains,
    })

    invalid_repo = int(out["template_repository"].isna().sum())
    out = out.dropna(subset=["template_repository"]).copy()

    conflicts = (
        out.dropna(subset=["domain"])
        .groupby("template_repository")["domain"]
        .nunique()
    )
    if (conflicts > 1).any():
        raise ValueError(
            f"Conflicting domain labels in {domain_file}: "
            f"{conflicts[conflicts > 1].index.tolist()[:10]}"
        )

    missing_domain = int(out["domain"].isna().sum())
    parsed_domain = int(out["domain"].notna().sum())

    print(
        f"[INFO] Domain file {Path(domain_file).name}: "
        f"rows={len(df):,}, valid_repo={len(out):,}, "
        f"parsed_domain={parsed_domain:,}, missing_domain={missing_domain:,}, "
        f"invalid_repo={invalid_repo:,}, source={domain_source}"
    )

    return out.drop_duplicates("template_repository")




def load_rq1_detailed(path):
    df = pd.read_csv(path)
    for col in ["language", "generated_repo", "template_repo", "commit_count"]:
        if col not in df.columns:
            raise ValueError(f"{path} is missing '{col}'. Available: {list(df.columns)}")

    work = df.copy()
    work["language"] = work["language"].apply(canonicalize_language)
    work["generated_repository"] = work["generated_repo"].apply(normalize_repo_reference)
    work["template_repository"] = work["template_repo"].apply(normalize_repo_reference)
    work["commit_count"] = pd.to_numeric(work["commit_count"], errors="coerce")

    if "evolution_level" not in work.columns:
        def classify(n):
            if pd.isna(n): return None
            if n == 1: return "Not evolved"
            if 2 <= n <= 6: return "Slightly evolved"
            if 7 <= n <= 20: return "Moderately evolved"
            if n > 20: return "Substantially evolved"
            return None
        work["evolution_level"] = work["commit_count"].apply(classify)

    family = (
        work.dropna(subset=["template_repository", "generated_repository"])
        .groupby(["language", "template_repository"])["generated_repository"]
        .nunique()
        .rename("template_family_size")
        .reset_index()
    )

    pair_vars = (
        work[["language", "template_repository", "generated_repository", "commit_count", "evolution_level"]]
        .dropna(subset=["template_repository", "generated_repository"])
        .sort_values("commit_count", na_position="last")
        .drop_duplicates(["language", "template_repository", "generated_repository"], keep="last")
        .merge(family, on=["language", "template_repository"], how="left", validate="many_to_one")
    )
    return pair_vars


def _parse_reference_date(reference_date):
    if reference_date is None:
        return None
    ts = pd.to_datetime(reference_date, errors="raise", utc=True)
    return ts


def _metadata_identity_columns(df):
    repo_col = find_column(
        df, GENERATED_METADATA_REPO_CANDIDATES, required=False
    )
    username_col = find_column(
        df, ["username", "owner", "repo_owner"], required=False
    )
    project_col = find_column(
        df, ["project_name", "name", "repo_name"], required=False
    )
    return repo_col, username_col, project_col


def _build_repository_series(df):
    repo_col, username_col, project_col = _metadata_identity_columns(df)

    if repo_col is not None:
        return df[repo_col].apply(normalize_repo_reference)

    if username_col is not None and project_col is not None:
        return df.apply(
            lambda r: normalize_repo_from_owner_name(
                r[username_col], r[project_col]
            ),
            axis=1,
        )

    raise ValueError(
        "Could not identify a repository column. Expected a URL/full-name "
        "column or owner/name columns."
    )


def _build_age_frame(
    df,
    language,
    source_path,
    reference_date=None,
    target_repositories=None,
):
    created_col = find_column(df, CREATED_AT_CANDIDATES)
    observation_col = find_column(
        df, OBSERVATION_DATE_CANDIDATES, required=False
    )

    repos = _build_repository_series(df)
    created = pd.to_datetime(df[created_col], errors="coerce", utc=True)

    fixed_ref = _parse_reference_date(reference_date)

    if fixed_ref is not None:
        observed = pd.Series(
            fixed_ref,
            index=df.index,
            dtype="datetime64[ns, UTC]",
        )
        reference_source = "fixed_age_reference_date"
    elif observation_col is not None:
        observed = pd.to_datetime(
            df[observation_col], errors="coerce", utc=True
        )
        reference_source = observation_col
    else:
        # Do not silently calculate age relative to today's date or to the
        # newest repository in the file. The creation timestamp is still
        # retained, while age remains missing until a valid observation
        # date is supplied.
        observed = pd.Series(
            pd.NaT,
            index=df.index,
            dtype="datetime64[ns, UTC]",
        )
        reference_source = "missing"

    age_days = (observed - created).dt.total_seconds() / 86400.0
    # Protect against malformed dates where observation precedes creation.
    age_days = age_days.where(age_days >= 0)

    out = pd.DataFrame({
        "language": language,
        "generated_repository": repos,
        "generated_created_at": created,
        "age_reference_at": observed,
        "age_reference_source": reference_source,
        "repository_age_days": age_days,
        "repository_age_years": age_days / 365.25,
        "generated_metadata_source": str(source_path),
    })

    out = out.dropna(subset=["generated_repository"]).copy()

    if target_repositories is not None:
        out = out[
            out["generated_repository"].isin(target_repositories)
        ].copy()

    return out


def _deduplicate_generated_metadata(metadata):
    if metadata.empty:
        return metadata

    work = metadata.copy()

    # Prefer rows with a valid creation timestamp and then a valid age.
    work["_created_ok"] = work["generated_created_at"].notna().astype(int)
    work["_age_ok"] = work["repository_age_years"].notna().astype(int)

    work = work.sort_values(
        ["generated_repository", "_age_ok", "_created_ok"],
        ascending=[True, False, False],
    )

    # Report conflicting creation timestamps before choosing one record.
    conflicts = (
        work.dropna(subset=["generated_created_at"])
        .groupby("generated_repository")["generated_created_at"]
        .nunique()
    )
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        print(
            f"[WARN] {len(conflicts):,} generated repositories have "
            "conflicting created_at values across metadata files. "
            "The first highest-quality record is retained."
        )

    work = work.drop_duplicates(
        ["language", "generated_repository"], keep="first"
    )

    return work.drop(columns=["_created_ok", "_age_ok"])


def load_generated_metadata(
    path,
    language,
    reference_date=None,
    target_repositories=None,
):
    df = pd.read_csv(path)

    out = _build_age_frame(
        df,
        language=language,
        source_path=path,
        reference_date=reference_date,
        target_repositories=target_repositories,
    )

    if (
        reference_date is None
        and (out["age_reference_source"] == "missing").all()
    ):
        print(
            f"[WARN] {path}: created_at was found, but no data_update/"
            "collection-date column was found. generated_created_at is "
            "populated, but repository age remains missing. Supply "
            "--age-reference-date to calculate age."
        )

    return _deduplicate_generated_metadata(out)


def _csv_has_generated_metadata_columns(path):
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return False

    created_col = find_column(
        header, CREATED_AT_CANDIDATES, required=False
    )
    if created_col is None:
        return False

    repo_col, username_col, project_col = _metadata_identity_columns(header)

    return (
        repo_col is not None
        or (username_col is not None and project_col is not None)
    )



def resolve_generated_report_file(
    generated_reports_dir,
    language,
    override_pattern=None,
):
    """
    Resolve the generated-repository metadata file for one language.

    Default filenames:
      C#         -> generated_GHRep_from_templates_Csharp_with_commits.csv
      Java       -> generated_GHRep_from_templates_Java_with_commits.csv
      Python     -> generated_GHRep_from_templates_Python_with_commits.csv
      TypeScript -> generated_GHRep_from_templates_TypeScript_with_commits.csv
      JavaScript -> generated_GHRep_from_templates_JavaScript_with_commits.csv
    """
    root = Path(generated_reports_dir).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    if override_pattern:
        matches = sorted(root.rglob(override_pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise FileNotFoundError(
                f"No file matching override pattern '{override_pattern}' "
                f"under {root}"
            )
        raise ValueError(
            f"Override pattern '{override_pattern}' matched multiple files "
            f"under {root}: {matches}"
        )

    filename = LANGUAGE_CONFIG[language]["generated_report_file"]

    # Prefer the file directly under rq1/reports, but allow subdirectories.
    direct = root / filename
    if direct.exists():
        return direct

    matches = sorted(root.rglob(filename))
    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Generated-report file for {language} was not found.\n"
            f"Expected: {filename}\n"
            f"Under: {root}"
        )

    raise ValueError(
        f"Multiple generated-report files found for {language}: {matches}"
    )


def discover_generated_metadata(
    data_location,
    language,
    target_repositories,
    reference_date=None,
    file_pattern="*.csv",
    source_label="metadata",
):
    """
    Recursively inspect matching CSVs and retain only rows whose normalized
    repository identifier appears in the RQ2 generated-repository set.

    This is intentionally repository-driven rather than filename-driven:
    files do not need to encode the programming language in their names.
    """
    data_location = Path(data_location).expanduser().resolve()

    if not data_location.exists():
        raise FileNotFoundError(data_location)

    csv_files = sorted(data_location.rglob(file_pattern))

    candidates = [
        path for path in csv_files
        if _csv_has_generated_metadata_columns(path)
    ]

    if not candidates:
        print(
            f"[WARN] No {source_label} CSV matching '{file_pattern}' with "
            f"repository identity + created_at was found under {data_location}."
        )
        return pd.DataFrame(columns=[
            "language",
            "generated_repository",
            "generated_created_at",
            "age_reference_at",
            "age_reference_source",
            "repository_age_days",
            "repository_age_years",
            "generated_metadata_source",
        ])

    frames = []
    matched_files = []

    for path in candidates:
        try:
            df = pd.read_csv(path, low_memory=False)
            frame = _build_age_frame(
                df,
                language=language,
                source_path=path,
                reference_date=reference_date,
                target_repositories=target_repositories,
            )

            if not frame.empty:
                frames.append(frame)
                matched_files.append(path)

        except Exception as exc:
            print(f"[WARN] Skipping {source_label} candidate {path}: {exc}")

    if not frames:
        print(
            f"[WARN] No RQ2 generated repositories for {language} were "
            f"matched to {source_label} under {data_location}."
        )
        return pd.DataFrame(columns=[
            "language",
            "generated_repository",
            "generated_created_at",
            "age_reference_at",
            "age_reference_source",
            "repository_age_days",
            "repository_age_years",
            "generated_metadata_source",
        ])

    out = _deduplicate_generated_metadata(
        pd.concat(frames, ignore_index=True)
    )

    age_count = int(out["repository_age_years"].notna().sum())
    created_count = int(out["generated_created_at"].notna().sum())

    print(
        f"[INFO] {language}: {source_label} matched {len(out):,}/"
        f"{len(target_repositories):,} generated repositories from "
        f"{len(matched_files):,} CSV file(s); created_at={created_count:,}, "
        f"age={age_count:,}."
    )

    if age_count == 0 and reference_date is None:
        print(
            "[WARN] created_at was matched, but repository ages could not "
            "be calculated because no observation/data_update date was found. "
            "Re-run with --age-reference-date YYYY-MM-DD if age is required."
        )

    return out


def load_pair_summary(path, language):
    df = pd.read_csv(path)
    template_col = find_column(df, PAIR_TEMPLATE_CANDIDATES)
    generated_col = find_column(df, PAIR_GENERATED_CANDIDATES)

    for col in ["persistent_issue_count", "no_longer_present_issue_count", "template_issue_count"]:
        if col not in df.columns:
            raise ValueError(f"{path} is missing '{col}'. Available: {list(df.columns)}")

    out = df.copy()
    out["language"] = language
    out["template_repository"] = out[template_col].apply(normalize_repo_reference)
    out["generated_repository"] = out[generated_col].apply(normalize_repo_reference)

    for col in ["persistent_issue_count", "no_longer_present_issue_count", "template_issue_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "sampling_weight" not in out.columns:
        out["sampling_weight"] = 1.0
    else:
        out["sampling_weight"] = pd.to_numeric(out["sampling_weight"], errors="coerce").fillna(1.0)
        out.loc[out["sampling_weight"] <= 0, "sampling_weight"] = 1.0

    out["template_vulnerability_observations"] = out["persistent_issue_count"] + out["no_longer_present_issue_count"]
    out["persistence_rate_recomputed"] = np.where(
        out["template_vulnerability_observations"] > 0,
        out["persistent_issue_count"] / out["template_vulnerability_observations"],
        np.nan,
    )

    keep = [
        "language", "template_repository", "generated_repository",
        "persistent_issue_count", "no_longer_present_issue_count",
        "template_issue_count", "template_vulnerability_observations",
        "persistence_rate_recomputed", "sampling_weight",
    ]
    for col in ["new_issue_count", "persistence_rate", "persistence_percentage",
                "selection_probability", "sampling_method", "evolution_band", "pair_id"]:
        if col in out.columns:
            keep.append(col)

    return (
        out[keep]
        .dropna(subset=["template_repository", "generated_repository"])
        .drop_duplicates(["language", "template_repository", "generated_repository"], keep="first")
    )


def safe_log1p(series):
    x = pd.to_numeric(series, errors="coerce")
    return np.log1p(x.where(x >= 0))


def main():
    args = parse_args()
    rq1_path = args.rq1_detailed.expanduser().resolve()
    rq2_root = args.rq2_root.expanduser().resolve()
    templates_dir = args.templates_dir.expanduser().resolve()
    domains_dir = args.domains_dir.expanduser().resolve() if args.domains_dir else None
    data_location = args.data_location.expanduser().resolve() if args.data_location else None
    generated_reports_dir = args.generated_reports_dir.expanduser().resolve() if args.generated_reports_dir else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    explicit_domain_files = build_override_map(args.domain_file)
    generated_metadata_files = build_override_map(args.generated_metadata)
    rq1_pairs = load_rq1_detailed(rq1_path)

    all_rows, quality_rows, unmatched_domain_rows = [], [], []

    for language in LANGUAGE_ORDER:
        pair_file = resolve_pair_file(rq2_root, args.rq2_pair_pattern, language)
        template_file = resolve_template_file(templates_dir, args.templates_pattern, language)
        domain_file = resolve_domain_file(language, explicit_domain_files, domains_dir, args.domain_pattern)

        pairs = load_pair_summary(pair_file, language)
        owners = build_template_owner_map(template_file)

        data = pairs.merge(owners, on="template_repository", how="left", validate="many_to_one")

        rq1_language = rq1_pairs[rq1_pairs["language"] == language].copy()
        data = data.merge(
            rq1_language[["template_repository", "generated_repository", "commit_count", "evolution_level", "template_family_size"]],
            on=["template_repository", "generated_repository"],
            how="left",
            validate="one_to_one",
        )

        if domain_file is not None:
            domains = build_domain_map(domain_file)
            data = data.merge(domains, on="template_repository", how="left", validate="many_to_one")
        else:
            data["domain"] = pd.NA

        if domain_file is not None:
            missing_domain_pairs = data[data["domain"].isna()].copy()
            if not missing_domain_pairs.empty:
                unmatched_summary = (
                    missing_domain_pairs
                    .groupby(["language", "template_repository"], dropna=False)
                    .agg(
                        rq2_pairs=("generated_repository", "nunique"),
                        pairs_with_template_vulnerabilities=(
                            "template_vulnerability_observations",
                            lambda s: int((s > 0).sum())
                        ),
                    )
                    .reset_index()
                )
                unmatched_summary["domain_file"] = str(domain_file)
                unmatched_domain_rows.append(unmatched_summary)

        target_generated_repositories = set(
            data["generated_repository"].dropna().unique()
        )

        # Generated-repository metadata priority:
        # 1) explicit per-language --generated-metadata
        # 2) rq1/reports/generated_GHRep_from_templates_<LANGUAGE>_with_commits.csv (preferred study source)
        # 3) optional --data-location fallback
        if language in generated_metadata_files:
            metadata = load_generated_metadata(
                generated_metadata_files[language],
                language,
                args.age_reference_date,
                target_repositories=target_generated_repositories,
            )
        elif generated_reports_dir is not None and generated_reports_dir.exists():
            try:
                generated_report_file = resolve_generated_report_file(
                    generated_reports_dir,
                    language,
                    args.generated_reports_pattern,
                )

                metadata = load_generated_metadata(
                    generated_report_file,
                    language,
                    args.age_reference_date,
                    target_repositories=target_generated_repositories,
                )

                print(
                    f"[INFO] {language}: using generated-report file "
                    f"{generated_report_file.name}; matched "
                    f"{len(metadata):,}/{len(target_generated_repositories):,} "
                    "RQ2 generated repositories."
                )

            except (FileNotFoundError, ValueError) as exc:
                print(f"[WARN] {exc}")
                metadata = pd.DataFrame()
        elif data_location is not None:
            metadata = discover_generated_metadata(
                data_location,
                language,
                target_generated_repositories,
                args.age_reference_date,
                file_pattern="*.csv",
                source_label="data-location",
            )
        else:
            metadata = pd.DataFrame()

        if not metadata.empty:
            data = data.merge(
                metadata,
                on=["language", "generated_repository"],
                how="left",
                validate="one_to_one",
            )
        else:
            data["generated_created_at"] = pd.NaT
            data["age_reference_at"] = pd.NaT
            data["age_reference_source"] = pd.NA
            data["repository_age_days"] = np.nan
            data["repository_age_years"] = np.nan
            data["generated_metadata_source"] = pd.NA

        data["log_commit_count"] = safe_log1p(data["commit_count"])
        data["log_template_family_size"] = safe_log1p(data["template_family_size"])
        data["log_template_issue_count"] = safe_log1p(data["template_issue_count"])

        data["has_template_vulnerabilities"] = data["template_vulnerability_observations"] > 0
        data["ownership_available"] = data["owner_type"].notna()
        data["domain_available"] = data["domain"].notna()
        data["commit_count_available"] = data["commit_count"].notna()
        data["age_available"] = data["repository_age_years"].notna()

        quality_rows.append({
            "Language": language,
            "Pair summary": str(pair_file),
            "Template metadata": str(template_file),
            "Domain file": str(domain_file) if domain_file else "",
            "Pairs": len(data),
            "Pairs with template vulnerabilities": int(data["has_template_vulnerabilities"].sum()),
            "Owner matched": int(data["owner_type"].notna().sum()),
            "Domain matched": int(data["domain"].notna().sum()),
            "Commit count matched": int(data["commit_count"].notna().sum()),
            "Created at matched": int(data["generated_created_at"].notna().sum()),
            "Generated metadata sources": int(data["generated_metadata_source"].nunique(dropna=True)),
            "Age matched": int(data["repository_age_years"].notna().sum()),
            "Unique templates": int(data["template_repository"].nunique()),
            "Unique generated repositories": int(data["generated_repository"].nunique()),
        })
        all_rows.append(data)

    dataset = pd.concat(all_rows, ignore_index=True)

    initial_model_columns = [
        "persistent_issue_count", "no_longer_present_issue_count", "sampling_weight",
        "log_commit_count", "owner_type", "log_template_family_size", "language", "domain",
    ]
    complete = dataset.loc[
        dataset["has_template_vulnerabilities"] &
        dataset[initial_model_columns].notna().all(axis=1)
    ].copy()

    summary_rows = []
    for language, group in dataset.groupby("language", sort=False):
        summary_rows.append({
            "Language": language,
            "Pairs": len(group),
            "Pairs with template vulnerabilities": int(group["has_template_vulnerabilities"].sum()),
            "Templates": int(group["template_repository"].nunique()),
            "Generated repositories": int(group["generated_repository"].nunique()),
            "Median commit count": group["commit_count"].median(),
            "Mean commit count": group["commit_count"].mean(),
            "Median template family size": group["template_family_size"].median(),
            "Mean template family size": group["template_family_size"].mean(),
            "Domains": int(group["domain"].nunique(dropna=True)),
            "Complete cases (initial model)": int(complete["language"].eq(language).sum()),
        })
    variable_summary = pd.DataFrame(summary_rows)

    domain_distribution = (
        dataset.dropna(subset=["domain"])
        .groupby(["language", "domain"])
        .agg(templates=("template_repository", "nunique"), pairs=("generated_repository", "size"))
        .reset_index()
        .sort_values(["language", "templates"], ascending=[True, False])
    )

    numeric_candidates = [
        "commit_count", "log_commit_count", "template_family_size",
        "log_template_family_size", "template_issue_count",
        "log_template_issue_count", "repository_age_years",
        "persistence_rate_recomputed",
    ]
    corr = dataset[numeric_candidates].corr(method="spearman", min_periods=20)
    correlations = (
        corr.stack().rename("spearman_rho").reset_index()
        .rename(columns={"level_0": "variable_1", "level_1": "variable_2"})
    )
    correlations = correlations[
        correlations["variable_1"] < correlations["variable_2"]
    ].sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)

    paths = {
        "dataset": output_dir / "rq3_analysis_dataset.csv",
        "complete": output_dir / "rq3_analysis_dataset_complete_cases.csv",
        "summary": output_dir / "rq3_variable_summary.csv",
        "quality": output_dir / "rq3_data_quality.csv",
        "domain": output_dir / "rq3_domain_distribution.csv",
        "corr": output_dir / "rq3_correlation_candidates.csv",
        "unmatched_domains": output_dir / "rq3_unmatched_domain_templates.csv",
    }

    dataset.to_csv(paths["dataset"], index=False)
    complete.to_csv(paths["complete"], index=False)
    variable_summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(quality_rows).to_csv(paths["quality"], index=False)
    domain_distribution.to_csv(paths["domain"], index=False)
    correlations.to_csv(paths["corr"], index=False)

    if unmatched_domain_rows:
        unmatched_domains = pd.concat(
            unmatched_domain_rows,
            ignore_index=True,
        ).sort_values(
            ["language", "rq2_pairs"],
            ascending=[True, False],
        )
    else:
        unmatched_domains = pd.DataFrame(columns=[
            "language",
            "template_repository",
            "rq2_pairs",
            "pairs_with_template_vulnerabilities",
            "domain_file",
        ])

    unmatched_domains.to_csv(
        paths["unmatched_domains"],
        index=False,
    )

    print("\nRQ3 DATASET")
    print("=" * 90)
    print(f"Rows: {len(dataset):,}")
    print(f"Templates: {dataset['template_repository'].nunique():,}")
    print(f"Generated repositories: {dataset['generated_repository'].nunique():,}")
    print(f"Pairs with >=1 template vulnerability: {int(dataset['has_template_vulnerabilities'].sum()):,}")
    print(f"Complete cases for initial model: {len(complete):,}")

    print("\nMATCHING / DATA QUALITY")
    print("=" * 90)
    print(pd.DataFrame(quality_rows).to_string(index=False))

    print("\nSuggested initial model:")
    print("persistent vs not-persistent ~ log_commit_count + owner_type + log_template_family_size + language + domain")
    print("\nRepository age uses created_at from rq1/reports/generated_GHRep_from_templates_<LANGUAGE>_with_commits.csv plus either a data_update/collection date in those reports or --age-reference-date.")

    print("\nSaved:")
    for p in paths.values():
        print(f"- {p}")


if __name__ == "__main__":
    main()
