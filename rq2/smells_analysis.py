#!/usr/bin/env python3
import os
import subprocess
import shutil
import requests
import time
import pandas as pd
import json
from collections import Counter

# -----------------------------
# CONFIGURATION
# -----------------------------
#SONARQUEBE 2
#SONAR_URL = "http://localhost:9001"
#SONAR_TOKEN = "sqa_5bdb2fef18703dca3070dad7592aad617e1d48b7"     # MUST be admin-level token

#SONARQUBE 3
SONAR_URL = "http://localhost:9002"
SONAR_TOKEN = "squ_f75043d923b1b7855a7c6e44ba031a9997df5881"     # MUST be admin-level token


REPO_DIR = "./temp_repos"
GENERATED_BASED_REPOSITORIES = "sample/persistence_sample_balanced/"
OUTPUT_CSV_BASE = "vulnerability_results_"
INPUT_CSV = "selected_sample.csv"
JSON_OUTPUT_DIR = "vulnerability_issues/JavaScript/vulnerability_details"
METRICS = ["ncloc", "files", "functions", "vulnerabilities", "security_hotspots", "code_smells", "bugs"]

# Path to sonar-scanner (Linux)
SONAR_SCANNER = "/opt/sonar-scanner/bin/sonar-scanner"

os.makedirs(REPO_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def run_cmd(cmd):
    """Run shell command and stream output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout)
    return result.returncode == 0


def get_vulnerability_details(project_key, repo_name):
    """Fetch detailed vulnerability information and severity counts."""
    url = f"{SONAR_URL}/api/issues/search"
    params = {
        "componentKeys": project_key,
        "types": "VULNERABILITY",
        "ps": 500
    }

    all_issues = []
    page = 1

    while True:
        params["p"] = page
        r = requests.get(url, auth=(SONAR_TOKEN, ""), params=params)

        if r.status_code != 200:
            print(f"⚠️ Failed to get issues for {repo_name}: {r.text}")
            break

        data = r.json()
        issues = data.get("issues", [])
        all_issues.extend(issues)

        if len(issues) < 500:
            break

        page += 1

    # Save raw JSON
    if all_issues:
        json_path = os.path.join(JSON_OUTPUT_DIR, f"{repo_name}_vulnerabilities.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_issues, f, indent=2)
        print(f"💾 Saved {len(all_issues)} vulnerability issues to {json_path}")

    severities = [issue.get("severity", "UNKNOWN") for issue in all_issues]
    return dict(Counter(severities))


def analyze_repo(git_url):
    """Clone, scan, collect results, delete repo."""
    # Extract owner + repo
    segments = git_url.rstrip("/").split("/")
    owner = segments[-2]
    repo_name = segments[-1].replace(".git", "")

    repo_path = os.path.join(REPO_DIR, repo_name)

    # Skip logic handled externally by processed_urls

    # Clone repo
    if not run_cmd(["git", "clone", "--depth", "1", git_url, repo_path]):
        print(f"⚠️ Failed to clone {git_url}")
        return None

    repo_path_abs = os.path.abspath(repo_path).replace("\\", "/")

    # UNIQUE PROJECT KEY (important)
    project_key = f"{owner}__{repo_name}".lower()

    sonar_cmd = [
        SONAR_SCANNER,
        f"-Dsonar.projectKey={project_key}",
        f"-Dsonar.projectName={owner}/{repo_name}",
        f"-Dsonar.sources={repo_path_abs}",
        "-Dsonar.scm.exclusions.disabled=true",
        "-Dsonar.inclusions=**/*",
        "-Dsonar.exclusions=**/*.java,**/.github/workflows/**,**/.git/**,**/node_modules/**",
        f"-Dsonar.host.url={SONAR_URL}",
        f"-Dsonar.token={SONAR_TOKEN}",
        f"-Dsonar.sourceEncoding=UTF-8"
    ]

    print(f"🔍 Analyzing {repo_name}...")
    run_cmd(sonar_cmd)

    time.sleep(10)  # allow SQ to finalize

    # Fetch metrics
    metrics_str = ",".join(METRICS)
    api_url = f"{SONAR_URL}/api/measures/component"
    params = {"component": project_key, "metricKeys": metrics_str}

    r = requests.get(api_url, auth=(SONAR_TOKEN, ""), params=params)

    if r.status_code != 200:
        print(f"⚠️ Failed to get metrics for {repo_name}: {r.text}")
        result = None
    else:
        data = r.json()
        measures = {
            m["metric"]: m["value"]
            for m in data.get("component", {}).get("measures", [])
        }

        result = {
            "repo_url": git_url,
            "owner": owner,
            "repo": repo_name,
            **measures
        }

        # severity counts
        severity_counts = get_vulnerability_details(project_key, repo_name)
        for sev in ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]:
            result[f"vuln_{sev.lower()}"] = severity_counts.get(sev, 0)

    shutil.rmtree(repo_path, ignore_errors=True)
    print(f"🧹 Deleted local clone of {repo_name}")

    return result

metric_cols = ["bugs", "code_smells", "vulnerabilities", "security_hotspots", "vuln_blocker", "vuln_critical",
                   "vuln_major", "vuln_minor", "vuln_info", "ncloc", "files", "functions"]

def delete_all_sonar_projects():
    """Delete all SonarQube projects using admin-level token."""
    print("🗑️ Fetching all SonarQube projects…")

    r = requests.get(f"{SONAR_URL}/api/projects/search",
                     auth=(SONAR_TOKEN, ""))

    if r.status_code != 200:
        print("❌ Failed to fetch project list:", r.text)
        return

    projects = r.json().get("components", [])
    print(f"📌 Found {len(projects)} projects to delete.")

    for p in projects:
        key = p["key"]
        print(f" → Deleting {key}…")

        r2 = requests.post(f"{SONAR_URL}/api/projects/delete?project={key}",
                           auth=(SONAR_TOKEN, ""))

        if r2.status_code == 204:
            print(f"   ✔ Deleted {key}")
        else:
            print(f"   ❌ Failed: {r2.status_code} — {r2.text}")

    print("✅ SonarQube reset complete.")

def compute_project_key_from_url(repo_url: str) -> str:
    """
    Convert GitHub URL into the SAME Sonar projectKey used during scanning.
    Example:
        https://github.com/NHSDigital/rap-package-template
        → NHSDigital__rap-package-template
    """
    cleaned = repo_url.rstrip("/").replace(".git", "")
    if cleaned.startswith("https://github.com/"):
        cleaned = cleaned.replace("https://github.com/", "")
    owner, repo = cleaned.split("/")[-2], cleaned.split("/")[-1]
    return f"{owner}__{repo}"


def _to_int_or_none(val):
    try:
        if pd.isna(val):
            return None
        s = str(val).strip()
        if s == "":
            return None
        return int(float(s))
    except Exception:
        return None


def repair_incomplete_results(output_csv: str):
    """
    For each row in the CSV:
      - recompute the SonarQube project key from repo_url
      - fetch metrics from SonarQube
      - if CSV row has all metrics empty/zero BUT Sonar has some non-zero,
        update that row with the Sonar values (and refresh vuln_* counts too).
    """
    if not os.path.exists(output_csv):
        print("❌ No output CSV found.")
        return

    df = pd.read_csv(output_csv)
    #metric_cols = METRICS
    severity_cols = [c for c in df.columns if c.startswith("vuln_")]

    print(f"🔧 Repairing CSV based on SonarQube as source of truth…")
    print(f"   Metrics columns: {metric_cols}")
    print(f"   Severity columns: {severity_cols}")

    repaired = 0

    for idx, row in df.iterrows():
        repo_url = row.get("repo_url")
        repo_name = row.get("repo")

        if not isinstance(repo_url, str) or not repo_url.startswith("http"):
            # No usable URL → can't map to projectKey
            continue

        project_key = compute_project_key_from_url(repo_url)

        # Current CSV metrics as ints (or None)
        curr_vals = [_to_int_or_none(row.get(col)) for col in metric_cols]
        curr_all_zero_or_empty = all(v in (None, 0) for v in curr_vals)

        # If row clearly has some non-zero metric already, skip it
        if not curr_all_zero_or_empty:
            continue

        # Ask SonarQube what the metrics are
        metrics_str = ",".join(METRICS)
        api_url = f"{SONAR_URL}/api/measures/component"
        params = {"component": project_key, "metricKeys": metrics_str}

        r = requests.get(api_url, auth=(SONAR_TOKEN, ""), params=params)
        if r.status_code != 200:
            # project might not exist or token privilege issues
            print(f"⚠️ [{repo_name}] Failed to fetch metrics for {project_key}: {r.text}")
            continue

        data = r.json()
        measures = {
            m["metric"]: m["value"]
            for m in data.get("component", {}).get("measures", [])
        }

        if not measures:
            # Sonar really has nothing → then 0s in CSV are correct
            continue

        sonar_nums = [_to_int_or_none(measures.get(col, "0")) for col in metric_cols]
        sonar_has_nonzero = any((v is not None and v != 0) for v in sonar_nums)

        # If Sonar also reports all zeros → nothing to fix
        if not sonar_has_nonzero:
            continue

        # At this point:
        #  - CSV row has all metrics empty/0
        #  - Sonar says there is something non-zero
        print(f"🔁 Fixing row {idx} for {repo_name} ({project_key})")
        print(f"    CSV:   {dict(zip(metric_cols, curr_vals))}")
        print(f"    Sonar: {dict(zip(metric_cols, sonar_nums))}")

        # Update metric columns
        for col in metric_cols:
            df.at[idx, col] = measures.get(col, "0")

        # Also refresh severity breakdown from Sonar issues API
        severity_counts = get_vulnerability_details(project_key, repo_name)
        for sev in ["BLOCKER", "CRITICAL", "MAJOR", "MINOR", "INFO"]:
            col = f"vuln_{sev.lower()}"
            if col in df.columns:
                df.at[idx, col] = severity_counts.get(sev, 0)

        repaired += 1

    df.to_csv(output_csv, index=False)
    print(f"\n💾 Repair complete. Updated {repaired} row(s).")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    programming_language = "JavaScript"
    df = pd.read_csv(GENERATED_BASED_REPOSITORIES + programming_language + "/" + INPUT_CSV)

    df["data_update"] = pd.to_datetime(
        df["data_update"],
        errors="coerce",
        utc=True,
    )

    start_date = pd.Timestamp("2023-10-18", tz="UTC")

    repo_urls = (
        df.loc[
            (df["commit_count"] > 1)
            & (df["is_template"] == False)
            & (df["data_update"] >= start_date),
            "original_link",
        ]
        .dropna()
        .tolist()
    )

    #repo_urls = df["original_link"].dropna().tolist()

    OUTPUT_CSV = "vulnerability_issues/" +programming_language + "/" + OUTPUT_CSV_BASE + programming_language + ".csv"
    # Track processed URLs
    processed_urls = set()
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV, on_bad_lines="skip")
        if "repo_url" in existing.columns:
            processed_urls = set(existing["repo_url"].dropna().tolist())

    print(f"📄 Loaded {len(processed_urls)} processed repos.")

    no_available = []
    # Process each repository
    for i, git_url in enumerate(repo_urls, 1):
        print(f"\n=== ({i}/{len(repo_urls)}) {git_url} ===")

        if git_url in processed_urls:
            print(f"⏩ Skipping (already processed)")
            continue

        info = analyze_repo(git_url)

        if info:
            pd.DataFrame([info]).to_csv(
                OUTPUT_CSV,
                mode="a",
                header=not os.path.exists(OUTPUT_CSV),
                index=False
            )
            print("💾 Saved result.")
        else:
            no_available.append(git_url)

    print(no_available)

if __name__ == "__main__":
    main()