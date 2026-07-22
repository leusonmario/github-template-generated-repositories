import base64

import requests
import time
import csv
from collections import OrderedDict

import config

BASE_URL = "https://api.github.com/search/repositories"
UA = "GH-Sampler/1.0"

# ----- Config -----
PER_PAGE = 100                     # max
MAX_PAGES = 10                     # hard GitHub cap
PAGE_SLEEP = 0.5                   # between pages
QUERY_SLEEP = 0.8                  # between different queries (avoid abuse)
SPLIT_THRESHOLD = 1000             # GitHub returns only first 1000 hits
RETRY = 3                          # network retries
TIMEOUT = 30

from datetime import date, timedelta
from dateutil.relativedelta import relativedelta  # pip install python-dateutil

def cutoff_iso():
    # get commits from the last two years
    cutoff = date.today() - timedelta(days=1) - relativedelta(years=2)
    print(cutoff)
    #return cutoff.isoformat()
    return '2023-10-18'

def gh_get(url, headers, params):
    for attempt in range(RETRY):
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if r.status_code == 200:
            return r
        # Handle secondary rate limits / abuse detection
        if r.status_code in (403, 429):
            reset = r.headers.get("X-RateLimit-Reset")
            wait = 60 if reset is None else max(1, int(reset) - int(time.time()) + 1)
            time.sleep(wait)
            continue
        # transient?
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r  # just in case

def total_count_for_range(base_query, star_min, star_max, headers):
    # Build stars qualifier
    if star_max is None:
        stars_q = f"stars:>={star_min}"
    else:
        stars_q = f"stars:{star_min}..{star_max}"
    q = f"{base_query} {stars_q}".strip()
    params = {"q": q, "per_page": 1}  # only want total_count
    r = gh_get(BASE_URL, headers, params)
    data = r.json()
    return data.get("total_count", 0)

def split_ranges_adaptively(base_query, min_stars, max_stars, headers):
    ranges = []
    stack = [(min_stars, max_stars)]
    while stack:
        lo, hi = stack.pop()
        cnt = total_count_for_range(base_query, lo, hi, headers)
        if cnt <= SPLIT_THRESHOLD:
            ranges.append((lo, hi, cnt))
            time.sleep(QUERY_SLEEP)
            continue

        # Need to split
        if hi is None:
            # open-ended: split by growing the lower bound geometrically
            # pick a pivot by multiplying lo (avoid infinite loop on low lo)
            pivot = max(lo + 1, int(lo * 2))  # crude but effective
            left = (lo, pivot - 1)
            right = (pivot, None)
        else:
            if hi <= lo:
                # degenerate; fall back to accept
                ranges.append((lo, hi, cnt))
                continue
            pivot = (lo + hi) // 2
            left = (lo, pivot)
            right = (pivot + 1, hi)

        # Push right then left so left is processed first (LIFO)
        stack.append(right)
        stack.append(left)
        time.sleep(QUERY_SLEEP)

    # Sort by lo ascending
    ranges.sort(key=lambda x: (x[0], float("inf") if x[1] is None else x[1]))
    return ranges


def save_repo_info_to_csv(rows, filename_base="all_templates", language=""):
    filename_final = f"{filename_base}_{language}.csv"
    fieldnames = [
        "id", "username", "project_name", "original_link_repository",
        "stars", "forks", "created_at", "updated_at", "language", "is_template", "owner_type", "README",
        "repo_description", "topics"
    ]
    # Write header if new
    try:
        with open(filename_final, "x", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
    except FileExistsError:
        pass
    # Append
    with open(filename_final, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerows(rows)

def fetch_readme(repo_full_name):
    """Fetch README content from GitHub API."""
    url = f"https://api.github.com/repos/{repo_full_name}/readme"
    headers = {"Authorization": f"token {config.GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            content = r.json().get("content", "")
            return base64.b64decode(content).decode("utf-8", errors="ignore")
        else:
            return None
    except Exception as e:
        print(f"⚠️ Error fetching {repo_full_name}: {e}")
        return None

def fetch_range(base_query, star_min, star_max, headers, language):
    """
    Fetch repositories from GitHub based on a query and star range,
    including topics for each repository.
    """

    if star_max is None:
        stars_q = f"stars:>={star_min}"
    else:
        stars_q = f"stars:{star_min}..{star_max}"
    q = f"{base_query} {stars_q}".strip()

    seen_ids = set()
    out_rows = []       # only valid template repos
    all_rows = []       # all repos (at least name info)

    for page in range(1, MAX_PAGES + 1):
        params = {
            "q": q,
            "sort": "stars",
            "order": "desc",
            "per_page": PER_PAGE,
            "page": page
        }
        # --- Important: Include 'application/vnd.github.mercy-preview+json' for topics ---
        # (still needed for topics API field in search results)
        r = gh_get(BASE_URL, headers, params)
        data = r.json()
        items = data.get("items", []) or []
        if not items:
            break

        for repo in items:
            rid = repo.get("id")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)

            full_name = repo.get("full_name", "")
            username, project_name = (full_name.split("/", 1) + [""])[:2]

            # --- Fetch topics separately (most reliable way) ---
            topics = fetch_topics(full_name, headers)

            # --- always store repo basic info ---
            all_rows.append({
                "id": rid,
                "username": username,
                "project_name": project_name,
                "original_link_repository": repo.get("html_url"),
                "stars": repo.get("stargazers_count", 0),
                "forks": repo.get("forks_count", 0),
                "created_at": repo.get("created_at"),
                "updated_at": repo.get("updated_at"),
                "language": language,
                "is_template": repo.get("is_template", False),
                "owner_type": (repo.get("owner") or {}).get("type", ""),
                "topics": ", ".join(topics) if topics else "",
                "README": fetch_readme(full_name),
                "repo_description": repo.get("description") or "",
            })

            # --- only keep actual templates for detailed CSV ---
            # if repo.get("is_template"):
            #     out_rows.append(all_rows[-1])

        if len(items) < PER_PAGE:
            break
        time.sleep(PAGE_SLEEP)

    # Save *all* repos (including non-templates)
    if all_rows:
        save_repo_info_to_csv(all_rows, filename_base="all_templates__new_search", language=language)

    return out_rows


def fetch_topics(full_name, headers):
    """
    Fetch topics for a given repository using the GitHub API.
    """
    url = f"https://api.github.com/repos/{full_name}/topics"
    # topics require a special Accept header
    headers_with_preview = headers.copy()
    headers_with_preview["Accept"] = "application/vnd.github.mercy-preview+json"

    r = requests.get(url, headers=headers_with_preview)
    if r.status_code == 200:
        data = r.json()
        return data.get("names", [])
    else:
        print(f"Warning: could not fetch topics for {full_name} (status {r.status_code})")
        return []

def mine_language(language, token, filename_base, min_stars=0, max_stars=None):
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": UA,
    }

    cutoff = cutoff_iso()

    # Base query: narrow to language, template-ish keywords, public, non-archived, non-fork.
    # Tip: include readme in text search; it helps surface templates.
    base_query = (
        f"language:{language} "
        f"fork:false archived:false is:public "
        f"pushed:>={cutoff} "
        f"template:true"
    )

    print(base_query)

    print(f"\n=== {language} ===")
    print("Building adaptive star ranges...")
    ranges = split_ranges_adaptively(base_query, min_stars, max_stars, headers)
    for lo, hi, cnt in ranges:
        print(f"  range {lo}..{('∞' if hi is None else hi)} => ~{cnt} repos")

    # Mine each range
    global_seen = set()
    for lo, hi, _ in ranges:
        rows = fetch_range(base_query, lo, hi, headers, language)
        # Dedup across ranges via id
        new_rows = []
        for r in rows:
            if r["id"] in global_seen:
                continue
            global_seen.add(r["id"])
            new_rows.append(r)
        if new_rows:
            save_repo_info_to_csv(new_rows, filename_base, language)
            print(f"Saved {len(new_rows)} rows for range {lo}..{('∞' if hi is None else hi)}")
        time.sleep(QUERY_SLEEP)

def main():
    # ---------- YOUR SETTINGS ----------
    import config  # expects GITHUB_TOKEN
    filename_base = "all_templates"
    languages = ["TypeScript", "Python", "JavaScript", "Java", "C#"]
    min_stars = 2
    max_stars = None  # None = open ended (>= min_stars)
    # -----------------------------------

    for lang in languages:
        mine_language(
            language=lang,
            token=config.GITHUB_TOKEN,
            filename_base=filename_base,
            min_stars=min_stars,
            max_stars=max_stars
        )

if __name__ == "__main__":
    main()
