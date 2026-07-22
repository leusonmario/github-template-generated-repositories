import csv
import time
import random
import requests
import config
from datetime import datetime, timedelta, timezone
import sys

csv.field_size_limit(sys.maxsize)

max_int = sys.maxsize

while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

BASE_URL = "https://api.github.com"

HEADERS = {
    "Authorization": f"Bearer {config.GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

session = requests.Session()
session.headers.update(HEADERS)

repo_cache = {}

# Pauses to reduce GitHub secondary rate limits
REPO_REQUEST_PAUSE = 0.8
SEARCH_PAGE_PAUSE = 2.0
TEMPLATE_PAUSE = 10.0

# GitHub Search exposes at most 1,000 results per query
GITHUB_SEARCH_LIMIT = 1000
PER_PAGE = 100
MAX_PAGES = 10


def is_secondary_rate_limit(response):
    if response.status_code not in (403, 429):
        return False

    try:
        message = response.json().get("message", "").lower()
    except Exception:
        message = response.text.lower()

    return (
        "secondary rate limit" in message
        or "abuse detection" in message
        or "you have exceeded a secondary rate limit" in message
    )


def compute_wait_time(response, attempt):
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        return int(retry_after) + 5

    remaining = response.headers.get("X-RateLimit-Remaining")
    reset = response.headers.get("X-RateLimit-Reset")

    if remaining == "0" and reset:
        reset_time = int(reset)
        return max(reset_time - int(time.time()), 0) + 10

    # Secondary rate limit fallback.
    # Wait at least 1 minute and increase gradually.
    base = min(60 * (2 ** attempt), 15 * 60)
    jitter = random.randint(10, 45)
    return base + jitter


def safe_get(url, params=None, max_retries=8):
    for attempt in range(max_retries):
        response = session.get(url, params=params)

        if response.status_code == 200:
            return response

        if is_secondary_rate_limit(response):
            wait = compute_wait_time(response, attempt)
            print(
                f"Secondary rate limit detected. "
                f"Waiting {wait} seconds before retrying the same request."
            )
            time.sleep(wait)
            continue

        if response.status_code in (500, 502, 503, 504):
            wait = min(30 * (2 ** attempt), 5 * 60)
            print(f"GitHub server error {response.status_code}. Waiting {wait} seconds.")
            time.sleep(wait)
            continue

        return response

    raise RuntimeError(f"Request failed after {max_retries} retries: {url}")


def get_repo_details(full_name):
    if full_name in repo_cache:
        return repo_cache[full_name]

    url = f"{BASE_URL}/repos/{full_name}"

    try:
        response = safe_get(url)
    except RuntimeError as e:
        print(f"Failed permanently to get {full_name}: {e}")
        return None

    if response.status_code != 200:
        print(f"Failed to get {full_name}: {response.status_code} - {response.text}")
        return None

    data = response.json()
    repo_cache[full_name] = data
    return data


def append_rows(rows, filename):
    if not rows:
        return

    fieldnames = [
        "username",
        "project_name",
        "original_link",
        "stars",
        "forks",
        "commits",
        "data_creation",
        "data_update",
        "is_template",
        "target_template_repository",
    ]

    try:
        with open(filename, "x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(rows)


def append_failed_template(filename, language, target_template, query, reason):
    fieldnames = ["language", "target_template", "query", "reason"]

    try:
        with open(filename, "x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        pass

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "language": language,
            "target_template": target_template,
            "query": query,
            "reason": reason,
        })


def normalize_date(date_value):
    """
    Accepts values such as:
    - 2020-01-01
    - 2020-01-01T12:34:56Z
    - empty/None

    Returns YYYY-MM-DD or None.
    """
    if not date_value:
        return None

    return str(date_value)[:10]


def today_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def split_date_range(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    midpoint = start + (end - start) / 2

    left_start = start.strftime("%Y-%m-%d")
    left_end = midpoint.strftime("%Y-%m-%d")

    right_start = (midpoint + timedelta(days=1)).strftime("%Y-%m-%d")
    right_end = end.strftime("%Y-%m-%d")

    return (left_start, left_end), (right_start, right_end)


def build_query(language, target_template, start_date=None, end_date=None):
    query_parts = [
        f"language:{language}",
        "in:name,description,readme",
        "fork:false",
        f"\"{target_template}\"",
    ]

    if start_date and end_date:
        query_parts.append(f"created:{start_date}..{end_date}")
    elif start_date:
        query_parts.append(f"created:>={start_date}")

    return " ".join(query_parts)


def get_search_total_count(query):
    """
    Gets only the first page to inspect total_count.
    """
    params = {
        "q": query,
        "sort": "updated",
        "order": "desc",
        "per_page": 1,
        "page": 1,
    }

    try:
        response = safe_get(f"{BASE_URL}/search/repositories", params=params)
    except RuntimeError as e:
        print(f"Could not get total_count for query: {query}")
        print(e)
        return None, False

    if response.status_code != 200:
        print(f"Count query failed: {response.status_code} - {response.text}")
        return None, False

    data = response.json()
    return data.get("total_count", 0), True


def fetch_query_results(query, max_pages=MAX_PAGES, per_page=PER_PAGE):
    """
    Fetches up to the accessible GitHub Search window for a query.
    This should be used only when total_count <= 1000, except as a fallback.
    """
    repos_by_full_name = {}

    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        }

        try:
            response = safe_get(f"{BASE_URL}/search/repositories", params=params)
        except RuntimeError as e:
            print(f"Search failed permanently on page {page}: {e}")
            return repos_by_full_name, False

        if response.status_code != 200:
            print(f"Search failed: {response.status_code} - {response.text}")
            return repos_by_full_name, False

        data = response.json()
        repos = data.get("items", [])

        if not repos:
            break

        for repo in repos:
            repos_by_full_name[repo["full_name"]] = repo

        time.sleep(SEARCH_PAGE_PAUSE)

    return repos_by_full_name, True


def collect_candidates_with_optional_date_split(
    language,
    target_template,
    start_date=None,
    end_date=None,
    depth=0,
    max_depth=20,
):
    """
    First checks total_count.

    If total_count <= 1000:
        fetches normally with pagination.

    If total_count > 1000:
        recursively splits by created date range and fetches each subquery.

    Returns:
        repos_by_full_name, completed, representative_query
    """
    query = build_query(
        language=language,
        target_template=target_template,
        start_date=start_date,
        end_date=end_date,
    )

    indent = "  " * depth

    total_count, ok = get_search_total_count(query)
    if not ok:
        return {}, False, query

    print(f"{indent}Candidate search returned total_count={total_count} for query: {query}")

    if total_count == 0:
        return {}, True, query

    if total_count <= GITHUB_SEARCH_LIMIT:
        repos, completed = fetch_query_results(query)
        return repos, completed, query

    print(
        f"{indent}total_count={total_count} is higher than {GITHUB_SEARCH_LIMIT}. "
        f"Splitting by creation date."
    )

    if not start_date:
        start_date = "2008-01-01"

    if not end_date:
        end_date = today_utc()

    if start_date >= end_date:
        print(
            f"{indent}Cannot split further because start_date >= end_date "
            f"({start_date}..{end_date}). Fetching first 1,000 only."
        )
        repos, completed = fetch_query_results(query)
        return repos, False, query

    if depth >= max_depth:
        print(
            f"{indent}Max split depth reached for {query}. "
            f"Fetching first 1,000 only."
        )
        repos, completed = fetch_query_results(query)
        return repos, False, query

    (left_start, left_end), (right_start, right_end) = split_date_range(start_date, end_date)

    all_repos = {}
    completed_all = True

    left_repos, left_completed, _ = collect_candidates_with_optional_date_split(
        language=language,
        target_template=target_template,
        start_date=left_start,
        end_date=left_end,
        depth=depth + 1,
        max_depth=max_depth,
    )
    all_repos.update(left_repos)
    completed_all = completed_all and left_completed

    right_repos, right_completed, _ = collect_candidates_with_optional_date_split(
        language=language,
        target_template=target_template,
        start_date=right_start,
        end_date=right_end,
        depth=depth + 1,
        max_depth=max_depth,
    )
    all_repos.update(right_repos)
    completed_all = completed_all and right_completed

    return all_repos, completed_all, query


def mine_generated_repos_for_template(target_template, language, template_created_at=None):
    """
    Candidate discovery:
    - occurrence of target_template in name, description, or README
    - same language
    - excludes forks
    - only repositories created after the template creation date

    If candidate total_count > 1000:
    - recursively splits by created date range

    Final confirmation:
    - template_repository.full_name == target_template
    """
    start_date = normalize_date(template_created_at)

    candidates_by_full_name, completed, query = collect_candidates_with_optional_date_split(
        language=language,
        target_template=target_template,
        start_date=start_date,
        end_date=None,
    )

    print(f"Total accessible candidate repositories after splitting/deduplication: {len(candidates_by_full_name)}")

    selected = []

    for full_name, repo in candidates_by_full_name.items():
        repo_data = get_repo_details(full_name)
        if not repo_data:
            time.sleep(REPO_REQUEST_PAUSE)
            continue

        template_info = repo_data.get("template_repository")

        if template_info and template_info.get("full_name", "").lower() == target_template.lower():
            username, project_name = full_name.split("/", 1)

            selected.append({
                "username": username,
                "project_name": project_name,
                "original_link": repo["html_url"],
                "stars": repo["stargazers_count"],
                "forks": repo["forks_count"],
                "commits": None,
                "data_creation": repo["created_at"],
                "data_update": repo["updated_at"],
                "is_template": repo.get("is_template"),
                "target_template_repository": template_info["full_name"],
            })

            print(f"Confirmed: {full_name} generated from {target_template}")

        time.sleep(REPO_REQUEST_PAUSE)

    return selected, completed, query

def parse_stars(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0

if __name__ == "__main__":
    language = "C#"
    start_line = 0  # MeteorDevelopment/meteor-addon-template

    input_file = f"../data_collection/all_templates_{language}.csv"
    output_file = f"output/generated_GHRep_from_templates_{language}.csv"
    failed_file = f"output/failed_generated_GHRep_from_templates_{language}.csv"

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if i < start_line or parse_stars(row.get("stars")) < 10:
                continue

            target_template = f"{row['username']}/{row['project_name']}"

            print(f"\nSearching generated repositories from: {target_template}")

            template_created_at = row.get("data_creation") or row.get("created_at")

            rows, completed, query = mine_generated_repos_for_template(
                target_template=target_template,
                language=language,
                template_created_at=template_created_at,
            )

            append_rows(rows, output_file)

            if not completed:
                append_failed_template(
                    filename=failed_file,
                    language=language,
                    target_template=target_template,
                    query=query,
                    reason="Search did not complete fully. Some query/date range may still exceed GitHub's 1,000-result limit.",
                )
                print(f"Marked as incomplete: {target_template}")

            time.sleep(TEMPLATE_PAUSE)