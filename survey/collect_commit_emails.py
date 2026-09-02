#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


REQUIRED_COLUMNS = {"generated_repository", "language", "owner_type"}


def log(msg: str) -> None:
    print(msg, flush=True)


def git_env() -> dict:
    """
    Disable all interactive Git credential/password prompts.

    This prevents private/deleted/inaccessible repositories from blocking
    the script waiting for a username/password.
    """
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    env["GIT_ASKPASS"] = "echo"
    env["SSH_ASKPASS"] = "echo"
    return env


def run(
    cmd: List[str],
    check: bool = True,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        env=git_env(),
        timeout=timeout,
    )


def normalize_repo(value: str) -> Optional[str]:
    """Normalize common GitHub repository forms to owner/repo."""
    if value is None:
        return None

    s = str(value).strip().strip(" <>\"'")
    s = s.rstrip(".,;:")

    if not s:
        return None

    if s.startswith("git@github.com:"):
        s = s[len("git@github.com:"):]
    elif s.startswith("ssh://git@github.com/"):
        s = s[len("ssh://git@github.com/"):]
    elif s.startswith("http://") or s.startswith("https://"):
        parsed = urlparse(s)
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            return None
        s = parsed.path.lstrip("/")

    s = s.split("#", 1)[0].split("?", 1)[0].strip("/")

    if s.endswith(".git"):
        s = s[:-4]

    parts = s.split("/")
    if len(parts) < 2:
        return None

    owner = parts[0].strip()
    repo = parts[1].strip().rstrip(".,;:")

    if not owner or not repo:
        return None

    return f"{owner}/{repo}"


def repo_dir_name(repo_slug: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", repo_slug)
    digest = hashlib.sha1(repo_slug.encode("utf-8")).hexdigest()[:8]
    return f"{safe}__{digest}.git"


def clean_git_error(stderr: str) -> str:
    """
    Compact Git errors into one CSV-safe line.
    """
    msg = " ".join((stderr or "").strip().split())
    return msg[:2000]


def classify_clone_error(stderr: str) -> str:
    """
    Give a simple reason useful for later inspection.
    """
    s = (stderr or "").lower()

    if (
        "repository not found" in s
        or "could not read username" in s
        or "authentication failed" in s
        or "terminal prompts disabled" in s
        or "permission denied" in s
    ):
        return "inaccessible_or_private"

    if "could not resolve host" in s or "failed to connect" in s:
        return "network_error"

    if "remote: not found" in s:
        return "repository_not_found"

    return "clone_failed"


def clone_repo(
    repo_slug: str,
    repos_dir: Path,
    clone_timeout: int,
) -> Path:
    """
    Bare partial clone: enough for commit metadata, without a working tree.

    Interactive authentication is disabled. If a repository is private,
    deleted, renamed, or otherwise inaccessible, this function fails
    immediately instead of asking for credentials.
    """
    repos_dir.mkdir(parents=True, exist_ok=True)

    target = repos_dir / repo_dir_name(repo_slug)
    clone_url = f"https://github.com/{repo_slug}.git"

    if target.exists():
        shutil.rmtree(target, ignore_errors=True)

    try:
        cp = run(
            [
                "git",
                "clone",
                "--bare",
                "--filter=blob:none",
                clone_url,
                str(target),
            ],
            check=False,
            timeout=clone_timeout,
        )
    except subprocess.TimeoutExpired:
        shutil.rmtree(target, ignore_errors=True)
        raise RuntimeError(f"clone_timeout_after_{clone_timeout}_seconds")

    if cp.returncode != 0:
        shutil.rmtree(target, ignore_errors=True)
        reason = classify_clone_error(cp.stderr)
        detail = clean_git_error(cp.stderr)
        raise RuntimeError(f"{reason}: {detail}")

    return target


def canonical_email(email: str) -> str:
    return email.strip().lower()


def valid_email(email: str) -> bool:
    if not email or "@" not in email:
        return False
    local, _, domain = email.rpartition("@")
    return bool(local and domain and "." in domain)


def read_input(input_csv: Path) -> Dict[str, Tuple[str, str]]:
    """
    Return:
        normalized_repository -> (language, owner_type)

    Fail if a normalized repository has conflicting metadata.
    """
    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise SystemExit("Input CSV has no header.")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise SystemExit(
                f"Missing required columns: {', '.join(sorted(missing))}"
            )

        tmp = {}

        for row in reader:
            repo = normalize_repo(row.get("generated_repository", ""))
            if not repo:
                continue

            language = (row.get("language") or "").strip()
            owner_type = (row.get("owner_type") or "").strip()

            if repo not in tmp:
                tmp[repo] = {
                    "languages": set(),
                    "owner_types": set(),
                }

            if language:
                tmp[repo]["languages"].add(language)

            if owner_type:
                tmp[repo]["owner_types"].add(owner_type)

    result = {}

    for repo, meta in tmp.items():
        if len(meta["languages"]) > 1:
            raise SystemExit(
                f"Conflicting language values for {repo}: "
                f"{sorted(meta['languages'])}"
            )

        if len(meta["owner_types"]) > 1:
            raise SystemExit(
                f"Conflicting owner_type values for {repo}: "
                f"{sorted(meta['owner_types'])}"
            )

        language = next(iter(meta["languages"]), "")
        owner_type = next(iter(meta["owner_types"]), "")

        result[repo] = (language, owner_type)

    return result


def collect_author_emails(repo_path: Path) -> Dict[str, int]:
    """
    Parse all commits reachable from all refs and count commits per author email.

    Returns:
        email -> number_of_commits
    """
    counts = defaultdict(int)

    proc = subprocess.Popen(
        [
            "git",
            "-C",
            str(repo_path),
            "log",
            "--all",
            "--format=%ae",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=git_env(),
    )

    assert proc.stdout is not None

    for line in proc.stdout:
        email = canonical_email(line)
        if valid_email(email):
            counts[email] += 1

    stderr = proc.stderr.read() if proc.stderr else ""
    rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"git_log_failed: {clean_git_error(stderr)}")

    return dict(counts)


def write_main_output(
    output_csv: Path,
    rows: List[Dict[str, str]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "repository",
        "language",
        "owner_type",
        "email",
        "number_of_commits",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_failures(
    failures_csv: Path,
    rows: List[Dict[str, str]],
) -> None:
    failures_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "repository",
        "language",
        "owner_type",
        "error",
    ]

    with failures_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Clone GitHub repositories, collect commit-author emails, "
            "and report inaccessible repositories separately."
        )
    )

    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input RQ3 CSV.",
    )

    p.add_argument(
        "--output",
        type=Path,
        default=Path("practitioner_emails.csv"),
        help="Main output CSV.",
    )

    p.add_argument(
        "--failures",
        type=Path,
        default=Path("failed_repositories.csv"),
        help="CSV containing repositories that could not be processed.",
    )

    p.add_argument(
        "--repos-dir",
        type=Path,
        default=Path("cloned_repositories"),
        help="Temporary directory used for repository clones.",
    )

    p.add_argument(
        "--clone-timeout",
        type=int,
        default=120,
        help="Maximum seconds allowed for one clone attempt. Default: 120.",
    )

    p.add_argument(
        "--keep-repositories",
        action="store_true",
        help=(
            "Keep repositories after processing. "
            "By default each repository is deleted immediately after analysis."
        ),
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        log(run(["git", "--version"]).stdout.strip())
    except Exception:
        raise SystemExit("Git is required and must be available in PATH.")

    repo_metadata = read_input(args.input)

    log(f"[INFO] Unique repositories to process: {len(repo_metadata):,}")
    log("[INFO] Interactive Git username/password prompts are disabled.")

    output_rows: List[Dict[str, str]] = []
    failure_rows: List[Dict[str, str]] = []

    for i, (repo, (language, owner_type)) in enumerate(
        sorted(repo_metadata.items()),
        start=1,
    ):
        log(f"[{i}/{len(repo_metadata)}] {repo}")

        repo_path = None

        try:
            repo_path = clone_repo(
                repo,
                args.repos_dir,
                clone_timeout=args.clone_timeout,
            )

            email_counts = collect_author_emails(repo_path)

            for email, count in sorted(email_counts.items()):
                output_rows.append(
                    {
                        "repository": repo,
                        "language": language,
                        "owner_type": owner_type,
                        "email": email,
                        "number_of_commits": count,
                    }
                )

            log(f"  -> SUCCESS: {len(email_counts):,} unique author emails")

        except Exception as exc:
            error = str(exc)
            log(f"  -> FAILED: {error}")

            failure_rows.append(
                {
                    "repository": repo,
                    "language": language,
                    "owner_type": owner_type,
                    "error": error,
                }
            )

        finally:
            # Checkpoint both files after every repository.
            write_main_output(args.output, output_rows)
            write_failures(args.failures, failure_rows)

            # Default: delete repository after analysis.
            if (
                not args.keep_repositories
                and repo_path is not None
                and repo_path.exists()
            ):
                shutil.rmtree(repo_path, ignore_errors=True)

    unique_emails = len({row["email"] for row in output_rows})

    log("")
    log("[DONE]")
    log(f"[DONE] Successful email/repository rows: {len(output_rows):,}")
    log(f"[DONE] Unique emails overall: {unique_emails:,}")
    log(f"[DONE] Failed repositories: {len(failure_rows):,}")
    log(f"[DONE] Main output: {args.output.resolve()}")
    log(f"[DONE] Failures: {args.failures.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
