#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import ipaddress
import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


REQUIRED_COLUMNS = {
    "repository",
    "language",
    "owner_type",
    "email",
    "number_of_commits",
}


# ============================================================
# Known bot / automation / service identities
# ============================================================

KNOWN_AUTOMATION_KEYWORDS = [
    "dependabot",
    "dependabot-preview",
    "renovate",
    "renovatebot",
    "greenkeeper",
    "snyk",
    "whitesource",
    "github-actions",
    "github_actions",
    "githubactions",
    "actions-user",
    "actions-template-sync",
    "web-flow",
    "bors",
    "mergify",
    "codecov",
    "coveralls",
    "travis",
    "circleci",
    "appveyor",
    "azure-pipelines",
    "semantic-release",
    "semantic_release",
    "release-please",
    "release_please",
    "allcontributors",
    "all-contributors",
    "stale",
    "pre-commit-ci",
    "pre_commit_ci",
    "imgbot",
    "weblate",
    "gitee",
    "octopus-deploy",
    "octopusdeploy",
    "deepsource",
]

EXACT_AUTOMATION_EMAILS = {
    "action@github.com",
    "actions@github.com",
    "github-actions@github.com",
    "github-actions[bot]@github.com",
    "support+actions@github.com",
    "copilot@github.com",
    "cursoragent@cursor.com",
    "bot@deepsource.io",
    "snyk-bot@snyk.io",
    "semantic-release-bot@martynus.net",
    "twinkle_bot@github.com",
    "support@codefactor.io",
    "imgbothelp@gmail.com",
}

BOT_TOKEN_RE = re.compile(
    r"(?:\[bot\]|(?:^|[._+\-])bot(?:$|[._+\-]))",
    flags=re.IGNORECASE,
)

NOREPLY_LOCAL_RE = re.compile(
    r"^(?:no[-_.]?reply|noreply|do[-_.]?not[-_.]?reply|donotreply|mailer[-_.]?daemon)(?:\+.*)?$",
    flags=re.IGNORECASE,
)


# ============================================================
# Invalid / uncontactable domains
# ============================================================

PLACEHOLDER_DOMAINS = {
    "example.com",
    "example.org",
    "example.net",
    "example.edu",
    "test.com",
    "test.org",
    "invalid",
    "localhost",
}

LOCAL_SUFFIXES = (
    ".local",
    ".lan",
    ".home",
    ".internal",
    ".localdomain",
    ".guest",
    ".invalid",
    ".test",
    ".localhost",
)

NOREPLY_DOMAIN_PATTERNS = (
    "users.noreply.github.com",
    "user.noreply.github.com",
    "noreply.users.github.com",
    "users.noreply.gitlab.com",
    "user.noreply.gitee.com",
    "users.noreply.hosted.weblate.org",
)


# ============================================================
# Manual-review categories
# ============================================================

GENERIC_ROLE_LOCALS = {
    "admin",
    "administrator",
    "contact",
    "dev",
    "developer",
    "developers",
    "engineering",
    "hello",
    "help",
    "info",
    "mail",
    "office",
    "root",
    "security",
    "service",
    "services",
    "support",
    "team",
    "webmaster",
}

SUSPICIOUS_TYPO_DOMAINS = {
    "gamil.com",
    "gmial.com",
    "gmaol.com",
    "gmamil.com",
    "gmai.com",
    "gmal.com",
    "gmail.co",
    "gmail.con",
    "gmail.om",
    "gmail.cm",
    "outlok.com",
    "outllok.com",
    "outlook.co",
    "hotnail.com",
    "hotmai.com",
    "hotmail.co",
    "yaho.com",
    "yahoo.co",
}

AMBIGUOUS_AUTOMATION_WORDS = (
    "robot",
    "automation",
)


# ============================================================
# Email validation
# ============================================================

EMAIL_RE = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
    r"[A-Z]{2,63}$",
    flags=re.IGNORECASE,
)

DOMAIN_LABEL_RE = re.compile(
    r"^[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?$",
    flags=re.IGNORECASE,
)


def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_email(value) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip().lower()


def split_email(email: str) -> Tuple[str, str]:
    local, sep, domain = email.rpartition("@")
    if not sep:
        return "", ""
    return local, domain.rstrip(".")


def contains_control_or_non_ascii(email: str) -> bool:
    for ch in email:
        if unicodedata.category(ch).startswith("C"):
            return True
        if ord(ch) > 127:
            return True
    return False


def is_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value.strip("[]"))
        return True
    except ValueError:
        return False


def is_local_or_private_host(domain: str) -> bool:
    d = domain.lower().rstrip(".")

    if is_ip_address(d):
        return True

    if d in {"localhost", "localhost.localdomain"}:
        return True

    if any(d.endswith(suffix) for suffix in LOCAL_SUFFIXES):
        return True

    if d.startswith("ip-") and (
        ".internal" in d
        or ".compute.internal" in d
        or ".ec2.internal" in d
    ):
        return True

    return False


def valid_email_syntax(email: str) -> bool:
    if not EMAIL_RE.fullmatch(email):
        return False

    local, domain = split_email(email)

    if not local or not domain:
        return False

    if len(email) > 254 or len(local) > 64:
        return False

    if local.startswith(".") or local.endswith(".") or ".." in local:
        return False

    if any(
        not DOMAIN_LABEL_RE.fullmatch(label)
        for label in domain.split(".")
    ):
        return False

    return True


# ============================================================
# Classification
# ============================================================

def classify_email(email: str) -> Tuple[str, Optional[str]]:
    if not email:
        return "remove", "empty_email"

    if contains_control_or_non_ascii(email):
        return "remove", "malformed_or_non_ascii_email"

    if not valid_email_syntax(email):
        return "remove", "invalid_email_syntax"

    local, domain = split_email(email)
    local = local.lower()
    domain = domain.lower()
    base_local = local.split("+", 1)[0]

    if is_local_or_private_host(domain):
        return "remove", "local_or_nonpublic_host"

    if domain in PLACEHOLDER_DOMAINS:
        return "remove", "placeholder_or_test_address"

    if (
        NOREPLY_LOCAL_RE.fullmatch(local)
        or "noreply" in domain
        or "no-reply" in domain
        or "no_reply" in domain
        or any(domain.endswith(p) for p in NOREPLY_DOMAIN_PATTERNS)
    ):
        return "remove", "noreply_address"

    if email in EXACT_AUTOMATION_EMAILS:
        return "remove", "known_bot_or_automation"

    if domain == "github.com":
        return "remove", "known_service_address"

    if BOT_TOKEN_RE.search(local):
        return "remove", "known_bot_or_automation"

    if any(keyword in local for keyword in KNOWN_AUTOMATION_KEYWORDS):
        return "remove", "known_bot_or_automation"

    if domain in SUSPICIOUS_TYPO_DOMAINS:
        return "review", "possible_mistyped_domain"

    if base_local in GENERIC_ROLE_LOCALS:
        return "review", "generic_or_role_account"

    if any(word in local for word in AMBIGUOUS_AUTOMATION_WORDS):
        return "review", "possible_automation_identity"

    return "keep", None


# ============================================================
# Global deduplication by EMAIL ONLY
# ============================================================

def join_unique(values) -> str:
    items = sorted({
        str(v).strip()
        for v in values
        if not pd.isna(v) and str(v).strip()
    })
    return ";".join(items)


def sum_commits(values) -> int:
    vals = pd.to_numeric(values, errors="coerce").fillna(0)
    return int(vals.sum())


def aggregate_by_email(df: pd.DataFrame, include_reason: bool = False) -> pd.DataFrame:
    """
    Produce exactly ONE ROW PER UNIQUE EMAIL across the entire dataset.

    Repository, language, and owner_type values are combined with semicolons.
    number_of_commits is summed across all repositories associated with the email.
    """
    if df.empty:
        cols = [
            "email",
            "repository",
            "language",
            "owner_type",
            "number_of_commits",
        ]
        if include_reason:
            cols.append("reason")
        return pd.DataFrame(columns=cols)

    agg = {
        "repository": join_unique,
        "language": join_unique,
        "owner_type": join_unique,
        "number_of_commits": sum_commits,
    }

    if include_reason:
        agg["reason"] = join_unique

    return (
        df
        .groupby("email", as_index=False, dropna=False)
        .agg(agg)
    )


# ============================================================
# Cleaning
# ============================================================

def clean_practitioner_emails(input_file: Path) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    df = pd.read_csv(input_file)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    print(f"Original rows: {len(df):,}")

    df["repository"] = df["repository"].apply(normalize_text)
    df["language"] = df["language"].apply(normalize_text)
    df["owner_type"] = df["owner_type"].apply(normalize_text)
    df["email"] = df["email"].apply(normalize_email)
    df["number_of_commits"] = pd.to_numeric(
        df["number_of_commits"],
        errors="coerce"
    ).fillna(0).astype(int)

    decisions = df["email"].apply(classify_email)
    df["decision"] = decisions.apply(lambda x: x[0])
    df["reason"] = decisions.apply(lambda x: x[1] or "")

    cleaned_raw = df[df["decision"] == "keep"].copy()
    removed_raw = df[df["decision"] == "remove"].copy()
    review_raw = df[df["decision"] == "review"].copy()

    # IMPORTANT: group ONLY by email, globally.
    cleaned = aggregate_by_email(cleaned_raw)
    removed = aggregate_by_email(removed_raw, include_reason=True)
    review = aggregate_by_email(review_raw, include_reason=True)

    cleaned = cleaned.sort_values("email")
    removed = removed.sort_values("email")
    review = review.sort_values("email")

    output_dir = input_file.parent

    cleaned_file = output_dir / "cleaned_practitioner_emails.csv"
    removed_file = output_dir / "removed_practitioner_emails.csv"
    review_file = output_dir / "manual_review_practitioner_emails.csv"

    cleaned.to_csv(cleaned_file, index=False, quoting=csv.QUOTE_MINIMAL)
    removed.to_csv(removed_file, index=False, quoting=csv.QUOTE_MINIMAL)
    review.to_csv(review_file, index=False, quoting=csv.QUOTE_MINIMAL)

    print()
    print("[DONE]")
    print(f"Input rows: {len(df):,}")
    print(f"Input unique emails: {df['email'].nunique():,}")
    print(f"Cleaned rows / unique emails: {len(cleaned):,}")
    print(f"Removed rows / unique emails: {len(removed):,}")
    print(f"Manual-review rows / unique emails: {len(review):,}")

    # Sanity check: cleaned email must be globally unique.
    if cleaned["email"].duplicated().any():
        raise RuntimeError("Unexpected duplicate email in cleaned output.")

    print("\nSanity check: cleaned output contains one row per unique email.")

    if not removed.empty:
        print("\nRemoved by reason:")
        reason_counts = (
            removed_raw["reason"]
            .value_counts()
        )
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count:,}")

    if not review.empty:
        print("\nManual review by reason:")
        reason_counts = (
            review_raw["reason"]
            .value_counts()
        )
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count:,}")

    print(f"\nSaved cleaned file to: {cleaned_file}")
    print(f"Saved removed file to: {removed_file}")
    print(f"Saved manual-review file to: {review_file}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean practitioner_emails.csv and globally deduplicate by email."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Original practitioner_emails.csv",
    )

    args = parser.parse_args()
    clean_practitioner_emails(args.input.resolve())


if __name__ == "__main__":
    main()
