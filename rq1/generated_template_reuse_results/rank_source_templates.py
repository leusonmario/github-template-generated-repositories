#!/usr/bin/env python3
"""Rank source templates by number of generated repositories that are templates.

Input: generated_repositories_marked_as_templates.csv produced by
       analyze_generated_templates.py.
Only rows with is_template=True and valid positive integer commit_count are used.
One generated repository is counted once. Original data is not modified.
"""

import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


def normalize_repo(value):
    """Normalize owner/repo, GitHub HTTPS URLs, and git@github.com URLs."""
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.casefold() in {'nan', 'none', 'null'}:
        return None
    if value.startswith('git@github.com:'):
        value = value[len('git@github.com:'):]
    elif value.lower().startswith(('http://', 'https://')):
        parsed = urlparse(value)
        if (parsed.hostname or '').casefold() not in {'github.com', 'www.github.com'}:
            return None
        value = parsed.path
    elif value.lower().startswith(('github.com/', 'www.github.com/')):
        value = value.split('/', 1)[1]
    parts = value.split('?', 1)[0].split('#', 1)[0].strip('/').split('/')
    if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
        return None
    owner, repo = parts[0].strip(), parts[1].strip()
    if repo.lower().endswith('.git'):
        repo = repo[:-4]
    return f'{owner}/{repo}'.casefold() if repo else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', type=Path,
                        help='Output generated_repositories_marked_as_templates.csv')
    parser.add_argument('--output_dir', type=Path,
                        default=Path('generated_template_source_ranking'))
    parser.add_argument('--top', type=int, default=10,
                        help='Number of source templates to show in terminal (default: 10)')
    args = parser.parse_args()
    if args.top < 1:
        parser.error('--top must be >= 1')

    df = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    required = {'is_template', 'commit_count'}
    missing = required - set(df.columns)
    if missing:
        parser.error(f'Missing required columns: {sorted(missing)}')

    # Prefer the normalized columns produced by the previous script.
    source_column = ('normalized_source_template' if 'normalized_source_template' in df
                     else 'target_template_repository')
    repo_column = 'normalized_repository' if 'normalized_repository' in df else 'original_link'
    for column in (source_column, repo_column):
        if column not in df:
            parser.error(f'Missing required column: {column}')

    df['source_template'] = df[source_column].map(normalize_repo)
    df['generated_repository'] = df[repo_column].map(normalize_repo)
    df['commits_numeric'] = pd.to_numeric(df['commit_count'], errors='coerce')
    valid_commits = (df['commits_numeric'].notna()
                     & (df['commits_numeric'] >= 1)
                     & (df['commits_numeric'] % 1 == 0))
    true_template = df['is_template'].astype(str).str.strip().str.casefold().isin(
        {'true', '1', 'yes', 'y', 't'})
    valid = df[true_template & valid_commits].dropna(
        subset=['source_template', 'generated_repository']).copy()
    # If a generated repository occurs twice, retain it only once.
    # A conflicting source identity is reported and requires manual review.
    conflicting = valid.groupby('generated_repository')['source_template'].nunique()
    conflicting = conflicting[conflicting > 1]
    if len(conflicting):
        print(f'WARNING: {len(conflicting)} generated repository identities have conflicting source templates.')
        print('Review these records before interpreting the rankings.')
    valid = valid.drop_duplicates(subset=['generated_repository', 'source_template'])
    valid['same_owner'] = valid.apply(
        lambda r: r['generated_repository'].split('/', 1)[0]
        == r['source_template'].split('/', 1)[0], axis=1)
    valid['one_commit'] = valid['commits_numeric'].eq(1)
    valid['over_20_commits'] = valid['commits_numeric'].gt(20)
    valid['generated_owner'] = valid['generated_repository'].str.split('/').str[0]
    if 'language' not in valid:
        valid['language'] = 'Unknown'

    ranking = (valid.groupby('source_template', as_index=False)
               .agg(generated_templates_count=('generated_repository', 'nunique'),
                    generated_owners_count=('generated_owner', 'nunique'),
                    same_owner_count=('same_owner', 'sum'),
                    one_commit_count=('one_commit', 'sum'),
                    over_20_commits_count=('over_20_commits', 'sum'),
                    median_commits=('commits_numeric', 'median'),
                    mean_commits=('commits_numeric', 'mean'),
                    languages=('language', lambda s: ', '.join(sorted(set(s)))))
               .sort_values(['generated_templates_count', 'source_template'],
                            ascending=[False, True]).reset_index(drop=True))
    total_unique = valid['generated_repository'].nunique()
    ranking.insert(0, 'rank', range(1, len(ranking) + 1))
    ranking['share_of_unique_generated_templates_pct'] = (
        100 * ranking['generated_templates_count'] / total_unique
        if total_unique else 0).round(2)
    ranking['mean_commits'] = ranking['mean_commits'].round(2)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output / 'source_templates_ranked.csv', index=False)
    detail_columns = ['source_template', 'generated_repository', 'commits_numeric',
                      'language', 'same_owner']
    valid[detail_columns].sort_values(
        ['source_template', 'commits_numeric'], ascending=[True, False]
    ).to_csv(output / 'generated_templates_by_source.csv', index=False)
    if len(conflicting):
        valid[valid['generated_repository'].isin(conflicting.index)][detail_columns].to_csv(
            output / 'conflicting_source_references.csv', index=False)

    print(f'Input rows: {len(df):,}')
    print(f'Valid template-marked rows: {len(valid):,}')
    print(f'Unique generated templates: {total_unique:,}')
    print(f'Distinct source templates: {len(ranking):,}')
    print(f'\nTop {min(args.top, len(ranking))} source templates:')
    print(ranking[['source_template', 'generated_templates_count', 'languages']]
          .head(args.top).to_string(index=False))
    print(f'\nTop {min(args.top, len(ranking))} account for '
          f'{ranking.head(args.top)["generated_templates_count"].sum()} '
          f'of {total_unique} unique generated templates '
          f'({100 * ranking.head(args.top)["generated_templates_count"].sum() / total_unique:.2f}%).'
          if total_unique else '')
    print(f'\nSaved: {output / "source_templates_ranked.csv"}')
    print(f'Saved: {output / "generated_templates_by_source.csv"}')


if __name__ == '__main__':
    main()
