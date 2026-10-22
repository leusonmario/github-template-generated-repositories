#!/usr/bin/env python3
"""Select repository-level RQ2 cases for explanatory follow-up analysis.

Inputs: previously generated, *validated* SonarQube pair_summary.csv files.
No matching or SonarQube analysis is rerun.

Usage from quality_issues_and_security_hotspots/:

    python -u select_rq2_repository_cases.py --results-root vulnerability_issues
      --result-version validated --sample-root ../sample/persistence_sample_balanced
      --output-dir rq2_repository_cases

--sample-root is optional. If omitted, evolution_band from pair_summary.csv
is retained, but actual commit counts may be unavailable.

Outputs:
  all_repository_pairs.csv                 All pairs and rates
  ranked_quality_issue_repositories.csv   Eligible pairs sorted within language
  ranked_security_hotspot_repositories.csv
  within_template_variation.csv           Family-level min/max/median and spread
  all_template_family_rankings.csv         All families ranked by pooled persistence
  lowest_persistence_template_families.csv Bottom families per language/finding type
  highest_persistence_template_families.csv Top families per language/finding type
  lowest_persistence_family_repositories.csv Low-persistence examples from bottom families
  highest_persistence_family_repositories.csv High-persistence examples from top families
  within_template_contrasts.csv           Matched low/high cases in one family
  selected_repository_cases.csv           Deduplicated review shortlist
  selection_report.txt                    Selection policy and coverage

Important: 'not persistent' and 'repository exclusive' mean *unmatched under
our adopted matching criteria*, not demonstrably fixed or newly introduced.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

LANGUAGES = {"C#": "CSharp", "Java": "Java", "Python": "Python", "TypeScript": "TypeScript", "JavaScript": "JavaScript"}
KINDS = {
    "issues": ("template_issue_count", "persistent_issue_count", "no_longer_present_issue_count", "new_issue_count"),
    "hotspots": ("template_hotspot_count", "persistent_hotspot_count", "no_longer_present_hotspot_count", "new_hotspot_count"),
}


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=Path("vulnerability_issues"))
    p.add_argument("--result-version", default="validated", help="Directory suffix in issue_persistence_results_<version>.")
    p.add_argument("--output-dir", type=Path, default=Path("rq2_repository_cases"))
    p.add_argument("--sample-root", type=Path, default=None,
                   help="Optional parent of LANGUAGE/selected_sample.csv for commit counts; or a directory with language-named CSVs.")
    p.add_argument("--sample-csv", type=Path, action="append", default=[],
                   help="Additional optional selected_sample.csv (repeat for different languages).")
    p.add_argument("--min-issues", type=int, default=20,
                   help="Minimum source-template issue occurrences in a pair (default: 20).")
    p.add_argument("--min-hotspots", type=int, default=3,
                   help="Minimum source-template hotspot occurrences in a pair (default: 3).")
    p.add_argument("--top-n", type=int, default=3,
                   help="High- and low-persistence cases per language and kind (default: 3 each).")
    p.add_argument("--max-per-template", type=int, default=2,
                   help="Cap cases per source template in each high/low ranking (default: 2).")
    p.add_argument("--family-n", type=int, default=3,
                   help="Contrasting template families per language and kind (default: 3).")
    p.add_argument("--min-family-pairs", type=int, default=2)
    p.add_argument("--min-family-spread", type=float, default=20.0,
                   help="Minimum within-family persistence spread in percentage points (default: 20).")
    p.add_argument("--family-extremes", type=int, default=5,
                   help="How many lowest/highest persistence template families to select per language and concern type (default: 5).")
    p.add_argument("--family-cases", type=int, default=2,
                   help="How many representative repositories to select from each extreme family (default: 2).")
    p.add_argument("--min-ranking-family-pairs", type=int, default=2,
                   help="Minimum generated repository pairs per template for extreme family selection (default: 2).")
    p.add_argument("--min-family-issues", type=int, default=100,
                   help="Minimum total template-issue occurrences across a family for selection (default: 100).")
    p.add_argument("--min-family-hotspots", type=int, default=10,
                   help="Minimum total template-hotspot occurrences across a family for selection (default: 10).")
    p.add_argument("--family-rank-metric", choices=["weighted", "unweighted"], default="weighted",
                   help="Rank families by weighted pooled persistence where valid, or unweighted pooled persistence (default: weighted).")
    p.add_argument("--top-rules", type=int, default=3,
                   help="For selected pairs, list top N persistent/unmatched rules from per-pair JSON files (default: 3; 0 to skip).")
    return p.parse_args()


def numeric(val, label, allow_blank=False):
    if val is None or str(val).strip() == "":
        if allow_blank: return None
        raise ValueError(f"Missing {label}")
    try:
        value = float(val)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {label}: {val!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {label}: {val!r}")
    return value


def count(val, label):
    v = numeric(val, label)
    if v < 0 or not v.is_integer():
        raise ValueError(f"Expected nonnegative integer {label}, got {val!r}")
    return int(v)


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header: {path}")
        return list(reader), set(reader.fieldnames)


def write_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def repo_url(slug):
    parts = str(slug).strip().split("/")
    if len(parts) != 2 or not all(parts):
        return ""
    return "https://github.com/" + "/".join(quote(x, safe="-._~") for x in parts)


def read_pair_summary(root, version):
    records=[]
    locations=[]
    for language, folder in LANGUAGES.items():
        path=root / folder / f"issue_persistence_results_{version}" / "pair_summary.csv"
        if not path.is_file():
            raise SystemExit(f"Missing {path}\nUse --result-version to select the actual matching result folder.")
        data, header=read_rows(path)
        needed={"pair_id","template_repository","generated_repository"}
        for labels in KINDS.values(): needed.update(labels)
        missing=needed-header
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        locations.append(f"{language}: {path} ({len(data)} pairs)")
        seen=set()
        for row in data:
            rid=str(row["pair_id"]).strip()
            if not rid: raise ValueError(f"Empty pair_id in {path}")
            if rid in seen: raise ValueError(f"Duplicate pair ID {rid} in {path}")
            seen.add(rid)
            template=str(row["template_repository"]).strip()
            project=str(row["generated_repository"]).strip()
            if not template or not project: raise ValueError(f"Missing repository slug: {rid}")
            record={
                "language": language,
                "pair_id": rid,
                "template_repository": template,
                "generated_repository": project,
                "template_url": repo_url(template),
                "generated_url": repo_url(project),
                "evolution_band": row.get("evolution_band", ""),
                "sampling_weight": row.get("sampling_weight", ""),
                "sampling_method": row.get("sampling_method", ""),
                "selection_probability": row.get("selection_probability", ""),
                "commit_count": "",
                "family_population_n": "",
                "pair_output_dir": str(path.parent / "pairs" / rid),
            }
            for kind, labels in KINDS.items():
                t,p,unmatched,exclusive = [count(row[k], f"{rid} {k}") for k in labels]
                if t != p+unmatched:
                    raise ValueError(f"{rid} {kind}: template {t} != matched {p} + unmatched {unmatched}")
                record[f"{kind}_template"] = t
                record[f"{kind}_persistent"] = p
                record[f"{kind}_unmatched_template"] = unmatched
                record[f"{kind}_unmatched_generated"] = exclusive
                record[f"{kind}_persistence_pct"] = round(100*p/t, 4) if t else None
            records.append(record)
    return records, locations


def candidate_sample_files(args):
    paths=list(args.sample_csv)
    if args.sample_root:
        root=args.sample_root
        for lang, folder in LANGUAGES.items():
            for name in [folder, lang]:
                path=root/name/"selected_sample.csv"
                if path.is_file() and path not in paths:
                    paths.append(path)
        # Some users keep a single selected_sample.csv in the directory.
        flat=root/"selected_sample.csv"
        if flat.is_file() and flat not in paths:
            paths.append(flat)
    return paths


def augment_commits(records, files):
    by_pair={(r["language"], r["template_repository"].casefold(), r["generated_repository"].casefold()):r for r in records}
    by_slugs=defaultdict(list)
    for r in records:
        by_slugs[(r["template_repository"].casefold(),r["generated_repository"].casefold())].append(r)
    found=0
    for path in files:
        rows, fields=read_rows(path)
        for row in rows:
            project=(row.get("repo_slug") or (str(row.get("username") or "")+"/"+str(row.get("project_name") or ""))).strip()
            template=(row.get("template_slug") or row.get("target_template_repository") or "").strip()
            if not project or not template: continue
            hits=by_slugs.get((template.casefold(),project.casefold()), [])
            for r in hits:
                if row.get("commit_count") not in (None, ""):
                    val=numeric(row["commit_count"], "commit_count")
                    if val is not None and val>=0:
                        r["commit_count"]=int(val) if val.is_integer() else val
                        found+=1
                for src,dst in [("family_population_n","family_population_n"),("sampling_weight","sampling_weight"),("selection_probability","selection_probability")]:
                    if row.get(src) and not r.get(dst): r[dst]=row[src]
                if row.get("evolution_band") and not r.get("evolution_band"): r["evolution_band"]=row["evolution_band"]
    return found


def sorted_records(rows, kind, direction):
    k=f"{kind}_persistence_pct"
    den=f"{kind}_template"
    # By default favor high denominator among ties, because 100% of 1 is weak evidence.
    if direction=="high":
        return sorted(rows,key=lambda r:(-r[k],-r[den],r["template_repository"].casefold(),r["generated_repository"].casefold()))
    return sorted(rows,key=lambda r:(r[k],-r[den],r["template_repository"].casefold(),r["generated_repository"].casefold()))


def qualified(records,kind,min_den):
    return [r for r in records if r[f"{kind}_template"]>=min_den]


def select_with_cap(candidates,n,cap):
    picked=[]; used=Counter()
    for r in candidates:
        group=(r["language"],r["template_repository"].casefold())
        if used[group]>=cap: continue
        picked.append(r)
        used[group]+=1
        if len(picked)>=n: break
    return picked


def family_analysis(eligible,kind,min_size,min_spread):
    families=defaultdict(list)
    for r in eligible:
        families[(r["language"],r["template_repository"].casefold())].append(r)
    stats=[]; contrasts=[]
    for (language,_),members in families.items():
        if len(members)<min_size: continue
        ordered=sorted_records(members,kind,"low")
        low,high=ordered[0],ordered[-1]
        # max() for tie gives larger denominator to the high case, more interpretable.
        high=min(sorted_records(members,kind,"high"),key=lambda x:(-x[f"{kind}_persistence_pct"],-x[f"{kind}_template"]))
        low=sorted_records(members,kind,"low")[0]
        # If persistence all equal, no meaningful contrasting cases.
        pct=[r[f"{kind}_persistence_pct"] for r in members]
        spread=max(pct)-min(pct)
        d={"language":language,"kind":kind,"template_repository":low["template_repository"],
           "template_url":low["template_url"],"eligible_pair_count":len(members),
           "min_persistence_pct":min(pct),"median_persistence_pct":round(statistics.median(pct),4),
           "max_persistence_pct":max(pct),"spread_pp":round(spread,4),
           "low_generated_repository":low["generated_repository"],"high_generated_repository":high["generated_repository"],
           "low_pair_id":low["pair_id"],"high_pair_id":high["pair_id"],
           "low_commit_count":low["commit_count"],"high_commit_count":high["commit_count"],
           "low_evolution_band":low["evolution_band"],"high_evolution_band":high["evolution_band"]}
        stats.append(d)
        if spread>=min_spread and high["pair_id"]!=low["pair_id"]:
            contrasts.append((d,low,high))
    stats.sort(key=lambda r:(r["language"],-r["spread_pp"],-r["eligible_pair_count"],r["template_repository"].casefold()))
    contrasts.sort(key=lambda x:(x[0]["language"],-x[0]["spread_pp"],-x[0]["eligible_pair_count"]))
    return stats,contrasts


def template_family_rankings(records, kind, min_total, min_pairs, rank_metric, take):
    """Rank template families by the pooled matched/observed counts, not max spread.

    A family's concern denominator is the sum across its analyzed generated-repo
    pairs. This is not the number of distinct SonarQube findings in its template.
    """
    families=defaultdict(list)
    for r in records:
        if r[f"{kind}_template"] > 0:
            families[(r["language"],r["template_repository"].casefold())].append(r)
    rows=[]
    for (language,_),members in families.items():
        total=sum(x[f"{kind}_template"] for x in members)
        persistent=sum(x[f"{kind}_persistent"] for x in members)
        unmatched_generated=sum(x[f"{kind}_unmatched_generated"] for x in members)
        unmatched_template=sum(x[f"{kind}_unmatched_template"] for x in members)
        assert total == persistent+unmatched_template
        raw_weights=[]
        for m in members:
            val=m.get("sampling_weight","")
            try:
                w=float(val) if str(val).strip() else None
                if w is not None and (w<=0 or not math.isfinite(w)):w=None
            except (TypeError,ValueError):w=None
            raw_weights.append(w)
        if all(w is not None for w in raw_weights):
            weights=raw_weights
            weight_status="sampling_weight"
        elif all(w is None for w in raw_weights):
            weights=[1.0]*len(members)
            weight_status="unweighted"
        else:
            weights=None
            weight_status="incomplete_weights"
        weighted_pct=None
        if weights is not None:
            weighted_den=sum(w*m[f"{kind}_template"] for w,m in zip(weights,members))
            weighted_num=sum(w*m[f"{kind}_persistent"] for w,m in zip(weights,members))
            weighted_pct=round(100*weighted_num/weighted_den,4) if weighted_den else None
        micro_pct=round(100*persistent/total,4)
        rank_value=weighted_pct if rank_metric=="weighted" and weighted_pct is not None else micro_pct
        pct_values=[m[f"{kind}_persistence_pct"] for m in members]
        low=sorted_records(members,kind,"low")[0]
        high=sorted_records(members,kind,"high")[0]
        qualified=len(members)>=min_pairs and total>=min_total
        rows.append({
            "language":language,"kind":kind,
            "template_repository":members[0]["template_repository"],
            "template_url":members[0]["template_url"],
            "pairs_with_concern":len(members),
            "template_finding_observations":total,
            "persistent_observations":persistent,
            "unmatched_template_observations":unmatched_template,
            "unmatched_generated_observations":unmatched_generated,
            "micro_persistence_pct":micro_pct,
            "weighted_micro_persistence_pct":weighted_pct if weighted_pct is not None else "",
            "weighting_status":weight_status,
            "ranking_persistence_pct":rank_value,
            "ranking_metric_used":("weighted_micro" if rank_metric=="weighted" and weighted_pct is not None else "micro"),
            "min_repository_persistence_pct":min(pct_values),
            "max_repository_persistence_pct":max(pct_values),
            "spread_pp":round(max(pct_values)-min(pct_values),4),
            "low_repository":low["generated_repository"],
            "low_repository_url":low["generated_url"],
            "high_repository":high["generated_repository"],
            "high_repository_url":high["generated_url"],
            "eligible_for_extremes":int(qualified),
        })
    ranked_low=[]; ranked_high=[]
    for language in LANGUAGES:
        langrows=[x for x in rows if x["language"]==language and x["eligible_for_extremes"]]
        low_order=sorted(langrows,key=lambda x:(x["ranking_persistence_pct"],-x["template_finding_observations"],x["template_repository"].casefold()))
        high_order=sorted(langrows,key=lambda x:(-x["ranking_persistence_pct"],-x["template_finding_observations"],x["template_repository"].casefold()))
        ranked_low.extend([{**row,"rank_in_language":n} for n,row in enumerate(low_order[:take],1)])
        ranked_high.extend([{**row,"rank_in_language":n} for n,row in enumerate(high_order[:take],1)])
    rows.sort(key=lambda x:(list(LANGUAGES).index(x["language"]),x["ranking_persistence_pct"],-x["template_finding_observations"],x["template_repository"].casefold()))
    return rows,ranked_low,ranked_high,families


def load_json_list(path):
    if not path.is_file(): return None
    try:
        data=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return None
    return data if isinstance(data,list) else None


def top_rule_counts(pair_dir,kind,group,limit):
    filenames={
        ("issues","persistent"):("matched_issues.json","template_issue"),
        ("issues","not_persistent"):("not_persistent_template_issues.json","template_issue"),
        ("issues","generated_unmatched"):("new_generated_repository_issues.json","project_issue"),
        ("hotspots","persistent"):("matched_hotspots.json","template_hotspot"),
        ("hotspots","not_persistent"):("not_persistent_template_hotspots.json","template_hotspot"),
        ("hotspots","generated_unmatched"):("new_generated_repository_hotspots.json","project_hotspot"),
    }
    file,field=filenames[(kind,group)]
    data=load_json_list(Path(pair_dir)/file)
    if data is None:return ""
    ctr=Counter()
    for item in data:
        finding=item.get(field,{}) if isinstance(item,dict) else {}
        if not isinstance(finding,dict):continue
        rule=finding.get("rule_key") or finding.get("rule") or "UNKNOWN"
        if isinstance(rule,dict):rule=rule.get("key") or "UNKNOWN"
        ctr[str(rule)]+=1
    return " | ".join(f"{rule}: {c}" for rule,c in sorted(ctr.items(),key=lambda x:(-x[1],x[0]))[:limit])


def main():
    args=get_args()
    for flag in ("min_issues","min_hotspots","top_n","max_per_template","family_n","min_family_pairs","family_extremes","family_cases","min_ranking_family_pairs","min_family_issues","min_family_hotspots","top_rules"):
        if getattr(args,flag)<0: raise SystemExit(f"--{flag.replace('_','-')} must not be negative")
    if args.top_n==0 or args.max_per_template==0 or args.family_n==0 or args.min_family_pairs<2 or args.min_ranking_family_pairs<1 or args.family_extremes<1 or args.family_cases<1:
        raise SystemExit("top-n, max-per-template, family-n must be positive; min-family-pairs >=2")
    if not 0<=args.min_family_spread<=100:
        raise SystemExit("--min-family-spread must be between 0 and 100")
    records, input_info=read_pair_summary(args.results_root,args.result_version)
    sample_files=candidate_sample_files(args)
    if args.sample_root and not sample_files:
        print(f"NOTE: no selected_sample.csv found under {args.sample_root}; commit counts may remain unavailable.")
    found=augment_commits(records,sample_files)
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"all_repository_pairs.csv",records)
    selected={}  # (language, kind, pair_id) -> row with roles
    family_stats=[]; contrast_rows=[]; all_family_ranks=[]; low_family_ranks=[]; high_family_ranks=[]; report=[]
    report.append("RQ2 repository case-selection report")
    report.append("These are descriptive examples, not a statistically representative sample of mechanisms.")
    report.append("A low match rate does not prove that a concern was fixed; an unmatched generated finding does not prove it was newly introduced.")
    report.append(f"Minimum template issue count: {args.min_issues}; hotspot count: {args.min_hotspots}")
    report.append(f"Top/bottom per language and concern type: {args.top_n}; max per template per extreme: {args.max_per_template}")
    report.append(f"Within-family contrasts per language/kind: {args.family_n}; min family pairs: {args.min_family_pairs}; min spread: {args.min_family_spread} pp")
    report.append("Single-repository persistence is matched/template, so survey sampling weights do not alter its rank.")
    report.extend(input_info)
    report.append(f"Optional sample files: {len(sample_files)}; matched commit-count records: {found}")

    def add_case(r,kind,role,spread=""):
        key=(r["language"],kind,r["pair_id"])
        if key not in selected:
            out=dict(r)
            out.update({"kind":kind,"selection_reasons":[],"family_spread_pp":spread,
                        "finding_template":r[f"{kind}_template"],
                        "finding_persistent":r[f"{kind}_persistent"],
                        "finding_unmatched_template":r[f"{kind}_unmatched_template"],
                        "finding_unmatched_generated":r[f"{kind}_unmatched_generated"],
                        "persistence_pct":r[f"{kind}_persistence_pct"]})
            selected[key]=out
        case=selected[key]
        if role not in case["selection_reasons"]:case["selection_reasons"].append(role)
        if spread!="":case["family_spread_pp"]=spread

    for kind,threshold in (("issues",args.min_issues),("hotspots",args.min_hotspots)):
        eligible=qualified(records,kind,threshold)
        rows=[]
        for language in LANGUAGES:
            lang_rows=[r for r in eligible if r["language"]==language]
            for direction in ("low","high"):
                for rank,r in enumerate(sorted_records(lang_rows,kind,direction),1):
                    rows.append({**r,"rank_direction":direction,"rank_in_language":rank})
                choice=select_with_cap(sorted_records(lang_rows,kind,direction),args.top_n,args.max_per_template)
                for case in choice:add_case(case,kind,f"{direction}_persistence")
            report.append(f"{language} {kind}: {len(lang_rows)} eligible pairs / {sum(r['language']==language for r in records)} total")
        write_csv(output/("ranked_quality_issue_repositories.csv" if kind=="issues" else "ranked_security_hotspot_repositories.csv"),rows)
        family_all,family_low,family_high,grouped_families=template_family_rankings(
            records,kind,
            args.min_family_issues if kind=="issues" else args.min_family_hotspots,
            args.min_ranking_family_pairs,args.family_rank_metric,args.family_extremes
        )
        all_family_ranks.extend(family_all)
        low_family_ranks.extend(family_low)
        high_family_ranks.extend(family_high)
        for direction,ranked in (("low",family_low),("high",family_high)):
            for family in ranked:
                members=grouped_families[(family["language"],family["template_repository"].casefold())]
                # Explicitly choose the most extreme repos WITHIN the selected
                # low/high family. Select one or more per family even if the
                # family has little internal variation.
                reliable_members=qualified(members,kind,threshold)
                candidates=reliable_members if reliable_members else members
                choices=sorted_records(candidates,kind,direction)[:args.family_cases]
                for case in choices:
                    add_case(case,kind,f"{direction}_persistence_family")
        summaries, contrasts=family_analysis(eligible,kind,args.min_family_pairs,args.min_family_spread)
        family_stats.extend(summaries)
        for language in LANGUAGES:
            top=[c for c in contrasts if c[0]["language"]==language][:args.family_n]
            for d,low,high in top:
                add_case(low,kind,"within_family_low",d["spread_pp"])
                add_case(high,kind,"within_family_high",d["spread_pp"])
                contrast_rows.append({
                    **d,
                    "low_generated_url":low["generated_url"],"high_generated_url":high["generated_url"],
                    "low_template_finding_count":low[f"{kind}_template"],
                    "high_template_finding_count":high[f"{kind}_template"],
                    "low_persistent_count":low[f"{kind}_persistent"],
                    "high_persistent_count":high[f"{kind}_persistent"],
                    "low_unmatched_template_count":low[f"{kind}_unmatched_template"],
                    "high_unmatched_template_count":high[f"{kind}_unmatched_template"],
                    "low_unmatched_generated_count":low[f"{kind}_unmatched_generated"],
                    "high_unmatched_generated_count":high[f"{kind}_unmatched_generated"],
                    "low_pair_output_dir":low["pair_output_dir"],"high_pair_output_dir":high["pair_output_dir"],
                })
            report.append(f"{language} {kind}: {len([x for x in summaries if x['language']==language])} families with >= {args.min_family_pairs} eligible pairs, {len([x for x in contrasts if x[0]['language']==language])} with >= {args.min_family_spread:g} pp spread, {len(top)} contrasts selected")
    write_csv(output/"all_template_family_rankings.csv",all_family_ranks)
    write_csv(output/"lowest_persistence_template_families.csv",low_family_ranks)
    write_csv(output/"highest_persistence_template_families.csv",high_family_ranks)
    write_csv(output/"within_template_variation.csv",family_stats)
    write_csv(output/"within_template_contrasts.csv",contrast_rows)

    shortlist=[]
    for case in selected.values():
        case["selection_reasons"]=";".join(case["selection_reasons"])
        for role,col in [("persistent","top_persistent_rules"),("not_persistent","top_unmatched_template_rules"),("generated_unmatched","top_unmatched_generated_rules")]:
            case[col]=(top_rule_counts(case["pair_output_dir"],case["kind"],role,args.top_rules) if args.top_rules else "")
        shortlist.append(case)
    order={name:i for i,name in enumerate(LANGUAGES)}
    shortlist.sort(key=lambda r:(order[r["language"]],r["kind"],r["persistence_pct"],-r["finding_template"],r["generated_repository"].casefold()))
    write_csv(output/"selected_repository_cases.csv",shortlist)
    write_csv(output/"lowest_persistence_family_repositories.csv",
              [r for r in shortlist if "low_persistence_family" in r["selection_reasons"]])
    write_csv(output/"highest_persistence_family_repositories.csv",
              [r for r in shortlist if "high_persistence_family" in r["selection_reasons"]])
    report.append(f"Ranked {len(all_family_ranks)} template families, shortlisted {len(low_family_ranks)} LOW and {len(high_family_ranks)} HIGH families")
    report.append(f"Family extreme thresholds: pairs >= {args.min_ranking_family_pairs}; issues >= {args.min_family_issues}, hotspots >= {args.min_family_hotspots}; rank metric: {args.family_rank_metric}")
    report.append("Families are ranked by the pooled rate across their eligible observed pairs, not by a single repository's extreme rate.")
    report.append(f"Selected {len(shortlist)} distinct (language, kind, repository) cases; {len(contrast_rows)} within-family contrasts")
    report.append("For larger/less restrictive samples, lower --min-issues / --min-hotspots, or increase --top-n / --family-n.")
    report.append("Inspect actual code changes before interpreting unmatched findings as repaired or introduced.")
    (output/"selection_report.txt").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("\n".join(report[-12:]))
    print(f"\nResults written to: {output.resolve()}")
    print("Start with lowest_persistence_template_families.csv and lowest_persistence_family_repositories.csv")

if __name__=="__main__":
    main()

