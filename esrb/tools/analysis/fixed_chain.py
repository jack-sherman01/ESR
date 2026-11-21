"""Verify fixed-chain outputs and export paired regression data."""
from __future__ import annotations


import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import time

from esrb.benchmark.fixed_chain import decide, severity
from esrb.utils.hashing import hash_artifact, hash_dict


def read(path):
    with path.open() as f:
        return json.load(f)


def load_panel(folder, root, cases, archive_cache):
    packet = read(folder / "panel.json")
    rows = packet["rows"]
    assert [x["seed"] for x in rows] == [case["seed"] for case in cases[:len(rows)]]
    assert packet["unmeasured_seeds"] == [case["seed"] for case in cases[len(rows):]]
    details = {}
    for row, case in zip(rows, cases):
        descriptor = row["evidence"]
        archive = root / descriptor["file"]
        assert archive.resolve().is_relative_to(root)
        if archive not in archive_cache:
            assert hash_artifact(archive) == descriptor["sha256"], archive
            with gzip.open(archive, "rt") as f:
                evidence = json.load(f)
            assert evidence["case"] == case
            assert evidence["task"] == {k: v for k, v in row.items() if k != "evidence"}
            if row["reset_id"] is not None:
                assert hash_dict(evidence["initial"]) == row["reset_id"]
                assert severity(evidence["oracle"]["rules"]) == row["severity"]
            else:
                assert row["severity"] == "UNKNOWN"
            archive_cache[archive] = len(evidence.get("telemetry", {}))
        details[row["seed"]] = descriptor["file"]
    return rows, details


def analyse(root):
    status = read(root / "progress.json")
    decision_rows, paired, witnesses, unfinished = [], [], [], []
    cache = {}
    totals = defaultdict(Counter)
    for chain in sorted(root.glob("chain_*")):
        if not (chain / "protocol.json").is_file():
            unfinished.append(chain.name)
            continue
        protocol = read(chain / "protocol.json")
        cases = protocol["cases"]
        seeds = [x["seed"] for x in cases]
        scene, tier = protocol["scenario"], protocol["difficulty"]
        totals[scene]["chains"] += 1
        if not (chain / "initial/episodes/panel.json").is_file():
            unfinished.append(chain.name)
            continue
        parent, parent_files = load_panel(chain / "initial/episodes", root, cases, cache)
        active = read(chain / "initial/policy.json")["version"]
        for folder in sorted(chain.glob("generation_*")):
            if not (folder / "decision.json").is_file():
                unfinished.append(str(folder.relative_to(root)))
                continue
            saved = read(folder / "decision.json")
            if "candidate_id" not in saved:
                totals[scene][saved["reason"]] += 1
                continue
            child, child_files = load_panel(folder / "episodes", root, cases, cache)
            result = decide(seeds, parent, child)
            legacy_parent = isinstance(saved.get("parent"), str)
            assert all(saved[k] == v for k, v in result.items()
                       if k != "parent" or not legacy_parent), folder
            assert saved.get("parent_id", saved.get("parent")) == active
            artifact = read(folder / "policy.json")
            assert artifact["version"] == saved["candidate_id"] and artifact["parent"] == active
            key = str(folder.parent.relative_to(root))
            gen = int(folder.name.split("_")[-1])
            summaries = ("mean_score", "measured", "unmeasured", "successes", "mean_success_steps")
            outcome = {"chain": key, "scenario": scene, "difficulty": tier, "generation": gen,
                       "parent": active, "candidate": saved["candidate_id"], "accept": saved["accept"],
                       "active_after": saved["active_after"], "reason": saved["reason"],
                       "regression_pairs": len(saved["safe_to_violation_seeds"])}
            for role in ("parent", "candidate"):
                outcome.update({f"{role}_{field}": result[role][field] for field in summaries})
                outcome.update({f"{role}_{k.lower()}": v for k, v in result[role]["safety"].items()})
            decision_rows.append(outcome)
            totals[scene]["candidate_versions"] += 1
            totals[scene]["promotions"] += saved["accept"]
            totals[scene]["regressing_updates"] += bool(saved["safe_to_violation_seeds"])
            totals[scene]["regression_pairs"] += outcome["regression_pairs"]
            regressions = set(saved["safe_to_violation_seeds"])
            for p, c in zip(parent, child):
                row = {"chain": key, "scenario": scene, "difficulty": tier, "generation": gen,
                       "seed": c["seed"], "parent": p["policy"], "candidate": c["policy"],
                       "parent_score": p["score"], "candidate_score": c["score"],
                       "parent_severity": p["severity"], "candidate_severity": c["severity"],
                       "candidate_violations": "|".join(c["violations"]),
                       "reset_matched": p["reset_id"] is not None and p["reset_id"] == c["reset_id"],
                       "parent_archive": parent_files[p["seed"]], "candidate_archive": child_files[c["seed"]]}
                paired.append(row)
                if c["seed"] in regressions:
                    witnesses.append(row)
                    totals[scene]["severe_regression_pairs"] += c["severity"] == "SEVERE"
            if saved["accept"]:
                active = saved["candidate_id"]
                parent, parent_files = child, child_files
            assert saved["active_after"] == active
    return status, totals, decision_rows, paired, witnesses, unfinished, len(cache)


def export(rows, path):
    with path.open("w", newline="") as f:
        if rows:
            out = csv.DictWriter(f, fieldnames=list(rows[0]))
            out.writeheader()
            out.writerows(rows)


def main(args=None):
    from esrb.cli.options import options_tools_analyze_fixed_chain
    parser = argparse.ArgumentParser()
    options_tools_analyze_fixed_chain(parser)
    args = parser.parse_args() if args is None else args
    if args.wait:
        while read(args.run / "progress.json")["status"] == "RUNNING":
            time.sleep(20)
    status, totals, decisions, paired, witnesses, unfinished, cached = analyse(args.run.resolve())
    args.out.mkdir(parents=True)
    export(decisions, args.out / "decisions.csv")
    export(paired, args.out / "paired_episodes.csv")
    export(witnesses, args.out / "safety_regressions.csv")
    receipt = {"status": status["status"], "verified_archives": cached,
               "decisions": len(decisions), "paired_rows": len(paired),
               "scenes": {k: dict(v) for k, v in totals.items()}, "incomplete": unfinished}
    (args.out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    report = [
        "# ESRB 固定 10 seed 安全退化探索", "",
        f"运行状态 **{status['status']}**。验证 {cached} 份仿真归档和 {len(decisions)} 个父子门控决定。",
        f"测到 {len(witnesses)} 条同初态父安全→子违规记录，涉及 {len({x['scenario'] for x in witnesses})} 个任务。",
        "",
        "| 任务 | 链 | 测量候选 | 晋升 | 退化更新 | 父安全→子违规样本 | 严重退化样本 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, counts in totals.items():
        numbers = [counts[key] for key in ("chains", "candidate_versions", "promotions",
                                           "regressing_updates", "regression_pairs", "severe_regression_pairs")]
        report.append("| " + " | ".join([name, *map(str, numbers)]) + " |")
    if witnesses:
        report += ["", "## 每类退化的一条记录", "",
                   "| 任务 | 难度 | Seed | 等级 | 违规规则 |",
                   "|---|---|---:|---|---|"]
        selected = set()
        for witness in witnesses:
            key = (witness["scenario"], witness["candidate_severity"], witness["candidate_violations"])
            if key in selected:
                continue
            selected.add(key)
            report.append("| " + " | ".join(str(witness[k]).replace("|", "、") for k in
                          ("scenario", "difficulty", "seed", "candidate_severity", "candidate_violations")) + " |")
    report += ["", "同一条链内所有策略共享 10 个 seed 和实际初态。轻微违规计数达到 3 次时无法晋升，"
                    "仍保存剩余测量；确认严重违规后停止该候选的后续场景。未知和未测均不计作通过。"
                    "完整均分需要十条有效得分。",
               "", "初始候选来自普通参数优化。难度和下一条链根据先前数据调整，结果用于寻找例子。",
               "同一 seed 在一条链中重复比较，因此表中配对记录数包含可追溯的重复场景。",
               "", "decisions.csv 记录均分和决定，paired_episodes.csv 保存配对索引，"
                   "safety_regressions.csv 指向父、子原始压缩轨迹。",
               f"未完成面板保留于原目录，共 {len(unfinished)} 项。"]
    (args.out / "FINAL_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(receipt))
