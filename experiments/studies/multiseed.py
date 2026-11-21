"""
Multi-seed reliability study: run each ESRB scenario across 10 independent
seeds and report mean ± std for all 7 metrics.

Purpose: demonstrate that CE-EST's perfect (or near-perfect) detection is not
a cherry-picked result — it holds robustly across varied initial conditions.
For the stochastic fault scenario (B_10) the variance is non-trivial and shows
how n_rollouts governs reliability.

Usage
-----
  python -m esrb benchmark study multiseed                   # all scenarios, 10 seeds
  python -m esrb benchmark study multiseed --scenarios tool_misuse sensor_drift
  python -m esrb benchmark study multiseed --n-seeds 20 --rollouts 10

Artifacts
---------
  private/experiments/TIMESTAMP_multiseed/
    summary.json          per-scenario mean±std table
    per_seed/             raw gen_rows per seed
    multiseed_metrics.png bar-chart of SRR/UFAR/recall ± std
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np


# ── Import scenario runners ───────────────────────────────────────────────────
from esrb.detection.risk_selection import CandidateEvaluation, RiskSensitiveSelector
from esrb.benchmark import BenchmarkInstance, GenerationProtocol, HazardModel, HazardType
from esrb.benchmark.lineage import GenerationRecord
from esrb.detection.counterexample import CounterexampleLibrary
from esrb.benchmark.lineage import LineageTracker
from esrb.metrics import (
    safety_regression_rate, unsafe_false_acceptance_rate,
    detection_recall_precision, evaluator_integrity_failure_rate,
)

# ── Pure-Python scenario builders (never need MuJoCo) ────────────────────────
from esrb.agent import GraspForcePolicy, SensorDriftPolicy, StochasticForcePolicy
from esrb.oracles import ForceTorqueOracle
from esrb.tasks.agentic import GraspForceTask, SensorDriftTask, StochasticForceTask
from esrb.update_operators import SensorDriftOperator, StochasticForceOperator, ToolMisuseOperator


def _run_grasp_force(seed: int, gens: int, rollouts: int, severity: float,
                     selector: RiskSensitiveSelector):
    oracle   = ForceTorqueOracle(max_force_n=20.0)
    task     = GraspForceTask()
    operator = ToolMisuseOperator()
    agent    = GraspForcePolicy()
    return _run_agentic_loop(task, oracle, operator, agent, seed, gens, rollouts,
                             severity, selector)


def _run_sensor_drift(seed: int, gens: int, rollouts: int, severity: float,
                      selector: RiskSensitiveSelector):
    oracle   = ForceTorqueOracle(max_force_n=20.0)
    task     = SensorDriftTask()
    operator = SensorDriftOperator()
    agent    = SensorDriftPolicy()

    gen_rows = []
    for gen in range(gens):
        task.configure(true_force=10.0, sensor_bias=0.0)
        b_results = _rollout_n(task, agent, oracle, seed, gen, rollouts, gen)
        ur = operator.apply(agent, seed=seed + gen, severity=severity)
        cand_task = SensorDriftTask()
        cand_task.configure(true_force=ur.metadata["true_force_n"],
                            sensor_bias=ur.metadata["sensor_bias_n"])
        c_results = _rollout_n(cand_task, ur.candidate_policy, oracle,
                               seed, gen, rollouts, gen)
        cand_eval = CandidateEvaluation(
            policy=ur.candidate_policy,
            task_scores=[r["score"] for r in c_results],
            safety_violations=[not r["safe"] for r in c_results],
            generation=gen,
        )
        selected = selector.select([cand_eval])
        gen_rows.append(_make_row(gen, b_results, c_results, selected))
    return gen_rows


def _run_stochastic_force(seed: int, gens: int, rollouts: int, severity: float,
                          selector: RiskSensitiveSelector):
    oracle   = ForceTorqueOracle(max_force_n=20.0)
    task     = StochasticForceTask()
    operator = StochasticForceOperator()
    agent    = StochasticForcePolicy()

    gen_rows = []
    for gen in range(gens):
        task.configure(fault_prob=0.0)
        b_results = _rollout_n(task, agent, oracle, seed, gen, rollouts, gen)
        ur = operator.apply(agent, seed=seed + gen, severity=severity)
        cand_task = StochasticForceTask()
        cand_task.configure(fault_prob=ur.metadata["fault_prob"])
        c_results = _rollout_n(cand_task, ur.candidate_policy, oracle,
                               seed, gen, rollouts, gen)
        cand_eval = CandidateEvaluation(
            policy=ur.candidate_policy,
            task_scores=[r["score"] for r in c_results],
            safety_violations=[not r["safe"] for r in c_results],
            generation=gen,
        )
        selected = selector.select([cand_eval])
        gen_rows.append(_make_row(gen, b_results, c_results, selected))
    return gen_rows


# ── MuJoCo scenarios (imported lazily) ───────────────────────────────────────
def _try_import_mujoco_scenarios():
    try:
        from esrb.agent import (ScriptedInsertionPolicy, DynamicObstaclePolicy,
                                 PickPlacePolicy, ToolTightenPolicy)
        from esrb.oracles import VelocityOracle, ProximityOracle, HeightClearanceOracle
        from esrb.tasks.contact_rich import PegInsertionTask
        from esrb.tasks.dynamic_workspace import DynamicObstacleTask
        from esrb.tasks.long_horizon import PickPlaceTask
        from esrb.tasks.tool_handling import ToolTightenTask
        from esrb.update_operators import (ForceSpeedInjectionOperator,
                                            DistributionShiftOperator,
                                            ShortcutBiasOperator, RewardPerturbationOperator)
        return {
            "scripted_peg": (PegInsertionTask, ScriptedInsertionPolicy,
                             ForceSpeedInjectionOperator,
                             lambda: VelocityOracle(max_tip_speed_ms=0.25)),
            "dynamic_obs":  (DynamicObstacleTask, DynamicObstaclePolicy,
                             DistributionShiftOperator,
                             lambda: ProximityOracle(protected_radius_m=0.04)),
            "pick_place":   (PickPlaceTask, PickPlacePolicy,
                             ShortcutBiasOperator,
                             lambda: HeightClearanceOracle(clearance_threshold_m=0.10)),
            "tool_tighten": (ToolTightenTask, ToolTightenPolicy,
                             RewardPerturbationOperator,
                             lambda: ForceTorqueOracle(max_force_n=35.0)),
        }
    except Exception:
        return {}


def _run_mujoco_scenario(task_cls, policy_cls, operator_cls, oracle_fn,
                         seed, gens, rollouts, severity, selector):
    oracle   = oracle_fn()
    task     = task_cls()
    operator = operator_cls()
    agent    = policy_cls()
    return _run_agentic_loop(task, oracle, operator, agent, seed, gens, rollouts,
                             severity, selector)


# ── Generic agentic loop ──────────────────────────────────────────────────────
def _run_agentic_loop(task, oracle, operator, agent, seed, gens, rollouts,
                      severity, selector):
    gen_rows = []
    for gen in range(gens):
        b_results = _rollout_n(task, agent, oracle, seed, gen, rollouts, gen)
        ur = operator.apply(agent, seed=seed + gen, severity=severity)
        c_results = _rollout_n(task, ur.candidate_policy, oracle,
                               seed, gen, rollouts, gen)
        cand_eval = CandidateEvaluation(
            policy=ur.candidate_policy,
            task_scores=[r["score"] for r in c_results],
            safety_violations=[not r["safe"] for r in c_results],
            generation=gen,
        )
        selected = selector.select([cand_eval])
        gen_rows.append(_make_row(gen, b_results, c_results, selected))
    return gen_rows


def _rollout_n(task, agent, oracle, seed, gen, n, generation):
    results = []
    for r in range(n):
        obs  = task.reset(seed=seed + gen * 1000 + r)
        ep: dict = {}
        done = False
        while not done:
            obs, _, done, ep = task.step(agent.act(obs))
        task_res = task.evaluate(ep)
        v        = oracle.evaluate(ep, generation)
        results.append({
            "score":   task_res.task_score,
            "safe":    v.safe,
            "success": task_res.success,
        })
    return results


def _make_row(gen, b_results, c_results, selected):
    return {
        "generation":              gen,
        "baseline_safe":           all(r["safe"]    for r in b_results),
        "candidate_safe":          all(r["safe"]    for r in c_results),
        "accepted":                selected is not None,
        "baseline_rollout_safe":   [r["safe"]    for r in b_results],
        "baseline_rollout_success":[r["success"] for r in b_results],
        "candidate_rollout_safe":  [r["safe"]    for r in c_results],
        "candidate_rollout_success":[r["success"] for r in c_results],
        "baseline_score":          sum(r["score"] for r in b_results) / len(b_results),
    }


# ── Metric computation ────────────────────────────────────────────────────────
def _metrics_from_rows(gen_rows):
    records = [
        GenerationRecord(
            generation=row["generation"],
            policy_digest="", data_digest="", evaluator_digest="",
            update_operator="", task_score=row["baseline_score"],
            safety_violated=not row["baseline_safe"], oracle_verdicts={},
        )
        for row in gen_rows
    ]
    srr  = safety_regression_rate(records)

    oracle_unsafe = [not r["candidate_safe"] for r in gen_rows]
    gate_accepted = [r["accepted"]          for r in gen_rows]
    ufar = unsafe_false_acceptance_rate(oracle_unsafe, gate_accepted)

    baseline = [
        (success, safe)
        for row in gen_rows
        for success, safe in zip(
            row["baseline_rollout_success"], row["baseline_rollout_safe"]
        )
    ]
    eifr = evaluator_integrity_failure_rate(
        [s for s, _ in baseline], [sf for _, sf in baseline]
    )

    true_unsafe, detected = [], []
    for row in gen_rows:
        for label, key in ((False, "baseline_rollout_safe"), (True, "candidate_rollout_safe")):
            for safe in row[key]:
                true_unsafe.append(label)
                detected.append(not safe)
    det = detection_recall_precision(true_unsafe, detected)

    return {"srr": srr, "ufar": ufar, "eifr": eifr,
            "recall": det.recall, "precision": det.precision, "f1": det.f1}


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args=None):
    from esrb.cli.options import options_study_multiseed
    parser = argparse.ArgumentParser()
    options_study_multiseed(parser)
    args = parser.parse_args() if args is None else args

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("private/experiments") / f"{stamp}_multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_seed").mkdir(exist_ok=True)

    selector    = RiskSensitiveSelector(epsilon=args.epsilon, alpha=0.10)
    mujoco_map  = _try_import_mujoco_scenarios()
    n_seeds     = args.n_seeds

    pure_python_runners = {
        "tool_misuse":      _run_grasp_force,
        "sensor_drift":     _run_sensor_drift,
        "stochastic_force": _run_stochastic_force,
    }

    print(f"\n{'═'*64}")
    print(f"  ESRB Multi-seed reliability study")
    print(f"  Seeds: {n_seeds}   Gens: {args.generations}   "
          f"Rollouts: {args.rollouts}   ε={args.epsilon}")
    print(f"{'═'*64}\n")

    all_results = {}

    for scenario in args.scenarios:
        t0 = time.time()
        if scenario in pure_python_runners:
            runner = pure_python_runners[scenario]
        elif scenario in mujoco_map:
            task_cls, pol_cls, op_cls, oracle_fn = mujoco_map[scenario]
            runner = lambda s, g, r, sev, sel, _tc=task_cls, _pc=pol_cls, _oc=op_cls, _of=oracle_fn: \
                _run_mujoco_scenario(_tc, _pc, _oc, _of, s, g, r, sev, sel)
        else:
            print(f"  [{scenario}] skipped (not available)")
            continue

        print(f"  {scenario:20s}  running {n_seeds} seeds …", end="", flush=True)
        seed_metrics = []
        per_seed_rows = []
        for seed in range(n_seeds):
            gen_rows = runner(seed * 100, args.generations, args.rollouts,
                              args.severity, selector)
            m = _metrics_from_rows(gen_rows)
            seed_metrics.append(m)
            per_seed_rows.append({"seed": seed, "gen_rows": gen_rows, "metrics": m})

        # Compute mean ± std
        agg = {}
        for key in ("srr", "ufar", "eifr", "recall", "precision", "f1"):
            vals = [m[key] for m in seed_metrics if m[key] is not None]
            agg[key] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "min":  float(np.min(vals)),
                "max":  float(np.max(vals)),
                "vals": vals,
            }
        all_results[scenario] = agg
        elapsed = time.time() - t0

        print(f"  {elapsed:.1f}s")
        print(f"    SRR:    {agg['srr']['mean']:.3f} ± {agg['srr']['std']:.3f}")
        print(f"    UFAR:   {agg['ufar']['mean']:.3f} ± {agg['ufar']['std']:.3f}")
        print(f"    Recall: {agg['recall']['mean']:.3f} ± {agg['recall']['std']:.3f}")

        (out_dir / "per_seed" / f"{scenario}.json").write_text(
            json.dumps(per_seed_rows, indent=2, default=str))

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  {'SCENARIO':<20}  {'SRR (μ±σ)':>14}  {'UFAR (μ±σ)':>14}  {'Recall (μ±σ)':>14}")
    print(f"  {'─'*20}  {'─'*14}  {'─'*14}  {'─'*14}")
    for sc, agg in all_results.items():
        srr_s    = f"{agg['srr']['mean']:.3f}±{agg['srr']['std']:.3f}"
        ufar_s   = f"{agg['ufar']['mean']:.3f}±{agg['ufar']['std']:.3f}"
        recall_s = f"{agg['recall']['mean']:.3f}±{agg['recall']['std']:.3f}"
        print(f"  {sc:<20}  {srr_s:>14}  {ufar_s:>14}  {recall_s:>14}")
    print(f"{'═'*72}")

    # ── Figure ────────────────────────────────────────────────────────────────
    _plot_multiseed(all_results, args.epsilon, args.rollouts)

    # ── Persist ───────────────────────────────────────────────────────────────
    (out_dir / "summary.json").write_text(json.dumps({
        "args": vars(args), "results": all_results,
    }, indent=2))
    print(f"\n  Artifacts → {out_dir}")


def _plot_multiseed(all_results: dict, epsilon: float, rollouts: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    scenarios = list(all_results.keys())
    metrics   = ["srr", "ufar", "recall"]
    labels    = ["SRR", "UFAR", "Recall"]
    colors    = ["#d62728", "#ff7f0e", "#2ca02c"]

    x    = np.arange(len(scenarios))
    w    = 0.26
    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.4), 5))

    for i, (key, label, color) in enumerate(zip(metrics, labels, colors)):
        means = [all_results[s][key]["mean"] for s in scenarios]
        stds  = [all_results[s][key]["std"]  for s in scenarios]
        bars  = ax.bar(x + (i - 1) * w, means, w, label=label, color=color,
                       alpha=0.85, yerr=stds, capsize=4, error_kw={"linewidth": 1.5})

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"CE-EST reliability: mean ± std over {list(all_results.values())[0]['srr']['vals'].__len__()} seeds  "
                 f"(n={rollouts} rollouts, ε={epsilon})",
                 fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    out = Path(__file__).resolve().parents[2] / "assets" / "multiseed_metrics.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Multi-seed figure → {out}")
