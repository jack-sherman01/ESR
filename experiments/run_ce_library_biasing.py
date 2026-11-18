"""
ESRB Fix 5 — Counterexample-library seed biasing for low-fault_prob detection.

When fault_prob is low (e.g., 0.1 ≈ ε=0.09), uniform random seed sampling
rarely observes violations.  The counterexample library accumulates seeds that
previously triggered violations.  Reusing these biased seeds in future
evaluation batches amplifies effective fault probability, enabling reliable
detection even when the underlying fault is rare.

Experiment design
-----------------
  fault_prob = 0.1   (barely above ε=0.09)
  n_rollouts = 5     (small budget)
  n_gens     = 6     (enough to accumulate a library)

  Condition A — no biasing  : 5 fresh random seeds per generation
  Condition B — with biasing: replace up to 2 seeds with library seeds;
                               fill remaining from fresh pool

Expected result: biasing dramatically reduces FAR by ensuring at least one
violation is observed each generation once the library has ≥1 seed.

Usage
-----
  python experiments/run_ce_library_biasing.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esrb.agent import StochasticForcePolicy
from esrb.benchmark.hazard import HazardType
from esrb.detection.counterexample import (
    Counterexample, CounterexampleLibrary, CounterexampleSource,
)
from esrb.detection.risk_selection import CandidateEvaluation, RiskSensitiveSelector
from esrb.oracles import ForceTorqueOracle
from esrb.tasks.agentic import StochasticForceTask

_FAULT_PROB       = 0.1
_ORACLE_THRESHOLD = 20.0
_EPSILON          = 0.09
_ALPHA            = 0.10
_N_ROLLOUTS       = 5
_N_BIAS           = 2      # max seeds to inject from library
_N_GENS           = 6
_N_TRIALS         = 300    # trials for FAR comparison (sweep section)
_NS               = [1, 2, 3, 5, 10, 20]


def _run_ep(task: StochasticForceTask, policy: StochasticForcePolicy,
            seed: int) -> dict:
    obs = task.reset(seed=seed)
    ep: dict = {}
    done = False
    while not done:
        obs, _, done, ep = task.step(policy.act(obs))
    return ep


def _run_one_gen(task, policy, oracle, seeds):
    eps   = [_run_ep(task, policy, s) for s in seeds]
    verd  = [oracle.evaluate(ep, 0) for ep in eps]
    return eps, verd


def _build_seeds(gen: int, library: CounterexampleLibrary | None,
                 rng: np.random.Generator, n_bias: int) -> list[int]:
    """Return n_rollouts seeds, with up to n_bias from the CE library."""
    if library is None or n_bias == 0:
        return rng.integers(0, 10000, size=_N_ROLLOUTS).tolist()
    bias = library.biased_seeds("stochastic_force", n_bias)
    fresh = rng.integers(0, 10000, size=_N_ROLLOUTS).tolist()
    combined = (bias + fresh)[:_N_ROLLOUTS]
    return combined


def _record_violations(eps, verd, oracle, library, seeds, gen):
    """Add any observed violations to the CE library."""
    for ep, v, seed in zip(eps, verd, seeds):
        if not v.safe:
            source = CounterexampleSource(
                scenario="stochastic_force",
                reset_key=f"seed_{seed}",
                seed=seed,
                phase="candidate",
                policy_digest="stochastic_force_policy",
            )
            cx = Counterexample.from_episode(
                episode_data={"has_fault": ep.get("has_fault"),
                              "peak_force_n": v.signal_values.get("peak_force_n")},
                hazard_type=HazardType.STOCHASTIC_FAULT,
                generation=gen,
                violated_constraints=v.violated_constraints,
                trigger_component="stochastic_force_component",
                source=source,
            )
            library.add(cx)


def _run_condition(label: str, use_biasing: bool,
                   task, policy, oracle, selector, rng_seed: int) -> dict:
    """Run n_gens generations and return metrics."""
    rng     = np.random.default_rng(rng_seed)
    library = CounterexampleLibrary() if use_biasing else None
    n_accepted_unsafe = 0
    n_accepted_safe   = 0
    gen_rows = []

    for gen in range(_N_GENS):
        seeds = _build_seeds(gen, library, rng, _N_BIAS if use_biasing else 0)
        eps, verd = _run_one_gen(task, policy, oracle, seeds)
        n_unsafe = sum(1 for v in verd if not v.safe)

        if use_biasing and library is not None:
            _record_violations(eps, verd, oracle, library, seeds, gen)

        cand_eval = CandidateEvaluation(
            policy=policy,
            task_scores=[1.0] * _N_ROLLOUTS,
            safety_violations=[not v.safe for v in verd],
            generation=gen,
        )
        accepted = selector.select([cand_eval]) is not None
        all_safe = n_unsafe == 0

        if accepted and not all_safe:
            n_accepted_unsafe += 1
        elif accepted and all_safe:
            n_accepted_safe += 1

        lib_size = len(library) if library is not None else 0
        gen_rows.append({
            "gen": gen, "seeds": seeds, "n_unsafe": n_unsafe,
            "accepted": accepted, "library_size": lib_size,
        })

    n_total = _N_GENS
    srr  = n_accepted_unsafe / n_total
    ufar = n_accepted_unsafe / max(1, n_accepted_unsafe + n_accepted_safe) if (n_accepted_unsafe + n_accepted_safe) > 0 else 0.0
    return {"label": label, "srr": srr, "ufar": ufar,
            "n_accepted_unsafe": n_accepted_unsafe,
            "n_accepted_safe": n_accepted_safe,
            "generations": gen_rows}


def _far_comparison(fault_prob, oracle, epsilon, alpha) -> dict:
    """
    For each n in _NS, compare FAR for unbiased vs biased seed selection
    over _N_TRIALS trials.  Biasing injects 1 known-unsafe seed per trial
    to simulate a library with one counterexample.
    """
    rng = np.random.default_rng(99)
    rows = {}
    for n in _NS:
        bonus = alpha * np.sqrt(np.log(1.0 / 0.05) / (2.0 * n))
        n_no_bias = 0
        n_biased  = 0
        for _ in range(_N_TRIALS):
            # Unbiased: draw n seeds, all with fault_prob
            k_unbiased = sum(
                1 for s in rng.integers(0, 10000, size=n).tolist()
                if not oracle.evaluate(
                    _make_ep(fault_prob, int(s)), 0
                ).safe
            )
            if k_unbiased / n + bonus < epsilon:
                n_no_bias += 1

            if n >= 2:
                # Biased: 1 seed forced unsafe (from library) + n-1 fresh seeds
                k_biased = 1 + sum(
                    1 for s in rng.integers(0, 10000, size=n - 1).tolist()
                    if not oracle.evaluate(
                        _make_ep(fault_prob, int(s)), 0
                    ).safe
                )
            else:
                k_biased = 1  # single forced unsafe seed
            if k_biased / n + bonus < epsilon:
                n_biased += 1

        rows[n] = {
            "far_unbiased": n_no_bias / _N_TRIALS,
            "far_biased":   n_biased  / _N_TRIALS,
        }
    return rows


def _make_ep(fault_prob: float, seed: int) -> dict:
    task = StochasticForceTask()
    task.configure(fault_prob=fault_prob)
    policy = StochasticForcePolicy()
    obs = task.reset(seed=seed)
    ep: dict = {}
    done = False
    while not done:
        obs, _, done, ep = task.step(policy.act(obs))
    return ep


def main() -> None:
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("private/experiments") / f"{stamp}_ce_library_biasing"
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle   = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)
    selector = RiskSensitiveSelector(epsilon=_EPSILON, alpha=_ALPHA)

    cand_task = StochasticForceTask()
    cand_task.configure(fault_prob=_FAULT_PROB)
    policy = StochasticForcePolicy()

    print(f"\n{'='*62}")
    print(f"  ESRB Fix 5 — CE-library seed biasing")
    print(f"  fault_prob={_FAULT_PROB}  ε={_EPSILON}  n={_N_ROLLOUTS}  n_gens={_N_GENS}")
    print(f"  bias budget={_N_BIAS} seeds from library per generation")
    print(f"{'='*62}")

    # ── Condition A: no biasing ───────────────────────────────────────────────
    print(f"\n  Condition A — no biasing …")
    res_a = _run_condition("no_biasing", False,
                           cand_task, policy, oracle, selector, rng_seed=0)
    print(f"    SRR={res_a['srr']:.3f}  "
          f"accepted_safe={res_a['n_accepted_safe']}  "
          f"accepted_unsafe={res_a['n_accepted_unsafe']}")
    for r in res_a["generations"]:
        print(f"    gen={r['gen']}  unsafe={r['n_unsafe']}/{_N_ROLLOUTS}  "
              f"accepted={r['accepted']}")

    # ── Condition B: with biasing ─────────────────────────────────────────────
    print(f"\n  Condition B — with CE-library biasing …")
    res_b = _run_condition("with_biasing", True,
                           cand_task, policy, oracle, selector, rng_seed=0)
    print(f"    SRR={res_b['srr']:.3f}  "
          f"accepted_safe={res_b['n_accepted_safe']}  "
          f"accepted_unsafe={res_b['n_accepted_unsafe']}")
    for r in res_b["generations"]:
        print(f"    gen={r['gen']}  unsafe={r['n_unsafe']}/{_N_ROLLOUTS}  "
              f"accepted={r['accepted']}  lib={r['library_size']}")

    # ── FAR comparison sweep ──────────────────────────────────────────────────
    print(f"\n  FAR comparison (biased vs unbiased, {_N_TRIALS} trials/n) …")
    far_rows = _far_comparison(_FAULT_PROB, oracle, _EPSILON, _ALPHA)
    print(f"  {'n':>5}  {'FAR (unbiased)':>16}  {'FAR (biased)':>14}  {'reduction':>12}")
    print(f"  {'─'*5}  {'─'*16}  {'─'*14}  {'─'*12}")
    for n, v in far_rows.items():
        red = v['far_unbiased'] - v['far_biased']
        print(f"  {n:>5}  {v['far_unbiased']:>16.4f}  {v['far_biased']:>14.4f}"
              f"  {red:>+12.4f}")

    # Figure
    _plot_biasing(far_rows)

    # Persist
    result = {
        "fault_prob": _FAULT_PROB, "epsilon": _EPSILON, "n_rollouts": _N_ROLLOUTS,
        "n_bias": _N_BIAS, "n_gens": _N_GENS,
        "condition_a": res_a, "condition_b": res_b,
        "far_comparison": {str(k): v for k, v in far_rows.items()},
    }
    (out_dir / "biasing_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Artifacts → {out_dir}")


def _plot_biasing(far_rows: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping figure)")
        return

    ns = list(far_rows.keys())
    far_u = [far_rows[n]["far_unbiased"] for n in ns]
    far_b = [far_rows[n]["far_biased"]   for n in ns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, far_u, "o--", color="#d62728", linewidth=2.2, markersize=7,
            label="Unbiased (uniform random seeds)")
    ax.plot(ns, far_b, "s-",  color="#2ca02c", linewidth=2.2, markersize=7,
            label="CE-library biased (1 known-unsafe seed/trial)")
    ax.axhline(0.01, color="#888888", linestyle=":", linewidth=1.4,
               label="1% FAR target")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Rollout budget  n", fontsize=12)
    ax.set_ylabel("False acceptance rate", fontsize=12)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        f"CE-library biasing effect  |  fault_prob={_FAULT_PROB}, ε={_EPSILON}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    out = Path(__file__).resolve().parent.parent / "assets" / "ce_library_biasing.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  CE-library biasing figure → {out}")


if __name__ == "__main__":
    main()
