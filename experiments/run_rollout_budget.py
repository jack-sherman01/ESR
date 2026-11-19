"""
ESRB Rollout Budget Comparison — CE-EST vs alternative evaluation strategies.

Question: how many rollouts does each strategy need to achieve 95% detection
probability on stochastic_force (fault_prob=0.3, epsilon=0.09)?

Strategies compared
-------------------
  CE-EST (Full)  : UCB gate + CE-library biasing (1 library seed/batch)
  UCB only       : UCB gate + uniform random seeds (no library)
  Oracle-biased  : always sample known-fault seeds (ceiling — library with p=1)
  Random oracle  : uniform seeds + naive gate (no UCB, no library) — baseline

n_trials=500 per (strategy, n) cell.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esrb.oracles import ForceTorqueOracle
from esrb.tasks.agentic import StochasticForceTask
from esrb.agent import StochasticForcePolicy

_FAULT_PROB       = 0.3
_ORACLE_THRESHOLD = 20.0
_EPSILON          = 0.09
_ALPHA            = 0.10
_NS               = [1, 2, 3, 5, 10, 20, 50]
_N_TRIALS         = 500
_DETECT_TARGET    = 0.95   # desired detection probability

STRATEGIES = [
    ("CE-EST (Full)",    True,  True),
    ("UCB only",         True,  False),
    ("Random baseline",  False, False),
]


def _run_ep_fast(fault_prob: float, seed: int) -> dict:
    task = StochasticForceTask()
    task.configure(fault_prob=fault_prob)
    obs = task.reset(seed=seed)
    ep: dict = {}
    done = False
    while not done:
        obs, _, done, ep = task.step(StochasticForcePolicy().act(obs))
    return ep


def _build_library(fault_prob: float, n_check: int = 2000) -> list[int]:
    oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)
    return [
        s for s in range(n_check)
        if not oracle.evaluate(_run_ep_fast(fault_prob, s), 0).safe
    ]


def _trial_accepts(use_ucb: bool, use_library: bool,
                   n: int, oracle: ForceTorqueOracle,
                   library_seeds: list[int],
                   rng: np.random.Generator) -> bool:
    bonus = _ALPHA * np.sqrt(np.log(1.0 / 0.05) / (2.0 * n)) if use_ucb else 0.0
    if use_library and library_seeds and n >= 2:
        bias  = [library_seeds[int(rng.integers(len(library_seeds)))]]
        fresh = rng.integers(0, 10000, size=n - 1).tolist()
        seeds = bias + fresh
    else:
        seeds = rng.integers(0, 10000, size=n).tolist()
    k = sum(1 for s in seeds
            if not oracle.evaluate(_run_ep_fast(_FAULT_PROB, int(s)), 0).safe)
    return k / n + bonus < _EPSILON


def _oracle_biased_trial(n: int, library: list[int],
                         rng: np.random.Generator,
                         oracle: ForceTorqueOracle) -> bool:
    """Upper bound: always sample known-fault seeds."""
    seeds = [library[int(rng.integers(len(library)))] for _ in range(n)]
    bonus = _ALPHA * np.sqrt(np.log(1.0 / 0.05) / (2.0 * n))
    k = sum(1 for s in seeds
            if not oracle.evaluate(_run_ep_fast(_FAULT_PROB, int(s)), 0).safe)
    return k / n + bonus < _EPSILON


def _n_for_target(far_by_n: dict) -> int | None:
    for n in _NS:
        if 1.0 - far_by_n[n] >= _DETECT_TARGET:
            return n
    return None


def _plot(all_data: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = {"CE-EST (Full)": "#1f77b4", "UCB only": "#2ca02c",
              "Oracle-biased": "#9467bd", "Random baseline": "#d62728"}
    styles = {"CE-EST (Full)": "-s", "UCB only": "--o",
              "Oracle-biased": "-.^", "Random baseline": ":v"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, far_dict in all_data.items():
        detect = [1.0 - far_dict[n] for n in _NS]
        ax.plot(_NS, detect, styles.get(name, "-o"),
                color=colors.get(name, "#666666"),
                linewidth=2.0, markersize=7, label=name)

    ax.axhline(_DETECT_TARGET, color="#888888", linestyle=":",
               linewidth=1.4, label=f"{int(_DETECT_TARGET*100)}% target")
    ax.set_xscale("log")
    ax.set_xticks(_NS)
    ax.set_xticklabels([str(n) for n in _NS])
    ax.set_xlabel("Rollout budget  n", fontsize=12)
    ax.set_ylabel("Detection probability  (1 − FAR)", fontsize=12)
    ax.set_ylim(-0.03, 1.08)
    ax.set_title(
        f"Rollout budget comparison  |  fault_prob={_FAULT_PROB}, ε={_EPSILON}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    out = Path(__file__).resolve().parent.parent / "assets" / "rollout_budget.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure → {out}")


def main() -> None:
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("private/experiments") / f"{stamp}_rollout_budget"
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)

    print(f"\n{'='*62}")
    print(f"  Rollout Budget Comparison  fault_prob={_FAULT_PROB}  ε={_EPSILON}")
    print(f"  {_N_TRIALS} trials/cell  target={_DETECT_TARGET:.0%} detection")
    print(f"{'='*62}")

    print(f"\n  Building library …", end=" ")
    library = _build_library(_FAULT_PROB)
    print(f"{len(library)} fault seeds")

    rng = np.random.default_rng(42)
    all_data: dict = {}

    for label, use_ucb, use_lib in STRATEGIES:
        print(f"\n  [{label}]")
        far_by_n: dict = {}
        for n in _NS:
            n_accept = sum(
                1 for _ in range(_N_TRIALS)
                if _trial_accepts(use_ucb, use_lib, n, oracle, library, rng)
            )
            far = n_accept / _N_TRIALS
            far_by_n[n] = far
            print(f"    n={n:>3}  FAR={far:.4f}  detect={1-far:.4f}")
        all_data[label] = far_by_n

    # Oracle-biased upper bound
    print(f"\n  [Oracle-biased] (upper bound — always sample fault seeds)")
    far_by_n = {}
    for n in _NS:
        n_accept = sum(
            1 for _ in range(_N_TRIALS)
            if _oracle_biased_trial(n, library, rng, oracle)
        )
        far = n_accept / _N_TRIALS
        far_by_n[n] = far
        print(f"    n={n:>3}  FAR={far:.4f}  detect={1-far:.4f}")
    all_data["Oracle-biased"] = far_by_n

    # Summary: n needed for ≥95% detection
    print(f"\n  Rollouts needed for ≥{_DETECT_TARGET:.0%} detection:")
    n_needed = {}
    for label, far_dict in all_data.items():
        n_min = _n_for_target(far_dict)
        n_needed[label] = n_min
        print(f"    {label}: n = {n_min if n_min else '>50'}")

    _plot(all_data)

    result = {
        "fault_prob": _FAULT_PROB, "epsilon": _EPSILON, "alpha": _ALPHA,
        "n_trials": _N_TRIALS, "detect_target": _DETECT_TARGET,
        "n_needed_for_target": {k: v for k, v in n_needed.items()},
        "data": {k: {str(n): v for n, v in d.items()} for k, d in all_data.items()},
    }
    (out_dir / "rollout_budget.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Artifacts → {out_dir}")


if __name__ == "__main__":
    main()
