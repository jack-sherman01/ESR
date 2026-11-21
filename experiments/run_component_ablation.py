"""
ESRB CE-EST Component Ablation — 2×2 matrix (UCB gate × CE library).

Isolates the contribution of each CE-EST component on stochastic_force
(fault_prob=0.3, epsilon=0.09), running 500 trials per condition.

Conditions
----------
  Full CE-EST : UCB (Wilson-score)  ×  CE-library seed biasing
  No library  : UCB                 ×  uniform random seeds
  No UCB      : naive k/n gate      ×  CE-library seed biasing
  Baseline    : naive k/n gate      ×  uniform random seeds

The CE library is populated with seeds from the first 2000 that have
fault_draw < fault_prob.  In biased trials, one library seed is included
per rollout batch (n >= 2 required).

Metric: FAR = Pr[gate accepts an unsafe candidate] (should be near 0).
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
_N_BIAS           = 1

CONDITIONS = [
    ("Full CE-EST", True,  True),
    ("No library",  True,  False),
    ("No UCB",      False, True),
    ("Baseline",    False, False),
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


def _compute_far(use_ucb: bool, use_library: bool,
                 oracle: ForceTorqueOracle,
                 library_seeds: list[int]) -> dict[int, float]:
    rng = np.random.default_rng(42)
    result = {}
    for n in _NS:
        bonus    = _ALPHA * np.sqrt(np.log(1.0 / 0.05) / (2.0 * n)) if use_ucb else 0.0
        n_accept = 0
        for _ in range(_N_TRIALS):
            if use_library and library_seeds and n >= 2:
                bias  = [library_seeds[int(rng.integers(len(library_seeds)))]]
                fresh = rng.integers(0, 10000, size=n - 1).tolist()
                seeds = bias + fresh
            else:
                seeds = rng.integers(0, 10000, size=n).tolist()

            k = sum(
                1 for s in seeds
                if not oracle.evaluate(_run_ep_fast(_FAULT_PROB, int(s)), 0).safe
            )
            if k / n + bonus < _EPSILON:
                n_accept += 1
        result[n] = n_accept / _N_TRIALS
    return result


def _plot(all_data: dict) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib unavailable — skipping figure)")
        return ""

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    styles = ["-s", "--o", "-^", ":v"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, _, __), color, sty in zip(CONDITIONS, colors, styles):
        far = [all_data[label][n] for n in _NS]
        ax.plot(_NS, far, sty, color=color, linewidth=2.0,
                markersize=7, label=label)

    ax.axhline(0.01, color="#888888", linestyle=":", linewidth=1.2,
               label="1% FAR target")
    ax.set_xscale("log")
    ax.set_xticks(_NS)
    ax.set_xticklabels([str(n) for n in _NS])
    ax.set_xlabel("Rollout budget  n", fontsize=12)
    ax.set_ylabel("False acceptance rate", fontsize=12)
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(
        f"CE-EST component ablation  |  fault_prob={_FAULT_PROB}, ε={_EPSILON}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    out = Path(__file__).resolve().parent.parent / "assets" / "component_ablation.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def main() -> None:
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("private/experiments") / f"{stamp}_component_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)

    print(f"\n{'='*62}")
    print(f"  CE-EST Component Ablation  fault_prob={_FAULT_PROB}  ε={_EPSILON}")
    print(f"  {_N_TRIALS} trials/cell  n_bias={_N_BIAS}")
    print(f"{'='*62}")

    print(f"\n  Building CE library …", end=" ")
    library = _build_library(_FAULT_PROB)
    print(f"{len(library)} fault seeds in first 2000 "
          f"({100*len(library)/2000:.1f}%,  expected {_FAULT_PROB*100:.0f}%)")

    all_data: dict = {}
    for label, use_ucb, use_lib in CONDITIONS:
        print(f"\n  [{label}]  UCB={'yes' if use_ucb else 'no'}  "
              f"library={'yes' if use_lib else 'no'}")
        rows = _compute_far(use_ucb, use_lib, oracle, library)
        all_data[label] = rows
        print(f"  {'n':>5}  {'FAR':>8}")
        for n, far in rows.items():
            print(f"  {n:>5}  {far:>8.4f}")

    # Cross-condition summary
    print(f"\n  Summary (FAR)")
    print(f"  {'n':>5}", end="")
    for label, _, __ in CONDITIONS:
        print(f"  {label[:12]:>12}", end="")
    print()
    for n in _NS:
        print(f"  {n:>5}", end="")
        for label, _, __ in CONDITIONS:
            print(f"  {all_data[label][n]:>12.4f}", end="")
        print()

    fig_path = _plot(all_data)
    if fig_path:
        print(f"\n  Figure → {fig_path}")

    result = {
        "fault_prob": _FAULT_PROB, "epsilon": _EPSILON, "alpha": _ALPHA,
        "n_trials": _N_TRIALS, "library_size": len(library),
        "ns": _NS,
        "conditions": {
            label: {str(k): v for k, v in rows.items()}
            for label, rows in all_data.items()
        },
    }
    (out_dir / "component_ablation.json").write_text(json.dumps(result, indent=2))
    print(f"  Artifacts → {out_dir}")


if __name__ == "__main__":
    main()
