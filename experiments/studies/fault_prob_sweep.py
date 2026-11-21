"""
ESRB Fix 3 — FAR phase transition as a function of fault_prob.

Sweep fault_prob in {0.1, 0.2, 0.3, 0.5} and for each value compute
the false-acceptance rate (FAR) of the UCB gate and naive gate across
n_rollouts in [1, 2, 3, 5, 10, 20, 50].

Key result: when fault_prob is close to epsilon (0.09), the system needs
many rollouts to reliably detect the fault.  The phase transition is sharp:
  - fault_prob >> epsilon: FAR drops quickly (high p means k>0 is likely)
  - fault_prob ~ epsilon: FAR stays high until n is large (k=0 is probable)

Usage
-----
  python -m esrb benchmark study fault-prob-sweep
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


from esrb.oracles import ForceTorqueOracle
from esrb.tasks.agentic import StochasticForceTask
from esrb.agent import StochasticForcePolicy

_ORACLE_THRESHOLD = 20.0
_EPSILON          = 0.09
_ALPHA            = 0.10
_N_TRIALS         = 500
_FAULT_PROBS      = [0.1, 0.2, 0.3, 0.5]
_NS               = [1, 2, 3, 5, 10, 20, 50]


def _run_ep_fast(fault_prob: float, seed: int) -> dict:
    task = StochasticForceTask()
    task.configure(fault_prob=fault_prob)
    policy = StochasticForcePolicy()
    obs = task.reset(seed=seed)
    ep: dict = {}
    done = False
    while not done:
        obs, _, done, ep = task.step(policy.act(obs))
    return ep


def _sweep_one(fault_prob: float, oracle: ForceTorqueOracle) -> list[dict]:
    """Return rows of {n, far_ucb, far_naive} for one fault_prob."""
    rng = np.random.default_rng(42)
    rows = []
    for n in _NS:
        n_ucb = 0
        n_naive = 0
        bonus = _ALPHA * np.sqrt(np.log(1.0 / 0.05) / (2.0 * n))
        for _ in range(_N_TRIALS):
            seeds = rng.integers(0, 10000, size=n).tolist()
            k = sum(
                1 for s in seeds
                if not oracle.evaluate(_run_ep_fast(fault_prob, s), 0).safe
            )
            ucb_rate  = k / n + bonus
            naive_rate = k / n
            if ucb_rate  < _EPSILON:
                n_ucb   += 1
            if naive_rate < _EPSILON:
                n_naive += 1
        rows.append({
            "fault_prob": fault_prob,
            "n_rollouts": n,
            "far_ucb":    n_ucb   / _N_TRIALS,
            "far_naive":  n_naive / _N_TRIALS,
            "bonus":      bonus,
        })
    return rows


def _plot_phase_transition(all_rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  (matplotlib not available — skipping figure)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    colors = cm.plasma(np.linspace(0.15, 0.85, len(_FAULT_PROBS)))

    for ax_idx, (gate_key, gate_label) in enumerate([
        ("far_ucb",   "UCB gate"),
        ("far_naive", "Naive gate"),
    ]):
        ax = axes[ax_idx]
        for fp, color in zip(_FAULT_PROBS, colors):
            subset = [r for r in all_rows if r["fault_prob"] == fp]
            ns  = [r["n_rollouts"] for r in subset]
            far = [r[gate_key]     for r in subset]
            ax.plot(ns, far, "o-", color=color, linewidth=2.0, markersize=6,
                    label=f"fault_prob={fp:.1f}")

        ax.axhline(0.01, color="#888888", linestyle=":", linewidth=1.2,
                   label="1% FAR target")
        ax.axvline(1, color="#cccccc", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xticks(_NS)
        ax.set_xticklabels([str(n) for n in _NS])
        ax.set_xlabel("Rollout budget  n", fontsize=12)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(gate_label, fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("False acceptance rate", fontsize=12)

    # Annotate phase boundary
    axes[0].text(
        1.05, 0.98,
        f"At n=1: UCB bonus > ε\n→ FAR=0 for ALL fault_prob",
        transform=axes[0].transAxes, fontsize=8.5, va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(
        f"FAR phase transition: UCB gate vs naive gate  |  ε={_EPSILON}, α={_ALPHA}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Phase-transition figure → {out_path}")


def main(args=None) -> None:
    oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)

    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("private/experiments") / f"{stamp}_fault_prob_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  ESRB Fix 3 — FAR phase transition  (ε={_EPSILON}, α={_ALPHA})")
    print(f"  fault_probs: {_FAULT_PROBS}")
    print(f"  n_rollouts:  {_NS}")
    print(f"  trials/cell: {_N_TRIALS}")
    print(f"{'='*62}")

    all_rows: list[dict] = []
    for fp in _FAULT_PROBS:
        print(f"\n  fault_prob={fp:.1f} …", end=" ", flush=True)
        rows = _sweep_one(fp, oracle)
        all_rows.extend(rows)
        # Print small summary table
        print(f"done")
        print(f"  {'n':>5}  {'FAR (UCB)':>10}  {'FAR (naive)':>12}")
        for r in rows:
            print(f"  {r['n_rollouts']:>5}  {r['far_ucb']:>10.4f}  {r['far_naive']:>12.4f}")

    # Figure
    fig_path = Path(__file__).resolve().parents[2] / "assets" / "fault_prob_phase_transition.png"
    _plot_phase_transition(all_rows, fig_path)

    # Persist
    (out_dir / "fault_prob_sweep.json").write_text(
        json.dumps(all_rows, indent=2)
    )
    print(f"\n  Artifacts → {out_dir}")
