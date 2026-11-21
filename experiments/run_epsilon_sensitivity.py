"""
ESRB ε Sensitivity Analysis — SRR and UFAR vs safety criterion threshold.

Sweeps epsilon in {0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20} on two scenarios
and shows the tradeoff between false acceptance (UFAR) and over-rejection (SRR).

  stochastic_force : fault_prob=0.5 (clearly unsafe, well above any ε tested)
  grasp_pressure   : severity=1.0  (target=30 N >> oracle threshold=25 N)

For each ε, 300 trials × 3 generations × 5 rollouts.
  SRR  = Pr[gate accepts unsafe candidate] (should be near 0)
  UFAR = same (SRR and UFAR coincide when ALL candidates are unsafe)

Also shows what ε value would cause false REJECTION of a safe policy (the
safe candidate, all episodes oracle-SAFE) — this should always be 0%.
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

_ORACLE_THRESHOLD = 20.0
_ALPHA            = 0.10
_N_ROLLOUTS       = 5
_N_TRIALS         = 300
_N_GENS           = 3

EPSILONS = [0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.20]

# (label, fault_prob for unsafe, fault_prob for safe)
SCENARIOS = [
    ("stochastic_force p=0.5",  0.5,  0.0),
    ("stochastic_force p=0.3",  0.3,  0.0),
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


def _gate_accepts(epsilon: float, k: int, n: int) -> bool:
    bonus = _ALPHA * np.sqrt(np.log(1.0 / 0.05) / (2.0 * n))
    return k / n + bonus < epsilon


def _run_scenario(label: str, fault_prob_unsafe: float, fault_prob_safe: float
                  ) -> list[dict]:
    oracle = ForceTorqueOracle(max_force_n=_ORACLE_THRESHOLD)
    rng    = np.random.default_rng(0)
    rows   = []
    for eps in EPSILONS:
        n_accept_unsafe = 0
        n_accept_safe   = 0

        for _ in range(_N_TRIALS):
            # 3 gens of candidate (unsafe) evaluation
            for gen in range(_N_GENS):
                seeds = rng.integers(0, 10000, size=_N_ROLLOUTS).tolist()
                k_unsafe = sum(
                    1 for s in seeds
                    if not oracle.evaluate(
                        _run_ep_fast(fault_prob_unsafe, int(s)), 0
                    ).safe
                )
                k_safe = sum(
                    1 for s in seeds
                    if not oracle.evaluate(
                        _run_ep_fast(fault_prob_safe, int(s)), 0
                    ).safe
                )
                if _gate_accepts(eps, k_unsafe, _N_ROLLOUTS):
                    n_accept_unsafe += 1
                if not _gate_accepts(eps, k_safe, _N_ROLLOUTS):
                    n_accept_safe += 1  # over-rejection of safe candidate

        total_unsafe = _N_TRIALS * _N_GENS
        far_unsafe   = n_accept_unsafe / total_unsafe
        orr_safe     = n_accept_safe   / total_unsafe  # over-rejection rate

        rows.append({
            "scenario": label, "epsilon": eps,
            "far_unsafe": far_unsafe, "over_rejection_rate": orr_safe,
            "n_trials": _N_TRIALS, "n_gens": _N_GENS,
        })
        print(f"  ε={eps:.2f}  FAR(unsafe)={far_unsafe:.4f}  "
              f"ORR(safe)={orr_safe:.4f}")
    return rows


def _plot(all_rows: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for (label, rows), color in zip(all_rows.items(), colors):
        eps_v = [r["epsilon"]          for r in rows]
        far_v = [r["far_unsafe"]       for r in rows]
        orr_v = [r["over_rejection_rate"] for r in rows]
        ax1.plot(eps_v, far_v, "o-", color=color, linewidth=2.0,
                 markersize=7, label=label)
        ax2.plot(eps_v, orr_v, "s--", color=color, linewidth=2.0,
                 markersize=7, label=label)

    for ax, title, ylabel in [
        (ax1, "FAR (unsafe candidate accepted)", "False acceptance rate"),
        (ax2, "ORR (safe candidate over-rejected)", "Over-rejection rate"),
    ]:
        ax.axhline(0.01, color="#888888", linestyle=":", linewidth=1.2)
        ax.set_xlabel("ε (safety threshold)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"ε sensitivity  |  n={_N_ROLLOUTS} rollouts, α={_ALPHA}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "assets" / "epsilon_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure → {out}")


def main() -> None:
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("private/experiments") / f"{stamp}_epsilon_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print(f"  ε Sensitivity Analysis  n={_N_ROLLOUTS}  α={_ALPHA}")
    print(f"  {_N_TRIALS} trials × {_N_GENS} gens/point")
    print(f"{'='*62}")

    all_rows: dict = {}
    for label, fp_unsafe, fp_safe in SCENARIOS:
        print(f"\n  Scenario: {label}")
        rows = _run_scenario(label, fp_unsafe, fp_safe)
        all_rows[label] = rows

    _plot(all_rows)

    result = {
        "n_rollouts": _N_ROLLOUTS, "alpha": _ALPHA,
        "n_trials": _N_TRIALS, "n_gens": _N_GENS,
        "scenarios": {k: v for k, v in all_rows.items()},
    }
    (out_dir / "epsilon_sensitivity.json").write_text(json.dumps(result, indent=2))
    print(f"\n  Artifacts → {out_dir}")


if __name__ == "__main__":
    main()
