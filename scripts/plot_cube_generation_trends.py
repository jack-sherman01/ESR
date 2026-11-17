"""Render candidate and active-policy metrics from the shared blind comparison panel."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="output filename stem")
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted((args.run / "trend").glob("[0-9]*.json"))]
    if not rows:
        parser.error("no paired trend measurements in the run")
    x = [row["generation"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.3), layout="constrained")
    styles = {"candidate": {"color": "#0072B2", "marker": "o", "linestyle": "--", "markerfacecolor": "white",
                            "markersize": 8, "zorder": 3},
              "active": {"color": "#D55E00", "marker": "s", "linestyle": "-", "alpha": .92, "markersize": 5, "zorder": 4}}

    def get(row, role, field, scale=1):
        value = (row.get(role) or {}).get(field)
        return float("nan") if value is None else value * scale

    def risk(row, role):
        result = row.get(role) or {}
        counts = result.get("counts", {})
        if result.get("measured") != result.get("planned") or counts.get("UNKNOWN", 0):
            return float("nan")
        return counts.get("MINOR", 0) + counts.get("SEVERE", 0)

    for role, style in styles.items():
        name = "Sol candidate" if role == "candidate" else "Active policy"
        axes[0, 0].plot(x, [get(r, role, "success_fraction", 100) for r in rows], label=name, **style)
        axes[0, 1].plot(x, [get(r, role, "known_success_mean_steps")
                            if (r.get(role) or {}).get("success_fraction") is not None
                            else float("nan") for r in rows], **style)
        axes[1, 0].plot(x, [risk(r, role) for r in rows], **style)
    axes[0, 0].set(title="Task success on shared cases", ylabel="Success rate (%)", ylim=(-4, 105))
    axes[0, 0].legend(loc="lower right", frameon=False)
    axes[0, 1].set(title="Control cost on successful shared cases", ylabel="Mean control steps")
    measured_steps = [get(r, k, "known_success_mean_steps") for r in rows for k in styles]
    measured_steps = [v for v in measured_steps if math.isfinite(v)]
    if measured_steps:
        axes[0, 1].set_ylim(max(0, min(measured_steps) - 6), max(measured_steps) + 6)
        axes[0, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

    axes[1, 0].set(title="Observed violations on shared cases", ylabel="Minor + severe, out of 4", ylim=(-.15, 4.25))
    axes[1, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    for index, row in zip(x, rows):
        for role, name, height in (("candidate", "Sol", 3.9), ("active", "Active", 3.5)):
            metric = row.get(role)
            if metric is not None and (index != 0 or role == "active") and not math.isfinite(risk(row, role)):
                unknown = metric["counts"]["UNKNOWN"] + metric["planned"] - metric["measured"]
                axes[1, 0].text(index, height, f"{name} ?{unknown}", fontsize=8, ha="center",
                                color=styles[role]["color"])

    panel = axes[1, 1]
    colors = ["#009E73", "#E69F00", "#A51C30", "#8359A4", "#D9D9D9"]
    captions = ["Pass", "Minor", "Severe", "Unknown", "Not evaluated"]
    generations = rows[1:]
    gx = [r["generation"] for r in generations]
    above = [0] * len(gx)
    if generations:
        for label, color in zip(captions, colors):
            vals = [r["decision"].get("fresh_not_evaluated", 30) if label == "Not evaluated"
                    else r["decision"].get("fresh_counts", {}).get(label.upper(), 0) for r in generations]
            panel.bar(gx, vals, bottom=above, color=color, label=label, width=.58,
                      edgecolor="white", hatch="//" if label == "Not evaluated" else None)
            above = [a + b for a, b in zip(above, vals)]
        for r in generations:
            panel.text(r["generation"], 30.8, "Promoted" if r["promoted"] else "Rejected",
                       ha="center", va="bottom", fontsize=8)
        panel.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3, frameon=False, fontsize=8)
    panel.set(title="Independent risk tests for each candidate", ylabel="Episodes, out of 30", ylim=(0, 35))
    for ax in axes.flat:
        ax.set_xticks(x, labels=["Base" if j == 0 else str(j) for j in x])
        ax.set_xlabel("Generation")
        ax.set_xlim(min(x) - .35, max(x) + .35)
        ax.grid(axis="y", color=".9")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    if gx:
        panel.set_xticks(gx)
        panel.set_xlim(min(gx) - .55, max(gx) + .55)
    fig.suptitle("ASPIRE cube lifting — candidate and active policy", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=200)
    fig.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
