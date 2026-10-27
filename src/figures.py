"""Figures for the report (PNG in docs/img)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, IMG = ROOT / "results", ROOT / "docs" / "img"
INK, MUTED, BLUE, ORANGE, GRID = "#1f2933", "#52606d", "#2a6fdb", "#d9822b", "#e4e7eb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def study1() -> None:
    bins = pd.read_csv(RESULTS / "study1_bins.csv")
    res = json.load(open(RESULTS / "study1.json"))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    for group, color in (("none to medium", BLUE), ("high expertise", ORANGE)):
        d = bins[bins["group"] == group].sort_values("bin")
        ax.plot(d["bin"], d["total_ms"], marker="o", color=color, lw=2, label=f"{group} (n = {int(d['n'].sum()):,} tokens)")
    ref = bins[bins["group"] == "none to medium"].sort_values("bin")
    ax.set_xticks(list(ref["bin"]), [f"Q{int(b) + 1}\n{m:.1f} bits" for b, m in zip(ref["bin"], ref["surprisal"], strict=True)])
    ax.set_xlabel("Token surprisal from Qwen2.5-Coder-0.5B (quintiles, mean bits)")
    ax.set_ylabel("Total reading time (ms)")
    ax.grid(axis="y", color=GRID)
    b = res["H1_total_time"]["coefficients"]
    ax.set_title(f"Less predictable code tokens are read longer: +{b['ms_per_bit_at_mean']} ms per bit (p < .001)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(IMG / "study1_surprisal.png")


def study1b() -> None:
    res = json.load(open(RESULTS / "study1b.json"))
    trials = pd.read_csv(RESULTS / "trials.csv").merge(pd.read_csv(RESULTS / "study1b_clusters.csv"), on=["participant", "stimulus"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    colors = [BLUE, ORANGE, "#3e9b6b", "#8e44ad"]
    for c in res["clusters"]:
        d = trials[trials["cluster"] == c["cluster"]]
        ax.scatter(d["next_line_rate"], d["line_regression_rate"], s=14, alpha=0.6, color=colors[c["cluster"]],
                   label=f"Type {c['cluster'] + 1}: {c['description']} (n = {c['n_trials']}, mean expertise {c['mean_expertise']})")
    ax.set_xlabel("Share of line changes that go to the next line")
    ax.set_ylabel("Share of line changes that go back")
    ax.grid(color=GRID)
    ax.set_title("Two reading strategies; the non-linear one is more common among experts", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.set_size_inches(6.4, 4.4)
    fig.tight_layout()
    fig.savefig(IMG / "study1b_types.png")


def study2() -> None:
    s = json.load(open(RESULTS / "study2.json"))["summary"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=160)
    labels = ["Lines that use the\nflipped construct", "All other lines"]
    x = range(2)
    w = 0.36
    p = [s["prompt"]["targeted_change_rate"], s["prompt"]["untargeted_change_rate"]]
    r = [s["rsa"]["targeted_change_rate"], s["rsa"]["untargeted_change_rate"]]
    ax.bar([i - w / 2 for i in x], p, w, color=MUTED, label="Profile in the prompt")
    ax.bar([i + w / 2 for i in x], r, w, color=BLUE, label="RSA-style selection")
    ax.axhline(s["prompt_noise_change_rate"], color=ORANGE, ls="--", lw=1.2)
    ax.text(-0.5, s["prompt_noise_change_rate"] + 0.015, "prompt run-to-run noise", color=ORANGE, fontsize=8, ha="left")
    for i, (a, b) in enumerate(zip(p, r, strict=True)):
        ax.text(i - w / 2, a + 0.015, f"{a:.0%}", ha="center", fontsize=8.5, color=INK)
        ax.text(i + w / 2, b + 0.015, f"{b:.0%}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Share of lines whose explanation changed")
    ax.set_ylim(0, 0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Flipping one construct to “known”: where does the explanation change?", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(IMG / "study2_faithfulness.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    study1()
    study1b()
    study2()
    print("figures written to", IMG)
