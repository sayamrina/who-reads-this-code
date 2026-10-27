"""Figures for the report (PNG in docs/img)."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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
    pv = b["surprisal_z"]["p"]
    ax.set_title(f"Less predictable code tokens are read longer: +{b['ms_per_bit_at_mean']} ms per bit "
                 f"({'p < .001' if pv < .001 else f'p = {pv:.3f}'})", fontsize=10, color=INK, loc="left")
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


def study3() -> None:
    p = RESULTS / "aoc" / "study3_scaling.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
    qwen = d[d["family"] == "Qwen2.5"].sort_values("size_b")
    other = d[d["family"] != "Qwen2.5"]
    ax.plot(qwen["size_b"], qwen["rho_confusing"], marker="o", color=BLUE, lw=2, label="Qwen2.5-Coder (0.5B to 32B)")
    ax.scatter(other["size_b"], other["rho_confusing"], color=MUTED, s=36, zorder=3, label="Other code models (published)")
    for _, r in other.iterrows():
        ax.annotate(r["model"].replace("-base", "").replace("deepseek-coder", "DeepSeek").replace("Artigenz-Coder-DS", "Artigenz"),
                    (r["size_b"], r["rho_confusing"]), textcoords="offset points",
                    xytext=(6, 4) if r["model"].startswith("deepseek-coder-6.7b") else (5, -10), fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 3, 10, 30], ["0.5B", "1B", "3B", "10B", "30B"])
    ax.set_xlabel("Model size (parameters, log scale)")
    ax.set_ylabel("Spearman ρ with the human\nconfusion response (confusing snippets)")
    ax.grid(axis="y", color=GRID)
    ax.set_title("Bigger code models do not track the brain’s confusion response better", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(IMG / "study3_scaling.png")


def study4() -> None:
    p = RESULTS / "aoc" / "study4.json"
    if not p.exists():
        return
    r = json.load(open(p))
    labels = ["Late frontal positivity\n(Fz, 400–800 ms)", "N400 window\n(Cz, CPz, Pz)", "P600 window\n(CPz, Pz, P3, P4)",
              "First-pass dwell\nin the AOI"]
    keys = ["H7_lfp_experience", "H8_n400", "H8_p600", "H9_first_pass"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=160, gridspec_kw={"width_ratios": [3, 1]})
    eeg = [(r[k]["coef"]["surprisal_z"]["b"], r[k]["coef"]["surprisal_z"]["se"], r[k]["coef"]["surprisal_z"]["p"]) for k in keys[:3]]
    ax = axes[0]
    for i, (b, se, pv) in enumerate(eeg):
        ax.errorbar(i, b, yerr=1.96 * se, fmt="o", color=BLUE if pv < .05 else MUTED, capsize=4, lw=2)
        ax.text(i + 0.08, b, f"p = {pv:.3f}" if pv >= .001 else "p < .001", fontsize=8, color=INK, va="center")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(3), labels[:3], fontsize=8)
    ax.set_xlim(-0.5, 2.7)
    ax.set_ylabel("µV per SD of AOI surprisal")
    ax.grid(axis="y", color=GRID)
    ax.set_title("Single-trial EEG (24 programmers)", fontsize=9.5, color=INK, loc="left")
    b, se, pv = (r[keys[3]]["coef"]["surprisal_z"][k] for k in ("b", "se", "p"))
    ax2 = axes[1]
    ax2.errorbar(0, (np.exp(b) - 1) * 100, yerr=[[ (np.exp(b) - np.exp(b - 1.96 * se)) * 100], [(np.exp(b + 1.96 * se) - np.exp(b)) * 100]],
                 fmt="o", color=BLUE, capsize=4, lw=2)
    ax2.axhline(0, color=MUTED, lw=1)
    ax2.set_xticks([0], [labels[3]], fontsize=8)
    ax2.set_ylabel("% longer per SD")
    ax2.grid(axis="y", color=GRID)
    ax2.set_title("Eye tracking", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(IMG / "study4_single_trial.png")


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    study1()
    study1b()
    study2()
    study3()
    study4()
    print("figures written to", IMG)
