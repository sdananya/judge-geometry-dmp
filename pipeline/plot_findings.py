"""One figure per novel finding (findings 1, 2, 4, 5; finding 3 already has
phase2_top10mass-vs-injected-prior). Same validated palette + provenance conventions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "metrics"))
from geometry import noise_floor_absdiff  # noqa: E402

OUT = ROOT / "results" / "phase2"
FIG = ROOT / "figures"
DATE = "2026-07-14"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED = ("#2a78d6", "#1baf7a", "#eda100", "#008300",
                                          "#4a3aa7", "#e34948")
PROV = "data: results/phase2/summary*.json | judge=claude-haiku unless labeled | n=150 dilemmas x 16 samples | 2026-07-14"


def ax_style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)


S = json.loads((OUT / "summary.json").read_text())


# ---- Finding 1: content-irrelevance --------------------------------------
def fig1():
    conds = ["dmp_empirical", "dmp_uniform", "shuffled_g0"]
    labels = ["human-fitted\nvalues", "uniform\nvalues", "scrambled\nvalues"]
    colors = [AQUA, YELLOW, GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0), facecolor=SURFACE)
    for ax, key, title, van in zip(
            axes, ["corr_with_human", "within_item_std"],
            ["agreement with humans", "judgment diversity (within-item std)"],
            [S["conditions"]["vanilla"]["corr_with_human"],
             S["conditions"]["vanilla"]["within_item_std"]]):
        ax_style(ax)
        ax.grid(True, axis="y", color="#e6e5e0", lw=0.7, zorder=0)
        vals = [S["conditions"][c][key] for c in conds]
        ax.bar(range(3), vals, width=0.6, color=colors, zorder=3)
        ax.axhline(van, color=INK2, lw=1.2, ls="--", zorder=4)
        ax.text(2.45, van, " vanilla", fontsize=8, color=INK2, va="center")
        ax.set_xticks(range(3), labels, fontsize=9)
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xlim(-0.6, 2.9)
    axes[0].set_ylim(0, 0.6)
    fig.suptitle("What's IN the profile doesn't matter — only that a profile exists",
                 fontsize=12.5, color=INK)
    fig.text(0.01, 0.005, PROV, fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    p = FIG / f"finding1_content-irrelevance_3dmp-variants_haiku_n150_{DATE}.png"
    fig.savefig(p, dpi=160)
    print(p)


# ---- Finding 2: forest plot of paired deltas ------------------------------
def fig2():
    keys = [("dmp_empirical_minus_vanilla", "DMP vs vanilla"),
            ("dmp_empirical_minus_shuffled_g0", "DMP vs scrambled values"),
            ("dmp_empirical_minus_diversity_instruction", "DMP vs 'voice one opinion'"),
            ("dmp_empirical_minus_random_persona", "DMP vs hobby personas")]
    fig, ax = plt.subplots(figsize=(7.8, 3.9), facecolor=SURFACE)
    ax_style(ax)
    ax.grid(True, axis="x", color="#e6e5e0", lw=0.7, zorder=0)
    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    for i, (k, lab) in enumerate(keys):
        d = S["bootstrap_deltas"][k]
        lo, hi = d["corr_ci95"]
        significant = lo > 0 or hi < 0
        col = RED if significant else BLUE
        ax.errorbar(d["corr_delta"], i, xerr=[[d["corr_delta"] - lo], [hi - d["corr_delta"]]],
                    fmt="o", ms=8, color=col, ecolor=col, elinewidth=1.6, capsize=4, zorder=3)
        ax.text(-0.155, i, lab, fontsize=9.5, ha="right", va="center", color=INK)
    ax.set_yticks([])
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.set_xlim(-0.15, 0.15)
    ax.set_xlabel("difference in agreement-with-humans\n(value profiles minus comparison), 95% CI",
                  fontsize=9.5, color=INK)
    ax.set_title("Value profiles never beat any control —\nand lose to hobby personas",
                  fontsize=11.5, color=INK, pad=10)
    ax.text(0.128, -0.55, "value profiles better →", fontsize=8, color=INK2, ha="right")
    ax.text(-0.128, -0.55, "← comparison better", fontsize=8, color=INK2, ha="left")
    fig.text(0.01, 0.005, PROV + " | paired item bootstrap B=1000", fontsize=6, color=INK2)
    fig.tight_layout(rect=(0.14, 0.03, 1, 1))
    p = FIG / f"finding2_paired-deltas-forest_dmp-vs-controls_haiku_n150_{DATE}.png"
    fig.savefig(p, dpi=160)
    print(p)


# ---- Finding 4: noise-floor diagram ---------------------------------------
def fig4():
    ns = np.arange(10, 210, 5)
    floor = [noise_floor_absdiff(0.5, int(n), int(n)) for n in ns]
    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE)
    ax_style(ax)
    ax.grid(True, color="#e6e5e0", lw=0.7, zorder=0)
    ax.plot(ns, floor, color=BLUE, lw=2.2, zorder=3)
    ax.fill_between(ns, 0, floor, color=BLUE, alpha=0.10, zorder=1)
    ax.text(120, 0.045, "IMPOSSIBLE REGION\n(even a perfect judge scores above the line)",
            fontsize=9, color=BLUE, ha="center")
    n_a = 32
    f_a = noise_floor_absdiff(0.5, n_a, n_a)
    ax.plot([n_a], [f_a], "o", ms=9, color=BLUE, zorder=4)
    ax.annotate(f"Paper A's protocol:\n~{n_a} judgments/side → floor ≈ {f_a:.2f}",
                (n_a, f_a), xytext=(52, 0.145), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.plot([n_a], [0.05], "X", ms=11, color=RED, zorder=4)
    ax.annotate("Paper A's reported gap on contested\ndilemmas (0.05) — below the floor",
                (n_a, 0.05), xytext=(75, 0.075), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.set_xlabel("number of judgments per side (humans and model)", fontsize=10, color=INK)
    ax.set_ylabel("smallest achievable |model P − human P|\non a 50/50 contested dilemma", fontsize=10, color=INK)
    ax.set_title("Finite votes put a floor under the alignment metric — Paper A's claim sits below it",
                  fontsize=11.5, color=INK, pad=12)
    ax.set_xlim(10, 205)
    ax.set_ylim(0, 0.19)
    fig.text(0.01, 0.005, "exact E|p̂_H − p̂_J| for two independent binomial estimates of the same p=0.5 "
             "(metrics/geometry.py noise_floor_absdiff, validated vs Monte Carlo) | 2026-07-14",
             fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = FIG / f"finding4_noise-floor-vs-n_paperA-claim_{DATE}.png"
    fig.savefig(p, dpi=160)
    print(p)


# ---- Finding 5: capability vs conditioning dumbbells ----------------------
def fig5():
    judges = [("gpt-4o-mini", json.loads((OUT / "summary_4omini.json").read_text())),
              ("Claude Haiku", {"vanilla": S["conditions"]["vanilla"],
                                "dmp_empirical": S["conditions"]["dmp_empirical"]}),
              ("Claude Sonnet", json.loads((OUT / "summary_sonnet.json").read_text()))]
    fig, ax = plt.subplots(figsize=(7.4, 4.2), facecolor=SURFACE)
    ax_style(ax)
    ax.grid(True, axis="y", color="#e6e5e0", lw=0.7, zorder=0)
    for i, (name, d) in enumerate(judges):
        v, m = d["vanilla"]["corr_with_human"], d["dmp_empirical"]["corr_with_human"]
        ax.plot([i, i], [v, m], color=INK2, lw=1.4, zorder=2)
        ax.plot([i], [v], "o", ms=10, color=BLUE, zorder=3)
        ax.plot([i], [m], "o", ms=10, color=AQUA, zorder=3)
        ax.annotate(f"{v:.2f}", (i, v), xytext=(14, -3), textcoords="offset points",
                    fontsize=8.5, color=BLUE)
        ax.annotate(f"{m:.2f}", (i, m), xytext=(14, -3), textcoords="offset points",
                    fontsize=8.5, color=AQUA)
    # capability arrow
    ax.annotate("", xy=(2.35, 0.552), xytext=(2.35, 0.345),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(2.42, 0.45, "capability:\n~0.19", fontsize=9, color=RED, va="center")
    ax.plot([], [], "o", ms=9, color=BLUE, label="vanilla")
    ax.plot([], [], "o", ms=9, color=AQUA, label="with value profiles (DMP)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xticks(range(3), [j[0] for j in judges], fontsize=10)
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylabel("agreement with humans (corr)", fontsize=10, color=INK)
    ax.set_title("Choosing a better judge moves alignment ~10× more than conditioning it",
                  fontsize=12, color=INK, pad=12)
    fig.text(0.01, 0.005, PROV + " | sonnet/4o-mini: results/phase2/summary_{sonnet,4omini}.json",
             fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = FIG / f"finding5_capability-vs-conditioning_3judges_n150_{DATE}.png"
    fig.savefig(p, dpi=160)
    print(p)


if __name__ == "__main__":
    fig1()
    fig2()
    fig4()
    fig5()
