"""Phase 2 figures. Colors follow each condition (fixed assignment, validated default
palette); every estimate carries a bootstrap CI; provenance footnote on every figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "metrics"))
from run_phase2 import CONDITIONS, N_SAMPLES, _matrices  # noqa: E402

OUT = ROOT / "results" / "phase2"
FIG = ROOT / "figures"
DATE = "2026-07-14"

COND_COLOR = {  # fixed entity->hue assignment (validated categorical palette, light mode)
    "vanilla": "#2a78d6",
    "dmp_empirical": "#1baf7a",
    "dmp_uniform": "#eda100",
    "shuffled_g0": "#008300",
    "random_persona": "#4a3aa7",
    "diversity_instruction": "#e34948",
}
COND_LABEL = {
    "vanilla": "vanilla",
    "dmp_empirical": "DMP (human-fitted values)",
    "dmp_uniform": "DMP (uniform values)",
    "shuffled_g0": "DMP (shuffled values)",
    "random_persona": "random hobby personas",
    "diversity_instruction": "“voice one opinion” instruction",
}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
PROV = ("data: results/phase2/judgments.jsonl (judge=claude-haiku-4-5, n=150 dilemmas x 16 "
        "samples x 6 conditions, T=1) | humans: Scruples verdict counts (>=40/item) | "
        f"script: pipeline/plot_phase2.py | {DATE}")


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, axis="y", color="#e6e5e0", lw=0.7, zorder=0)


def boot_stats():
    pids, mats, p_h, n_bin, bucket = _matrices()
    rng = np.random.default_rng(20260714)
    B = 1000
    bi = rng.integers(0, len(pids), size=(B, len(pids)))
    stats = {}
    for c in CONDITIONS:
        pm = np.nanmean(mats[c], axis=1)
        corr = float(np.corrcoef(pm, p_h)[0, 1])
        bc = [float(np.corrcoef(pm[i], p_h[i])[0, 1]) for i in bi]
        wstd = float(np.nanstd(mats[c], axis=1).mean())
        bw = [float(np.nanstd(mats[c][i], axis=1).mean()) for i in bi]
        stats[c] = {"corr": corr, "corr_ci": np.percentile(bc, [2.5, 97.5]),
                    "wstd": wstd, "wstd_ci": np.percentile(bw, [2.5, 97.5])}
    return stats, (pids, mats, p_h, n_bin, bucket)


def fig_dissociation(stats):
    fig, ax = plt.subplots(figsize=(7.6, 5.2), facecolor=SURFACE)
    style_ax(ax)
    ax.grid(True, axis="x", color="#e6e5e0", lw=0.7, zorder=0)
    for c in CONDITIONS:
        s = stats[c]
        ax.errorbar(s["wstd"], s["corr"],
                    xerr=[[s["wstd"] - s["wstd_ci"][0]], [s["wstd_ci"][1] - s["wstd"]]],
                    yerr=[[s["corr"] - s["corr_ci"][0]], [s["corr_ci"][1] - s["corr"]]],
                    fmt="o", ms=9, color=COND_COLOR[c], ecolor=COND_COLOR[c],
                    elinewidth=1.2, capsize=3, zorder=3)
    # direct labels, nudged to avoid collisions (checked visually after each render)
    off = {"vanilla": (0, 12, "center"), "dmp_empirical": (-10, -22, "right"),
           "dmp_uniform": (10, -26, "left"), "shuffled_g0": (0, 14, "center"),
           "random_persona": (0, 12, "center"), "diversity_instruction": (10, -4, "left")}
    for c in CONDITIONS:
        s = stats[c]
        dx, dy, ha = off[c]
        ax.annotate(COND_LABEL[c], (s["wstd"], s["corr"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, color=INK, ha=ha,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    ax.axhline(stats["vanilla"]["corr"], color="#c3c2b7", lw=1, ls="--", zorder=1)
    ax.text(0.001, stats["vanilla"]["corr"] - 0.012, "vanilla alignment", fontsize=8,
            color=INK2, ha="left")
    ax.set_xlabel("judgment diversity within a dilemma (std of 16 samples)  → “spread recovered”",
                  fontsize=10, color=INK)
    ax.set_ylabel("agreement with humans (corr of per-item means)  → “direction”",
                  fontsize=10, color=INK)
    ax.set_title("Conditioning adds diversity but does not move judgment toward humans",
                 fontsize=12, color=INK, pad=14)
    ax.set_xlim(0, 0.105)
    fig.text(0.01, 0.005, PROV, fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = FIG / f"phase2_spread-vs-alignment_6cond_haiku_n150_{DATE}.png"
    fig.savefig(p, dpi=160)
    print(p)


def fig_corr_bars(stats):
    fig, ax = plt.subplots(figsize=(7.6, 4.4), facecolor=SURFACE)
    style_ax(ax)
    xs = np.arange(len(CONDITIONS))
    for i, c in enumerate(CONDITIONS):
        s = stats[c]
        ax.bar(i, s["corr"], width=0.62, color=COND_COLOR[c], zorder=3)
        ax.errorbar(i, s["corr"], yerr=[[s["corr"] - s["corr_ci"][0]],
                                        [s["corr_ci"][1] - s["corr"]]],
                    fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=4)
        ax.text(i, 0.015, COND_LABEL[c], rotation=90, ha="center", va="bottom",
                fontsize=8.5, color="#ffffff" if c != "dmp_uniform" else INK, zorder=5)
    ax.set_xticks([])
    ax.set_ylabel("correlation with human verdict rates", fontsize=10, color=INK)
    ax.set_title("Human alignment barely differs across conditions (95% bootstrap CIs)",
                 fontsize=12, color=INK, pad=12)
    ax.set_ylim(0, 0.62)
    fig.text(0.01, 0.005, PROV, fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = FIG / f"phase2_corr-with-humans_6cond_bootstrapCI_haiku_n150_{DATE}.png"
    fig.savefig(p, dpi=160)
    print(p)


def fig_buckets(data):
    from geometry import noise_floor_absdiff
    pids, mats, p_h, n_bin, bucket = data
    buckets = sorted(set(bucket))
    floor = np.array([noise_floor_absdiff(p, n, N_SAMPLES) for p, n in zip(p_h, n_bin)])
    fig, ax = plt.subplots(figsize=(7.6, 4.6), facecolor=SURFACE)
    style_ax(ax)
    width = 0.34
    xs = np.arange(len(buckets))
    for k, c in enumerate(["vanilla", "dmp_empirical"]):
        pm = np.nanmean(mats[c], axis=1)
        means, err = [], []
        for b in buckets:
            d = np.abs(pm - p_h)[bucket == b]
            means.append(d.mean())
            err.append(1.96 * d.std() / np.sqrt(len(d)))
        ax.bar(xs + (k - 0.5) * width, means, width=width * 0.94, color=COND_COLOR[c],
               zorder=3, label=COND_LABEL[c])
        ax.errorbar(xs + (k - 0.5) * width, means, yerr=err, fmt="none", ecolor=INK,
                    elinewidth=1, capsize=2.5, zorder=4)
    fl = [floor[bucket == b].mean() for b in buckets]
    ax.plot(xs, fl, ls="--", color=INK2, lw=1.4, marker="_", ms=14, zorder=5,
            label="noise floor (perfect alignment)")
    ax.set_xticks(xs, [f"{b}\n(contested)" if b == "0.5-0.6" else b for b in buckets],
                  fontsize=9)
    ax.set_xlabel("human consensus bucket", fontsize=10, color=INK)
    ax.set_ylabel("mean |model P − human P| (unacceptable)", fontsize=10, color=INK)
    ax.set_title("Gap to human judgment distributions, by how contested the dilemma is",
                 fontsize=12, color=INK, pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(0.01, 0.005, PROV, fontsize=6, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = FIG / f"phase2_absdiff-by-consensus_vanilla-vs-dmp_floor_haiku_n150_{DATE}.png"
    fig.savefig(p, dpi=160)
    print(p)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    stats, data = boot_stats()
    fig_dissociation(stats)
    fig_corr_bars(stats)
    fig_buckets(data)
