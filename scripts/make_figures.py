"""Model-comparison figures, paired NDCG differences, and the pipeline diagram."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from recsys.data import ROOT  # noqa: E402
from recsys.plotting import AXIS, BLUE, INK, INK2, MUTED, apply_style, save  # noqa: E402

TABLES = ROOT / "reports" / "tables"


def paired_differences() -> None:
    """Paired bootstrap CI of per-user NDCG@10 differences between models."""
    pu = pd.read_parquet(TABLES / "per_user_test.parquet")
    pv = pu.pivot_table(index="userId", columns="family", values="NDCG@10")
    rng = np.random.default_rng(42)
    rows = []
    for a, b in [("uknn", "implicitmf"), ("implicitmf", "pop"), ("uknn", "pop"), ("implicitmf", "iknn_imp"), ("implicitmf", "biasedmf")]:
        d = (pv[a] - pv[b]).dropna().to_numpy()
        boot = d[rng.integers(0, len(d), (2000, len(d)))].mean(axis=1)
        rows.append({"a": a, "b": b, "users": len(d), "mean_diff": d.mean(), "lo": np.quantile(boot, 0.025),
                     "hi": np.quantile(boot, 0.975), "rel": d.mean() / pv[b].mean()})
    pd.DataFrame(rows).to_csv(TABLES / "paired_ndcg_differences.csv", index=False)


def model_figures() -> None:
    comp = pd.read_csv(TABLES / "model_comparison.csv").sort_values("ndcg")
    best = comp["ndcg"].idxmax()

    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(comp))
    ax.barh(y, comp["ndcg"], color=[BLUE if i == best else "#9ec5f4" for i in comp.index], height=0.62)
    ax.errorbar(comp["ndcg"], y, xerr=[comp["ndcg"] - comp["ndcg_lo"], comp["ndcg_hi"] - comp["ndcg"]], fmt="none", ecolor=INK2, capsize=3, lw=1)
    for yi, v, hi in zip(y, comp["ndcg"], comp["ndcg_hi"]):
        ax.text(hi + comp["ndcg"].max() * 0.02, yi, f"{v:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, comp["model"])
    ax.set_xlabel("NDCG@10 on 5,000 test users (95% bootstrap CI)")
    ax.set_title("Top-10 ranking accuracy by model")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, comp["ndcg_hi"].max() * 1.18)
    save(fig, "model_ndcg.png")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(comp["avg_pop"] * 100, comp["ndcg"], s=60, color=BLUE, zorder=3)
    for _, r in comp.iterrows():
        ax.annotate(r["model"], (r["avg_pop"] * 100, r["ndcg"]), textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=INK2)
    ax.set_xlabel("Avg. popularity of recommended movies (% of users who rated them)")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Accuracy vs. popularity bias")
    save(fig, "model_tradeoff.png")


def pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 3.1))
    ax.set_xlim(0, 13)
    ax.set_ylim(0.05, 2.9)
    ax.axis("off")
    boxes = [
        (0.1, "MovieLens 32M", "32.0M ratings\n2.0M tags · 87.6K movies", "#f0efec"),
        (2.25, "Clean & profile", "Parquet · EDA\nonboarding-burst filter", "#f0efec"),
        (4.4, "Train & compare", "LensKit: popularity, bias,\nk-NN, matrix factorization", "#e6f0fc"),
        (6.55, "Score new user", "fold-in from ≥5 ratings\n(no retraining)", "#e6f0fc"),
        (8.7, "When to watch", "holiday tags · recurring\nseasons · weekend rhythm", "#fdf1d6"),
        (10.85, "CineCompass UI", "cards · 'because you liked'\nfilters · movie calendar", "#e3f5ee"),
    ]
    for x, title, sub, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.9), 1.95, 1.9, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=AXIS, lw=1))
        ax.text(x + 0.975, 2.35, title, ha="center", va="center", fontsize=11.5, weight="bold", color=INK)
        ax.text(x + 0.975, 1.5, sub, ha="center", va="center", fontsize=9, color=INK2, linespacing=1.4)
    for x, _, _, _ in boxes[:-1]:
        ax.annotate("", xy=(x + 2.2, 1.85), xytext=(x + 1.97, 1.85), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
    ax.text(5.4, 0.5, "Offline test: last-5 temporal holdout\nNDCG@10 · recall · coverage · popularity · RMSE",
            ha="center", va="center", fontsize=9, color=INK2, linespacing=1.4)
    ax.text(9.675, 0.5, "Out-of-time check\n(fit ≤2018 → test 2019–23)", ha="center", va="center", fontsize=9, color=INK2, linespacing=1.4)
    save(fig, "pipeline_diagram.png")


if __name__ == "__main__":
    apply_style()
    pipeline_diagram()
    if (TABLES / "model_comparison.csv").exists():
        model_figures()
        paired_differences()
    print("figures done")
