"""Generate 300-DPI publication figures (PNG + PDF) for the SINAN LLM measurement study."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from src.analysis.accuracy import MODEL_SHORT

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_fig(fig: plt.Figure, out_dir: Path, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {stem}.png and {stem}.pdf")


def plot_fig1_accuracy_by_regime_task(tables_dir: Path, figures_dir: Path, models: list[str]):
    df = pd.read_csv(tables_dir / "accuracy_by_model_regime_task.csv")
    tasks = ["lookup", "aggregation", "comparison", "abstention"]
    regimes = ["N1", "N2", "N3"]
    regime_labels = {"N1": "N1 (Zero-Shot)", "N2": "N2 (Tool 1-Shot)", "N3": "N3 (Self-Repair)"}
    colors = {"N1": "#4c72b0", "N2": "#dd8452", "N3": "#55a868"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.flatten()
    x = np.arange(len(models))
    width = 0.25

    for ax, task in zip(axes, tasks):
        sub = df[df["task_type"] == task]
        for idx, reg in enumerate(regimes):
            sub_r = sub[sub["regime"] == reg].set_index("model")
            means = [sub_r.loc[m, "accuracy"] * 100 if m in sub_r.index else 0.0 for m in models]
            lo = [max(0.0, (sub_r.loc[m, "accuracy"] - sub_r.loc[m, "ci_low"]) * 100) if m in sub_r.index else 0.0 for m in models]
            hi = [max(0.0, (sub_r.loc[m, "ci_high"] - sub_r.loc[m, "accuracy"]) * 100) if m in sub_r.index else 0.0 for m in models]
            ax.bar(
                x + (idx - 1) * width,
                means,
                width=width,
                label=regime_labels[reg],
                color=colors[reg],
                yerr=[lo, hi],
                capsize=2.5,
                error_kw={"elinewidth": 1.0, "ecolor": "#333333"},
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_title(f"Task: {task.capitalize()}", fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], rotation=28, ha="right", fontsize=9.5)
        ax.set_ylim(0, 108)
        ax.set_ylabel("Accuracy (%)" if ax in (axes[0], axes[2]) else "")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Accuracy with 95% Bootstrap CI by Model, Regime, and Task Type", fontsize=14, fontweight="bold", y=1.06)
    plt.tight_layout()
    save_fig(fig, figures_dir, "fig1_accuracy_by_regime_task")


def plot_fig2_error_taxonomy(tables_dir: Path, figures_dir: Path, models: list[str]):
    df = pd.read_csv(tables_dir / "error_taxonomy.csv")
    df = df[df["model"].isin(models) & df["regime"].isin(["N2", "N3"])].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.34

    cats = [
        ("correct_pct", "Correct", "#55a868"),
        ("noisy_error_pct", "Noisy Error (Exec/Timeout)", "#c44e52"),
        ("silent_error_pct", "Silent Error (Wrong Output)", "#dd8452"),
        ("parse_or_empty_error_pct", "Parse / Empty Error", "#8172b3"),
    ]

    for r_idx, (reg, offset, hatch) in enumerate([("N2", -width / 2 - 0.02, ""), ("N3", width / 2 + 0.02, "//")]):
        sub = df[df["regime"] == reg].set_index("model")
        bottom = np.zeros(len(models))
        for col, label, color in cats:
            vals = np.array([sub.loc[m, col] if m in sub.index else 0.0 for m in models])
            ax.bar(
                x + offset,
                vals,
                width=width,
                bottom=bottom,
                color=color,
                edgecolor="white" if not hatch else "#222222",
                linewidth=0.6,
                hatch=hatch,
                label=label if r_idx == 0 else None,
            )
            bottom += vals
        # Annotate N2 / N3 above each bar
        for i in range(len(models)):
            ax.text(x[i] + offset, 101.5, reg, ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT.get(m, m) for m in models], rotation=22, ha="right", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Share of Trials (%)", fontsize=11)
    ax.set_title("Tool-Use Error Taxonomy: N2 (1-Shot) vs. N3 (Self-Repair)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=10)
    plt.tight_layout()
    save_fig(fig, figures_dir, "fig2_error_taxonomy")


def plot_fig3_gtheory_dstudy(tables_dir: Path, figures_dir: Path):
    df_vc = pd.read_csv(tables_dir / "gtheory.csv")
    df_ds = pd.read_csv(tables_dir / "gtheory_dstudy.csv")

    ov = df_vc[df_vc["scope"] == "Overall"].iloc[0]
    comps = [
        ("model", "Model (m)"),
        ("item", "Item (i)"),
        ("regime", "Regime (r)"),
        ("format", "Format (f)"),
        ("model_x_item", "Model x Item"),
        ("model_x_regime", "Model x Regime"),
        ("residual", "Residual (e)"),
    ]
    tot = ov["total"] if ov["total"] > 0 else 1.0
    labels = [lbl for _, lbl in comps]
    pcts = [100.0 * float(ov.get(k, 0.0)) / tot for k, _ in comps]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # Panel (a): Variance decomposition
    y_pos = np.arange(len(labels))
    palette = ["#4c72b0", "#dd8452", "#55a868", "#8172b3", "#c44e52", "#937860", "#8c8c8c"]
    bars = ax1.barh(y_pos, pcts, color=palette, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Proportion of Total Variance (%)", fontsize=10.5)
    ax1.set_title("(a) G-Study Variance Decomposition", fontweight="bold", fontsize=12)
    ax1.set_xlim(0, max(pcts + [10]) * 1.18)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, p in zip(bars, pcts):
        ax1.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2, f"{p:.1f}%", va="center", fontsize=9.5)

    # Panel (b): D-study curve
    scope_styles = {
        "Overall": ("#222222", "-", "o", 2.4),
        "N1": ("#4c72b0", "--", "s", 1.8),
        "N2": ("#dd8452", "-.", "^", 1.8),
        "N3": ("#55a868", ":", "D", 1.8),
    }
    for scope, (color, ls, marker, lw) in scope_styles.items():
        sub = df_ds[df_ds["scope"] == scope].sort_values("n_items")
        if sub.empty:
            continue
        ax2.plot(sub["n_items"], sub["rho"], color=color, linestyle=ls, marker=marker, linewidth=lw, markersize=5.5, label=scope)

    ax2.axhline(0.80, color="#888888", linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.text(101, 0.80, "0.80", va="center", fontsize=8.5, color="#666666")
    ax2.axhline(0.90, color="#888888", linestyle=":", linewidth=1.0, alpha=0.7)
    ax2.text(101, 0.90, "0.90", va="center", fontsize=8.5, color="#666666")
    ax2.axvline(36, color="#c44e52", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.text(37.5, 0.08, "Current\n(n=36)", fontsize=8.5, color="#c44e52")

    ax2.set_xlabel("Number of Benchmark Items (n)", fontsize=10.5)
    ax2.set_ylabel(r"Generalizability Coefficient ($E\rho^2$)", fontsize=10.5)
    ax2.set_title("(b) D-Study Reliability Curve", fontweight="bold", fontsize=12)
    ax2.set_xlim(0, 108)
    ax2.set_ylim(0, 1.03)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    save_fig(fig, figures_dir, "fig3_gtheory_dstudy")


def plot_fig4_cpu_pareto_frontier(tables_dir: Path, figures_dir: Path):
    cpu_csv = tables_dir / "cpu_bench.csv"
    if not cpu_csv.exists():
        logger.warning(f"{cpu_csv} not found; skipping fig4_cpu_pareto_frontier")
        return
    df = pd.read_csv(cpu_csv)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6.2))
    model_colors = {
        "qwen2.5:0.5b": "#4c72b0",
        "qwen2.5:1.5b": "#55a868",
        "qwen2.5:3b": "#dd8452",
        "qwen2.5:7b": "#c44e52",
    }
    regime_markers = {"N1": "o", "N2": "s", "N3": "^"}

    # Scale marker size proportionally to peak RAM (MB)
    min_ram = df["peak_ram_mb"].min()
    max_ram = max(df["peak_ram_mb"].max(), min_ram + 1.0)

    for _, r in df.iterrows():
        m = r["model"]
        reg = r["regime"]
        ram = r["peak_ram_mb"]
        size = 90 + 380 * (ram - min_ram) / (max_ram - min_ram)
        ax.scatter(
            r["mean_latency_s"],
            r["accuracy"] * 100,
            s=size,
            color=model_colors.get(m, "#333333"),
            marker=regime_markers.get(reg, "o"),
            alpha=0.85,
            edgecolors="#222222",
            linewidth=0.9,
            zorder=4,
        )
        short = MODEL_SHORT.get(m, m).replace("Qwen2.5-", "")
        ax.annotate(
            f"{short} {reg}",
            (r["mean_latency_s"], r["accuracy"] * 100),
            textcoords="offset points",
            xytext=(7, 5),
            fontsize=8.5,
            color="#222222",
        )

    # Compute Pareto frontier (minimize mean_latency_s, maximize accuracy)
    pts = df.sort_values("mean_latency_s")
    frontier = []
    best_acc = -1.0
    for _, r in pts.iterrows():
        if r["accuracy"] > best_acc:
            frontier.append((r["mean_latency_s"], r["accuracy"] * 100))
            best_acc = r["accuracy"]
    if len(frontier) >= 2:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, linestyle="--", color="#222222", linewidth=1.6, alpha=0.75, label="Pareto Frontier", zorder=3)

    # Custom legends for models, regimes, and RAM size
    from matplotlib.lines import Line2D
    model_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=MODEL_SHORT.get(m, m))
        for m, c in model_colors.items()
    ]
    regime_handles = [
        Line2D([0], [0], marker=mk, color="w", markerfacecolor="#666666", markeredgecolor="#222222", markersize=8.5, label=reg)
        for reg, mk in regime_markers.items()
    ]
    if len(frontier) >= 2:
        regime_handles.append(Line2D([0], [0], linestyle="--", color="#222222", linewidth=1.6, label="Pareto Frontier"))

    leg1 = ax.legend(handles=model_handles, title="Model (Color)", loc="lower right", frameon=True, fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=regime_handles, title="Regime (Shape)", loc="upper left", frameon=True, fontsize=9)

    lat_ratio = df["mean_latency_s"].max() / max(df["mean_latency_s"].min(), 1e-3)
    if lat_ratio > 8:
        ax.set_xscale("log")
        ax.set_xlabel("Mean Latency per Query (s, log scale)", fontsize=11)
    else:
        ax.set_xlabel("Mean Latency per Query (s)", fontsize=11)

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(-5, 108)
    ax.set_title("CPU-Only Pareto Frontier: Accuracy vs. Latency (Marker Size = Peak RAM)", fontsize=12.5, fontweight="bold", pad=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    save_fig(fig, figures_dir, "fig4_cpu_pareto_frontier")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables_dir = Path(cfg["outputs"]["tables_dir"])
    figures_dir = Path(cfg["outputs"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    models = cfg["llm"]["models"]

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

    plot_fig1_accuracy_by_regime_task(tables_dir, figures_dir, models)
    plot_fig2_error_taxonomy(tables_dir, figures_dir, models)
    plot_fig3_gtheory_dstudy(tables_dir, figures_dir)
    plot_fig4_cpu_pareto_frontier(tables_dir, figures_dir)


if __name__ == "__main__":
    main()
