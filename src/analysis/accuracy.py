"""Compute accuracy with 95% bootstrap CIs and error taxonomy tables."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

MODEL_SHORT = {
    "qwen2.5:0.5b": "Qwen2.5-0.5B",
    "qwen2.5:1.5b": "Qwen2.5-1.5B",
    "qwen2.5:3b": "Qwen2.5-3B",
    "qwen2.5:7b": "Qwen2.5-7B",
    "qwen2.5-coder:7b-instruct-q4_K_M": "Qwen2.5-Coder-7B",
    "llama3.2:3b": "Llama3.2-3B",
    "mistral:7b": "Mistral-7B",
}


def load_config(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using percentile bootstrap."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    if n == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def grouped_accuracy_ci(df: pd.DataFrame, group_cols: list[str], n_resamples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rows = []
    for keys, grp in df.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        mean, lo, hi = bootstrap_ci(grp["correct"].to_numpy(), n_resamples=n_resamples, seed=seed)
        row = dict(zip(group_cols, keys))
        row.update({
            "n": int(len(grp)),
            "n_correct": int(grp["correct"].sum()),
            "accuracy": round(mean, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def generate_tab2_accuracy_regimes(df: pd.DataFrame, models: list[str], tables_dir: Path, n_resamples: int = 1000):
    """Generate Table 2: Accuracy (95% bootstrap CI) by model across N1-CSV, N1-MD, N2, N3."""
    slices = [
        ("N1_csv", lambda d: d[(d["regime"] == "N1") & (d["format"] == "csv")]),
        ("N1_md", lambda d: d[(d["regime"] == "N1") & (d["format"] == "markdown")]),
        ("N2", lambda d: d[d["regime"] == "N2"]),
        ("N3", lambda d: d[d["regime"] == "N3"]),
    ]

    records = []
    for m in models:
        sub_m = df[df["model"] == m]
        rec = {"model": m}
        for label, fn in slices:
            vals = fn(sub_m)["correct"].to_numpy()
            mean, lo, hi = bootstrap_ci(vals, n_resamples=n_resamples, seed=42)
            rec[f"{label}_acc"] = mean
            rec[f"{label}_lo"] = lo
            rec[f"{label}_hi"] = hi
        records.append(rec)

    # Overall row
    ov = {"model": "Overall"}
    for label, fn in slices:
        vals = fn(df)["correct"].to_numpy()
        mean, lo, hi = bootstrap_ci(vals, n_resamples=n_resamples, seed=42)
        ov[f"{label}_acc"] = mean
        ov[f"{label}_lo"] = lo
        ov[f"{label}_hi"] = hi
    records.append(ov)

    # Markdown
    md = [
        "| Model | N1 (CSV) | N1 (Markdown) | N2 (Tool 1-shot) | N3 (Self-Repair) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in records:
        name = f"**{r['model']}**" if r["model"] == "Overall" else f"`{r['model']}`"
        cells = []
        for label, _ in slices:
            cells.append(f"{r[f'{label}_acc']*100:.1f}% [{r[f'{label}_lo']*100:.1f}, {r[f'{label}_hi']*100:.1f}]")
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    (tables_dir / "tab2_accuracy_regimes.md").write_text("\n".join(md) + "\n")

    # LaTeX
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Accuracy (\%) with 95\% bootstrap confidence intervals (1,000 resamples) across models and evaluation regimes on the 36-item SINAN benchmark ($n=3$ repetitions per item).}",
        r"\label{tab:accuracy_regimes}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & N1 (CSV) & N1 (Markdown) & N2 (Tool 1-shot) & N3 (Self-Repair) \\",
        r"\midrule",
    ]
    for r in records:
        is_ov = r["model"] == "Overall"
        if is_ov:
            tex.append(r"\midrule")
            m_str = r"\textbf{Overall}"
        else:
            m_str = r"\texttt{" + r["model"].replace("_", r"\_") + "}"
        cells = []
        for label, _ in slices:
            cells.append(f"{r[f'{label}_acc']*100:.1f} [{r[f'{label}_lo']*100:.1f}, {r[f'{label}_hi']*100:.1f}]")
        tex.append(f"{m_str} & " + " & ".join(cells) + r" \\")
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    (tables_dir / "tab2_accuracy_regimes.tex").write_text("\n".join(tex) + "\n")


def generate_tab3_error_taxonomy(df: pd.DataFrame, models: list[str], tables_dir: Path) -> pd.DataFrame:
    """Generate Table 3: Error taxonomy counts and percentages by model and regime."""
    error_cols = [
        "correct",
        "noisy_error",
        "silent_error",
        "arithmetic_or_reasoning_error",
        "parse_or_empty_error",
    ]
    rows = []
    for m in models + ["Overall"]:
        sub_m = df if m == "Overall" else df[df["model"] == m]
        for reg in ["N1", "N2", "N3"]:
            sub = sub_m[sub_m["regime"] == reg]
            n = len(sub)
            if n == 0:
                continue
            counts = sub["error_class"].value_counts().to_dict()
            rec = {"model": m, "regime": reg, "n": n}
            for ec in error_cols:
                c = int(counts.get(ec, 0))
                rec[f"{ec}_n"] = c
                rec[f"{ec}_pct"] = round(100.0 * c / n, 1)
            rows.append(rec)
    df_err = pd.DataFrame(rows)
    df_err.to_csv(tables_dir / "error_taxonomy.csv", index=False)

    # Markdown
    md = [
        "| Model | Regime | N | Correct (%) | Noisy Err (%) | Silent Err (%) | Arith./Reason. Err (%) | Parse/Empty (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_err.iterrows():
        name = f"**{r['model']}**" if r["model"] == "Overall" else f"`{r['model']}`"
        md.append(
            f"| {name} | {r['regime']} | {int(r['n'])} | "
            f"{r['correct_pct']:.1f}% ({int(r['correct_n'])}) | "
            f"{r['noisy_error_pct']:.1f}% ({int(r['noisy_error_n'])}) | "
            f"{r['silent_error_pct']:.1f}% ({int(r['silent_error_n'])}) | "
            f"{r['arithmetic_or_reasoning_error_pct']:.1f}% ({int(r['arithmetic_or_reasoning_error_n'])}) | "
            f"{r['parse_or_empty_error_pct']:.1f}% ({int(r['parse_or_empty_error_n'])}) |"
        )
    (tables_dir / "tab3_error_taxonomy.md").write_text("\n".join(md) + "\n")

    # LaTeX
    tex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Error taxonomy distribution (\%) across models and regimes (N1: zero-shot, N2: 1-shot Pandas tool, N3: self-repair).}",
        r"\label{tab:error_taxonomy}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Regime & $N$ & Correct (\%) & Noisy (\%) & Silent (\%) & Arith./Reason. (\%) & Parse/Empty (\%) \\",
        r"\midrule",
    ]
    for idx, r in df_err.iterrows():
        if r["model"] == "Overall" and r["regime"] == "N1":
            tex.append(r"\midrule")
        m_str = r"\textbf{Overall}" if r["model"] == "Overall" else r"\texttt{" + r["model"].replace("_", r"\_") + "}"
        tex.append(
            f"{m_str} & {r['regime']} & {int(r['n'])} & "
            f"{r['correct_pct']:.1f} & {r['noisy_error_pct']:.1f} & {r['silent_error_pct']:.1f} & "
            f"{r['arithmetic_or_reasoning_error_pct']:.1f} & {r['parse_or_empty_error_pct']:.1f} \\\\"
        )
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    (tables_dir / "tab3_error_taxonomy.tex").write_text("\n".join(tex) + "\n")
    return df_err


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n-resamples", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    responses_path = Path(cfg["outputs"]["responses"])
    out_path = Path(cfg["outputs"]["accuracy"])
    tables_dir = Path(cfg["outputs"]["tables_dir"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(responses_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No records found in {responses_path}")

    df.to_csv(out_path, index=False)
    models = [m for m in cfg["llm"]["models"] if m in set(df["model"])]

    # Grouped summaries with 95% bootstrap CIs
    by_mr = grouped_accuracy_ci(df, ["model", "regime"], n_resamples=args.n_resamples)
    by_mrt = grouped_accuracy_ci(df, ["model", "regime", "task_type"], n_resamples=args.n_resamples)
    by_mrtab = grouped_accuracy_ci(df, ["model", "regime", "table_name"], n_resamples=args.n_resamples)
    by_mf = grouped_accuracy_ci(df[df["regime"] == "N1"], ["model", "format"], n_resamples=args.n_resamples)
    by_reg = grouped_accuracy_ci(df, ["regime"], n_resamples=args.n_resamples)
    by_task = grouped_accuracy_ci(df, ["regime", "task_type"], n_resamples=args.n_resamples)
    by_tab = grouped_accuracy_ci(df, ["regime", "table_name"], n_resamples=args.n_resamples)

    by_mr.to_csv(tables_dir / "accuracy_by_model_regime.csv", index=False)
    by_mrt.to_csv(tables_dir / "accuracy_by_model_regime_task.csv", index=False)
    by_mrtab.to_csv(tables_dir / "accuracy_by_model_regime_table.csv", index=False)
    by_mf.to_csv(tables_dir / "accuracy_by_model_format.csv", index=False)
    by_reg.to_csv(tables_dir / "accuracy_by_regime.csv", index=False)
    by_task.to_csv(tables_dir / "accuracy_by_regime_task.csv", index=False)
    by_tab.to_csv(tables_dir / "accuracy_by_regime_table.csv", index=False)

    generate_tab2_accuracy_regimes(df, models, tables_dir, n_resamples=args.n_resamples)
    generate_tab3_error_taxonomy(df, models, tables_dir)

    logger.info(f"Overall accuracy by regime:\n{by_reg.to_string(index=False)}")
    logger.info(f"Saved accuracy tables to {tables_dir}")


if __name__ == "__main__":
    main()
