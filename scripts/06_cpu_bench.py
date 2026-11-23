"""CPU-only benchmark (num_gpu=0) on 10 monthly questions x N1/N2/N3 x 4 Qwen2.5 sizes."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure repo root is on sys.path when invoked as python scripts/06_cpu_bench.py
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.llm.client import OllamaClient
from src.llm.query import (
    load_completed_keys,
    run_n1_item,
    run_n2_n3_item,
)
from src.prompts.formats import format_table
from src.prompts.questions import TABLE_FILES, build_questions, load_tables

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def summarize_cpu_bench(jsonl_path: Path, tables_dir: Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    summary_rows = []
    for (model, regime), grp in df.groupby(["model", "regime"], sort=False):
        n = len(grp)
        n_correct = int(grp["correct"].sum())
        acc = n_correct / n if n > 0 else 0.0
        mean_lat = float(grp["elapsed_s"].mean())
        med_lat = float(grp["elapsed_s"].median())
        total_lat = float(grp["elapsed_s"].sum())
        peak_ram = float(grp["peak_ram_mb"].max())
        mean_prompt_tok = float(grp["prompt_tokens"].mean())
        mean_eval_tok = float(grp["eval_tokens"].mean())
        time_per_correct = (total_lat / n_correct) if n_correct > 0 else np.nan
        summary_rows.append({
            "model": model,
            "regime": regime,
            "n_items": n,
            "n_correct": n_correct,
            "accuracy": round(acc, 4),
            "mean_latency_s": round(mean_lat, 3),
            "median_latency_s": round(med_lat, 3),
            "peak_ram_mb": round(peak_ram, 1),
            "mean_prompt_tokens": round(mean_prompt_tok, 1),
            "mean_eval_tokens": round(mean_eval_tok, 1),
            "time_per_correct_s": round(time_per_correct, 3) if pd.notna(time_per_correct) else np.nan,
        })

    model_order = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b", "qwen2.5:7b"]
    regime_order = ["N1", "N2", "N3"]
    df_sum = pd.DataFrame(summary_rows)
    df_sum["_m"] = df_sum["model"].map({m: i for i, m in enumerate(model_order)}).fillna(99)
    df_sum["_r"] = df_sum["regime"].map({r: i for i, r in enumerate(regime_order)}).fillna(99)
    df_sum = df_sum.sort_values(["_m", "_r"]).drop(columns=["_m", "_r"]).reset_index(drop=True)

    tables_dir.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(tables_dir / "cpu_bench.csv", index=False)

    # Markdown table
    md_lines = [
        "| Model | Regime | Accuracy | Mean Latency (s) | Peak RAM (MB) | Mean Eval Tok | Time / Correct (s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_sum.iterrows():
        tpc = f"{r['time_per_correct_s']:.2f}" if pd.notna(r["time_per_correct_s"]) else "--"
        md_lines.append(
            f"| `{r['model']}` | {r['regime']} | {r['accuracy']*100:.1f}% ({int(r['n_correct'])}/{int(r['n_items'])}) "
            f"| {r['mean_latency_s']:.2f} | {r['peak_ram_mb']:.0f} | {r['mean_eval_tokens']:.1f} | {tpc} |"
        )
    (tables_dir / "tab5_cpu_efficiency.md").write_text("\n".join(md_lines) + "\n")

    # LaTeX table
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{CPU-only (\texttt{num\_gpu=0}) efficiency benchmark on 10 monthly SINAN items across Qwen2.5 model scales and regimes.}",
        r"\label{tab:cpu_efficiency}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Model & Regime & Accuracy (\%) & Mean Lat. (s) & Peak RAM (MB) & Eval Tok. & Time/Correct (s) \\",
        r"\midrule",
    ]
    for _, r in df_sum.iterrows():
        model_esc = r["model"].replace("_", r"\_")
        tpc = f"{r['time_per_correct_s']:.2f}" if pd.notna(r["time_per_correct_s"]) else "--"
        tex_lines.append(
            f"\\texttt{{{model_esc}}} & {r['regime']} & {r['accuracy']*100:.1f} & "
            f"{r['mean_latency_s']:.2f} & {r['peak_ram_mb']:.0f} & {r['mean_eval_tokens']:.1f} & {tpc} \\\\"
        )
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    (tables_dir / "tab5_cpu_efficiency.tex").write_text("\n".join(tex_lines) + "\n")
    logger.info(f"Saved CPU efficiency tables to {tables_dir}")
    return df_sum


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["data"]["processed_dir"])
    out_path = Path(cfg["outputs"]["cpu_bench"])
    tables_dir = Path(cfg["outputs"]["tables_dir"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.summarize_only:
        df_sum = summarize_cpu_bench(out_path, tables_dir)
        print(df_sum.to_string(index=False))
        return

    cpu_cfg = cfg["cpu_bench"]
    models = cpu_cfg["models"]
    qids = set(cpu_cfg["qids"])
    regimes = cpu_cfg["regimes"]
    seed = cpu_cfg.get("seed", 42)
    max_tokens = cpu_cfg.get("max_tokens", 256)
    temperature = cfg["llm"].get("temperature", 0.3)
    timeout = cfg["llm"].get("timeout", 180)
    sandbox_timeout = cfg["llm"].get("sandbox_timeout", 5)

    tables = load_tables(processed_dir)
    all_questions = build_questions()
    questions = [q for q in all_questions if q.qid in qids]
    df_monthly = tables["monthly"]
    csv_path = processed_dir / TABLE_FILES["monthly"]
    table_csv_str = format_table(df_monthly, "csv")
    expected_map = {q.qid: q.ground_truth_fn(df_monthly) for q in questions}

    completed, n2_cache = load_completed_keys(out_path)
    logger.info(f"CPU bench: {len(completed)} existing records in {out_path}")

    total_planned = len(models) * len(questions) * len(regimes)
    with open(out_path, "a") as fout, tqdm(total=total_planned, desc="CPU Bench (num_gpu=0)") as pbar:
        for model in models:
            client = OllamaClient(model=model, temperature=temperature, num_gpu=0)
            logger.info(f"=== CPU Bench model: {model} (num_gpu=0) ===")

            # N1 (csv)
            if "N1" in regimes:
                for q in questions:
                    key = (model, "N1", q.qid, "csv", 0)
                    if key in completed:
                        pbar.update(1)
                        continue
                    rec = run_n1_item(
                        client, q, table_csv_str, "csv", expected_map[q.qid], 0, seed, max_tokens, timeout
                    )
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    completed.add(key)
                    pbar.update(1)

            # N2 & N3 (csv)
            do_n2 = "N2" in regimes
            do_n3 = "N3" in regimes
            if do_n2 or do_n3:
                for q in questions:
                    k2 = (model, "N2", q.qid, "csv", 0)
                    k3 = (model, "N3", q.qid, "csv", 0)
                    need_n2 = do_n2 and (k2 not in completed)
                    need_n3 = do_n3 and (k3 not in completed)
                    if not need_n2 and not need_n3:
                        if do_n2:
                            pbar.update(1)
                        if do_n3:
                            pbar.update(1)
                        continue
                    n2_rec, n3_rec = run_n2_n3_item(
                        client=client,
                        q=q,
                        df=df_monthly,
                        csv_path=csv_path,
                        expected=expected_map[q.qid],
                        rep=0,
                        seed=seed,
                        max_tokens=max_tokens,
                        timeout=timeout,
                        sandbox_timeout=sandbox_timeout,
                        run_n2=need_n2,
                        run_n3=need_n3,
                        existing_n2=n2_cache.get((model, q.qid, 0)),
                    )
                    if n2_rec is not None:
                        fout.write(json.dumps(n2_rec) + "\n")
                        fout.flush()
                        completed.add(k2)
                        n2_cache[(model, q.qid, 0)] = n2_rec
                    if do_n2:
                        pbar.update(1)
                    if n3_rec is not None:
                        fout.write(json.dumps(n3_rec) + "\n")
                        fout.flush()
                        completed.add(k3)
                    if do_n3:
                        pbar.update(1)

    df_sum = summarize_cpu_bench(out_path, tables_dir)
    logger.info(f"\n{df_sum.to_string(index=False)}")


if __name__ == "__main__":
    main()
