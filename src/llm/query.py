"""Run multi-model, multi-regime (N1/N2/N3) LLM measurement experiments with JSONL checkpoint/resume."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

import ollama
import pandas as pd
import yaml
from tqdm import tqdm

from src.llm.client import OllamaClient
from src.llm.parse import is_correct, parse_response
from src.llm.sandbox import execute_pandas_code, extract_python_code
from src.prompts.formats import format_table
from src.prompts.questions import TABLE_FILES, Question, build_questions, load_tables

logger = logging.getLogger(__name__)

QUICKSTART_MODEL = "qwen2.5:0.5b"
QUICKSTART_QIDS = ("Q01", "Q02", "Q11", "Q13", "Q34")


def validate_environment(processed_dir: Path, models: list[str]) -> None:
    """Verify required processed CSV tables and local Ollama models are available."""
    missing_files = [
        str(processed_dir / fname)
        for fname in TABLE_FILES.values()
        if not (processed_dir / fname).exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required processed SINAN tables: {missing_files}. "
            f"Run `make data` or verify `data/processed/`."
        )
    try:
        listed = ollama.list()
        available = {
            m.get("name", m.get("model", ""))
            for m in (listed.get("models", []) if isinstance(listed, dict) else [])
        }
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to local Ollama server: {exc}. "
            f"Start Ollama with `ollama serve` before running inference."
        ) from exc

    missing_models = [m for m in models if m not in available and f"{m}:latest" not in available]
    if missing_models:
        pull_cmds = " && ".join(f"ollama pull {m}" for m in missing_models)
        raise RuntimeError(
            f"Missing local Ollama models: {missing_models}. "
            f"Pull them first with: {pull_cmds}"
        )


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_n1_prompt(question: str, table_str: str, fmt: str) -> str:
    return (
        f"You are given a table in {fmt.upper()} format. "
        f"Answer the following question using ONLY the table.\n"
        f"If the table does not contain the information needed to answer the question, "
        f"respond with \"Insufficient information\".\n\n"
        f"Table:\n{table_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer with the final value only, no explanation."
    )


def build_schema_summary(df: pd.DataFrame) -> str:
    dtypes_str = ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())
    head_csv = df.head(5).to_csv(index=False).strip()
    return (
        f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"- Columns and dtypes: {dtypes_str}\n"
        f"- First 5 rows (CSV):\n{head_csv}"
    )


def build_n2_prompt(question: str, df: pd.DataFrame) -> str:
    schema = build_schema_summary(df)
    return (
        f"A pandas DataFrame `df` is already loaded in memory with the following schema:\n"
        f"{schema}\n\n"
        f"Write Python code using pandas on `df` to answer the following question, "
        f"and print ONLY the final answer using `print(...)`.\n"
        f"If the question cannot be answered from the columns or data in `df`, "
        f"print \"Insufficient information\".\n"
        f"Do NOT read any files; `df` is already loaded. "
        f"Put your code inside a ```python ... ``` block.\n\n"
        f"Question: {question}"
    )


def build_n3_repair_prompt(exec_status: str, stderr: str) -> str:
    return (
        f"Your previous Python code failed with status `{exec_status}` and error:\n"
        f"```\n{stderr}\n```\n"
        f"Fix the code and output a corrected ```python ... ``` block using the pre-loaded "
        f"DataFrame `df` that prints ONLY the final answer using `print(...)`."
    )


def classify_error(regime: str, correct: bool, exec_status: str, predicted: str) -> str:
    if correct:
        return "correct"
    if regime in ("N2", "N3"):
        if exec_status != "ok":
            return "noisy_error"
        if predicted == "":
            return "parse_or_empty_error"
        return "silent_error"
    if predicted == "":
        return "parse_or_empty_error"
    return "arithmetic_or_reasoning_error"


def load_completed_keys(out_path: Path) -> tuple[set[tuple], dict[tuple, dict]]:
    """Load already-logged JSONL records for checkpoint/resume."""
    completed: set[tuple] = set()
    n2_cache: dict[tuple, dict] = {}
    if not out_path.exists():
        return completed, n2_cache
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec["model"], rec["regime"], rec["qid"], rec["format"], int(rec["repetition"]))
            completed.add(key)
            if rec["regime"] == "N2":
                n2_cache[(rec["model"], rec["qid"], int(rec["repetition"]))] = rec
    return completed, n2_cache


def run_n1_item(
    client: OllamaClient,
    q: Question,
    table_str: str,
    fmt: str,
    expected: str,
    rep: int,
    seed: int,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    prompt = build_n1_prompt(q.text, table_str, fmt)
    resp = client.query(prompt, max_tokens=max_tokens, seed=seed, timeout=timeout)
    predicted = parse_response(resp.content, q.answer_type)
    correct = is_correct(predicted, expected, q.answer_type)
    error_class = classify_error("N1", correct, "n/a", predicted)
    return {
        "model": client.model,
        "regime": "N1",
        "table_name": q.table_name,
        "qid": q.qid,
        "task_type": q.task_type,
        "answer_type": q.answer_type,
        "format": fmt,
        "repetition": rep,
        "response": resp.content,
        "code_generated": "",
        "exec_status": "n/a",
        "predicted": predicted,
        "expected": expected,
        "correct": int(correct),
        "error_class": error_class,
        "elapsed_s": resp.elapsed_s,
        "prompt_tokens": resp.prompt_tokens,
        "eval_tokens": resp.eval_tokens,
        "n_llm_calls": 1,
        "peak_ram_mb": resp.peak_ram_mb,
    }


def run_n2_n3_item(
    client: OllamaClient,
    q: Question,
    df: pd.DataFrame,
    csv_path: Path,
    expected: str,
    rep: int,
    seed: int,
    max_tokens: int,
    timeout: int,
    sandbox_timeout: int,
    run_n2: bool = True,
    run_n3: bool = True,
    existing_n2: Optional[dict[str, Any]] = None,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Run N2 (tool_1shot) and/or N3 (self_repair) for one (model, qid, rep)."""
    n2_prompt = build_n2_prompt(q.text, df)
    n2_rec = existing_n2
    n2_stderr = ""
    n2_raw_response = ""

    if n2_rec is None and (run_n2 or run_n3):
        resp = client.query(n2_prompt, max_tokens=max_tokens, seed=seed, timeout=timeout)
        n2_raw_response = resp.content
        code = extract_python_code(resp.content)
        sb = execute_pandas_code(code, csv_path, timeout=sandbox_timeout)
        n2_stderr = sb.stderr
        raw_out = sb.stdout if sb.exec_status == "ok" else ""
        # If abstention question and model refused in prose without code, check prose too?
        # Keep strict regime separation: N2/N3 evaluate stdout of executed code.
        predicted = parse_response(raw_out, q.answer_type) if sb.exec_status == "ok" else ""
        correct = is_correct(predicted, expected, q.answer_type) if sb.exec_status == "ok" else False
        error_class = classify_error("N2", correct, sb.exec_status, predicted)
        n2_rec = {
            "model": client.model,
            "regime": "N2",
            "table_name": q.table_name,
            "qid": q.qid,
            "task_type": q.task_type,
            "answer_type": q.answer_type,
            "format": "csv",
            "repetition": rep,
            "response": sb.stdout if sb.exec_status == "ok" else resp.content,
            "code_generated": code,
            "exec_status": sb.exec_status,
            "predicted": predicted,
            "expected": expected,
            "correct": int(correct),
            "error_class": error_class,
            "elapsed_s": round(resp.elapsed_s + sb.elapsed_s, 4),
            "prompt_tokens": resp.prompt_tokens,
            "eval_tokens": resp.eval_tokens,
            "n_llm_calls": 1,
            "peak_ram_mb": resp.peak_ram_mb,
            "_n2_raw_response": n2_raw_response,
            "_n2_stderr": n2_stderr,
        }

    n3_rec = None
    if run_n3 and n2_rec is not None:
        if n2_rec["exec_status"] == "ok":
            # No repair needed: N3 inherits N2's successful execution
            n3_rec = {k: v for k, v in n2_rec.items() if not k.startswith("_")}
            n3_rec["regime"] = "N3"
        else:
            # Re-execute N2 code if stderr wasn't cached (e.g. resumed from JSONL)
            stderr = n2_rec.get("_n2_stderr", "")
            raw_resp = n2_rec.get("_n2_raw_response", n2_rec.get("response", ""))
            if not stderr:
                sb_re = execute_pandas_code(n2_rec.get("code_generated", ""), csv_path, timeout=sandbox_timeout)
                stderr = sb_re.stderr
            repair_prompt = build_n3_repair_prompt(n2_rec["exec_status"], stderr)
            messages = [
                {"role": "user", "content": n2_prompt},
                {"role": "assistant", "content": raw_resp},
                {"role": "user", "content": repair_prompt},
            ]
            resp2 = client.chat(messages, max_tokens=max_tokens, seed=seed + 1000, timeout=timeout)
            code2 = extract_python_code(resp2.content)
            sb2 = execute_pandas_code(code2, csv_path, timeout=sandbox_timeout)
            raw_out2 = sb2.stdout if sb2.exec_status == "ok" else ""
            predicted2 = parse_response(raw_out2, q.answer_type) if sb2.exec_status == "ok" else ""
            correct2 = is_correct(predicted2, expected, q.answer_type) if sb2.exec_status == "ok" else False
            error_class2 = classify_error("N3", correct2, sb2.exec_status, predicted2)
            n3_rec = {
                "model": client.model,
                "regime": "N3",
                "table_name": q.table_name,
                "qid": q.qid,
                "task_type": q.task_type,
                "answer_type": q.answer_type,
                "format": "csv",
                "repetition": rep,
                "response": sb2.stdout if sb2.exec_status == "ok" else resp2.content,
                "code_generated": code2,
                "exec_status": sb2.exec_status,
                "predicted": predicted2,
                "expected": expected,
                "correct": int(correct2),
                "error_class": error_class2,
                "elapsed_s": round(float(n2_rec["elapsed_s"]) + resp2.elapsed_s + sb2.elapsed_s, 4),
                "prompt_tokens": int(n2_rec["prompt_tokens"]) + resp2.prompt_tokens,
                "eval_tokens": int(n2_rec["eval_tokens"]) + resp2.eval_tokens,
                "n_llm_calls": 2,
                "peak_ram_mb": max(float(n2_rec["peak_ram_mb"]), resp2.peak_ram_mb),
            }

    clean_n2 = None
    if run_n2 and existing_n2 is None and n2_rec is not None:
        clean_n2 = {k: v for k, v in n2_rec.items() if not k.startswith("_")}
    return clean_n2, n3_rec


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Run SINAN multi-model, multi-regime LLM measurement experiments.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML experiment config.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of Ollama models.")
    parser.add_argument("--regimes", nargs="*", default=None, help="Optional subset of regimes (N1, N2, N3).")
    parser.add_argument(
        "--quickstart",
        action="store_true",
        help="Run fast (<30s) smoke test: qwen2.5:0.5b x 5 items x 1 rep into quickstart_responses.jsonl.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["data"]["processed_dir"])
    out_path = Path(cfg["outputs"]["responses"])
    if args.quickstart:
        out_path = out_path.parent / "quickstart_responses.jsonl"
        if out_path.exists():
            out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    models = [QUICKSTART_MODEL] if args.quickstart else (args.models or cfg["llm"]["models"])
    validate_environment(processed_dir, models)

    tables = load_tables(processed_dir)
    questions = build_questions()
    if args.quickstart:
        questions = [q for q in questions if q.qid in QUICKSTART_QIDS]
    expected_map = {q.qid: q.ground_truth_fn(tables[q.table_name]) for q in questions}

    # Pre-format tables for N1
    formats = ["csv"] if args.quickstart else cfg.get("formats", ["csv", "markdown"])
    formatted_tables = {
        (tname, fmt): format_table(df, fmt)
        for tname, df in tables.items()
        for fmt in formats
    }

    regimes = args.regimes or cfg.get("regimes", ["N1", "N2", "N3"])
    seeds = cfg["experiment"].get("seeds", [42, 43, 44])
    n_rep = 1 if args.quickstart else cfg["experiment"].get("n_repetitions", len(seeds))
    temperature = cfg["llm"].get("temperature", 0.3)
    max_tokens = cfg["llm"].get("max_tokens", 512)
    timeout = cfg["llm"].get("timeout", 120)
    sandbox_timeout = cfg["llm"].get("sandbox_timeout", 5)

    completed, n2_cache = load_completed_keys(out_path)
    logger.info(f"Loaded {len(completed)} existing records from {out_path}")

    # Compute total planned records for progress bar
    total_planned = 0
    for _ in models:
        if "N1" in regimes:
            total_planned += len(questions) * len(formats) * n_rep
        if "N2" in regimes:
            total_planned += len(questions) * n_rep
        if "N3" in regimes:
            total_planned += len(questions) * n_rep

    with open(out_path, "a") as fout, tqdm(total=total_planned, desc="Experiment") as pbar:
        for model in models:
            client = OllamaClient(model=model, temperature=temperature)
            logger.info(f"=== Running model: {model} ===")

            # --- Regime N1 (zero_shot: csv & markdown) ---
            if "N1" in regimes:
                for fmt in formats:
                    for q in questions:
                        table_str = formatted_tables[(q.table_name, fmt)]
                        expected = expected_map[q.qid]
                        for rep in range(n_rep):
                            key = (model, "N1", q.qid, fmt, rep)
                            if key in completed:
                                pbar.update(1)
                                continue
                            seed = seeds[rep] if rep < len(seeds) else 42 + rep
                            rec = run_n1_item(
                                client, q, table_str, fmt, expected, rep, seed, max_tokens, timeout
                            )
                            fout.write(json.dumps(rec) + "\n")
                            fout.flush()
                            completed.add(key)
                            pbar.update(1)

            # --- Regimes N2 (tool_1shot) & N3 (self_repair) on csv ---
            do_n2 = "N2" in regimes
            do_n3 = "N3" in regimes
            if do_n2 or do_n3:
                for q in questions:
                    df = tables[q.table_name]
                    csv_path = processed_dir / TABLE_FILES[q.table_name]
                    expected = expected_map[q.qid]
                    for rep in range(n_rep):
                        k2 = (model, "N2", q.qid, "csv", rep)
                        k3 = (model, "N3", q.qid, "csv", rep)
                        need_n2 = do_n2 and (k2 not in completed)
                        need_n3 = do_n3 and (k3 not in completed)
                        if not need_n2 and not need_n3:
                            if do_n2:
                                pbar.update(1)
                            if do_n3:
                                pbar.update(1)
                            continue

                        seed = seeds[rep] if rep < len(seeds) else 42 + rep
                        existing_n2 = n2_cache.get((model, q.qid, rep))
                        n2_rec, n3_rec = run_n2_n3_item(
                            client=client,
                            q=q,
                            df=df,
                            csv_path=csv_path,
                            expected=expected,
                            rep=rep,
                            seed=seed,
                            max_tokens=max_tokens,
                            timeout=timeout,
                            sandbox_timeout=sandbox_timeout,
                            run_n2=need_n2,
                            run_n3=need_n3,
                            existing_n2=existing_n2,
                        )
                        if n2_rec is not None:
                            fout.write(json.dumps(n2_rec) + "\n")
                            fout.flush()
                            completed.add(k2)
                            n2_cache[(model, q.qid, rep)] = n2_rec
                        if do_n2:
                            pbar.update(1)
                        if n3_rec is not None:
                            fout.write(json.dumps(n3_rec) + "\n")
                            fout.flush()
                            completed.add(k3)
                        if do_n3:
                            pbar.update(1)

    logger.info(f"Finished! Total completed records in {out_path}: {len(completed)}")


if __name__ == "__main__":
    main()
