"""Error analysis and validity diagnostics."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_empty_rate(df: pd.DataFrame, col: str = "predicted") -> float:
    """Compute fraction of empty or NaN predictions (fixes pandas NaN-to-'nan' bug)."""
    if df.empty or col not in df.columns:
        return 0.0
    series = df[col]
    is_empty = series.isna() | (series.astype(str).str.strip() == "") | (series.astype(str).str.strip().str.lower() == "nan")
    return float(is_empty.mean())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    acc_path = Path(cfg["outputs"]["accuracy"])
    # keep_default_na=False preserves empty strings "" rather than converting to NaN
    df = pd.read_csv(acc_path, keep_default_na=False)
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0)

    logger.info("Error by task type:")
    logger.info(f"\n{df.groupby('task_type')['correct'].agg(['mean', 'count'])}")
    logger.info("\nError by format:")
    logger.info(f"\n{df.groupby('format')['correct'].agg(['mean', 'count'])}")
    if "regime" in df.columns:
        logger.info("\nAccuracy by regime:")
        logger.info(f"\n{df.groupby('regime')['correct'].agg(['mean', 'count'])}")
    if "error_class" in df.columns:
        logger.info("\nError class distribution:")
        logger.info(f"\n{df.groupby(['regime', 'error_class']).size().unstack(fill_value=0) if 'regime' in df.columns else df['error_class'].value_counts()}")
    logger.info("\nWorst questions:")
    logger.info(f"\n{df.groupby('qid')['correct'].mean().sort_values().head(10)}")
    empty_rate = compute_empty_rate(df, "predicted")
    logger.info(f"\nEmpty/unparsed prediction rate: {empty_rate:.4f}")


if __name__ == "__main__":
    main()
