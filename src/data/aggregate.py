"""Aggregate cleaned SINAN data into small tables."""
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def aggregate_monthly(df):
    """Aggregate case counts by notification year and month (preserves late-2023 notifications)."""
    return df.groupby(["year", "month"]).size().reset_index(name="cases")


def aggregate_weekly(df):
    """Aggregate case counts by epidemiological week_year and week."""
    return df.groupby(["week_year", "week"]).size().reset_index(name="cases")


def aggregate_by_sex(df):
    """Aggregate case counts by year and CS_SEXO (preserves blank and ignored 'I' entries)."""
    if "CS_SEXO" not in df.columns:
        return pd.DataFrame()
    return df.groupby(["year", "CS_SEXO"]).size().reset_index(name="cases")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed_dir = Path(cfg["data"]["processed_dir"])
    clean_file = processed_dir / "sinan_clean.parquet"
    if not clean_file.exists():
        raise FileNotFoundError(
            f"Cleaned SINAN parquet not found at {clean_file}. Run `python -m src.data.preprocess` first."
        )
    df = pd.read_parquet(clean_file)
    tables = {
        "monthly": aggregate_monthly(df),
        "weekly": aggregate_weekly(df),
        "by_sex": aggregate_by_sex(df),
    }
    for name, table in tables.items():
        if table.empty:
            continue
        out_file = processed_dir / f"table_{name}.csv"
        table.to_csv(out_file, index=False)
        logger.info(f"Saved {name}: {out_file} ({len(table)} rows)")


if __name__ == "__main__":
    main()
