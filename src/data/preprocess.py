"""Clean and preprocess SINAN chikungunya microdata."""
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CONFIRMED_CODES = {"13"}


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def preprocess(df, state_code="31"):
    """Filter SINAN microdata to confirmed cases (CLASSI_FIN=13) in the target UF state and derive date/week fields."""
    if "SG_UF" in df.columns:
        df = df[df["SG_UF"].astype(str) == state_code]
    if "CLASSI_FIN" in df.columns:
        df = df[df["CLASSI_FIN"].astype(str).isin(CONFIRMED_CODES)]
    if "DT_NOTIFIC" in df.columns:
        df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"], format="%Y%m%d", errors="coerce")
        df["year"] = df["DT_NOTIFIC"].dt.year
        df["month"] = df["DT_NOTIFIC"].dt.month
    if "SEM_NOT" in df.columns:
        df["week"] = df["SEM_NOT"].astype(str).str[-2:].astype(int)
        df["week_year"] = df["SEM_NOT"].astype(str).str[:4].astype(int)
    logger.info(f"After filtering: {len(df)} confirmed cases in state {state_code}")
    return df.reset_index(drop=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"sinan_{cfg['data']['disease']}_{cfg['data']['year']}.parquet"
    if not raw_file.exists():
        raise FileNotFoundError(
            f"Raw SINAN parquet not found at {raw_file}. Run `python -m src.data.download` first."
        )
    df = pd.read_parquet(raw_file)
    df_clean = preprocess(df, state_code=cfg["data"]["state"])
    out_file = processed_dir / "sinan_clean.parquet"
    df_clean.to_parquet(out_file, index=False)
    logger.info(f"Saved cleaned data to {out_file}")


if __name__ == "__main__":
    main()
