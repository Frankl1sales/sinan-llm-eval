"""Download SINAN microdata via PySUS 2.x."""
import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml
from pysus.ftp import sinan as sinan_download

logger = logging.getLogger(__name__)


def load_config(config_path="configs/config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_sinan(disease, year, output_dir):
    """Download SINAN data using the PySUS 2.x API."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading SINAN {disease} for year {year}...")
    
    # PySUS 2.x: one-line download returning a DataFrame
    df = sinan_download(disease=disease, year=year, as_dataframe=True)
    
    out_file = output_path / f"sinan_{disease}_{year}.parquet"
    df.to_parquet(out_file, index=False)
    logger.info(f"Saved {len(df)} records to {out_file}")
    return out_file


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    download_sinan(cfg["data"]["disease"], cfg["data"]["year"], cfg["data"]["raw_dir"])


if __name__ == "__main__":
    main()
