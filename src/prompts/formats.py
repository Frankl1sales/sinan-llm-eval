"""Format a pandas DataFrame into textual representations."""
import pandas as pd


def to_csv_string(df):
    return df.to_csv(index=False)


def to_markdown_string(df):
    return df.to_markdown(index=False)


def to_json_string(df):
    return df.to_json(orient="records", indent=2)


FORMATTERS = {
    "csv": to_csv_string,
    "markdown": to_markdown_string,
    "json": to_json_string,
}


def format_table(df, fmt):
    if fmt not in FORMATTERS:
        raise ValueError(f"Unknown format: {fmt}")
    return FORMATTERS[fmt](df)
