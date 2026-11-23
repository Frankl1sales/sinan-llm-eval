"""Define the 36-item deterministic Pandas-grounded question bank."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

TABLE_FILES = {
    "monthly": "table_monthly.csv",
    "weekly": "table_weekly.csv",
    "by_sex": "table_by_sex.csv",
}


@dataclass(frozen=True)
class Question:
    qid: str
    table_name: str
    task_type: str
    answer_type: str
    text: str
    ground_truth_fn: Callable[[pd.DataFrame], str]


def load_tables(processed_dir: str | Path = "data/processed") -> dict[str, pd.DataFrame]:
    """Load all processed SINAN summary tables."""
    pdir = Path(processed_dir)
    return {
        name: pd.read_csv(pdir / fname)
        for name, fname in TABLE_FILES.items()
    }


def build_questions() -> list[Question]:
    """Return all 36 deterministic Pandas-grounded benchmark questions."""
    return [
        # =====================================================================
        # MONTHLY TABLE (15 items: 5 lookup, 5 aggregation, 5 comparison)
        # Real trap: row (2023, 12, 21) precedes 2024 rows.
        # =====================================================================
        # --- Aggregation (5 items; Q01 reclassified to aggregation) ---
        Question(
            "Q01", "monthly", "aggregation", "numeric",
            "What is the total number of cases in 2024?",
            lambda df: str(int(df[df["year"] == 2024]["cases"].sum())),
        ),
        Question(
            "Q06", "monthly", "aggregation", "numeric",
            "What is the average number of cases per month in 2024?",
            lambda df: f"{df[df['year'] == 2024]['cases'].mean():.2f}",
        ),
        Question(
            "Q07", "monthly", "aggregation", "numeric",
            "What is the total number of cases in Q1 (January, February, and March) of 2024?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["month"].isin([1, 2, 3]))]["cases"].sum())),
        ),
        Question(
            "Q08", "monthly", "aggregation", "numeric",
            "What is the total number of cases in June and July 2024 combined?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["month"].isin([6, 7]))]["cases"].sum())),
        ),
        Question(
            "Q10", "monthly", "aggregation", "numeric",
            "What is the total number of cases in the last three months of 2024 (October, November, and December)?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["month"] >= 10)]["cases"].sum())),
        ),
        # --- Lookup (5 items) ---
        Question(
            "Q02", "monthly", "lookup", "numeric",
            "How many cases were reported in January 2024?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["month"] == 1)]["cases"].sum())),
        ),
        Question(
            "Q03", "monthly", "lookup", "numeric",
            "How many cases were reported in June 2024?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["month"] == 6)]["cases"].sum())),
        ),
        Question(
            "Q04", "monthly", "lookup", "numeric",
            "How many cases were reported in December 2024?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["month"] == 12)]["cases"].sum())),
        ),
        Question(
            "Q05", "monthly", "lookup", "numeric",
            "How many cases were reported in March 2024?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["month"] == 3)]["cases"].sum())),
        ),
        Question(
            "Q09", "monthly", "lookup", "numeric",
            "How many cases were reported in December 2023?",
            lambda df: str(int(df[(df["year"] == 2023) & (df["month"] == 12)]["cases"].sum())),
        ),
        # --- Comparison (5 items: month, boolean, numeric) ---
        Question(
            "Q11", "monthly", "comparison", "month",
            "Which month had the highest number of cases in 2024?",
            lambda df: str(int(df[df["year"] == 2024].groupby("month")["cases"].sum().idxmax())),
        ),
        Question(
            "Q12", "monthly", "comparison", "month",
            "Which month had the lowest number of cases in 2024?",
            lambda df: str(int(df[df["year"] == 2024].groupby("month")["cases"].sum().idxmin())),
        ),
        Question(
            "Q13", "monthly", "comparison", "boolean",
            "Did January 2024 have more cases than February 2024?",
            lambda df: "yes" if df[(df["year"] == 2024) & (df["month"] == 1)]["cases"].sum()
            > df[(df["year"] == 2024) & (df["month"] == 2)]["cases"].sum() else "no",
        ),
        Question(
            "Q14", "monthly", "comparison", "boolean",
            "Did June 2024 have fewer cases than July 2024?",
            lambda df: "yes" if df[(df["year"] == 2024) & (df["month"] == 6)]["cases"].sum()
            < df[(df["year"] == 2024) & (df["month"] == 7)]["cases"].sum() else "no",
        ),
        Question(
            "Q15", "monthly", "comparison", "numeric",
            "What is the difference between the highest and lowest monthly case counts in 2024?",
            lambda df: str(int(
                df[df["year"] == 2024].groupby("month")["cases"].sum().max()
                - df[df["year"] == 2024].groupby("month")["cases"].sum().min()
            )),
        ),

        # =====================================================================
        # WEEKLY TABLE (12 items: 4 lookup, 4 aggregation, 4 comparison)
        # =====================================================================
        # --- Lookup (4 items) ---
        Question(
            "Q16", "weekly", "lookup", "numeric",
            "How many cases were reported in epidemiological week 1 of 2024?",
            lambda df: str(int(df[(df["week_year"] == 2024) & (df["week"] == 1)]["cases"].sum())),
        ),
        Question(
            "Q17", "weekly", "lookup", "numeric",
            "How many cases were reported in epidemiological week 9 of 2024?",
            lambda df: str(int(df[(df["week_year"] == 2024) & (df["week"] == 9)]["cases"].sum())),
        ),
        Question(
            "Q18", "weekly", "lookup", "numeric",
            "How many cases were reported in epidemiological week 26 of 2024?",
            lambda df: str(int(df[(df["week_year"] == 2024) & (df["week"] == 26)]["cases"].sum())),
        ),
        Question(
            "Q19", "weekly", "lookup", "numeric",
            "How many cases were reported in epidemiological week 52 of 2024?",
            lambda df: str(int(df[(df["week_year"] == 2024) & (df["week"] == 52)]["cases"].sum())),
        ),
        # --- Aggregation (4 items) ---
        Question(
            "Q20", "weekly", "aggregation", "numeric",
            "What is the total number of cases across epidemiological weeks 1 to 4 of 2024?",
            lambda df: str(int(df[(df["week_year"] == 2024) & (df["week"].between(1, 4))]["cases"].sum())),
        ),
        Question(
            "Q21", "weekly", "aggregation", "numeric",
            "What is the total number of cases across epidemiological weeks 50 to 52 of 2024?",
            lambda df: str(int(df[(df["week_year"] == 2024) & (df["week"].between(50, 52))]["cases"].sum())),
        ),
        Question(
            "Q22", "weekly", "aggregation", "numeric",
            "What is the total number of cases across all epidemiological weeks in 2024?",
            lambda df: str(int(df[df["week_year"] == 2024]["cases"].sum())),
        ),
        Question(
            "Q23", "weekly", "aggregation", "numeric",
            "What is the average number of cases per epidemiological week in 2024?",
            lambda df: f"{df[df['week_year'] == 2024]['cases'].mean():.2f}",
        ),
        # --- Comparison (4 items) ---
        Question(
            "Q24", "weekly", "comparison", "numeric",
            "Which epidemiological week number had the highest number of cases in 2024?",
            lambda df: str(int(df[df["week_year"] == 2024].groupby("week")["cases"].sum().idxmax())),
        ),
        Question(
            "Q25", "weekly", "comparison", "numeric",
            "Which epidemiological week number had the lowest number of cases in 2024?",
            lambda df: str(int(df[df["week_year"] == 2024].groupby("week")["cases"].sum().idxmin())),
        ),
        Question(
            "Q26", "weekly", "comparison", "boolean",
            "Did epidemiological week 10 have more cases than epidemiological week 11 in 2024?",
            lambda df: "yes" if df[(df["week_year"] == 2024) & (df["week"] == 10)]["cases"].sum()
            > df[(df["week_year"] == 2024) & (df["week"] == 11)]["cases"].sum() else "no",
        ),
        Question(
            "Q27", "weekly", "comparison", "numeric",
            "How many more cases were reported in epidemiological week 9 than in epidemiological week 1 of 2024?",
            lambda df: str(int(
                df[(df["week_year"] == 2024) & (df["week"] == 9)]["cases"].sum()
                - df[(df["week_year"] == 2024) & (df["week"] == 1)]["cases"].sum()
            )),
        ),

        # =====================================================================
        # BY_SEX TABLE (6 items: 2 lookup, 2 aggregation, 2 comparison)
        # Real traps: 2023 rows, blank CS_SEXO (2024,,1), and ignored 'I' (164).
        # =====================================================================
        # --- Lookup (2 items) ---
        Question(
            "Q28", "by_sex", "lookup", "numeric",
            "How many cases were reported for females (CS_SEXO = F) in 2024?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["CS_SEXO"] == "F")]["cases"].sum())),
        ),
        Question(
            "Q29", "by_sex", "lookup", "numeric",
            "How many cases were reported with ignored sex (CS_SEXO = I) in 2024?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["CS_SEXO"] == "I")]["cases"].sum())),
        ),
        # --- Aggregation (2 items) ---
        Question(
            "Q30", "by_sex", "aggregation", "numeric",
            "What is the total number of cases in 2024 across all sex categories (including ignored and blank)?",
            lambda df: str(int(df[df["year"] == 2024]["cases"].sum())),
        ),
        Question(
            "Q31", "by_sex", "aggregation", "numeric",
            "What is the combined number of male (M) and female (F) cases in 2024?",
            lambda df: str(int(df[(df["year"] == 2024) & (df["CS_SEXO"].isin(["M", "F"]))]["cases"].sum())),
        ),
        # --- Comparison (2 items) ---
        Question(
            "Q32", "by_sex", "comparison", "category",
            "Which sex category (CS_SEXO) had the highest number of cases in 2024?",
            lambda df: str(df[df["year"] == 2024].groupby("CS_SEXO")["cases"].sum().idxmax()),
        ),
        Question(
            "Q33", "by_sex", "comparison", "boolean",
            "Were there more female (F) cases than male (M) cases in 2024?",
            lambda df: "yes" if df[(df["year"] == 2024) & (df["CS_SEXO"] == "F")]["cases"].sum()
            > df[(df["year"] == 2024) & (df["CS_SEXO"] == "M")]["cases"].sum() else "no",
        ),

        # =====================================================================
        # ABSTENTION (3 items on monthly table — unanswerable from the table)
        # =====================================================================
        Question(
            "Q34", "monthly", "abstention", "abstention",
            "How many deaths due to chikungunya were reported in 2024?",
            lambda df: "ABSTAIN",
        ),
        Question(
            "Q35", "monthly", "abstention", "abstention",
            "How many dengue cases were reported in February 2024?",
            lambda df: "ABSTAIN",
        ),
        Question(
            "Q36", "monthly", "abstention", "abstention",
            "How many chikungunya cases were reported in the state of São Paulo (SP) in 2024?",
            lambda df: "ABSTAIN",
        ),
    ]
