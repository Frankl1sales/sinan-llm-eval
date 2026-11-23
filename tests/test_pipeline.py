"""Unit tests for question bank, isolated Pandas sandbox, and G-theory MoM estimator."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.gtheory import REGIME_FACTORS_SIMPLE, compute_rho, estimate_variance_components_mom
from src.llm.sandbox import execute_pandas_code, extract_python_code
from src.prompts.questions import build_questions, load_tables


def test_question_bank_structure():
    qs = build_questions()
    assert len(qs) == 36
    assert len({q.qid for q in qs}) == 36
    tables = load_tables("data/processed")
    by_tt = pd.Series([(q.table_name, q.task_type) for q in qs]).value_counts().to_dict()
    assert by_tt[("monthly", "lookup")] == 5
    assert by_tt[("monthly", "aggregation")] == 5
    assert by_tt[("monthly", "comparison")] == 5
    assert by_tt[("weekly", "lookup")] == 4
    assert by_tt[("weekly", "aggregation")] == 4
    assert by_tt[("weekly", "comparison")] == 4
    assert by_tt[("by_sex", "lookup")] == 2
    assert by_tt[("by_sex", "aggregation")] == 2
    assert by_tt[("by_sex", "comparison")] == 2
    assert by_tt[("monthly", "abstention")] == 3
    # Verify Q01 reclassified to aggregation
    q01 = next(q for q in qs if q.qid == "Q01")
    assert q01.task_type == "aggregation"
    # Verify all ground truth functions execute cleanly
    for q in qs:
        gt = q.ground_truth_fn(tables[q.table_name])
        assert isinstance(gt, str) and len(gt) > 0


def test_sandbox_execution_and_isolation():
    csv_path = "data/processed/table_monthly.csv"
    res = execute_pandas_code('print(df[df["year"] == 2024]["cases"].sum())', csv_path, timeout=5)
    assert res.exec_status == "ok"
    assert res.stdout == "149884"

    # Blocks file writes
    res_file = execute_pandas_code('open("/tmp/test_blocked.txt", "w").write("hi")', csv_path, timeout=5)
    assert res_file.exec_status == "runtime_error"

    # Blocks network
    res_net = execute_pandas_code("import socket; print(socket.gethostname())", csv_path, timeout=5)
    assert res_net.exec_status == "runtime_error"

    # Extracts markdown fenced code
    code = extract_python_code("Here is code:\n```python\nprint(1+1)\n```")
    assert code == "print(1+1)"


def test_gtheory_mom_nonnegative_and_rho():
    rng = np.random.default_rng(123)
    rows = []
    model_eff = {"m1": 0.9, "m2": 0.5, "m3": 0.1}
    for m, eff in model_eff.items():
        for i in range(12):
            for rep in range(3):
                val = int(rng.random() < eff)
                rows.append({"model": m, "qid": f"Q{i:02d}", "correct": val})
    df = pd.DataFrame(rows)
    vc = estimate_variance_components_mom(df, REGIME_FACTORS_SIMPLE)
    assert all(v >= 0.0 for v in vc.values())
    assert vc["model"] > 0.05
    rho = compute_rho(vc, n_items=12, n_regimes=1.0, obs_per_model_item=3.0)
    assert 0.5 < rho <= 1.0
