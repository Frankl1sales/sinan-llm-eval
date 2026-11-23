"""Unit tests for typed response parsing, correctness scoring, and validity diagnostics."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.validity import compute_empty_rate
from src.llm.parse import (
    extract_abstention,
    extract_boolean,
    extract_category,
    extract_month,
    extract_number,
    is_correct,
    parse_response,
)


class TestNumericParsing:
    def test_last_number_extracted(self):
        assert parse_response("In 2024, there were 149,884 cases.", "numeric") == "149884"
        assert parse_response("Step 1: 24647 + 40005 = 64652", "numeric") == "64652"
        assert parse_response("12,490.33", "numeric") == "12490.33"
        assert parse_response("The answer is -42.", "numeric") == "-42"
        assert parse_response("", "numeric") == ""
        assert parse_response("No numbers here", "numeric") == ""

    def test_numeric_tolerance(self):
        assert is_correct("149884", "149884", "numeric")
        assert is_correct("150500", "149884", "numeric")  # ~0.41% < 1%
        assert not is_correct("152000", "149884", "numeric")  # ~1.41% > 1%
        assert is_correct("0.005", "0", "numeric")
        assert not is_correct("0.05", "0", "numeric")
        assert not is_correct("", "100", "numeric")


class TestBooleanParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Yes", "yes"),
            ("yes.", "yes"),
            ("TRUE", "yes"),
            ("Sim", "yes"),
            ("Yes, January had more cases.", "yes"),
            ("No", "no"),
            ("False", "no"),
            ("Não", "no"),
            ("Is January higher than February? No.", "no"),
            ("", ""),
            ("Maybe", ""),
        ],
    )
    def test_boolean_normalization(self, raw, expected):
        assert parse_response(raw, "boolean") == expected

    def test_boolean_correctness(self):
        assert is_correct("yes", "yes", "boolean")
        assert is_correct("True", "yes", "boolean")
        assert is_correct("false", "no", "boolean")
        assert not is_correct("yes", "no", "boolean")
        assert not is_correct("", "no", "boolean")


class TestMonthParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("February", "2"),
            ("Feb", "2"),
            ("Fevereiro", "2"),
            ("Fev", "2"),
            ("October", "10"),
            ("Outubro", "10"),
            ("Out", "10"),
            ("Dezembro", "12"),
            ("Dec", "12"),
            ("Março", "3"),
            ("2", "2"),
            ("Month 10", "10"),
            ("2024-02", "2"),
            ("The peak month in 2024 was February (40,005 cases).", "2"),
            ("", ""),
            ("2024", ""),
        ],
    )
    def test_month_normalization(self, raw, expected):
        assert parse_response(raw, "month") == expected

    def test_month_correctness(self):
        assert is_correct("2", "2", "month")
        assert is_correct("February", "2", "month")
        assert is_correct("10", "10", "month")
        assert not is_correct("1", "2", "month")
        assert not is_correct("", "2", "month")


class TestCategoryParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("F", "F"),
            ("f", "F"),
            ("`F`", "F"),
            ("**M**", "M"),
            ("I.", "I"),
            ("Female", "F"),
            ("Feminino", "F"),
            ("Male", "M"),
            ("Ignored", "I"),
            ("The sex category with the highest count is F.", "F"),
            ("CS_SEXO = M", "M"),
            ("", ""),
        ],
    )
    def test_category_normalization(self, raw, expected):
        assert parse_response(raw, "category") == expected

    def test_category_correctness(self):
        assert is_correct("F", "F", "category")
        assert is_correct("f", "F", "category")
        assert not is_correct("M", "F", "category")
        assert not is_correct("", "F", "category")


class TestAbstentionParsing:
    @pytest.mark.parametrize(
        "raw",
        [
            "Insufficient information in the table.",
            "Not available",
            "Cannot be determined from the provided data.",
            "Unknown",
            "This information is not in the table.",
            "N/A",
            "ABSTAIN",
            "Não disponível na tabela.",
        ],
    )
    def test_abstention_detected(self, raw):
        pred = parse_response(raw, "abstention")
        assert pred == "ABSTAIN"
        assert is_correct(pred, "ABSTAIN", "abstention")

    def test_hallucinated_answer_fails_abstention(self):
        pred = parse_response("149884", "abstention")
        assert pred != "ABSTAIN"
        assert pred != ""
        assert not is_correct(pred, "ABSTAIN", "abstention")

    def test_empty_response_abstention(self):
        assert parse_response("", "abstention") == ""
        assert not is_correct("", "ABSTAIN", "abstention")


class TestValidityEmptyRate:
    def test_nan_and_empty_detected(self):
        df = pd.DataFrame({"predicted": ["42", "", np.nan, "nan", "  "]})
        rate = compute_empty_rate(df, "predicted")
        assert pytest.approx(rate) == 0.8

    def test_no_empty(self):
        df = pd.DataFrame({"predicted": ["42", "yes", "F"]})
        assert compute_empty_rate(df, "predicted") == 0.0
