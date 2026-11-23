"""Typed parsing and correctness evaluation for LLM responses."""
from __future__ import annotations

import re
from typing import Optional

MONTH_MAP = {
    "january": "1", "janeiro": "1", "jan": "1",
    "february": "2", "fevereiro": "2", "feb": "2", "fev": "2",
    "march": "3", "março": "3", "marco": "3", "mar": "3",
    "april": "4", "abril": "4", "apr": "4", "abr": "4",
    "may": "5", "maio": "5", "mai": "5",
    "june": "6", "junho": "6", "jun": "6",
    "july": "7", "julho": "7", "jul": "7",
    "august": "8", "agosto": "8", "aug": "8", "ago": "8",
    "september": "9", "setembro": "9", "sept": "9", "sep": "9", "set": "9",
    "october": "10", "outubro": "10", "oct": "10", "out": "10",
    "november": "11", "novembro": "11", "nov": "11",
    "december": "12", "dezembro": "12", "dec": "12", "dez": "12",
}

# Sort keys longest-first so full names match before abbreviations
_MONTH_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(MONTH_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

_BOOL_MAP = {
    "yes": "yes", "true": "yes", "sim": "yes", "verdadeiro": "yes",
    "no": "no", "false": "no", "não": "no", "nao": "no", "falso": "no",
}

_BOOL_PATTERN = re.compile(
    r"\b(yes|no|true|false|sim|não|nao|verdadeiro|falso)\b",
    re.IGNORECASE,
)

_CATEGORY_WORD_MAP = {
    "female": "F",
    "females": "F",
    "feminino": "F",
    "male": "M",
    "males": "M",
    "masculino": "M",
    "ignored": "I",
    "ignorado": "I",
}

ABSTENTION_PHRASES = (
    "insufficient",
    "not available",
    "cannot be determined",
    "can't be determined",
    "cant be determined",
    "unknown",
    "not in the table",
    "not in table",
    "n/a",
    "not provided",
    "no data",
    "not present",
    "cannot answer",
    "can't answer",
    "cannot be answered",
    "can't be answered",
    "unable to determine",
    "no information",
    "not found",
    "missing",
    "does not contain",
    "doesn't contain",
    "not possible to determine",
    "unanswerable",
    "unavailable",
    "abstain",
    "not mentioned",
    "no record",
    "not included",
    "não disponível",
    "insuficiente",
    "não consta",
)


def _clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if s.lower() == "nan":
        return ""
    return s


def extract_number(text: Optional[str]) -> str:
    """Extract the LAST number in the response (the final answer)."""
    s = _clean_text(text)
    if not s:
        return ""
    # Match signed numbers with optional thousands commas and optional decimals
    matches = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", s)
    if not matches:
        return ""
    return matches[-1].replace(",", "")


def extract_boolean(text: Optional[str]) -> str:
    """Normalize yes/no/true/false (EN/PT) to 'yes' or 'no'."""
    s = _clean_text(text)
    if not s:
        return ""
    stripped = re.sub(r"^[*`\"'\s]+|[*`\"'.!?\s]+$", "", s)
    lower_stripped = stripped.lower()
    if lower_stripped in _BOOL_MAP:
        return _BOOL_MAP[lower_stripped]

    # Check if response starts with a boolean answer (e.g. "Yes, February has...")
    start_match = re.match(
        r"^[*`\"'\s]*(yes|no|true|false|sim|não|nao|verdadeiro|falso)\b",
        s,
        re.IGNORECASE,
    )
    if start_match:
        return _BOOL_MAP[start_match.group(1).lower()]

    matches = _BOOL_PATTERN.findall(s)
    if not matches:
        return ""
    return _BOOL_MAP[matches[-1].lower()]


def extract_month(text: Optional[str]) -> str:
    """Map EN/PT month names/abbreviations to '1'-'12' or extract month int."""
    s = _clean_text(text)
    if not s:
        return ""

    # 1. Look for explicit month names / abbreviations (prefer the last match)
    name_matches = _MONTH_PATTERN.findall(s)
    if name_matches:
        return MONTH_MAP[name_matches[-1].lower()]

    # 2. Look for YYYY-MM or MM/YYYY patterns
    ym_match = re.findall(r"\b\d{4}[-/](0?[1-9]|1[0-2])\b", s)
    if ym_match:
        return str(int(ym_match[-1]))

    # 3. Look for standalone integers in 1..12 (excluding 4-digit years)
    num_matches = re.findall(r"\b(\d+)\b", s)
    valid_months = [str(int(m)) for m in num_matches if 1 <= int(m) <= 12]
    if valid_months:
        return valid_months[-1]

    return ""


def extract_category(text: Optional[str]) -> str:
    """Extract and normalize a categorical label (e.g., F, M, I)."""
    s = _clean_text(text)
    if not s:
        return ""

    # Strip common markdown/quote wrappers
    cleaned = re.sub(r"[*`\"']", "", s).strip()
    cleaned_no_punct = cleaned.rstrip(".,;:!?").strip()

    # Exact short category token (e.g., "F", "M", "I")
    if re.fullmatch(r"[A-Za-z0-9_-]{1,3}", cleaned_no_punct):
        return cleaned_no_punct.upper()

    # Check explicit category words (e.g., "Female", "Male", "Ignored")
    word_pattern = re.compile(
        r"\b(" + "|".join(_CATEGORY_WORD_MAP.keys()) + r")\b",
        re.IGNORECASE,
    )
    word_matches = word_pattern.findall(cleaned)
    if word_matches:
        return _CATEGORY_WORD_MAP[word_matches[-1].lower()]

    # Check standalone single-letter SINAN sex codes (F, M, I)
    code_matches = re.findall(r"\b([FMI])\b", cleaned)
    if code_matches:
        return code_matches[-1].upper()

    # Fallback: normalized uppercase token if concise
    if len(cleaned_no_punct) <= 24:
        return cleaned_no_punct.upper()
    return ""


def extract_abstention(text: Optional[str]) -> str:
    """Detect refusal/missing-evidence phrases -> 'ABSTAIN', else non-empty raw or ''."""
    s = _clean_text(text)
    if not s:
        return ""
    lower = s.lower()
    for phrase in ABSTENTION_PHRASES:
        if phrase in lower:
            return "ABSTAIN"
    # Model attempted to answer instead of abstaining
    return s[:64]


def parse_response(text: Optional[str], answer_type: str = "numeric") -> str:
    """Typed parser dispatching by answer_type."""
    atype = (answer_type or "numeric").strip().lower()
    if atype == "numeric":
        return extract_number(text)
    if atype == "boolean":
        return extract_boolean(text)
    if atype == "month":
        return extract_month(text)
    if atype == "category":
        return extract_category(text)
    if atype == "abstention":
        return extract_abstention(text)
    raise ValueError(f"Unknown answer_type: {answer_type}")


def is_correct(
    predicted: Optional[str],
    expected: Optional[str],
    answer_type: str = "numeric",
    tolerance: float = 0.01,
) -> bool:
    """Compare parsed prediction against expected ground truth for the given answer_type."""
    p_str = _clean_text(predicted)
    e_str = _clean_text(expected)
    if p_str == "" or e_str == "":
        return False

    atype = (answer_type or "numeric").strip().lower()
    if atype == "numeric":
        try:
            p = float(p_str.replace(",", ""))
            e = float(e_str.replace(",", ""))
            if e == 0:
                return abs(p) < tolerance
            return abs(p - e) / abs(e) < tolerance
        except ValueError:
            return False

    if atype == "boolean":
        p_norm = _BOOL_MAP.get(p_str.lower(), p_str.lower())
        e_norm = _BOOL_MAP.get(e_str.lower(), e_str.lower())
        return p_norm == e_norm

    if atype == "month":
        p_norm = MONTH_MAP.get(p_str.lower(), p_str)
        e_norm = MONTH_MAP.get(e_str.lower(), e_str)
        try:
            return int(p_norm) == int(e_norm)
        except ValueError:
            return p_norm.strip().lower() == e_norm.strip().lower()

    if atype == "category":
        return p_str.strip().upper() == e_str.strip().upper()

    if atype == "abstention":
        return p_str.strip().upper() == e_str.strip().upper() == "ABSTAIN"

    return p_str.strip().lower() == e_str.strip().lower()
