from __future__ import annotations
import re

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.IGNORECASE)
_NON_WORD_RE = re.compile(r"[^a-z0-9\s<>]", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    """Basic text cleaning for phishing/spam classification."""
    if text is None:
        return ""
    t = str(text).strip().lower()
    t = _URL_RE.sub("<URL>", t)
    t = _EMAIL_RE.sub("<EMAIL>", t)
    t = _NON_WORD_RE.sub(" ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t

def coerce_label(label: str) -> int:
    """Map common labels to binary: phishing/spam=1, ham/legit=0."""
    if label is None:
        raise ValueError("Label is None")
    s = str(label).strip().lower()
    phishing_tokens = {"phishing", "spam", "malicious", "1", "true", "yes"}
    legit_tokens = {"ham", "legit", "legitimate", "benign", "0", "false", "no", "normal"}
    if s in phishing_tokens:
        return 1
    if s in legit_tokens:
        return 0
    if "phish" in s or "spam" in s or "mal" in s:
        return 1
    if "ham" in s or "legit" in s or "benign" in s or "normal" in s:
        return 0
    raise ValueError(f"Unrecognised label value: {label!r}")
