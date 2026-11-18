#!/usr/bin/env python3
"""Shared constants and text utilities for the OCR brand analysis."""

from __future__ import annotations

import re
import unicodedata

BRANDS = {
    "bang-olufsen": "Bang & Olufsen",
    "fortnum-mason": "Fortnum & Mason",
    "herman-miller": "Herman Miller",
    "jellycat": "Jellycat",
    "jo-malone-london": "Jo Malone London",
    "kodak": "Kodak",
    "lego": "LEGO",
    "oura": "Oura",
    "sennheiser": "Sennheiser",
    "stanley": "Stanley",
}

BRAND_ORDER = list(BRANDS)

# Regular expressions are applied to text normalized with normalize_for_matching().
# Longer aliases precede shorter aliases so that one occurrence is not counted twice.
BRAND_PATTERNS = {
    "bang-olufsen": (
        r"\bbang\s*(?:&|and)?\s*olufsen\b|\bb\s*\.?\s*&\s*\.?\s*o\b"
        r"|\bb\s*\.\s*o\b|\bb\s+(?:and\s+)?o\b"
    ),
    "fortnum-mason": r"\bfortnum\s*(?:(?:&|and)\s*)?mason\b|\bfortnum'?s\b",
    "herman-miller": r"\bherman\s*miller\b",
    "jellycat": r"\bjellycat\b",
    "jo-malone-london": r"\bjo\s*malone(?:\s*london)?\b",
    "kodak": r"\bkodak\b",
    "lego": r"\blego\b",
    "oura": r"\boura\b",
    "sennheiser": r"\bsennheiser\b",
    "stanley": r"\bstanley(?:\s+1913)?\b",
}

BRAND_ALIASES = {
    "bang-olufsen": ["Bang & Olufsen", "Bang and Olufsen", "B&O", "B O"],
    "fortnum-mason": ["Fortnum & Mason", "Fortnum and Mason", "Fortnum Mason", "Fortnum's"],
    "herman-miller": ["Herman Miller"],
    "jellycat": ["Jellycat"],
    "jo-malone-london": ["Jo Malone", "Jo Malone London"],
    "kodak": ["Kodak"],
    "lego": ["LEGO"],
    "oura": ["Oura"],
    "sennheiser": ["Sennheiser"],
    "stanley": ["Stanley", "Stanley 1913"],
}

# Tokens are maximal alphanumeric runs, optionally joined by an internal apostrophe.
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?")
ASCII_TOKEN_RE = re.compile(r"^[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?$")

# Standard English function words. Content and promotional words are intentionally
# retained (e.g. "new", "collection", "shop", "gift", "limited").
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "us", "via", "per", "also", "just", "get", "got",
}


def normalize_for_matching(text: str) -> str:
    """Casefold and normalize punctuation/whitespace before brand-name matching."""
    text = unicodedata.normalize("NFKC", str(text))
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    text = text.casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_brand_matches(text: str, brand: str) -> list[str]:
    """Return non-overlapping alias matches for a target brand."""
    normalized = normalize_for_matching(text)
    pattern = re.compile(BRAND_PATTERNS[brand], flags=re.IGNORECASE)
    return [match.group(0) for match in pattern.finditer(normalized)]


def tokenize(text: str) -> list[str]:
    """Extract ASCII/alphanumeric tokens from a recognized text region."""
    return TOKEN_RE.findall(str(text))


def normalize_token(token: str) -> str:
    """Lowercase a token and remove a possessive suffix for vocabulary counting."""
    token = unicodedata.normalize("NFKC", token).casefold().replace("\u2019", "'")
    token = re.sub(r"'s$", "", token)
    token = token.strip("'")
    return token


# Short but interpretable tokens that would otherwise fail the vowel/length rule.
ALLOWED_SHORT_TOKENS = {
    "uk", "us", "tv", "hd", "ml", "oz", "pc", "pcs", "id", "ai", "av",
    "4k", "2d", "3d", "5g", "8k", "xl", "xs",
}


def is_content_token(token: str) -> bool:
    """Keep interpretable alphabetic/alphanumeric vocabulary tokens.

    This rejects obvious OCR noise such as single-letter fragments, purely
    numeric strings, and tokens beginning with a digit unless explicitly allowed.
    It does not attempt to spell-check genuine words.
    """
    if not ASCII_TOKEN_RE.match(token):
        return False
    if len(token) < 2:
        return False
    if token in STOPWORDS:
        return False
    if re.fullmatch(r"\d+", token):
        return False
    if token in ALLOWED_SHORT_TOKENS:
        return True
    if token[0].isdigit():
        return False
    if len(token) < 3:
        return False
    letters = re.sub(r"[^a-z]", "", token)
    if len(letters) < 2:
        return False
    if not re.search(r"[aeiouy]", letters):
        return False
    return True


def brand_token_set(brand: str) -> set[str]:
    """Exact alias tokens and concatenated aliases for a target brand."""
    tokens: set[str] = set()
    aliases = BRAND_ALIASES[brand] + [BRANDS[brand]]
    for alias in aliases:
        for token in tokenize(alias):
            normalized = normalize_token(token)
            if normalized:
                tokens.add(normalized)
        compact = re.sub(r"[^a-z0-9]", "", normalize_for_matching(alias))
        if len(compact) >= 2:
            tokens.add(compact)
    return tokens


def brand_fuzzy_bases(brand: str) -> set[str]:
    """Distinctive brand bases used to suppress likely OCR misspellings."""
    bases: set[str] = set()
    aliases = BRAND_ALIASES[brand] + [BRANDS[brand]]
    for alias in aliases:
        compact = re.sub(r"[^a-z0-9]", "", normalize_for_matching(alias))
        if len(compact) >= 4:
            bases.add(compact)
        for token in tokenize(alias):
            normalized = normalize_token(token)
            if len(normalized) >= 5:
                bases.add(normalized)
    return bases


def levenshtein_within(left: str, right: str, limit: int = 2) -> bool:
    """Return True when the edit distance is at most limit (banded DP)."""
    if abs(len(left) - len(right)) > limit:
        return False
    if left == right:
        return True
    previous = list(range(len(right) + 1))
    for row, left_char in enumerate(left, start=1):
        current = [row] + [0] * len(right)
        row_min = current[0]
        for column, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current[column] = min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + cost,
            )
            row_min = min(row_min, current[column])
        if row_min > limit:
            return False
        previous = current
    return previous[-1] <= limit


def is_brand_variant(token: str, exact: set[str], fuzzy_bases: set[str]) -> bool:
    """Exact or near-miss target-brand token treated as brand, not content."""
    if token in exact:
        return True
    for base in fuzzy_bases:
        if len(base) >= 4 and levenshtein_within(token, base, limit=2):
            return True
    return False


def parse_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    """Convert an EasyOCR quadrilateral to x_min, y_min, x_max, y_max."""
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def interval_summary(values: list[int]) -> dict[str, float]:
    """Descriptive statistics used for per-brand distributions."""
    if not values:
        return {
            "mean": float("nan"),
            "sd": float("nan"),
            "median": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    series = sorted(values)
    n = len(series)
    mean = sum(series) / n
    variance = sum((value - mean) ** 2 for value in series) / (n - 1) if n > 1 else 0.0

    def percentile(fraction: float) -> float:
        if n == 1:
            return float(series[0])
        position = (n - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, n - 1)
        weight = position - lower
        return series[lower] * (1 - weight) + series[upper] * weight

    return {
        "mean": mean,
        "sd": variance**0.5,
        "median": percentile(0.5),
        "q1": percentile(0.25),
        "q3": percentile(0.75),
        "min": float(series[0]),
        "max": float(series[-1]),
    }
