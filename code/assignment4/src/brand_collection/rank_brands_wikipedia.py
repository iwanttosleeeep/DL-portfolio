#!/usr/bin/env python3
"""Rank Lane Crawford brands by English Wikipedia popularity.

For every brand in data/all_brands.csv this script:

1. Asks the English Wikipedia search API for the single best result.
2. Keeps that result only when the title clearly matches the brand and the
   page is about a company or brand (verified from page categories and the
   Wikidata short description returned by the API).
3. Sums monthly user pageviews for the last 12 complete months from the
   Wikimedia Pageviews REST API.
4. Writes assignment-4/data/brand_ranking.csv sorted by total pageviews.

The script deliberately does not hard-code any article titles or view counts;
all titles, categories, descriptions, and numbers come from the APIs at run
time.

Usage:
    python3 src/collection/rank_brands_wikipedia.py
    python3 src/collection/rank_brands_wikipedia.py --limit 20 --output /tmp/test.csv
"""

from __future__ import annotations

import argparse
import calendar
import csv
import json
import re
import time
import unicodedata
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

INPUT_CSV = Path("data/all_brands.csv")
OUTPUT_CSV = Path("assignment-4/data/brand_ranking.csv")
USER_AGENT = (
    "STAT6207-Assignment4-BrandRanking/1.0 "
    "(educational research; en.wikipedia pageview comparison; "
    "contact: stat6207-student@example.com)"
)
REQUEST_DELAY_SECONDS = 0.2
SEARCH_API = "https://en.wikipedia.org/w/api.php"
PAGEVIEWS_API = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/monthly/{start}/{end}"
)

CATEGORY_TERMS = (
    "company",
    "companies",
    "brand",
    "brands",
    "retailer",
    "retailers",
    "manufacturer",
    "manufacturers",
    "conglomerate",
    "corporation",
    "corporations",
    "enterprise",
    "enterprises",
    "fashion house",
    "fashion houses",
    "luxury goods",
    "brewery",
    "breweries",
    "distillery",
    "distilleries",
    "winery",
    "wineries",
    "champagne",
    "producer",
    "producers",
    "watchmaker",
    "watchmakers",
    "jeweller",
    "jewellers",
    "jewelry",
    "jewellery",
    "perfumery",
    "publisher",
    "publishing companies",
)

DESCRIPTION_TERMS = (
    "company",
    "brand",
    "manufacturer",
    "retailer",
    "conglomerate",
    "corporation",
    "enterprise",
    "fashion house",
    "luxury",
    "brewery",
    "distillery",
    "winery",
    "champagne",
    "watchmaker",
    "jeweller",
    "jewelry",
    "perfumery",
    "publisher",
    "record label",
)

# Normalise common letters that NFKD does not decompose into ASCII.
TRANSLITERATION = str.maketrans(
    {
        "ø": "o",
        "Ø": "O",
        "œ": "oe",
        "Œ": "OE",
        "æ": "ae",
        "Æ": "AE",
        "ß": "ss",
        "đ": "d",
        "Đ": "D",
        "ð": "d",
        "Ð": "D",
        "ł": "l",
        "Ł": "L",
        "þ": "th",
        "Þ": "TH",
        "ı": "i",
        "@": "a",
    }
)

LEGAL_SUFFIXES = (
    "ltd.",
    "ltd",
    "limited",
    "inc.",
    "inc",
    "corp.",
    "corp",
    "sl",
    "s.l.",
    "gmbh",
    "s.a.",
    "sa",
    "spa",
    "a/s",
    "as",
    "plc",
)


class RateLimiter:
    """Enforce a minimum delay between all outgoing requests."""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.last_request = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_request = time.monotonic()


def normalize(value: str) -> str:
    """Case-, accent-, and punctuation-insensitive comparison key."""
    value = value.translate(TRANSLITERATION)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def brand_variants(brand_name: str) -> list[tuple[str, str]]:
    """Return plausible name variants and their normalized forms."""
    raw_variants = [brand_name]

    without_parenthetical = re.sub(r"\s*\([^)]*\)\s*$", "", brand_name).strip()
    if without_parenthetical and without_parenthetical != brand_name:
        raw_variants.append(without_parenthetical)

    if ":" in brand_name:
        before_colon = brand_name.split(":", 1)[0].strip()
        if before_colon:
            raw_variants.append(before_colon)

    for variant in list(raw_variants):
        lowered = variant.casefold().strip()
        for suffix in LEGAL_SUFFIXES:
            if lowered.endswith(" " + suffix) and len(lowered) > len(suffix) + 3:
                raw_variants.append(variant[: -(len(suffix) + 1)].strip())
                break

    variants: list[tuple[str, str]] = []
    seen: set[str] = set()
    for variant in raw_variants:
        normalized = normalize(variant)
        if normalized and normalized not in seen:
            seen.add(normalized)
            variants.append((variant, normalized))
    return variants


def contains_contiguous(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    width = len(needle)
    return any(haystack[index : index + width] == needle for index in range(len(haystack) - width + 1))


def title_matches_brand(title: str, brand_name: str) -> tuple[bool, str | None]:
    """Check whether a Wikipedia title clearly contains the brand name."""
    title_without_parenthetical = re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()
    title_normalized = normalize(title_without_parenthetical)
    title_tokens = title_normalized.split()

    for variant, variant_normalized in brand_variants(brand_name):
        if not variant_normalized:
            continue
        variant_tokens = variant_normalized.split()

        if title_normalized == variant_normalized:
            return True, variant

        # A one-word brand may appear as one word inside a longer article title,
        # e.g. "Beats" -> "Beats Electronics".
        if len(variant_tokens) == 1 and variant_tokens[0] in title_tokens:
            return True, variant

        # A multi-word brand should appear as a contiguous phrase. This catches
        # titles such as "Bang & Olufsen" without matching accidental fragments.
        if len(variant_tokens) >= 2 and contains_contiguous(title_tokens, variant_tokens):
            return True, variant

    return False, None


def request_json(
    url: str,
    limiter: RateLimiter,
    *,
    allow_404: bool = False,
    attempts: int = 4,
    timeout: int = 40,
):
    """Fetch JSON with retries while keeping the requested request spacing."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        limiter.wait()
        request = Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            last_error = error
            if error.code == 404 and allow_404:
                return None
            if error.code in {429, 500, 502, 503, 504} and attempt < attempts - 1:
                time.sleep(2**attempt)
                continue
            raise
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < attempts - 1:
                time.sleep(2**attempt)
                continue
            raise

    assert last_error is not None
    raise last_error


def search_first_result(brand_name: str, limiter: RateLimiter) -> dict | None:
    """Return the single top search result plus page categories/description."""
    query = urlencode(
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": brand_name,
            "gsrlimit": "1",
            "gsrnamespace": "0",
            "prop": "categories|pageprops",
            "cllimit": "max",
            "clshow": "!hidden",
            "redirects": "1",
            "format": "json",
            "formatversion": "2",
        }
    )
    payload = request_json(f"{SEARCH_API}?{query}", limiter)
    pages = payload.get("query", {}).get("pages") or []
    if not pages:
        return None
    return sorted(pages, key=lambda page: page.get("index", 0))[0]


def is_person_page(page: dict) -> bool:
    """Reject biographies even when their categories mention company founders."""
    pageprops = page.get("pageprops") or {}
    description = pageprops.get("wikibase-shortdesc", "").casefold()
    if re.search(r"\(born\s+\d{4}\)", description):
        return True

    for category in page.get("categories") or []:
        name = category.get("title", "").removeprefix("Category:").casefold()
        if name == "living people" or "births" in name or "deaths" in name:
            return True
    return False


def company_or_brand_evidence(page: dict) -> tuple[bool, list[str]]:
    """Decide from API metadata whether a page is about a company or brand."""
    evidence: list[str] = []

    categories = page.get("categories") or []
    for category in categories:
        category_name = category.get("title", "").removeprefix("Category:")
        lowered = category_name.casefold()
        if any(term in lowered for term in CATEGORY_TERMS):
            evidence.append(f"category '{category_name}'")

    description = (page.get("pageprops") or {}).get("wikibase-shortdesc", "")
    if description:
        lowered = description.casefold()
        if any(term in lowered for term in DESCRIPTION_TERMS):
            evidence.append(f"description '{description}'")

    return bool(evidence), evidence


def classify_brand(brand_name: str, limiter: RateLimiter) -> dict:
    """Run steps 1-2 for one brand and return a result row."""
    page = search_first_result(brand_name, limiter)

    if page is None:
        return {
            "brand_name": brand_name,
            "wiki_title": "not_found",
            "match_reason": "no search results returned; not_found",
            "pageviews_12m": 0,
        }

    title = page.get("title", "")
    pageprops = page.get("pageprops") or {}

    if "disambiguation" in pageprops:
        return {
            "brand_name": brand_name,
            "wiki_title": "not_found",
            "match_reason": (
                f"first result '{title}' is a disambiguation page, not a company or brand; "
                "not_found"
            ),
            "pageviews_12m": 0,
        }

    title_ok, matched_variant = title_matches_brand(title, brand_name)
    if not title_ok:
        return {
            "brand_name": brand_name,
            "wiki_title": "not_found",
            "match_reason": (
                f"first result '{title}' does not clearly contain the brand name; not_found"
            ),
            "pageviews_12m": 0,
        }

    if is_person_page(page):
        return {
            "brand_name": brand_name,
            "wiki_title": "not_found",
            "match_reason": (
                f"first result '{title}' matches '{matched_variant}' but is a biography "
                "of a person, not a company or brand; not_found"
            ),
            "pageviews_12m": 0,
        }

    is_company_or_brand, evidence = company_or_brand_evidence(page)
    if not is_company_or_brand:
        description = pageprops.get("wikibase-shortdesc", "no short description")
        return {
            "brand_name": brand_name,
            "wiki_title": "not_found",
            "match_reason": (
                f"first result '{title}' matches '{matched_variant}' but is not a company "
                f"or brand (description: {description}); not_found"
            ),
            "pageviews_12m": 0,
        }

    pageviews = pageviews_for_title(title, limiter)
    evidence_text = "; ".join(evidence[:3])
    return {
        "brand_name": brand_name,
        "wiki_title": title,
        "match_reason": (
            f"first result; title matches brand variant '{matched_variant}'; "
            f"company/brand evidence: {evidence_text}"
        ),
        "pageviews_12m": pageviews,
    }


def last_12_full_months(today: date) -> tuple[str, str]:
    """Return YYYYMMDD bounds for the 12 months before the current month."""
    end_year = today.year
    end_month = today.month - 1
    if end_month == 0:
        end_year -= 1
        end_month = 12

    end_day = calendar.monthrange(end_year, end_month)[1]

    start_year = end_year
    start_month = end_month - 11
    while start_month <= 0:
        start_year -= 1
        start_month += 12

    return (
        f"{start_year:04d}{start_month:02d}01",
        f"{end_year:04d}{end_month:02d}{end_day:02d}",
    )


def pageviews_for_title(title: str, limiter: RateLimiter) -> int:
    """Sum monthly user pageviews for the last 12 complete months."""
    start, end = last_12_full_months(date.today())
    article = quote(title.replace(" ", "_"), safe="")
    url = PAGEVIEWS_API.format(article=article, start=start, end=end)
    payload = request_json(url, limiter, allow_404=True)
    if not payload:
        return 0
    return sum(int(item.get("views", 0)) for item in payload.get("items", []))


def read_brands(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row["brand_name"].strip() for row in reader if row.get("brand_name", "").strip()]


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    results = sorted(
        results,
        key=lambda row: (-int(row["pageviews_12m"]), row["brand_name"].casefold()),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["brand_name", "wiki_title", "match_reason", "pageviews_12m"],
        )
        writer.writeheader()
        writer.writerows(results)


def print_top(results: list[dict], count: int = 40) -> None:
    ranked = sorted(
        results,
        key=lambda row: (-int(row["pageviews_12m"]), row["brand_name"].casefold()),
    )
    print(
        f"{'rank':>4}  {'brand_name':<29}  {'wiki_title':<38}  "
        f"{'pageviews_12m':>13}  match_reason"
    )
    print("-" * 150)
    for index, row in enumerate(ranked[:count], start=1):
        print(
            f"{index:>4}  {row['brand_name']:<29}  {row['wiki_title']:<38}  "
            f"{int(row['pageviews_12m']):>13,}  {row['match_reason']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N brands (useful for a quick test).",
    )
    args = parser.parse_args()

    brands = read_brands(args.input)
    if args.limit is not None:
        brands = brands[: args.limit]

    limiter = RateLimiter(REQUEST_DELAY_SECONDS)
    results: list[dict] = []
    total = len(brands)

    for position, brand_name in enumerate(brands, start=1):
        try:
            result = classify_brand(brand_name, limiter)
        except Exception as error:  # Keep a long run alive; report the failure clearly.
            result = {
                "brand_name": brand_name,
                "wiki_title": "not_found",
                "match_reason": f"request failed: {type(error).__name__}: {error}; not_found",
                "pageviews_12m": 0,
            }
        results.append(result)
        print(
            f"[{position:>3}/{total}] {brand_name} -> {result['wiki_title']} "
            f"({int(result['pageviews_12m']):,})",
            flush=True,
        )

    write_results(results, args.output)
    print(f"\nWrote {len(results)} rows to {args.output}")
    print(f"Pageview window: {last_12_full_months(date.today())[0]} to "
          f"{last_12_full_months(date.today())[1]}\n")
    print_top(results, count=40)


if __name__ == "__main__":
    main()
