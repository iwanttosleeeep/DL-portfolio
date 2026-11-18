#!/usr/bin/env python3
"""Collect the Lane Crawford Hong Kong Home & Lifestyle brand directory.

The directory page is server-rendered and contains both the A-Z listing and a
"New Brands" block. This script makes a few ordinary requests with a normal
browser User-Agent, parses the brand entries, unions the responses,
deduplicates by brand ID, and writes data/all_brands.csv plus a short
data/README.md.

Usage:
    python3 src/collection/collect_brands.py
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import time
import unicodedata
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

SOURCE_URL = "https://www.lanecrawford.com.hk/brand/home-lifestyle/"
CATEGORY = "Home & Lifestyle"
EXPECTED_LETTERS = ["#"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)


def fetch_html(url: str, timeout: int = 30) -> str:
    """Download the directory page without attempting to bypass access controls."""
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def clean_name(raw: str) -> str:
    """Unescape HTML entities and remove only the markup's indentation."""
    return html.unescape(raw).strip()


def parse_directory(
    page: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter[str]]:
    """Return A-Z brands, New Brands brands, and counts by A-Z section."""
    # The page includes a Handlebars template inside a <script> tag. Removing
    # scripts prevents placeholder entries such as "{{ brand.brandName }}".
    page = re.sub(r"<script\b.*?</script>", "", page, flags=re.IGNORECASE | re.DOTALL)

    section_pattern = re.compile(
        r'<section\b[^>]*\bid="([^"]*)"[^>]*\bclass="[^"]*brands-list__letter[^"]*"[^>]*>'
        r"(.*?)</section>",
        re.IGNORECASE | re.DOTALL,
    )
    item_pattern = re.compile(
        r'<li\b[^>]*class="[^"]*brands-list__brand[^"]*"[^>]*>(.*?)</li>',
        re.IGNORECASE | re.DOTALL,
    )
    anchor_pattern = re.compile(
        r'<a\b[^>]*href="([^"]+)"[^>]*>.*?'
        r'<span\b[^>]*class="[^"]*brands-list__brand-name[^"]*"[^>]*>(.*?)</span>',
        re.IGNORECASE | re.DOTALL,
    )

    az_brands: list[dict[str, str]] = []
    letter_counts: Counter[str] = Counter()
    for section_id, body in section_pattern.findall(page):
        letter = html.unescape(section_id).strip()
        if letter.startswith("_"):
            letter = letter[1:]
        if not letter or "{{" in letter:
            continue
        items = item_pattern.findall(body)
        letter_counts[letter] += len(items)
        for item in items:
            match = anchor_pattern.search(item)
            if not match:
                continue
            url, name = match.groups()
            az_brands.append({"brand_name": clean_name(name), "brand_url": url})

    new_pattern = re.compile(
        r'<section\b[^>]*class="[^"]*brands-new[^"]*"[^>]*>(.*?)</section>',
        re.IGNORECASE | re.DOTALL,
    )
    new_anchor_pattern = re.compile(
        r'<a\b[^>]*href="([^"]+)"[^>]*>.*?'
        r'<span\b[^>]*class="[^"]*brands-new__brand-name[^"]*"[^>]*>(.*?)</span>',
        re.IGNORECASE | re.DOTALL,
    )
    new_brands: list[dict[str, str]] = []
    new_section = new_pattern.search(page)
    if new_section:
        for url, name in new_anchor_pattern.findall(new_section.group(1)):
            new_brands.append({"brand_name": clean_name(name), "brand_url": url})

    return az_brands, new_brands, letter_counts


def brand_identity(brand: dict[str, str]) -> str:
    """Use the numeric brand ID as identity, falling back to a normalized name."""
    match = re.search(r"/brand/(\d+)/", brand["brand_url"])
    if match:
        return f"id:{match.group(1)}"
    normalized = unicodedata.normalize("NFKC", brand["brand_name"]).casefold()
    return "name:" + re.sub(r"\s+", " ", normalized).strip()


def merge_unique(brand_lists: list[list[dict[str, str]]]) -> list[dict[str, str]]:
    """Merge brand entries, keeping the first displayed name for each identity."""
    unique: OrderedDict[str, dict[str, str]] = OrderedDict()
    for brands in brand_lists:
        for brand in brands:
            key = brand_identity(brand)
            if key not in unique:
                unique[key] = {
                    "brand_name": brand["brand_name"],
                    "brand_url": urljoin(SOURCE_URL, brand["brand_url"]),
                    "category": CATEGORY,
                }
    return list(unique.values())


def collect_directory(
    attempts: int = 4, delay: float = 1.0
) -> tuple[
    list[dict[str, str]],
    Counter[str],
    set[str],
    set[str],
    list[tuple[int, int]],
]:
    """Load the page a few times and union entries.

    The public site intermittently serves an incomplete A-Z list (398 rather
    than 407 entries on the collection date) from one backend. A small number
    of ordinary GET requests, with a delay, lets us capture the complete union
    without bypassing any access control. No single response is modified.
    """
    az_lists: list[list[dict[str, str]]] = []
    new_lists: list[list[dict[str, str]]] = []
    letter_counts: Counter[str] = Counter()
    response_counts: list[tuple[int, int]] = []

    for attempt in range(attempts):
        page = fetch_html(SOURCE_URL)
        az_brands, new_brands, counts = parse_directory(page)
        az_lists.append(az_brands)
        new_lists.append(new_brands)
        response_counts.append((len(az_brands), len(new_brands)))
        for letter, count in counts.items():
            letter_counts[letter] = max(letter_counts[letter], count)
        if attempt < attempts - 1:
            time.sleep(delay)

    # Use the largest A-Z response as the ordering base, then add any brands
    # that only appeared in the other responses. This keeps the site's A-Z
    # order while still taking the union.
    base = max(az_lists, key=len)
    other_lists = [brands for brands in az_lists if brands is not base]
    brands = merge_unique([base, *other_lists, *new_lists])

    az_ids = {brand_identity(brand) for brands in az_lists for brand in brands}
    new_ids = {brand_identity(brand) for brands in new_lists for brand in brands}
    return brands, letter_counts, az_ids, new_ids, response_counts


def validate(
    brands: list[dict[str, str]],
    letter_counts: Counter[str],
    a_z_ids: set[str],
    new_ids: set[str],
    response_counts: list[tuple[int, int]],
) -> None:
    """Fail loudly if the parsed directory is internally inconsistent."""
    errors: list[str] = []
    if not brands:
        errors.append("no brands were parsed")
    if len({brand_identity(b) for b in brands}) != len(brands):
        errors.append("duplicate brand identities remain after deduplication")
    if any(b["category"] != CATEGORY for b in brands):
        errors.append("unexpected category value")
    if any(not b["brand_name"] or not b["brand_url"] for b in brands):
        errors.append("empty brand name or URL")
    if not a_z_ids:
        errors.append("no A-Z brand IDs were parsed")
    if not new_ids:
        errors.append("no New Brands entries were parsed")
    output_ids = {brand_identity(b) for b in brands}
    missing_new = new_ids - output_ids
    if missing_new:
        errors.append(f"New Brands entries missing from output: {sorted(missing_new)}")
    missing_az = a_z_ids - output_ids
    if missing_az:
        errors.append(f"A-Z entries missing from output: {sorted(missing_az)}")
    if len(brands) < max((count for count, _ in response_counts), default=0):
        errors.append("merged output is smaller than a single page response")
    if sum(letter_counts.values()) != len(a_z_ids):
        errors.append("A-Z section counts do not match the unique A-Z brand total")

    missing_letters = [letter for letter in EXPECTED_LETTERS if letter not in letter_counts]
    unexpected_letters = [letter for letter in letter_counts if letter not in EXPECTED_LETTERS]
    if missing_letters:
        errors.append(f"missing A-Z sections: {missing_letters}")
    if unexpected_letters:
        errors.append(f"unexpected A-Z sections: {unexpected_letters}")
    if any(count == 0 for count in (letter_counts[letter] for letter in EXPECTED_LETTERS)):
        errors.append("an alphabetical section was empty")

    if errors:
        raise RuntimeError("Validation failed:\n- " + "\n- ".join(errors))


def write_csv(path: Path, brands: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["brand_name", "brand_url", "category"]
        )
        writer.writeheader()
        writer.writerows(brands)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_readme(
    path: Path,
    brands: list[dict[str, str]],
    letter_counts: Counter[str],
    new_count: int,
    new_overlap: int,
    response_counts: list[tuple[int, int]],
    collected_at: str,
) -> None:
    missing_letters = [letter for letter in EXPECTED_LETTERS if letter not in letter_counts]
    special_names = [b["brand_name"] for b in brands if any(ord(c) > 127 for c in b["brand_name"])]
    counts = ", ".join(f"{letter}: {letter_counts[letter]}" for letter in EXPECTED_LETTERS
                       if letter in letter_counts)
    responses = "; ".join(
        f"request {index}: {a_z_count} A-Z + {brand_count} New Brands"
        for index, (a_z_count, brand_count) in enumerate(response_counts, start=1)
    )
    content = f"""# Lane Crawford Home & Lifestyle Brands

- **Source URL:** {SOURCE_URL}
- **Collection date:** {collected_at}
- **Total unique brands:** {len(brands)}
- **Category:** {CATEGORY}

## Extraction method

The directory page was requested up to four times with ordinary HTTP GET
requests using a normal browser User-Agent and a one-second delay. The server-rendered
HTML was parsed with Python's standard library: every `brands-list__letter`
section (`#`, A-Z) and the `brands-new` section were read, HTML entities were
unescaped, and brand URLs were resolved to absolute
`https://www.lanecrawford.com.hk` URLs. Entries were deduplicated by the numeric
brand ID in each brand-page URL, preserving the first displayed brand name.
The responses were combined by union. The reproducible script is
`src/collection/collect_brands.py`.

## Validation

- Responses: {responses}.
- Alphabetical sections found: {len(letter_counts)} ({", ".join(letter_counts)}).
- Brands per section (maximum seen across responses): {counts}.
- Missing alphabetical sections: {", ".join(missing_letters) if missing_letters else "none"}.
- New Brands identities seen across responses: {new_count}; {new_overlap} also
  appear in the A-Z listing and were removed as duplicates.
- CSV was read back after writing and the row count matched the unique brand count.
- Original capitalization, punctuation, accents, and special characters were
  preserved (for example: {", ".join(special_names[:8])}).

## Access limitations / notes

- The figures reflect the public Hong Kong site on the collection date and may
  change as the brand directory is updated.
- The site intermittently served an incomplete A-Z directory from one backend
  (398 entries rather than 407). The union of the ordinary page loads was
  used to avoid dropping valid brands; no response was modified.
- Only publicly served HTML was used; no login, anti-bot bypass, or hidden API
  was used.
- Every extracted entry had a brand-page URL, so there was no missing brand URL
  at collection time.
- Brand names and availability can vary by store filter; the default "All
  Stores" directory was used.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/all_brands.csv"),
        help="CSV output path (default: data/all_brands.csv)",
    )
    parser.add_argument(
        "--readme", type=Path, default=Path("data/README.md"),
        help="README output path (default: data/README.md)",
    )
    args = parser.parse_args()

    brands, letter_counts, a_z_ids, new_ids, response_counts = collect_directory()
    new_overlap = len(a_z_ids & new_ids)
    validate(brands, letter_counts, a_z_ids, new_ids, response_counts)

    write_csv(args.output, brands)

    # Re-read the written CSV so validation covers the actual artifact.
    written = read_csv(args.output)
    if len(written) != len(brands) or list(written[0]) != ["brand_name", "brand_url", "category"]:
        raise RuntimeError("CSV read-back validation failed")

    collected_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    write_readme(
        args.readme,
        brands,
        letter_counts,
        len(new_ids),
        new_overlap,
        response_counts,
        collected_at,
    )

    print(f"Source: {SOURCE_URL}")
    print("Responses (A-Z, New Brands): " + ", ".join(
        f"{a_z} + {new}" for a_z, new in response_counts
    ))
    print(f"Unique A-Z brand IDs: {len(a_z_ids)}")
    print(f"Unique New Brands IDs: {len(new_ids)} "
          f"({new_overlap} already in A-Z)")
    print(f"Unique brands written: {len(brands)}")
    print(f"Alphabetical sections: {len(letter_counts)} "
          f"({', '.join(letter_counts)})")
    print(f"CSV: {args.output}")
    print(f"README: {args.readme}")


if __name__ == "__main__":
    main()
