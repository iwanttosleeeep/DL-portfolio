#!/usr/bin/env python3
"""Match Assignment 4 brands and verify their Lane Crawford listings.

Reads the shared directory at data/all_brands.csv, matches the ten selected
brands by normalized name, then checks each official brand page for its
Home & Lifestyle classification and current product listing. Product detail
pages are sampled to confirm online availability.

Output: assignment-4/data/selected_brands.csv
"""

from __future__ import annotations

import csv
import html
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

DIRECTORY_CSV = Path("data/all_brands.csv")
OUTPUT_CSV = Path("assignment-4/data/selected_brands.csv")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
SELECTED_BRANDS = [
    "Jellycat",
    "Stanley",
    "LEGO",
    "JO MALONE LONDON",
    "Marshall",
    "Bang & Olufsen",
    "Polaroid",
    "Fortnum & Mason",
    "Aesop",
    "JULY",
]
MAX_PRODUCTS_SAMPLED = 3
CROSS_BRAND_CHECKS = {
    "BE@RBRICK": {
        "brand_name": "TOYQUBE",
        "brand_url": "https://www.lanecrawford.com.hk/brand/92203753/toyqube/lifestyle/",
        "match_terms": ["BE@RBRICK", "BEARBRICK"],
    }
}


def normalize_name(value: str) -> str:
    """Case-, accent-, and punctuation-insensitive comparison key."""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


def fetch(url: str, timeout: int = 45) -> tuple[int, str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return (
            response.status,
            response.geturl(),
            response.read().decode("utf-8", errors="replace"),
        )


def parse_plp(page: str) -> dict:
    match = re.search(r"var plpData = (\{.*?\});\s*\n", page, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def fetch_listing(base_url: str) -> tuple[dict, list[dict]]:
    """Fetch every page of a brand PLP and return its data and products."""
    _, _, first_page = fetch(base_url)
    first_plp = parse_plp(first_page)
    products = list(first_plp.get("productList", []))
    page_count = int(first_plp.get("pagination", {}).get("length", 1) or 1)

    for page_number in range(2, page_count + 1):
        page_url = f"{base_url.rstrip('/')}/page/{page_number}/"
        _, _, page = fetch(page_url)
        plp = parse_plp(page)
        products.extend(plp.get("productList", []))
        time.sleep(0.5)

    unique = {}
    for product in products:
        unique[product.get("productId", "")] = product
    return first_plp, list(unique.values())


def classify_page(plp: dict) -> str:
    bu_titles = {
        str(item.get("title", "")).casefold()
        for item in plp.get("buLinkList", [])
    }
    dimensions = {
        str(item.get("dimensionName", "")).casefold()
        for item in plp.get("availableBU", [])
    }
    if "home & lifestyle" in bu_titles or "lifestyle" in dimensions:
        return "Yes"
    return "No"


def check_product_availability(page: str) -> tuple[str, dict[str, str], bool]:
    availability_match = re.search(
        r'itemprop="availability" content="([^"]+)"', page
    )
    availability = availability_match.group(1) if availability_match else ""
    attrs = dict(
        re.findall(
            r'(data-(?:availability-status|product-status|is-normal-status|'
            r'is-collect-only|show-collect|show-deliver))="([^"]*)"',
            page,
        )
    )
    online_available = (
        availability in {"InStock", "OnlineOnly"}
        and attrs.get("data-is-normal-status") == "true"
        and attrs.get("data-show-deliver") == "true"
    )
    return availability, attrs, online_available


def match_brand(directory: list[dict[str, str]], requested: str) -> dict[str, str] | None:
    key = normalize_name(requested)
    matches = [row for row in directory if normalize_name(row["brand_name"]) == key]
    return matches[0] if matches else None


def apply_cross_check(row: dict[str, str], notes: list[str]) -> None:
    """Check a related Lane Crawford brand for products matching this brand."""
    spec = CROSS_BRAND_CHECKS.get(row["brand_name"])
    if not spec:
        return

    row["cross_check_brand_name"] = spec["brand_name"]
    row["cross_check_brand_url"] = spec["brand_url"]
    try:
        plp, products = fetch_listing(spec["brand_url"])
    except Exception as exc:  # noqa: BLE001 - report access failures as data
        row["cross_check_availability_status"] = "Unable to verify"
        row["cross_check_notes"] = f"Cross-check request failed: {exc}"
        notes.append("Related-brand cross-check failed.")
        return

    total = int(plp.get("content", {}).get("length", len(products)) or 0)
    terms = [term.casefold() for term in spec["match_terms"]]
    matches = [
        product
        for product in products
        if any(
            term in str(product.get("productDisplayName", "")).casefold()
            for term in terms
        )
    ]
    row["cross_check_total_products"] = str(total)
    row["cross_check_matching_products"] = str(len(matches))

    if not matches:
        row["cross_check_availability_status"] = "No matching products"
        row["cross_check_example_product_url"] = ""
        row["cross_check_notes"] = (
            f"{spec['brand_name']} has {total} current lifestyle products; none "
            "have BE@RBRICK/BEARBRICK in the product name."
        )
        notes.append(
            f"{spec['brand_name']} cross-check found no matching product names."
        )
        return

    row["cross_check_example_product_url"] = urljoin(
        spec["brand_url"], matches[0]["productUrl"]
    )
    row["cross_check_availability_status"] = (
        "Matching products found; online availability not confirmed"
    )
    for product in matches[:MAX_PRODUCTS_SAMPLED]:
        product_url = urljoin(spec["brand_url"], product["productUrl"])
        try:
            _, _, product_page = fetch(product_url)
            availability, attrs, online_available = check_product_availability(
                product_page
            )
        except Exception:  # noqa: BLE001 - keep checking other matches
            continue
        if online_available:
            row["cross_check_availability_status"] = (
                "Matching products found; online availability confirmed"
            )
            row["cross_check_example_product_url"] = product_url
            row["cross_check_notes"] = (
                f"{product['productId']} | {product['productDisplayName']} | "
                f"availability={availability} | "
                f"deliver={attrs.get('data-show-deliver')}"
            )
            break
    else:
        row["cross_check_notes"] = (
            f"Matched {len(matches)} product(s), but sampled product pages did not "
            "confirm online availability."
        )
    notes.append(
        f"{spec['brand_name']} cross-check found {len(matches)} matching product(s)."
    )


def verify_brand(requested: str, match: dict[str, str] | None) -> dict[str, str]:
    row = {
        "requested_name": requested,
        "brand_name": "",
        "brand_url": "",
        "match_status": "Not matched",
        "category": "",
        "category_verified": "No",
        "brand_page_status": "",
        "products_listed": "No",
        "product_count": "0",
        "product_availability_status": "Unable to verify",
        "availability_evidence": "",
        "cross_check_brand_name": "",
        "cross_check_brand_url": "",
        "cross_check_total_products": "",
        "cross_check_matching_products": "",
        "cross_check_example_product_url": "",
        "cross_check_availability_status": "",
        "cross_check_notes": "",
        "notes": "",
    }
    if not match:
        row["notes"] = "No normalized name match in data/all_brands.csv."
        return row

    row["brand_name"] = match["brand_name"]
    row["brand_url"] = match["brand_url"]
    row["category"] = match["category"]
    row["match_status"] = (
        "Exact match"
        if requested == match["brand_name"]
        else "Matched (case/punctuation difference)"
    )

    try:
        status, final_url, page = fetch(match["brand_url"])
    except Exception as exc:  # noqa: BLE001 - report access failures as data
        row["product_availability_status"] = "Unable to verify"
        row["notes"] = f"Brand page request failed: {exc}"
        return row
    row["brand_page_status"] = str(status)

    plp = parse_plp(page)
    product_list = plp.get("productList", [])
    product_count = int(plp.get("content", {}).get("length", len(product_list)) or 0)
    row["product_count"] = str(product_count)
    row["products_listed"] = "Yes" if product_count else "No"
    row["category_verified"] = classify_page(plp)

    title_match = re.search(r"<title>\s*(.*?)\s*-\s*", page, re.DOTALL)
    title = html.unescape(title_match.group(1)).strip() if title_match else ""
    notes: list[str] = []
    if status != 200:
        notes.append(f"Brand page HTTP status {status}.")
    if final_url != match["brand_url"]:
        notes.append(f"Redirected to {final_url}.")
    if title and normalize_name(title) != normalize_name(match["brand_name"]):
        notes.append(f"Page title was {title!r}.")
    if row["category_verified"] != "Yes":
        notes.append("Home & Lifestyle BU was not present in the brand page data.")

    if not product_count:
        row["product_availability_status"] = (
            "No products listed; online availability not confirmed"
        )
        row["availability_evidence"] = "Brand page returned 0 products."
        notes.append("Requires review: brand is in the directory but has no current products.")
    else:
        for product in product_list[:MAX_PRODUCTS_SAMPLED]:
            product_url = urljoin(match["brand_url"], product["productUrl"])
            try:
                _, _, product_page = fetch(product_url)
                availability, attrs, online_available = check_product_availability(
                    product_page
                )
            except Exception as exc:  # noqa: BLE001 - keep checking other samples
                availability, attrs, online_available = "", {}, False
                notes.append(f"Product request failed for {product['productId']}: {exc}")
                continue

            if online_available:
                row["product_availability_status"] = (
                    "Products listed; online availability confirmed"
                )
                row["availability_evidence"] = (
                    f"{product['productId']} | {product['productDisplayName']} | "
                    f"{product_url} | availability={availability} | "
                    f"deliver={attrs.get('data-show-deliver')}"
                )
                break

        if not row["availability_evidence"]:
            row["product_availability_status"] = (
                "Products listed; online availability not confirmed in sampled products"
            )
            row["availability_evidence"] = (
                f"Sampled {min(MAX_PRODUCTS_SAMPLED, len(product_list))} product(s); "
                "none showed online availability."
            )
            notes.append("Requires review: confirm availability on the brand page.")

    if match["brand_name"] == "AESOP":
        notes.append(
            "Dual-listed under Beauty and Home & Lifestyle; verify intended category scope."
        )

    apply_cross_check(row, notes)
    row["notes"] = " ".join(notes)
    return row


def main() -> None:
    directory = list(csv.DictReader(DIRECTORY_CSV.open(encoding="utf-8")))
    rows = []
    for index, requested in enumerate(SELECTED_BRANDS):
        match = match_brand(directory, requested)
        rows.append(verify_brand(requested, match))
        if index < len(SELECTED_BRANDS) - 1:
            time.sleep(1.0)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "requested_name",
        "brand_name",
        "brand_url",
        "match_status",
        "category",
        "category_verified",
        "brand_page_status",
        "products_listed",
        "product_count",
        "product_availability_status",
        "availability_evidence",
        "cross_check_brand_name",
        "cross_check_brand_url",
        "cross_check_total_products",
        "cross_check_matching_products",
        "cross_check_example_product_url",
        "cross_check_availability_status",
        "cross_check_notes",
        "notes",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Verified at: {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}")
    print(f"Matched: {sum(r['match_status'] != 'Not matched' for r in rows)}/{len(rows)}")
    print(
        "Online availability confirmed: "
        f"{sum(r['product_availability_status'] == 'Products listed; online availability confirmed' for r in rows)}"
    )
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
