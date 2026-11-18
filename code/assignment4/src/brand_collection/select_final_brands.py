#!/usr/bin/env python3
"""Select ten Assignment 4 brands from the Wikipedia ranking by rule.

The selection is deterministic:

1. Jellycat (toys) and Stanley (drinkware) are required.
2. Walk assignment-4/data/brand_ranking.csv from the highest pageview count.
3. Skip rows whose Wikipedia title is not the Lane Crawford brand.
4. Verify each candidate's Lane Crawford page with the functions in
   verify_selected_brands.py and skip brands with no products listed.
5. Assign one category and allow at most two selected brands per category,
   including the two required brands.
6. Stop at ten selected brands.

Outputs:
  assignment-4/data/selected_brands.csv
  assignment-4/data/brand_ranking.csv  (adds selected/category/on_sale/skip_reason)

Usage:
    python3 src/collection/select_final_brands.py
"""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import verify_selected_brands as verify

RANKING_CSV = Path("assignment-4/data/brand_ranking.csv")
DIRECTORY_CSV = Path("data/all_brands.csv")
SELECTED_CSV = Path("assignment-4/data/selected_brands.csv")
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_USER_AGENT = (
    "STAT6207-Assignment4-BrandSelection/1.0 "
    "(educational research; en.wikipedia subject check; "
    "contact: stat6207-student@example.com)"
)
REQUEST_DELAY_SECONDS = 0.2
VERIFY_DELAY_SECONDS = 0.5
MAX_SELECTED = 10
MAX_PER_CATEGORY = 2

REQUIRED = {
    "jellycat": "toys",
    "stanley": "drinkware",
}

BASE_RANKING_FIELDS = [
    "brand_name",
    "wiki_title",
    "match_reason",
    "pageviews_12m",
]
EXTRA_RANKING_FIELDS = ["selected", "category", "on_sale", "skip_reason"]
SELECTED_FIELDS = [
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
    "assigned_category",
]

# Subjects that clearly are not the Lane Crawford Home & Lifestyle brand when
# they appear in the Wikipedia description or categories. These are name
# collisions, not industry guesses about legitimate brands.
MISMATCH_SUBJECT_TERMS = [
    "restaurant",
    "restaurants",
    "electric power",
    "utility",
    "utilities",
    "software",
    "sewing machine",
    "sewing machines",
    "airline",
    "airlines",
    "airport",
    "bank",
    "banks",
    "insurance",
    "railway",
    "railways",
    "metro",
    "football club",
    "football clubs",
    "sports club",
    "sports clubs",
    "political party",
    "political parties",
    "university",
    "universities",
    "college",
    "colleges",
    "school",
    "schools",
    "hospital",
    "hospitals",
    "museum",
    "museums",
    "church",
    "churches",
    "village",
    "villages",
    "lakes",
    "rivers",
    "mountains",
    "genus",
    "species",
    "album",
    "albums",
    "song",
    "songs",
    "novel",
    "novels",
    "newspaper",
    "newspapers",
    "magazine",
    "magazines",
    "video game",
    "video games",
    "television series",
    "aircraft",
    "ships",
    "military",
    "weapons",
]

CATEGORY_RULES = [
    (
        "toys",
        [
            "toy", "toys", "plush", "plushie", "plushies", "teddy",
            "teddies", "lego", "jellycat", "construction toy",
            "construction toys", "collectible toy", "collectible toys",
            "doll", "dolls", "stuffed toy", "stuffed toys", "puzzle",
            "puzzles", "board game", "board games",
        ],
    ),
    (
        "drinkware",
        [
            "drinkware", "tumbler", "tumblers", "bottle", "bottles",
            "water bottle", "water bottles", "flask", "flasks", "thermos",
            "cup", "cups", "mug", "mugs",
        ],
    ),
    (
        "cameras & imaging",
        [
            "camera", "cameras", "photographic", "photography",
            "film camera", "film cameras", "lens", "lenses", "drone",
            "drones", "imaging", "camcorder", "camcorders", "action camera",
            "action cameras", "photo",
        ],
    ),
    (
        "audio",
        [
            "audio", "speaker", "speakers", "loudspeaker", "loudspeakers",
            "headphone", "headphones", "earphone", "earphones", "earbud",
            "earbuds", "turntable", "turntables", "phonograph", "radio",
            "radios", "amplifier", "amplifiers", "musical instrument",
            "musical instruments", "sound",
        ],
    ),
    (
        "fashion",
        [
            "fashion", "clothing", "apparel", "jewellery", "jewelry",
            "jeweller", "jewellers", "jeweler", "jewelers", "handbag",
            "handbags", "bag", "bags", "shoe", "shoes", "sneaker",
            "sneakers", "bracelet", "bracelets", "necklace", "necklaces",
            "earring", "earrings", "watch", "watches", "scarf", "scarves",
            "sunglasses", "wallet", "wallets", "belt", "belts", "dress",
            "dresses", "shirt", "shirts", "jacket", "jackets", "coat",
            "coats", "trouser", "trousers", "skirt", "skirts", "couture",
            "accessory", "accessories",
        ],
    ),
    (
        "fragrance",
        [
            "fragrance", "fragrances", "perfume", "perfumes", "candle",
            "candles", "eau de parfum", "eau de toilette", "scented",
            "diffuser", "diffusers", "incense", "home fragrance",
        ],
    ),
    (
        "alcohol",
        [
            "champagne", "cognac", "whisky", "whiskey", "wine", "wines",
            "winery", "wineries", "sake", "gin", "vodka", "rum", "beer",
            "beers", "brewery", "breweries", "distillery", "distilleries",
            "liquor", "spirit", "spirits", "brandy", "tequila", "alcoholic",
        ],
    ),
    (
        "food",
        [
            "tea", "teas", "coffee", "chocolate", "chocolates", "biscuit",
            "biscuits", "marmalade", "shortbread", "jam", "jams", "food",
            "foods", "gourmet", "delicatessen", "delicatessens", "grocery",
            "groceries", "cake", "cakes", "cookie", "cookies", "honey",
            "hamper", "hampers", "oil", "oils", "vinegar", "sauce",
            "sauces", "confectionery",
        ],
    ),
    (
        "furniture",
        [
            "furniture", "chair", "chairs", "table", "tables", "sofa",
            "sofas", "bed", "beds", "cabinet", "cabinets", "stool",
            "stools", "desk", "desks", "shelf", "shelves", "bookcase",
            "bookcases", "lamp", "lamps", "lighting", "mirror", "mirrors",
            "rug", "rugs", "cushion", "cushions", "duvet", "duvets",
            "pillow", "pillows", "blanket", "blankets", "throw", "throws",
            "bedding", "mattress", "mattresses", "dresser", "dressers",
            "armchair", "armchairs", "ottoman", "ottomans", "bench",
            "benches", "sideboard", "sideboards", "wardrobe", "wardrobes",
            "home furnishing", "decor", "ceramic", "ceramics", "tableware",
            "glassware", "plate", "plates", "bowl", "bowls", "teapot",
            "teapots", "vase", "vases", "frame", "frames", "silverware",
            "chopstick", "chopsticks", "tray", "trays",
        ],
    ),
    (
        "personal electronics",
        [
            "electronics", "electronic", "technology", "activity tracker",
            "activity trackers", "tracker", "trackers", "wearable",
            "wearables", "charger", "chargers", "power bank", "power banks",
            "phone", "phones", "smartphone", "smartphones", "tablet",
            "tablets", "laptop", "laptops", "battery", "batteries", "cable",
            "cables", "adapter", "adapters", "keyboard", "keyboards",
            "mouse", "monitor", "monitors", "fitness",
        ],
    ),
]


def phrase_in_text(phrase: str, text: str) -> bool:
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
    return re.search(pattern, text) is not None


def fetch_wiki_subject(title: str, limiter: "RateLimiter") -> tuple[str, list[str]]:
    params = {
        "action": "query",
        "prop": "categories|pageprops",
        "titles": title,
        "format": "json",
        "formatversion": "2",
        "cllimit": "max",
        "clshow": "!hidden",
        "redirects": "1",
    }
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    limiter.wait()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": WIKI_USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=40) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pages = payload.get("query", {}).get("pages") or []
    if not pages:
        return "", []
    page = pages[0]
    description = (page.get("pageprops") or {}).get("wikibase-shortdesc", "")
    categories = [
        item["title"].removeprefix("Category:")
        for item in page.get("categories", [])
    ]
    return description, categories


class RateLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.last_request = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self.last_request = time.monotonic()


def mismatch_reason(description: str, categories: list[str]) -> str:
    # Use the short description as the primary subject. Categories alone are
    # too noisy: for example Fitbit has Android/iOS software categories but is
    # an activity-tracker brand, not a software company.
    text = (description or " ".join(categories)).casefold()
    hits = [term for term in MISMATCH_SUBJECT_TERMS if phrase_in_text(term, text)]
    if not hits:
        return ""
    subject = description or ", ".join(categories[:2]) or hits[0]
    return (
        "mismatched title: Wikipedia page is about "
        f"{subject!r} ({', '.join(hits[:3])}), not the Lane Crawford brand"
    )


def classify_category(
    brand_name: str,
    description: str,
    categories: list[str],
    product_text: str,
) -> str | None:
    normalized_brand = verify.normalize_name(brand_name)
    required_override = REQUIRED.get(normalized_brand)
    if required_override:
        return required_override

    wiki_text = f"{brand_name} {description} {' '.join(categories)}".casefold()
    product_text = product_text.casefold()

    for category, keywords in CATEGORY_RULES:
        if any(phrase_in_text(keyword, wiki_text) for keyword in keywords):
            return category

    for category, keywords in CATEGORY_RULES:
        if any(phrase_in_text(keyword, product_text) for keyword in keywords):
            return category

    return None


def main() -> None:
    ranking = list(csv.DictReader(RANKING_CSV.open(encoding="utf-8")))
    directory = list(csv.DictReader(DIRECTORY_CSV.open(encoding="utf-8")))

    wiki_limiter = RateLimiter(REQUEST_DELAY_SECONDS)
    wiki_cache: dict[str, tuple[str, list[str]]] = {}
    verify_cache: dict[str, dict[str, str] | None] = {}
    verification_count = 0

    def wiki_subject(title: str) -> tuple[str, list[str]]:
        if title not in wiki_cache:
            wiki_cache[title] = fetch_wiki_subject(title, wiki_limiter)
        return wiki_cache[title]

    def verify_brand(brand_name: str) -> dict[str, str] | None:
        nonlocal verification_count
        key = verify.normalize_name(brand_name)
        if key not in verify_cache:
            match = verify.match_brand(directory, brand_name)
            if not match:
                verify_cache[key] = None
            else:
                try:
                    verify_cache[key] = verify.verify_brand(brand_name, match)
                except Exception as exc:  # noqa: BLE001 - report as data
                    verify_cache[key] = {
                        "requested_name": brand_name,
                        "brand_name": brand_name,
                        "brand_url": match["brand_url"],
                        "match_status": "Matched",
                        "category": match["category"],
                        "category_verified": "No",
                        "brand_page_status": "",
                        "products_listed": "No",
                        "product_count": "0",
                        "product_availability_status": "Unable to verify",
                        "availability_evidence": f"Verification failed: {exc}",
                        "cross_check_brand_name": "",
                        "cross_check_brand_url": "",
                        "cross_check_total_products": "",
                        "cross_check_matching_products": "",
                        "cross_check_example_product_url": "",
                        "cross_check_availability_status": "",
                        "cross_check_notes": "",
                        "notes": f"Verification failed: {exc}",
                    }
                verification_count += 1
                time.sleep(VERIFY_DELAY_SECONDS)
        return verify_cache[key]

    def verify_info(row: dict[str, str]) -> dict[str, str] | None:
        if row["wiki_title"] == "not_found" and verify.normalize_name(
            row["brand_name"]
        ) not in REQUIRED:
            return None
        return verify_brand(row["brand_name"])

    def on_sale_for(row: dict[str, str], info: dict[str, str] | None) -> str:
        if info is None:
            return ""
        if info.get("notes", "").startswith("Verification failed"):
            return "Unknown"
        try:
            return "Yes" if int(info.get("product_count", "0")) > 0 else "No"
        except ValueError:
            return "Unknown"

    selected: list[dict[str, str]] = []
    selected_category: dict[str, str] = {}
    category_counts: dict[str, int] = {}
    selected_keys: set[str] = set()
    skipped: list[tuple[dict[str, str], str]] = []
    skip_reasons: dict[str, str] = {}

    # Required brands are selected before the ranking walk. They still count
    # against their category limits.
    for required_name, category in REQUIRED.items():
        row = next(
            (
                candidate
                for candidate in ranking
                if verify.normalize_name(candidate["brand_name"]) == required_name
            ),
            None,
        )
        if row is None:
            raise SystemExit(f"Required brand {required_name!r} is not in the ranking")
        info = verify_brand(row["brand_name"])
        on_sale = on_sale_for(row, info)
        if on_sale != "Yes":
            raise SystemExit(
                f"Required brand {row['brand_name']} has no products listed "
                f"(on_sale={on_sale!r}); cannot satisfy required selection."
            )
        selected.append({"row": row, "info": info, "category": category})
        selected_keys.add(verify.normalize_name(row["brand_name"]))
        selected_category[verify.normalize_name(row["brand_name"])] = category
        category_counts[category] = category_counts.get(category, 0) + 1

    # Walk the ranking from the top.
    for row in ranking:
        if len(selected) >= MAX_SELECTED:
            break

        key = verify.normalize_name(row["brand_name"])
        if key in selected_keys:
            skip_reasons[key] = ""  # required brand, but not a skip
            continue

        if row["wiki_title"] == "not_found":
            skip_reasons[key] = "Wikipedia title not_found"
            skipped.append((row, skip_reasons[key]))
            continue

        description, categories = wiki_subject(row["wiki_title"])
        mismatch = mismatch_reason(description, categories)
        info = verify_info(row)
        on_sale = on_sale_for(row, info)

        if mismatch:
            skip_reasons[key] = mismatch
            skipped.append((row, skip_reasons[key]))
            continue

        if on_sale != "Yes":
            reason = (
                "Lane Crawford verification failed"
                if on_sale == "Unknown"
                else f"no products listed (product_count={info.get('product_count', '0') if info else '0'})"
            )
            skip_reasons[key] = reason
            skipped.append((row, reason))
            continue

        product_text = info.get("availability_evidence", "") if info else ""
        category = classify_category(
            row["brand_name"], description, categories, product_text
        )
        if category is None:
            reason = "no matching category"
            skip_reasons[key] = reason
            skipped.append((row, reason))
            continue

        if category_counts.get(category, 0) >= MAX_PER_CATEGORY:
            reason = f"category '{category}' already has {MAX_PER_CATEGORY} selected brands"
            skip_reasons[key] = reason
            skipped.append((row, reason))
            continue

        selected.append({"row": row, "info": info, "category": category})
        selected_keys.add(key)
        selected_category[key] = category
        category_counts[category] = category_counts.get(category, 0) + 1

    # Annotate every row. Verify the remaining Wikipedia matches so that
    # on_sale and mismatch reasons are complete, but do not add selections.
    for row in ranking:
        key = verify.normalize_name(row["brand_name"])
        if key in selected_keys:
            continue
        if key in skip_reasons:
            continue
        if row["wiki_title"] == "not_found":
            skip_reasons[key] = "Wikipedia title not_found"
            continue

        description, categories = wiki_subject(row["wiki_title"])
        mismatch = mismatch_reason(description, categories)
        info = verify_info(row)
        on_sale = on_sale_for(row, info)

        if mismatch:
            skip_reasons[key] = mismatch
        elif on_sale != "Yes":
            skip_reasons[key] = (
                "Lane Crawford verification failed"
                if on_sale == "Unknown"
                else f"no products listed (product_count={info.get('product_count', '0') if info else '0'})"
            )
        else:
            skip_reasons[key] = "not reached: 10-brand limit reached"

    # Write selected_brands.csv using the existing verification columns.
    SELECTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SELECTED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_FIELDS)
        writer.writeheader()
        for item in selected:
            info = dict(item["info"] or {})
            output = {field: info.get(field, "") for field in SELECTED_FIELDS}
            output["assigned_category"] = item["category"]
            writer.writerow(output)

    # Write brand_ranking.csv with the four extra columns.
    with RANKING_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=BASE_RANKING_FIELDS + EXTRA_RANKING_FIELDS
        )
        writer.writeheader()
        for row in ranking:
            key = verify.normalize_name(row["brand_name"])
            info = (
                verify_cache.get(key)
                if row["wiki_title"] != "not_found" or key in REQUIRED
                else None
            )
            is_selected = key in selected_keys
            category = selected_category.get(key, "") if is_selected else ""
            skip_reason = "" if is_selected else skip_reasons.get(key, "")
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in BASE_RANKING_FIELDS},
                    "selected": "Yes" if is_selected else "No",
                    "category": category,
                    "on_sale": on_sale_for(row, info),
                    "skip_reason": skip_reason,
                }
            )

    print(f"Verification requests: {verification_count}")
    print(f"Selected {len(selected)} brands:")
    for index, item in enumerate(selected, start=1):
        row = item["row"]
        info = item["info"] or {}
        print(
            f"  {index:>2}. {row['brand_name']:<24} "
            f"wiki={row['wiki_title']:<28} "
            f"category={item['category']:<22} "
            f"products={info.get('product_count', '0')}"
        )

    print("\nSkipped during the ranking walk:")
    for row, reason in skipped:
        print(f"  {row['brand_name']:<24} -> {reason}")


if __name__ == "__main__":
    main()
