"""Shared settings for Step 3 of the solo STAT6207 Assignment 4 project."""

from __future__ import annotations

import csv
import os
import re
import sys
from datetime import timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[2]  # Assignment 4/ (script in src/<collection_folder>/)
ACCOUNTS_DIR = REPO_ROOT / "assignment-4" / "data" / "accounts"
MANIFEST_DIR = REPO_ROOT / "assignment-4" / "data" / "manifest"

SELECTED_BRANDS_CSV = REPO_ROOT / "assignment-4" / "data" / "selected_brands.csv"


def load_brand_slugs() -> list[str]:
    """Use the same selection file and slug columns as build_accounts.py."""
    if not SELECTED_BRANDS_CSV.is_file():
        fail(f"Not found: {SELECTED_BRANDS_CSV}")
    with SELECTED_BRANDS_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        slug_col = next((col for col in ("slug", "brand_slug", "lc_slug") if col in headers), None)
        if slug_col is not None:
            slugs = {row[slug_col].strip() for row in reader if row.get(slug_col, "").strip()}
        elif "brand_url" in headers:
            slugs = set()
            for row in reader:
                path = urlsplit(row.get("brand_url") or "").path
                match = re.fullmatch(r"/brand/[0-9]+/([a-z0-9-]+)/[^/]+/?", path)
                if not match:
                    fail(f"{SELECTED_BRANDS_CSV.name} line {reader.line_num}: invalid brand_url")
                slugs.add(match[1])
        else:
            fail(f"{SELECTED_BRANDS_CSV.name} needs a slug column or a brand_url column")
    if not slugs:
        fail(f"No brand slugs found in {SELECTED_BRANDS_CSV}")
    return sorted(slugs)


HKT = timezone(timedelta(hours=8))
IMAGES_PER_BRAND = 40
MAX_OFFICIAL = 30
MAX_PER_ACCOUNT = 10             # non-official accounts
MAX_PER_OFFICIAL_ACCOUNT = 20    # higher allowance for an official account
REQUEST_DELAY_SECONDS = 6          # between Instagram requests, on top of instaloader's limiter

# Brand mention for posts from non-official accounts. Always counted: an
# @mention or photo tag of an approved official account, or a hashtag equal to
# an approved official handle. In addition:
#   BRAND_HASHTAGS: extra hashtags (without #, lowercase);
#   BRAND_NAMES: phrases matched in the caption as whole words (lowercase).
# For ambiguous names (e.g. Stanley, Oura), prefer specific product/brand
# phrases over bare words to avoid unrelated matches.
BRAND_HASHTAGS = {
    "jellycat": ["jellycat"],
    "stanley": ["stanley1913", "stanleyquencher", "stanleytumbler"],
    "lego": ["lego"],
    "kodak": ["kodakfilm", "kodakcamera", "kodakportra", "kodakgold", "kodakmoment"],
    "bang-olufsen": ["bangolufsen", "bangandolufsen", "beoplay", "beosound"],
    "sennheiser": ["sennheiser", "sennheisermomentum"],
    "fortnum-mason": ["fortnums", "fortnumandmason", "fortnummason"],
    "jo-malone-london": ["jomalone", "jomalonelondon"],
    "herman-miller": ["hermanmiller", "hermanmillerchair", "aeronchair"],
    "oura": ["ouraring", "ourahealth", "ouraring4"],
}
BRAND_NAMES = {
    "jellycat": ["jellycat"],
    "stanley": ["stanley tumbler", "stanley quencher", "stanley flask", "stanley 1913"],
    "lego": ["lego"],
    "kodak": ["kodak film", "kodak camera", "kodak portra", "kodak gold", "kodak"],
    "bang-olufsen": ["bang & olufsen", "bang and olufsen", "bang&olufsen"],
    "sennheiser": ["sennheiser"],
    "fortnum-mason": ["fortnum & mason", "fortnum and mason", "fortnum's", "fortnums"],
    "jo-malone-london": ["jo malone", "jo malone london"],
    "herman-miller": ["herman miller", "herman-miller", "aeron chair"],
    "oura": ["oura ring", "ouraring", "oura health"],
}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def images_root() -> Path:
    """LC_IMAGES_ROOT from the environment, else from the repo-root .env."""
    value = os.environ.get("LC_IMAGES_ROOT") or read_dotenv(REPO_ROOT / ".env").get("LC_IMAGES_ROOT")
    if not value:
        fail("LC_IMAGES_ROOT is not set (environment or repo-root .env).")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        fail(f"LC_IMAGES_ROOT does not exist: {root}")
    if root == REPO_ROOT or REPO_ROOT in root.parents:
        fail("LC_IMAGES_ROOT must be outside the repository.")
    return root


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items() if k is not None}
                for row in csv.DictReader(handle)]


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def approved_accounts(slug: str) -> dict[str, str]:
    """Approved handles for the brand from Step 2 -> account_type."""
    path = ACCOUNTS_DIR / f"{slug}.csv"
    if not path.is_file():
        fail(f"Not found: {path} (finish prompt 01 first)")
    return {r["handle"].lower(): r["account_type"] for r in read_csv(path) if r["status"] == "approved"}
