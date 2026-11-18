#!/usr/bin/env python3
"""Build the Step 2 account list for one brand (prompts/01_find_accounts.md).

Inputs (all in assignment-4/data/accounts/)
    <slug>_candidates.csv   accounts found by the agent
    <slug>_review.xlsx      your review (edited in Excel; created by this script)

Fill only `decision` for every account and `creator_basis` for creators
(`reject_reason` is optional). Older review files remain compatible. The script
stamps metadata_source, reviewed_by ("solo" by default) and reviewed_at
(the date a decision is first seen). Valid brands come from
assignment-4/data/selected_brands.csv.

Outputs (all in assignment-4/data/accounts/)
    <slug>_review.xlsx      rewritten: candidate columns refreshed, member columns kept
    <slug>_review.csv       plain copy of the review, for readable diffs (never edit)
    <slug>.csv              merged list with computed status
    <slug>_summary.json     counts and shortfall flags

Usage
    python src/accounts_collection/build_accounts.py --brand stanley
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

REPO_ROOT = Path(__file__).resolve().parents[2]  # Assignment 4/ (script is in src/accounts_collection/)
ACCOUNTS_DIR = REPO_ROOT / "assignment-4" / "data" / "accounts"
SELECTED_BRANDS_CSV = REPO_ROOT / "assignment-4" / "data" / "selected_brands.csv"
ACCOUNT_TYPES = ["official", "retailer", "media", "creator"]
NON_OFFICIAL_TYPES = ["retailer", "media", "creator"]
CREATOR_MIN_FOLLOWERS = 10_000
MIN_APPROVED_ACCOUNTS = 4        # 40 images at most 10 per account
MIN_APPROVED_NON_OFFICIAL = 2    # official supplies at most 20 of the 40

CANDIDATE_COLUMNS = ["handle", "account_type", "profile_url", "evidence_url", "found_via"]
# Review sheet: script-owned columns first, then the member's columns.
SCRIPT_COLUMNS = ["handle", "account_type", "profile_url", "evidence_url", "found_via"]
# Older workbooks did not have found_via; keep them readable.
REQUIRED_REVIEW_COLUMNS = ["handle", "account_type", "profile_url", "evidence_url"]
MEMBER_COLUMNS = ["decision", "reject_reason", "creator_basis"]        # chosen by the reviewer
LEGACY_COLUMN = "follower_count"     # older review files only; kept, never shown for new rows
STAMP_COLUMNS = ["metadata_source", "reviewed_by", "reviewed_at"]       # filled by this script
STORED_COLUMNS = MEMBER_COLUMNS + [LEGACY_COLUMN] + STAMP_COLUMNS
OUTPUT_COLUMNS = ["brand_slug", "handle", "account_type", "profile_url", "evidence_url",
                  "found_via", "creator_basis", "follower_count", "metadata_source", "reviewed_by",
                  "reviewed_at", "status", "status_reason"]
HKT = timezone(timedelta(hours=8))
REJECT_REASONS = ["personal", "private", "fan", "reseller", "below_threshold",
                  "no_evidence", "off_topic", "duplicate", "other"]
DROPDOWNS = {
    "decision": ["approve", "reject"],
    "reject_reason": REJECT_REASONS,
    "creator_basis": ["verified", "creator_label", "10k_plus"],
}
SHEET = "review"
VALIDATED_ROWS = 300          # dropdowns cover this many rows
HANDLE_RE = re.compile(r"^[a-z0-9._]{1,30}$")
FOLLOWER_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kmb万萬]?)$", re.IGNORECASE)
FOLLOWER_MULTIPLIERS = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000,
                        "万": 10_000, "萬": 10_000}


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


# --- parsing -------------------------------------------------------------------

def parse_followers(value: str) -> int | None:
    """Convert a count as Instagram displays it ("12,345", "45.6K", "1.2M",
    "4.5萬") to an integer. Abbreviated counts are as precise as the display."""
    match = FOLLOWER_RE.match(value.replace(",", "").strip())
    if not match:
        return None
    number, suffix = match.groups()
    if "." in number and not suffix:
        return None
    return int(Decimal(number) * FOLLOWER_MULTIPLIERS[suffix.lower()])


def cell_text(value) -> str:
    """Excel cell value as text, whatever type Excel stored."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


# --- files ---------------------------------------------------------------------

def load_brand_slugs() -> set[str]:
    """Read the chosen brand slugs from the existing selection file."""
    if not SELECTED_BRANDS_CSV.is_file():
        fail(f"Not found: {SELECTED_BRANDS_CSV}")
    with SELECTED_BRANDS_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        slug_col = next((col for col in ("slug", "brand_slug", "lc_slug")
                         if col in headers), None)
        if slug_col is None and "brand_url" not in headers:
            fail(f"{SELECTED_BRANDS_CSV.name} needs a slug column "
                 "(slug, brand_slug, or lc_slug) or a brand_url column")
        if slug_col is not None:
            slugs = {row[slug_col].strip() for row in reader if row.get(slug_col, "").strip()}
        else:
            slugs = set()
            for row in reader:
                path = urlsplit(row.get("brand_url") or "").path
                match = re.fullmatch(r"/brand/[0-9]+/([a-z0-9-]+)/[^/]+/?", path)
                if not match:
                    fail(f"{SELECTED_BRANDS_CSV.name} line {reader.line_num}: invalid brand_url")
                slugs.add(match[1])
    if not slugs:
        fail(f"No brand slugs found in {SELECTED_BRANDS_CSV}")
    return slugs


def load_candidates(path: Path) -> tuple[list[dict[str, str]], int]:
    if not path.is_file():
        fail(f"Not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in CANDIDATE_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            fail(f"{path.name} is missing columns: {', '.join(missing)}")
        raw = [{k: (v or "").strip() for k, v in r.items() if k} for r in reader]
    rows, seen, dropped = [], set(), 0
    for line_no, row in enumerate(raw, start=2):
        handle = row["handle"].lstrip("@").lower()
        if not HANDLE_RE.match(handle):
            fail(f"{path.name} line {line_no}: invalid handle {row['handle']!r}")
        if row["account_type"] not in ACCOUNT_TYPES:
            fail(f"{path.name} line {line_no}: account_type must be one of {ACCOUNT_TYPES}")
        if handle in seen:
            dropped += 1
            continue
        seen.add(handle)
        rows.append({**row, "handle": handle})
    return rows, dropped


def read_review(path: Path) -> dict[str, dict[str, str]]:
    """Member columns from the existing workbook, keyed by handle."""
    if not path.is_file():
        return {}
    sheet = load_workbook(path, data_only=True)[SHEET]
    header = [cell_text(c.value) for c in sheet[1]]
    missing = [c for c in REQUIRED_REVIEW_COLUMNS + ["decision", "reject_reason"]
               if c not in header]
    if missing:
        fail(f"{path.name} is missing columns: {', '.join(missing)}")
    reviews: dict[str, dict[str, str]] = {}
    for values in sheet.iter_rows(min_row=2, values_only=True):
        row = {name: cell_text(v) for name, v in zip(header, values)}
        handle = row.get("handle", "").lstrip("@").lower()
        if not handle:
            continue
        if handle in reviews:
            fail(f"{path.name} has more than one row for {handle}")
        reviews[handle] = {c: row.get(c, "") for c in STORED_COLUMNS}
    return reviews


def write_review(xlsx_path: Path, csv_path: Path, rows: list[dict[str, str]]) -> None:
    legacy = [LEGACY_COLUMN] if any(r.get(LEGACY_COLUMN) for r in rows) else []
    columns = SCRIPT_COLUMNS + MEMBER_COLUMNS + legacy + STAMP_COLUMNS
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET
    ws.append(columns)
    for row in rows:
        ws.append([row.get(c, "") for c in columns])

    member_fill = PatternFill("solid", fgColor="FFF2CC")
    for idx, name in enumerate(columns, start=1):
        letter = ws.cell(row=1, column=idx).column_letter
        is_url = name.endswith("_url") or name == "found_via"
        ws.cell(row=1, column=idx).font = Font(bold=True)
        if name in MEMBER_COLUMNS:
            ws.cell(row=1, column=idx).fill = member_fill
        ws.column_dimensions[letter].width = 34 if is_url else max(14, len(name) + 2)
        for r in range(2, VALIDATED_ROWS + 2):
            cell = ws.cell(row=r, column=idx)
            cell.number_format = "@"            # text, so Excel keeps dates and counts as typed
            if is_url and cell.value:
                cell.hyperlink = cell.value
                cell.style = "Hyperlink"
                cell.number_format = "@"
                if name == "found_via":
                    cell.alignment = Alignment(wrap_text=True)
        if name in DROPDOWNS:
            dv = DataValidation(type="list", formula1='"' + ",".join(DROPDOWNS[name]) + '"',
                                allow_blank=True)
            dv.add(f"{letter}2:{letter}{VALIDATED_ROWS + 1}")
            ws.add_data_validation(dv)
    ws.freeze_panes = "B2"
    try:
        wb.save(xlsx_path)
    except PermissionError:
        fail(f"Cannot write {xlsx_path.name}: close it in Excel and re-run.")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --- status --------------------------------------------------------------------

def validate_review(cand: dict[str, str], rev: dict[str, str]) -> str | None:
    """Return the first invalid member field, or None."""
    if rev["decision"] not in DROPDOWNS["decision"]:
        return "decision"
    if rev["reject_reason"] and rev["reject_reason"] not in REJECT_REASONS:
        return "reject_reason"
    if rev["creator_basis"] and rev["creator_basis"] not in DROPDOWNS["creator_basis"]:
        return "creator_basis"
    if rev["decision"] == "approve" and cand["account_type"] == "creator" and not rev["creator_basis"]:
        # Legacy rows: a follower count stands in for creator_basis.
        if not rev[LEGACY_COLUMN]:
            return "creator_basis"
        if parse_followers(rev[LEGACY_COLUMN]) is None:
            return "follower_count"
    return None


def stamp(rev: dict[str, str], member: str) -> dict[str, str]:
    """Fill the script columns the first time a decision is seen; clear them if
    the decision is removed."""
    if not rev["decision"]:
        return {**rev, "metadata_source": "", "reviewed_by": "", "reviewed_at": ""}
    if rev["reviewed_at"]:
        return rev
    return {**rev, "metadata_source": "manual", "reviewed_by": member,
            "reviewed_at": datetime.now(HKT).date().isoformat()}


def decide(cand: dict[str, str], rev: dict[str, str]) -> tuple[str, str]:
    if not rev["decision"]:
        return "pending", "awaiting_review"
    invalid = validate_review(cand, rev)
    if invalid:
        return "pending", f"invalid_review:{invalid}"
    if rev["decision"] == "reject":
        return "rejected", f"member_reject:{rev['reject_reason'] or 'unspecified'}"
    if not cand["evidence_url"]:
        return "rejected", "no_evidence_url"
    if (cand["account_type"] == "creator" and not rev["creator_basis"]
            and parse_followers(rev[LEGACY_COLUMN]) < CREATOR_MIN_FOLLOWERS):
        return "rejected", "creator_below_threshold"
    return "approved", "all_checks_passed"


# --- main ----------------------------------------------------------------------

def build(slug: str, member: str) -> None:
    base = ACCOUNTS_DIR / slug
    candidates, dropped = load_candidates(base.with_name(f"{slug}_candidates.csv"))
    xlsx_path = base.with_name(f"{slug}_review.xlsx")
    reviews = {h: stamp(r, member) for h, r in read_review(xlsx_path).items()}

    empty = {c: "" for c in STORED_COLUMNS}
    review_rows = [{**{c: cand[c] for c in SCRIPT_COLUMNS}, **reviews.get(cand["handle"], empty)}
                   for cand in candidates]
    current = {c["handle"] for c in candidates}
    orphans = sorted(h for h in reviews if h not in current)
    # Keep reviews whose candidate was removed, at the end, so nothing typed is lost.
    review_rows += [{"handle": h, **reviews[h]} for h in orphans]
    write_review(xlsx_path, base.with_name(f"{slug}_review.csv"), review_rows)

    output = []
    for cand in candidates:
        rev = reviews.get(cand["handle"], empty)
        status, reason = decide(cand, rev)
        followers = parse_followers(rev["follower_count"]) if rev["follower_count"] else None
        output.append({
            "brand_slug": slug, "handle": cand["handle"], "account_type": cand["account_type"],
            "profile_url": cand["profile_url"], "evidence_url": cand["evidence_url"],
            "found_via": cand["found_via"], "creator_basis": rev["creator_basis"],
            "follower_count": str(followers) if followers is not None else rev["follower_count"],
            "metadata_source": rev["metadata_source"], "reviewed_by": rev["reviewed_by"],
            "reviewed_at": rev["reviewed_at"], "status": status, "status_reason": reason,
        })
    with base.with_name(f"{slug}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output)

    counts = {t: {s: 0 for s in ("approved", "rejected", "pending")} for t in ACCOUNT_TYPES}
    for r in output:
        counts[r["account_type"]][r["status"]] += 1
    approved_total = sum(c["approved"] for c in counts.values())
    approved_non_official = sum(counts[t]["approved"] for t in NON_OFFICIAL_TYPES)
    summary = {
        "brand_slug": slug,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidates": len(output),
        "counts_by_type_and_status": counts,
        "approved_total": approved_total,
        "approved_non_official": approved_non_official,
        "pending_total": sum(c["pending"] for c in counts.values()),
        "status_reasons": dict(sorted(Counter(r["status_reason"] for r in output).items())),
        "shortfall_accounts": approved_total < MIN_APPROVED_ACCOUNTS,
        "shortfall_non_official": approved_non_official < MIN_APPROVED_NON_OFFICIAL,
        "types_with_no_approved": [t for t in ACCOUNT_TYPES if counts[t]["approved"] == 0],
        "candidate_duplicates_dropped": dropped,
        "review_rows_without_candidate": orphans,
    }
    base.with_name(f"{slug}_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                      encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--brand", required=True, help="slug in selected_brands.csv")
    parser.add_argument("--member", default="solo",
                        help="optional reviewer label (default: solo; legacy flag supported)")
    args = parser.parse_args()
    slugs = load_brand_slugs()
    if args.brand not in slugs:
        fail(f"Unknown brand {args.brand!r}; choose one of: {', '.join(sorted(slugs))}")
    build(args.brand, args.member)


if __name__ == "__main__":
    main()
