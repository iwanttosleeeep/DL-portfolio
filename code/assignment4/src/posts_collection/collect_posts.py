#!/usr/bin/env python3
"""Step 3: verify Instagram photo posts and write the image manifest.

Candidate URLs live in assignment-4/data/manifest/<slug>_post_urls.xlsx,
created using --init. The workbook keeps three columns:
    post_url, verified, url_source

Verification workflow:
- For a post owned by an APPROVED official account, the script fetches
  metadata, checks the owner and photo-only media type, and writes
  verified=auto in the workbook. No human review is required.
- For non-official accounts, the human MUST mark verified=yes or no after
  checking post ownership and brand relevance. Blank means pending review.
- Human verified=no always excludes a post. Invalid/unavailable posts,
  videos, Reels, mixed/video carousels and unapproved owners never download.
- An explicit human verified=yes is sufficient brand-relevance evidence for
  non-official posts; caption/hashtag matches are recorded, not required.

The script fetches metadata for candidate URLs, marks verified official posts,
then downloads eligible photo posts with per-account limits and SHA-256
image de-duplication. Only successfully saved, unique images count toward the
required 40 per brand. Images/ is not cleaned destructively on re-runs;
the manifest identifies currently selected images.

Files written to assignment-4/data/manifest/:
    <slug>_post_urls.xlsx   review workbook (auto marked for official photos)
    <slug>_post_urls.csv    review states, including verified=auto
    <slug>_fetch_log.csv    cached metadata (captions never saved)
    <slug>.csv              image-selection manifest
    <slug>_summary.json     counts and shortfalls

Usage (from Assignment 4/):
    python3 src/posts_collection/collect_posts.py --brand stanley --init
    python3 src/posts_collection/collect_posts.py --brand stanley
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image, UnidentifiedImageError

from _common import (BRAND_HASHTAGS, BRAND_NAMES, HKT, IMAGES_PER_BRAND,
                     MANIFEST_DIR, MAX_OFFICIAL, MAX_PER_ACCOUNT, MAX_PER_OFFICIAL_ACCOUNT, REQUEST_DELAY_SECONDS,
                     approved_accounts, fail, images_root, load_brand_slugs,
                     read_csv, write_csv)

if TYPE_CHECKING:
    import instaloader

URL_SHEET = "urls"
URL_COLUMNS = ["post_url", "verified", "url_source"]
URL_SOURCES = ["agent_search", "member_added"]
SHORTCODE_RE = re.compile(r"instagram\.com/p/([A-Za-z0-9_-]+)(?:[/?#]|$)", re.IGNORECASE)
LOG_COLUMNS = ["shortcode", "post_url", "fetched_at", "fetch_status", "fetch_error",
               "owner", "post_datetime_utc", "post_date", "post_type", "brand_match",
               "image_url", "image_file", "image_sha256", "collection_method",
               "download_error"]
MANIFEST_COLUMNS = ["brand_slug", "post_url", "url_source", "shortcode", "handle", "account_type",
                    "post_datetime_utc", "post_date", "post_type", "brand_match",
                    "selected", "selection_reason", "image_id", "image_path",
                    "collection_method", "file_format", "width", "height"]
FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
class Blocked(Exception):
    pass


# --- URL workbook ----------------------------------------------------------------

def init_workbook(path: Path) -> None:
    if path.exists():
        fail(f"{path.name} already exists.")
    wb = Workbook()
    ws = wb.active
    ws.title = URL_SHEET
    ws.append(URL_COLUMNS)
    ws.column_dimensions["A"].width = 65
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 16
    review_options = DataValidation(type="list", formula1='"yes,no,auto"', allow_blank=True)
    review_options.error = "Choose yes or no for non-official posts; auto is set by the script."
    review_options.showErrorMessage = True
    ws.add_data_validation(review_options)
    review_options.add("B2:B400")
    source_options = DataValidation(type="list", formula1='"' + ",".join(URL_SOURCES) + '"',
                                    allow_blank=True)
    ws.add_data_validation(source_options)
    source_options.add("C2:C400")
    # Do not pre-create 400 empty rows: that makes openpyxl.append() start
    # at row 401, outside the dropdown validation ranges.
    for cell in ws[1]:
        cell.number_format = "@"
    wb.save(path)
    print(f"Created {path}. Review only non-official photo posts (yes/no); official photos are marked auto after metadata checks.")


def read_urls(xlsx: Path, csv_copy: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Read all candidates, including pending and rejected URLs.

    Do not require human review before fetching metadata: ownership can only
    be confirmed from that metadata, and determines whether review is needed.
    """
    if not xlsx.is_file():
        fail(f"Not found: {xlsx} (run with --init first)")
    wb = load_workbook(xlsx)
    ws = wb[URL_SHEET]
    headers = [ws.cell(row=1, column=col).value for col in (1, 2, 3)]
    if headers[:2] != ["post_url", "verified"] or headers[2] not in (None, "url_source"):
        fail(f"{xlsx.name} needs columns post_url, verified and url_source (A–C).")

    reviewed, filled = [], 0
    for row_number in range(2, ws.max_row + 1):
        url_cell, verified_cell, source_cell = (ws.cell(row=row_number, column=c) for c in (1, 2, 3))
        if url_cell.value is None or not str(url_cell.value).strip():
            continue
        url = str(url_cell.value).strip()
        verified = str(verified_cell.value).strip().lower() if verified_cell.value is not None else ""
        if verified not in {"yes", "no", "auto", ""}:
            fail(f"{xlsx.name} row {row_number}: verified must be yes, no, auto or blank.")
        source = str(source_cell.value).strip() if source_cell.value is not None else ""
        if not source:
            source = "member_added"
            source_cell.value, source_cell.number_format = source, "@"
            filled += 1
        elif source not in URL_SOURCES:
            fail(f"{xlsx.name} row {row_number}: url_source must be one of {URL_SOURCES}.")
        match = SHORTCODE_RE.search(url)
        reviewed.append({"post_url": url, "verified": verified, "url_source": source,
                         "shortcode": match.group(1) if match else "", "row_number": row_number})
    if filled:
        ws.cell(row=1, column=3).value = "url_source"
        try:
            wb.save(xlsx)
        except PermissionError:
            fail(f"Cannot write {xlsx.name}: close it in Excel and re-run.")

    if not reviewed:
        fail(f"No post URLs in {xlsx.name}. Add genuine post URLs first.")

    out, seen = [], {}
    for row in reviewed:
        code = row["shortcode"]
        if code:
            if code in seen:
                previous = seen[code]
                if previous["verified"] != row["verified"] and (previous["verified"] or row["verified"]):
                    fail(f"Conflicting review values for duplicate post {code} at rows "
                         f"{previous['row_number']} and {row['row_number']}; resolve in Excel.")
                continue
            seen[code] = row
        out.append({"pasted_url": row["post_url"], "shortcode": code,
                    "url_source": row["url_source"], "verified": row["verified"],
                    "row_number": row["row_number"]})
    return out, reviewed


def sync_official_verification(xlsx: Path, urls: list[dict[str, str]],
                               reviewed: list[dict[str, str]], accounts: dict[str, str],
                               log: dict[str, dict[str, str]]) -> int:
    """Mark metadata-confirmed official photo posts `auto`, preserving human yes/no.

    Remove stale auto status if the owner is no longer an approved official
    account, or the cached metadata no longer qualifies. Do not alter any
    manually entered yes/no values.
    """
    wb = load_workbook(xlsx)
    ws = wb[URL_SHEET]
    changes = 0
    for row in reviewed:
        if row["verified"] in {"yes", "no"}:
            continue
        rec = log.get(row["shortcode"], {})
        auto_ok = (rec.get("fetch_status") == "ok"
                   and accounts.get(rec.get("owner", "").lower()) == "official"
                   and rec.get("post_type") in {"image", "carousel_image"}
                   and bool(rec.get("image_url")))
        new = "auto" if auto_ok else ""
        if new != row["verified"]:
            ws.cell(row=row["row_number"], column=2, value=new)
            row["verified"] = new
            changes += 1

    # Update existing workbooks' dropdown as well as new workbooks.
    validation_changed = False
    for validation in ws.data_validations.dataValidation:
        if 'B2' in str(validation.sqref) and validation.type == "list":
            if validation.formula1 != '"yes,no,auto"':
                validation.formula1 = '"yes,no,auto"'
                validation_changed = True
    if changes or validation_changed:
        try:
            wb.save(xlsx)
        except PermissionError:
            fail(f"Cannot update official verification in {xlsx.name}: close Excel and rerun.")
    state_by_row = {row["row_number"]: row["verified"] for row in reviewed}
    for row in urls:
        row["verified"] = state_by_row[row["row_number"]]
    return changes


# --- Instagram ---------------------------------------------------------------------

def make_loader() -> instaloader.Instaloader:
    import instaloader

    return instaloader.Instaloader(
        sleep=True, quiet=True, download_videos=False, download_video_thumbnails=False,
        download_geotags=False, download_comments=False, save_metadata=False,
        compress_json=False, post_metadata_txt_pattern="", max_connection_attempts=1,
        request_timeout=30, resume_prefix=None, iphone_support=False,
        fatal_status_codes=[401, 429],
    )


def first_image(post: instaloader.Post) -> tuple[str, str]:
    """Accept photo posts only; never substitute a video's cover image."""
    if post.typename == "GraphImage" and not post.is_video:
        return "image", post.url
    if post.typename == "GraphSidecar":
        nodes = list(post.get_sidecar_nodes())
        if nodes and all(not node.is_video for node in nodes):
            return "carousel_image", nodes[0].display_url
        return "non_image", ""  # Includes mixed photo/video carousels.
    return "non_image", ""  # Reels, GraphVideo and unknown types.



def brand_match(post: instaloader.Post, slug: str, account_type: str, officials: set[str]) -> str:
    if account_type == "official":
        return "official_account"
    tagged = {h.lower() for h in post.tagged_users} | {h.lower() for h in post.caption_mentions}
    if tagged & officials:
        return "tag_or_mention"
    hashtags = {h.lower() for h in post.caption_hashtags}
    if hashtags & (officials | set(BRAND_HASHTAGS[slug])):
        return "hashtag"
    caption = (post.caption or "").lower()
    for name in BRAND_NAMES[slug]:
        if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", caption):
            return "caption_name"
    return ""


def fetch(loader, shortcode: str, slug: str, accounts: dict[str, str]) -> dict:
    """One request. Raises Blocked if Instagram refuses."""
    from instaloader import Post
    from instaloader.exceptions import (AbortDownloadException, ConnectionException,
                                        InstaloaderException, LoginRequiredException,
                                        QueryReturnedNotFoundException, TooManyRequestsException)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        post = Post.from_shortcode(loader.context, shortcode)
        owner = post.owner_username.lower()
        post_type, image_url = first_image(post)
        officials = {h for h, t in accounts.items() if t == "official"}
        match = brand_match(post, slug, accounts[owner], officials) if owner in accounts else ""
        utc = post.date_utc.replace(tzinfo=timezone.utc)
        return {"fetched_at": now, "fetch_status": "ok", "owner": owner,
                "post_datetime_utc": utc.isoformat(), "post_date": utc.astimezone(HKT).date().isoformat(),
                "post_type": post_type, "brand_match": match, "image_url": image_url}
    except QueryReturnedNotFoundException as exc:
        return {"fetched_at": now, "fetch_status": "unavailable",
                "fetch_error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    except (LoginRequiredException, TooManyRequestsException, AbortDownloadException,
            ConnectionException) as exc:
        raise Blocked(f"{type(exc).__name__}: {exc}") from exc
    except InstaloaderException as exc:
        return {"fetched_at": now, "fetch_status": "unavailable",
                "fetch_error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def download(loader, url: str, dl_dir: Path, shortcode: str) -> Path:
    from instaloader.exceptions import ConnectionException

    dl_dir.mkdir(parents=True, exist_ok=True)
    for old in dl_dir.glob(shortcode + ".*"):
        old.unlink()
    loader.download_pic(str(dl_dir / shortcode), url, datetime.now())
    files = sorted(dl_dir.glob(shortcode + ".*"))
    if not files:
        raise ConnectionException("download produced no file")
    return files[0]


# --- images -----------------------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_info(path: Path) -> tuple[str, int, int] | None:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.format, img.size[0], img.size[1]
    except (UnidentifiedImageError, OSError, SyntaxError):
        return None


# --- main ---------------------------------------------------------------------------

def run(slug: str) -> None:
    from instaloader.exceptions import (AbortDownloadException, InstaloaderException,
                                        LoginRequiredException, TooManyRequestsException)

    root = images_root()
    brand_dir = root / slug
    images_dir, shots_dir, dl_dir = brand_dir / "images", brand_dir / "screenshots", brand_dir / "_downloads"
    for d in (images_dir, shots_dir):
        d.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    urls, reviewed = read_urls(MANIFEST_DIR / f"{slug}_post_urls.xlsx",
                               MANIFEST_DIR / f"{slug}_post_urls.csv")
    accounts = approved_accounts(slug)
    if not accounts:
        fail("No approved accounts for this brand.")

    log_path = MANIFEST_DIR / f"{slug}_fetch_log.csv"
    log = {r["shortcode"]: r for r in read_csv(log_path)} if log_path.is_file() else {}

    def save_log() -> None:
        write_csv(log_path, LOG_COLUMNS, sorted(log.values(), key=lambda r: r["shortcode"]))

    loader = make_loader()

    # 1. Fetch metadata for candidates that were not explicitly rejected.
    # Fetching metadata is necessary to determine the actual post owner;
    # an unreviewed non-official post is NEVER downloaded at this stage.
    todo = [u for u in urls if u["shortcode"] and u["verified"] != "no"
            and u["shortcode"] not in log]
    for i, item in enumerate(todo, start=1):
        code = item["shortcode"]
        try:
            result = fetch(loader, code, slug, accounts)
        except Blocked as exc:
            save_log()
            fail(f"Instagram blocked the request ({exc}). Progress saved: "
                 f"{len(log)} posts fetched, {len(todo) - i + 1} left. Re-run later to resume.")
        log[code] = {c: "" for c in LOG_COLUMNS} | result | {
            "shortcode": code, "post_url": f"https://www.instagram.com/p/{code}/"}
        save_log()
        print(f"fetched {i}/{len(todo)}: {code} {result['fetch_status']}")
        time.sleep(REQUEST_DELAY_SECONDS)

    # 2. Confirm official ownership and update the review workbook. Non-official
    # decisions remain solely the human's responsibility.
    auto_count = sync_official_verification(
        MANIFEST_DIR / f"{slug}_post_urls.xlsx", urls, reviewed, accounts, log)
    write_csv(MANIFEST_DIR / f"{slug}_post_urls.csv", URL_COLUMNS, reviewed)
    if auto_count:
        print(f"Official-photo verification updated for {auto_count} URL(s).")

    # 3. Determine eligibility. Only metadata-verified official photos or
    # human-approved non-official photos may reach the image downloader.
    rows = []
    for item in urls:
        code = item["shortcode"]
        rec = log.get(code, {})
        owner = rec.get("owner", "")
        member_added = item["url_source"] == "member_added" and item["verified"] == "yes"
        row = {"brand_slug": slug, "shortcode": code,
               "post_url": rec.get("post_url") or item["pasted_url"],
               "url_source": item["url_source"],
               "handle": owner,
               "account_type": accounts.get(owner) or ("creator" if member_added else ""),
               "post_datetime_utc": rec.get("post_datetime_utc", ""),
               "post_date": rec.get("post_date", ""), "post_type": rec.get("post_type", ""),
               "brand_match": rec.get("brand_match", ""), "selected": "no"}
        if item["verified"] == "no":
            row["selection_reason"] = "human_rejected"
        elif not code:
            row["selection_reason"] = "invalid_url"
        elif rec.get("fetch_status") != "ok":
            row["selection_reason"] = "fetch_unavailable"
        elif owner not in accounts and not member_added:
            row["selection_reason"] = "owner_not_approved"
        elif rec.get("post_type") not in {"image", "carousel_image"}:
            row["selection_reason"] = "non_image_post"
        elif not rec.get("image_url") and not (shots_dir / f"{code}.png").is_file():
            row["selection_reason"] = "no_image"
        elif accounts.get(owner) == "official" and item["verified"] in {"auto", "yes"}:
            row["selection_reason"] = "eligible"
        elif member_added:
            # A human-supplied yes is sufficient even when the owner is not in
            # the original approved-account lists.
            row["selection_reason"] = "eligible"
        elif accounts.get(owner) != "official" and item["verified"] == "yes":
            # Manual yes is sufficient evidence of brand relevance. Keep the
            # caption/hashtag brand_match flag without treating it as a gate.
            row["selection_reason"] = "eligible"
        elif accounts.get(owner) != "official":
            row["selection_reason"] = "awaiting_human_review"
        else:
            row["selection_reason"] = "official_not_verified"
        rows.append(row)

    # 3–4. Round-robin across accounts (handle order), newest first within each
    # account, until the caps are reached; download as posts are taken.
    hashes: set[str] = set()

    def take(row: dict) -> bool:
        """Return True only for a successfully obtained, unique image."""
        rec, code = log[row["shortcode"]], row["shortcode"]
        source = None
        screenshot = shots_dir / f"{code}.png"
        if rec.get("image_file") and (brand_dir / rec["image_file"]).is_file():
            source = brand_dir / rec["image_file"]
        elif screenshot.is_file():
            source, rec["collection_method"] = screenshot, "screenshot"
        elif rec.get("image_url") and not rec.get("download_error"):
            # Image URLs expire; on failure, re-fetch the post once for a fresh URL.
            for attempt in (1, 2):
                try:
                    source = download(loader, rec["image_url"], dl_dir, code)
                    rec.update(collection_method="fetch", image_file=f"_downloads/{source.name}")
                    break
                except (LoginRequiredException, TooManyRequestsException, AbortDownloadException) as exc:
                    save_log()
                    fail(f"Instagram blocked the download ({type(exc).__name__}). Progress saved; re-run later.")
                except InstaloaderException as exc:
                    rec["download_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                    if attempt == 2:
                        break
                    time.sleep(REQUEST_DELAY_SECONDS)
                    try:
                        fresh = fetch(loader, code, slug, accounts)
                    except Blocked as blocked:
                        save_log()
                        fail(f"Instagram blocked the request ({blocked}). Progress saved; re-run later.")
                    if fresh.get("fetch_status") != "ok" or not fresh.get("image_url"):
                        break
                    rec["image_url"], rec["download_error"] = fresh["image_url"], ""
            time.sleep(REQUEST_DELAY_SECONDS)
        if source is None:
            row["selection_reason"] = "awaiting_screenshot"
            return False  # An unavailable image never consumes one of the 40 slots.

        info = image_info(source)
        if info is None or info[0] not in FORMATS:
            row["selection_reason"] = "invalid_image"
            return False
        digest = sha256(source)
        if digest in hashes:
            row["selection_reason"] = "duplicate_image"
            return False
        hashes.add(digest)
        name = f"{row['handle']}_{row['post_date'].replace('-', '')}_{digest[:12]}.{FORMATS[info[0]]}"
        target = images_dir / name
        if source != target:
            shutil.copy2(source, target)
        rec["image_sha256"] = digest
        row.update(selected="yes", selection_reason="selected", image_id=digest,
                   image_path=f"{slug}/images/{name}", collection_method=rec["collection_method"],
                   file_format=FORMATS[info[0]], width=info[1], height=info[2])
        return True

    queues: dict[str, list[dict]] = {}
    for row in sorted((r for r in rows if r["selection_reason"] == "eligible"),
                      key=lambda r: (r["post_datetime_utc"], r["shortcode"]), reverse=True):
        queues.setdefault(row["handle"], []).append(row)
    per_account, n_official, n_total = Counter(), 0, 0

    def can_take(handle: str) -> bool:
        is_official = accounts.get(handle) == "official"
        cap = MAX_PER_OFFICIAL_ACCOUNT if is_official else MAX_PER_ACCOUNT
        if per_account[handle] >= cap:
            return False
        return not (is_official and n_official >= MAX_OFFICIAL)

    progress = True
    while n_total < IMAGES_PER_BRAND and progress:
        progress = False
        for handle in sorted(queues):
            if n_total >= IMAGES_PER_BRAND:
                break
            # One slot per account per round; skipped posts (duplicate or
            # invalid image) do not use the account's turn.
            while queues[handle] and can_take(handle):
                row = queues[handle].pop(0)
                if take(row):
                    per_account[handle] += 1
                    n_official += accounts.get(handle) == "official"
                    n_total += 1
                    progress = True
                    break

    for handle, queue in queues.items():
        for row in queue:
            is_official = accounts.get(handle) == "official"
            cap = MAX_PER_OFFICIAL_ACCOUNT if is_official else MAX_PER_ACCOUNT
            if per_account[handle] >= cap:
                row["selection_reason"] = "account_cap"
            elif is_official and n_official >= MAX_OFFICIAL:
                row["selection_reason"] = "official_cap"
            else:
                row["selection_reason"] = "brand_cap"
    save_log()

    # Do not delete historical or manually added image files on a rerun.
    # Downstream analysis must use only files marked selected=yes in manifest.
    keep = {Path(r["image_path"]).name for r in rows if r["selected"] == "yes"}
    unreferenced = sorted(f.name for f in images_dir.iterdir()
                          if f.is_file() and f.name not in keep)

    rows.sort(key=lambda r: (r["selected"] != "yes", r["handle"], r["post_datetime_utc"]))
    write_csv(MANIFEST_DIR / f"{slug}.csv", MANIFEST_COLUMNS, rows)

    selected = [r for r in rows if r["selected"] == "yes"]
    by_type = Counter(r["account_type"] for r in selected)
    days = sorted(r["post_date"] for r in selected)
    summary = {
        "brand_slug": slug,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_images": IMAGES_PER_BRAND, "max_official": MAX_OFFICIAL,
        "max_per_account": MAX_PER_ACCOUNT,
        "max_per_official_account": MAX_PER_OFFICIAL_ACCOUNT,
        "urls_listed": len(reviewed),
        "urls_by_source_and_verified": {
            src: {v: sum(r["url_source"] == src and r["verified"] == ("" if v == "pending" else v)
                         for r in reviewed)
                  for v in ("yes", "no", "auto", "pending")}
            for src in URL_SOURCES},
        "urls_verified_unique": sum(u["verified"] in {"yes", "auto"} for u in urls),
        "official_auto_verified": sum(u["verified"] == "auto" for u in urls),
        "awaiting_human_review": sum(r["selection_reason"] == "awaiting_human_review" for r in rows),
        "unreferenced_image_files": len(unreferenced),
        "selected_total": len(selected),
        "complete": len(selected) == IMAGES_PER_BRAND,
        "shortfall": IMAGES_PER_BRAND - len(selected),
        "awaiting_screenshot": sum(r["selection_reason"] == "awaiting_screenshot" for r in rows),
        "selected_by_account_type": {t: by_type[t] for t in ["official", "retailer", "media", "creator"]},
        "official_share": round(by_type["official"] / len(selected), 4) if selected else None,
        "selected_by_account": dict(sorted(Counter(r["handle"] for r in selected).items())),
        "selected_by_url_source": {src: sum(r["url_source"] == src for r in selected)
                                   for src in URL_SOURCES},
        "selected_by_method": dict(sorted(Counter(r["collection_method"] for r in selected).items())),
        "selected_by_brand_match": dict(sorted(Counter(r["brand_match"] for r in selected).items())),
        "selected_by_post_type": dict(sorted(Counter(r["post_type"] for r in selected).items())),
        "approved_accounts_without_images": sorted(set(accounts) - {r["handle"] for r in selected}),
        "earliest_selected_post_date": days[0] if days else None,
        "latest_selected_post_date": days[-1] if days else None,
        "rows_by_reason": dict(sorted(Counter(r["selection_reason"] for r in rows).items())),
    }
    (MANIFEST_DIR / f"{slug}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["complete"]:
        fail(f"{slug} is incomplete: {len(selected)}/{IMAGES_PER_BRAND} images. "
             "Progress saved; add valid image posts and rerun.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--brand", required=True, choices=load_brand_slugs())
    parser.add_argument("--init", action="store_true", help="create the empty URL workbook")
    args = parser.parse_args()
    if args.init:
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        init_workbook(MANIFEST_DIR / f"{args.brand}_post_urls.xlsx")
    else:
        run(args.brand)


if __name__ == "__main__":
    main()
