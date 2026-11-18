#!/usr/bin/env python3
"""Apply the human-reviewed manual_review_pending.csv decisions."""

from __future__ import annotations

import csv
import re
import time
from collections import Counter
from datetime import timezone
from pathlib import Path

from openpyxl import load_workbook

import collect_posts as cp
from _common import (HKT, MANIFEST_DIR, REQUEST_DELAY_SECONDS, approved_accounts,
                     images_root, read_csv, write_csv)

REVIEW_FILE = MANIFEST_DIR / "manual_review_pending.csv"
PROGRESS_FILE = MANIFEST_DIR / "manual_review_progress.csv"
PROGRESS_COLUMNS = ["brand_slug", "handle", "post_url", "shortcode", "verified",
                    "image_selection", "status", "detail", "image_file", "image_sha256"]
SHORTCODE_RE = re.compile(r"instagram\.com/p/([A-Za-z0-9_-]+)", re.IGNORECASE)


def post_from_shortcode(loader, code: str):
    from instaloader import Post
    from instaloader.exceptions import (AbortDownloadException, ConnectionException,
                                        InstaloaderException, LoginRequiredException,
                                        QueryReturnedNotFoundException, TooManyRequestsException)
    try:
        return Post.from_shortcode(loader.context, code), None
    except QueryReturnedNotFoundException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    except (LoginRequiredException, TooManyRequestsException, AbortDownloadException,
            ConnectionException) as exc:
        raise cp.Blocked(f"{type(exc).__name__}: {exc}") from exc
    except InstaloaderException as exc:
        return None, f"{type(exc).__name__}: {exc}"


def download_image(loader, url: str, dl_dir: Path, code: str) -> Path:
    from instaloader.exceptions import (AbortDownloadException, ConnectionException,
                                        LoginRequiredException, TooManyRequestsException)
    try:
        return cp.download(loader, url, dl_dir, code)
    except (LoginRequiredException, TooManyRequestsException, AbortDownloadException,
            ConnectionException) as exc:
        raise cp.Blocked(f"{type(exc).__name__}: {exc}") from exc


def media_urls(post) -> tuple[str, list[str]]:
    if post.typename == "GraphImage" and not post.is_video:
        return "image", [post.url]
    if post.typename == "GraphSidecar":
        nodes = list(post.get_sidecar_nodes())
        if not nodes or any(node.is_video for node in nodes):
            return "non_image", []
        return "carousel_image", [node.display_url for node in nodes]
    return "non_image", []


def choose_index(selection: str, count: int) -> int:
    value = (selection or "").strip().lower()
    if not value:
        return 0
    if value == "last":
        return count - 1
    if not value.isdigit():
        raise ValueError(f"invalid image_selection {selection!r}")
    index = int(value) - 1
    if index < 0 or index >= count:
        raise ValueError(f"image_selection {selection!r} out of range for {count} image(s)")
    return index


def update_workbook(slug: str, decisions: dict[str, str]) -> None:
    path = MANIFEST_DIR / f"{slug}_post_urls.xlsx"
    wb = load_workbook(path)
    ws = wb["urls"]
    changes = 0
    for row_number in range(2, ws.max_row + 1):
        url = ws.cell(row=row_number, column=1).value
        if not url:
            continue
        match = SHORTCODE_RE.search(str(url))
        if not match:
            continue
        decision = decisions.get(match.group(1))
        if decision is None:
            continue
        if ws.cell(row=row_number, column=2).value != decision:
            ws.cell(row=row_number, column=2, value=decision)
            changes += 1
    if changes:
        wb.save(path)
    print(f"{slug}: updated {changes} review decision(s) in {path.name}")


def main() -> None:
    rows = read_csv(REVIEW_FILE)
    if not rows:
        cp.fail(f"No rows in {REVIEW_FILE}")
    decisions: dict[str, dict[str, str]] = {}
    for row in rows:
        match = SHORTCODE_RE.search(row.get("post_url", ""))
        if not match:
            cp.fail(f"Invalid post_url in {REVIEW_FILE}: {row.get('post_url')}")
        if row.get("verified") not in {"yes", "no"}:
            cp.fail(f"{row['post_url']}: verified must be yes or no.")
        decisions[match.group(1)] = row

    progress = {r["post_url"]: r for r in read_csv(PROGRESS_FILE)} if PROGRESS_FILE.is_file() else {}
    root = images_root()
    loader = cp.make_loader()
    by_brand: dict[str, list[dict]] = {}
    for row in rows:
        by_brand.setdefault(row["brand_slug"], []).append(row)

    for slug in sorted(by_brand):
        accounts = approved_accounts(slug)
        brand_dir = root / slug
        dl_dir = brand_dir / "_downloads"
        log_path = MANIFEST_DIR / f"{slug}_fetch_log.csv"
        log = {r["shortcode"]: r for r in read_csv(log_path)} if log_path.is_file() else {}

        def save_log() -> None:
            write_csv(log_path, cp.LOG_COLUMNS, sorted(log.values(), key=lambda r: r["shortcode"]))

        for row in by_brand[slug]:
            match = SHORTCODE_RE.search(row["post_url"])
            code = match.group(1)
            base = {"brand_slug": slug, "handle": row["handle"], "post_url": row["post_url"],
                    "shortcode": code, "verified": row["verified"],
                    "image_selection": row.get("image_selection", "")}
            if row["verified"] == "no":
                progress[row["post_url"]] = base | {"status": "human_rejected", "detail": ""}
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                continue

            rec = log.get(code, {})
            rec.setdefault("shortcode", code)
            rec.setdefault("post_url", f"https://www.instagram.com/p/{code}/")
            existing_file = rec.get("image_file", "")
            if (rec.get("fetch_status") == "ok" and existing_file
                    and (brand_dir / existing_file).is_file()):
                progress[row["post_url"]] = base | {"status": "already_downloaded",
                                                    "detail": "", "image_file": existing_file,
                                                    "image_sha256": rec.get("image_sha256", "")}
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                continue

            if row["handle"].lower() not in accounts:
                progress[row["post_url"]] = base | {"status": "owner_not_approved", "detail": ""}
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                continue

            try:
                post, error = post_from_shortcode(loader, code)
            except cp.Blocked as exc:
                save_log()
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                cp.fail(f"Instagram blocked the request ({exc}). Progress saved; re-run later to resume.")
                return
            if post is None:
                rec.update({"fetch_status": "unavailable", "fetch_error": error})
                log[code] = rec
                save_log()
                progress[row["post_url"]] = base | {"status": "failed", "detail": error}
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            post_type, urls = media_urls(post)
            utc = post.date_utc.replace(tzinfo=timezone.utc)
            rec.update({"owner": post.owner_username.lower(), "post_type": post_type,
                        "post_datetime_utc": utc.isoformat(),
                        "post_date": utc.astimezone(HKT).date().isoformat()})
            if post_type == "non_image":
                rec.update({"fetch_status": "ok", "fetch_error": "", "image_url": ""})
                log[code] = rec
                save_log()
                progress[row["post_url"]] = base | {"status": "excluded_non_image", "detail": ""}
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            try:
                index = choose_index(row.get("image_selection", ""), len(urls))
            except ValueError as exc:
                rec.update({"fetch_status": "unavailable", "fetch_error": str(exc)})
                log[code] = rec
                save_log()
                progress[row["post_url"]] = base | {"status": "failed", "detail": str(exc)}
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            chosen = urls[index]
            try:
                source = download_image(loader, chosen, dl_dir, code)
            except cp.Blocked:
                save_log()
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                cp.fail("Instagram blocked the download. Progress saved; re-run later to resume.")
                return
            except Exception as exc:
                rec.update({"fetch_status": "unavailable", "fetch_error": f"download: {type(exc).__name__}: {exc}",
                            "image_url": ""})
                log[code] = rec
                save_log()
                progress[row["post_url"]] = base | {"status": "failed",
                                                    "detail": f"download: {type(exc).__name__}: {exc}"}
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                time.sleep(REQUEST_DELAY_SECONDS)
                continue

            digest = cp.sha256(source)
            info = cp.image_info(source)
            rec.update({"fetch_status": "ok", "fetch_error": "", "image_url": chosen,
                        "image_file": f"{dl_dir.name}/{source.name}", "image_sha256": digest,
                        "collection_method": "fetch", "download_error": ""})
            log[code] = rec
            save_log()
            progress[row["post_url"]] = base | {"status": "downloaded", "detail": f"image {index + 1}/{len(urls)}",
                                                "image_file": rec["image_file"], "image_sha256": digest}
            write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
            print(f"{slug} {code}: downloaded {index + 1}/{len(urls)} "
                  f"({info[1]}x{info[2]})" if info else f"{slug} {code}: downloaded")
            time.sleep(REQUEST_DELAY_SECONDS)

    for slug in sorted(by_brand):
        update_workbook(slug, {code: d["verified"] for code, d in decisions.items()
                               if d["brand_slug"] == slug})

    counts = Counter(r["status"] for r in progress.values())
    print(f"Progress statuses: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
