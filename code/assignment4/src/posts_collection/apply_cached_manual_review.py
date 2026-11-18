#!/usr/bin/env python3
"""Download reviewed manual posts from cached image URLs only."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from openpyxl import load_workbook

import collect_posts as cp
from _common import (MANIFEST_DIR, MAX_PER_ACCOUNT, MAX_PER_OFFICIAL_ACCOUNT,
                     REQUEST_DELAY_SECONDS, approved_accounts, images_root,
                     read_csv, write_csv)

REVIEW_FILE = MANIFEST_DIR / "manual_review_remaining.csv"
PROGRESS_FILE = MANIFEST_DIR / "manual_review_progress.csv"
PROGRESS_COLUMNS = ["brand_slug", "handle", "post_url", "shortcode", "verified",
                    "image_selection", "status", "detail", "image_file", "image_sha256"]


def update_workbook(slug: str, codes: set[str]) -> None:
    path = MANIFEST_DIR / f"{slug}_post_urls.xlsx"
    wb = load_workbook(path)
    ws = wb["urls"]
    changed = 0
    for row_number in range(2, ws.max_row + 1):
        url = ws.cell(row=row_number, column=1).value
        if not url:
            continue
        code = cp.SHORTCODE_RE.search(str(url))
        if code and code.group(1) in codes and ws.cell(row=row_number, column=2).value != "yes":
            ws.cell(row=row_number, column=2, value="yes")
            changed += 1
    if changed:
        wb.save(path)
    print(f"{slug}: marked {changed} reviewed post(s) verified=yes")


def download_cached(loader, url: str, dl_dir: Path, code: str) -> Path:
    from instaloader.exceptions import (AbortDownloadException, ConnectionException,
                                        LoginRequiredException, TooManyRequestsException)
    try:
        return cp.download(loader, url, dl_dir, code)
    except (LoginRequiredException, TooManyRequestsException, AbortDownloadException,
            ConnectionException) as exc:
        raise cp.Blocked(f"{type(exc).__name__}: {exc}") from exc


def main() -> None:
    rows = read_csv(REVIEW_FILE)
    if not rows:
        cp.fail(f"No rows in {REVIEW_FILE}")
    progress = {r["post_url"]: r for r in read_csv(PROGRESS_FILE)}
    root = images_root()
    loader = cp.make_loader()

    by_brand: dict[str, list[dict]] = {}
    for row in rows:
        by_brand.setdefault(row["brand_slug"], []).append(row)

    for slug in sorted(by_brand):
        accounts = approved_accounts(slug)
        manifest = read_csv(MANIFEST_DIR / f"{slug}.csv")
        selected = Counter(r["handle"] for r in manifest if r.get("selected") == "yes")
        log_path = MANIFEST_DIR / f"{slug}_fetch_log.csv"
        log = {r["shortcode"]: r for r in read_csv(log_path)}
        dl_dir = root / slug / "_downloads"

        def save_log() -> None:
            write_csv(log_path, cp.LOG_COLUMNS, sorted(log.values(), key=lambda r: r["shortcode"]))

        downloadable = []
        for row in by_brand[slug]:
            code = row["shortcode"]
            rec = log.get(code, {})
            base = {"brand_slug": slug, "handle": row["handle"], "post_url": row["post_url"],
                    "shortcode": code, "verified": row["verified"],
                    "image_selection": row.get("image_selection", ""),
                    "image_file": rec.get("image_file", ""),
                    "image_sha256": rec.get("image_sha256", "")}
            if row.get("image_selection", "").strip():
                detail = ("only the first-image URL is cached; the selected image "
                          f"{row['image_selection']} requires fresh metadata")
                progress[row["post_url"]] = base | {"status": "missing_cached_selection", "detail": detail}
                rec.update({"fetch_status": "unavailable", "fetch_error": detail})
                log[code] = rec
                continue
            if not rec.get("image_url"):
                progress[row["post_url"]] = base | {"status": "missing_cached_url",
                                                    "detail": "no cached image URL"}
                rec.update({"fetch_status": "unavailable", "fetch_error": "no cached image URL"})
                log[code] = rec
                continue
            downloadable.append((row, rec, base))
        save_log()

        cap = {h: (MAX_PER_OFFICIAL_ACCOUNT if t == "official" else MAX_PER_ACCOUNT)
               for h, t in accounts.items()}
        downloadable.sort(key=lambda item: (item[0]["handle"],
                                            log[item[0]["shortcode"]].get("post_datetime_utc", "")),
                          reverse=False)
        by_account: dict[str, list] = {}
        for item in downloadable:
            by_account.setdefault(item[0]["handle"], []).append(item)
        for handle, items in by_account.items():
            items.sort(key=lambda item: log[item[0]["shortcode"]].get("post_datetime_utc", ""),
                       reverse=True)
            allowed = max(0, cap.get(handle, MAX_PER_ACCOUNT) - selected[handle])
            for index, (row, rec, base) in enumerate(items):
                if index >= allowed:
                    progress[row["post_url"]] = base | {
                        "status": "excluded_account_cap",
                        "detail": f"{handle} already has {selected[handle]} image(s); cap {cap.get(handle, MAX_PER_ACCOUNT)}"}
                    continue
                code = row["shortcode"]
                if rec.get("fetch_status") == "ok" and rec.get("image_file") and (root / slug / rec["image_file"]).is_file():
                    progress[row["post_url"]] = base | {"status": "already_downloaded", "detail": ""}
                    continue
                try:
                    source = download_cached(loader, rec["image_url"], dl_dir, code)
                except cp.Blocked as exc:
                    save_log()
                    write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                    cp.fail(f"Instagram blocked the cached-image download ({exc}). Progress saved; no retries.")
                    return
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {str(exc)[:200]}"
                    if "429" in detail or "Too Many" in detail:
                        save_log()
                        write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                        cp.fail(f"Cached image download was rate-limited ({detail}). Progress saved; no retries.")
                        return
                    rec.update({"fetch_status": "unavailable", "fetch_error": f"cached download: {detail}"})
                    log[code] = rec
                    save_log()
                    progress[row["post_url"]] = base | {"status": "failed_cached_url", "detail": detail}
                    write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                    time.sleep(REQUEST_DELAY_SECONDS)
                    continue
                digest = cp.sha256(source)
                rec.update({"fetch_status": "ok", "fetch_error": "", "image_file": f"{dl_dir.name}/{source.name}",
                            "image_sha256": digest, "collection_method": "fetch", "download_error": ""})
                log[code] = rec
                save_log()
                progress[row["post_url"]] = base | {
                    "status": "downloaded", "detail": "cached first image",
                    "image_file": rec["image_file"], "image_sha256": digest}
                print(f"{slug} {code}: downloaded cached image")
                write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))
                time.sleep(REQUEST_DELAY_SECONDS)
        write_csv(PROGRESS_FILE, PROGRESS_COLUMNS, list(progress.values()))

    for slug, items in by_brand.items():
        update_workbook(slug, {row["shortcode"] for row in items})

    counts = Counter(progress[r["post_url"]]["status"] for r in rows)
    print(f"Statuses for these 29: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
