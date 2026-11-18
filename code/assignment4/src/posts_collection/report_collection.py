#!/usr/bin/env python3
"""Consolidate Step 3 collection results for all selected brands."""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from _common import (ACCOUNTS_DIR, IMAGES_PER_BRAND, MANIFEST_DIR, MAX_OFFICIAL,
                     MAX_PER_ACCOUNT, MAX_PER_OFFICIAL_ACCOUNT, approved_accounts,
                     load_brand_slugs, read_csv, write_csv)

SHORTCODE_RE = re.compile(r"instagram\.com/p/([A-Za-z0-9_-]+)", re.IGNORECASE)
REASON_LABELS = {
    "non_image_post": "Reel/video or mixed-media carousel",
    "account_cap": "Excluded by per-account cap",
    "official_cap": "Excluded by 30 official-image brand cap",
    "brand_cap": "Brand target already met",
    "owner_not_approved": "Post owner not an approved account",
    "human_rejected": "Human rejected",
    "invalid_url": "Invalid URL",
    "duplicate_image": "Duplicate image (SHA-256)",
    "invalid_image": "Downloaded file not a valid image",
    "awaiting_screenshot": "Download unavailable; screenshot pending",
}
REPORT_COLUMNS = ["brand_slug", "handle", "account_type", "post_url", "shortcode",
                  "url_source", "verified", "fetch_status", "fetch_error", "post_type",
                  "post_date", "selected", "selection_reason", "selection_reason_label",
                  "image_path", "collection_method", "file_format", "width", "height",
                  "download_error"]
BRAND_COLUMNS = ["brand_slug", "urls_listed", "auto_verified", "human_yes", "human_no",
                 "pending_review", "selected_images", "target_images", "shortfall",
                 "official_selected", "non_official_selected", "awaiting_human_review",
                 "failed_urls", "excluded_non_image", "excluded_account_cap",
                 "excluded_owner_not_approved", "complete"]
ACCOUNT_COLUMNS = ["brand_slug", "handle", "account_type", "candidate_urls",
                   "auto_verified", "human_yes", "pending_review", "selected_images",
                   "per_account_cap", "remaining_to_account_cap", "awaiting_review",
                   "failed_urls", "excluded_posts"]
DOWNLOAD_COLUMNS = ["brand_slug", "handle", "account_type", "post_url", "post_date",
                    "post_type", "image_path", "collection_method", "file_format",
                    "width", "height"]
REVIEW_COLUMNS = ["brand_slug", "handle", "account_type", "post_url", "verified",
                  "url_source", "post_type", "post_date"]
PROGRESS_COLUMNS = ["brand_slug", "handle", "post_url", "shortcode", "verified",
                    "image_selection", "status", "detail", "image_file", "image_sha256"]


def shortcode(url: str) -> str:
    match = SHORTCODE_RE.search(url)
    return match.group(1) if match else ""


def markdown_table(columns: list[str], rows: list[dict]) -> str:
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = [str(row.get(col, "")).replace("|", "\\|") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    slugs = load_brand_slugs()
    all_rows: list[dict] = []
    brand_rows: list[dict] = []
    account_rows: list[dict] = []

    for slug in slugs:
        manifest = MANIFEST_DIR / f"{slug}.csv"
        log_path = MANIFEST_DIR / f"{slug}_fetch_log.csv"
        review_path = MANIFEST_DIR / f"{slug}_post_urls.csv"
        if not manifest.is_file():
            continue
        rows = read_csv(manifest)
        log = {r["shortcode"]: r for r in read_csv(log_path)} if log_path.is_file() else {}
        reviews = read_csv(review_path) if review_path.is_file() else []
        review_by_code = {}
        for review in reviews:
            code = shortcode(review.get("post_url", ""))
            if code:
                review_by_code[code] = review

        for row in rows:
            rec = log.get(row["shortcode"], {})
            review = review_by_code.get(row["shortcode"], {})
            reason = row.get("selection_reason", "")
            combined = dict(row)
            combined.update({
                "url_source": review.get("url_source", row.get("url_source", "")),
                "verified": review.get("verified", ""),
                "fetch_status": rec.get("fetch_status", ""),
                "fetch_error": rec.get("fetch_error", ""),
                "download_error": rec.get("download_error", ""),
                "selection_reason_label": REASON_LABELS.get(reason, reason),
            })
            all_rows.append(combined)

        selected = [r for r in rows if r.get("selected") == "yes"]
        reasons = Counter(r.get("selection_reason", "") for r in rows)
        verified = Counter(r.get("verified", "") for r in reviews)
        brand_rows.append({
            "brand_slug": slug,
            "urls_listed": len(reviews),
            "auto_verified": verified["auto"],
            "human_yes": verified["yes"],
            "human_no": verified["no"],
            "pending_review": verified[""],
            "selected_images": len(selected),
            "target_images": IMAGES_PER_BRAND,
            "shortfall": IMAGES_PER_BRAND - len(selected),
            "official_selected": sum(r.get("account_type") == "official" for r in selected),
            "non_official_selected": sum(r.get("account_type") != "official" for r in selected),
            "awaiting_human_review": reasons["awaiting_human_review"],
            "failed_urls": reasons["fetch_unavailable"],
            "excluded_non_image": reasons["non_image_post"],
            "excluded_account_cap": reasons["account_cap"],
            "excluded_owner_not_approved": reasons["owner_not_approved"],
            "complete": "yes" if len(selected) == IMAGES_PER_BRAND else "no",
        })

        accounts = approved_accounts(slug)
        for handle, account_type in sorted(accounts.items()):
            handle_rows = [r for r in rows if r.get("handle") == handle]
            account_reviews = [r for r in reviews
                               if shortcode(r.get("post_url", "")) in
                               {row["shortcode"] for row in handle_rows}]
            account_verified = Counter(r.get("verified", "") for r in account_reviews)
            handle_selected = [r for r in handle_rows if r.get("selected") == "yes"]
            cap = MAX_PER_OFFICIAL_ACCOUNT if account_type == "official" else MAX_PER_ACCOUNT
            account_rows.append({
                "brand_slug": slug,
                "handle": handle,
                "account_type": account_type,
                "candidate_urls": len(handle_rows),
                "auto_verified": account_verified["auto"],
                "human_yes": account_verified["yes"],
                "pending_review": account_verified[""],
                "selected_images": len(handle_selected),
                "per_account_cap": cap,
                "remaining_to_account_cap": max(0, cap - len(handle_selected)),
                "awaiting_review": sum(r.get("selection_reason") == "awaiting_human_review"
                                       for r in handle_rows),
                "failed_urls": sum(r.get("selection_reason") == "fetch_unavailable"
                                   for r in handle_rows),
                "excluded_posts": sum(r.get("selection_reason") in {
                    "non_image_post", "account_cap", "official_cap", "brand_cap",
                    "owner_not_approved", "human_rejected", "duplicate_image",
                    "invalid_image", "invalid_url", "awaiting_screenshot"}
                    for r in handle_rows),
            })

    downloads = [r for r in all_rows if r.get("selected") == "yes"]
    failed = [r for r in all_rows if r.get("selection_reason") == "fetch_unavailable"]
    awaiting = [r for r in all_rows if r.get("selection_reason") == "awaiting_human_review"]
    excluded = [r for r in all_rows if r.get("selection_reason") in set(REASON_LABELS) - {
        "awaiting_screenshot"}]
    exclusion_rows = [r for r in excluded if r.get("selection_reason") != "fetch_unavailable"]
    total_selected = len(downloads)
    total_shortfall = IMAGES_PER_BRAND * len(brand_rows) - total_selected

    write_csv(MANIFEST_DIR / "collection_report.csv", REPORT_COLUMNS, all_rows)
    write_csv(MANIFEST_DIR / "collection_report_by_brand.csv", BRAND_COLUMNS, brand_rows)
    write_csv(MANIFEST_DIR / "collection_report_by_account.csv", ACCOUNT_COLUMNS, account_rows)
    write_csv(MANIFEST_DIR / "collection_report_downloads.csv", DOWNLOAD_COLUMNS, downloads)
    write_csv(MANIFEST_DIR / "manual_review_remaining.csv", REVIEW_COLUMNS, awaiting)

    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    progress_path = MANIFEST_DIR / "manual_review_progress.csv"
    progress_rows = read_csv(progress_path) if progress_path.is_file() else []
    progress_counts = Counter(r.get("status", "") for r in progress_rows)
    if progress_rows:
        write_csv(MANIFEST_DIR / "manual_review_results.csv", PROGRESS_COLUMNS, progress_rows)
        unresolved_statuses = {"not_processed_http_429", "missing_cached_selection",
                               "missing_cached_url", "failed_cached_url"}
        write_csv(MANIFEST_DIR / "manual_review_remaining.csv", PROGRESS_COLUMNS,
                  [r for r in progress_rows if r.get("status") in unresolved_statuses])
    lines = [
        "# Step 3 Instagram Image Collection Report",
        "",
        f"Generated: {generated}",
        f"Brands reported: {len(brand_rows)}",
        f"Target: {IMAGES_PER_BRAND} images per brand ({IMAGES_PER_BRAND * len(brand_rows)} total)",
        f"Selected images: {total_selected}",
        f"Remaining shortfall: {total_shortfall}",
        f"Awaiting manual verification: {len(awaiting)}",
        f"Failed URLs: {len(failed)}",
        f"Excluded posts: {len(exclusion_rows)}",
        "",
    ]
    if progress_rows:
        lines += [
            "## Reviewed manual-review processing",
            "",
            f"Reviewed rows: {len(progress_rows)}",
            f"Downloaded: {progress_counts['downloaded']}",
            f"Skipped as human `no`: {progress_counts['human_rejected']}",
            f"Excluded by account cap: {progress_counts['excluded_account_cap']}",
            f"Not processed (selected image not cached): {progress_counts['missing_cached_selection']}",
            f"Failed cached URLs: {progress_counts['failed_cached_url'] + progress_counts['missing_cached_url']}",
            "The 3 unprocessed rows selected a non-first carousel image, but the fetch logs",
            "cache only the first-image URL; they need fresh metadata when Instagram access returns.",
            f"Status file: `{MANIFEST_DIR / 'manual_review_results.csv'}`",
            "",
        ]
    lines += [
        "## Brand summary",
        "",
        markdown_table(["Brand", "URLs", "Auto", "Selected", "Target", "Shortfall",
                        "Official", "Awaiting review", "Failed", "Non-image", "Cap", "Other"],
                       [{
                           "Brand": r["brand_slug"], "URLs": r["urls_listed"],
                           "Auto": r["auto_verified"], "Selected": r["selected_images"],
                           "Target": r["target_images"], "Shortfall": r["shortfall"],
                           "Official": r["official_selected"],
                           "Awaiting review": r["awaiting_human_review"],
                           "Failed": r["failed_urls"],
                           "Non-image": r["excluded_non_image"],
                           "Cap": r["excluded_account_cap"],
                           "Other": r["excluded_owner_not_approved"],
                       } for r in brand_rows]),
        "",
        "## Account summary",
        "",
        markdown_table(["Brand", "Handle", "Type", "URLs", "Selected", "Cap",
                        "Remaining to cap", "Awaiting review", "Failed"],
                       [{
                           "Brand": r["brand_slug"], "Handle": r["handle"],
                           "Type": r["account_type"], "URLs": r["candidate_urls"],
                           "Selected": r["selected_images"], "Cap": r["per_account_cap"],
                           "Remaining to cap": r["remaining_to_account_cap"],
                           "Awaiting review": r["awaiting_review"],
                           "Failed": r["failed_urls"],
                       } for r in account_rows]),
        "",
        "## Failed URLs",
        "",
    ]
    if failed:
        lines.append(markdown_table(["Brand", "URL", "Handle", "Error"],
                                    [{"Brand": r["brand_slug"], "URL": r["post_url"],
                                      "Handle": r.get("handle", ""),
                                      "Error": r.get("fetch_error", "")} for r in failed]))
    else:
        lines.append("None.")
    lines += ["", "## Posts awaiting manual verification", ""]
    if awaiting:
        lines.append(markdown_table(["Brand", "URL", "Handle", "Type", "Post type"],
                                    [{"Brand": r["brand_slug"], "URL": r["post_url"],
                                      "Handle": r.get("handle", ""),
                                      "Type": r.get("account_type", ""),
                                      "Post type": r.get("post_type", "")}
                                     for r in awaiting]))
    else:
        lines.append("None.")
    lines += ["", "## Excluded posts", ""]
    exclusion_rows = [r for r in excluded if r.get("selection_reason") != "fetch_unavailable"]
    if exclusion_rows:
        lines.append(markdown_table(["Brand", "URL", "Handle", "Post type", "Reason"],
                                    [{"Brand": r["brand_slug"], "URL": r["post_url"],
                                      "Handle": r.get("handle", ""),
                                      "Post type": r.get("post_type", ""),
                                      "Reason": r.get("selection_reason_label", "")}
                                     for r in exclusion_rows]))
    else:
        lines.append("None.")
    lines += ["", "## Successfully downloaded image manifests", "",
              f"Full row-level results: `{MANIFEST_DIR / 'collection_report.csv'}`",
              f"Downloaded images only: `{MANIFEST_DIR / 'collection_report_downloads.csv'}`",
              f"Per-brand summary: `{MANIFEST_DIR / 'collection_report_by_brand.csv'}`",
              f"Per-account summary: `{MANIFEST_DIR / 'collection_report_by_account.csv'}`",
              f"Remaining reviewed but unprocessed list: `{MANIFEST_DIR / 'manual_review_remaining.csv'}`",
              "Pending decisions must be entered as yes/no in the `verified` column of each",
              "`<slug>_post_urls.xlsx` workbook; this report does not change those decisions.",
              ""]
    (MANIFEST_DIR / "collection_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Selected images: {total_selected}")
    print(f"Remaining shortfall: {total_shortfall}")
    print(f"Awaiting manual verification: {len(awaiting)}")
    print(f"Failed URLs: {len(failed)}")
    print(f"Report: {MANIFEST_DIR / 'collection_report.md'}")


if __name__ == "__main__":
    main()
