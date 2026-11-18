#!/usr/bin/env python3
"""Independent audit of the OCR outputs, statistics, figures, and samples."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import textwrap
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from ocr_common import BRANDS, BRAND_ORDER

AUDIT_PATTERNS = {
    "bang-olufsen": (
        r"\bbang\s*(?:&|and)?\s*olufsen\b|\bb\s*\.?\s*&\s*\.?\s*o\b"
        r"|\bb\s*\.\s*o\b|\bb\s+(?:and\s+)?o\b"
    ),
    "fortnum-mason": r"\bfortnum\s*(?:(?:&|and)\s*)?mason\b|\bfortnum'?s\b",
    "herman-miller": r"\bherman\s*miller\b",
    "jellycat": r"\bjellycat\b",
    "jo-malone-london": r"\bjo\s*malone(?:\s*london)?\b",
    "kodak": r"\bkodak\b",
    "lego": r"\blego\b",
    "oura": r"\boura\b",
    "sennheiser": r"\bsennheiser\b",
    "stanley": r"\bstanley(?:\s+1913)?\b",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)?")

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"

CHECKS: list[dict] = []


def add_check(name: str, passed: bool, details: str = "") -> None:
    CHECKS.append({"check": name, "passed": bool(passed), "details": details})
    logger.info("%s | %s | %s", "PASS" if passed else "FAIL", name, details)


def configure_logging(output_dir: Path) -> logging.Logger:
    global logger
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ocr_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "logs" / "audit.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    return re.sub(r"\s+", " ", text.casefold()).strip()


def audit_matches(text: str, brand: str) -> list[str]:
    return [match.group(0) for match in re.finditer(AUDIT_PATTERNS[brand], audit_normalize(text))]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_dataset(root: Path, manifest: pd.DataFrame) -> None:
    actual_ids: set[str] = set()
    for brand in BRAND_ORDER:
        paths = sorted((root / brand).glob("*.jpg"))
        add_check(f"dataset_{brand}_40_images", len(paths) == 40, f"found {len(paths)}")
        for path in paths:
            actual_ids.add(f"{brand}/{path.name}")
    add_check("dataset_400_unique_filenames", len(actual_ids) == 400, f"found {len(actual_ids)}")

    manifest_ids = set(manifest["image_id"])
    add_check("manifest_matches_disk_filenames", manifest_ids == actual_ids,
              f"{len(manifest_ids)} manifest IDs, {len(actual_ids)} disk IDs")

    hash_mismatches = []
    for row in manifest.itertuples():
        path = root / row.brand / row.filename
        if not path.exists() or sha256(path) != row.sha256:
            hash_mismatches.append(row.image_id)
    add_check("dataset_files_unchanged_sha256", not hash_mismatches,
              f"{len(hash_mismatches)} mismatches" if hash_mismatches else "all 400 match frozen manifest")


def verify_raw_and_images(raw: list[dict], image_df: pd.DataFrame, detection_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    threshold = float(metadata["confidence_threshold"])
    ids = [record["image_id"] for record in raw]
    add_check("raw_records_400", len(raw) == 400, f"found {len(raw)}")
    add_check("raw_records_unique", len(set(ids)) == len(ids), f"{len(set(ids))} unique")
    add_check("raw_no_inference_errors", all(not record.get("error") for record in raw),
              f"{sum(bool(record.get('error')) for record in raw)} errors")
    add_check("image_results_400", len(image_df) == 400, f"found {len(image_df)}")
    add_check("image_results_unique", image_df["image_id"].nunique() == 400,
              f"{image_df['image_id'].nunique()} unique")
    counts = image_df.groupby("brand").size().to_dict()
    add_check("image_results_40_per_brand", all(counts.get(brand) == 40 for brand in BRAND_ORDER), str(counts))
    add_check("image_results_all_zero_text_present", int((image_df["region_count"] == 0).sum()) > 0,
              f"{int((image_df['region_count'] == 0).sum())} zero-text images retained")

    add_check("detections_sum_raw_region_count",
              len(detection_df) == sum(record["raw_region_count"] for record in raw),
              f"{len(detection_df)} detection rows")
    kept_from_detections = int(detection_df["kept"].sum())
    add_check("detections_sum_kept_region_count",
              kept_from_detections == int(image_df["region_count"].sum()),
              f"{kept_from_detections} kept detections")
    add_check("detection_kept_flag_matches_threshold",
              bool(((detection_df["confidence"] >= threshold) == (detection_df["kept"] == 1)).all()),
              f"threshold={threshold}")
    geometry_ok = bool(
        ((detection_df["bbox_x_min"] <= detection_df["bbox_x_max"])
         & (detection_df["bbox_y_min"] <= detection_df["bbox_y_max"])).all()
    )
    add_check("detection_bbox_geometry_valid", geometry_ok, "all x_min<=x_max and y_min<=y_max")

    mismatch_rows = []
    image_lookup = image_df.set_index("image_id")
    for record in raw:
        image_id = record["image_id"]
        kept = [d for d in record["detections"] if float(d["confidence"]) >= threshold]
        region_count = len(kept)
        token_count = sum(len(TOKEN_RE.findall(str(d["text"]))) for d in kept)
        matches = audit_matches(" ".join(str(d["text"]) for d in kept), record["brand"])
        observed = {
            "region_count": region_count,
            "token_count": token_count,
            "has_text": int(bool(kept)),
            "brand_mentions": len(matches),
            "brand_name_present": int(bool(matches)),
        }
        reported = image_lookup.loc[image_id]
        differences = {
            key: (reported[key], value)
            for key, value in observed.items()
            if int(reported[key]) != int(value)
        }
        if differences:
            mismatch_rows.append({"image_id": image_id, **{f"{k}_reported": v[0] for k, v in differences.items()},
                                  **{f"{k}_recomputed": v[1] for k, v in differences.items()}})
    add_check("independent_image_metrics_reproduced", not mismatch_rows,
              f"{len(mismatch_rows)} mismatches")
    return pd.DataFrame(mismatch_rows)


def verify_brand_statistics(image_df: pd.DataFrame, brand_summary: pd.DataFrame,
                            brand_name_summary: pd.DataFrame, brand_name_matches: pd.DataFrame) -> None:
    expected_counts = {brand: 40 for brand in BRAND_ORDER}
    add_check("brand_summary_denominators_40",
              all(int(row.n_images) == expected_counts[row.brand] for row in brand_summary.itertuples()),
              "all n_images=40")
    proportion_ok = True
    details = []
    for row in brand_summary.itertuples():
        expected_text = row.images_with_text / 40
        expected_notext = row.images_no_text / 40
        expected_mention = row.images_with_brand_mention / 40
        if abs(row.proportion_with_text - expected_text) > 1e-6:
            proportion_ok = False
            details.append(f"{row.brand} text proportion")
        if abs(row.proportion_no_text - expected_notext) > 1e-6:
            proportion_ok = False
            details.append(f"{row.brand} no-text proportion")
        if abs(row.proportion_images_with_brand_mention - expected_mention) > 1e-6:
            proportion_ok = False
            details.append(f"{row.brand} mention proportion")
    add_check("brand_proportions_use_40_denominator", proportion_ok, "; ".join(details) or "all consistent")

    stats_mismatches = []
    for brand in BRAND_ORDER:
        subset = image_df[image_df["brand"] == brand]
        reported = brand_summary[brand_summary["brand"] == brand].iloc[0]
        expected = {
            "images_with_text": int(subset["has_text"].sum()),
            "images_no_text": int(subset["no_text"].sum()),
            "total_regions": int(subset["region_count"].sum()),
            "mean_regions_per_image": round(float(subset["region_count"].mean()), 4),
            "mean_tokens_per_image": round(float(subset["token_count"].mean()), 4),
            "total_tokens": int(subset["token_count"].sum()),
            "images_with_brand_mention": int(subset["brand_name_present"].sum()),
            "total_brand_mentions": int(subset["brand_mentions"].sum()),
        }
        for key, value in expected.items():
            if abs(float(reported[key]) - float(value)) > 1e-6:
                stats_mismatches.append(f"{brand}:{key}")
    add_check("brand_statistics_recomputed_from_images", not stats_mismatches,
              "; ".join(stats_mismatches) or "all brand statistics match")

    name_mismatches = []
    for brand in BRAND_ORDER:
        subset = image_df[image_df["brand"] == brand]
        reported = brand_name_summary[brand_name_summary["brand"] == brand].iloc[0]
        expected_images = int(subset["brand_name_present"].sum())
        expected_mentions = int(subset["brand_mentions"].sum())
        if int(reported["images_with_brand_mention"]) != expected_images:
            name_mismatches.append(f"{brand}:images")
        if int(reported["total_brand_mentions"]) != expected_mentions:
            name_mismatches.append(f"{brand}:mentions")
    add_check("brand_name_summary_recomputed", not name_mismatches,
              "; ".join(name_mismatches) or "all brand-name counts match")
    add_check("brand_name_matches_has_400_rows", len(brand_name_matches) == 400,
              f"{len(brand_name_matches)} rows")

    overall_mentions = int(image_df["brand_mentions"].sum())
    add_check("brand_name_matches_sum_to_total",
              int(brand_name_matches["brand_mentions"].sum()) == overall_mentions,
              f"{overall_mentions} mentions")


def verify_vocabulary(vocabulary: pd.DataFrame, vocabulary_summary: pd.DataFrame) -> None:
    add_check("text_frequency_nonempty", len(vocabulary) > 0, f"{len(vocabulary)} rows")
    pct_ok = bool((abs(vocabulary["image_pct"] - 100 * vocabulary["image_df"] / 40) < 0.01).all())
    add_check("text_frequency_pct_uses_40_denominator", pct_ok, "max deviation <0.01 pp")
    add_check("text_frequency_df_between_1_and_40",
              bool(vocabulary["image_df"].between(1, 40).all()), "all terms observed in 1-40 images")
    add_check("text_frequency_term_count_ge_df",
              bool((vocabulary["term_count"] >= vocabulary["image_df"]).all()),
              "every total count is at least its image frequency")

    exact_brand_tokens = {
        "bang-olufsen": {"bang", "olufsen", "bangolufsen", "bangandolufsen", "bo", "b", "o"},
        "fortnum-mason": {"fortnum", "mason", "fortnumandmason", "fortnummason", "s"},
        "herman-miller": {"herman", "miller", "hermanmiller"},
        "jellycat": {"jellycat"},
        "jo-malone-london": {"jo", "malone", "london", "jomalone", "jomalonelondon"},
        "kodak": {"kodak"},
        "lego": {"lego"},
        "oura": {"oura"},
        "sennheiser": {"sennheiser"},
        "stanley": {"stanley", "stanley1913"},
    }
    leaked = []
    for brand, terms in exact_brand_tokens.items():
        found = set(vocabulary[vocabulary["brand"] == brand]["ngram"]) & terms
        if found:
            leaked.append(f"{brand}:{sorted(found)}")
    add_check("target_brand_tokens_excluded_from_content", not leaked, "; ".join(leaked) or "no exact brand tokens")

    summary_mismatch = []
    for brand in BRAND_ORDER:
        for ngram_type, top_n in (("unigram", 10), ("bigram", 5)):
            expected = vocabulary[(vocabulary["brand"] == brand) & (vocabulary["ngram_type"] == ngram_type)].sort_values(
                ["image_df", "term_count"], ascending=[False, False]
            ).head(top_n)
            reported = vocabulary_summary[
                (vocabulary_summary["brand"] == brand) & (vocabulary_summary["ngram_type"] == ngram_type)
            ]
            if expected["ngram"].tolist() != reported["ngram"].tolist():
                summary_mismatch.append(f"{brand}:{ngram_type}")
    add_check("brand_vocabulary_summary_matches_text_frequency", not summary_mismatch,
              "; ".join(summary_mismatch) or "top terms consistent")


def verify_figures(output_dir: Path, brand_summary: pd.DataFrame, brand_name_summary: pd.DataFrame,
                   image_df: pd.DataFrame, vocabulary: pd.DataFrame) -> None:
    figures = {
        "fig1_text_presence.png": "figure_text_presence_data.csv",
        "fig2_text_density.png": "figure_text_density_data.csv",
        "fig3_brand_name_visibility.png": "figure_brand_name_visibility_data.csv",
        "fig4_text_volume_distribution.png": "figure_text_volume_distribution_data.csv",
        "fig5_vocabulary.png": "figure_vocabulary_data.csv",
    }
    for figure_name, data_name in figures.items():
        figure_path = output_dir / "figures" / figure_name
        exists = figure_path.exists() and figure_path.stat().st_size > 10_000
        dimensions = ""
        if figure_path.exists():
            with Image.open(figure_path) as image:
                dimensions = f"{image.size[0]}x{image.size[1]}"
        add_check(f"figure_exists_{figure_name}", exists, dimensions)

    fig1 = pd.read_csv(output_dir / "data" / "figure_text_presence_data.csv")
    fig1_ok = all(
        abs(row.proportion_with_text - row.images_with_text / 40) < 1e-6
        for row in fig1.itertuples()
    ) and len(fig1) == 10
    add_check("fig1_data_agrees_with_brand_summary", fig1_ok, "10 brands, proportions use /40")

    fig2 = pd.read_csv(output_dir / "data" / "figure_text_density_data.csv")
    merged = fig2.merge(brand_summary[["brand", "mean_regions_per_image", "mean_tokens_per_image"]], on="brand", suffixes=("_fig", "_summary"))
    fig2_ok = (
        len(merged) == 10
        and (abs(merged["mean_regions_per_image_fig"] - merged["mean_regions_per_image_summary"]) < 1e-6).all()
        and (abs(merged["mean_tokens_per_image_fig"] - merged["mean_tokens_per_image_summary"]) < 1e-6).all()
    )
    add_check("fig2_data_agrees_with_brand_summary", bool(fig2_ok), "means match brand_summary.csv")

    fig3 = pd.read_csv(output_dir / "data" / "figure_brand_name_visibility_data.csv")
    merged3 = fig3.merge(
        brand_name_summary[["brand", "images_with_brand_mention", "total_brand_mentions"]], on="brand", suffixes=("_fig", "_summary")
    )
    fig3_ok = (
        len(merged3) == 10
        and (merged3["images_with_brand_mention_fig"] == merged3["images_with_brand_mention_summary"]).all()
        and (merged3["total_brand_mentions_fig"] == merged3["total_brand_mentions_summary"]).all()
    )
    add_check("fig3_data_agrees_with_brand_name_summary", bool(fig3_ok), "image counts and mentions match")

    fig4 = pd.read_csv(output_dir / "data" / "figure_text_volume_distribution_data.csv")
    merged4 = fig4.merge(image_df[["image_id", "region_count", "token_count"]], on="image_id", suffixes=("_fig", "_csv"))
    fig4_ok = (
        len(fig4) == 400
        and (merged4["region_count_fig"] == merged4["region_count_csv"]).all()
        and (merged4["token_count_fig"] == merged4["token_count_csv"]).all()
    )
    add_check("fig4_data_agrees_with_image_results", bool(fig4_ok), "400 image counts match")

    fig5 = pd.read_csv(output_dir / "data" / "figure_vocabulary_data.csv")
    fig5_mismatch = []
    for brand in BRAND_ORDER:
        expected = vocabulary[
            (vocabulary["brand"] == brand) & (vocabulary["ngram_type"] == "unigram")
        ].sort_values(["image_df", "term_count"], ascending=[False, False]).head(7)
        reported = fig5[fig5["brand"] == brand].sort_values(["image_df", "term_count"], ascending=[False, False])
        if expected["ngram"].tolist() != reported["ngram"].tolist():
            fig5_mismatch.append(brand)
    add_check("fig5_data_agrees_with_text_frequency", not fig5_mismatch,
              "; ".join(fig5_mismatch) or "top-7 terms match")


def verify_metadata(output_dir: Path, metadata: dict) -> None:
    add_check("pretrained_weights_declared", metadata.get("pretrained_weights") is True, "metadata pretrained_weights=true")
    add_check("no_training_or_fine_tuning", metadata.get("training_or_fine_tuning") == "none",
              str(metadata.get("training_or_fine_tuning")))
    add_check("confidence_threshold_recorded", "confidence_threshold" in metadata,
              f"{metadata.get('confidence_threshold')}")
    weights_ok = True
    weight_details = []
    for role, info in metadata.get("model_files", {}).items():
        path = Path(info["filename"])
        if not path.exists() or sha256(path) != info["sha256"]:
            weights_ok = False
            weight_details.append(role)
    add_check("pretrained_weight_files_and_hashes_match", weights_ok,
             "; ".join(weight_details) or "CRAFT detector and English recognizer hashes verified")
    add_check("languages_recorded", bool(metadata.get("languages")), str(metadata.get("languages")))
    add_check("device_recorded", bool(metadata.get("device_used")), str(metadata.get("device_used")))


def verify_brand_matching() -> None:
    cases = [
        ("bang-olufsen", "Bang & Olufsen", 1),
        ("bang-olufsen", "BANG AND OLUFSEN", 1),
        ("bang-olufsen", "B&O", 1),
        ("bang-olufsen", "B.O.", 1),
        ("bang-olufsen", "B and O", 1),
        ("bang-olufsen", "BangOlufsen", 1),
        ("fortnum-mason", "Fortnum & Mason", 1),
        ("fortnum-mason", "Fortnum's", 1),
        ("fortnum-mason", "FORTNUM AND MASON", 1),
        ("fortnum-mason", "FortnumandMason", 1),
        ("herman-miller", "Herman Miller", 1),
        ("herman-miller", "HermanMiller", 1),
        ("jellycat", "Jellycat", 1),
        ("jo-malone-london", "Jo Malone London", 1),
        ("jo-malone-london", "JO MALONE", 1),
        ("jo-malone-london", "JoMaloneLondon", 1),
        ("kodak", "Kodak", 1),
        ("kodak", "KODAK 200", 1),
        ("lego", "LEGO", 1),
        ("lego", "Lego bricks", 1),
        ("oura", "Oura Ring", 1),
        ("oura", "aura", 0),
        ("sennheiser", "Sennheiser", 1),
        ("sennheiser", "Sennheisser", 0),
        ("stanley", "Stanley", 1),
        ("stanley", "Stanley 1913", 1),
        ("stanley", "Stanleyville", 0),
        ("lego", "LEGOland", 0),
        ("kodak", "Kodakery", 0),
    ]
    rows = []
    for brand, text, expected in cases:
        actual = len(audit_matches(text, brand))
        rows.append({"brand": brand, "text": text, "expected_matches": expected,
                     "actual_matches": actual, "passed": actual == expected})
    frame = pd.DataFrame(rows)
    add_check("brand_match_synthetic_cases", bool(frame["passed"].all()),
              f"{int(frame['passed'].sum())}/{len(frame)} cases passed")
    frame.to_csv(audit_dir() / "brand_match_tests.csv", index=False)


def audit_dir() -> Path:
    return Path("results/ocr/audit").resolve()


def build_visual_samples(raw: list[dict], image_df: pd.DataFrame) -> pd.DataFrame:
    raw_lookup = {record["image_id"]: record for record in raw}
    notes: dict[tuple[str, str], str] = {}
    existing_path = audit_dir() / "visual_sample_manifest.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        for row in existing.itertuples():
            if isinstance(row.manual_observation, str) and row.manual_observation.strip():
                notes[(row.sample_category, row.image_id)] = row.manual_observation
    selections = []
    for category in ("text_heavy", "no_text", "brand_mention", "ocr_errors"):
        for brand in BRAND_ORDER:
            subset = image_df[image_df["brand"] == brand]
            if category == "text_heavy":
                row = subset.sort_values(["region_count", "token_count"], ascending=[False, False]).iloc[0]
                reason = "most retained text regions in brand"
            elif category == "no_text":
                zero = subset[subset["region_count"] == 0].sort_values("filename")
                if zero.empty:
                    row = subset.sort_values(["region_count", "filename"], ascending=[True, True]).iloc[0]
                    reason = "fewest retained text regions (no zero-text image)"
                else:
                    row = zero.iloc[0]
                    reason = "zero retained text regions"
            elif category == "brand_mention":
                row = subset.sort_values(["brand_mentions", "region_count"], ascending=[False, False]).iloc[0]
                reason = "most target-brand mentions in brand"
            else:
                row = subset.sort_values(
                    ["low_conf_region_count", "raw_region_count", "mean_confidence"],
                    ascending=[False, False, True],
                ).iloc[0]
                reason = "most low-confidence raw OCR regions in brand"
            record = raw_lookup[row["image_id"]]
            snippets = [d["text"] for d in record["detections"] if float(d["confidence"]) >= 0.30]
            selections.append(
                {
                    "sample_category": category,
                    "brand": brand,
                    "brand_display": BRANDS[brand],
                    "filename": row["filename"],
                    "image_id": row["image_id"],
                    "selection_reason": reason,
                    "region_count": int(row["region_count"]),
                    "raw_region_count": int(row["raw_region_count"]),
                    "low_conf_region_count": int(row["low_conf_region_count"]),
                    "token_count": int(row["token_count"]),
                    "brand_mentions": int(row["brand_mentions"]),
                    "mean_confidence": row["mean_confidence"],
                    "text_excerpt": " | ".join(snippets)[:240],
                    "manual_observation": notes.get((category, row["image_id"]), ""),
                }
            )
    frame = pd.DataFrame(selections)
    frame.to_csv(audit_dir() / "visual_sample_manifest.csv", index=False)
    return frame


def make_contact_sheet(category: str, samples: pd.DataFrame, raw: list[dict], output_dir: Path) -> None:
    try:
        font = ImageFont.truetype(FONT_PATH, 15)
        font_bold = ImageFont.truetype(FONT_PATH, 17)
    except OSError:
        font = ImageFont.load_default()
        font_bold = font
    raw_lookup = {record["image_id"]: record for record in raw}
    tile_width, tile_height = 650, 690
    sheet = Image.new("RGB", (tile_width * 2, tile_height * 5), "white")
    draw_sheet = ImageDraw.Draw(sheet)
    legend = {
        "text_heavy": "Green boxes = retained text regions (confidence >= 0.30).",
        "no_text": "No retained text regions; image shown without boxes.",
        "brand_mention": "Green boxes = retained regions; title shows matched brand mentions.",
        "ocr_errors": "Green = retained (>=0.30); red = rejected low-confidence regions.",
    }[category]
    draw_sheet.text((10, 5), f"{category} visual sample - {legend}", fill="#111111", font=font_bold)

    for index, row in samples.iterrows():
        record = raw_lookup[row["image_id"]]
        image_path = Path(record["image_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        original_width, original_height = image.size
        image.thumbnail((tile_width - 20, 500))
        display_width, display_height = image.size
        column = index % 2
        row_index = index // 2
        origin_x = column * tile_width
        origin_y = row_index * tile_height
        image_x = origin_x + (tile_width - display_width) // 2
        image_y = origin_y + 135
        sheet.paste(image, (image_x, image_y))
        draw = ImageDraw.Draw(sheet)
        scale_x = display_width / original_width
        scale_y = display_height / original_height
        for detection in record["detections"]:
            confidence = float(detection["confidence"])
            kept = confidence >= 0.30
            if category == "no_text":
                continue
            if category == "ocr_errors":
                color = "#1E8449" if kept else "#C0392B"
            else:
                if not kept:
                    continue
                color = "#1E8449"
            x_min, y_min, x_max, y_max = detection["bbox"]
            box = [
                image_x + float(x_min) * scale_x,
                image_y + float(y_min) * scale_y,
                image_x + float(x_max) * scale_x,
                image_y + float(y_max) * scale_y,
            ]
            draw.rectangle(box, outline=color, width=2)

        title = f"{row['brand_display']} | {row['filename']}"
        metrics = (
            f"kept regions={row['region_count']}, tokens={row['token_count']}, "
            f"mentions={row['brand_mentions']}, raw={row['raw_region_count']}"
        )
        excerpt = textwrap.shorten(str(row["text_excerpt"]).replace("\n", " "), width=95, placeholder="...")
        draw.text((origin_x + 10, origin_y + 32), title, fill="#111111", font=font_bold)
        draw.text((origin_x + 10, origin_y + 55), metrics, fill="#333333", font=font)
        draw.text((origin_x + 10, origin_y + 78), f"OCR: {excerpt}", fill="#555555", font=font)
        draw.text((origin_x + 10, origin_y + 101), f"Selected: {row['selection_reason']}", fill="#555555", font=font)

    sheet.save(output_dir / "samples" / f"contact_{category}.png", dpi=(150, 150))


def main() -> None:
    global logger
    output_dir = Path("results/ocr").resolve()
    audit_output = audit_dir()
    audit_output.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)

    metadata = json.loads((output_dir / "model_metadata.json").read_text(encoding="utf-8"))
    raw = load_jsonl(output_dir / "data" / "ocr_raw.jsonl")
    manifest = pd.read_csv(output_dir / "data" / "dataset_manifest.csv")
    image_df = pd.read_csv(output_dir / "data" / "ocr_image_results.csv")
    detection_df = pd.read_csv(output_dir / "data" / "ocr_detections.csv")
    brand_summary = pd.read_csv(output_dir / "data" / "brand_summary.csv")
    brand_name_summary = pd.read_csv(output_dir / "data" / "brand_name_summary.csv")
    brand_name_matches = pd.read_csv(output_dir / "data" / "brand_name_matches.csv")
    vocabulary = pd.read_csv(output_dir / "data" / "text_frequency.csv")
    vocabulary_summary = pd.read_csv(output_dir / "data" / "brand_vocabulary_summary.csv")

    verify_dataset(output_dir.parent.parent / "images", manifest)
    mismatches = verify_raw_and_images(raw, image_df, detection_df, metadata)
    mismatches.to_csv(audit_output / "independent_count_mismatches.csv", index=False)
    verify_brand_statistics(image_df, brand_summary, brand_name_summary, brand_name_matches)
    verify_vocabulary(vocabulary, vocabulary_summary)
    verify_figures(output_dir, brand_summary, brand_name_summary, image_df, vocabulary)
    verify_metadata(output_dir, metadata)
    verify_brand_matching()

    samples = build_visual_samples(raw, image_df)
    for category in ("text_heavy", "no_text", "brand_mention", "ocr_errors"):
        make_contact_sheet(category, samples[samples["sample_category"] == category].reset_index(drop=True), raw, output_dir)

    checks = pd.DataFrame(CHECKS)
    checks.to_csv(audit_output / "audit_checks.csv", index=False)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks_total": len(checks),
        "checks_passed": int(checks["passed"].sum()),
        "checks_failed": int((~checks["passed"]).sum()),
        "all_checks_passed": bool(checks["passed"].all()),
        "images_processed": len(image_df),
        "brands": int(image_df["brand"].nunique()),
        "images_with_text": int(image_df["has_text"].sum()),
        "images_no_text": int(image_df["no_text"].sum()),
        "total_kept_regions": int(image_df["region_count"].sum()),
        "total_tokens": int(image_df["token_count"].sum()),
        "images_with_brand_mention": int(image_df["brand_name_present"].sum()),
        "total_brand_mentions": int(image_df["brand_mentions"].sum()),
        "visual_sample_images": len(samples),
        "contact_sheets": sorted(path.name for path in (output_dir / "samples").glob("contact_*.png")),
        "failed_checks": checks[~checks["passed"]]["check"].tolist(),
    }
    (audit_output / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Audit complete: %d/%d checks passed", summary["checks_passed"], summary["checks_total"])
    if summary["checks_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
