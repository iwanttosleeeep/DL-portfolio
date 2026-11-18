#!/usr/bin/env python3
"""Build image-level, brand-level, brand-name, vocabulary, and figure outputs."""

from __future__ import annotations

import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ocr_common import (
    BRANDS,
    BRAND_ORDER,
    brand_fuzzy_bases,
    brand_token_set,
    find_brand_matches,
    interval_summary,
    is_brand_variant,
    is_content_token,
    normalize_token,
    tokenize,
)

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
TEXT_COLOR = "#2E86AB"
NO_TEXT_COLOR = "#D9D9D9"
BRAND_COLOR = "#C0392B"
GRID_COLOR = "#D9D9D9"


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def configure_logging(output_dir: Path) -> logging.Logger:
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ocr_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "logs" / "analysis.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_raw(raw_path: Path) -> list[dict]:
    records = []
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def build_tables(records: list[dict], metadata: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold = float(metadata["confidence_threshold"])
    image_rows = []
    detection_rows = []
    for record in records:
        brand = record["brand"]
        kept = [d for d in record["detections"] if float(d["confidence"]) >= threshold]
        region_texts = [str(d["text"]) for d in kept]
        all_region_texts = [str(d["text"]) for d in record["detections"]]
        tokens: list[str] = []
        for text in region_texts:
            tokens.extend(tokenize(text))
        normalized_tokens = [normalize_token(token) for token in tokens]
        content_tokens = [
            normalize_token(token)
            for token in tokens
            if is_content_token(normalize_token(token))
        ]
        # Match on the concatenated recognized text so that a brand name split
        # across two OCR regions (e.g. "FORTNUM" + "& MASON") is still counted.
        matches = find_brand_matches(" ".join(region_texts), brand)
        all_matches = find_brand_matches(" ".join(all_region_texts), brand)
        confidences = [float(d["confidence"]) for d in kept]
        text_content = " | ".join(region_texts)
        image_rows.append(
            {
                "brand": brand,
                "brand_display": BRANDS[brand],
                "filename": record["filename"],
                "image_id": record["image_id"],
                "image_width": record["image_width"],
                "image_height": record["image_height"],
                "raw_region_count": int(record["raw_region_count"]),
                "region_count": len(kept),
                "low_conf_region_count": int(record["raw_region_count"]) - len(kept),
                "has_text": int(bool(kept)),
                "no_text": int(not kept),
                "token_count": len(tokens),
                "unique_token_count": len(set(normalized_tokens)),
                "content_token_count": len(content_tokens),
                "char_count": int(sum(len(text) for text in region_texts)),
                "mean_confidence": round(safe_mean(confidences), 6) if confidences else "",
                "max_confidence": round(max(confidences), 6) if confidences else "",
                "min_confidence": round(min(confidences), 6) if confidences else "",
                "brand_mentions": len(matches),
                "brand_mentions_all_raw": len(all_matches),
                "brand_name_present": int(bool(matches)),
                "matched_brand_text": "; ".join(matches),
                "text_content": text_content,
                "elapsed_s": record.get("elapsed_s", ""),
            }
        )
        for detection in record["detections"]:
            bbox = detection["bbox"]
            width = float(bbox[2]) - float(bbox[0])
            height = float(bbox[3]) - float(bbox[1])
            confidence = float(detection["confidence"])
            detection_rows.append(
                {
                    "brand": brand,
                    "brand_display": BRANDS[brand],
                    "filename": record["filename"],
                    "image_id": record["image_id"],
                    "region_id": int(detection["region_id"]),
                    "text": str(detection["text"]),
                    "confidence": round(confidence, 6),
                    "kept": int(confidence >= threshold),
                    "bbox_x_min": round(float(bbox[0]), 2),
                    "bbox_y_min": round(float(bbox[1]), 2),
                    "bbox_x_max": round(float(bbox[2]), 2),
                    "bbox_y_max": round(float(bbox[3]), 2),
                    "box_width": round(width, 2),
                    "box_height": round(height, 2),
                    "box_area": round(max(width, 0.0) * max(height, 0.0), 2),
                    "polygon_json": json.dumps(detection["polygon"], ensure_ascii=False),
                    "token_count": len(tokenize(str(detection["text"]))),
                    "char_count": len(str(detection["text"])),
                }
            )
    image_df = pd.DataFrame(image_rows)
    detection_df = pd.DataFrame(detection_rows)
    return image_df, detection_df


def build_brand_summary(image_df: pd.DataFrame, detection_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand in BRAND_ORDER:
        subset = image_df[image_df["brand"] == brand]
        detections = detection_df[(detection_df["brand"] == brand) & (detection_df["kept"] == 1)]
        regions = [int(value) for value in subset["region_count"].tolist()]
        tokens = [int(value) for value in subset["token_count"].tolist()]
        chars = [int(value) for value in subset["char_count"].tolist()]
        region_stats = interval_summary(regions)
        token_stats = interval_summary(tokens)
        char_stats = interval_summary(chars)
        images_with_text = int(subset["has_text"].sum())
        images_no_text = int(subset["no_text"].sum())
        images_with_mention = int(subset["brand_name_present"].sum())
        total_mentions = int(subset["brand_mentions"].sum())
        confidences = detections["confidence"].astype(float).tolist()
        rows.append(
            {
                "brand": brand,
                "brand_display": BRANDS[brand],
                "n_images": len(subset),
                "images_with_text": images_with_text,
                "proportion_with_text": round(images_with_text / len(subset), 4),
                "images_no_text": images_no_text,
                "proportion_no_text": round(images_no_text / len(subset), 4),
                "images_text_fraction": f"{images_with_text}/{len(subset)}",
                "total_regions": int(subset["region_count"].sum()),
                "mean_regions_per_image": round(region_stats["mean"], 4),
                "sd_regions_per_image": round(region_stats["sd"], 4),
                "median_regions_per_image": round(region_stats["median"], 4),
                "q1_regions_per_image": round(region_stats["q1"], 4),
                "q3_regions_per_image": round(region_stats["q3"], 4),
                "min_regions_per_image": int(region_stats["min"]),
                "max_regions_per_image": int(region_stats["max"]),
                "total_tokens": int(subset["token_count"].sum()),
                "mean_tokens_per_image": round(token_stats["mean"], 4),
                "sd_tokens_per_image": round(token_stats["sd"], 4),
                "median_tokens_per_image": round(token_stats["median"], 4),
                "q1_tokens_per_image": round(token_stats["q1"], 4),
                "q3_tokens_per_image": round(token_stats["q3"], 4),
                "min_tokens_per_image": int(token_stats["min"]),
                "max_tokens_per_image": int(token_stats["max"]),
                "total_chars": int(subset["char_count"].sum()),
                "mean_chars_per_image": round(char_stats["mean"], 4),
                "raw_regions_total": int(subset["raw_region_count"].sum()),
                "retained_region_share": round(
                    int(subset["region_count"].sum()) / max(int(subset["raw_region_count"].sum()), 1), 4
                ),
                "mean_region_confidence": round(safe_mean(confidences), 4) if confidences else "",
                "median_region_confidence": round(float(np.median(confidences)), 4) if confidences else "",
                "images_with_brand_mention": images_with_mention,
                "proportion_images_with_brand_mention": round(images_with_mention / len(subset), 4),
                "brand_mention_fraction": f"{images_with_mention}/{len(subset)}",
                "total_brand_mentions": total_mentions,
                "mean_brand_mentions_all_images": round(total_mentions / len(subset), 4),
                "brand_mentions_per_100_images": round(100 * total_mentions / len(subset), 2),
            }
        )
    return pd.DataFrame(rows)


def build_brand_name_tables(image_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_rows = []
    summary_rows = []
    for brand in BRAND_ORDER:
        subset = image_df[image_df["brand"] == brand]
        variant_counts: Counter[str] = Counter()
        for value in subset["matched_brand_text"]:
            if value:
                for item in str(value).split("; "):
                    if item:
                        variant_counts[item] += 1
        images_with = int(subset["brand_name_present"].sum())
        total_mentions = int(subset["brand_mentions"].sum())
        summary_rows.append(
            {
                "brand": brand,
                "brand_display": BRANDS[brand],
                "n_images": len(subset),
                "images_with_brand_mention": images_with,
                "proportion_images_with_brand_mention": round(images_with / len(subset), 4),
                "total_brand_mentions": total_mentions,
                "mean_mentions_all_images": round(total_mentions / len(subset), 4),
                "mentions_per_100_images": round(100 * total_mentions / len(subset), 2),
                "max_mentions_single_image": int(subset["brand_mentions"].max()),
                "images_with_multiple_mentions": int((subset["brand_mentions"] >= 2).sum()),
                "matched_variants_json": json.dumps(dict(variant_counts), ensure_ascii=False, sort_keys=True),
            }
        )
        for _, row in subset.iterrows():
            match_rows.append(
                {
                    "brand": brand,
                    "brand_display": BRANDS[brand],
                    "filename": row["filename"],
                    "image_id": row["image_id"],
                    "brand_mentions": int(row["brand_mentions"]),
                    "brand_name_present": int(row["brand_name_present"]),
                    "matched_brand_text": row["matched_brand_text"],
                    "brand_mentions_all_raw": int(row["brand_mentions_all_raw"]),
                    "text_content": row["text_content"],
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(match_rows)


def build_vocabulary(image_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for brand in BRAND_ORDER:
        subset = image_df[image_df["brand"] == brand]
        excluded = brand_token_set(brand)
        fuzzy_bases = brand_fuzzy_bases(brand)
        unigram_counts: Counter[str] = Counter()
        unigram_df: Counter[str] = Counter()
        bigram_counts: Counter[str] = Counter()
        bigram_df: Counter[str] = Counter()
        for text in subset["text_content"]:
            image_unigrams: set[str] = set()
            image_bigrams: set[str] = set()
            image_unigram_counts: Counter[str] = Counter()
            image_bigram_counts: Counter[str] = Counter()
            for region in str(text).split(" | "):
                content = []
                for token in tokenize(region):
                    normalized = normalize_token(token)
                    if is_brand_variant(normalized, excluded, fuzzy_bases):
                        continue
                    if is_content_token(normalized):
                        content.append(normalized)
                image_unigram_counts.update(content)
                image_unigrams.update(content)
                for left, right in zip(content, content[1:]):
                    bigram = f"{left} {right}"
                    image_bigram_counts[bigram] += 1
                    image_bigrams.add(bigram)
            unigram_counts.update(image_unigram_counts)
            bigram_counts.update(image_bigram_counts)
            unigram_df.update(image_unigrams)
            bigram_df.update(image_bigrams)
        for ngram_type, counts, document_frequency in (
            ("unigram", unigram_counts, unigram_df),
            ("bigram", bigram_counts, bigram_df),
        ):
            ordered = sorted(document_frequency, key=lambda term: (-document_frequency[term], -counts[term], term))
            for rank, term in enumerate(ordered, start=1):
                rows.append(
                    {
                        "brand": brand,
                        "brand_display": BRANDS[brand],
                        "n_images": len(subset),
                        "ngram_type": ngram_type,
                        "ngram": term,
                        "term_count": int(counts[term]),
                        "image_df": int(document_frequency[term]),
                        "image_pct": round(100 * document_frequency[term] / len(subset), 2),
                        "rank": rank,
                    }
                )
    vocabulary = pd.DataFrame(rows)
    summary_rows = []
    for brand in BRAND_ORDER:
        subset = vocabulary[vocabulary["brand"] == brand]
        for ngram_type, top_n in (("unigram", 10), ("bigram", 5)):
            selected = subset[subset["ngram_type"] == ngram_type].sort_values(
                ["image_df", "term_count"], ascending=[False, False]
            ).head(top_n)
            for _, row in selected.iterrows():
                summary_rows.append(
                    {
                        "brand": brand,
                        "brand_display": BRANDS[brand],
                        "ngram_type": ngram_type,
                        "rank": int(row["rank"]),
                        "ngram": row["ngram"],
                        "term_count": int(row["term_count"]),
                        "image_df": int(row["image_df"]),
                        "image_pct": float(row["image_pct"]),
                    }
                )
    return vocabulary, pd.DataFrame(summary_rows)


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_text_presence(brand_summary: pd.DataFrame, figures_dir: Path, data_dir: Path) -> None:
    data = brand_summary.sort_values("proportion_with_text", ascending=True)
    source = data[
        ["brand", "brand_display", "n_images", "images_with_text", "proportion_with_text", "images_no_text", "proportion_no_text"]
    ]
    source.to_csv(data_dir / "figure_text_presence_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(data))
    text_pct = data["proportion_with_text"].to_numpy() * 100
    no_text_pct = data["proportion_no_text"].to_numpy() * 100
    ax.barh(y, text_pct, color=TEXT_COLOR, edgecolor="white", height=0.68, label="At least one region")
    ax.barh(y, no_text_pct, left=text_pct, color=NO_TEXT_COLOR, edgecolor="white", height=0.68, label="No recognized text")
    for index, row in enumerate(data.itertuples()):
        ax.text(
            min(text_pct[index] + 1.5, 96),
            index,
            f"{row.images_with_text}/40 ({text_pct[index]:.1f}%)",
            va="center",
            fontsize=8.5,
            color="#222222",
        )
    ax.set_yticks(y, data["brand_display"].tolist())
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the 40 brand images (%)")
    ax.set_title("Images with OCR-recognized text, by brand")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=1)
    save_figure(fig, figures_dir / "fig1_text_presence.png")


def plot_text_density(brand_summary: pd.DataFrame, figures_dir: Path, data_dir: Path) -> None:
    data = brand_summary.sort_values("mean_regions_per_image", ascending=True)
    source = data[
        [
            "brand",
            "brand_display",
            "n_images",
            "mean_regions_per_image",
            "sd_regions_per_image",
            "mean_tokens_per_image",
            "sd_tokens_per_image",
            "total_regions",
            "total_tokens",
        ]
    ]
    source.to_csv(data_dir / "figure_text_density_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    y = np.arange(len(data))
    metrics = [
        ("mean_regions_per_image", "sd_regions_per_image", "Detected text regions per image", "Mean regions"),
        ("mean_tokens_per_image", "sd_tokens_per_image", "Alphanumeric tokens per image", "Mean tokens"),
    ]
    for ax, (mean_col, sd_col, title, xlabel) in zip(axes, metrics):
        values = data[mean_col].to_numpy()
        errors = data[sd_col].fillna(0).to_numpy()
        ax.barh(y, values, xerr=errors, color=PALETTE[0], alpha=0.9, edgecolor="white", height=0.68,
                error_kw={"ecolor": "#333333", "elinewidth": 0.9, "capsize": 2.5})
        for index, value in enumerate(values):
            ax.text(value + max(values) * 0.02, index, f"{value:.2f}", va="center", fontsize=8.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, data["brand_display"].tolist())
    axes[0].set_xlim(left=0)
    axes[1].set_xlim(left=0)
    fig.suptitle("OCR text density: mean +/- SD across 40 images per brand", fontsize=13, y=1.02)
    save_figure(fig, figures_dir / "fig2_text_density.png")


def plot_brand_name_visibility(brand_name_summary: pd.DataFrame, figures_dir: Path, data_dir: Path) -> None:
    data = brand_name_summary.sort_values("proportion_images_with_brand_mention", ascending=True)
    source = data[
        [
            "brand",
            "brand_display",
            "n_images",
            "images_with_brand_mention",
            "proportion_images_with_brand_mention",
            "total_brand_mentions",
            "mentions_per_100_images",
        ]
    ]
    source.to_csv(data_dir / "figure_brand_name_visibility_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    y = np.arange(len(data))
    percentages = data["proportion_images_with_brand_mention"].to_numpy() * 100
    mention_rates = data["total_brand_mentions"].to_numpy()
    axes[0].barh(y, percentages, color=BRAND_COLOR, alpha=0.9, edgecolor="white", height=0.68)
    for index, row in enumerate(data.itertuples()):
        axes[0].text(
            percentages[index] + 1.5,
            index,
            f"{row.images_with_brand_mention}/40 ({percentages[index]:.1f}%)",
            va="center",
            fontsize=8.5,
        )
    axes[0].set_xlim(0, max(percentages.max() * 1.28, 10))
    axes[0].set_xlabel("Images with target-brand mention (%)")
    axes[0].set_title("Image-level brand-name presence")
    axes[0].grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axes[0].set_axisbelow(True)

    axes[1].barh(y, mention_rates, color=PALETTE[1], alpha=0.9, edgecolor="white", height=0.68)
    for index, row in enumerate(data.itertuples()):
        axes[1].text(
            mention_rates[index] + max(mention_rates.max() * 0.02, 0.2),
            index,
            f"{row.total_brand_mentions}",
            va="center",
            fontsize=8.5,
        )
    axes[1].set_xlim(0, max(mention_rates.max() * 1.20, 5))
    axes[1].set_xlabel("Total target-brand mentions across 40 images")
    axes[1].set_title("Total textual mentions")
    axes[1].grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    axes[1].set_axisbelow(True)
    axes[0].set_yticks(y, data["brand_display"].tolist())
    fig.suptitle("OCR-recognized target-brand visibility (case-insensitive alias matching)", fontsize=13, y=1.02)
    save_figure(fig, figures_dir / "fig3_brand_name_visibility.png")


def plot_text_volume_distribution(image_df: pd.DataFrame, brand_summary: pd.DataFrame, figures_dir: Path, data_dir: Path) -> None:
    order = brand_summary.sort_values("mean_regions_per_image", ascending=True)["brand"].tolist()
    source = image_df[["brand", "brand_display", "filename", "image_id", "region_count", "token_count"]].copy()
    source.to_csv(data_dir / "figure_text_volume_distribution_data.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharey=True)
    for ax, column, title, color in (
        (axes[0], "region_count", "Detected text regions per image", PALETTE[0]),
        (axes[1], "token_count", "Alphanumeric tokens per image", PALETTE[2]),
    ):
        values = [image_df[image_df["brand"] == brand][column].to_numpy() for brand in order]
        box = ax.boxplot(
            values,
            orientation="horizontal",
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops={"color": "#111111", "linewidth": 1.2},
            whiskerprops={"color": "#666666", "linewidth": 0.8},
            capprops={"color": "#666666", "linewidth": 0.8},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
            patch.set_edgecolor("#555555")
        rng = np.random.default_rng(6207)
        for index, series in enumerate(values, start=1):
            jitter = rng.uniform(-0.16, 0.16, size=len(series))
            ax.scatter(series, np.full(len(series), index) + jitter, s=9, color=color, alpha=0.75, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("Count per image")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    axes[0].set_yticks(np.arange(1, len(order) + 1), [BRANDS[brand] for brand in order])
    fig.suptitle("Distribution of OCR text volume across the 40 images per brand", fontsize=13, y=1.02)
    save_figure(fig, figures_dir / "fig4_text_volume_distribution.png")


def plot_vocabulary(brand_summary: pd.DataFrame, vocabulary: pd.DataFrame, figures_dir: Path, data_dir: Path) -> None:
    order = brand_summary.sort_values("mean_regions_per_image", ascending=False)["brand"].tolist()
    rows = []
    fig, axes = plt.subplots(5, 2, figsize=(11.8, 14.5))
    for ax, brand in zip(axes.ravel(), order):
        subset = vocabulary[(vocabulary["brand"] == brand) & (vocabulary["ngram_type"] == "unigram")].sort_values(
            ["image_df", "term_count"], ascending=[False, False]
        ).head(7)
        subset = subset.sort_values(["image_df", "term_count"], ascending=[True, True])
        labels = subset["ngram"].tolist()
        values = subset["image_pct"].to_numpy()
        ax.barh(np.arange(len(subset)), values, color=PALETTE[0], alpha=0.88, height=0.65)
        for index, (value, df) in enumerate(zip(values, subset["image_df"].tolist())):
            ax.text(value + max(values.max() * 0.02, 0.4), index, f"{df}/40", va="center", fontsize=8)
        ax.set_yticks(np.arange(len(subset)), labels)
        ax.set_xlim(0, max(values.max() * 1.22, 12))
        ax.set_title(BRANDS[brand], fontsize=11, loc="left")
        ax.set_xlabel("% of brand images containing term")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for _, row in subset.iterrows():
            rows.append(
                {
                    "brand": brand,
                    "brand_display": BRANDS[brand],
                    "ngram": row["ngram"],
                    "term_count": int(row["term_count"]),
                    "image_df": int(row["image_df"]),
                    "image_pct": float(row["image_pct"]),
                }
            )
    pd.DataFrame(rows).to_csv(data_dir / "figure_vocabulary_data.csv", index=False)
    fig.suptitle("Most frequent content unigrams per brand (brand and near-miss tokens excluded)", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(fig, figures_dir / "fig5_vocabulary.png")


def main() -> None:
    output_dir = Path("results/ocr").resolve()
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    configure_plotting()

    metadata = json.loads((output_dir / "model_metadata.json").read_text(encoding="utf-8"))
    records = load_raw(data_dir / "ocr_raw.jsonl")
    if len(records) != 400:
        raise RuntimeError(f"Expected 400 raw OCR records, found {len(records)}")
    image_df, detection_df = build_tables(records, metadata)
    if len(image_df) != 400:
        raise RuntimeError(f"Expected 400 image rows, found {len(image_df)}")
    if image_df.groupby("brand").size().ne(40).any():
        raise RuntimeError("Brand image counts are not 40 each")

    image_df.to_csv(data_dir / "ocr_image_results.csv", index=False)
    detection_df.to_csv(data_dir / "ocr_detections.csv", index=False)

    brand_summary = build_brand_summary(image_df, detection_df)
    brand_name_summary, brand_name_matches = build_brand_name_tables(image_df)
    vocabulary, vocabulary_summary = build_vocabulary(image_df)

    brand_summary.to_csv(data_dir / "brand_summary.csv", index=False)
    brand_name_summary.to_csv(data_dir / "brand_name_summary.csv", index=False)
    brand_name_matches.to_csv(data_dir / "brand_name_matches.csv", index=False)
    vocabulary.to_csv(data_dir / "text_frequency.csv", index=False)
    vocabulary_summary.to_csv(data_dir / "brand_vocabulary_summary.csv", index=False)

    plot_text_presence(brand_summary, figures_dir, data_dir)
    plot_text_density(brand_summary, figures_dir, data_dir)
    plot_brand_name_visibility(brand_name_summary, figures_dir, data_dir)
    plot_text_volume_distribution(image_df, brand_summary, figures_dir, data_dir)
    plot_vocabulary(brand_summary, vocabulary, figures_dir, data_dir)

    logger.info("Image rows: %d; detection rows: %d", len(image_df), len(detection_df))
    logger.info("Total kept regions: %d; total tokens: %d", int(image_df["region_count"].sum()), int(image_df["token_count"].sum()))
    logger.info("Images with text: %d/400; images with target-brand mention: %d/400",
                int(image_df["has_text"].sum()), int(image_df["brand_name_present"].sum()))
    logger.info("Wrote CSVs and five figures under %s", output_dir)


if __name__ == "__main__":
    main()
