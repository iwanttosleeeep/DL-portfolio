#!/usr/bin/env python3
"""Build brand-level tables and figures from the raw YOLO detection outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
SAMPLES = ROOT / "samples"
IMAGE_ROOT = ROOT.parent.parent / "images"
BRAND_ORDER = [
    "Bang & Olufsen",
    "Fortnum & Mason",
    "Herman Miller",
    "Jellycat",
    "Jo Malone London",
    "Kodak",
    "LEGO",
    "Oura",
    "Sennheiser",
    "Stanley",
]
PALETTE = {
    "blue": "#2E6F9E",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "grey": "#B8B8B8",
    "dark": "#264653",
    "red": "#B23A48",
}


def pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else np.nan


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    image_df = pd.read_csv(DATA / "image_level_detections.csv")
    detection_df = pd.read_csv(DATA / "object_detections.csv")
    if len(image_df) != 400:
        raise RuntimeError(f"Expected 400 image rows, found {len(image_df)}")
    counts = image_df.groupby("brand").size()
    if len(counts) != 10 or not (counts == 40).all():
        raise RuntimeError("Expected exactly 40 images for each of 10 brands")
    if image_df["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image IDs in image-level table")
    if int(image_df["object_count"].sum()) != len(detection_df):
        raise RuntimeError("Detection rows do not match summed object counts")
    if set(image_df["brand_display"]) != set(BRAND_ORDER):
        raise RuntimeError("Unexpected brand display names")
    return image_df, detection_df


def build_tables(image_df: pd.DataFrame, detection_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    brand_images = (
        image_df.groupby("brand_display", as_index=False)
        .agg(
            images=("filename", "size"),
            images_with_detection=("has_detection", "sum"),
            images_without_detection=("zero_detection", "sum"),
            total_detections=("object_count", "sum"),
            mean_objects_per_image=("object_count", "mean"),
            sd_objects_per_image=("object_count", "std"),
            median_objects_per_image=("object_count", "median"),
            min_objects_per_image=("object_count", "min"),
            max_objects_per_image=("object_count", "max"),
        )
        .set_index("brand_display")
        .reindex(BRAND_ORDER)
        .reset_index()
    )
    brand_images["pct_images_with_detection"] = brand_images.apply(
        lambda row: pct(row["images_with_detection"], row["images"]), axis=1
    )
    brand_images["pct_images_without_detection"] = brand_images.apply(
        lambda row: pct(row["images_without_detection"], row["images"]), axis=1
    )
    brand_images["se_objects_per_image"] = (
        brand_images["sd_objects_per_image"] / np.sqrt(brand_images["images"])
    )
    brand_images["mean_objects_per_detected_image"] = brand_images.apply(
        lambda row: row["total_detections"] / row["images_with_detection"]
        if row["images_with_detection"]
        else np.nan,
        axis=1,
    )

    confidence_stats = (
        detection_df.groupby("brand_display", as_index=False)
        .agg(
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            max_confidence=("confidence", "max"),
        )
    )
    brand_images = brand_images.merge(confidence_stats, on="brand_display", how="left")

    brand_class = (
        detection_df.groupby(["brand_display", "class_name"], as_index=False)
        .agg(
            detection_count=("class_name", "size"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
        )
    )
    image_class = (
        detection_df.groupby(["brand_display", "filename", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "detections_in_image"})
    )
    images_with_class = (
        image_class.groupby(["brand_display", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "images_with_class"})
    )
    brand_class = brand_class.merge(images_with_class, on=["brand_display", "class_name"], how="left")
    brand_totals = brand_images.set_index("brand_display")["total_detections"]
    brand_image_counts = brand_images.set_index("brand_display")["images"]
    brand_class["brand_total_detections"] = brand_class["brand_display"].map(brand_totals)
    brand_class["brand_image_count"] = brand_class["brand_display"].map(brand_image_counts)
    brand_class["pct_brand_detections"] = (
        100.0 * brand_class["detection_count"] / brand_class["brand_total_detections"]
    )
    brand_class["pct_images_with_class"] = (
        100.0 * brand_class["images_with_class"] / brand_class["brand_image_count"]
    )
    brand_class["mean_detections_per_image"] = (
        brand_class["detection_count"] / brand_class["brand_image_count"]
    )
    brand_class = brand_class.sort_values(
        ["brand_display", "detection_count", "images_with_class", "class_name"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    brand_class["class_rank_within_brand"] = brand_class.groupby("brand_display").cumcount() + 1

    overall_class = (
        detection_df.groupby("class_name", as_index=False)
        .agg(
            detection_count=("class_name", "size"),
            images_with_class=("image_id", "nunique"),
            brands_present=("brand_display", "nunique"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            mean_area_fraction=("area_fraction", "mean"),
        )
        .sort_values(["detection_count", "images_with_class", "class_name"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    overall_class["pct_all_detections"] = 100.0 * overall_class["detection_count"] / len(detection_df)
    overall_class["pct_images_with_class"] = 100.0 * overall_class["images_with_class"] / len(image_df)
    overall_class["pct_brands_present"] = 100.0 * overall_class["brands_present"] / len(BRAND_ORDER)
    overall_class["class_rank"] = np.arange(1, len(overall_class) + 1)

    major_classes = (
        overall_class.sort_values(["images_with_class", "detection_count"], ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    major_classes["major_rank"] = np.arange(1, len(major_classes) + 1)
    major_classes = major_classes[
        ["major_rank", "class_name", "images_with_class", "pct_images_with_class", "detection_count", "brands_present"]
    ]

    image_prop = brand_class.pivot(index="brand_display", columns="class_name", values="pct_images_with_class")
    image_prop = image_prop.reindex(index=BRAND_ORDER, columns=major_classes["class_name"]).fillna(0.0)
    image_prop_long = image_prop.reset_index().rename_axis(None, axis=1)
    image_prop_long.to_csv(DATA / "brand_class_image_proportions_wide.csv", index=False)

    detection_share = brand_class.pivot(index="brand_display", columns="class_name", values="pct_brand_detections")
    detection_share = detection_share.reindex(index=BRAND_ORDER, columns=major_classes["class_name"]).fillna(0.0)
    detection_share_long = detection_share.reset_index().rename_axis(None, axis=1)
    detection_share_long.to_csv(DATA / "brand_class_detection_shares_wide.csv", index=False)

    top_classes = (
        brand_class[brand_class["class_rank_within_brand"] <= 5]
        .copy()
        .sort_values(["brand_display", "class_rank_within_brand"])
    )

    all_classes = sorted(overall_class["class_name"].tolist())
    count_matrix = (
        brand_class.pivot(index="brand_display", columns="class_name", values="detection_count")
        .reindex(index=BRAND_ORDER, columns=all_classes)
        .fillna(0.0)
        .to_numpy()
    )
    n_brands = len(BRAND_ORDER)
    similarity = np.ones((n_brands, n_brands))
    for i in range(n_brands):
        for j in range(i + 1, n_brands):
            numerator = np.abs(count_matrix[i] - count_matrix[j]).sum()
            denominator = count_matrix[i].sum() + count_matrix[j].sum()
            distance = numerator / denominator if denominator else 0.0
            similarity[i, j] = similarity[j, i] = 1.0 - distance
    similarity_df = pd.DataFrame(similarity, index=BRAND_ORDER, columns=BRAND_ORDER)
    similarity_df.index.name = "brand_display"
    similarity_df.to_csv(DATA / "brand_bray_curtis_similarity.csv")

    brand_images["top_class_by_detections"] = brand_images["brand_display"].map(
        brand_class.groupby("brand_display").first()["class_name"]
    )
    brand_images["top_class_pct_of_brand_detections"] = brand_images["brand_display"].map(
        brand_class.groupby("brand_display").first()["pct_brand_detections"]
    )
    brand_images["top_class_pct_images"] = brand_images["brand_display"].map(
        brand_class.groupby("brand_display").first()["pct_images_with_class"]
    )

    tables = {
        "brand_summary": brand_images,
        "brand_class_counts": brand_class,
        "overall_class_summary": overall_class,
        "major_classes": major_classes,
        "brand_top_classes": top_classes,
        "brand_bray_curtis_similarity": similarity_df.reset_index(),
    }

    brand_images.to_csv(DATA / "brand_summary.csv", index=False)
    brand_class.to_csv(DATA / "brand_class_counts.csv", index=False)
    overall_class.to_csv(DATA / "overall_class_summary.csv", index=False)
    major_classes.to_csv(DATA / "major_classes.csv", index=False)
    top_classes.to_csv(DATA / "brand_top_classes.csv", index=False)

    summary = {
        "total_images": int(len(image_df)),
        "total_detections": int(len(detection_df)),
        "overall_mean_objects_per_image": round(float(image_df["object_count"].mean()), 4),
        "overall_sd_objects_per_image": round(float(image_df["object_count"].std()), 4),
        "overall_median_objects_per_image": float(image_df["object_count"].median()),
        "images_without_detection": int(image_df["zero_detection"].sum()),
        "pct_images_without_detection": round(float(pct(image_df["zero_detection"].sum(), len(image_df))), 4),
        "major_classes_definition": "Top 15 COCO classes by number of dataset images containing the class",
        "mean_offdiagonal_bray_curtis_similarity": round(
            float(similarity[~np.eye(n_brands, dtype=bool)].mean()), 4
        ),
        "min_brand_mean_objects": {
            "brand": brand_images.sort_values("mean_objects_per_image").iloc[0]["brand_display"],
            "value": round(float(brand_images["mean_objects_per_image"].min()), 4),
        },
        "max_brand_mean_objects": {
            "brand": brand_images.sort_values("mean_objects_per_image").iloc[-1]["brand_display"],
            "value": round(float(brand_images["mean_objects_per_image"].max()), 4),
        },
        "max_brand_zero_detection_pct": {
            "brand": brand_images.sort_values("pct_images_without_detection").iloc[-1]["brand_display"],
            "value": round(float(brand_images["pct_images_without_detection"].max()), 4),
        },
    }
    (DATA / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return {
        **tables,
        "image_prop": image_prop,
        "detection_share": detection_share,
        "similarity_matrix": similarity,
        "summary": summary,
        "brand_images": brand_images,
        "brand_class": brand_class,
        "overall_class": overall_class,
        "image_df": image_df,
        "detection_df": detection_df,
    }


def figure_coverage(tables: dict) -> None:
    brand_images = tables["brand_images"]
    y = np.arange(len(brand_images))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.15, 1]})
    axes[0].barh(y, brand_images["images_with_detection"], color=PALETTE["teal"], label="Images with >=1 detection")
    axes[0].barh(
        y,
        brand_images["images_without_detection"],
        left=brand_images["images_with_detection"],
        color=PALETTE["grey"],
        label="Images with no detection",
    )
    for ypos, with_det, without in zip(y, brand_images["images_with_detection"], brand_images["images_without_detection"]):
        axes[0].text(with_det / 2, ypos, f"{int(with_det)}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        if without:
            axes[0].text(with_det + without / 2, ypos, f"{int(without)}", ha="center", va="center", color=PALETTE["dark"], fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(brand_images["brand_display"])
    axes[0].set_xlim(0, 40)
    axes[0].set_xlabel("Images processed (n = 40 per brand)")
    axes[0].set_title("A. Coverage and zero-detection images")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=8,
    )

    axes[1].barh(y, brand_images["total_detections"], color=PALETTE["blue"])
    for ypos, value in zip(y, brand_images["total_detections"]):
        axes[1].text(value + 3, ypos, f"{int(value)}", va="center", fontsize=9)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Total detected objects across 40 images")
    axes[1].set_title("B. Total object detections")
    axes[1].set_xlim(0, brand_images["total_detections"].max() * 1.18)
    fig.suptitle("YOLO11n coverage of the 400-image Instagram dataset", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig01_coverage_and_total_detections.png")


def figure_mean_objects(tables: dict) -> None:
    brand_images = tables["brand_images"]
    y = np.arange(len(brand_images))[::-1]
    overall = tables["summary"]["overall_mean_objects_per_image"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(
        y,
        brand_images["mean_objects_per_image"],
        xerr=brand_images["sd_objects_per_image"],
        color=PALETTE["blue"],
        error_kw={"ecolor": PALETTE["dark"], "capsize": 3, "elinewidth": 1},
    )
    for ypos, mean, sd in zip(y, brand_images["mean_objects_per_image"], brand_images["sd_objects_per_image"]):
        ax.text(mean + sd + 0.12, ypos, f"{mean:.2f} +/- {sd:.2f}", va="center", fontsize=9)
    ax.axvline(overall, color=PALETTE["orange"], linestyle="--", linewidth=1.5, label=f"Overall mean = {overall:.2f}")
    ax.set_yticks(y)
    ax.set_yticklabels(brand_images["brand_display"])
    ax.set_xlabel("Detected objects per image (mean +/- SD; n = 40 images)")
    ax.set_title("Average number of detected objects per image by brand")
    ax.legend(loc="lower right")
    ax.set_xlim(0, brand_images["mean_objects_per_image"].max() + brand_images["sd_objects_per_image"].max() + 0.8)
    fig.tight_layout()
    save_figure(fig, "fig02_mean_objects_per_image.png")


def figure_zero_detection(tables: dict) -> None:
    brand_images = tables["brand_images"]
    y = np.arange(len(brand_images))[::-1]
    overall = tables["summary"]["pct_images_without_detection"]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(y, brand_images["pct_images_without_detection"], color=PALETTE["orange"])
    for ypos, value, count in zip(y, brand_images["pct_images_without_detection"], brand_images["images_without_detection"]):
        ax.text(value + 0.5, ypos, f"{value:.1f}% ({int(count)}/40)", va="center", fontsize=9)
    ax.axvline(overall, color=PALETTE["dark"], linestyle="--", linewidth=1.5, label=f"Overall = {overall:.1f}% (42/400)")
    ax.set_yticks(y)
    ax.set_yticklabels(brand_images["brand_display"])
    ax.set_xlabel("Images with no YOLO detections (%)")
    ax.set_title("Proportion of images with zero detections")
    ax.legend(loc="lower right")
    ax.set_xlim(0, brand_images["pct_images_without_detection"].max() + 7)
    fig.tight_layout()
    save_figure(fig, "fig03_zero_detection_proportions.png")


def figure_overall_classes(tables: dict) -> None:
    overall = tables["overall_class"].head(15).iloc[::-1]
    y = np.arange(len(overall))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].barh(y, overall["detection_count"], color=PALETTE["teal"])
    for ypos, value in zip(y, overall["detection_count"]):
        axes[0].text(value + 3, ypos, f"{int(value)}", va="center", fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(overall["class_name"])
    axes[0].set_xlabel("Number of detections")
    axes[0].set_title("A. Most frequent detected classes")
    axes[0].set_xlim(0, overall["detection_count"].max() * 1.15)

    axes[1].barh(y, overall["pct_images_with_class"], color=PALETTE["gold"])
    for ypos, value in zip(y, overall["pct_images_with_class"]):
        axes[1].text(value + 1, ypos, f"{value:.1f}%", va="center", fontsize=9)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Images containing the class (%)")
    axes[1].set_title("B. Image prevalence")
    axes[1].set_xlim(0, overall["pct_images_with_class"].max() * 1.18)
    fig.suptitle("Overall COCO object classes detected across 400 images", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig04_overall_top_classes.png")


def heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame | np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    cmap: str,
    value_format: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate_threshold: float = 0.0,
) -> None:
    values = np.asarray(matrix, dtype=float)
    vmin = 0.0 if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if value >= annotate_threshold:
                color = "white" if value > (vmax * 0.58) else "#222222"
                ax.text(j, i, value_format.format(value), ha="center", va="center", fontsize=7, color=color)
    colorbar = plt.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(cbar_label, fontsize=9)
    colorbar.ax.tick_params(labelsize=8)


def figure_heatmaps(tables: dict) -> None:
    image_prop = tables["image_prop"]
    fig, ax = plt.subplots(figsize=(13, 6.4))
    heatmap(
        ax,
        image_prop,
        image_prop.index.tolist(),
        image_prop.columns.tolist(),
        "YlGnBu",
        "{:.0f}",
        "% of brand images containing class",
        vmin=0,
        vmax=100,
        annotate_threshold=0.5,
    )
    ax.set_title("Share of images containing each major object class, by brand")
    ax.set_xlabel("Major COCO class (top 15 by overall image prevalence)")
    ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, "fig05_heatmap_class_image_proportion.png")

    detection_share = tables["detection_share"]
    fig, ax = plt.subplots(figsize=(13, 6.4))
    heatmap(
        ax,
        detection_share,
        detection_share.index.tolist(),
        detection_share.columns.tolist(),
        "OrRd",
        "{:.0f}",
        "% of brand detections",
        vmin=0,
        vmax=100,
        annotate_threshold=1.0,
    )
    ax.set_title("Composition of detected objects within each brand")
    ax.set_xlabel("Major COCO class (top 15 by overall image prevalence)")
    ax.set_ylabel("")
    fig.tight_layout()
    save_figure(fig, "fig06_heatmap_detection_composition.png")


def figure_top_classes_by_brand(tables: dict) -> None:
    top = tables["brand_top_classes"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 9.2), sharex=False)
    axes = axes.ravel()
    for ax, brand in zip(axes, BRAND_ORDER):
        subset = top[top["brand_display"] == brand].sort_values("detection_count")
        colors = plt.cm.Blues(np.linspace(0.38, 0.82, len(subset)))
        y = np.arange(len(subset))
        ax.barh(y, subset["detection_count"], color=colors)
        for ypos, (_, row) in zip(y, subset.iterrows()):
            ax.text(
                row["detection_count"] + max(subset["detection_count"]) * 0.03,
                ypos,
                f"{int(row['detection_count'])} det. ({row['pct_images_with_class']:.0f}% img.)",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(subset["class_name"], fontsize=9)
        ax.set_title(brand, fontsize=11)
        ax.set_xlabel("Number of detections")
        ax.set_xlim(0, max(subset["detection_count"]) * 1.55)
    fig.suptitle("Top five detected object classes within each brand", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "fig07_top_classes_by_brand.png")


def figure_similarity(tables: dict) -> None:
    similarity = tables["similarity_matrix"]
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    heatmap(
        ax,
        similarity,
        BRAND_ORDER,
        BRAND_ORDER,
        "YlGn",
        "{:.2f}",
        "Bray-Curtis similarity (1 = identical composition)",
        vmin=0,
        vmax=1,
        annotate_threshold=0,
    )
    ax.set_title("Similarity of detected-object composition between brands")
    fig.tight_layout()
    save_figure(fig, "fig08_brand_composition_similarity.png")


def figure_confidences(tables: dict) -> None:
    detection_df = tables["detection_df"]
    groups = [detection_df.loc[detection_df["brand_display"] == brand, "confidence"].to_numpy() for brand in BRAND_ORDER]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    box = ax.boxplot(
        groups,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["dark"], "linewidth": 1.5},
        whiskerprops={"color": PALETTE["dark"]},
        capprops={"color": PALETTE["dark"]},
    )
    for patch, color in zip(box["boxes"], plt.cm.tab10(np.linspace(0, 1, len(BRAND_ORDER)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    rng = np.random.default_rng(42)
    for position, values in enumerate(groups, start=1):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(values, position + jitter, s=8, color=PALETTE["dark"], alpha=0.22, linewidths=0)
    medians = [np.median(values) for values in groups]
    for position, median in enumerate(medians, start=1):
        ax.text(1.01, position, f"{median:.2f}", va="center", fontsize=8, color=PALETTE["dark"])
    ax.axvline(0.25, color=PALETTE["red"], linestyle="--", linewidth=1.2, label="Confidence threshold = 0.25")
    ax.set_yticks(np.arange(1, len(BRAND_ORDER) + 1))
    ax.set_yticklabels(BRAND_ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 1.05)
    ax.set_xlabel("Detection confidence")
    ax.set_title("Distribution of YOLO confidence scores by brand")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig09_confidence_distribution.png")


def figure_object_count_distribution(tables: dict) -> None:
    image_df = tables["image_df"]
    groups = [image_df.loc[image_df["brand_display"] == brand, "object_count"].to_numpy() for brand in BRAND_ORDER]
    fig, ax = plt.subplots(figsize=(12, 5.8))
    box = ax.boxplot(
        groups,
        orientation="horizontal",
        patch_artist=True,
        showfliers=False,
        medianprops={"color": PALETTE["dark"], "linewidth": 1.5},
        whiskerprops={"color": PALETTE["dark"]},
        capprops={"color": PALETTE["dark"]},
    )
    for patch, color in zip(box["boxes"], plt.cm.tab10(np.linspace(0, 1, len(BRAND_ORDER)))):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    rng = np.random.default_rng(42)
    for position, values in enumerate(groups, start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        ax.scatter(values + jitter * 0.0, position + jitter, s=13, color=PALETTE["dark"], alpha=0.28, linewidths=0)
    means = [values.mean() for values in groups]
    for position, mean in enumerate(means, start=1):
        ax.plot(mean, position, marker="D", markersize=5, color=PALETTE["orange"], zorder=3)
    ax.plot([], [], marker="D", markersize=5, color=PALETTE["orange"], linestyle="None", label="Mean")
    ax.set_yticks(np.arange(1, len(BRAND_ORDER) + 1))
    ax.set_yticklabels(BRAND_ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Detected objects per image")
    ax.set_title("Distribution of object counts per image by brand")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "fig10_object_count_distribution.png")


def select_sample_images(image_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand in BRAND_ORDER:
        subset = image_df[image_df["brand_display"] == brand].copy()
        median_value = subset["object_count"].median()
        subset["median_distance"] = (subset["object_count"] - median_value).abs()
        median_row = subset.sort_values(["median_distance", "filename"]).iloc[0].copy()
        median_row["selection"] = "median_count"
        rows.append(median_row)
        max_row = subset.sort_values(["object_count", "filename"], ascending=[False, True]).iloc[0].copy()
        max_row["selection"] = "max_count"
        if max_row["filename"] != median_row["filename"]:
            rows.append(max_row)
        else:
            alt = subset.sort_values(["object_count", "filename"], ascending=[False, True]).iloc[1].copy()
            alt["selection"] = "second_highest_count"
            rows.append(alt)
    return pd.DataFrame(rows).drop_duplicates(subset=["image_id"])


def render_samples(tables: dict) -> None:
    image_df = tables["image_df"]
    detection_df = tables["detection_df"]
    selected = select_sample_images(image_df)
    output_dir = SAMPLES / "annotated"
    output_dir.mkdir(parents=True, exist_ok=True)
    font_path = font_manager.findfont("DejaVu Sans")
    label_font = ImageFont.truetype(font_path, 18)
    small_font = ImageFont.truetype(font_path, 14)
    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    color_map = {}
    rendered = []
    for _, row in selected.iterrows():
        brand_slug = row["brand"]
        source = IMAGE_ROOT / brand_slug / row["filename"]
        image = Image.open(source).convert("RGB")
        detections = detection_df[detection_df["image_id"] == row["image_id"]]
        draw = ImageDraw.Draw(image)
        line_width = max(2, int(round(min(image.size) / 300)))
        for _, detection in detections.iterrows():
            class_name = detection["class_name"]
            if class_name not in color_map:
                color_map[class_name] = tuple(int(value * 255) for value in palette[len(color_map) % len(palette)][:3])
            color = color_map[class_name]
            box = [detection["x1"], detection["y1"], detection["x2"], detection["y2"]]
            draw.rectangle(box, outline=color, width=line_width)
            label = f"{class_name} {detection['confidence']:.2f}"
            text_box = draw.textbbox((0, 0), label, font=small_font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            x, y = detection["x1"], max(0, detection["y1"] - text_height - 4)
            draw.rectangle([x, y, x + text_width + 6, y + text_height + 4], fill=color)
            draw.text((x + 3, y + 1), label, fill="black", font=small_font)
        brand_dir = output_dir / brand_slug
        brand_dir.mkdir(parents=True, exist_ok=True)
        output_path = brand_dir / row["filename"]
        image.save(output_path, quality=92)
        rendered.append(
            {
                "brand": row["brand_display"],
                "filename": row["filename"],
                "image_id": row["image_id"],
                "object_count": int(row["object_count"]),
                "selection": row["selection"],
                "annotated_path": str(output_path),
            }
        )

    tile_size = 360
    label_height = 52
    columns = 4
    rows_count = int(np.ceil(len(rendered) / columns))
    sheet = Image.new("RGB", (columns * tile_size, rows_count * (tile_size + label_height)), "white")
    sheet_draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(rendered):
        column = index % columns
        row_number = index // columns
        x0 = column * tile_size
        y0 = row_number * (tile_size + label_height)
        image = Image.open(item["annotated_path"]).convert("RGB")
        thumbnail = ImageOps.pad(image, (tile_size, tile_size), color="white", method=Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, (x0, y0))
        sheet_draw.text((x0 + 8, y0 + tile_size + 4), item["brand"], fill="#111111", font=label_font)
        sheet_draw.text(
            (x0 + 8, y0 + tile_size + 27),
            f"{item['filename']} | {item['object_count']} objects",
            fill="#444444",
            font=small_font,
        )
    sheet.save(FIGURES / "fig11_annotated_detection_samples.png", quality=95)
    pd.DataFrame(rendered).to_csv(DATA / "sample_images_rendered.csv", index=False)


def main() -> None:
    setup_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    image_df, detection_df = load_inputs()
    tables = build_tables(image_df, detection_df)
    figure_coverage(tables)
    figure_mean_objects(tables)
    figure_zero_detection(tables)
    figure_overall_classes(tables)
    figure_heatmaps(tables)
    figure_top_classes_by_brand(tables)
    figure_similarity(tables)
    figure_confidences(tables)
    figure_object_count_distribution(tables)
    render_samples(tables)
    print(json.dumps(tables["summary"], indent=2))


if __name__ == "__main__":
    main()
