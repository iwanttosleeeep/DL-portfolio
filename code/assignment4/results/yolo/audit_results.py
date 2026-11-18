#!/usr/bin/env python3
"""Independently audit the raw YOLO outputs, derived statistics, and figures."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ultralytics
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
AUDIT = ROOT / "audit"
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
BRAND_SLUG = {
    "Bang & Olufsen": "bang-olufsen",
    "Fortnum & Mason": "fortnum-mason",
    "Herman Miller": "herman-miller",
    "Jellycat": "jellycat",
    "Jo Malone London": "jo-malone-london",
    "Kodak": "kodak",
    "LEGO": "lego",
    "Oura": "oura",
    "Sennheiser": "sennheiser",
    "Stanley": "stanley",
}
TOL = 1e-6
checks: list[dict] = []


def add_check(name: str, passed: bool, details: str, category: str = "general") -> None:
    checks.append(
        {
            "category": category,
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "details": details,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_series(left: pd.Series, right: pd.Series, name: str, category: str, tolerance: float = TOL) -> None:
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    if len(left) != len(right):
        add_check(name, False, f"row count mismatch: {len(left)} vs {len(right)}", category)
        return
    numeric = pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right)
    if numeric:
        difference = (left.astype(float) - right.astype(float)).abs()
        maximum = float(difference.max()) if len(difference) else 0.0
        add_check(name, maximum <= tolerance, f"max absolute difference = {maximum:.12g}", category)
    else:
        mismatches = int((left.astype(str) != right.astype(str)).sum())
        add_check(name, mismatches == 0, f"mismatched rows = {mismatches}", category)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT / "model_metadata.json").read_text(encoding="utf-8"))
    image_df = pd.read_csv(DATA / "image_level_detections.csv", keep_default_na=False)
    detection_df = pd.read_csv(DATA / "object_detections.csv", keep_default_na=False)
    brand_summary = pd.read_csv(DATA / "brand_summary.csv")
    brand_class = pd.read_csv(DATA / "brand_class_counts.csv")
    overall_class = pd.read_csv(DATA / "overall_class_summary.csv")
    major_classes = pd.read_csv(DATA / "major_classes.csv")
    top_classes = pd.read_csv(DATA / "brand_top_classes.csv")
    image_prop = pd.read_csv(DATA / "brand_class_image_proportions_wide.csv")
    detection_share = pd.read_csv(DATA / "brand_class_detection_shares_wide.csv")
    similarity = pd.read_csv(DATA / "brand_bray_curtis_similarity.csv", index_col=0)
    analysis_summary = json.loads((DATA / "analysis_summary.json").read_text(encoding="utf-8"))

    add_check("Metadata file exists", (ROOT / "model_metadata.json").exists(), str(ROOT / "model_metadata.json"), "metadata")
    add_check("Weights file exists", Path(metadata["weights_file"]).exists(), metadata["weights_file"], "model")
    actual_weights_hash = sha256(Path(metadata["weights_file"]))
    add_check("Weights SHA-256 matches metadata", actual_weights_hash == metadata["weights_sha256"], actual_weights_hash, "model")
    add_check("Weight file size matches metadata", Path(metadata["weights_file"]).stat().st_size == metadata["weights_size_bytes"], f"{Path(metadata['weights_file']).stat().st_size} bytes", "model")
    add_check("Ultralytics version matches runtime", metadata["ultralytics_version"] == ultralytics.__version__, f"metadata={metadata['ultralytics_version']} runtime={ultralytics.__version__}", "model")
    add_check("Torch version matches runtime", metadata["torch_version"] == torch.__version__, f"metadata={metadata['torch_version']} runtime={torch.__version__}", "model")
    add_check("MPS was available at inference", bool(metadata["mps_available"]) is True, f"mps_available={metadata['mps_available']}", "model")
    add_check("Inference device recorded as mps", metadata["device"] == "mps", f"device={metadata['device']}", "model")
    add_check("COCO class count is 80", metadata["class_count"] == 80, f"class_count={metadata['class_count']}", "model")
    add_check("Task is object detection", metadata["task"] == "detect", f"task={metadata['task']}", "model")

    disk_brands = sorted([path.name for path in IMAGE_ROOT.iterdir() if path.is_dir() and not path.name.startswith(".")])
    add_check("Exactly 10 dataset brand directories", len(disk_brands) == 10, f"found {len(disk_brands)}", "dataset")
    r_image_counts = image_df.groupby("brand").size().to_dict()
    r_disk_counts = {brand: len(list((IMAGE_ROOT / brand).glob("*.jpg"))) for brand in disk_brands}
    add_check("Dataset has 400 JPEG images on disk", sum(r_disk_counts.values()) == 400, f"found {sum(r_disk_counts.values())}", "dataset")
    add_check("Every dataset brand has 40 images on disk", all(value == 40 for value in r_disk_counts.values()), json.dumps(r_disk_counts, sort_keys=True), "dataset")
    add_check("Image-level table has 400 rows", len(image_df) == 400, f"found {len(image_df)}", "coverage")
    add_check("Every brand has 40 image-level rows", all(r_image_counts.get(brand) == 40 for brand in disk_brands), json.dumps(r_image_counts, sort_keys=True), "coverage")
    add_check("Image IDs are unique", not image_df["image_id"].duplicated().any(), f"duplicates={int(image_df['image_id'].duplicated().sum())}", "coverage")
    add_check("All image-level brands are represented in metadata order", set(image_df["brand"]) == set(disk_brands), f"brands={sorted(image_df['brand'].unique())}", "coverage")
    disk_files = {(brand, path.name) for brand in disk_brands for path in (IMAGE_ROOT / brand).glob("*.jpg")}
    result_files = set(zip(image_df["brand"], image_df["filename"], strict=True))
    add_check("Image-level results exactly match files on disk", disk_files == result_files, f"disk={len(disk_files)} results={len(result_files)}", "coverage")
    add_check("Zero-detection images are present in image-level results", int(image_df["zero_detection"].sum()) == 42, f"zero_detection_rows={int(image_df['zero_detection'].sum())}", "coverage")
    add_check("Zero-detection rows have object_count 0", (image_df.loc[image_df["zero_detection"] == 1, "object_count"] == 0).all(), f"checked={int(image_df['zero_detection'].sum())}", "coverage")

    dimensions_ok = True
    dimension_details = []
    for row in image_df.itertuples(index=False):
        with Image.open(IMAGE_ROOT / row.brand / row.filename) as image:
            if image.size != (int(row.image_width), int(row.image_height)):
                dimensions_ok = False
                dimension_details.append(f"{row.image_id}: {(image.size)} vs {(row.image_width, row.image_height)}")
    add_check("Recorded image dimensions match source files", dimensions_ok, "; ".join(dimension_details[:5]) if dimension_details else "all 400 images match", "coverage")

    detection_groups = {key: group for key, group in detection_df.groupby(["brand", "filename"], sort=False)}
    sum_ok = int(detection_df.shape[0]) == int(image_df["object_count"].sum())
    add_check("Detection table rows equal summed object counts", sum_ok, f"detections={len(detection_df)} summed={int(image_df['object_count'].sum())}", "raw")
    all_counts_ok = True
    all_classes_ok = True
    all_summary_ok = True
    all_json_ok = True
    all_top_ok = True
    all_index_ok = True
    all_conf_stats_ok = True
    detail_errors = []
    for row in image_df.itertuples(index=False):
        group = detection_groups.get((row.brand, row.filename), detection_df.iloc[0:0])
        if len(group) != int(row.object_count):
            all_counts_ok = False
            detail_errors.append(f"{row.image_id}: table={len(group)} image={row.object_count}")
        class_counter = Counter(group["class_name"].tolist())
        unique_expected = sorted(class_counter, key=lambda name: (-class_counter[name], name))
        if ";".join(unique_expected) != row.detected_classes:
            all_classes_ok = False
            detail_errors.append(f"{row.image_id}: detected_classes mismatch")
        expected_summary = "; ".join(f"{name} ({class_counter[name]})" for name in unique_expected)
        if expected_summary != row.detection_summary:
            all_summary_ok = False
            detail_errors.append(f"{row.image_id}: detection_summary mismatch")
        parsed_counts = json.loads(row.class_counts_json)
        if dict(sorted(parsed_counts.items())) != dict(sorted(class_counter.items())):
            all_json_ok = False
            detail_errors.append(f"{row.image_id}: class_counts_json mismatch")
        expected_top = unique_expected[0] if unique_expected else ""
        if expected_top != row.top_class:
            all_top_ok = False
            detail_errors.append(f"{row.image_id}: top_class mismatch")
        if len(group):
            indices = sorted(group["detection_index"].astype(int).tolist())
            if indices != list(range(1, len(group) + 1)):
                all_index_ok = False
                detail_errors.append(f"{row.image_id}: detection_index not contiguous")
            if int(group["is_top1"].sum()) != 1 or int(group.loc[group["detection_index"].astype(int) == 1, "is_top1"].iloc[0]) != 1:
                all_index_ok = False
                detail_errors.append(f"{row.image_id}: is_top1 invalid")
            max_confidence = round(float(group["confidence"].max()), 6)
            mean_confidence = round(float(group["confidence"].mean()), 6)
            if abs(max_confidence - float(row.max_confidence)) > TOL or abs(mean_confidence - float(row.mean_confidence)) > TOL:
                all_conf_stats_ok = False
                detail_errors.append(f"{row.image_id}: confidence stats mismatch")
    add_check("Per-image detection counts match object_count", all_counts_ok, "; ".join(detail_errors[:5]) if detail_errors else "all 400 images", "raw")
    add_check("Per-image detected_classes match detection table", all_classes_ok, "; ".join(detail_errors[:5]) if detail_errors else "all 400 images", "raw")
    add_check("Per-image detection_summary matches detection table", all_summary_ok, "; ".join(detail_errors[:5]) if detail_errors else "all 400 images", "raw")
    add_check("Per-image class_counts_json matches detection table", all_json_ok, "; ".join(detail_errors[:5]) if detail_errors else "all 400 images", "raw")
    add_check("Per-image top_class matches detection table", all_top_ok, "; ".join(detail_errors[:5]) if detail_errors else "all 400 images", "raw")
    add_check("Detection indices and is_top1 flags are valid", all_index_ok, "; ".join(detail_errors[:5]) if detail_errors else "all 400 images", "raw")
    add_check("Per-image confidence statistics match detection table", all_conf_stats_ok, "; ".join(detail_errors[:5]) if detail_errors else "all 400 images", "raw")

    boxes_valid = (
        (detection_df["x2"] > detection_df["x1"])
        & (detection_df["y2"] > detection_df["y1"])
        & (detection_df["confidence"] >= 0.25)
        & (detection_df["confidence"] <= 1.0)
    ).all()
    add_check("Bounding boxes and confidence values are valid", bool(boxes_valid), f"detections={len(detection_df)}", "raw")
    bounds = detection_df.merge(
        image_df[["image_id", "image_width", "image_height"]], on="image_id", how="left", validate="many_to_one"
    )
    inside = (
        (bounds["x1"] >= -1e-6)
        & (bounds["y1"] >= -1e-6)
        & (bounds["x2"] <= bounds["image_width"] + 1e-6)
        & (bounds["y2"] <= bounds["image_height"] + 1e-6)
    )
    add_check("All boxes fall inside their source image", bool(inside.all()), f"outside={int((~inside).sum())}", "raw")
    add_check("No duplicate derived detection rows", not detection_df.duplicated(["image_id", "detection_index"]).any(), f"duplicates={int(detection_df.duplicated(['image_id', 'detection_index']).sum())}", "raw")

    recomputed_brand = (
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
    recomputed_brand["pct_images_without_detection"] = 100.0 * recomputed_brand["images_without_detection"] / recomputed_brand["images"]
    recomputed_brand["mean_objects_per_detected_image"] = recomputed_brand["total_detections"] / recomputed_brand["images_with_detection"]
    confidence_by_brand = detection_df.groupby("brand_display").agg(
        mean_confidence=("confidence", "mean"),
        median_confidence=("confidence", "median"),
        max_confidence=("confidence", "max"),
    ).reindex(BRAND_ORDER).reset_index()
    recomputed_brand = recomputed_brand.merge(confidence_by_brand, on="brand_display", how="left")
    published = brand_summary.set_index("brand_display").reindex(BRAND_ORDER).reset_index()
    for column in [
        "images",
        "images_with_detection",
        "images_without_detection",
        "total_detections",
        "mean_objects_per_image",
        "sd_objects_per_image",
        "median_objects_per_image",
        "min_objects_per_image",
        "max_objects_per_image",
        "pct_images_without_detection",
        "mean_objects_per_detected_image",
        "mean_confidence",
        "median_confidence",
        "max_confidence",
    ]:
        compare_series(published[column], recomputed_brand[column], f"brand_summary.{column} reproducible", "statistics", 1e-6)
    denominator_ok = np.allclose(
        published["pct_images_without_detection"],
        published["images_without_detection"] / published["images"] * 100,
        atol=1e-9,
    )
    add_check("Brand percentages use 40-image denominators", bool(denominator_ok), "pct = images_without / images * 100", "statistics")
    add_check("All brands have 40 images in summary", (published["images"] == 40).all(), f"images={published['images'].tolist()}", "statistics")
    add_check("Detection total equals 1178", int(published["total_detections"].sum()) == 1178, f"total={int(published['total_detections'].sum())}", "statistics")
    add_check("Zero-detection total equals 42", int(published["images_without_detection"].sum()) == 42, f"total={int(published['images_without_detection'].sum())}", "statistics")

    recomputed_brand_class = (
        detection_df.groupby(["brand_display", "class_name"], as_index=False)
        .agg(detection_count=("class_name", "size"))
    )
    image_class_counts = detection_df.groupby(["brand_display", "filename", "class_name"], as_index=False).size()
    images_with_class = image_class_counts.groupby(["brand_display", "class_name"], as_index=False).size().rename(columns={"size": "images_with_class"})
    recomputed_brand_class = recomputed_brand_class.merge(images_with_class, on=["brand_display", "class_name"], how="left")
    recomputed_brand_class["pct_brand_detections"] = 100.0 * recomputed_brand_class["detection_count"] / recomputed_brand_class["brand_display"].map(published.set_index("brand_display")["total_detections"])
    recomputed_brand_class["pct_images_with_class"] = 100.0 * recomputed_brand_class["images_with_class"] / 40.0
    merged_class = brand_class.merge(
        recomputed_brand_class,
        on=["brand_display", "class_name"],
        how="outer",
        suffixes=("_published", "_recomputed"),
        validate="one_to_one",
        indicator=True,
    )
    add_check("Brand-class rows match recomputed set", (merged_class["_merge"] == "both").all(), f"merge_counts={merged_class['_merge'].value_counts().to_dict()}", "statistics")
    class_count_ok = np.allclose(
        merged_class["detection_count_published"].fillna(-1),
        merged_class["detection_count_recomputed"].fillna(-2),
        atol=1e-9,
    )
    add_check("Brand-class detection counts reproducible", bool(class_count_ok), "compared published and recomputed counts", "statistics")
    class_images_ok = np.allclose(
        merged_class["images_with_class_published"].fillna(-1),
        merged_class["images_with_class_recomputed"].fillna(-2),
        atol=1e-9,
    )
    add_check("Brand-class image counts reproducible", bool(class_images_ok), "compared published and recomputed image counts", "statistics")
    class_pct_ok = np.allclose(
        merged_class["pct_images_with_class_published"].fillna(-1),
        merged_class["pct_images_with_class_recomputed"].fillna(-2),
        atol=1e-9,
    )
    add_check("Brand-class image percentages use 40-image denominators", bool(class_pct_ok), "pct = images_with_class / 40 * 100", "statistics")

    recomputed_overall = (
        detection_df.groupby("class_name", as_index=False)
        .agg(
            detection_count=("class_name", "size"),
            images_with_class=("image_id", "nunique"),
            brands_present=("brand_display", "nunique"),
            mean_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
            mean_area_fraction=("area_fraction", "mean"),
        )
    )
    recomputed_overall["pct_all_detections"] = 100.0 * recomputed_overall["detection_count"] / len(detection_df)
    recomputed_overall["pct_images_with_class"] = 100.0 * recomputed_overall["images_with_class"] / len(image_df)
    merged_overall = overall_class.merge(
        recomputed_overall,
        on="class_name",
        how="outer",
        suffixes=("_published", "_recomputed"),
        validate="one_to_one",
        indicator=True,
    )
    add_check("Overall class rows match recomputed set", (merged_overall["_merge"] == "both").all(), f"merge_counts={merged_overall['_merge'].value_counts().to_dict()}", "statistics")
    for column in ["detection_count", "images_with_class", "brands_present", "mean_confidence", "median_confidence", "mean_area_fraction", "pct_all_detections", "pct_images_with_class"]:
        left = merged_overall[f"{column}_published"].fillna(-1)
        right = merged_overall[f"{column}_recomputed"].fillna(-2)
        maximum = float((left.astype(float) - right.astype(float)).abs().max())
        add_check(f"overall_class_summary.{column} reproducible", maximum <= 1e-9, f"max absolute difference = {maximum:.12g}", "statistics")
    overall_denominator_ok = np.allclose(
        overall_class["pct_images_with_class"],
        overall_class["images_with_class"] / 400.0 * 100,
        atol=1e-9,
    )
    add_check("Overall image percentages use 400-image denominators", bool(overall_denominator_ok), "pct = images_with_class / 400 * 100", "statistics")
    add_check("Overall detection count equals 1178", int(overall_class["detection_count"].sum()) == 1178, f"total={int(overall_class['detection_count'].sum())}", "statistics")
    add_check("Overall class detection counts sum to detection table", int(overall_class["detection_count"].sum()) == len(detection_df), f"{int(overall_class['detection_count'].sum())} vs {len(detection_df)}", "statistics")

    expected_major = (
        overall_class.sort_values(["images_with_class", "detection_count"], ascending=False)
        .head(15)["class_name"]
        .tolist()
    )
    add_check("Major classes are top 15 by image prevalence", major_classes["class_name"].tolist() == expected_major, f"{major_classes['class_name'].tolist()}", "statistics")

    prop_expected = brand_class.pivot(index="brand_display", columns="class_name", values="pct_images_with_class")
    prop_expected = prop_expected.reindex(index=BRAND_ORDER, columns=major_classes["class_name"]).fillna(0.0).reset_index()
    prop_expected.columns.name = None
    prop_published = image_prop.set_index("brand_display").reindex(BRAND_ORDER).reset_index()
    prop_expected = prop_expected.set_index("brand_display").reindex(BRAND_ORDER).reset_index()
    add_check("Image-proportion wide table brands match", prop_published["brand_display"].tolist() == prop_expected["brand_display"].tolist(), f"brands={prop_published['brand_display'].tolist()}", "statistics")
    max_prop_diff = float((prop_published.drop(columns=["brand_display"]).to_numpy() - prop_expected.drop(columns=["brand_display"]).to_numpy()).__abs__().max())
    add_check("Image-proportion wide table reproducible", max_prop_diff <= 1e-9, f"max absolute difference = {max_prop_diff:.12g}", "statistics")

    share_expected = brand_class.pivot(index="brand_display", columns="class_name", values="pct_brand_detections")
    share_expected = share_expected.reindex(index=BRAND_ORDER, columns=major_classes["class_name"]).fillna(0.0).reset_index()
    share_published = detection_share.set_index("brand_display").reindex(BRAND_ORDER).reset_index()
    share_expected.columns.name = None
    share_expected = share_expected.set_index("brand_display").reindex(BRAND_ORDER).reset_index()
    max_share_diff = float((share_published.drop(columns=["brand_display"]).to_numpy() - share_expected.drop(columns=["brand_display"]).to_numpy()).__abs__().max())
    add_check("Detection-share wide table reproducible", max_share_diff <= 1e-9, f"max absolute difference = {max_share_diff:.12g}", "statistics")

    matches = detection_df.groupby(["brand_display", "class_name"]).size().rename("n").reset_index()
    matrix = matches.pivot(index="brand_display", columns="class_name", values="n").reindex(index=BRAND_ORDER).fillna(0.0).to_numpy()
    recomputed_similarity = np.ones((len(BRAND_ORDER), len(BRAND_ORDER)))
    for i in range(len(BRAND_ORDER)):
        for j in range(i + 1, len(BRAND_ORDER)):
            distance = np.abs(matrix[i] - matrix[j]).sum() / (matrix[i].sum() + matrix[j].sum())
            recomputed_similarity[i, j] = recomputed_similarity[j, i] = 1.0 - distance
    published_similarity = similarity.reindex(index=BRAND_ORDER, columns=BRAND_ORDER).to_numpy()
    max_similarity_diff = float(np.abs(published_similarity - recomputed_similarity).max())
    add_check("Bray-Curtis similarity matrix reproducible", max_similarity_diff <= 1e-9, f"max absolute difference = {max_similarity_diff:.12g}", "statistics")
    add_check("Bray-Curtis matrix is symmetric", bool(np.allclose(published_similarity, published_similarity.T, atol=1e-12)), "checked transpose", "statistics")
    add_check("Bray-Curtis diagonal equals 1", bool(np.allclose(np.diag(published_similarity), 1.0, atol=1e-12)), f"diagonal={np.diag(published_similarity).tolist()}", "statistics")
    add_check("Bray-Curtis values lie in [0, 1]", bool((published_similarity >= 0).all() and (published_similarity <= 1).all()), f"min={published_similarity.min():.6f} max={published_similarity.max():.6f}", "statistics")

    top_expected = brand_class[brand_class["class_rank_within_brand"] <= 5].sort_values(["brand_display", "class_rank_within_brand"])
    top_published = top_classes.sort_values(["brand_display", "class_rank_within_brand"])
    add_check("Top-five table has 50 rows", len(top_published) == 50, f"found {len(top_published)}", "statistics")
    add_check("Top-five table classes match recomputed ranks", top_published["class_name"].tolist() == top_expected["class_name"].tolist(), f"first mismatch comparison rows={len(top_expected)}", "statistics")
    add_check("Analysis summary total images matches", analysis_summary["total_images"] == 400, f"total_images={analysis_summary['total_images']}", "statistics")
    add_check("Analysis summary total detections matches", analysis_summary["total_detections"] == len(detection_df), f"total_detections={analysis_summary['total_detections']}", "statistics")
    add_check("Analysis summary zero-detection count matches", analysis_summary["images_without_detection"] == int(image_df["zero_detection"].sum()), f"zero={analysis_summary['images_without_detection']}", "statistics")

    expected_figures = [
        "fig01_coverage_and_total_detections.png",
        "fig02_mean_objects_per_image.png",
        "fig03_zero_detection_proportions.png",
        "fig04_overall_top_classes.png",
        "fig05_heatmap_class_image_proportion.png",
        "fig06_heatmap_detection_composition.png",
        "fig07_top_classes_by_brand.png",
        "fig08_brand_composition_similarity.png",
        "fig09_confidence_distribution.png",
        "fig10_object_count_distribution.png",
        "fig11_annotated_detection_samples.png",
    ]
    figure_rows = []
    figures_ok = True
    for filename in expected_figures:
        path = FIGURES / filename
        exists = path.exists() and path.stat().st_size > 1000
        size = path.stat().st_size if path.exists() else 0
        if exists:
            with Image.open(path) as image:
                width, height = image.size
        else:
            width, height = 0, 0
        figures_ok = figures_ok and exists and width > 500 and height > 300
        figure_rows.append({"figure": filename, "exists": exists, "size_bytes": size, "width": width, "height": height})
    add_check("All expected figures exist and are non-trivial", figures_ok, f"checked {len(expected_figures)} figures", "figures")
    pd.DataFrame(figure_rows).to_csv(AUDIT / "audit_figures.csv", index=False)

    reinference_rows = []
    reinference_ok = True
    model = YOLO(metadata["weights_file"])
    add_check("Model reloads 80 COCO labels", len(model.names) == 80 and model.task == "detect", f"classes={len(model.names)} task={model.task}", "model")
    model_names = {int(key): value for key, value in model.names.items()}
    add_check("Model labels match metadata labels", model_names == {int(key): value for key, value in metadata["class_labels"].items()}, "compared 80 labels", "model")
    for brand in BRAND_ORDER:
        filename = f"{BRAND_SLUG[brand]}_01.jpg"
        path = IMAGE_ROOT / BRAND_SLUG[brand] / filename
        image_id = f"{BRAND_SLUG[brand]}/{filename}"
        original = detection_df[detection_df["image_id"] == image_id].reset_index(drop=True)
        result = model.predict(
            str(path),
            imgsz=640,
            conf=0.25,
            iou=0.45,
            max_det=300,
            device="mps",
            rect=False,
            verbose=False,
        )[0]
        classes = [int(value) for value in result.boxes.cls.detach().cpu().tolist()]
        confidences = [float(value) for value in result.boxes.conf.detach().cpu().tolist()]
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        fresh = pd.DataFrame(
            {
                "class_id": classes,
                "class_name": [result.names[class_id] for class_id in classes],
                "confidence": confidences,
                "x1": boxes[:, 0] if len(boxes) else [],
                "y1": boxes[:, 1] if len(boxes) else [],
                "x2": boxes[:, 2] if len(boxes) else [],
                "y2": boxes[:, 3] if len(boxes) else [],
            }
        ).reset_index(drop=True)
        count_match = len(original) == len(fresh)
        class_match = Counter(original["class_id"]) == Counter(fresh["class_id"])
        matched_original = []
        matched_fresh = []
        if count_match and class_match:
            for class_id in sorted(set(original["class_id"].tolist())):
                original_indices = original.index[original["class_id"].astype(int) == class_id].tolist()
                fresh_indices = fresh.index[fresh["class_id"].astype(int) == class_id].tolist()
                while original_indices and fresh_indices:
                    best = None
                    for oi in original_indices:
                        for fi in fresh_indices:
                            o = original.loc[oi, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
                            f = fresh.loc[fi, ["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
                            x1 = max(o[0], f[0])
                            y1 = max(o[1], f[1])
                            x2 = min(o[2], f[2])
                            y2 = min(o[3], f[3])
                            intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                            area_o = max(0.0, o[2] - o[0]) * max(0.0, o[3] - o[1])
                            area_f = max(0.0, f[2] - f[0]) * max(0.0, f[3] - f[1])
                            union = area_o + area_f - intersection
                            iou = intersection / union if union else 0.0
                            if best is None or iou > best[0]:
                                best = (iou, oi, fi)
                    matched_original.append(best[1])
                    matched_fresh.append(best[2])
                    original_indices.remove(best[1])
                    fresh_indices.remove(best[2])
        if matched_original:
            original_matched = original.loc[matched_original].reset_index(drop=True)
            fresh_matched = fresh.loc[matched_fresh].reset_index(drop=True)
            confidence_diff = float(
                (original_matched["confidence"].to_numpy() - fresh_matched["confidence"].to_numpy()).__abs__().max()
            )
            box_diff = float(
                np.abs(
                    original_matched[["x1", "y1", "x2", "y2"]].to_numpy()
                    - fresh_matched[["x1", "y1", "x2", "y2"]].to_numpy()
                ).max()
            )
        elif count_match and class_match:
            confidence_diff = np.nan
            box_diff = np.nan
        else:
            confidence_diff = np.nan
            box_diff = np.nan
        row_ok = count_match and class_match and len(matched_original) == len(original) and (confidence_diff <= 1e-5 if len(matched_original) else True) and (box_diff <= 1e-3 if len(matched_original) else True)
        reinference_ok = reinference_ok and row_ok
        reinference_rows.append(
            {
                "brand": brand,
                "filename": filename,
                "original_count": int(len(original)),
                "reinferred_count": int(len(fresh)),
                "count_match": bool(count_match),
                "class_multiset_match": bool(class_match),
                "max_confidence_abs_diff": confidence_diff,
                "max_box_coordinate_abs_diff": box_diff,
                "status": "PASS" if row_ok else "FAIL",
            }
        )
    reinference_df = pd.DataFrame(reinference_rows)
    reinference_df.to_csv(AUDIT / "audit_reinference_check.csv", index=False)
    add_check("Independent re-inference reproduces one image per brand", reinference_ok, f"checked={len(reinference_df)} images", "reproducibility")

    image_manifest_rows = []
    for brand in disk_brands:
        for path in sorted((IMAGE_ROOT / brand).glob("*.jpg")):
            image_manifest_rows.append(
                {
                    "brand": brand,
                    "filename": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    image_manifest = pd.DataFrame(image_manifest_rows)
    image_manifest.to_csv(AUDIT / "image_manifest_sha256.csv", index=False)
    add_check("Image manifest contains 400 unique hashes", len(image_manifest) == 400 and image_manifest["sha256"].nunique() == 400, f"rows={len(image_manifest)} unique_hashes={image_manifest['sha256'].nunique()}", "dataset")

    visual_path = AUDIT / "audit_visual_inspection.csv"
    if visual_path.exists():
        visual = pd.read_csv(visual_path)
        visual_counts = detection_df.groupby("filename").size().to_dict()
        visual_count_match = all(int(row.object_count) == visual_counts.get(row.sample_image, 0) for row in visual.itertuples())
        visual_brand_match = set(visual["brand"]) == set(BRAND_ORDER)
        add_check(
            "Visual inspection sample covers all brands and matches detection counts",
            visual_count_match and visual_brand_match,
            f"rows={len(visual)} brands={visual['brand'].nunique()} count_match={visual_count_match}",
            "visual",
        )
    else:
        add_check("Visual inspection sample file exists", False, str(visual_path), "visual")

    sample_rows = pd.read_csv(DATA / "sample_images_rendered.csv")
    sample_files_ok = all(Path(path).exists() for path in sample_rows["annotated_path"])
    add_check("Annotated sample images exist for visual review", sample_files_ok, f"rows={len(sample_rows)}", "visual")

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(AUDIT / "audit_checks.csv", index=False)
    summary = {
        "checks_total": int(len(checks_df)),
        "checks_passed": int((checks_df["status"] == "PASS").sum()),
        "checks_failed": int((checks_df["status"] == "FAIL").sum()),
        "failed_checks": checks_df.loc[checks_df["status"] == "FAIL", "check"].tolist(),
        "weights_sha256": actual_weights_hash,
        "reinference_images_checked": int(len(reinference_df)),
        "reinference_failures": int((reinference_df["status"] == "FAIL").sum()),
    }
    (AUDIT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["checks_failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
