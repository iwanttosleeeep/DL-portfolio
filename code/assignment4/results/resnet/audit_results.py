#!/usr/bin/env python3
"""Independent self-audit of the ResNet18 inference and analysis outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torchvision.models import ResNet18_Weights, resnet18

BRAND_DISPLAY = {
    "bang-olufsen": "Bang & Olufsen",
    "fortnum-mason": "Fortnum & Mason",
    "herman-miller": "Herman Miller",
    "jellycat": "Jellycat",
    "jo-malone-london": "Jo Malone London",
    "kodak": "Kodak",
    "lego": "LEGO",
    "oura": "Oura",
    "sennheiser": "Sennheiser",
    "stanley": "Stanley",
}
BRAND_ORDER = list(BRAND_DISPLAY)
DISPLAY_TO_BRAND = {value: key for key, value in BRAND_DISPLAY.items()}
EXPECTED_WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
EXPECTED_PARAMETERS = 11_689_512

VISUAL_NOTES = {
    "bang-olufsen_16.jpg": (
        "Person listening with over-ear headphones. Predicted 'lab coat' (0.06) and 'groom' "
        "(0.05); headphones are not a salient ImageNet class at this framing."
    ),
    "bang-olufsen_22.jpg": (
        "Close-up of a round black Bang & Olufsen speaker. Predicted 'lens cap' (0.92) because "
        "of the round black object; speakers are not an ImageNet-1K class."
    ),
    "fortnum-mason_16.jpg": (
        "Busy tea-table/store scene with a person holding a tray. Predicted 'scale' (0.06) and "
        "'tray' (0.05); the scene is diffuse and no single product dominates."
    ),
    "fortnum-mason_22.jpg": (
        "Wicker hamper filled with Fortnum & Mason products. Predicted 'hamper' (1.00), which is "
        "semantically appropriate but not brand-specific."
    ),
    "herman-miller_28.jpg": (
        "Furniture showroom containing many chairs. Predicted 'folding chair' (0.09) and 'desk' "
        "(0.06); low confidence and a generic furniture label."
    ),
    "herman-miller_06.jpg": (
        "Dining table surrounded by colorful chairs. Predicted 'folding chair' (0.96); plausible "
        "generic furniture content, but not a specific Herman Miller design."
    ),
    "jellycat_06.jpg": (
        "Grid of small plush toys on a blue background. Predicted 'wall clock' (0.12) and 'tray' "
        "(0.06); the flat-lay layout defeats object recognition."
    ),
    "jellycat_04.jpg": (
        "Jellycat patisserie-themed plush display. Predicted 'teddy' (0.99), a generic plush label; "
        "it does not identify Jellycat as a brand."
    ),
    "jo-malone-london_29.jpg": (
        "Woman inside a Jo Malone boutique. Predicted 'water jug' (0.10) and 'wine bottle' (0.05); "
        "the perfume products are small or absent."
    ),
    "jo-malone-london_26.jpg": (
        "Two Jo Malone perfume bottles. Predicted 'perfume' (1.00), which is apt but generic."
    ),
    "kodak_01.jpg": (
        "Dark close-up of a camera lens. Predicted 'loupe' (0.13) and 'hourglass' (0.06); the "
        "object is not recognizable as a camera from this crop."
    ),
    "kodak_26.jpg": (
        "Several compact Kodak cameras. Predicted 'switch' (0.99); a confident but incorrect label. "
        "ImageNet has 'Polaroid camera' and 'reflex camera' but no general compact-camera class."
    ),
    "lego_34.jpg": (
        "Colorful SpongeBob-themed LEGO packaging/poster. Predicted 'balloon' (0.05) and 'ocarina' "
        "(0.03); graphic packaging is not well represented."
    ),
    "lego_33.jpg": (
        "LEGO minifigures in a play scene. Predicted 'rubber eraser' (0.95); a confident but wrong "
        "label for small plastic figures."
    ),
    "oura_14.jpg": (
        "App interface overlaid on lifestyle imagery. Predicted 'web site' (0.07) and 'laptop' "
        "(0.06); UI screenshots have no clear ImageNet object."
    ),
    "oura_25.jpg": (
        "Hand holding a silver Oura ring. Predicted 'loupe' (0.99); the ring shape is confused with "
        "a jeweler's loupe, and 'ring' is not a suitable ImageNet class."
    ),
    "sennheiser_26.jpg": (
        "Trade-show booth photograph. Predicted 'cellular telephone' (0.05) and 'odometer' (0.05); "
        "the product is not salient in the event scene."
    ),
    "sennheiser_35.jpg": (
        "Hand holding a Sennheiser earbud charging case. Predicted 'joystick' (0.93); a confident "
        "but wrong label because earbud cases are absent from ImageNet-1K."
    ),
    "stanley_11.jpg": (
        "Dark Stanley product/accessory on a striped surface. Predicted 'Polaroid camera' (0.08) "
        "and 'backpack' (0.07); ambiguous framing and low confidence."
    ),
    "stanley_18.jpg": (
        "Stanley insulated tumbler. Predicted 'coffee mug' (0.92), a plausible generic drinkware "
        "label rather than brand identification."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/resnet"))
    parser.add_argument("--device", default="mps")
    return parser.parse_args()


def shannon_entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def pairwise_block(similarity: np.ndarray, indices: np.ndarray) -> np.ndarray:
    block = similarity[np.ix_(indices, indices)]
    mask = ~np.eye(len(indices), dtype=bool)
    return block[mask]


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict] = []

    def add(category: str, name: str, passed: bool, details: str = "") -> None:
        checks.append(
            {
                "category": category,
                "check": name,
                "passed": bool(passed),
                "details": details,
            }
        )

    metadata = json.loads((output_dir / "model_metadata.json").read_text())
    predictions = pd.read_csv(data_dir / "image_predictions.csv")
    brand_summary = pd.read_csv(data_dir / "brand_summary.csv")
    class_counts = pd.read_csv(data_dir / "brand_top1_class_counts.csv")
    overall_counts = pd.read_csv(data_dir / "overall_top1_classes.csv")
    top5_counts = pd.read_csv(data_dir / "brand_top5_class_counts.csv")
    wide_counts = pd.read_csv(data_dir / "brand_top1_class_shares_wide.csv")
    within_df = pd.read_csv(data_dir / "within_brand_pairwise_similarity.csv")
    within_between = pd.read_csv(data_dir / "within_between_similarity.csv")
    brand_pairs = pd.read_csv(data_dir / "brand_similarity_pairs.csv")
    brand_sim = pd.read_csv(data_dir / "brand_similarity_mean_pairwise.csv", index_col=0)
    centroid_sim = pd.read_csv(data_dir / "brand_similarity_centroid.csv", index_col=0)
    pca_df = pd.read_csv(data_dir / "pca_coordinates.csv")
    explained_df = pd.read_csv(data_dir / "pca_explained_variance.csv")
    analysis_summary = json.loads((data_dir / "analysis_summary.json").read_text())
    embeddings = np.load(data_dir / "image_embeddings.npz", allow_pickle=False)
    features = embeddings["features"]
    logits = embeddings["logits"]
    probabilities = embeddings["probabilities"]
    image_ids = embeddings["image_ids"].astype(str)
    embedding_brands = embeddings["brands"].astype(str)
    embedding_filenames = embeddings["filenames"].astype(str)
    class_names = embeddings["class_names"].astype(str).tolist()

    print("Audit: dataset and manifest ...")
    brand_dirs = sorted(
        p.name for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    add("dataset", "ten_brand_directories", len(brand_dirs) == 10, f"found {len(brand_dirs)}")
    on_disk: dict[str, list[Path]] = {}
    for brand in brand_dirs:
        on_disk[brand] = sorted(dataset_root.joinpath(brand).glob("*.jpg"))
        add("dataset", f"image_count_{brand}", len(on_disk[brand]) == 40, f"{len(on_disk[brand])} images")
    add("dataset", "total_disk_images", sum(len(v) for v in on_disk.values()) == 400, f"{sum(len(v) for v in on_disk.values())}")

    manifest = pd.read_csv(data_dir / "image_manifest_sha256.csv")
    manifest_rows = []
    for brand in brand_dirs:
        for path in on_disk[brand]:
            with Image.open(path) as image:
                width, height = image.size
            manifest_rows.append(
                {
                    "brand": brand,
                    "filename": path.name,
                    "image_id": f"{brand}/{path.name}",
                    "size_bytes": path.stat().st_size,
                    "width": width,
                    "height": height,
                    "sha256": sha256(path),
                }
            )
    manifest_check = pd.DataFrame(manifest_rows)
    manifest_check.to_csv(audit_dir / "image_manifest_sha256.csv", index=False)
    add("dataset", "manifest_row_count", len(manifest) == 400, f"manifest has {len(manifest)} rows")
    add("dataset", "manifest_matches_disk", set(manifest["image_id"]) == set(manifest_check["image_id"]))
    add("dataset", "manifest_hashes_match", bool((manifest.sort_values("image_id")["sha256"].to_numpy() == manifest_check.sort_values("image_id")["sha256"].to_numpy()).all()))
    add("dataset", "manifest_dimensions_match", bool((manifest.sort_values("image_id")[["width", "height"]].to_numpy() == manifest_check.sort_values("image_id")[["width", "height"]].to_numpy()).all()))

    print("Audit: model and metadata ...")
    weights_path = Path(metadata["weights_local_path"])
    add("model", "weights_file_exists", weights_path.exists(), str(weights_path))
    add("model", "weights_sha256", weights_path.exists() and sha256(weights_path) == EXPECTED_WEIGHTS_SHA256, metadata["weights_sha256"])
    add("model", "weights_enum", metadata["weights_enum"] == "ResNet18_Weights.IMAGENET1K_V1", metadata["weights_enum"])
    add("model", "architecture", metadata["architecture"] == "resnet18", metadata["architecture"])
    add("model", "parameter_count", metadata["parameter_count"] == EXPECTED_PARAMETERS, str(metadata["parameter_count"]))
    add("model", "class_count", metadata["class_count"] == 1000, str(metadata["class_count"]))
    add("model", "torch_version", metadata["torch_version"] == torch.__version__, f"{metadata['torch_version']} vs {torch.__version__}")
    add("model", "torchvision_version", metadata["torchvision_version"] == torchvision.__version__, f"{metadata['torchvision_version']} vs {torchvision.__version__}")
    add("model", "inference_device_mps", metadata["device"] == "mps" and torch.backends.mps.is_available(), metadata["device"])
    add("model", "inference_mode_only", metadata["inference_settings"]["mode"].startswith("eval"), metadata["inference_settings"]["mode"])
    preprocessing = metadata["inference_settings"]["preprocessing"]
    add("model", "preprocessing_settings", preprocessing["resize_size"] == 256 and preprocessing["crop_size"] == 224 and preprocessing["mean"] == [0.485, 0.456, 0.406] and preprocessing["std"] == [0.229, 0.224, 0.225])
    saved_classes = json.loads((output_dir / "imagenet_classes.json").read_text())
    weight_classes = list(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])
    add("model", "class_names_complete", len(saved_classes) == 1000 and all(saved_classes[str(i)] == weight_classes[i] for i in range(1000)))

    dummy = resnet18(weights=None)
    add("model", "parameter_count_recomputes", sum(p.numel() for p in dummy.parameters()) == EXPECTED_PARAMETERS)

    print("Audit: coverage and prediction consistency ...")
    add("coverage", "prediction_row_count", len(predictions) == 400, f"{len(predictions)} rows")
    add("coverage", "unique_image_ids", predictions["image_id"].nunique() == 400, f"{predictions['image_id'].nunique()} unique")
    per_brand = predictions.groupby("brand").size().to_dict()
    add("coverage", "forty_per_brand", len(per_brand) == 10 and all(v == 40 for v in per_brand.values()), json.dumps(per_brand))
    add("coverage", "prediction_ids_match_disk", set(predictions["image_id"]) == set(manifest_check["image_id"]))
    add("coverage", "embedding_ids_match_predictions", np.array_equal(predictions["image_id"].to_numpy(), image_ids))
    add("coverage", "embedding_brands_match", np.array_equal(predictions["brand"].to_numpy(), embedding_brands))
    add("coverage", "embedding_filenames_match", np.array_equal(predictions["filename"].to_numpy(), embedding_filenames))

    add("embeddings", "features_shape", features.shape == (400, 512), str(features.shape))
    add("embeddings", "logits_shape", logits.shape == (400, 1000), str(logits.shape))
    add("embeddings", "probabilities_shape", probabilities.shape == (400, 1000), str(probabilities.shape))
    add("embeddings", "features_float32", features.dtype == np.float32, str(features.dtype))
    add("embeddings", "features_finite", bool(np.isfinite(features).all()))
    add("embeddings", "logits_finite", bool(np.isfinite(logits).all()))
    add("embeddings", "probabilities_finite", bool(np.isfinite(probabilities).all()))
    add("embeddings", "no_zero_feature_rows", bool((np.linalg.norm(features, axis=1) > 0).all()))
    add("embeddings", "feature_norm_column_matches", float(np.max(np.abs(np.linalg.norm(features, axis=1) - predictions["feature_l2_norm"].to_numpy()))) < 1e-5)

    softmax_from_logits = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    add("predictions", "softmax_recomputes_probabilities", float(np.max(np.abs(softmax_from_logits - probabilities))) < 1e-7)
    top5_ids = np.array(
        [[int(value) for value in row.split(";")] for row in predictions["top5_class_ids"]]
    )
    top5_probs = np.array(
        [[float(value) for value in row.split(";")] for row in predictions["top5_probs"]]
    )
    add("predictions", "top5_has_five_ids", top5_ids.shape == (400, 5), str(top5_ids.shape))
    add("predictions", "top1_is_first_top5_id", bool((top5_ids[:, 0] == predictions["top1_class_id"].to_numpy()).all()))
    add("predictions", "top5_probabilities_descending", bool((np.diff(top5_probs, axis=1) <= 1e-10).all()))
    add("predictions", "top5_probabilities_in_range", bool(((top5_probs > 0) & (top5_probs <= 1)).all()))
    add("predictions", "top5_probability_sum", bool((predictions["top5_prob_sum"] <= 1.0000001).all()))
    add("predictions", "top1_probability_matches_softmax", float(np.max(np.abs(predictions["top1_prob"].to_numpy() - softmax_from_logits[np.arange(400), top5_ids[:, 0]]))) < 1e-6)
    add("predictions", "top5_probabilities_match_softmax", float(np.max(np.abs(top5_probs - softmax_from_logits[np.arange(400)[:, None], top5_ids]))) < 1e-6)
    add("predictions", "top1_class_names_match_ids", all(saved_classes[str(i)] == name for i, name in zip(predictions["top1_class_id"], predictions["top1_class"])))
    add("predictions", "top5_class_names_match_ids", all(all(saved_classes[str(i)] == name for i, name in zip(ids, row.split(";"))) for ids, row in zip(top5_ids, predictions["top5_classes"])))
    probabilities64 = probabilities.astype(np.float64)
    entropy_bits = np.array(
        [
            float(-(row[row > 0] * np.log2(row[row > 0])).sum())
            for row in probabilities64
        ]
    )
    add("predictions", "entropy_recomputes", float(np.max(np.abs(entropy_bits - predictions["prediction_entropy_nats"].to_numpy() / np.log(2)))) < 1e-6)
    add("predictions", "margin_recomputes", float(np.max(np.abs((top5_probs[:, 0] - top5_probs[:, 1]) - predictions["top1_margin"].to_numpy()))) < 1e-7)
    add("predictions", "top5_sum_recomputes", float(np.max(np.abs(top5_probs.sum(axis=1) - predictions["top5_prob_sum"].to_numpy()))) < 1e-7)
    pred_dims = (
        predictions.sort_values(by=["brand", "filename"])[["image_width", "image_height"]].to_numpy()
    )
    disk_dims = (
        manifest_check.sort_values(by=["brand", "filename"])[["width", "height"]].to_numpy()
    )
    add("predictions", "image_dimensions_match", bool((pred_dims == disk_dims).all()))

    print("Audit: brand-level statistics ...")
    for brand in BRAND_ORDER:
        group = predictions[predictions["brand"] == brand]
        row = brand_summary[brand_summary["brand"] == brand].iloc[0]
        counts = group["top1_class"].value_counts()
        entropy_value = shannon_entropy(counts.to_numpy())
        top5_union = {name for value in group["top5_classes"] for name in value.split(";")}
        checks_brand = {
            "unique_top1_classes": int(group["top1_class"].nunique()) == int(row["unique_top1_classes"]),
            "top1_mode_class": counts.index[0] == row["top1_mode_class"],
            "top1_mode_count": int(counts.iloc[0]) == int(row["top1_mode_count"]),
            "top1_mode_share": abs(counts.iloc[0] / 40 - float(row["top1_mode_share"])) < 1e-6,
            "shannon_entropy_bits": abs(entropy_value - float(row["shannon_entropy_bits"])) < 1e-6,
            "normalized_entropy": abs(entropy_value / np.log2(40) - float(row["normalized_entropy_log2_40"])) < 1e-6,
            "effective_classes": abs(2**entropy_value - float(row["effective_number_of_classes"])) < 1e-5,
            "top5_union_classes": len(top5_union) == int(row["top5_union_classes"]),
            "mean_top1_prob": abs(group["top1_prob"].mean() - float(row["mean_top1_prob"])) < 1e-6,
            "median_top1_prob": abs(group["top1_prob"].median() - float(row["median_top1_prob"])) < 1e-6,
            "sd_top1_prob": abs(group["top1_prob"].std(ddof=1) - float(row["sd_top1_prob"])) < 1e-6,
            "share_top1_prob_ge_0_5": abs((group["top1_prob"] >= 0.5).mean() - float(row["share_top1_prob_ge_0_5"])) < 1e-6,
            "mean_top5_prob_sum": abs(group["top5_prob_sum"].mean() - float(row["mean_top5_prob_sum"])) < 1e-6,
            "mean_top1_margin": abs(group["top1_margin"].mean() - float(row["mean_top1_margin"])) < 1e-6,
            "mean_prediction_entropy_bits": abs(group["prediction_entropy_nats"].mean() / np.log(2) - float(row["mean_prediction_entropy_bits"])) < 1e-6,
        }
        for name, passed in checks_brand.items():
            add("statistics", f"{brand}_{name}", passed)

    for brand in BRAND_ORDER:
        group = predictions[predictions["brand"] == brand]
        recomputed = group["top1_class"].value_counts()
        saved_rows = class_counts[class_counts["brand"] == brand]
        ok = set(saved_rows["class_name"]) == set(recomputed.index) and all(
            int(row["image_count"]) == int(recomputed[row["class_name"]])
            for _, row in saved_rows.iterrows()
        )
        add("statistics", f"{brand}_class_counts_reproduce", ok)
        add("statistics", f"{brand}_class_shares_sum_to_one", abs(saved_rows["share_of_brand_images"].sum() - 1.0) < 1e-6)

    for _, row in wide_counts.iterrows():
        group = predictions[predictions["brand"] == row["brand"]]
        recomputed = group["top1_class"].value_counts()
        ok = all(
            int(row[column]) == int(recomputed.get(column, 0))
            for column in wide_counts.columns
            if column not in {"brand", "brand_display"}
        )
        add("statistics", f"{row['brand']}_wide_table_reproduces", ok)

    top5_ok = True
    for brand in BRAND_ORDER:
        group = predictions[predictions["brand"] == brand]
        recomputed: dict[str, int] = {}
        for value in group["top5_classes"]:
            for name in value.split(";"):
                recomputed[name] = recomputed.get(name, 0) + 1
        saved = top5_counts[top5_counts["brand"] == brand]
        ok = set(saved["class_name"]) == set(recomputed) and all(
            int(row["image_count_top5"]) == recomputed[row["class_name"]]
            for _, row in saved.iterrows()
        )
        top5_ok = top5_ok and ok
        add("statistics", f"{brand}_top5_counts_reproduce", ok)
        add("statistics", f"{brand}_top5_counts_sum_to_200", int(saved["image_count_top5"].sum()) == 200)
    add("statistics", "top5_counts_total_reproduce", top5_ok)

    overall_recomputed = predictions["top1_class"].value_counts()
    overall_ok = set(overall_counts["top1_class"]) == set(overall_recomputed.index) and all(
        int(row["image_count"]) == int(overall_recomputed[row["top1_class"]])
        for _, row in overall_counts.iterrows()
    )
    add("statistics", "overall_counts_reproduce", overall_ok)
    add("statistics", "overall_counts_sum_to_400", int(overall_counts["image_count"].sum()) == 400)
    add("statistics", "overall_shares_use_400", bool(np.allclose(overall_counts["image_count"] / 400, overall_counts["share_of_400_images"], atol=1e-8)))

    print("Audit: similarity analysis ...")
    cosine = np.load(data_dir / "cosine_similarity_matrix.npy")
    add("similarity", "cosine_matrix_shape", cosine.shape == (400, 400), str(cosine.shape))
    add("similarity", "cosine_matrix_symmetric", float(np.max(np.abs(cosine - cosine.T))) < 1e-6)
    add("similarity", "cosine_matrix_unit_diagonal", float(np.max(np.abs(np.diag(cosine) - 1.0))) < 1e-6)
    add("similarity", "cosine_matrix_range", float(cosine.min()) >= -1.0 - 1e-6 and float(cosine.max()) <= 1.0 + 1e-6, f"[{cosine.min():.4f}, {cosine.max():.4f}]")
    recomputed_cosine = (features.astype(np.float64) / np.linalg.norm(features, axis=1, keepdims=True)) @ (
        features.astype(np.float64) / np.linalg.norm(features, axis=1, keepdims=True)
    ).T
    add("similarity", "cosine_matrix_recomputes", float(np.max(np.abs(cosine - recomputed_cosine))) < 2e-6, f"max diff {float(np.max(np.abs(cosine - recomputed_cosine))):.2e}")

    within_values = []
    between_values = []
    for i in range(400):
        for j in range(i + 1, 400):
            if embedding_brands[i] == embedding_brands[j]:
                within_values.append(recomputed_cosine[i, j])
            else:
                between_values.append(recomputed_cosine[i, j])
    within_values = np.array(within_values)
    between_values = np.array(between_values)
    for kind, values in [("within", within_values), ("between", between_values)]:
        row = within_between[within_between["pair_type"] == kind].iloc[0]
        prefix = "within" if kind == "within" else "between"
        add("similarity", f"{kind}_pair_count", int(len(values)) == int(row["n_pairs"]), f"{len(values)}")
        for statistic, function in [
            ("mean_cosine", np.mean),
            ("sd_cosine", lambda v: np.std(v, ddof=1)),
            ("median_cosine", np.median),
            ("q05_cosine", lambda v: np.quantile(v, 0.05)),
            ("q95_cosine", lambda v: np.quantile(v, 0.95)),
            ("min_cosine", np.min),
            ("max_cosine", np.max),
        ]:
            add("similarity", f"{prefix}_{statistic}", abs(float(function(values)) - float(row[statistic])) < 1e-6)

    within_ok = True
    for brand in BRAND_ORDER:
        indices = np.where(embedding_brands == brand)[0]
        values = pairwise_block(recomputed_cosine, indices)
        row = within_df[within_df["brand"] == brand].iloc[0]
        ok = (
            int(len(values)) == int(row["n_pairs"])
            and abs(values.mean() - float(row["within_mean_cosine"])) < 1e-6
            and abs(values.std(ddof=1) - float(row["within_sd_cosine"])) < 1e-6
            and abs(np.median(values) - float(row["within_median_cosine"])) < 1e-6
        )
        within_ok = within_ok and ok
        add("similarity", f"{brand}_within_stats", ok)
    add("similarity", "within_stats_all_brands", within_ok)

    brand_sim_recomputed = np.zeros((10, 10))
    for i, brand_a in enumerate(BRAND_ORDER):
        idx_a = np.where(embedding_brands == brand_a)[0]
        for j, brand_b in enumerate(BRAND_ORDER):
            idx_b = np.where(embedding_brands == brand_b)[0]
            values = (
                pairwise_block(recomputed_cosine, idx_a)
                if i == j
                else recomputed_cosine[np.ix_(idx_a, idx_b)].ravel()
            )
            brand_sim_recomputed[i, j] = values.mean()
    saved_brand_sim = brand_sim.reindex([BRAND_DISPLAY[b] for b in BRAND_ORDER]).reindex(
        columns=[BRAND_DISPLAY[b] for b in BRAND_ORDER]
    ).to_numpy()
    add("similarity", "brand_similarity_matrix_recomputes", float(np.max(np.abs(saved_brand_sim - brand_sim_recomputed))) < 2e-6, f"max diff {float(np.max(np.abs(saved_brand_sim - brand_sim_recomputed))):.2e}")
    add("similarity", "brand_similarity_symmetric", float(np.max(np.abs(brand_sim_recomputed - brand_sim_recomputed.T))) < 1e-9)

    normalized = features.astype(np.float64) / np.linalg.norm(features, axis=1, keepdims=True)
    centroids = np.vstack([normalized[embedding_brands == brand].mean(axis=0) for brand in BRAND_ORDER])
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_recomputed = centroids @ centroids.T
    saved_centroid = centroid_sim.reindex([BRAND_DISPLAY[b] for b in BRAND_ORDER]).reindex(
        columns=[BRAND_DISPLAY[b] for b in BRAND_ORDER]
    ).to_numpy()
    add("similarity", "centroid_similarity_recomputes", float(np.max(np.abs(saved_centroid - centroid_recomputed))) < 2e-6)
    add("similarity", "centroid_diagonal_unit", float(np.max(np.abs(np.diag(centroid_recomputed) - 1))) < 1e-9)

    pairs_ok = True
    for _, row in brand_pairs.iterrows():
        i = BRAND_ORDER.index(row["brand_a"])
        j = BRAND_ORDER.index(row["brand_b"])
        saved = float(row["mean_pairwise_cosine"])
        recomputed = recomputed_cosine[
            np.ix_(np.where(embedding_brands == row["brand_a"])[0], np.where(embedding_brands == row["brand_b"])[0])
        ].mean()
        pairs_ok = pairs_ok and abs(saved - recomputed) < 1e-6
    add("similarity", "brand_pair_table_recomputes", pairs_ok)
    add("similarity", "brand_pair_table_complete", len(brand_pairs) == 45, f"{len(brand_pairs)} pairs")

    print("Audit: PCA and figures ...")
    explained = explained_df["explained_variance_ratio"].to_numpy()
    add("pca", "coordinates_shape", pca_df.shape[0] == 400 and {"pc1", "pc2"}.issubset(pca_df.columns), str(pca_df.shape))
    add("pca", "coordinates_finite", bool(np.isfinite(pca_df[["pc1", "pc2"]].to_numpy()).all()))
    add("pca", "coordinates_centered", float(np.max(np.abs(pca_df[["pc1", "pc2"]].mean())) < 1e-8), f"mean {pca_df[['pc1','pc2']].mean().to_dict()}")
    recomputed_pca = PCA(n_components=50, svd_solver="full").fit_transform(normalized)
    summary_pc = np.array(
        [
            analysis_summary["pca_pc1_explained_variance"],
            analysis_summary["pca_pc2_explained_variance"],
        ]
    )
    add("pca", "explained_variance_recomputes", float(np.max(np.abs(explained[:2] - summary_pc))) < 1e-6)
    corr_pc1 = float(np.corrcoef(recomputed_pca[:, 0], pca_df["pc1"].to_numpy())[0, 1])
    corr_pc2 = float(np.corrcoef(recomputed_pca[:, 1], pca_df["pc2"].to_numpy())[0, 1])
    add("pca", "pc1_recomputes", abs(corr_pc1) > 0.999999, f"corr {corr_pc1:.10f}")
    add("pca", "pc2_recomputes", abs(corr_pc2) > 0.999999, f"corr {corr_pc2:.10f}")

    silhouette = float(silhouette_score(normalized, embedding_brands, metric="cosine"))
    add("pca", "silhouette_recomputes", abs(silhouette - float(analysis_summary["silhouette_cosine_brand_labels"])) < 1e-6, f"{silhouette:.6f}")
    add("summary", "within_between_means", abs(float(within_values.mean()) - float(analysis_summary["within_brand_cosine_mean"])) < 1e-6 and abs(float(between_values.mean()) - float(analysis_summary["between_brand_cosine_mean"])) < 1e-6)
    add("summary", "pca_variance_values", abs(float(explained[:2].sum()) - float(analysis_summary["pca_pc1_pc2_cumulative"])) < 1e-6)

    expected_figures = [
        "fig01_overall_top1_classes.png",
        "fig02_brand_top1_heatmap.png",
        "fig03_brand_diversity.png",
        "fig04_top1_confidence.png",
        "fig05_brand_similarity_heatmap.png",
        "fig06_pca_embeddings.png",
        "fig07_within_vs_between.png",
        "fig08_example_predictions.png",
        "fig09_top5_class_heatmap.png",
    ]
    figure_rows = []
    for name in expected_figures:
        path = figures_dir / name
        exists = path.exists()
        width = height = 0
        if exists:
            with Image.open(path) as image:
                width, height = image.size
        size_bytes = path.stat().st_size if exists else 0
        figure_rows.append(
            {
                "figure": name,
                "exists": exists,
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
            }
        )
        add("figures", f"{name}_exists", exists and size_bytes > 10_000 and width >= 1000 and height >= 500, f"{width}x{height}, {size_bytes} bytes")
    pd.DataFrame(figure_rows).to_csv(audit_dir / "audit_figures.csv", index=False)

    print("Audit: visual inspection sample ...")
    visual_rows = []
    for brand in BRAND_ORDER:
        group = predictions[predictions["brand"] == brand].sort_values("top1_prob")
        for tag, row in [("lowest_confidence", group.iloc[0]), ("highest_confidence", group.iloc[-1])]:
            note = VISUAL_NOTES.get(row["filename"], "")
            visual_rows.append(
                {
                    "brand": brand,
                    "brand_display": BRAND_DISPLAY[brand],
                    "selection": tag,
                    "filename": row["filename"],
                    "image_id": row["image_id"],
                    "top1_class": row["top1_class"],
                    "top1_prob": row["top1_prob"],
                    "top2_class": row["top2_class"],
                    "top2_prob": row["top2_prob"],
                    "visual_note": note,
                }
            )
    visual_df = pd.DataFrame(visual_rows)
    visual_df.to_csv(audit_dir / "audit_visual_samples.csv", index=False)
    add("visual", "visual_sample_covers_all_brands", visual_df["brand"].nunique() == 10 and len(visual_df) == 20)
    add("visual", "visual_notes_completed", bool((visual_df["visual_note"].str.len() > 20).all()))
    add("visual", "example_sheet_exists", (figures_dir / "fig08_example_predictions.png").exists())

    print("Audit: independent re-inference ...")
    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval().to(device)
    extractor = torch.nn.Sequential(*(list(model.children())[:-1])).eval().to(device)
    transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
    selection = []
    for brand in BRAND_ORDER:
        for suffix in ["01", "40"]:
            filename = f"{brand}_{suffix}.jpg"
            matches = predictions[(predictions["brand"] == brand) & (predictions["filename"] == filename)]
            if len(matches) != 1:
                raise RuntimeError(f"Missing audit image {brand}/{filename}")
            selection.append(matches.iloc[0])

    reinference_rows = []
    for start in range(0, len(selection), 10):
        batch_rows = selection[start : start + 10]
        tensors = []
        for row in batch_rows:
            with Image.open(dataset_root / row["brand"] / row["filename"]) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            logits_batch = model(batch)
            features_batch = extractor(batch).flatten(1)
        probs_batch = torch.softmax(logits_batch.to("cpu", dtype=torch.float32), dim=1)
        new_features = features_batch.to("cpu", dtype=torch.float32).numpy()
        for local_index, row in enumerate(batch_rows):
            original_index = int(np.where(image_ids == row["image_id"])[0][0])
            new_top_probs, new_top_ids = torch.topk(probs_batch[local_index], k=5)
            new_ids = [int(value) for value in new_top_ids.tolist()]
            saved_ids = [int(value) for value in str(row["top5_class_ids"]).split(";")]
            prob_difference = float(
                np.max(np.abs(new_top_probs.numpy() - np.array([float(v) for v in str(row["top5_probs"]).split(";")])))
            )
            feature_cosine = float(
                np.dot(new_features[local_index], features[original_index])
                / (np.linalg.norm(new_features[local_index]) * np.linalg.norm(features[original_index]))
            )
            passed = new_ids == saved_ids and prob_difference < 5e-5 and feature_cosine > 0.9999
            reinference_rows.append(
                {
                    "image_id": row["image_id"],
                    "saved_top1_class": row["top1_class"],
                    "rerun_top1_class": class_names[new_ids[0]],
                    "top5_ids_match": new_ids == saved_ids,
                    "max_abs_top5_prob_difference": prob_difference,
                    "feature_cosine": feature_cosine,
                    "passed": passed,
                }
            )
    reinference_df = pd.DataFrame(reinference_rows)
    reinference_df.to_csv(audit_dir / "audit_reinference_check.csv", index=False)
    add("reproducibility", "reinference_count", len(reinference_df) == 20, f"{len(reinference_df)} images")
    add("reproducibility", "reinference_top5_ids", bool(reinference_df["top5_ids_match"].all()))
    add("reproducibility", "reinference_probability_tolerance", float(reinference_df["max_abs_top5_prob_difference"].max()) < 5e-5, f"max diff {reinference_df['max_abs_top5_prob_difference'].max():.2e}")
    add("reproducibility", "reinference_feature_cosine", float(reinference_df["feature_cosine"].min()) > 0.9999, f"min cosine {reinference_df['feature_cosine'].min():.8f}")
    add("reproducibility", "reinference_all_passed", bool(reinference_df["passed"].all()))

    print("Audit: documentation ...")
    analysis_path = output_dir / "analysis.md"
    add("documentation", "analysis_document_exists", analysis_path.exists() and analysis_path.stat().st_size > 1000, f"{analysis_path.stat().st_size if analysis_path.exists() else 0} bytes")
    if analysis_path.exists():
        analysis_text = analysis_path.read_text()
        add("documentation", "analysis_has_limitations_section", "limitation" in analysis_text.lower())
        add("documentation", "analysis_has_no_ground_truth_accuracy_claim", "ground-truth accuracy" not in analysis_text.lower() or "cannot" in analysis_text.lower())

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(audit_dir / "audit_checks.csv", index=False)
    category_counts = checks_df.groupby("category").agg(
        checks=("check", "size"), passed=("passed", "sum")
    ).reset_index()
    category_counts["failed"] = category_counts["checks"] - category_counts["passed"]
    failed = checks_df[~checks_df["passed"]]
    summary = {
        "total_checks": int(len(checks_df)),
        "passed_checks": int(checks_df["passed"].sum()),
        "failed_checks": int((~checks_df["passed"]).sum()),
        "categories": category_counts.to_dict(orient="records"),
        "failures": failed.to_dict(orient="records"),
        "reinference": {
            "images": int(len(reinference_df)),
            "max_abs_top5_prob_difference": float(reinference_df["max_abs_top5_prob_difference"].max()),
            "min_feature_cosine": float(reinference_df["feature_cosine"].min()),
        },
    }
    with (audit_dir / "audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    if summary["failed_checks"]:
        print("\nFAILED CHECKS:")
        print(failed.to_string(index=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
