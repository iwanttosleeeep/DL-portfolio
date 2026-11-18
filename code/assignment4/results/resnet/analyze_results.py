#!/usr/bin/env python3
"""Quantitative brand analysis of ResNet18 predictions and embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

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
COLORS = plt.get_cmap("tab10").colors
BRAND_COLORS = {brand: COLORS[i % 10] for i, brand in enumerate(BRAND_ORDER)}
SIM_CMAP = LinearSegmentedColormap.from_list(
    "sim", ["#3b4cc0", "#f7f7f7", "#b40426"]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/resnet"))
    parser.add_argument("--top-n-classes", type=int, default=15)
    return parser.parse_args()


def shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())


def pairwise_cosine_block(similarity: np.ndarray, indices: np.ndarray) -> np.ndarray:
    block = similarity[np.ix_(indices, indices)]
    if len(indices) < 2:
        return np.array([], dtype=float)
    mask = ~np.eye(len(indices), dtype=bool)
    return block[mask]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(data_dir / "image_predictions.csv")
    embeddings = np.load(data_dir / "image_embeddings.npz", allow_pickle=False)
    features = embeddings["features"].astype(np.float64)
    image_ids = embeddings["image_ids"].astype(str)
    brands_array = embeddings["brands"].astype(str)
    filenames = embeddings["filenames"].astype(str)
    logits = embeddings["logits"]

    if predictions.shape[0] != 400 or features.shape != (400, 512):
        raise RuntimeError("Unexpected input shapes; rerun inference")

    if not np.array_equal(predictions["image_id"].to_numpy(), image_ids):
        raise RuntimeError("Prediction rows and embedding rows are not aligned")

    brand_counts = predictions.groupby("brand").size().to_dict()
    if len(brand_counts) != 10 or any(value != 40 for value in brand_counts.values()):
        raise RuntimeError(f"Unexpected brand counts: {brand_counts}")

    print("Building image-level similarity ...")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / norms
    cosine = normalized @ normalized.T
    np.save(data_dir / "cosine_similarity_matrix.npy", cosine.astype(np.float32))

    pair_types: list[str] = []
    pair_values: list[float] = []
    brand_pair_values: list[float] = []
    for i in range(len(image_ids)):
        for j in range(i + 1, len(image_ids)):
            pair_types.append("within" if brands_array[i] == brands_array[j] else "between")
            pair_values.append(float(cosine[i, j]))
        for j in range(i + 1, len(image_ids)):
            if brands_array[i] == brands_array[j]:
                brand_pair_values.append(float(cosine[i, j]))
    pair_types_array = np.array(pair_types)
    pair_values_array = np.array(pair_values)

    within_values = pair_values_array[pair_types_array == "within"]
    between_values = pair_values_array[pair_types_array == "between"]

    within_between = pd.DataFrame(
        [
            {
                "pair_type": kind,
                "n_pairs": int(len(values)),
                "mean_cosine": round(float(values.mean()), 6),
                "sd_cosine": round(float(values.std(ddof=1)), 6),
                "median_cosine": round(float(np.median(values)), 6),
                "q05_cosine": round(float(np.quantile(values, 0.05)), 6),
                "q95_cosine": round(float(np.quantile(values, 0.95)), 6),
                "min_cosine": round(float(values.min()), 6),
                "max_cosine": round(float(values.max()), 6),
            }
            for kind, values in [("within", within_values), ("between", between_values)]
        ]
    )
    within_between.to_csv(data_dir / "within_between_similarity.csv", index=False)

    print("Computing within-brand image similarity ...")
    within_rows = []
    for brand in BRAND_ORDER:
        indices = np.where(brands_array == brand)[0]
        values = pairwise_cosine_block(cosine, indices)
        within_rows.append(
            {
                "brand": brand,
                "brand_display": BRAND_DISPLAY[brand],
                "n_images": int(len(indices)),
                "n_pairs": int(len(values)),
                "within_mean_cosine": round(float(values.mean()), 6),
                "within_sd_cosine": round(float(values.std(ddof=1)), 6),
                "within_median_cosine": round(float(np.median(values)), 6),
                "within_min_cosine": round(float(values.min()), 6),
                "within_max_cosine": round(float(values.max()), 6),
            }
        )
    within_df = pd.DataFrame(within_rows)
    within_df.to_csv(data_dir / "within_brand_pairwise_similarity.csv", index=False)

    print("Computing brand-level similarity matrices ...")
    brand_sim = np.zeros((10, 10))
    brand_pair_rows = []
    for i, brand_a in enumerate(BRAND_ORDER):
        idx_a = np.where(brands_array == brand_a)[0]
        for j, brand_b in enumerate(BRAND_ORDER):
            idx_b = np.where(brands_array == brand_b)[0]
            if i == j:
                values = pairwise_cosine_block(cosine, idx_a)
            else:
                values = cosine[np.ix_(idx_a, idx_b)].ravel()
            brand_sim[i, j] = float(values.mean())
            if i < j:
                brand_pair_rows.append(
                    {
                        "brand_a": brand_a,
                        "brand_b": brand_b,
                        "brand_a_display": BRAND_DISPLAY[brand_a],
                        "brand_b_display": BRAND_DISPLAY[brand_b],
                        "n_image_pairs": int(len(values)),
                        "mean_pairwise_cosine": round(float(values.mean()), 6),
                        "sd_pairwise_cosine": round(float(values.std(ddof=1)), 6),
                        "median_pairwise_cosine": round(float(np.median(values)), 6),
                        "min_pairwise_cosine": round(float(values.min()), 6),
                        "max_pairwise_cosine": round(float(values.max()), 6),
                    }
                )

    brand_sim_df = pd.DataFrame(brand_sim, index=BRAND_ORDER, columns=BRAND_ORDER)
    brand_sim_display = brand_sim_df.copy()
    brand_sim_display.index = [BRAND_DISPLAY[b] for b in BRAND_ORDER]
    brand_sim_display.columns = [BRAND_DISPLAY[b] for b in BRAND_ORDER]
    brand_sim_display.round(6).to_csv(data_dir / "brand_similarity_mean_pairwise.csv")

    brand_centroids = np.vstack(
        [normalized[brands_array == brand].mean(axis=0) for brand in BRAND_ORDER]
    )
    brand_centroids = brand_centroids / np.linalg.norm(brand_centroids, axis=1, keepdims=True)
    centroid_sim = brand_centroids @ brand_centroids.T
    centroid_df = pd.DataFrame(centroid_sim, index=BRAND_ORDER, columns=BRAND_ORDER)
    centroid_display = centroid_df.copy()
    centroid_display.index = [BRAND_DISPLAY[b] for b in BRAND_ORDER]
    centroid_display.columns = [BRAND_DISPLAY[b] for b in BRAND_ORDER]
    centroid_display.round(6).to_csv(data_dir / "brand_similarity_centroid.csv")

    brand_pairs = pd.DataFrame(brand_pair_rows).sort_values(
        "mean_pairwise_cosine", ascending=False
    )
    brand_pairs.to_csv(data_dir / "brand_similarity_pairs.csv", index=False)

    print("Computing per-image brand similarity diagnostics ...")
    image_sim_rows = []
    for index, image_id in enumerate(image_ids):
        same = np.where((brands_array == brands_array[index]) & (np.arange(400) != index))[0]
        other = np.where(brands_array != brands_array[index])[0]
        same_values = cosine[index, same]
        nearest_local = same[np.argmax(same_values)]
        image_sim_rows.append(
            {
                "image_id": image_id,
                "brand": brands_array[index],
                "filename": filenames[index],
                "mean_cosine_to_same_brand": round(float(same_values.mean()), 6),
                "max_cosine_to_same_brand": round(float(same_values.max()), 6),
                "nearest_same_brand_image": image_ids[nearest_local],
                "mean_cosine_to_other_brands": round(float(cosine[index, other].mean()), 6),
                "same_minus_other_mean": round(
                    float(same_values.mean() - cosine[index, other].mean()), 6
                ),
            }
        )
    image_sim = pd.DataFrame(image_sim_rows)
    image_sim.to_csv(data_dir / "image_brand_similarity.csv", index=False)

    with_brand_stats = within_df.set_index("brand")
    for column in ["within_mean_cosine", "within_sd_cosine"]:
        image_sim[column] = image_sim["brand"].map(with_brand_stats[column])

    print("Computing classification summaries ...")
    overall_counts = (
        predictions.groupby(["top1_class_id", "top1_class"])
        .size()
        .reset_index(name="image_count")
        .sort_values(["image_count", "top1_class"], ascending=[False, True])
    )
    overall_counts["share_of_400_images"] = overall_counts["image_count"] / 400
    brand_membership = (
        predictions.groupby("top1_class")["brand"].apply(lambda s: sorted(set(s))).to_dict()
    )
    overall_counts["n_brands"] = overall_counts["top1_class"].map(
        lambda name: len(brand_membership[name])
    )
    overall_counts["brands"] = overall_counts["top1_class"].map(
        lambda name: ";".join(brand_membership[name])
    )
    overall_counts.to_csv(data_dir / "overall_top1_classes.csv", index=False)

    class_counts_rows = []
    class_share_rows = []
    for brand in BRAND_ORDER:
        group = predictions[predictions["brand"] == brand]
        counts = (
            group.groupby(["top1_class_id", "top1_class"])
            .size()
            .reset_index(name="image_count")
            .sort_values(["image_count", "top1_class"], ascending=[False, True])
        )
        counts["brand"] = brand
        counts["brand_display"] = BRAND_DISPLAY[brand]
        counts["share_of_brand_images"] = counts["image_count"] / len(group)
        class_counts_rows.append(counts)
    class_counts = pd.concat(class_counts_rows, ignore_index=True)[
        [
            "brand",
            "brand_display",
            "top1_class_id",
            "top1_class",
            "image_count",
            "share_of_brand_images",
        ]
    ]
    class_counts.rename(columns={"top1_class_id": "class_id", "top1_class": "class_name"}).to_csv(
        data_dir / "brand_top1_class_counts.csv", index=False
    )

    top_classes_for_heatmap = overall_counts.head(args.top_n_classes)["top1_class"].tolist()
    heatmap_rows = []
    for brand in BRAND_ORDER:
        group = predictions[predictions["brand"] == brand]
        counts = group["top1_class"].value_counts()
        row = {"brand": brand, "brand_display": BRAND_DISPLAY[brand]}
        for class_name in top_classes_for_heatmap:
            row[class_name] = int(counts.get(class_name, 0))
        heatmap_rows.append(row)
    heatmap_df = pd.DataFrame(heatmap_rows)
    heatmap_df.to_csv(data_dir / "brand_top1_class_shares_wide.csv", index=False)

    top5_rows = []
    for brand in BRAND_ORDER:
        group = predictions[predictions["brand"] == brand]
        counts: dict[str, int] = {}
        for value in group["top5_classes"]:
            for class_name in value.split(";"):
                counts[class_name] = counts.get(class_name, 0) + 1
        for class_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            top5_rows.append(
                {
                    "brand": brand,
                    "brand_display": BRAND_DISPLAY[brand],
                    "class_name": class_name,
                    "image_count_top5": count,
                    "share_of_brand_images": count / len(group),
                }
            )
    top5_counts = pd.DataFrame(top5_rows)
    top5_counts.to_csv(data_dir / "brand_top5_class_counts.csv", index=False)

    print("Computing brand summaries ...")
    summary_rows = []
    for brand in BRAND_ORDER:
        group = predictions[predictions["brand"] == brand]
        counts = group["top1_class"].value_counts()
        entropy_bits = shannon_entropy(counts.to_numpy())
        top5_union = {name for value in group["top5_classes"] for name in value.split(";")}
        overall_ranking = overall_counts.set_index("top1_class")["image_count"]
        overall_share = counts / overall_ranking
        within_row = within_df[within_df["brand"] == brand].iloc[0]
        other_similarities = {
            other: brand_sim[BRAND_ORDER.index(brand), BRAND_ORDER.index(other)]
            for other in BRAND_ORDER
            if other != brand
        }
        most_similar = max(other_similarities, key=other_similarities.get)
        summary_rows.append(
            {
                "brand": brand,
                "brand_display": BRAND_DISPLAY[brand],
                "n_images": int(len(group)),
                "unique_top1_classes": int(group["top1_class"].nunique()),
                "top1_mode_class": counts.index[0],
                "top1_mode_count": int(counts.iloc[0]),
                "top1_mode_share": round(float(counts.iloc[0] / len(group)), 6),
                "top1_classes_covering_50pct": int(
                    np.searchsorted(
                        (counts.cumsum() / len(group)).to_numpy(), 0.5, side="left"
                    )
                    + 1
                ),
                "shannon_entropy_bits": round(entropy_bits, 6),
                "normalized_entropy_log2_40": round(entropy_bits / np.log2(40), 6),
                "effective_number_of_classes": round(float(2**entropy_bits), 6),
                "top5_union_classes": int(len(top5_union)),
                "mean_top1_prob": round(float(group["top1_prob"].mean()), 6),
                "median_top1_prob": round(float(group["top1_prob"].median()), 6),
                "sd_top1_prob": round(float(group["top1_prob"].std(ddof=1)), 6),
                "share_top1_prob_ge_0_5": round(float((group["top1_prob"] >= 0.5).mean()), 6),
                "mean_top5_prob_sum": round(float(group["top5_prob_sum"].mean()), 6),
                "mean_top1_margin": round(float(group["top1_margin"].mean()), 6),
                "mean_prediction_entropy_bits": round(
                    float(group["prediction_entropy_nats"].mean() / np.log(2)), 6
                ),
                "within_brand_cosine_mean": round(float(within_row["within_mean_cosine"]), 6),
                "within_brand_cosine_sd": round(float(within_row["within_sd_cosine"]), 6),
                "mean_cosine_to_other_brands": round(
                    float(
                        np.mean(
                            [
                                brand_sim[BRAND_ORDER.index(brand), BRAND_ORDER.index(other)]
                                for other in BRAND_ORDER
                                if other != brand
                            ]
                        )
                    ),
                    6,
                ),
                "most_similar_other_brand": most_similar,
                "most_similar_other_brand_cosine": round(
                    float(other_similarities[most_similar]), 6
                ),
                "dominant_class_share_of_all_occurrences": round(
                    float(overall_share.iloc[0]), 6
                ),
            }
        )
    brand_summary = pd.DataFrame(summary_rows)
    brand_summary.to_csv(data_dir / "brand_summary.csv", index=False)

    print("Computing PCA and clustering diagnostics ...")
    pca = PCA(n_components=50, svd_solver="full")
    pca_coords_full = pca.fit_transform(normalized)
    pca_2d = pca_coords_full[:, :2]
    pca_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "brand": brands_array,
            "brand_display": [BRAND_DISPLAY[b] for b in brands_array],
            "filename": filenames,
            "pc1": pca_coords_full[:, 0],
            "pc2": pca_coords_full[:, 1],
            "top1_class": predictions["top1_class"].to_numpy(),
            "top1_prob": predictions["top1_prob"].to_numpy(),
        }
    )
    pca_df.to_csv(data_dir / "pca_coordinates.csv", index=False)
    explained = pca.explained_variance_ratio_
    pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": np.cumsum(explained),
        }
    ).to_csv(data_dir / "pca_explained_variance.csv", index=False)

    silhouette = float(silhouette_score(normalized, brands_array, metric="cosine"))

    overall_entropy = shannon_entropy(overall_counts["image_count"].to_numpy())
    analysis_summary = {
        "n_images": int(len(predictions)),
        "n_brands": int(len(BRAND_ORDER)),
        "images_per_brand": 40,
        "unique_top1_classes_overall": int(predictions["top1_class"].nunique()),
        "overall_top1_entropy_bits": round(overall_entropy, 6),
        "overall_top1_effective_classes": round(float(2**overall_entropy), 6),
        "overall_mean_top1_prob": round(float(predictions["top1_prob"].mean()), 6),
        "overall_median_top1_prob": round(float(predictions["top1_prob"].median()), 6),
        "overall_share_top1_prob_ge_0_5": round(float((predictions["top1_prob"] >= 0.5).mean()), 6),
        "overall_top1_class": str(overall_counts.iloc[0]["top1_class"]),
        "overall_top1_class_count": int(overall_counts.iloc[0]["image_count"]),
        "overall_top1_class_share": round(float(overall_counts.iloc[0]["image_count"] / 400), 6),
        "within_brand_cosine_mean": round(float(within_values.mean()), 6),
        "between_brand_cosine_mean": round(float(between_values.mean()), 6),
        "within_minus_between_cosine": round(
            float(within_values.mean() - between_values.mean()), 6
        ),
        "silhouette_cosine_brand_labels": round(silhouette, 6),
        "pca_pc1_explained_variance": round(float(explained[0]), 6),
        "pca_pc2_explained_variance": round(float(explained[1]), 6),
        "pca_pc1_pc2_cumulative": round(float(explained[:2].sum()), 6),
        "pca_10_component_cumulative": round(float(explained[:10].sum()), 6),
        "most_similar_brand_pair": brand_pairs.iloc[0][
            ["brand_a_display", "brand_b_display", "mean_pairwise_cosine"]
        ].to_dict(),
        "least_similar_brand_pair": brand_pairs.iloc[-1][
            ["brand_a_display", "brand_b_display", "mean_pairwise_cosine"]
        ].to_dict(),
        "most_heterogeneous_brand": brand_summary.sort_values(
            "shannon_entropy_bits", ascending=False
        ).iloc[0][["brand_display", "shannon_entropy_bits", "unique_top1_classes"]].to_dict(),
        "most_concentrated_brand": brand_summary.sort_values(
            "top1_mode_share", ascending=False
        ).iloc[0][["brand_display", "top1_mode_class", "top1_mode_share"]].to_dict(),
    }
    analysis_summary["most_similar_brand_pair"]["mean_pairwise_cosine"] = round(
        float(analysis_summary["most_similar_brand_pair"]["mean_pairwise_cosine"]), 6
    )
    analysis_summary["least_similar_brand_pair"]["mean_pairwise_cosine"] = round(
        float(analysis_summary["least_similar_brand_pair"]["mean_pairwise_cosine"]), 6
    )
    with (data_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis_summary, handle, indent=2)
        handle.write("\n")

    print("Creating figures ...")
    brand_labels = [BRAND_DISPLAY[b] for b in BRAND_ORDER]

    top_overall = overall_counts.head(args.top_n_classes).iloc[::-1]
    top_names = top_overall["top1_class"].tolist()
    composition = (
        predictions[predictions["top1_class"].isin(top_names)]
        .groupby(["top1_class", "brand"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=top_names, columns=BRAND_ORDER, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(10, 6.2))
    left = np.zeros(len(top_names))
    for brand in BRAND_ORDER:
        values = composition[brand].to_numpy()
        ax.barh(top_names, values, left=left, color=BRAND_COLORS[brand], label=BRAND_DISPLAY[brand])
        left += values
    for y, (count, share) in enumerate(
        zip(top_overall["image_count"], top_overall["share_of_400_images"])
    ):
        ax.text(count + 0.3, y, f"{count} ({share:.1%})", va="center", fontsize=8)
    ax.set_xlabel("Images (of 400)")
    ax.set_title(f"Overall top-{args.top_n_classes} ResNet18 top-1 ImageNet classes, by brand")
    ax.set_xlim(0, top_overall["image_count"].max() * 1.22)
    ax.legend(fontsize=7, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig01_overall_top1_classes.png")
    plt.close(fig)

    heat = heatmap_df.set_index("brand_display")[top_classes_for_heatmap]
    heat = heat.reindex(brand_labels)
    heat_share = heat / 40.0
    fig, ax = plt.subplots(figsize=(11, 6.2))
    image = ax.imshow(heat_share.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(len(top_classes_for_heatmap)))
    ax.set_xticklabels(top_classes_for_heatmap, rotation=45, ha="right")
    ax.set_yticks(range(len(brand_labels)))
    ax.set_yticklabels(brand_labels)
    for i in range(heat_share.shape[0]):
        for j in range(heat_share.shape[1]):
            value = heat_share.to_numpy()[i, j]
            if value > 0:
                ax.text(
                    j,
                    i,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.35 else "black",
                )
    fig.colorbar(image, ax=ax, label="Share of brand images")
    ax.set_title("Top-1 class composition by brand (share of 40 images)")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig02_brand_top1_heatmap.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order_unique = brand_summary.sort_values("unique_top1_classes", ascending=True)
    axes[0].barh(
        [BRAND_DISPLAY[b] for b in order_unique["brand"]],
        order_unique["unique_top1_classes"],
        color=[BRAND_COLORS[b] for b in order_unique["brand"]],
    )
    for y, value in enumerate(order_unique["unique_top1_classes"]):
        axes[0].text(value + 0.3, y, str(value), va="center", fontsize=8)
    axes[0].set_xlabel("Distinct top-1 ImageNet classes (of 40 images)")
    axes[0].set_title("Label variety")
    axes[0].set_xlim(0, order_unique["unique_top1_classes"].max() * 1.18)

    order_entropy = brand_summary.sort_values("shannon_entropy_bits", ascending=True)
    axes[1].barh(
        [BRAND_DISPLAY[b] for b in order_entropy["brand"]],
        order_entropy["shannon_entropy_bits"],
        color=[BRAND_COLORS[b] for b in order_entropy["brand"]],
    )
    for y, value in enumerate(order_entropy["shannon_entropy_bits"]):
        axes[1].text(value + 0.03, y, f"{value:.2f}", va="center", fontsize=8)
    axes[1].set_xlabel("Shannon entropy of top-1 labels (bits)")
    axes[1].set_title("Class-distribution diversity")
    axes[1].set_xlim(0, order_entropy["shannon_entropy_bits"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig03_brand_diversity.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    data = [predictions.loc[predictions["brand"] == brand, "top1_prob"].to_numpy() for brand in BRAND_ORDER]
    box = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, brand in zip(box["boxes"], BRAND_ORDER):
        patch.set_facecolor(BRAND_COLORS[brand])
        patch.set_alpha(0.65)
    for median in box["medians"]:
        median.set_color("black")
    rng = np.random.default_rng(0)
    for index, (brand, values) in enumerate(zip(BRAND_ORDER, data), start=1):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(index + jitter, values, s=8, color=BRAND_COLORS[brand], alpha=0.55, linewidths=0)
        ax.scatter(index, values.mean(), marker="D", s=30, color="black", zorder=3)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([BRAND_DISPLAY[b] for b in BRAND_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Top-1 softmax probability")
    ax.set_title("Top-1 confidence by brand (box; diamond = mean; points = images)")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig04_top1_confidence.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(brand_sim, cmap=SIM_CMAP, vmin=0.50, vmax=0.70)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(brand_labels, rotation=45, ha="right")
    ax.set_yticklabels(brand_labels)
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                f"{brand_sim[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(brand_sim[i, j]) > 0.55 else "black",
            )
    fig.colorbar(image, ax=ax, label="Mean pairwise cosine similarity")
    ax.set_title(
        "Brand-level visual similarity of ResNet18 embeddings\n"
        "Diagonal = within-brand mean; off-diagonal = mean cross-brand image-pair cosine",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig05_brand_similarity_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7))
    for brand in BRAND_ORDER:
        mask = brands_array == brand
        ax.scatter(
            pca_2d[mask, 0],
            pca_2d[mask, 1],
            s=26,
            color=BRAND_COLORS[brand],
            label=BRAND_DISPLAY[brand],
            alpha=0.75,
            linewidths=0.3,
            edgecolors="white",
        )
        centroid = pca_2d[mask].mean(axis=0)
        ax.scatter(
            centroid[0],
            centroid[1],
            marker="X",
            s=170,
            color=BRAND_COLORS[brand],
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
        ax.annotate(
            BRAND_DISPLAY[brand],
            centroid,
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
        )
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
    ax.set_title("PCA projection of 512-d ResNet18 embeddings")
    ax.legend(markerscale=1.2, fontsize=7, loc="best", frameon=True, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig06_pca_embeddings.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(min(pair_values_array.min(), -0.2), 1.0, 70)
    ax.hist(between_values, bins=bins, color="#9aa6b2", alpha=0.75, density=True, label="Between-brand pairs")
    ax.hist(within_values, bins=bins, color="#b2182b", alpha=0.65, density=True, label="Within-brand pairs")
    ax.axvline(between_values.mean(), color="#4d5966", linestyle="--", linewidth=1.2)
    ax.axvline(within_values.mean(), color="#7f0000", linestyle="--", linewidth=1.2)
    ax.text(
        within_values.mean(),
        ax.get_ylim()[1] * 0.9,
        f" within mean {within_values.mean():.3f}",
        color="#7f0000",
        fontsize=8,
    )
    ax.text(
        between_values.mean(),
        ax.get_ylim()[1] * 0.78,
        f" between mean {between_values.mean():.3f}",
        color="#4d5966",
        fontsize=8,
    )
    ax.set_xlabel("Cosine similarity of 512-d feature vectors")
    ax.set_ylabel("Density")
    ax.set_title("Within-brand vs between-brand image similarity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig07_within_vs_between.png")
    plt.close(fig)

    print("Creating annotated example sheet ...")
    selection = []
    for brand in BRAND_ORDER:
        group = predictions[predictions["brand"] == brand].sort_values("top1_prob")
        selection.append(group.iloc[-1])
        selection.append(group.iloc[0])
    ncols = 4
    nrows = int(np.ceil(len(selection) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.1 * nrows))
    axes = axes.ravel()
    for ax, row in zip(axes, selection):
        path = args.dataset_root.resolve() / row["brand"] / row["filename"]
        with Image.open(path) as image:
            display = image.convert("RGB")
            display.thumbnail((600, 600))
            ax.imshow(display)
        ax.axis("off")
        label = (
            f"{row['brand_display']} | {row['filename']}\n"
            f"top-1: {row['top1_class']} ({row['top1_prob']:.2f})\n"
            f"top-2: {row['top2_class']} ({row['top2_prob']:.2f})"
        )
        ax.set_title(label, fontsize=8, loc="left")
    for ax in axes[len(selection) :]:
        ax.axis("off")
    fig.suptitle(
        "Example predictions: highest- and lowest-confidence image per brand",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(figures_dir / "fig08_example_predictions.png")
    plt.close(fig)

    top5_classes = (
        top5_counts.groupby("class_name")["image_count_top5"].sum().sort_values(ascending=False)
    )
    top5_heat_classes = top5_classes.head(args.top_n_classes).index.tolist()
    top5_heat = []
    for brand in BRAND_ORDER:
        row = {"brand": brand, "brand_display": BRAND_DISPLAY[brand]}
        values = top5_counts[top5_counts["brand"] == brand].set_index("class_name")["image_count_top5"]
        for class_name in top5_heat_classes:
            row[class_name] = int(values.get(class_name, 0))
        top5_heat.append(row)
    top5_heat_df = (
        pd.DataFrame(top5_heat)
        .set_index("brand_display")
        .drop(columns=["brand"])
        .reindex(brand_labels)
    )
    fig, ax = plt.subplots(figsize=(11, 6.2))
    image = ax.imshow(top5_heat_df.to_numpy() / 40.0, aspect="auto", cmap="Purples", vmin=0)
    ax.set_xticks(range(len(top5_heat_classes)))
    ax.set_xticklabels(top5_heat_classes, rotation=45, ha="right")
    ax.set_yticks(range(len(brand_labels)))
    ax.set_yticklabels(brand_labels)
    for i in range(top5_heat_df.shape[0]):
        for j in range(top5_heat_df.shape[1]):
            value = top5_heat_df.to_numpy()[i, j] / 40.0
            if value > 0:
                ax.text(
                    j,
                    i,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.4 else "black",
                )
    fig.colorbar(image, ax=ax, label="Share of brand images with class in top-5")
    ax.set_title("Top-5 prediction context by brand")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig09_top5_class_heatmap.png")
    plt.close(fig)

    print(json.dumps(analysis_summary, indent=2))
    print(f"Saved analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
