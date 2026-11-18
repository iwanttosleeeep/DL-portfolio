#!/usr/bin/env python3
"""Run pretrained ResNet18 inference on the fixed 400-image Lane Crawford dataset.

The script performs inference only (no training or fine-tuning). It records the
top-5 ImageNet-1K predictions, the 512-dimensional penultimate feature vector,
and the full 1000-class logits for every image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import PIL
import sklearn
import torch
import torchvision
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18

matplotlib.use("Agg")

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(path: Path) -> tuple:
    stem = path.stem
    prefix, _, number = stem.rpartition("_")
    return (prefix, int(number) if number.isdigit() else 0, stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/resnet"))
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--batch", type=int, default=32)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "mps"}:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if requested == "mps":
            raise RuntimeError("MPS was requested but is not available")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    brands = sorted(
        p.name for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    image_paths: dict[str, list[Path]] = {}
    for brand in brands:
        paths = sorted(dataset_root.joinpath(brand).glob("*.jpg"), key=natural_key)
        if len(paths) != 40:
            raise RuntimeError(f"{brand} has {len(paths)} JPEG images, expected 40")
        image_paths[brand] = paths
    total_images = sum(len(paths) for paths in image_paths.values())
    if len(brands) != 10 or total_images != 400:
        raise RuntimeError(f"Expected 10 brands and 400 images, found {len(brands)} and {total_images}")

    weights = ResNet18_Weights.IMAGENET1K_V1
    class_names = list(weights.meta["categories"])
    transform = weights.transforms()

    device = select_device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)

    print(f"Loading torchvision ResNet18 with {weights} on {device} ...")
    model = resnet18(weights=weights).eval()
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1])).eval()
    model = model.to(device)
    feature_extractor = feature_extractor.to(device)

    parameter_count = int(sum(p.numel() for p in model.parameters()))
    if parameter_count != 11_689_512:
        raise RuntimeError(f"Unexpected ResNet18 parameter count: {parameter_count}")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "total_images": total_images,
        "brands": {brand: len(image_paths[brand]) for brand in brands},
        "model_family": "torchvision ResNet18",
        "architecture": "resnet18",
        "weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "weights_url": weights.url,
        "weights_local_path": str(Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]),
        "weights_sha256": sha256(Path(torch.hub.get_dir()) / "checkpoints" / weights.url.rsplit("/", 1)[-1]),
        "parameter_count": parameter_count,
        "class_count": len(class_names),
        "task": "ImageNet-1K image classification (1000 general categories)",
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pillow_version": PIL.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "inference_settings": {
            "batch_size": args.batch,
            "precision": "FP32",
            "mode": "eval + torch.inference_mode",
            "training": False,
            "fine_tuning": False,
            "augmentation": False,
            "softmax_dim": 1,
            "top_k": 5,
            "feature_layer": "penultimate representation after global average pooling, before fc",
            "feature_dimension": 512,
            "preprocessing": {
                "input_mode": "RGB",
                "resize_size": 256,
                "crop_size": 224,
                "interpolation": "bilinear",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "source": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
            },
        },
    }
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "imagenet_classes.json").open("w", encoding="utf-8") as handle:
        json.dump({str(i): name for i, name in enumerate(class_names)}, handle, indent=2)
        handle.write("\n")

    records = [
        (brand, path, f"{brand}/{path.name}") for brand in brands for path in image_paths[brand]
    ]

    manifest_rows: list[dict] = []
    for brand, path, image_id in records:
        with Image.open(path) as image:
            width, height = image.size
        manifest_rows.append(
            {
                "brand": brand,
                "filename": path.name,
                "image_id": image_id,
                "size_bytes": path.stat().st_size,
                "width": width,
                "height": height,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(data_dir / "image_manifest_sha256.csv", index=False)
    dimensions = {row["image_id"]: (row["width"], row["height"]) for row in manifest_rows}

    all_features: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    image_rows: list[dict] = []
    run_started = time.perf_counter()
    inference_seconds = 0.0

    for start in range(0, len(records), args.batch):
        batch_records = records[start : start + args.batch]
        tensors = []
        for _, path, _ in batch_records:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            tensors.append(transform(rgb))
        batch = torch.stack(tensors).to(device)

        if device.type == "mps":
            torch.mps.synchronize()
        batch_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(batch)
            features = feature_extractor(batch).flatten(1)
        if device.type == "mps":
            torch.mps.synchronize()
        inference_seconds += time.perf_counter() - batch_started

        logits_cpu = logits.detach().to("cpu", dtype=torch.float32)
        features_cpu = features.detach().to("cpu", dtype=torch.float32)
        probs = torch.softmax(logits_cpu, dim=1)
        top_probs, top_ids = torch.topk(probs, k=5, dim=1)

        all_features.append(features_cpu.numpy())
        all_logits.append(logits_cpu.numpy())
        all_probs.append(probs.numpy())

        for row_index, (brand, path, image_id) in enumerate(batch_records):
            prob_row = probs[row_index]
            entropy_nats = float(-(prob_row * torch.log(prob_row.clamp_min(1e-12))).sum())
            ids = [int(value) for value in top_ids[row_index].tolist()]
            values = [float(value) for value in top_probs[row_index].tolist()]
            image_rows.append(
                {
                    "brand": brand,
                    "brand_display": BRAND_DISPLAY.get(brand, brand),
                    "filename": path.name,
                    "image_id": image_id,
                    "image_width": dimensions[image_id][0],
                    "image_height": dimensions[image_id][1],
                    "top1_class_id": ids[0],
                    "top1_class": class_names[ids[0]],
                    "top1_prob": round(values[0], 8),
                    "top2_class_id": ids[1],
                    "top2_class": class_names[ids[1]],
                    "top2_prob": round(values[1], 8),
                    "top3_class_id": ids[2],
                    "top3_class": class_names[ids[2]],
                    "top3_prob": round(values[2], 8),
                    "top4_class_id": ids[3],
                    "top4_class": class_names[ids[3]],
                    "top4_prob": round(values[3], 8),
                    "top5_class_id": ids[4],
                    "top5_class": class_names[ids[4]],
                    "top5_prob": round(values[4], 8),
                    "top5_class_ids": ";".join(str(value) for value in ids),
                    "top5_classes": ";".join(class_names[value] for value in ids),
                    "top5_probs": ";".join(f"{value:.8f}" for value in values),
                    "top5_prob_sum": round(sum(values), 8),
                    "top1_margin": round(values[0] - values[1], 8),
                    "prediction_entropy_nats": round(entropy_nats, 8),
                    "feature_l2_norm": round(float(torch.linalg.vector_norm(features_cpu[row_index])), 8),
                }
            )
        print(f"processed {min(start + args.batch, len(records))}/{len(records)} images", flush=True)

    features_array = np.concatenate(all_features, axis=0).astype(np.float32)
    logits_array = np.concatenate(all_logits, axis=0).astype(np.float32)
    probs_array = np.concatenate(all_probs, axis=0).astype(np.float32)
    image_ids = np.array([image_id for _, _, image_id in records])
    brands_array = np.array([brand for brand, _, _ in records])
    filenames_array = np.array([path.name for _, path, _ in records])

    np.savez_compressed(
        data_dir / "image_embeddings.npz",
        features=features_array,
        logits=logits_array,
        probabilities=probs_array,
        image_ids=image_ids,
        brands=brands_array,
        filenames=filenames_array,
        class_names=np.array(class_names),
    )

    predictions = pd.DataFrame(image_rows)
    predictions.to_csv(data_dir / "image_predictions.csv", index=False)

    elapsed = time.perf_counter() - run_started
    summary = {
        "processed_images": len(predictions),
        "brands": len(brands),
        "features_shape": list(features_array.shape),
        "logits_shape": list(logits_array.shape),
        "probabilities_shape": list(probs_array.shape),
        "elapsed_seconds_total": round(elapsed, 3),
        "inference_seconds": round(inference_seconds, 3),
        "images_per_second_inference": round(len(records) / inference_seconds, 3),
        "device": str(device),
    }
    with (data_dir / "inference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
