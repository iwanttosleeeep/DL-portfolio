#!/usr/bin/env python3
"""Run pretrained YOLO inference on the fixed 400-image Lane Crawford dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import ultralytics
from PIL import Image
from ultralytics import YOLO

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
    parser.add_argument("--output-dir", type=Path, default=Path("results/yolo"))
    parser.add_argument("--weights", type=Path, default=Path("results/yolo/weights/yolo11n.pt"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    weights = args.weights.resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    brands = sorted([p.name for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")])
    image_paths: dict[str, list[Path]] = {}
    for brand in brands:
        paths = sorted(dataset_root.joinpath(brand).glob("*.jpg"), key=natural_key)
        if len(paths) != 40:
            raise RuntimeError(f"{brand} has {len(paths)} JPEG images, expected 40")
        image_paths[brand] = paths

    total_images = sum(len(paths) for paths in image_paths.values())
    if total_images != 400:
        raise RuntimeError(f"Dataset contains {total_images} images, expected 400")
    if not weights.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {weights}")

    model = YOLO(str(weights))
    if len(model.names) != 80:
        raise RuntimeError(f"Expected 80 COCO classes, found {len(model.names)}")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "total_images": total_images,
        "brands": {brand: len(image_paths[brand]) for brand in brands},
        "model_family": "Ultralytics YOLO11",
        "weights_file": str(weights),
        "weights_size_bytes": weights.stat().st_size,
        "weights_sha256": sha256(weights),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": args.device,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "task": model.task,
        "parameter_count": int(sum(parameter.numel() for parameter in model.model.parameters())),
        "class_count": len(model.names),
        "class_labels": {int(key): value for key, value in model.names.items()},
        "inference_settings": {
            "imgsz": args.imgsz,
            "conf_threshold": args.conf,
            "iou_threshold": args.iou,
            "max_det": args.max_det,
            "batch": args.batch,
            "half_precision": False,
            "rect": False,
            "agnostic_nms": False,
            "augment": False,
            "classes": "all 80 COCO classes",
            "stream": True,
            "source": "one brand directory per prediction call",
        },
    }
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    image_rows: list[dict] = []
    detection_rows: list[dict] = []
    run_log: list[str] = []
    run_started = time.perf_counter()

    for brand in brands:
        brand_started = time.perf_counter()
        results = model.predict(
            source=str(dataset_root / brand),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            device=args.device,
            batch=args.batch,
            rect=False,
            agnostic_nms=False,
            augment=False,
            stream=True,
            verbose=False,
        )
        brand_rows = 0
        brand_detections = 0
        for result in results:
            path = Path(result.path)
            filename = path.name
            image_id = f"{brand}/{filename}"
            with Image.open(path) as image:
                width, height = image.size
            boxes = result.boxes
            count = int(len(boxes))
            classes = [int(value) for value in boxes.cls.detach().cpu().tolist()]
            confidences = [round(float(value), 6) for value in boxes.conf.detach().cpu().tolist()]
            coordinates = boxes.xyxy.detach().cpu().numpy()
            class_counts = Counter(result.names[class_id] for class_id in classes)
            ordered_counts = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
            unique_classes = sorted(class_counts, key=lambda name: (-class_counts[name], name))
            top_class = ordered_counts[0][0] if ordered_counts else ""
            summary = "; ".join(f"{name} ({number})" for name, number in ordered_counts)

            speed = result.speed if isinstance(result.speed, dict) else {}
            image_rows.append(
                {
                    "brand": brand,
                    "brand_display": BRAND_DISPLAY.get(brand, brand),
                    "filename": filename,
                    "image_id": image_id,
                    "image_width": width,
                    "image_height": height,
                    "object_count": count,
                    "has_detection": int(count > 0),
                    "zero_detection": int(count == 0),
                    "unique_class_count": len(class_counts),
                    "detected_classes": ";".join(unique_classes),
                    "top_class": top_class,
                    "detection_summary": summary,
                    "class_counts_json": json.dumps(dict(sorted(class_counts.items())), sort_keys=True),
                    "max_confidence": round(max(confidences), 6) if confidences else "",
                    "mean_confidence": round(float(np.mean(confidences)), 6) if confidences else "",
                    "speed_preprocess_ms": round(float(speed.get("preprocess", 0.0)), 3),
                    "speed_inference_ms": round(float(speed.get("inference", 0.0)), 3),
                    "speed_postprocess_ms": round(float(speed.get("postprocess", 0.0)), 3),
                }
            )
            for detection_index, (class_id, confidence, box) in enumerate(
                zip(classes, confidences, coordinates, strict=True), start=1
            ):
                x1, y1, x2, y2 = [float(value) for value in box]
                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                detection_rows.append(
                    {
                        "brand": brand,
                        "brand_display": BRAND_DISPLAY.get(brand, brand),
                        "filename": filename,
                        "image_id": image_id,
                        "detection_index": detection_index,
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": round(confidence, 6),
                        "x1": round(x1, 3),
                        "y1": round(y1, 3),
                        "x2": round(x2, 3),
                        "y2": round(y2, 3),
                        "box_width": round(box_width, 3),
                        "box_height": round(box_height, 3),
                        "x_center_norm": round((x1 + x2) / (2 * width), 6),
                        "y_center_norm": round((y1 + y2) / (2 * height), 6),
                        "width_norm": round(box_width / width, 6),
                        "height_norm": round(box_height / height, 6),
                        "area_fraction": round((box_width * box_height) / (width * height), 6),
                        "is_top1": int(detection_index == 1),
                    }
                )
            brand_rows += 1
            brand_detections += count
        elapsed = time.perf_counter() - brand_started
        expected = len(image_paths[brand])
        if brand_rows != expected:
            raise RuntimeError(f"{brand}: received {brand_rows} results, expected {expected}")
        line = (
            f"{brand}: images={brand_rows} detections={brand_detections} "
            f"elapsed_seconds={elapsed:.2f} images_per_second={brand_rows / elapsed:.2f}"
        )
        print(line, flush=True)
        run_log.append(line)

    image_df = pd.DataFrame(image_rows).sort_values(["brand", "filename"], kind="stable")
    detection_df = pd.DataFrame(detection_rows).sort_values(
        ["brand", "filename", "detection_index"], kind="stable"
    )
    if len(image_df) != 400:
        raise RuntimeError(f"Image-level table contains {len(image_df)} rows, expected 400")
    if image_df["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image IDs found in image-level results")
    if set(image_df["brand"].unique()) != set(brands):
        raise RuntimeError("Brand set in image-level results does not match dataset")
    if int(image_df["object_count"].sum()) != len(detection_df):
        raise RuntimeError("Detection table row count does not match summed object counts")

    image_csv = data_dir / "image_level_detections.csv"
    detection_csv = data_dir / "object_detections.csv"
    image_df.to_csv(image_csv, index=False)
    detection_df.to_csv(detection_csv, index=False)

    total_elapsed = time.perf_counter() - run_started
    run_log.append(
        f"total: images={len(image_df)} detections={len(detection_df)} "
        f"elapsed_seconds={total_elapsed:.2f} images_per_second={len(image_df) / total_elapsed:.2f}"
    )
    (output_dir / "logs" / "inference_log.txt").write_text("\n".join(run_log) + "\n", encoding="utf-8")

    print(f"Wrote {image_csv} ({len(image_df)} rows)")
    print(f"Wrote {detection_csv} ({len(detection_df)} rows)")
    print(f"Total detections: {len(detection_df)}")
    print(f"Total elapsed: {total_elapsed:.2f} s")


if __name__ == "__main__":
    main()
