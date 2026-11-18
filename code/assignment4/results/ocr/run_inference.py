#!/usr/bin/env python3
"""Run pretrained EasyOCR inference on the fixed 400-image Lane Crawford dataset.

The script is inference-only. It writes one raw JSONL record per image, a dataset
manifest with SHA-256 hashes, and model/inference metadata. Image-level and
brand-level tables are produced by analyze_results.py from the raw records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import cv2
import certifi

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import easyocr  # noqa: E402  (certificate path must be set before model download)

from ocr_common import BRANDS, BRAND_ORDER, parse_bbox, tokenize  # noqa: E402

MODEL_FILENAMES = {
    "detector": "craft_mlt_25k.pth",
    "english_recognizer": "english_g2.pth",
}

DETECTOR_SETTINGS = {
    "min_size": 10,
    "text_threshold": 0.7,
    "low_text": 0.4,
    "link_threshold": 0.4,
    "canvas_size": 2560,
    "mag_ratio": 1.0,
    "slope_ths": 0.1,
    "ycenter_ths": 0.5,
    "height_ths": 0.5,
    "width_ths": 0.5,
    "add_margin": 0.1,
    "threshold": 0.2,
    "bbox_min_score": 0.2,
    "bbox_min_size": 3,
    "max_candidates": 0,
    "rotation_info": None,
}

RECOGNIZER_SETTINGS = {
    "decoder": "greedy",
    "beamWidth": 5,
    "batch_size": 1,
    "workers": 0,
    "allowlist": None,
    "blocklist": None,
    "paragraph": False,
    "contrast_ths": 0.1,
    "adjust_contrast": 0.5,
    "filter_ths": 0.003,
}


class ImageTimeout(Exception):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise ImageTimeout("per-image OCR timeout")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_key(path: Path) -> tuple:
    prefix, _, number = path.stem.rpartition("_")
    return (prefix, int(number) if number.isdigit() else 0, path.stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ocr"))
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--langs", nargs="+", default=["en"])
    parser.add_argument("--conf-threshold", type=float, default=0.30)
    parser.add_argument("--image-timeout", type=int, default=300)
    parser.add_argument("--fallback-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ocr_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "logs" / "run.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def load_resume_records(path: Path, fingerprint: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("config_fingerprint") != fingerprint:
                continue
            if record.get("error"):
                continue
            records[record["image_id"]] = record
    return records


def build_reader(langs: list[str], device: str, logger: logging.Logger):
    logger.info("Loading EasyOCR pretrained reader (langs=%s, device=%s)", langs, device)
    started = time.perf_counter()
    reader = easyocr.Reader(langs, gpu=device, verbose=False)
    logger.info("Reader loaded in %.2f s; resolved device=%s", time.perf_counter() - started, reader.device)
    return reader


def run_image(reader, path: Path, brand: str, threshold: float, timeout: int) -> dict:
    started = time.perf_counter()
    error = ""
    raw = []
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)
    try:
        raw = reader.readtext(
            str(path),
            **DETECTOR_SETTINGS,
            **RECOGNIZER_SETTINGS,
        )
    except ImageTimeout as exc:
        error = f"timeout:{exc}"
    except Exception as exc:  # noqa: BLE001 - record failures for audit
        error = f"{type(exc).__name__}:{exc}"
    finally:
        signal.alarm(0)
        if reader.device == "mps" and torch.backends.mps.is_available():
            # Without this, MPS allocation growth degraded per-image time from
            # roughly 1-4 s to 86 s within 30 images during the pilot run.
            torch.mps.empty_cache()
    elapsed = time.perf_counter() - started

    detections = []
    for region_index, item in enumerate(raw, start=1):
        polygon, text, confidence = item[0], str(item[1]), float(item[2])
        x_min, y_min, x_max, y_max = parse_bbox(polygon)
        kept = confidence >= threshold
        detections.append(
            {
                "region_id": region_index,
                "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon],
                "bbox": [round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)],
                "text": text,
                "confidence": round(confidence, 6),
                "kept": bool(kept),
                "token_count": len(tokenize(text)) if kept else 0,
            }
        )

    return {
        "brand": brand,
        "filename": path.name,
        "image_id": f"{brand}/{path.name}",
        "image_path": str(path),
        "image_sha256": sha256(path),
        "image_width": int(reader_image_width(path)),
        "image_height": int(reader_image_height(path)),
        "elapsed_s": round(elapsed, 3),
        "error": error,
        "raw_region_count": len(detections),
        "detections": detections,
    }


def reader_image_width(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as image:
        return image.size[0]


def reader_image_height(path: Path) -> int:
    from PIL import Image

    with Image.open(path) as image:
        return image.size[1]


def write_progress(path: Path, records: dict[str, dict], order: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for image_id in order:
            record = records.get(image_id)
            if record is not None:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)

    brands = [brand for brand in BRAND_ORDER if (dataset_root / brand).is_dir()]
    if len(brands) != 10:
        raise RuntimeError(f"Expected 10 brand directories, found {len(brands)}: {brands}")

    image_paths: dict[str, list[Path]] = {}
    for brand in brands:
        paths = sorted((dataset_root / brand).glob("*.jpg"), key=natural_key)
        if len(paths) != 40:
            raise RuntimeError(f"{brand} has {len(paths)} JPEG images, expected 40")
        image_paths[brand] = paths
    total_images = sum(len(paths) for paths in image_paths.values())
    if total_images != 400:
        raise RuntimeError(f"Dataset contains {total_images} images, expected 400")
    if args.device == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but unavailable; using CPU")
        args.device = "cpu"

    config = {
        "model_family": "EasyOCR",
        "model_version": easyocr.__version__,
        "languages": args.langs,
        "device": args.device,
        "confidence_threshold": args.conf_threshold,
        "detector": DETECTOR_SETTINGS,
        "recognizer": RECOGNIZER_SETTINGS,
        "tokenization": "maximal ASCII alphanumeric run, internal apostrophe allowed",
    }
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    manifest_rows = []
    for brand in brands:
        for path in image_paths[brand]:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
                mode = image.mode
            manifest_rows.append(
                {
                    "brand": brand,
                    "brand_display": BRANDS[brand],
                    "filename": path.name,
                    "image_id": f"{brand}/{path.name}",
                    "path": str(path),
                    "file_size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(data_dir / "dataset_manifest.csv", index=False)
    logger.info("Dataset manifest written: %d images, %d unique SHA-256 values", len(manifest), manifest["sha256"].nunique())

    partial_path = data_dir / "ocr_raw.partial.jsonl"
    records = load_resume_records(partial_path, fingerprint) if args.resume else {}
    if records:
        logger.info("Resuming from %d completed image records", len(records))

    ordered_ids = [f"{brand}/{path.name}" for brand in brands for path in image_paths[brand]]
    reader = build_reader(args.langs, args.device, logger)

    run_started = time.perf_counter()
    for brand in brands:
        brand_started = time.perf_counter()
        brand_done = 0
        brand_failed = 0
        for path in image_paths[brand]:
            image_id = f"{brand}/{path.name}"
            if image_id in records:
                brand_done += 1
                continue
            record = run_image(reader, path, brand, args.conf_threshold, args.image_timeout)
            record["config_fingerprint"] = fingerprint
            records[image_id] = record
            if record["error"]:
                brand_failed += 1
                logger.error("FAILED %s | %s", image_id, record["error"])
            brand_done += 1
            if brand_done % 10 == 0 or record["error"]:
                write_progress(partial_path, records, ordered_ids)
                logger.info(
                    "%s: %d/%d images | last=%.2fs, %d raw regions",
                    brand,
                    brand_done,
                    len(image_paths[brand]),
                    record["elapsed_s"],
                    record["raw_region_count"],
                )
        write_progress(partial_path, records, ordered_ids)
        logger.info(
            "%s complete: %d images, %d failures, %.1fs",
            brand,
            brand_done,
            brand_failed,
            time.perf_counter() - brand_started,
        )

    failed_ids = [image_id for image_id in ordered_ids if records.get(image_id, {}).get("error")]
    if failed_ids and args.fallback_cpu:
        logger.warning("Retrying %d failed images on CPU", len(failed_ids))
        cpu_reader = build_reader(args.langs, "cpu", logger)
        for image_id in failed_ids:
            brand, filename = image_id.split("/", 1)
            path = dataset_root / brand / filename
            record = run_image(cpu_reader, path, brand, args.conf_threshold, args.image_timeout)
            record["config_fingerprint"] = fingerprint
            record["device"] = "cpu_fallback"
            records[image_id] = record
            write_progress(partial_path, records, ordered_ids)
            logger.info("CPU retry %s | error=%s | %d regions", image_id, record["error"], record["raw_region_count"])

    missing = [image_id for image_id in ordered_ids if image_id not in records]
    failed = [image_id for image_id in ordered_ids if records.get(image_id, {}).get("error")]
    if missing or failed:
        raise RuntimeError(f"Incomplete OCR run: {len(missing)} missing, {len(failed)} failed")

    raw_path = data_dir / "ocr_raw.jsonl"
    write_progress(raw_path, records, ordered_ids)
    if partial_path.exists():
        partial_path.unlink()

    model_dir = Path.home() / ".EasyOCR" / "model"
    model_files = {}
    for role, filename in MODEL_FILENAMES.items():
        path = model_dir / filename
        model_files[role] = {
            "filename": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256(path) if path.exists() else None,
        }

    elapsed_total = time.perf_counter() - run_started
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "total_images": total_images,
        "brands": {brand: len(image_paths[brand]) for brand in brands},
        "task": "optical character recognition (pretrained inference only)",
        "model_family": "EasyOCR",
        "model_architecture": "CRAFT text detector + english_g2 recognition network",
        "easyocr_version": easyocr.__version__,
        "pretrained_weights": True,
        "training_or_fine_tuning": "none",
        "languages": args.langs,
        "language_note": (
            "English-only recognition was used for the final run. EasyOCR also ships "
            "Chinese simplified/traditional recognition models, but the bilingual pilot "
            "changed English predictions and was not used for the frozen final run."
        ),
        "model_files": model_files,
        "model_storage_directory": str(model_dir),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "device_requested": args.device,
        "device_used": "mps" if args.device == "mps" else "cpu",
        "device_note": (
            "Apple Silicon MPS was used. A pilot without empty_cache() degraded from "
            "about 1-4 s per image to 16-86 s within 30 images and was aborted; "
            "torch.mps.empty_cache() after every image restored stable 1-5 s inference."
        ),
        "mps_empty_cache_after_each_image": True,
        "confidence_threshold": args.conf_threshold,
        "confidence_threshold_definition": (
            "Regions with recognition confidence < threshold remain in raw detections "
            "but are excluded from image-level has_text, region, and token counts."
        ),
        "detector_settings": DETECTOR_SETTINGS,
        "recognizer_settings": RECOGNIZER_SETTINGS,
        "token_definition": (
            "Maximal ASCII alphanumeric run, optionally joined by an internal apostrophe; "
            "Chinese characters are not counted as English/alphanumeric tokens."
        ),
        "per_image_timeout_seconds": args.image_timeout,
        "cpu_fallback_enabled": args.fallback_cpu,
        "config_fingerprint": fingerprint,
        "processed_images": len(ordered_ids),
        "run_elapsed_seconds": round(elapsed_total, 2),
        "images_per_second": round(len(ordered_ids) / elapsed_total, 4),
    }
    with (output_dir / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    logger.info("Wrote %s and model_metadata.json", raw_path)
    logger.info("Processed %d images in %.1f s", len(ordered_ids), elapsed_total)


if __name__ == "__main__":
    main()
