from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VISION_SOURCE = Path(__file__).with_name("vision_ocr.swift")
VISION_BINARY = ROOT / "output" / "stage3" / "bin" / "vision_ocr"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class OCRResult:
    text: str
    method: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text.strip()) and self.error is None


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _ensure_vision_binary() -> tuple[Path | None, str | None]:
    if not shutil.which("swiftc"):
        return None, "swiftc is not available, so the macOS Vision OCR helper cannot be built."
    if not VISION_SOURCE.exists():
        return None, f"Vision OCR source is missing: {VISION_SOURCE}"
    try:
        needs_build = (
            not VISION_BINARY.exists()
            or VISION_BINARY.stat().st_mtime < VISION_SOURCE.stat().st_mtime
        )
        if needs_build:
            VISION_BINARY.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    "swiftc",
                    "-O",
                    str(VISION_SOURCE),
                    "-o",
                    str(VISION_BINARY),
                    "-framework",
                    "Vision",
                    "-framework",
                    "ImageIO",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                return None, f"Could not build the macOS Vision OCR helper: {detail[-500:]}"
        return VISION_BINARY, None
    except subprocess.TimeoutExpired:
        return None, "Building the macOS Vision OCR helper timed out."
    except OSError as exc:
        return None, f"Could not prepare the macOS Vision OCR helper: {exc}"


def _run_vision(path: Path) -> OCRResult:
    binary, error = _ensure_vision_binary()
    if error:
        return OCRResult("", "macOS Vision", error)
    try:
        result = subprocess.run(
            [str(binary), str(path)],
            capture_output=True,
            text=True,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        return OCRResult("", "macOS Vision", "OCR timed out after 45 seconds.")
    except OSError as exc:
        return OCRResult("", "macOS Vision", f"Could not run Vision OCR: {exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return OCRResult("", "macOS Vision", f"Vision OCR failed: {detail[-500:]}")
    text = _clean_text(result.stdout)
    if not text:
        return OCRResult("", "macOS Vision", "No readable text was detected in the image.")
    return OCRResult(text, "macOS Vision")


def _run_tesseract(path: Path) -> OCRResult:
    binary = shutil.which("tesseract")
    if not binary:
        return OCRResult("", "Tesseract", "Tesseract is not installed.")
    try:
        result = subprocess.run(
            [binary, str(path), "stdout", "-l", "eng", "--psm", "6"],
            capture_output=True,
            text=True,
            timeout=45,
        )
    except subprocess.TimeoutExpired:
        return OCRResult("", "Tesseract", "OCR timed out after 45 seconds.")
    except OSError as exc:
        return OCRResult("", "Tesseract", f"Could not run Tesseract: {exc}")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return OCRResult("", "Tesseract", f"Tesseract OCR failed: {detail[-500:]}")
    text = _clean_text(result.stdout)
    if not text:
        return OCRResult("", "Tesseract", "No readable text was detected in the image.")
    return OCRResult(text, "Tesseract")


def ocr_image(path: str | Path) -> OCRResult:
    image_path = Path(path)
    if not image_path.exists():
        return OCRResult("", "none", f"Image not found: {image_path}")
    if image_path.suffix.lower() not in ALLOWED_SUFFIXES:
        return OCRResult("", "none", "Unsupported image type. Use PNG, JPG, or JPEG.")

    if shutil.which("tesseract"):
        result = _run_tesseract(image_path)
        if result.ok:
            return result
        vision_result = _run_vision(image_path)
        if vision_result.ok:
            return vision_result
        return OCRResult(
            "",
            "Tesseract and macOS Vision",
            f"Tesseract: {result.error} Vision: {vision_result.error}",
        )
    return _run_vision(image_path)


def ocr_bytes(data: bytes, suffix: str) -> OCRResult:
    suffix = suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return OCRResult("", "none", "Unsupported image type. Use PNG, JPG, or JPEG.")
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        handle.write(data)
        handle.flush()
        handle.close()
        return ocr_image(handle.name)
    finally:
        try:
            Path(handle.name).unlink(missing_ok=True)
        except OSError:
            pass
