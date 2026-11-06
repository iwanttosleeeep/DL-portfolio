from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import requests

from app.app_logic import maybe_judge_document
from llm_judge import JudgeError, check_ollama, judge_document
from ocr_tools.ocr import ocr_image
from retrieval_service import RetrievalService


SMOKE_DIR = ROOT / "output" / "stage3" / "smoke"
APP = SRC / "app" / "streamlit_app.py"
TEXT_QUERY = "What is the total amount due on Peak Pest Control invoice INV-2024-892?"
EXPECTED_DOC = "invoice-026.md"
RESULTS: dict = {"checks": [], "details": {}}


def record(name: str, passed: bool, detail: object = None) -> bool:
    RESULTS["checks"].append({"name": name, "passed": bool(passed), "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")
    return bool(passed)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_image_query() -> Path:
    text = "\n".join(
        [
            "INVOICE",
            "PEAK PEST CONTROL",
            "Invoice Number: INV-2024-892",
            "Invoice Date: 11/14/2023",
            "Total Amount Due: $2,895.15",
        ]
    )
    font = ImageFont.truetype(
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        28,
    )
    image = Image.new("RGB", (920, 270), "white")
    draw = ImageDraw.Draw(image)
    draw.multiline_text((30, 30), text, fill="black", font=font, spacing=12)
    path = SMOKE_DIR / "image_query.png"
    image.save(path)
    return path


def check_retrieval(service: RetrievalService) -> None:
    top10 = service.search(TEXT_QUERY, top_k=10)
    record(
        "text query returns Top-10",
        len(top10["ranked_details"]) == 10,
        [item["filename"] for item in top10["ranked_details"]],
    )
    record(
        "text query recommends expected document",
        top10["source_docs"] == [EXPECTED_DOC],
        top10["source_docs"],
    )
    record(
        "text query ranks expected document first",
        top10["ranked_details"][0]["filename"] == EXPECTED_DOC,
        top10["ranked_details"][0],
    )
    top20 = service.search(TEXT_QUERY, top_k=20)
    record(
        "text query returns Top-20",
        len(top20["ranked_details"]) == 20,
        len(top20["ranked_details"]),
    )
    record(
        "Top-20 contains expected document",
        EXPECTED_DOC in [item["filename"] for item in top20["ranked_details"]],
        top20["ranked_details"][0],
    )


def check_adaptive_count(service: RetrievalService) -> None:
    query = "What do all the invoices in the dataset add up to when combined?"
    result = service.search(query, top_k=10)
    record(
        "adaptive source_docs count is independent of UI Top-K",
        len(result["ranked_details"]) == 10
        and result["recommended_count"] == 25
        and len(result["source_docs"]) == 25,
        {
            "ui_top_k": len(result["ranked_details"]),
            "recommended_count": result["recommended_count"],
            "method": result["method"],
        },
    )


def check_ocr(service: RetrievalService) -> None:
    image_path = make_image_query()
    ocr = ocr_image(image_path)
    record("OCR succeeds", ocr.ok, {"method": ocr.method, "error": ocr.error})
    record(
        "OCR extracts invoice identifier",
        "INV-2024-892" in ocr.text,
        ocr.text,
    )
    RESULTS["details"]["ocr_text"] = ocr.text
    if ocr.ok:
        ocr_result = service.search(ocr.text, top_k=10)
        record(
            "image OCR query recommends expected document",
            ocr_result["source_docs"] == [EXPECTED_DOC],
            ocr_result["source_docs"],
        )
        record(
            "image OCR query ranks expected document first",
            ocr_result["ranked_details"][0]["filename"] == EXPECTED_DOC,
            ocr_result["ranked_details"][0],
        )
    blank_path = SMOKE_DIR / "blank_image.png"
    Image.new("RGB", (500, 200), "white").save(blank_path)
    blank = ocr_image(blank_path)
    record(
        "OCR failure is reported for a blank image",
        not blank.ok and bool(blank.error),
        {"method": blank.method, "error": blank.error},
    )


def check_judge_disabled(service: RetrievalService) -> None:
    document = service.doc_by_name[EXPECTED_DOC]
    called = {"value": False}

    def should_not_run(**kwargs):
        called["value"] = True
        raise AssertionError("judge was called while disabled")

    result = maybe_judge_document(
        False,
        TEXT_QUERY,
        {"filename": document.filename, "text": document.text},
        judge_fn=should_not_run,
    )
    record(
        "judge is not called while disabled",
        result is None and not called["value"],
        {"result": result, "called": called["value"]},
    )


def check_app() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    record("Streamlit app runs without exceptions", not app.exception, [e.value for e in app.exception])
    app.text_area[0].input(TEXT_QUERY)
    app.button[0].click().run()
    record(
        "Streamlit text query runs without exceptions",
        not app.exception,
        [e.value for e in app.exception],
    )
    record(
        "Streamlit shows 10 ranked cards",
        len(app.expander) == 11,
        len(app.expander),
    )
    record(
        "Streamlit judge button is absent when disabled",
        not any(button.label == "Evaluate Retrieval with Local LLM" for button in app.button),
        [button.label for button in app.button],
    )
    app.select_slider[0].set_value(20).run()
    record(
        "Streamlit Top-20 runs without exceptions",
        not app.exception,
        [e.value for e in app.exception],
    )
    record(
        "Streamlit shows 20 ranked cards",
        len(app.expander) == 21,
        len(app.expander),
    )


def check_server() -> None:
    port = 8517
    log_path = SMOKE_DIR / "streamlit_smoke_server.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(APP),
            "--server.headless=true",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats=false",
        ],
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    healthy = False
    try:
        for _ in range(40):
            if process.poll() is not None:
                break
            try:
                response = requests.get(
                    f"http://localhost:{port}/_stcore/health",
                    timeout=1,
                )
                if response.status_code == 200 and response.text.strip() == "ok":
                    healthy = True
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log_file.close()
    record("Streamlit server health endpoint is ok", healthy, str(log_path))


def check_stage2_and_predictions() -> None:
    predictions = ROOT / "output" / "final" / "test_predictions.json"
    before = sha256(predictions)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(SRC)
    process = subprocess.run(
        [sys.executable, str(SRC / "run_stage2_eval.py")],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    record(
        "Stage 2 evaluation reruns successfully",
        process.returncode == 0,
        process.stdout[-1000:] if process.returncode else "ok",
    )
    metrics = json.loads((ROOT / "output" / "stage2" / "metrics.json").read_text())
    best = metrics["configs"]["D_hybrid_plus_intent"]
    record(
        "Stage 2 validation F1 is unchanged",
        best["macro_f1"] == 1.0 and best["exact_matches"] == 125,
        {"macro_f1": best["macro_f1"], "exact_matches": best["exact_matches"]},
    )
    after = sha256(predictions)
    record(
        "final test predictions are unchanged",
        before == after,
        {"before": before, "after": after},
    )


def check_judge_availability(service: RetrievalService) -> bool:
    document = service.doc_by_name[EXPECTED_DOC]
    try:
        judge_document(
            TEXT_QUERY,
            document.filename,
            document.text,
            host="http://127.0.0.1:1",
            timeout=2.0,
        )
        failure_reported = False
        failure_detail = "no error raised"
    except JudgeError as exc:
        failure_reported = True
        failure_detail = str(exc)
    record(
        "Ollama connection failure is reported",
        failure_reported,
        failure_detail,
    )

    available, message = check_ollama()
    RESULTS["details"]["ollama"] = message
    record("local Ollama judge available", available, message)
    return available


def check_app_judge() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    app.text_area[0].input(TEXT_QUERY)
    app.button[0].click().run()
    app.toggle[0].set_value(True).run()
    judge_button = [
        button
        for button in app.button
        if button.label == "Evaluate Retrieval with Local LLM"
    ]
    record(
        "Streamlit judge button appears when enabled",
        bool(judge_button),
        [button.label for button in app.button],
    )
    if not judge_button:
        return
    judge_button[0].click().run(timeout=180)
    record(
        "Streamlit judge runs without exceptions",
        not app.exception,
        [e.value for e in app.exception],
    )
    successes = [item.value for item in app.success if item.value.startswith("Acceptable:")]
    acceptable = successes[0] if successes else None
    markdown = [item.value for item in app.markdown]
    justification = None
    for index, value in enumerate(markdown):
        if value == "#### Judgment" and index + 1 < len(markdown):
            justification = markdown[index + 1]
            break
    RESULTS["details"]["judge"] = {
        "displayed_decision": acceptable,
        "justification": justification,
    }
    record(
        "Streamlit judge displays an acceptable decision",
        acceptable is not None,
        acceptable,
    )
    record(
        "Streamlit judge displays a justification",
        bool(justification),
        justification,
    )


def main() -> int:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    service = RetrievalService()
    check_retrieval(service)
    check_adaptive_count(service)
    check_ocr(service)
    check_judge_disabled(service)
    check_app()
    check_server()
    check_stage2_and_predictions()
    if check_judge_availability(service):
        check_app_judge()
    RESULTS["all_passed"] = all(item["passed"] for item in RESULTS["checks"])
    path = SMOKE_DIR / "smoke_test_results.json"
    path.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"all_passed={RESULTS['all_passed']} results={path}")
    return 0 if RESULTS["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
