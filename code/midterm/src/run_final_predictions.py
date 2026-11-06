from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from rag.corpus import load_documents
from rag.fields import extract_fields, extract_query_signals
from rag.system import RetrievalSystem

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "New Folder With Items" / "test_questions.json"
DOC_DIR = ROOT / "New Folder With Items" / "documents"
METRICS_PATH = ROOT / "output" / "stage2" / "metrics.json"
FINAL_DIR = ROOT / "output" / "final"
FINAL_PATH = FINAL_DIR / "test_predictions.json"


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best_name = metrics["best_config"]
    config = metrics["best_config_settings"]
    documents = load_documents(DOC_DIR)
    test_questions = load_questions(TEST_PATH)
    document_names = {document.filename for document in documents}
    fields = {document.filename: extract_fields(document) for document in documents}

    system = RetrievalSystem(documents, **config)
    predictions = []
    traces = []
    for question in test_questions:
        result = system.retrieve(question["question"])
        source_docs = list(dict.fromkeys(result["source_docs"]))
        predictions.append(
            {
                "id": question["id"],
                "question": question["question"],
                "source_docs": source_docs,
                "answer": "",
            }
        )
        traces.append(
            {
                "id": question["id"],
                "method": result["method"],
                "intent": result["intent"],
                "source_docs": source_docs,
                "top_results": result.get("top_results", []),
                "detail": result.get("detail", {}),
            }
        )

    FINAL_PATH.write_text(
        json.dumps({"questions": predictions}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (FINAL_DIR / "test_prediction_trace.json").write_text(
        json.dumps(
            {"selected_config": best_name, "config": config, "questions": traces},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    reloaded = json.loads(FINAL_PATH.read_text(encoding="utf-8"))
    output_questions = reloaded.get("questions", [])
    original_by_id = {question["id"]: question for question in test_questions}
    output_ids = [item.get("id") for item in output_questions]
    output_by_id = {item.get("id"): item for item in output_questions}
    trace_by_id = {item["id"]: item for item in traces}

    ids_exactly_once = (
        len(output_ids) == len(set(output_ids))
        and set(output_ids) == set(original_by_id)
        and output_ids == [question["id"] for question in test_questions]
    )
    question_texts_match = all(
        output_by_id.get(question_id, {}).get("question") == original["question"]
        for question_id, original in original_by_id.items()
    )
    source_docs_exist = all(
        isinstance(item.get("source_docs"), list)
        and item["source_docs"]
        and all(filename in document_names for filename in item["source_docs"])
        for item in output_questions
    )
    answer_empty = all(item.get("answer") == "" for item in output_questions)
    no_duplicate_docs = all(
        len(item["source_docs"]) == len(set(item["source_docs"])) for item in output_questions
    )
    exact_schema = all(
        set(item) == {"id", "question", "source_docs", "answer"} for item in output_questions
    )
    generated_by_algorithm = (
        len(traces) == len(test_questions)
        and all(
            trace_by_id[item["id"]]["source_docs"] == item["source_docs"]
            and bool(trace_by_id[item["id"]]["method"])
            for item in output_questions
        )
        and metrics["best_config"] == best_name
    )

    signal_checks = []
    for item in output_questions:
        signals = extract_query_signals(item["question"], list(fields.values()))
        if not signals["invoice_numbers"] and not signals["account_numbers"]:
            continue
        predicted = fields[item["source_docs"][0]]
        ok = True
        if signals["invoice_numbers"]:
            ok = ok and predicted.invoice_number in signals["invoice_numbers"]
        if signals["account_numbers"]:
            ok = ok and predicted.account_number in signals["account_numbers"]
        signal_checks.append({"id": item["id"], "identifier_match": ok})

    checks = {
        "valid_json": isinstance(reloaded, dict) and isinstance(output_questions, list),
        "question_count": len(output_questions),
        "expected_question_count": len(test_questions),
        "question_count_matches": len(output_questions) == len(test_questions),
        "ids_exactly_once": ids_exactly_once,
        "question_texts_match": question_texts_match,
        "source_docs_exist": source_docs_exist,
        "answer_empty": answer_empty,
        "no_duplicate_source_docs": no_duplicate_docs,
        "exact_four_field_schema": exact_schema,
        "generated_by_selected_algorithm": generated_by_algorithm,
        "all_source_docs_nonempty": all(item["source_docs"] for item in output_questions),
        "identifier_signal_consistency": all(item["identifier_match"] for item in signal_checks),
    }
    required = [
        "valid_json",
        "question_count_matches",
        "ids_exactly_once",
        "question_texts_match",
        "source_docs_exist",
        "answer_empty",
        "no_duplicate_source_docs",
        "exact_four_field_schema",
        "generated_by_selected_algorithm",
        "all_source_docs_nonempty",
    ]
    if not all(checks[key] for key in required):
        raise RuntimeError(f"final prediction validation failed: {checks}")

    method_counts = Counter(trace["method"] for trace in traces)
    intent_counts = Counter(trace["intent"].get("kind") for trace in traces)
    report = {
        "selected_config": best_name,
        "config": config,
        "validation_macro_f1": metrics["configs"][best_name]["macro_f1"],
        "validation_exact_matches": metrics["configs"][best_name]["exact_matches"],
        "checks": checks,
        "identifier_signal_checks": signal_checks,
        "method_counts": dict(sorted(method_counts.items())),
        "intent_counts": dict(sorted(intent_counts.items())),
        "document_count": len(documents),
    }
    (FINAL_DIR / "test_predictions_checks.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print(f"wrote {FINAL_PATH}")


if __name__ == "__main__":
    main()
