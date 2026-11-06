from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rag.corpus import load_documents
from rag.system import RetrievalSystem

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "New Folder With Items" / "documents"
QA_PATH = ROOT / "New Folder With Items" / "qa_pairs.json"
EVALUATOR = ROOT / "New Folder With Items" / "evaluate_retrieval_f1.py"
OUTDIR = ROOT / "output" / "stage2"
PRED_DIR = OUTDIR / "eval_predictions"

CONFIGS = {
    "A_baseline": {"use_hybrid": False, "use_intent": False},
    "B_hybrid_field_rerank": {"use_hybrid": True, "use_intent": False},
    "C_intent_routing": {"use_hybrid": False, "use_intent": True},
    "D_hybrid_plus_intent": {"use_hybrid": True, "use_intent": True},
}
TIE_BREAK = {"D_hybrid_plus_intent": 0, "C_intent_routing": 1, "B_hybrid_field_rerank": 2, "A_baseline": 3}


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def question_category(question: dict) -> str:
    truth = question["source_docs"]
    if len(truth) > 1:
        return "aggregate_multi_doc"
    text = question["question"].casefold()
    if text.startswith(("which customer", "which account holder", "which company")):
        return "aggregate_global"
    stem = Path(truth[0]).stem
    for prefix, label in (
        ("credit-card-bill", "credit_card_bill"),
        ("invoice", "invoice"),
        ("bank-statement", "bank_statement"),
        ("utility-bill", "utility_bill"),
        ("pharmacy-receipt", "pharmacy_receipt"),
        ("receipt", "receipt"),
        ("policy-", "policy"),
    ):
        if stem.startswith(prefix):
            return label
    return "unknown"


def score_sets(pred: set[str], truth: set[str]) -> tuple[float, float, float]:
    correct = pred & truth
    precision = len(correct) / len(pred) if pred else 0.0
    recall = len(correct) / len(truth)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "macro_precision": sum(row["precision"] for row in rows) / n,
        "macro_recall": sum(row["recall"] for row in rows) / n,
        "macro_f1": sum(row["f1"] for row in rows) / n,
        "exact_matches": sum(1 for row in rows if row["exact"]),
        "imperfect": sum(1 for row in rows if not row["exact"]),
        "avg_returned_docs": sum(len(row["pred"]) for row in rows) / n,
    }


def category_metrics(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    result = {}
    for category, items in sorted(grouped.items()):
        n = len(items)
        result[category] = {
            "n": n,
            "macro_precision": sum(row["precision"] for row in items) / n,
            "macro_recall": sum(row["recall"] for row in items) / n,
            "macro_f1": sum(row["f1"] for row in items) / n,
            "exact_matches": sum(1 for row in items if row["exact"]),
        }
    return result


def run_official(truth: Path, pred: Path, log_path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(EVALUATOR), "--truth", str(truth), "--pred", str(pred), "--verbose"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"official evaluator failed for {pred.name}: {result.stderr}")
    output = result.stdout
    return {
        "scored": int(re.search(r"scored\s+: (\d+)", output).group(1)),
        "precision": float(re.search(r"macro precision\s+: ([\d.]+)", output).group(1)),
        "recall": float(re.search(r"macro recall\s+: ([\d.]+)", output).group(1)),
        "f1": float(re.search(r"macro F1\s+: ([\d.]+)", output).group(1)),
    }


def plot_comparison(metrics: dict, outdir: Path) -> None:
    names = list(CONFIGS)
    f1 = [metrics[name]["macro_f1"] for name in names]
    precision = [metrics[name]["macro_precision"] for name in names]
    recall = [metrics[name]["macro_recall"] for name in names]
    exact = [metrics[name]["exact_matches"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = list(range(len(names)))
    width = 0.25
    axes[0].bar([v - width for v in x], precision, width, label="Precision")
    axes[0].bar(x, recall, width, label="Recall")
    axes[0].bar([v + width for v in x], f1, width, label="F1")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Macro score")
    axes[0].set_title("Validation retrieval comparison")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()
    axes[1].bar(names, exact, color="#3b6ea5")
    axes[1].set_ylim(0, 125)
    axes[1].set_ylabel("Exact-set matches (max 125)")
    axes[1].set_title("Exact-set matches by configuration")
    axes[1].tick_params(axis="x", rotation=15)
    for index, value in enumerate(exact):
        axes[1].text(index, value + 1.5, str(value), ha="center", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "comparison.png", dpi=180)
    plt.close(fig)

    categories = list(metrics[names[0]]["by_category"])
    short = {
        "credit_card_bill": "credit card",
        "invoice": "invoice",
        "receipt": "receipt",
        "bank_statement": "bank",
        "utility_bill": "utility",
        "pharmacy_receipt": "pharmacy",
        "policy": "policy",
        "aggregate_multi_doc": "multi-doc aggregate",
        "aggregate_global": "global extreme",
    }
    fig, ax = plt.subplots(figsize=(15, 6))
    x = list(range(len(categories)))
    width = 0.2
    for offset, name in enumerate(names):
        values = [metrics[name]["by_category"][category]["macro_f1"] for category in categories]
        ax.bar([v + (offset - 1.5) * width for v in x], values, width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([short.get(category, category) for category in categories], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("Validation macro F1 by question category")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "per_category.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_documents(DOC_DIR)
    questions = load_questions(QA_PATH)
    if len(questions) != 125:
        raise RuntimeError(f"expected 125 validation questions, found {len(questions)}")

    metrics = {}
    all_traces = {}
    for name, config in CONFIGS.items():
        system = RetrievalSystem(documents, **config)
        rows = []
        predictions = []
        traces = []
        for question in questions:
            result = system.retrieve(question["question"])
            pred = list(dict.fromkeys(result["source_docs"]))
            truth = question["source_docs"]
            pred_set, truth_set = set(pred), set(truth)
            precision, recall, f1 = score_sets(pred_set, truth_set)
            category = question_category(question)
            rows.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "category": category,
                    "truth": truth,
                    "pred": pred,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "exact": pred_set == truth_set,
                    "method": result["method"],
                }
            )
            predictions.append(
                {"id": question["id"], "question": question["question"], "source_docs": pred}
            )
            traces.append(
                {
                    "id": question["id"],
                    "method": result["method"],
                    "intent": result["intent"],
                    "source_docs": pred,
                }
            )

        pred_path = PRED_DIR / f"{name}.json"
        pred_path.write_text(
            json.dumps({"questions": predictions}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        official = run_official(QA_PATH, pred_path, OUTDIR / f"official_eval_{name}.log")
        summary = macro_metrics(rows)
        if abs(official["f1"] - round(summary["macro_f1"], 4)) > 1e-9:
            raise RuntimeError(
                f"metric mismatch for {name}: own={summary['macro_f1']}, official={official['f1']}"
            )
        if abs(official["precision"] - round(summary["macro_precision"], 4)) > 1e-9:
            raise RuntimeError(f"precision mismatch for {name}")
        if abs(official["recall"] - round(summary["macro_recall"], 4)) > 1e-9:
            raise RuntimeError(f"recall mismatch for {name}")
        if official["scored"] != 125:
            raise RuntimeError(f"official evaluator scored {official['scored']} of 125 for {name}")
        summary.update(
            {
                "official": official,
                "by_category": category_metrics(rows),
                "failures": [row for row in rows if not row["exact"]],
                "predictions_file": str(pred_path.relative_to(ROOT)),
            }
        )
        metrics[name] = summary
        all_traces[name] = traces

    best_name = max(
        CONFIGS,
        key=lambda name: (
            metrics[name]["macro_f1"],
            metrics[name]["macro_precision"],
            metrics[name]["macro_recall"],
            metrics[name]["exact_matches"],
            -TIE_BREAK[name],
        ),
    )

    payload = {
        "validation_questions": len(questions),
        "selection_rule": "max macro F1, then precision, recall, exact matches; preference D>C>B>A only on a complete tie",
        "best_config": best_name,
        "best_config_settings": CONFIGS[best_name],
        "configs": metrics,
    }
    (OUTDIR / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTDIR / "retrieval_traces.json").write_text(
        json.dumps(all_traces, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (OUTDIR / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["configuration", "macro_precision", "macro_recall", "macro_f1", "exact_matches", "imperfect", "avg_docs"]
        )
        for name in CONFIGS:
            item = metrics[name]
            writer.writerow(
                [
                    name,
                    f"{item['macro_precision']:.6f}",
                    f"{item['macro_recall']:.6f}",
                    f"{item['macro_f1']:.6f}",
                    item["exact_matches"],
                    item["imperfect"],
                    f"{item['avg_returned_docs']:.4f}",
                ]
            )

    lines = ["# Stage 2 validation comparison", "", "| Configuration | P | R | F1 | Exact | Imperfect | Avg docs |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in CONFIGS:
        item = metrics[name]
        lines.append(
            f"| {name} | {item['macro_precision']:.4f} | {item['macro_recall']:.4f} | "
            f"{item['macro_f1']:.4f} | {item['exact_matches']} | {item['imperfect']} | "
            f"{item['avg_returned_docs']:.2f} |"
        )
    lines.extend(["", f"Selected configuration: **{best_name}**", ""])
    (OUTDIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    plot_comparison(metrics, OUTDIR)

    error_lines = ["# Stage 2 error analysis", "", "## Summary", ""]
    for name in CONFIGS:
        item = metrics[name]
        error_lines.append(
            f"- **{name}**: F1={item['macro_f1']:.4f}, exact={item['exact_matches']}/125, "
            f"imperfect={item['imperfect']}"
        )
    error_lines.extend(["", "## Per-configuration failures", ""])
    for name in CONFIGS:
        failures = metrics[name]["failures"]
        error_lines.append(f"### {name} ({len(failures)} failures)")
        if not failures:
            error_lines.append("No failures on the 125 validation questions.")
        for row in failures:
            error_lines.append(
                f"- `{row['id']}` [{row['category']}] F1={row['f1']:.3f} | {row['question']}"
            )
            error_lines.append(f"  - truth: {row['truth']}")
            error_lines.append(f"  - predicted: {row['pred']}")
            error_lines.append(f"  - method: `{row['method']}`")
        error_lines.append("")

    error_lines.extend(["", "## Improvement effects", ""])
    failure_ids = {name: {row["id"] for row in metrics[name]["failures"]} for name in CONFIGS}
    for base, candidate in (
        ("A_baseline", "B_hybrid_field_rerank"),
        ("A_baseline", "C_intent_routing"),
        ("C_intent_routing", "D_hybrid_plus_intent"),
    ):
        fixed = sorted(failure_ids[base] - failure_ids[candidate])
        regressed = sorted(failure_ids[candidate] - failure_ids[base])
        delta = metrics[candidate]["macro_f1"] - metrics[base]["macro_f1"]
        error_lines.append(
            f"- **{base} -> {candidate}**: F1 {delta:+.4f}; exact "
            f"{metrics[base]['exact_matches']} -> {metrics[candidate]['exact_matches']}; "
            f"fixed={fixed or 'none'}; regressed={regressed or 'none'}"
        )
    error_lines.extend(
        [
            "",
            "Improvement 1 is a modest independent gain: it fixes `q101` and `q113` but",
            "regresses `q117`, for a net +1 exact match and +0.0080 macro F1. All 100",
            "structured single-document questions remain correct.",
            "",
            "Improvement 2 fixes every policy and dataset-level failure, reaching 125/125.",
            "Configuration D ties C on validation F1, so Improvement 1 is neutral when",
            "combined with intent routing rather than additive on this validation set.",
            "",
            "No validation failure remains for C or D. This is a validation ceiling and",
            "does not imply that unseen test questions will be perfect.",
        ]
    )
    (OUTDIR / "error_analysis.md").write_text("\n".join(error_lines), encoding="utf-8")

    print(json.dumps({name: {k: metrics[name][k] for k in ("macro_precision", "macro_recall", "macro_f1", "exact_matches", "imperfect")} for name in CONFIGS}, indent=2))
    print(f"best configuration: {best_name}")


if __name__ == "__main__":
    main()
