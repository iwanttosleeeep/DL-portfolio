import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rag.corpus import load_documents
from rag.tfidf_baseline import TFIDFRetriever

KS = (1, 3, 5, 10)


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def macro_scores(pred_sets: list[set[str]], truth_sets: list[set[str]]) -> dict:
    precisions, recalls, f1s = [], [], []
    for pred, truth in zip(pred_sets, truth_sets):
        correct = pred & truth
        precision = len(correct) / len(pred) if pred else 0.0
        recall = len(correct) / len(truth) if truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    n = len(f1s)
    return {
        "n": n,
        "macro_precision": sum(precisions) / n,
        "macro_recall": sum(recalls) / n,
        "macro_f1": sum(f1s) / n,
        "perfect": sum(1 for x in f1s if x == 1.0),
        "imperfect": sum(1 for x in f1s if x < 1.0),
    }


def question_category(q: dict) -> str:
    truth = q["source_docs"]
    if len(truth) > 1:
        return "aggregate_multi_doc"
    text = q["question"].lower()
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


def category_table(rows: list[dict], key: str) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    table = {}
    for cat, items in grouped.items():
        table[cat] = {
            "n": len(items),
            "exact_set_match": sum(1 for x in items if x["correct_top1"]),
            "f1": sum(x[key] for x in items) / len(items),
            "mrr": sum(1 for x in items if x["best_truth_rank"] == 1) / len(items),
        }
    return dict(sorted(table.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-dir", type=Path, default=Path("New Folder With Items/documents"))
    parser.add_argument("--qa", type=Path, default=Path("New Folder With Items/qa_pairs.json"))
    parser.add_argument("--outdir", type=Path, default=Path("output/stage1"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    docs = load_documents(args.doc_dir)
    questions = load_questions(args.qa)
    retriever = TFIDFRetriever().fit(docs)

    predictions_by_k = {k: [] for k in KS}
    rankings = []
    rows_by_k = {k: [] for k in KS}

    for q in questions:
        truth = q["source_docs"]
        truth_set = set(truth)
        ranked = retriever.rank_all(q["question"])
        ranked_names = [r["filename"] for r in ranked]
        truth_ranks = {d: ranked_names.index(d) + 1 if d in ranked_names else None for d in truth}
        best_truth_rank = min(v for v in truth_ranks.values() if v is not None)
        category = question_category(q)

        rankings.append(
            {
                "id": q["id"],
                "question": q["question"],
                "category": category,
                "truth": truth,
                "truth_ranks": truth_ranks,
                "best_truth_rank": best_truth_rank,
                "top20": ranked[:20],
            }
        )

        for k in KS:
            pred = ranked_names[:k]
            predictions_by_k[k].append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "source_docs": pred,
                    "retrieval_scores": [round(r["score"], 6) for r in ranked[:k]],
                }
            )
            pred_set = set(pred)
            correct = pred_set & truth_set
            precision = len(correct) / len(pred_set) if pred_set else 0.0
            recall = len(correct) / len(truth_set)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows_by_k[k].append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "category": category,
                    "truth": truth,
                    "pred": pred,
                    "correct_top1": pred_set == truth_set,
                    "best_truth_rank": best_truth_rank,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

    truth_sets = [set(q["source_docs"]) for q in questions]
    metrics = {
        "corpus": {
            "documents": len(docs),
            "doc_types": dict(sorted(Counter(d.doc_type for d in docs).items())),
            "vocabulary_size": len(retriever.vectorizer.vocabulary_),
            "avg_doc_words": sum(len(d.text.split()) for d in docs) / len(docs),
        },
        "data": {
            "questions": len(questions),
            "single_doc_questions": sum(1 for q in questions if len(q["source_docs"]) == 1),
            "multi_doc_questions": sum(1 for q in questions if len(q["source_docs"]) > 1),
            "total_truth_links": sum(len(q["source_docs"]) for q in questions),
            "unique_truth_docs": len({d for q in questions for d in q["source_docs"]}),
        },
        "config": {
            "method": "TF-IDF word 1-2 grams, sublinear_tf, English stop words, cosine similarity",
            "ks_evaluated": list(KS),
        },
        "by_k": {},
    }

    for k in KS:
        (args.outdir / f"predictions_top{k}.json").write_text(
            json.dumps({"questions": predictions_by_k[k]}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        metrics["by_k"][str(k)] = {
            **macro_scores([set(r["pred"]) for r in rows_by_k[k]], truth_sets),
            "mean_recall_at_k": sum(
                len(set(r["pred"]) & set(r["truth"])) / len(r["truth"]) for r in rows_by_k[k]
            )
            / len(rows_by_k[k]),
        }

    top1_rows = rows_by_k[1]
    metrics["top1_by_category"] = category_table(top1_rows, "f1")
    metrics["mrr_top1"] = sum(
        (1.0 / r["best_truth_rank"] if r["best_truth_rank"] else 0.0) for r in top1_rows
    ) / len(top1_rows)

    failure_buckets = {
        "multi_doc_partial": [],
        "exact_top1_but_truth_rank_2_to_5": [],
        "truth_rank_6_to_20": [],
        "truth_outside_top20": [],
    }
    for row in top1_rows:
        if row["correct_top1"]:
            continue
        if len(row["truth"]) > 1:
            failure_buckets["multi_doc_partial"].append(row)
        elif row["best_truth_rank"] <= 5:
            failure_buckets["exact_top1_but_truth_rank_2_to_5"].append(row)
        elif row["best_truth_rank"] <= 20:
            failure_buckets["truth_rank_6_to_20"].append(row)
        else:
            failure_buckets["truth_outside_top20"].append(row)

    metrics["failure_buckets"] = {k: len(v) for k, v in failure_buckets.items()}
    (args.outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.outdir / "rankings.json").write_text(
        json.dumps({"questions": rankings}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.outdir / "top1_details.json").write_text(
        json.dumps({"questions": top1_rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = []
    report.append("# Stage 1 Baseline: TF-IDF Document Retrieval\n")
    report.append("## Corpus and validation data\n")
    report.append(f"- Documents: **{len(docs)}** across {len(metrics['corpus']['doc_types'])} types")
    for doc_type, count in metrics["corpus"]["doc_types"].items():
        report.append(f"  - `{doc_type}`: {count}")
    report.append(f"- Questions: **{len(questions)}**")
    report.append(
        f"- Single-document questions: {metrics['data']['single_doc_questions']}; "
        f"multi-document questions: {metrics['data']['multi_doc_questions']}"
    )
    report.append(f"- Truth doc links: {metrics['data']['total_truth_links']}\n")
    report.append("## Fixed-k retrieval results\n")
    report.append("| k | Macro P | Macro R | Macro F1 | Perfect | Imperfect | Recall@k |")
    report.append("|---:|---:|---:|---:|---:|---:|---:|")
    for k in KS:
        m = metrics["by_k"][str(k)]
        report.append(
            f"| {k} | {m['macro_precision']:.4f} | {m['macro_recall']:.4f} | "
            f"{m['macro_f1']:.4f} | {m['perfect']} | {m['imperfect']} | "
            f"{m['mean_recall_at_k']:.4f} |"
        )
    report.append(f"\nMRR (best truth rank): **{metrics['mrr_top1']:.4f}**\n")
    report.append("## Top-1 F1 by question category\n")
    report.append("| Category | n | Exact set match | F1 | Top-1 hit rate |")
    report.append("|---|---:|---:|---:|---:|")
    for cat, m in metrics["top1_by_category"].items():
        report.append(
            f"| {cat} | {m['n']} | {m['exact_set_match']} | {m['f1']:.4f} | {m['mrr']:.4f} |"
        )
    report.append("\n## Top-1 failure buckets\n")
    for name, items in failure_buckets.items():
        report.append(f"### {name} ({len(items)})\n")
        for row in items:
            report.append(
                f"- `{row['id']}` truth-rank={row['best_truth_rank']} F1={row['f1']:.3f} | "
                f"{row['question']}"
            )
            report.append(f"  - truth: {row['truth']}")
            report.append(f"  - top1 : {row['pred'][0]}")
        report.append("")
    (args.outdir / "failure_analysis.md").write_text("\n".join(report), encoding="utf-8")

    print(json.dumps({k: metrics[k] for k in ("corpus", "data", "by_k")}, indent=2))
    print(f"\nwrote outputs to {args.outdir}")


if __name__ == "__main__":
    main()
