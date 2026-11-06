from __future__ import annotations

from .corpus import Document
from .fields import DocFields, extract_fields, extract_query_signals
from .hybrid import HybridRetriever
from .intents import (
    category_documents,
    classify_intent,
    filter_documents,
    select_policy_document,
)
from .tfidf_baseline import TFIDFRetriever


class RetrievalSystem:
    def __init__(
        self,
        documents: list[Document],
        use_hybrid: bool = False,
        use_intent: bool = False,
    ):
        self.documents = documents
        self.use_hybrid = use_hybrid
        self.use_intent = use_intent
        self.fields: list[DocFields] = [extract_fields(document) for document in documents]
        self.fields_by_name = {item.filename: item for item in self.fields}
        self.baseline = TFIDFRetriever().fit(documents)
        self.hybrid = (
            HybridRetriever(documents, self.fields, self.baseline) if use_hybrid else None
        )
        self.mode = (
            "hybrid" if use_hybrid else "baseline",
            "intent" if use_intent else "direct",
        )

    def _rank(self, question: str) -> list[dict]:
        if self.hybrid is not None:
            return self.hybrid.rank_all(question)
        return self.baseline.rank_all(question)

    def _numeric_value(self, item: DocFields, field: str) -> float | None:
        if field == "largest_deposit":
            values = [row["amount"] for row in item.transactions if row["amount"] > 0]
            return max(values) if values else None
        if field == "largest_charge":
            values = [row["amount"] for row in item.transactions if row["amount"] < 0]
            return min(values) if values else None
        return item.amounts.get(field)

    def _global_extreme(self, category: str, field: str, direction: str) -> tuple[list[str], dict]:
        if category == "purchases":
            allowed = {"receipt", "pharmacy_receipt"}
        else:
            allowed = {category}
        valued = []
        for item in self.fields:
            if item.doc_type not in allowed:
                continue
            value = self._numeric_value(item, field)
            if value is not None:
                valued.append((value, item.filename))
        if not valued:
            return [], {"values": []}
        target = max(value for value, _ in valued) if direction == "max" else min(
            value for value, _ in valued
        )
        matches = sorted(filename for value, filename in valued if abs(value - target) < 1e-9)
        return matches, {
            "field": field,
            "direction": direction,
            "target_value": target,
            "values": [
                {"filename": filename, "value": value}
                for value, filename in sorted(valued, reverse=direction == "max")
            ],
        }

    def retrieve(self, question: str) -> dict:
        rank = self._rank(question)
        rank_hint = [row["filename"] for row in rank]
        trace = {
            "method": f"{self.mode[0]}_single",
            "intent": {"kind": "single"},
            "top_results": rank[:10],
        }

        if not self.use_intent:
            trace["source_docs"] = rank_hint[:1]
            return trace

        signals = extract_query_signals(question, self.fields)
        intent = classify_intent(question, signals)
        kind = intent["kind"]

        if kind == "policy":
            filename = select_policy_document(question, intent.get("concept"), self.fields, rank_hint)
            if filename:
                trace.update(
                    {
                        "method": "policy_intent",
                        "intent": intent,
                        "source_docs": [filename],
                    }
                )
                return trace

        elif kind == "global_extreme":
            matches, detail = self._global_extreme(
                intent["category"], intent["field"], intent["direction"]
            )
            if matches:
                trace.update(
                    {
                        "method": "global_extreme",
                        "intent": intent,
                        "source_docs": matches,
                        "detail": detail,
                    }
                )
                return trace

        elif kind == "aggregate_sum":
            matches = category_documents(self.fields, intent["category"])
            if matches:
                values = [
                    self.fields_by_name[filename].amounts.get(intent.get("field") or "")
                    for filename in matches
                ]
                numeric = [value for value in values if value is not None]
                trace.update(
                    {
                        "method": "aggregate_sum",
                        "intent": intent,
                        "source_docs": matches,
                        "detail": {
                            "document_count": len(matches),
                            "aggregate_value": sum(numeric) if numeric else None,
                        },
                    }
                )
                return trace

        elif kind == "filtered_set":
            matches = filter_documents(
                self.fields, intent["category"], intent["condition"], intent["value"]
            )
            if matches:
                trace.update(
                    {
                        "method": "filtered_set",
                        "intent": intent,
                        "source_docs": matches,
                    }
                )
                return trace

        trace["source_docs"] = rank_hint[:1]
        return trace
