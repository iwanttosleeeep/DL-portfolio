from __future__ import annotations

import time
from pathlib import Path

from rag.corpus import Document, load_documents
from rag.system import RetrievalSystem


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOC_DIR = ROOT / "New Folder With Items" / "documents"


def _preview(text: str, limit: int = 700) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit].rstrip() + "..."


class RetrievalService:
    def __init__(self, doc_dir: str | Path = DEFAULT_DOC_DIR):
        self.doc_dir = Path(doc_dir)
        self.documents: list[Document] = load_documents(self.doc_dir)
        self.system = RetrievalSystem(
            self.documents,
            use_hybrid=True,
            use_intent=True,
        )
        self.doc_by_name = {document.filename: document for document in self.documents}

    def _detail(self, filename: str, score: float | None, rank: int | None) -> dict:
        document = self.doc_by_name.get(filename)
        if document is None:
            return {
                "filename": filename,
                "score": score,
                "rank": rank,
                "doc_type": "unknown",
                "preview": "",
                "text": "",
            }
        return {
            "filename": filename,
            "score": score,
            "rank": rank,
            "doc_type": document.doc_type,
            "preview": _preview(document.text),
            "text": document.text,
        }

    def search(self, query: str, top_k: int = 10) -> dict:
        query = " ".join(query.split())
        if not query:
            raise ValueError("Enter a question or provide an image containing text.")
        if top_k < 1:
            raise ValueError("Top-K must be at least 1.")

        started = time.perf_counter()
        ranked = self.system._rank(query)
        trace = self.system.retrieve(query)
        elapsed_ms = (time.perf_counter() - started) * 1000

        score_by_name = {row["filename"]: row["score"] for row in ranked}
        rank_by_name = {row["filename"]: row["rank"] for row in ranked}
        recommended = list(dict.fromkeys(trace.get("source_docs", [])))
        ranked_top = ranked[: min(top_k, len(ranked))]

        return {
            "query": query,
            "top_k": len(ranked_top),
            "requested_top_k": top_k,
            "method": trace.get("method"),
            "intent": trace.get("intent", {}),
            "source_docs": recommended,
            "recommended_count": len(recommended),
            "recommended_details": [
                self._detail(
                    filename,
                    score_by_name.get(filename),
                    rank_by_name.get(filename),
                )
                for filename in recommended
            ],
            "ranked": ranked_top,
            "ranked_details": [
                self._detail(
                    row["filename"],
                    row["score"],
                    row["rank"],
                )
                for row in ranked_top
            ],
            "detail": trace.get("detail", {}),
            "elapsed_ms": elapsed_ms,
        }
