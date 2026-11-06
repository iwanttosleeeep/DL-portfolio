from __future__ import annotations

from typing import Callable

from llm_judge import judge_document


def maybe_judge_document(
    enabled: bool,
    query: str,
    document: dict,
    judge_fn: Callable[..., object] | None = None,
    **kwargs,
):
    if not enabled:
        return None
    fn = judge_fn or judge_document
    return fn(
        query=query,
        filename=document["filename"],
        document_text=document["text"],
        **kwargs,
    )
