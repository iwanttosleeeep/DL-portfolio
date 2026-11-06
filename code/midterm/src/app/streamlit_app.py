from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from app.app_logic import maybe_judge_document
from llm_judge import DEFAULT_MODEL, JudgeError, judge_document
from ocr_tools.ocr import ocr_bytes
from retrieval_service import RetrievalService


@st.cache_resource(show_spinner=False)
def get_service() -> RetrievalService:
    return RetrievalService()


@st.cache_data(show_spinner=False)
def cached_ocr(data: bytes, suffix: str) -> dict:
    result = ocr_bytes(data, suffix)
    return {
        "text": result.text,
        "method": result.method,
        "error": result.error,
    }


def _score_text(score: float | None) -> str:
    return f"{score:.4f}" if score is not None else "n/a"


def _intent_caption(intent: dict) -> str:
    kind = intent.get("kind", "unknown")
    parts = [f"Intent: {kind}"]
    for key in ("concept", "category", "field", "direction", "condition", "value"):
        if intent.get(key) not in (None, "", []):
            parts.append(f"{key}: {intent[key]}")
    return " | ".join(parts)


def _render_recommended(result: dict) -> None:
    st.subheader("Recommended source_docs (adaptive)")
    count = result["recommended_count"]
    st.caption(
        f"{count} document(s) selected by the validated query-intent logic. "
        "This adaptive count is separate from the UI Top-K setting."
    )
    if count == 0:
        st.warning("The system did not return a recommended document for this query.")
        return
    lines = [
        "| Filename | Score | Rank | Type |",
        "|---|---:|---:|---|",
    ]
    for item in result["recommended_details"]:
        rank = item["rank"] if item["rank"] is not None else "-"
        lines.append(
            f"| {item['filename']} | {_score_text(item['score'])} | {rank} | {item['doc_type']} |"
        )
    st.markdown("\n".join(lines))


def _render_top_k(result: dict) -> None:
    st.subheader("Top-K ranked documents")
    st.caption(
        f"Showing {result['top_k']} ranked document(s) for inspection. "
        "This display limit does not change the adaptive source_docs above."
    )
    for item in result["ranked_details"]:
        label = (
            f"{item['rank']}. {item['filename']} — score {_score_text(item['score'])}"
        )
        with st.expander(label):
            st.caption(
                f"Type: {item['doc_type']} | Rank: {item['rank']} | "
                f"Score: {_score_text(item['score'])}"
            )
            st.markdown("**Relevant text preview**")
            st.write(item["preview"] or "No preview available.")
            st.markdown("**Full document content**")
            st.code(item["text"], language="markdown")


def _render_judge(result: dict, active_query: str) -> None:
    st.subheader("Optional local LLM-as-Judge")
    st.caption(
        f"Model: {DEFAULT_MODEL}. This judge checks relevance and sufficiency only; "
        "it does not answer the question."
    )
    judge_enabled = st.toggle(
        "Enable local LLM judge",
        value=False,
        help="No Ollama request is made unless this is enabled and the evaluation button is pressed.",
    )
    if not judge_enabled:
        st.caption("Judge disabled: no Ollama request has been made.")
        return

    candidate_map: dict[str, dict] = {}
    for item in result["recommended_details"] + result["ranked_details"]:
        candidate_map.setdefault(item["filename"], item)
    candidates = list(candidate_map)
    if not candidates:
        st.warning("No retrieved documents are available to judge.")
        return

    default = [item["filename"] for item in result["recommended_details"][:3]]
    if not default:
        default = candidates[:3]
    query_key = hashlib.sha1(active_query.encode("utf-8")).hexdigest()[:10]
    selected = st.multiselect(
        "Documents to judge (maximum 3 per run)",
        options=candidates,
        default=default,
        key=f"judge_docs_{query_key}",
        help="The local 4B model is intentionally limited to at most three documents per run.",
    )
    if len(selected) > 3:
        st.warning("Only the first three selected documents will be judged.")
        selected = selected[:3]

    if st.button("Evaluate Retrieval with Local LLM", type="primary"):
        if not selected:
            st.warning("Select at least one document to judge.")
        else:
            judge_key = f"{query_key}|{','.join(selected)}"
            results = []
            errors = 0
            with st.status("Evaluating retrieval with the local LLM...", expanded=True) as status:
                for index, filename in enumerate(selected, start=1):
                    status.write(f"Judging {index}/{len(selected)}: {filename}")
                    try:
                        judgment = maybe_judge_document(
                            True,
                            active_query,
                            candidate_map[filename],
                            judge_fn=judge_document,
                            timeout=180.0,
                        )
                        results.append(
                            {
                                "filename": judgment.filename,
                                "acceptable": judgment.acceptable,
                                "justification": judgment.justification,
                            }
                        )
                    except JudgeError as exc:
                        errors += 1
                        results.append({"filename": filename, "error": str(exc)})
                    except Exception as exc:
                        errors += 1
                        results.append(
                            {
                                "filename": filename,
                                "error": f"Unexpected judge failure: {exc}",
                            }
                        )
                if errors == len(selected):
                    status.update(label="Judgment failed", state="error")
                else:
                    status.update(label="Judgment complete", state="complete")
            st.session_state["judge_results"] = results
            st.session_state["judge_key"] = judge_key

    judge_key = f"{query_key}|{','.join(selected)}"
    stored = st.session_state.get("judge_results")
    if stored and st.session_state.get("judge_key") == judge_key:
        st.markdown("#### Judgment")
        for item in stored:
            if "error" in item:
                st.error(f"{item['filename']}: {item['error']}")
                continue
            label = f"Acceptable: {item['acceptable']} — {item['filename']}"
            if item["acceptable"] == "Yes":
                st.success(label)
            elif item["acceptable"] == "No":
                st.error(label)
            else:
                st.warning(label)
            st.write(item["justification"])


def _render_results(service: RetrievalService, active_query: str, top_k: int) -> None:
    st.divider()
    st.subheader("Retrieval results")
    st.caption(f"Query: {active_query}")
    with st.spinner("Retrieving documents with the validated Stage 2 pipeline..."):
        try:
            result = service.search(active_query, top_k=top_k)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            return

    st.caption(
        f"Pipeline: {result['method']} | {_intent_caption(result['intent'])} | "
        f"Retrieval time: {result['elapsed_ms']:.1f} ms"
    )
    with st.expander("Retrieval rationale and intent details"):
        st.json({"intent": result["intent"], "detail": result["detail"]})

    _render_recommended(result)
    st.divider()
    _render_top_k(result)
    st.divider()
    _render_judge(result, active_query)


def main() -> None:
    st.set_page_config(
        page_title="Document Retrieval Studio",
        layout="wide",
    )
    st.title("Document Retrieval Studio")
    st.write(
        "Ask a question or upload an image containing text. Retrieval uses the validated "
        "Stage 2 system (`D_hybrid_plus_intent`)."
    )

    with st.sidebar:
        st.header("Retrieval settings")
        top_k = st.select_slider(
            "UI Top-K display",
            options=[1, 3, 5, 10, 20],
            value=10,
            help="This only controls how many ranked documents are shown.",
        )
        st.caption(
            "The recommended source_docs count is chosen adaptively by query intent and "
            "is not affected by this Top-K display setting."
        )
        st.info("Pipeline: D_hybrid_plus_intent\n\nHybrid retrieval + query-intent routing.")

    try:
        service = get_service()
    except Exception as exc:
        st.error(f"Could not load the retrieval system: {exc}")
        return

    st.session_state.setdefault("active_query", "")
    st.session_state.setdefault("active_source", "")

    text_tab, image_tab = st.tabs(["Text query", "Image query"])

    with text_tab:
        text_query = st.text_area(
            "Natural-language question",
            height=110,
            placeholder="Example: What is the total amount due on invoice INV-2024-892?",
            key="text_query_input",
        )
        if st.button("Retrieve documents", type="primary", key="retrieve_text"):
            if not text_query.strip():
                st.warning("Enter a question before retrieving.")
            else:
                st.session_state["active_query"] = text_query.strip()
                st.session_state["active_source"] = "Text query"
                st.session_state.pop("judge_results", None)

    with image_tab:
        uploaded = st.file_uploader(
            "Upload an image (PNG, JPG, or JPEG)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
            suffix = Path(uploaded.name).suffix.lower()
            st.image(image_bytes, caption=uploaded.name, use_container_width=True)
            with st.spinner("Extracting text from the image..."):
                ocr = cached_ocr(image_bytes, suffix)
            if ocr["error"]:
                st.error(f"OCR failed: {ocr['error']}")
                st.caption("No retrieval is performed when OCR fails.")
            else:
                st.success(f"OCR completed using {ocr['method']}.")
                digest = hashlib.sha1(image_bytes).hexdigest()[:10]
                extracted = st.text_area(
                    "Extracted text (editable)",
                    value=ocr["text"],
                    height=190,
                    key=f"ocr_text_{digest}",
                )
                st.caption(f"Extracted {len(ocr['text'])} characters.")
                if st.button("Retrieve using extracted text", type="primary", key="retrieve_image"):
                    if not extracted.strip():
                        st.warning("The extracted text is empty; edit it or upload another image.")
                    else:
                        st.session_state["active_query"] = extracted.strip()
                        st.session_state["active_source"] = "Image OCR"
                        st.session_state.pop("judge_results", None)
        else:
            st.caption(
                "PNG, JPG, and JPEG are supported. OCR uses local macOS Vision "
                "(or Tesseract if installed)."
            )

    active_query = st.session_state.get("active_query", "").strip()
    if active_query:
        st.caption(f"Active query source: {st.session_state.get('active_source', '')}")
        _render_results(service, active_query, top_k)
    else:
        st.info("Enter a text query or upload an image to begin.")


if __name__ == "__main__":
    main()
