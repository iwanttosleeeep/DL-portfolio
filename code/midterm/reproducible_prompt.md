Reproduce the STAT6207 Midterm RAG system on an M1 Mac with 8 GB RAM. Use existing `.venv`, `activate.sh`, local tools, and minimal dependencies. Preserve `New Folder With Items/`; create all outputs from scratch.

Inputs: `New Folder With Items/documents/*.md`, `qa_pairs.json` (125 validation questions), `test_questions.json` (unlabeled), and `evaluate_retrieval_f1.py`. Load all Markdown, preserve filenames, and infer types from prefixes.

1. Baseline: word TF-IDF 1–2 grams, sublinear TF, English stop words, `min_df=1`, cosine similarity; predict Top-1 and evaluate all validation questions with the official evaluator. Save predictions, metrics, failure analysis, plots under `output/stage1/`.

2. Improvement 1: fuse normalized word TF-IDF, `char_wb` 3–5 gram TF-IDF, and BM25 (`k1=1.5`, `b=0.75`) at 0.45/0.40/0.15. Parse names, companies, invoice/account IDs, dates, amounts, payment methods, and bank transactions; rerank with exact bonuses for IDs, names, companies, dates, and type hints. No hardcoded question-to-document mappings.

3. Improvement 2: route policy, global-extreme, aggregate-sum, and filtered-set questions. Use reusable policy concepts/synonyms; parse numeric fields for extrema/sums; return all category documents for sums and content matches for filters. Fall back to the best single document. Never use validation IDs or labels.

4. Evaluate A baseline, B +Improvement 1, C +Improvement 2, D both on all validation questions with the official evaluator. Report macro P/R/F1, exact matches, per-category results, and plots. Select by validation only: highest F1, then precision, recall, exact matches; on a complete tie prefer D, C, B, A.

5. After selection, generate `output/final/test_predictions.json` with this exact shape: `{"questions":[{"id":"test-id","question":"...","source_docs":["file.md"],"answer":""}]}`. Preserve every ID, question text, and order; use original filenames; leave `answer` empty. Validate one entry per question, exact text, existing documents, no duplicates, non-empty `source_docs`, four fields. Never tune on test questions.

6. Build a Streamlit app using the selected pipeline. Support text queries and PNG/JPG/JPEG uploads with local OCR (macOS Vision Swift helper, Tesseract fallback); show image and extracted text, allow corrections, report OCR failure. Offer Top-K including 10 and 20; show filename, score, preview, and expandable full content. Show adaptive `source_docs` separately from UI Top-K. Pin Streamlit 1.40.2 with protobuf 4.25.9.

7. Add an optional local Ollama judge using `qwen3.5:4b-q4_K_M`. Keep it off by default; call only after enabling it and pressing "Evaluate Retrieval with Local LLM". Send the query and at most three documents, truncate inputs, limit output tokens, and request only relevance/sufficiency plus a short justification, never an answer. Handle connection, model, timeout errors.

8. Save code, metrics, figures, predictions, OCR/LLM modules, requirements, a README with the launch command, and smoke-test results. Verify by execution: text query, Top-10/20, rendered-image OCR, real Ollama judgment if available, no judge call when disabled, official validation F1, and identical prediction SHA-256 across two runs. Audit aggregate/extreme predictions against raw evidence without changing parameters.
