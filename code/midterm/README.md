# Midterm: document retrieval, OCR and optional local LLM judge

[Results page](https://iwanttosleeeep.github.io/DL-portfolio/midterm.html)

`src/` is an unchanged snapshot of the original exam implementation. It contains
the word TF-IDF baseline, hybrid/field-aware ranker, intent router, evaluation and
prediction scripts, Streamlit app, OCR helper, local Ollama judge and smoke suite.
The saved results are published under `../../results/midterm/`; no experiment was
rerun for this portfolio update. The two prompt files preserve the final Stage 4
prompt and explanation. The prompt has 440 whitespace-delimited words and refers
to the original exam environment, including its `activate.sh` setup.

## Data and setup

Use Python 3.11 and the original course-provided dataset. The dataset and question
files are not bundled in this public source snapshot. Restore this layout within
`code/midterm/` before running:

```text
New Folder With Items/
  documents/*.md
  qa_pairs.json
  test_questions.json
  evaluate_retrieval_f1.py
src/
```

From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python src/run_baseline_eval.py
PYTHONPATH=src python src/plot_baseline.py
PYTHONPATH=src python src/run_stage2_eval.py
PYTHONPATH=src python src/run_final_predictions.py
streamlit run src/app/streamlit_app.py
```

Stage 2 writes `output/stage2/metrics.json`, required by the final prediction
script. Commands generate new local outputs; the public `results/midterm/` files
are an archive, not the scripts' output directory. Requirements capture the
relevant versions recorded in the exam environment, not a newly tested clean
installation. The core retrieval path requires no neural model weights.

## Optional components

OCR uses a system Tesseract executable when available, with macOS Vision through
the Swift helper as another path. The verified exam run used macOS Vision `.fast`;
building it needs the Xcode Command Line Tools. Extracted text is editable.

The judge requires local Ollama with `qwen3.5:4b-q4_K_M`. It is off by default and
runs only after enabling the option and pressing the evaluation button. It judges
relevance/sufficiency; it does not generate answers or change retrieval outputs.

With the original data and optional tools available:

```bash
PYTHONPATH=src python src/stage3_smoke_test.py
```

This suite generates outputs and may call Ollama. The archived exam suite passed
28/28 checks; that is historical evidence, not a guarantee about a new machine.

## Evaluation scope

125 validation questions: A F1 0.9007, B 0.9087, C/D 1.0000. D was selected by
tie-break, not a measured advantage over C. All 116 test predictions passed
schema/integrity checks, but test F1 is unknown because labels are unavailable.
GitHub Pages hosts the static results page; the Streamlit application runs locally.
The full PDF report, original course handout, raw corpus/questions, full session transcript,
virtual environment, model caches and native binaries are not redistributed here.
