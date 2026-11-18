# STAT6207 Coursework Portfolio

Static coursework website, extending the original Assignment 1 layout.

## Repository layout

```text
index.html                 Portfolio homepage
assignment1.html           Image encoders and KNN
assignment2.html           Time-series forecasting
assignment3.html           FaceNet and CNN comparison
midterm.html               Document retrieval, OCR and optional LLM judge
assignment4.html           Lane Crawford brand imagery analysis
style.css                  Shared page styles
assets/assignment1/        Assignment 1 presentation images
assets/assignment2/        Assignment 2 presentation figures
assets/assignment3/        Assignment 3 presentation figures
assets/midterm/            Original midterm report figures
code/assignment1/          Assignment 1 original Python scripts
code/assignment2/          Sessions 1–3 and Transformer scripts
code/assignment3/          Dataset preparation and experiment scripts
code/midterm/              Retrieval pipeline, app, OCR, judge and prompt
results/                   Published aggregate result evidence
results/midterm/           Midterm aggregate metrics and checks
```

Assignment 1's former `code/*.py` and `assets/*` files have been moved into their
assignment-specific folders without changing their contents. All website image
references use the new locations; the public `assignment1.html` URL is unchanged.

## Assignment pages

- [Portfolio](https://iwanttosleeeep.github.io/DL-portfolio/)
- [Assignment 1](https://iwanttosleeeep.github.io/DL-portfolio/assignment1.html): pretrained encoders and KNN.
- [Assignment 2](https://iwanttosleeeep.github.io/DL-portfolio/assignment2.html): LSTM and Transformer forecasting; repeated-run and generated-code audits.
- [Assignment 3](https://iwanttosleeeep.github.io/DL-portfolio/assignment3.html): FaceNet freezing strategies, CNN comparison and inference audit.

- [Midterm](https://iwanttosleeeep.github.io/DL-portfolio/midterm.html): document retrieval with hybrid ranking, intent routing, OCR and an optional local judge.

- [Assignment 4](https://iwanttosleeeep.github.io/DL-portfolio/assignment4.html): YOLO11n, ResNet18 and EasyOCR analysis of 400 curated brand images.

## Content and provenance

Assignment 1 retains the original published results and analysis, with updated presentation and resource paths. Assignment 2 follows the audited final assignment report and Session 3 result file. Assignment 3 follows the final report and saved validation/inference summaries. The website describes the original assignment experiments, not later environment-verification reruns. Assignment 3 external images have no ground-truth labels; prediction distributions must not be interpreted as accuracy.

`code/assignment1/`, `code/assignment2/` and `code/assignment3/` contain original Python source snapshots. Running them requires their original data, directory layout and Python dependencies; model weights and full datasets are not distributed here. Some Assignment 1 scripts retain historical machine-specific paths; see [the Assignment 1 code guide](code/assignment1/README.md) before running them. The webpages themselves need no Python environment or build step. `results/` contains compact aggregate evidence, and `assets/` contains presentation figures and selected examples.

## Midterm archive

The midterm page follows the final exam report and saved Stage 1–4 artifacts.
Source code is preserved in `code/midterm/src/`; the final 440-word prompt and
setup guide accompany it. The four report figures are copied unchanged; the full PDF remains local.
Metrics and check summaries exclude raw question/document text and machine-local
logs. The course handout, raw corpus/questions, session transcript, environment,
models and native binaries are not redistributed. Full test predictions remain in
the original local exam folder; the public archive includes counts and checksum.

Validation F1 is 0.9007 / 0.9087 / 1.0000 / 1.0000 for A/B/C/D. C and D tie;
D wins only by tie-break. Test labels are unavailable, so no test F1 is claimed.
The archived 28/28 smoke checks are historical exam evidence. Publication did not
rerun the experiments. GitHub Pages serves the results; Streamlit runs locally.

The repository was renamed from `stat6207-portfolio` to `DL-portfolio`; links use
the current GitHub repository and Pages base URL.

## Local preview

From this directory:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Open http://127.0.0.1:8765/. The same HTML/CSS and assets are served on GitHub Pages.

## Assignment 4 archive

`code/assignment4/` preserves original collection, inference, analysis and audit
scripts. `results/assignment4/` contains corrected brand-level tables, cross-model
intersections, similarity pairs, sensitivity checks and verification summaries.
`assets/assignment4/` contains six statistical figures copied from the completed
project. The page follows the final 12 November report and corrected integration.

The public snapshot uses aggregate figures and tables. The original image corpus,
full report, transcripts, account tracker, model weights and runtime caches remain
in the local coursework archive. Model inference was not rerun for publication.
The original audit had three machine-check failures; the later corrected
integration passed 2,145/2,145 consistency checks. Dataset curation and source
concentration limit representativeness, and recognition accuracy is unmeasured.
