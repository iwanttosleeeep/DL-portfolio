# Assignment 4 — Lane Crawford brand imagery

[Results page](https://iwanttosleeeep.github.io/DL-portfolio/assignment4.html)

This directory preserves the completed project's collection, model inference,
analysis and self-audit Python scripts, plus the corrected integration workflow.
Source files retain their original contents and relative project paths.

## Layout and inputs

- `src/` contains brand, account and post collection scripts.
- `results/yolo/` contains the YOLO11n inference, analysis and audit scripts.
- `results/resnet/` contains ResNet18 inference, analysis and audit scripts.
- `results/ocr/` contains EasyOCR inference, shared parsing, analysis and audit code.
- `integrated_analysis/` contains the corrected cross-model analysis and verifier.

Restore the original project inputs before execution. Scripts expect `images/`
with ten brand folders and 40 JPEG files per brand, the course-project manifests
under `assignment-4/data/`, and original experiment outputs beneath
`results/{yolo,resnet,ocr}/data/`. The integration workflow also reads audit and
preparation evidence. Its verifier requires the original frozen image hashes,
OCR JSONL, detection tables and ResNet NPZ embeddings.

The compact public tables live separately at `../../results/assignment4/`.
They form a result archive. The full frozen images, weights, manifests, tracker,
session transcripts and PDF remain in the local project archive. Original report
figure names are mapped to presentation names in `../../assets/assignment4/`.

## Running the original scripts

Run commands from this directory after restoring the original project structure.
Inspect each script's arguments and dependencies first.

```bash
python results/yolo/run_inference.py --device cpu --weights /path/to/yolo11n.pt
python results/resnet/run_inference.py --device cpu
python results/ocr/run_inference.py --device cpu
```

The YOLO experiment used YOLO11n. Obtain the original `yolo11n.pt` weights and
retain the recorded confidence, NMS and preprocessing settings. ResNet uses
torchvision's ImageNet-1K V1 weights. EasyOCR uses CRAFT plus `english_g2`.
The OCR-specific requirements are archived in `results/ocr/requirements.txt`.
Other imports include torch, torchvision, ultralytics, numpy, pandas, scipy,
scikit-learn, matplotlib and Pillow. Refer to original run metadata for versions;
the three experiments used separate recorded dependency combinations.

Once original model outputs and supporting evidence are available:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python integrated_analysis/analyze_brands.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python integrated_analysis/verify_analysis.py
```

These commands can generate derived files and require the original paths. The
published source snapshots were syntax-checked; portfolio preparation reused
saved outputs without repeating model inference or testing a clean installation.

## Corrected results

The historical self-analysis scripts preserve their original behavior, including
issues identified by the independent audit. The portfolio uses corrected
`integrated_analysis` outputs. Oura's supplementary modal-class share is 0.60;
Jellycat's toyshop top-one count is 3/40; within-brand cosine uses 780 unique
unordered pairs; the primary YOLO similarity uses normalized class composition.

The original independent audit recorded 855/858 passing machine checks. The later
integration verifier passed 2,145/2,145 consistency checks. These describe source
agreement and arithmetic. Sampling qualifications and model errors are documented
on the results page. Metric definitions and original source paths accompany the
public aggregate tables in `../../results/assignment4/METRICS.md`.
