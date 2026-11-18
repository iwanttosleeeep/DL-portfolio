# Metrics and source map

All paths below are relative to the Assignment 4 workspace. Derived tables reside in `integrated_analysis/data/`. Images remain the frozen 40 per brand.

| Measure | Definition | Principal numerical source |
|---|---|---|
| YOLO object count | Number of retained detection boxes at confidence >=0.25 | `results/yolo/data/object_detections.csv`; zeros from image manifest |
| Class image prevalence | Images with >=1 detection of that class /40 | Same detection CSV, unique image IDs |
| Leading YOLO class | Highest total detection count; ties resolved by image prevalence, then class name | Same CSV; the profile reports images containing that class, not its detection count |
| Normalized object composition | Per-brand class detection counts divided by that brand's total detections, using all 80 COCO classes | Same detection CSV + class names in YOLO metadata |
| Normalized composition similarity | `1 - 0.5*sum(abs(p_a-p_b))`; equivalent to 1 minus Bray-Curtis for unit-sum vectors | Newly derived composition table |
| ResNet modal class share | Fraction of the 40 images assigned the modal top-one class | `results/resnet/data/image_predictions.csv` |
| Modal-class ties | All classes with maximal top-one frequency; one original modal label is retained for display | Same prediction CSV; do not infer unique dominance for Sennheiser |
| Shannon entropy | `-sum(p*log2(p))` over the brand's 40 top-one labels | Same prediction CSV; distribution diversity, not per-image uncertainty |
| Corrected modal-class share of all occurrences | That brand's count of its modal label / number of images with that label in the entire dataset | Same prediction CSV; supplementary, not used as brand identity/recognition score |
| Feature cohesion | Mean cosine over the 780 unordered pairs of L2-normalized 512-dimensional features within a brand | `results/resnet/data/image_embeddings.npz` |
| Within-minus-between gap | Within-brand mean minus mean over the brand's 40*360 image pairs to the other nine brands | Same NPZ; not accuracy or statistical significance |
| Cross-brand feature similarity | Mean cosine over 40*40=1600 pairs for each distinct brand pair | Same NPZ; matrix diagonal stores within-brand cohesion, although the figure masks it |
| OCR text presence | >=1 region retained at recognition confidence >=0.30 | `results/ocr/data/ocr_raw.jsonl` and audited image table |
| OCR text volume | Retained regions/tokens per image; mean and median use all 40 including zeros | Raw JSONL; ASCII/alphanumeric token rule from original experiment |
| OCR target-name presence | At least one exact normalized target alias matched in concatenated retained OCR strings | Audited `ocr_image_results.csv`; no new aliases or thresholds introduced |
| Vocabulary document frequency | Number of images with the filtered term; distinct from total term count | Audited `results/ocr/data/text_frequency.csv` |
| Post-weighting sensitivity | Average within each post, then average across that brand's distinct posts | Joined image table + canonical post URLs |
| Jo Malone source strata | Official `jomalonelondon` handle (27) versus other known accounts (13) | Canonical collection manifest; missing account-type cells do not override a known official handle |

`image_level_integrated.csv` has one row per image, namespaced fields from each model, canonical post/account metadata, and recomputed mean feature similarities. `unified_brand_profile.csv` has one row per brand. Suffixes `_pct` are percentages, not fractions; cosine values are unitless; entropy is in bits. Full precision is retained in CSVs (up to 12 significant digits); report and figure labels are rounded for reading.

The four-column percentage panel does not combine its measures into a score. No automated product/lifestyle classification was added. Those descriptions are comparative interpretations tied to model categories and visually inspected examples.
