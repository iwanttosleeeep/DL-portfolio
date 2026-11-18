#!/usr/bin/env python3
"""Assemble the analytical narrative and source-derived unified table. No source writes."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[1];O=R/'integrated_analysis'
table=(O/'UNIFIED_PROFILE_TABLE.md').read_text().strip()
text='''# STAT6207 Assignment 4 — Integrated Brand Analysis

**Topic 2: Lane Crawford Brand Discovery · Step 2 · 12 November 2026**

## Analytical overview

The three pretrained models describe different aspects of the same curated Instagram images. Their combination separates **generic product/category cues, scene context, feature-space cohesion, and readable text**. These dimensions do not form a single ranking of brand strength.

Four findings are most useful. First, category evidence can be strong even when OCR rarely recovers a brand name: Jellycat and Herman Miller are the clearest cases. Second, semantic label diversity and visual feature cohesion differ: Sennheiser has many top-one labels but relatively high within-brand feature similarity. Third, OCR can add product/name evidence when visual categories are misleading, as in Kodak, while text-rich images can still yield no target-name match, as in Fortnum & Mason. Fourth, similar brand-level category rates can conceal complementary image selections: Jo Malone's bottle detections and perfume classifications overlap in only one image.

The analysis below is restricted to **400 frozen images, 40 per brand, from 350 distinct Instagram posts**. It describes representation in this collected sample. No model was retrained or rerun, no source image or original experiment was edited, and no final PDF is produced in this step.

## 1. Research design and provenance

### Brand selection

The project extracted 407 unique Home & Lifestyle brand-page entries from Lane Crawford's Hong Kong directory. English Wikipedia pageviews for the preceding twelve complete months supplied an initial attention proxy. Initial selection required Jellycat and Stanley, considered ranked candidates with positive recorded Lane Crawford product availability, rejected mismatched Wikipedia subjects, and limited each assigned category to two brands. Subsequent explicit user decisions replaced Hennessy and Dom Pérignon to avoid alcohol-account collection barriers, and Fitbit because its sole listed product was reported discontinued. Sennheiser, Herman Miller, and Jo Malone London were added from the remaining ranking.

The final ten span toys (Jellycat, LEGO), audio (Bang & Olufsen, Sennheiser), drinkware (Stanley), cameras/imaging (Kodak), fragrance (Jo Malone London), food (Fortnum & Mason), furniture (Herman Miller), and personal electronics (Oura). This is a purposive selection shaped by category coverage, availability, access, and user priorities. It is not a random draw or simply the ten most popular brands. Wikipedia pageviews are not Instagram popularity, sales, or Google Trends; Stanley's unmatched Wikipedia record is not evidence of zero interest.

**Evidence:** `data/all_brands.csv`; `assignment-4/data/selected_brands.csv`; `assignment-4/data/brand_ranking.csv`; discovery transcripts; `final_audit/brand_selection_verification.csv`.

### Discovery, human review, and manual curation

AI-assisted public-source searches identified relevant official, retailer, media, and creator accounts and candidate post URLs; human review supplied account/post approvals and selected images. The initial account workflow produced 56 approved accounts. Later user-reviewed posts and manually collected images expanded the sources. Automated retrieval checked metadata and logged successes, rejections, and access interruptions. The final image set combines 224 records labelled `fetch`, 175 `manual_collection`, and one user-supplied attachment.

Final selection, relevance review, reduction of repetitions, and balancing were **manual curation decisions**, with scripts supporting file organization, renaming, hashing, and verification. The recorded balancing stage reduced a 443-image cleaned set by 44 exclusions and added one Sennheiser image, producing the final balanced set. This manual curation was **not a separate model experiment and has no separate supplied AI-session transcript**. The collection transcript ends at an earlier 263-image stage; final manifests, mappings, exclusion decisions, and preparation records document subsequent organization. No missing transcript has been invented or treated as evidence of an additional model run.

All three models processed the same frozen image hashes and brand assignments. The independent audit confirmed readable RGB JPEGs, matching manifests, and no exact file or decoded-pixel duplicates. Image suffixes identify dataset rows, not original carousel positions. Fifty images are additional frames from posts already represented, so the sample contains 350 posts rather than 400 independent posts.

### Sampling qualifications

Equal image counts provide consistent image-level denominators, but do not balance account exposure, campaign, time, or image type. The final manifest has 129 blank account handles, 144 blank account types, 176 missing publication datetimes, 73 records with only `Creator` eligibility wording, and 47 unknown carousel indices; two Sennheiser post/index combinations conflict. These fields remain unresolved rather than imputed. The known dates span 2016–2026; there is no common recent sampling window. Historical availability and account assertions were not freshly authenticated online in Step 2.

Jo Malone London's known official account contributes **27/40 images (67.5%)**, exceeding the original 20-image per-official-account cap. This source concentration is retained and explicitly examined below. Neither all creators' eligibility nor universal compliance with every original account/media-selection rule is claimed.

**Evidence:** canonical `assignment-4/data/manifest/dataset.csv`; `final_audit/collection_provenance_by_brand.csv`; `final_audit/account_sampling_caps.csv`; `final_audit/balance_chain_verification.csv`; preparation evidence indexed in `final_audit/README.md`.

## 2. Integrated measurements and audit corrections

YOLO11n supplies retained COCO detection boxes at confidence 0.25, with fixed 640-square letterboxing and NMS IoU 0.45. ResNet18 uses ImageNet-1K V1 weights, RGB resize 256/center crop 224, and its 512-dimensional pooled representation. EasyOCR uses pretrained CRAFT plus `english_g2`, English-only, and a retained recognition-confidence threshold of 0.30. These are pretrained inference systems, with no training or fine-tuning. Exact weights/configurations remain in the original model metadata and independent audit.

The integration joins **one row per image** by brand/filename, then constructs brand profiles and same-image intersections. All ordinary image proportions divide by 40; zero-detection and zero-text cases remain in the denominator. The unit is an image, not an independent consumer observation.

| Dimension | Operational measure | Interpretation boundary |
|---|---|---|
| Detected object/context | Images containing a YOLO class; boxes per image | Model detections, not verified products or humans |
| Recognizable generic category | ResNet modal top-one class and its image count; selected top-five evidence | ImageNet categories, not brand classification |
| Classification diversity | Shannon entropy `H = -Σ p log2(p)` across the 40 top-one labels | Across-image label diversity; not per-image uncertainty or complete visual diversity |
| Visual feature cohesion | Mean cosine over 780 unique unordered within-brand image pairs | Feature-space resemblance; not commercial identity |
| Relative cohesion | Within-brand mean minus mean across that brand's 40×360 pairs to the other brands | Descriptive contrast, not accuracy or a significance test |
| Text presence/volume | Retained OCR regions/tokens; mean and median per image | Recognized text; not all visible text or pixel-area density |
| Target-name presence | Images with at least one normalized exact target alias in retained text | OCR-recognized textual name; not a validated logo/brand-visibility score |
| Vocabulary | Filtered unigram/bigram image document frequencies | Noisy textual evidence; frequency differs from total token count |

**Normalized object composition.** For brand b, each of the 80 COCO class counts is divided by that brand's total detections, giving a unit-sum vector p_b. The primary object similarity is `1 - 0.5 Σ |p_a - p_b|`, equivalent to one minus Bray–Curtis dissimilarity on the normalized vectors. Raw-count Bray–Curtis is retained in a separately labelled reference table. Normalization removes total-detection volume from this comparison; it does not remove detector bias, and images with many detections still contribute more events to the pooled composition. It is not an average of image-level compositions. Zero-detection images remain in image-prevalence measures but contribute no class events.

**Cross-brand feature similarity.** Each off-diagonal ResNet similarity is the mean cosine over 40×40=1,600 cross-brand image pairs. Within-brand cohesion uses only the 780 unordered pairs, consistently for the mean, SD, and pair count. All 7,800 unique within-brand pairs and 72,000 between-brand pairs are included in the overall comparison. Matrix diagonals contain within-brand cohesion but are masked in the comparative figure.

Applied corrections are isolated in new derived files:

- The ResNet supplementary dominant-class share is recomputed by looking up the modal label explicitly: brand modal count divided by its dataset-wide count. Oura is **3/5=0.60**, replacing the faulty supplementary value 1.0. This quantity is supplied for completeness and is not used as a brand-recognition score.
- Jellycat `toyshop` top-one frequency is **3/40 (7.5%)**; **11/40 (27.5%)** belongs to LEGO. Jellycat's `toyshop` top-five frequency remains 19/40.
- Primary YOLO similarity uses normalized composition; raw-count similarity is never relabelled as that measure.
- ResNet per-brand pair counts and SDs use 780 unique pairs instead of 1,560 directed entries.
- LEGO `pcs` vocabulary document frequency is **4/40 (10%)**.
- OCR measures are not called guaranteed lower bounds. ImageNet includes `loudspeaker`; imperfect product coverage is distinguished from class absence. Similarity does not establish the cause of a shared visual style.

**Reproducible sources:** `data/image_level_integrated.csv`, `data/unified_brand_profile.csv`, `data/corrected_resnet_modal_class_share.csv`, `data/resnet_unique_pair_statistics.csv`, `data/yolo_normalized_composition.csv`; detailed definitions in `METRICS.md`. All paths beginning `data/` in the remainder of this document refer to `integrated_analysis/data/`.

## 3. Unified quantitative profiles

The table integrates category, diversity/cohesion, and text measurements. It does not rank brands on an overall score. “Leading YOLO class” is selected by total detections, with ties resolved by image prevalence then class name; the displayed count is the number of images containing that class. An image can contain multiple classes; this is not a distribution of mutually exclusive image categories. ResNet modal labels are generic and can be inappropriate product descriptions.

__PROFILE_TABLE__

*Table 1. Every image count uses the denominator 40. H is top-one label-distribution entropy. “Within cosine / gap” reports mean within-brand feature similarity and its difference from mean similarity to the other nine brands. “OCR text / name” gives images with retained text and with a target-name match. Sennheiser has three tied modal labels—`stethoscope`, `lab coat`, and `lens cap`, each 3/40—so the displayed label is not uniquely dominant. Full precision and other indicators are in `data/unified_brand_profile.csv`.*

Figure 1 separates person-class detections, modal-label concentration, and OCR rates, then compares mean/median tokens. The shared percentage denominator does not make those dimensions equivalent. No product-versus-lifestyle score is constructed, because a COCO `person` label can identify minifigures or illustrations as well as humans, and absence of a suitable product label can reflect model vocabulary.

__FIG1__

*Figure 1. Integrated measurements for all ten brands. Mean–median differences expose text-heavy outliers that a presence percentage alone cannot reveal. Values are exported in `data/figure01_comparison_data.csv`.*

## 4. Cross-model findings

### 4.1 Generic category consistency and textual naming are separate dimensions

**Jellycat and Herman Miller have strong category cues without frequent OCR name recovery.** Jellycat has `teddy bear` detections in **33/40 images (82.5%)**, `teddy` top-one predictions in **22/40 (55%)**, and only **11/40 (27.5%)** OCR target-name images. The two visual signals occur together in **19 images**; **24 of the 33** YOLO plush-labelled images have no OCR target-name match. That pattern supports recurring generic plush content in the sample, rather than evidence that the model distinguishes Jellycat from other plush brands.

Herman Miller has chairs in **31/40 images (77.5%)**. All **eight** `desk` top-one images also contain a YOLO chair detection. Yet its target name is recovered in only **2/40 images (5%)**, and **30 of the 31** chair-containing images have no target-name match. Category/context evidence is therefore substantially more available than textual naming. The `_06` example has chair detections and a `folding chair` prediction with probability 0.9607; its only retained OCR string is `Ceres`, not the target name.

The two sets are not identical kinds of consistency. Jellycat has the lowest class entropy, **2.75 bits**, while Herman Miller's furniture/room labels yield **4.09 bits**. Nevertheless, Herman Miller has the highest feature cohesion, **0.665**, compared with Jellycat's **0.647**. Recurring plush categories and recurring furniture/room representations are distinct routes to consistency in these measurements.

**LEGO provides an important counterexample to person-based lifestyle interpretations.** Its `toyshop` top-one frequency is **11/40**, and its within-brand feature cosine is **0.656**. YOLO assigns `person` in **20/40 images**, but the `_23` example is a model building with minifigures, producing 13 person detections. The person rate cannot be read as the percentage of human-centred lifestyle posts. OCR finds retained text in **35/40 (87.5%)** but the target name in only **4/40 (10%)**; among text-positive images, only **4/35 (11.4%)** have a target-name match. Corrected vocabulary such as `brickset` (5 images) and `pcs` (4) adds retail/building context but does not repair a missed stylized wordmark.

### 4.2 Similar marginal counts can conceal different image-level evidence

Jo Malone London is the clearest example of complementarity rather than same-image agreement. YOLO detects `bottle` in **10 images**, and ResNet predicts `perfume` in **10**, but these signals overlap in **only one image**. Nine images are bottle-only and nine perfume-only; either signal appears in **19/40 images**. Of those 19, **14** also have an OCR target-name match.

The image examples explain why a single model is insufficient. In `_26`, YOLO returns zero detections, while ResNet predicts `perfume` at 0.9996 and OCR supplies two target-name matches in a visible perfume presentation. In `_31`, YOLO detects a bottle and cup within a fragrance/candle arrangement, and OCR recovers one name match; ResNet instead predicts `Crock Pot` at 0.1822. These are different successful and unsuccessful observations of local objects, global image content, and text. Differences in tasks, crop/letterbox preprocessing, and the image content can contribute; this analysis does not isolate a single causal mechanism.

| Brand and selected class pair | YOLO-positive | ResNet-positive | Both | YOLO only | ResNet only | Either | Either plus OCR name |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jellycat: `teddy bear` / `teddy` | 33 | 22 | 19 | 14 | 3 | 36 | 9 |
| Herman Miller: `chair` / `desk` | 31 | 8 | 8 | 23 | 0 | 31 | 1 |
| Jo Malone London: `bottle` / `perfume` | 10 | 10 | 1 | 9 | 9 | 19 | 14 |
| Kodak: `cell phone` / `Polaroid camera` | 15 | 6 | 4 | 11 | 2 | 17 | 11 |
| LEGO: `person` / `toyshop` | 20 | 11 | 7 | 13 | 4 | 24 | 3 |

*Table 2. Same-image intersections from `data/cross_model_image_overlap.csv`. Each row has 40 images. Class pairs were selected to investigate observed mechanisms, not as equivalent measures or ground-truth product indicators. In particular, Kodak cell phones and LEGO persons can be model confusions, and “either” does not represent validated product-recognition recall.*

### 4.3 Text can supply product/name information when visual labels are misleading

**Kodak has the most extensive OCR-readable material in the sample:** **38/40 images (95%)** contain retained text, **28/40 (70%)** contain a target-name match, and mean/median token counts are **11.975/10**. Its visual labels are less straightforward: YOLO labels cell phones in **15 images**, while ResNet predicts `Polaroid camera` in **6** and includes it in the top five for **12/40 (30%)**. Neither the low camera-labelled frequency nor the phone-labelled frequency is a verified account of which images actually contain cameras.

The `_26` image visibly presents seven compact cameras. YOLO produces seven `cell phone` detections, ResNet confidently predicts `switch` (0.9882), and OCR returns eight `Kodak` target-name matches. This demonstrates what OCR adds when packaging/name text survives but generic visual categories fail. The recognised `Polaroid camera` label, when present elsewhere, is also an ImageNet category rather than evidence that the Kodak image depicts the Polaroid brand.

Filtered vocabulary supports a product-information reading of much of the recognised text: `charmera` appears in **6** Kodak images, `film` and `camera` in **5 each**, and `digital` in **4**. These are sample image document frequencies, not market-level themes. Manual curation removed many weaker film-output images from the prior set, so the text-forward result must not be attributed solely to Kodak's underlying communications strategy.

**Fortnum & Mason illustrates a different limitation: abundant text does not guarantee target-name recovery.** It averages **11.20 tokens** and **9.325 regions** per image, with text present in **32/40 (80%)**, but name matches in **12/40 (30%)**. Its `hamper` top-one frequency is **7/40**, with `hamper` in the top five for **12/40**; cups, bowls, bottles, cakes, and dining tables provide food/retail context. Vocabulary such as `tea` and `christmas` (**5 images each**) supports that context.

In `_22`, ResNet identifies `hamper` (0.9999), YOLO detects only a vase, and OCR retains **35 regions** but matches the target name zero times. Brand text is visibly present; OCR fragments/intervening text prevent the exact alias match. Kodak and Fortnum thus differ not only in text volume but also in recoverable naming and OCR segmentation. Fortnum's lower matching rate cannot establish weaker actual branding.

### 4.4 High class diversity does not necessarily imply weak feature cohesion

Sennheiser has **30 distinct top-one labels**, entropy **4.77 bits**, and a modal frequency of only **3/40** (with a three-way tie). Yet its mean within-brand feature cosine, **0.657**, is second only to Herman Miller in this sample. Its within-minus-between gap is **0.055**. This shows that stable aspects of the feature representation can coexist with inconsistent generic class assignments; it does not identify which visual factors cause the cohesion.

YOLO assigns `person` in **33/40 Sennheiser images (82.5%)**, and OCR recovers the target name in **12/40 (30%)**. Recognised terms `sound` and `wireless` occur in **6 images each**, and `momentum` in **4**, adding audio-product semantics that the dominant ImageNet labels do not supply. The `_35` earbud-case example receives `joystick` at 0.9286 and no retained OCR text, so combination is not a guarantee that a difficult image will be rescued.

**Bang & Olufsen and Oura are less cohesive in this representation**: within means **0.612** and **0.609**, gaps **0.023** and **0.024**, and entropies **4.97** and **4.95 bits**, respectively. Their person-class image counts are **21/40** and **27/40**. Reviewed images include product presentations, people/product use, rooms, and graphics, making a contextual/mixed-content interpretation plausible; the model rates alone do not quantify human lifestyle content.

Their text profiles differ sharply. Bang & Olufsen has retained text in **10/40 (25%)**, name matches in **3/40 (7.5%)**, and mean/median tokens **1.525/0**. Oura has text in **29/40 (72.5%)**, name matches in **14/40 (35%)**, and mean/median tokens **10.20/6.5**. Oura vocabulary such as `ring` (**7 images**) and `sleep` (**4**) supplies product/feature information despite diffuse visual classes. Bang & Olufsen's `loudspeaker` appears in the top-five list for **6/40** images; ImageNet does include that class, although specialised products and presentation can still be poorly described. Low recovered text/name frequency in Bang & Olufsen is an OCR observation, not evidence that its images lack logos or deliberate typography.

__FIG2__

*Figure 2. Classification-distribution entropy and feature cohesion are different properties. The right panel compares each brand's mean within similarity with its mean similarity to all other brands. These are descriptive effects, not inferential estimates of brand separation. Data: `data/figure02_diversity_cohesion_data.csv`.*

### 4.5 Container/scene resemblance crosses commercial product categories

Stanley combines `person` in **21/40 images (52.5%)** with bottle/cup/vase detections and diffuse global labels: `bucket` is top one in **5/40**, while `coffee mug` appears in the top five in **7/40**. It has text in **32/40 (80%)**, target-name matches in **19/40 (47.5%)**, and mean/median tokens **3.725/2.5**. Reviewed examples include product displays and use/event imagery. Its vocabulary is comparatively sparse—top terms `company` and `day` occur in only **2 images each**—so a detailed campaign strategy cannot be inferred from those fragments.

The normalized object-composition matrix places **Jo Malone London–Stanley first (0.700638)**, despite their different focal categories. Bottle/cup/vase/container assignments plausibly contribute; the source images and detector confusions prevent interpreting that as a shared product taxonomy. Their feature cosine is also relatively high (**0.618801**, second among 45 pairs), which corroborates some model-derived resemblance without explaining it commercially.

Similarity is nevertheless method dependent. **Sennheiser–Stanley is first by feature cosine (0.620490) but only 16th by object composition (0.439216)**. **Bang & Olufsen–LEGO is third by object composition (0.626866) but 35th by feature cosine (0.575292)**. Shared generic detections—particularly person assignments—can resemble one another even when the full image representations do not. These are complementary comparisons, not mutually contradictory results.

| Pair | Normalized YOLO composition | Rank /45 | ResNet mean cosine | Rank /45 |
|---|---:|---:|---:|---:|
| Jo Malone London–Stanley | 0.700638 | 1 | 0.618801 | 2 |
| Sennheiser–Stanley | 0.439216 | 16 | 0.620490 | 1 |
| Bang & Olufsen–LEGO | 0.626866 | 3 | 0.575292 | 35 |
| Jellycat–Kodak | 0.076288 | 45 | 0.551502 | 45 |

*Table 3. Selected comparisons expose agreement and disagreement between two definitions of resemblance. Values are not directly comparable across columns. All 45 pairs and raw-count references are in `data/cross_model_similarity_pairs.csv`.*

Across all 45 pair rankings, descriptive Spearman correlation is **0.6169**. No p-value or independence claim is attached: pairs share endpoints and derive from the same curated images. Jellycat–Kodak is least similar under both methods, supporting a broad content contrast in this set. That agreement does not validate a consumer/commercial distance. The overall within/between ResNet means are only **0.6401/0.5907**; the audited silhouette is **0.0377**, and the first two PCA components explain only **10.34%** of feature variance. Strong market segmentation or sharply separated visual clusters would be an overstatement.

__FIG3__

*Figure 3. Normalized object-class composition versus average feature cosine, with the same row order. Diagonals are omitted; colour scales intentionally differ. Matrix sources and definitions are explicit in the data filenames and `METRICS.md`. The former raw-count YOLO comparison is saved only as a separately named reference.*

## 5. Representative image-level interpretation

__FIG4__

*Figure 4. Eight deliberately selected diagnostic examples. A: generic plush evidence despite failed text recovery; B: furniture-label agreement without target naming; C: camera-category confusions with successful repeated name text; D/E: complementary local-object and global-category evidence for the same brand; F: a specialised product not rescued by any measured output; G: extensive OCR without successful alias matching; H: minifigures counted as persons. Probabilities express model output, not correctness. Full strings, counts, source image hashes, and observations are in `data/representative_examples.csv`.*

The comparison adds more than three independent brand summaries because it asks **which evidence co-occurs in the same images** and **which interpretations survive a change of measurement**. It also exposes cases where combining outputs would create a misleading conclusion: camera images should not become phone-oriented content solely because of YOLO, toy figures should not become human lifestyle subjects, and absent OCR name matches should not become absent branding.

## 6. Source concentration and post dependence

### Jo Malone official-account sensitivity

The 27 official-account images and 13 images from other known accounts have different measured content:

| Source stratum | Images | YOLO person images | OCR text images | OCR target-name images | Mean tokens |
|---|---:|---:|---:|---:|---:|
| `jomalonelondon` official | 27 | 5 (18.5%) | 21 | 19 (70.4%) | 4.407 |
| Other known accounts | 13 | 7 (53.8%) | 12 | 5 (38.5%) | 10.923 |

*Table 4. Within-sample source comparison; `data/jo_malone_source_strata.csv`. Other-account identity is known for these Jo Malone rows even where some account-type cells are blank.*

The overall Jo Malone name-match rate of 60% and person-class rate of 30% combine substantially different source strata. The official subset is less person-labelled and more name-positive, but the other subset has more recognized tokens on average. This demonstrates the distinction between textual name repetition and total text volume, and why a sample-level profile cannot be attributed entirely to a brand-wide strategy. These are descriptive differences with unequal sample sizes, not causal effects of official ownership. No image has been removed or replaced to satisfy the original cap.

### Multiple images from a post

As a secondary diagnostic, each image's primary measurements were averaged within its source post, then averaged across the brand's distinct posts. This gives equal weight to posts while retaining every image. Across YOLO person presence, OCR text presence, and OCR target-name presence, the largest image-weighted versus equal-post change is **5.73 percentage points** (LEGO person presence). Fortnum & Mason name presence changes from **30.0% to 35.56%**; Kodak remains the most text-positive set and Bang & Olufsen the least under this reweighting.

These modest changes for the selected rates do not remove account/campaign dependence, selective curation, or missing metadata. Jo Malone has 40 distinct posts, so equal-post weighting leaves its source concentration unchanged. The 40-image estimates remain the primary reported results; the sensitivity is labelled separately in `data/post_weighting_sensitivity.csv`.

## 7. Substantive interpretation and limits

The integrated evidence supports three broad conclusions about the **collected imagery**:

1. **Product-category recurrence, feature cohesion, and readable naming can diverge.** Jellycat and Herman Miller have strong generic product/context evidence despite limited textual naming; Sennheiser has cohesive features despite diverse generic labels. No single output adequately measures visual consistency or brand identity.
2. **Text contributes a separate descriptive channel.** It supplies names/product vocabulary in Kodak, Oura, Sennheiser, and Jo Malone images that generic visual labels can miss. But Fortnum segmentation failures, stylized logos, multilingual copy, and the threshold make recognized text an incomplete and error-prone observation. Sparse OCR vocabulary in Bang & Olufsen or Stanley does not imply sparse brand meaning.
3. **Cross-brand resemblance changes with the representation.** Container/person/context labels and whole-image features can support different pairings. Similarity helps characterize sampled content and investigate examples; it does not demonstrate commercial affinity, consumer preferences, brand equity, or a business recommendation.

Several constraints prevent stronger conclusions. The ten brands and images were purposively selected, and relevance/repetition filtering differed across collections. Forty images per brand do not represent each brand's complete account history or products. Multiple frames, recurring campaigns, source concentration, and uneven date coverage violate simple independent random-sample assumptions. No significance tests or population confidence intervals are reported.

The models are general-purpose and not ground-truth annotators. COCO lacks several focal product categories, ImageNet labels are imperfect for specialised products and marketing layouts, and high softmax probability can be wrong. Crop/resize choices alter available context. Feature similarities can reflect shared representation biases; their causes were not experimentally isolated. OCR is English-only, thresholded, sensitive to image size/font/angle, and uses exact normalized aliases. It can miss true text or introduce false text/matches; its counts are not guaranteed bounds on human-visible branding. Vocabulary filtering is brand-specific, includes edit-distance suppression and stop words, and can remove meaningful terms or retain noise. Bigrams are formed after filtering within an OCR region, so they are not always contiguous phrases in the original text.

There are no manual category labels or complete OCR transcriptions to estimate accuracy, precision, recall, or comparative detector reliability. The selected examples diagnose mechanisms rather than estimate error rates. Joining models is not an ensemble trained or validated for brand recognition. No claim about marketing effectiveness, buying intent, consumer preference, or commercial performance follows from these outputs.

## 8. Verification, reproducibility, and handoff to the PDF stage

`analyze_brands.py` regenerates the new CSVs, four integrated PNGs, summary, and unified table from the frozen inputs. `verify_analysis.py` separately reconciles joins, image hashes, class counts, raw OCR region/token counts, repaired modal shares, unique-pair feature statistics, both similarity definitions, same-image intersections, source strata, post weighting, vocabulary evidence, and figure-source tables. The figure review is recorded in `FIGURE_REVIEW.md`.

The numerical verification completed with **__PASSED__/__CHECKS__ checks passing**. These are consistency checks, not independent scientific trials or measures of accuracy. All **__PRESERVED__ original project/audit files** captured for this step remain byte-identical, including the **1,090 files** in the original audit baseline; no files were added to the protected source folders. Original experiment/audit documents are retained as historical records, with corrections applied only to new derived outputs and this narrative.

Reproduction from the workspace root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 integrated_analysis/analyze_brands.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 integrated_analysis/verify_analysis.py
python3 integrated_analysis/write_analysis_document.py
```

Dependencies and runtime versions are in `runtime_metadata.json`; definitions are in `METRICS.md`; numerical source hashes are in `data/source_file_hashes.csv`. The original frozen files and their relative workspace paths are required. The independent audit remains authoritative for source provenance and experiment verification; this step does not repeat model inference.

For the final report, Table 1 and Figures 1–3 provide the primary comparative structure; Table 2 and Figure 4 establish the image-level mechanisms. Table 4 and the post-weighting check make the sampling qualifications concrete. Figure 4 should receive a full landscape page or be split into its two rows rather than compressed to a small panel. Existing audited single-model figures can remain supplementary where they answer a specific question; reproducing all of them would obscure the integrated findings.

This completes the analytical foundation for Step 3. The final PDF remains to be created separately.
'''
for k,v in {'__PROFILE_TABLE__':table,**{f'__FIG{i}__':f'![Figure {i}](<{O/"figures"/name}>)' for i,name in enumerate(['fig01_unified_brand_comparison.png','fig02_diversity_and_cohesion.png','fig03_complementary_similarity.png','fig04_representative_cross_model_examples.png'],1)}}.items():text=text.replace(k,v)
verification=json.loads((O/'verification_summary.json').read_text())
for key,val in [('__PASSED__',verification['passed']),('__CHECKS__',verification['checks']),('__PRESERVED__',verification['preserved_original_and_audit_files'])]:text=text.replace(key,f'{val:,}')
(O/'BRAND_ANALYSIS.md').write_text(text)
print('Wrote',O/'BRAND_ANALYSIS.md','words',len(text.split()))
