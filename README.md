# Multimodal Reasoning Audit Pipeline for Vision-Language Models

A pipeline for evaluating Vision-Language Models (VLMs) on forensic footprint image analysis. Goes beyond final-answer accuracy to test vision-dependence, detect hallucinations, and score claim-level faithfulness using a **2-call Reasoner-Verifier loop**.

## Dataset

- **Source:** [CSAFE 2D Footwear Outsole Study](https://data.csafe.iastate.edu/2DFootwearOutsoleStudy/)
- **300 forensic footprint cases** (left/right variants, 5 views each)
- **90 cases manually annotated** using SAM-assisted ground truth creation
- Three evidence types per the annotation schema:
  - `outer_boundary` - complete outer contour (size estimation, overlap detection)
  - `pattern_region` - visible tread/sole patterns (shoe ID, pattern matching)
  - `unclear_region` - ambiguous/low-quality areas (uncertainty quantification)

> The `data/` folder contains a **sample of 1 case** (00101L/R) to illustrate the data format and pipeline outputs. Full dataset resides on the compute cluster.

## Pipeline

```
TIFF Images -> PNG Conversion -> SAM Masks -> Pre-Annotation -> Manual Review -> Benchmark -> VLM Evaluation -> Visualization
```

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Format Conversion | `data_preprocessing.ipynb` | Converts raw TIFF images to PNG, organized by case |
| 2. Mask Generation | `scripts/01_generate_sam_masks.py` | Generates segmentation masks using SAM (ViT-H) with tuned thresholds |
| 3. Pre-Annotation | `scripts/02_preannotation.py` | Rules-based classification of masks into evidence types using area, score, and rank heuristics |
| 4. Manual Review | `scripts/03_review_suggestions.py` | CLI tool with visual overlays for reviewing/accepting/rejecting suggestions |
| 5. Benchmark Creation | `scripts/04_create_benchmark.py` | Builds evaluation benchmark with train/val/test splits (70/15/15), grouped by case to prevent L/R leakage |
| 6. VLM Evaluation | `scripts/05_vlm_evaluation.py` | 2-call evaluation: Reasoner analyzes images, Verifier checks for hallucinations and overclaiming. Computes dual grounding scores (reasoner vs verifier-corrected) |
| 7. Stress Testing | `scripts/07_generate_stress_tests.py` | Generates degraded images (blur, noise, low contrast, crop, blank) for RQ5 uncertainty/abstention testing |
| 8. Visualization | `scripts/06_visualize_results.py` | Generates publication-ready figures for all RQs |

### Evaluation Architecture

The VLM evaluation uses a **Reasoner-Verifier loop** with GPT-4o:

1. **Reasoner (Call 1):** Receives footprint image(s) and a hypothesis prompt. Outputs structured JSON with evidence types found, confidence scores, and an abstention flag.
2. **Verifier (Call 2):** Receives the Reasoner's claims and a different view of the same footprint. Flags hallucinated regions, overclaimed evidence, and missed findings. Outputs corrected evidence labels using canonical types.

This two-call design separates generation from verification, enabling both hallucination detection (RQ4) and direct measurement of verifier impact on grounding (RQ6).

## Research Questions

| RQ | Question | Metric | Status |
|----|----------|--------|--------|
| RQ1 | How does domain shift affect VLM performance on forensic imagery? | Failure mode analysis | Weak - no baseline comparison |
| RQ2 | Does multi-view input improve evidence detection? | F1 delta (multi vs. single-view) | Addressed |
| RQ3 | How well are VLM findings grounded against gold labels? | Precision / Recall / F1 (reasoner + verifier-corrected) | Addressed |
| RQ4 | What is the hallucination rate in VLM forensic analysis? | Hallucination rate, verification verdict | Addressed |
| RQ5 | Do VLMs appropriately express uncertainty? | Abstention rate, confidence calibration, stress test performance | Addressed |
| RQ6 | Does adding a verifier improve reasoning reliability? | F1 improvement (reasoner vs verifier-corrected) | Addressed |

## Results

Test split: 56 items, 14 cases, GPT-4o, `detail=high` for all views. 0 items skipped.

### Grounding Performance (RQ3)

**Reasoner (raw free-text labels, normalized):**

| Evidence Type | Precision | Recall | F1 |
|---------------|-----------|--------|----|
| `outer_boundary` | 1.000 | 0.089 | 0.164 |
| `pattern_region` | 1.000 | 0.964 | 0.982 |
| `unclear_region` | 1.000 | 0.750 | 0.857 |
| **Overall** | **0.964** | **0.601** | **0.727** |

**After Verifier Correction (canonical labels):**

| Evidence Type | Precision | Recall | F1 |
|---------------|-----------|--------|----|
| `outer_boundary` | 1.000 | 0.393 | 0.564 |
| `pattern_region` | 1.000 | 0.357 | 0.526 |
| `unclear_region` | 1.000 | 0.571 | 0.727 |
| **Overall** | **1.000** | **0.476** | **0.476** |

### Hallucination & Verification (RQ4)

| Metric | Value |
|--------|-------|
| Hallucination Rate | 3.6% (2/56 items) |
| Overclaiming Rate | 1.8% (1/56 items) |
| Verifier Verdict | 55/56 supported |

### Multi-View vs Single-View (RQ2)

Both conditions use `detail=high` to eliminate resolution as a confound.

| Condition | Items | Avg F1 |
|-----------|-------|--------|
| Single-view | 42 | 0.812 |
| Multi-view | 14 | 0.471 |
| **Delta** | - | **-0.341** |

### Uncertainty Calibration (RQ5)

**On clean images:**

| Metric | Value |
|--------|-------|
| Abstention Rate | 0.0% |
| Avg. Confidence | 0.818 |
| Confidence at 0.8 bin (n=48) | Actual F1 = 0.78 |
| Confidence at 0.9 bin (n=8) | Actual F1 = 0.42 |

**Stress test (degraded images, 5 cases x 5 degradations = 25 items):**

| Degradation | Avg Confidence | Abstention Rate | Avg F1 |
|-------------|---------------|-----------------|--------|
| Heavy blur | 0.80 | 0% | 0.680 |
| Low contrast | 0.78 | 0% | 0.620 |
| Heavy noise | 0.80 | 0% | 0.560 |
| Center crop | 0.80 | 0% | 0.680 |
| Blank image | 0.90 | 0% (80% refused by safety filter) | 0.500 |
| *Clean baseline* | *0.818* | *0%* | *0.727* |

### Verification Impact (RQ6)

| Metric | Value |
|--------|-------|
| Reasoner avg F1 | 0.727 |
| Verifier-corrected avg F1 | 0.476 |
| **F1 change** | **-0.251** |

The verifier maintains perfect precision (1.0) but overcorrects, dropping recall significantly. It removes valid evidence labels rather than just flagging hallucinations.

### Key Findings

1. **Perfect precision, variable recall.** The model never hallucinates evidence types (precision = 1.0 across all types) but frequently fails to label them, especially `outer_boundary` (8.9% recall). The model describes boundaries but uses free-text instead of canonical labels.
2. **Multi-view reasoning degrades performance.** Multi-view items score 34.1% lower F1 than single-view at the same image detail level. GPT-4o cannot effectively cross-reference multiple forensic images simultaneously.
3. **Zero uncertainty calibration.** The model outputs confidence of 0.8 regardless of image quality, never abstains, and is more confident on blank images (0.9) than real footprints (0.8). The safety filter catches 80% of blank images, but the model's reasoning system has no uncertainty awareness.
4. **Verifier overcorrects.** The verification step reduces hallucinations (3.6% rate) but at the cost of recall, dropping overall F1 by 25.1%. The verifier is too conservative, removing correct labels along with incorrect ones.
5. **Binary confidence behavior.** The model uses only two confidence levels (0.8 and 0.9) with no correlation to actual performance. At 0.9 confidence, actual F1 is only 0.42.

## Repository Structure

```
├── config/
│   └── evidence_schema.json          # Evidence types + annotation rules
├── data/
│   ├── raw/                          # Raw TIFF images (sample: 00101L, 00101R)
│   ├── masks/                        # SAM-generated masks (.npz + metadata)
│   ├── suggestions/                  # Pre-annotation suggestions
│   ├── annotations/                  # Reviewed annotations (ground truth)
│   ├── visualizations/               # Mask overlay visualizations
│   ├── benchmark_v1.json             # Evaluation benchmark (360 items)
│   ├── case_splits.json              # Train/val/test splits
│   └── stress_test/
│       └── benchmark_stress.json     # Stress test benchmark for RQ5
├── scripts/
│   ├── 01_generate_sam_masks.py      # SAM ViT-H mask generation
│   ├── 02_preannotation.py           # Rules-based evidence classification
│   ├── 03_review_suggestions.py      # Interactive annotation review CLI
│   ├── 04_create_benchmark.py        # Benchmark assembly + case splits
│   ├── 05_vlm_evaluation.py          # Reasoner-Verifier evaluation loop
│   ├── 06_visualize_results.py       # Publication figure generation
│   └── 07_generate_stress_tests.py   # Degraded image generator for RQ5
├── data_preprocessing.ipynb          # TIFF to PNG conversion
├── requirements.txt
└── README.md
```

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Convert TIFF to PNG
#    Run data_preprocessing.ipynb

# 2. Generate SAM masks
python scripts/01_generate_sam_masks.py

# 3. Pre-annotate
python scripts/02_preannotation.py

# 4. Manual review
python scripts/03_review_suggestions.py

# 5. Create benchmark
python scripts/04_create_benchmark.py

# 6. Run VLM evaluation
python scripts/05_vlm_evaluation.py --split test
python scripts/05_vlm_evaluation.py --dry-run          # no API calls

# 7. Generate stress tests (RQ5)
python scripts/07_generate_stress_tests.py
python scripts/05_vlm_evaluation.py --benchmark data/stress_test/benchmark_stress.json

# 8. Generate figures
python scripts/06_visualize_results.py --results data/vlm_results/results.json
python scripts/06_visualize_results.py --results data/vlm_results/abstention.json --output-dir results/figures_stress
```

## Requirements

- Python 3.8+
- SAM checkpoint (ViT-H) for mask generation
- OpenAI API key for VLM evaluation (GPT-4o)

```bash
pip install -r requirements.txt
```
