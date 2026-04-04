# Multimodal Reasoning Audit Pipeline for Vision-Language Models

A pipeline for evaluating Vision-Language Models (VLMs) on forensic footprint image analysis. Goes beyond final-answer accuracy to test vision-dependence, detect hallucinations, and score claim-level faithfulness using a **2-call Reasoner-Verifier loop**.

## Dataset

- **Source:** [CSAFE 2D Footwear Outsole Study](https://data.csafe.iastate.edu/2DFootwearOutsoleStudy/)
- **300 forensic footprint cases** (left/right variants, 5 views each)
- **90 cases manually annotated** using SAM-assisted ground truth creation
- Three evidence types per the annotation schema:
  - `outer_boundary` — complete outer contour (size estimation, overlap detection)
  - `pattern_region` — visible tread/sole patterns (shoe ID, pattern matching)
  - `unclear_region` — ambiguous/low-quality areas (uncertainty quantification)

> The `data/` folder contains a **sample of 1 case** (00101L/R) to illustrate the data format and pipeline outputs.

## Pipeline

```
TIFF Images → PNG Conversion → SAM Masks → Pre-Annotation → Manual Review → Benchmark → VLM Evaluation
```

| Stage | Script | Description |
|-------|--------|-------------|
| 1. Format Conversion | `data_preprocessing.ipynb` | Converts raw TIFF images to PNG, organized by case |
| 2. Mask Generation | `scripts/01_generate_sam_masks.py` | Generates segmentation masks using SAM (ViT-H) with tuned thresholds |
| 3. Pre-Annotation | `scripts/02_preannotation.py` | Rules-based classification of masks into evidence types using area, score, and rank heuristics |
| 4. Manual Review | `scripts/03_review_suggestions.py` | CLI tool with visual overlays for reviewing/accepting/rejecting suggestions |
| 5. Benchmark Creation | `scripts/04_create_benchmark.py` | Builds evaluation benchmark with train/val/test splits (70/15/15), grouped by case to prevent L/R leakage |
| 6. VLM Evaluation | `scripts/05_vlm_evaluation.py` | 2-call evaluation: Reasoner analyzes images, Verifier checks for hallucinations and overclaiming |

### Evaluation Architecture

The VLM evaluation uses a **Reasoner-Verifier loop** with GPT-4o:

1. **Reasoner (Call 1):** Receives footprint image(s) and a hypothesis prompt. Outputs structured JSON with evidence types found, confidence scores, and an abstention flag.
2. **Verifier (Call 2):** Receives the Reasoner's claims and the original image(s). Flags hallucinated regions, overclaimed evidence, and missed findings.

This two-call design separates generation from verification, enabling fine-grained hallucination detection.

## Research Questions

| RQ | Question | Metric |
|----|----------|--------|
| RQ1 | How does domain shift affect VLM performance on forensic imagery? | Failure mode analysis |
| RQ2 | Does multi-view input improve evidence detection? | F1 improvement (multi vs. single-view) |
| RQ3 | How well are VLM findings grounded against gold labels? | Precision / Recall / F1 |
| RQ4 | What is the hallucination rate in VLM forensic analysis? | Hallucination rate, verification verdict |
| RQ5 | Do VLMs appropriately express uncertainty? | Abstention rate, confidence calibration |
| RQ6 | Are VLM findings consistent across views of the same case? | Cross-view consistency score |

## Results

Test split (56 items, 14 cases, GPT-4o). 0 items skipped.

### Grounding Performance (RQ3)

| Evidence Type | Precision | Recall | F1 |
|---------------|-----------|--------|----|
| `outer_boundary` | 0.609 | 0.525 | 0.564 |
| `pattern_region` | 0.880 | 0.946 | 0.912 |
| `unclear_region` | 1.000 | 0.732 | 0.845 |
| **Overall** | **0.830** | **0.734** | **0.774** |

### Hallucination & Verification (RQ4)

| Metric | Value |
|--------|-------|
| Hallucination Rate | 3.6% (2/56 items) |
| Overclaiming Rate | 1.8% (1/56 items) |
| Verifier Verdict | 55/56 supported |

### Multi-View vs Single-View (RQ2)

| Condition | Items | Avg F1 |
|-----------|-------|--------|
| Single-view | 42 | 0.788 |
| Multi-view | 14 | 0.564 |
| Delta | — | -0.224 |

### Uncertainty Calibration (RQ5)

| Metric | Value |
|--------|-------|
| Abstention Rate | 0.0% |
| Avg. Confidence (F1 = 1.0) | 0.800 |
| Avg. Confidence (F1 < 1.0) | 0.820 |

### Key Findings

- **High precision, moderate recall.** Precision is strong across all types (overall 0.830), but recall lags behind (overall 0.734). `outer_boundary` is the weakest (P=0.609, R=0.525, F1=0.564), dragging overall F1 to 0.774.
- **Multi-view hurts.** Multi-view items score 22.4% lower F1 than single-view, likely because images are sent at `detail="low"` to manage token cost, losing subtle boundary and degradation cues.
- **Unclear regions underdetected.** `unclear_region` has perfect precision but recall of 0.732 — the model misses degraded/ambiguous areas in roughly 1 in 4 items.
- **Weak confidence calibration.** The model is slightly more confident on incorrect predictions (0.82) than correct ones (0.80), and never abstains.

## Repository Structure

```
├── config/
│   └── evidence_schema.json        # Evidence types + annotation rules
├── data/
│   ├── raw/                        # Raw TIFF images (sample: 00101L, 00101R)
│   ├── masks/                      # SAM-generated masks (.npz + metadata)
│   ├── suggestions/                # Pre-annotation suggestions
│   ├── annotations/                # Reviewed annotations (ground truth)
│   ├── visualizations/             # Mask overlay visualizations
│   ├── benchmark_v1.json           # Evaluation benchmark
│   └── case_splits.json            # Train/val/test splits
├── vlm_results/
│   └── results.json                # GPT-4o evaluation results (test split)
├── scripts/
│   ├── 01_generate_sam_masks.py
│   ├── 02_preannotation.py
│   ├── 03_review_suggestions.py
│   ├── 04_create_benchmark.py
│   └── 05_vlm_evaluation.py
├── data_preprocessing.ipynb
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
python scripts/01_generate_sam_masks.py --data_dir data/processed --output_dir data/masks --sam_checkpoint <path>

# 3. Pre-annotate
python scripts/02_preannotation.py --masks_dir data/masks --output_dir data/suggestions --evidence_schema config/evidence_schema.json

# 4. Manual review
python scripts/03_review_suggestions.py --data_dir data/processed --masks_dir data/masks --suggestions_dir data/suggestions --output_dir data/annotations

# 5. Create benchmark
python scripts/04_create_benchmark.py

# 6. Run VLM evaluation
python scripts/05_vlm_evaluation.py              # all items
python scripts/05_vlm_evaluation.py --split test  # test split only
python scripts/05_vlm_evaluation.py --dry-run     # no API calls
```

## Requirements

- Python 3.8+
- SAM checkpoint (ViT-H) for mask generation
- OpenAI API key for VLM evaluation (GPT-4o)

```bash
pip install -r requirements.txt
```
