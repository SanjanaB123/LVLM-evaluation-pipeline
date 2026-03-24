# LVLM-evaluation-pipeline
Multimodal Reasoning Audit Pipeline for Vision-Language Models (VLMs) — evaluates VLMs beyond final-answer accuracy using vision-dependence tests, hallucination checks, and claim-level faithfulness scoring across VQA/OCR/document benchmarks.

# LVLM Evaluation Pipeline

This repository provides a pipeline for processing, annotating, and evaluating 2D footwear images using Segment Anything Model (SAM) and custom evidence type identification. The workflow includes image format conversion, mask generation, automated pre-annotation, manual review, benchmark creation, and VLM evaluation.

## Pipeline Overview

1. **TIFF to PNG Conversion**
   - Converts raw `.tiff` images to `.png` format, organizing them by case.
   - See: `data_preprocessing.ipynb`

2. **Mask Generation using SAM**
   - Generates segmentation masks for each image using Meta's Segment Anything Model (SAM).
   - See: `scripts/01_generate_sam_masks.py`

3. **Automated Pre-Annotation**
   - Suggests evidence regions automatically using mask properties and strict rules (e.g., largest mask as outer boundary).
   - See: `scripts/02_preannotation.py`

4. **Manual Review of Suggestions**
   - Tool for manual review of pre-annotation suggestions.
   - See: `scripts/03_review_suggestions.py`

5. **Benchmark Creation**
   - Builds a benchmark JSON from reviewed annotations for VLM evaluation.
   - See: `scripts/04_create_benchmark.py`

6. **VLM Evaluation**
   - Runs the evaluation pipeline using GPT-4V, with options for test splits and dry runs.
   - See: `scripts/05_vlm_evaluation.py`

## Folder Structure

- `data/raw/` - Raw TIFF images
- `data/processed/` - PNG images (after conversion)
- `data/masks/` - Generated segmentation masks
- `data/annotations/` - Manual or automated annotations
- `data/benchmark_v1.json` - Benchmark for VLM evaluation
- `config/evidence_schema.json` - Evidence type schema
- `scripts/` - All pipeline scripts

## Usage

### 1. Convert TIFF to PNG
Run the notebook or use the function in `data_preprocessing.ipynb`:

```python
tiff_to_png_by_case(input_dir="data/raw", output_dir="data/processed")
```

### 2. Generate SAM Masks
```bash
python scripts/01_generate_sam_masks.py --data_dir data/processed --output_dir data/masks --sam_checkpoint <path_to_sam_checkpoint>
```

### 3. Automated Pre-Annotation
```bash
python scripts/02_preannotation.py --masks_dir data/masks --output_dir data/annotations --evidence_schema config/evidence_schema.json
```

### 4. Manual Review (Optional/Planned)
```bash
python scripts/03_review_suggestions.py --data_dir data/processed --masks_dir data/masks --suggestions_dir data/annotations --output_dir data/reviewed_annotations
```

### 5. Create Benchmark
```bash
python scripts/04_create_benchmark.py
```

### 6. VLM Evaluation
```bash
python scripts/05_vlm_evaluation.py
# or for test split only
python scripts/05_vlm_evaluation.py --split test
# or dry run (no API calls)
python scripts/05_vlm_evaluation.py --dry-run
```
 
## Requirements

- Python 3.8+
- segment-anything
- numpy, opencv-python, matplotlib, tqdm, Pillow

Install dependencies:
```bash
pip install -r requirements.txt
```

## Evidence Types

Defined in `config/evidence_schema.json`. Typical types:
- `outer_boundary`
- `pattern_region`
- `unclear_region`
 
