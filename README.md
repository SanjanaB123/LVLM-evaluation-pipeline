# LVLM-evaluation-pipeline
Multimodal Reasoning Audit Pipeline for Vision-Language Models (VLMs) — evaluates VLMs beyond final-answer accuracy using vision-dependence tests, hallucination checks, and claim-level faithfulness scoring across VQA/OCR/document benchmarks.
 
 # LVLM Evaluation Pipeline
 
 This repository provides a pipeline for processing, annotating, and evaluating 2D footwear images using Segment Anything Model (SAM) and custom evidence type identification. The workflow includes image format conversion, mask generation, annotation, pre-annotation, review, and statistics.
 
 ## Pipeline Overview
 
 1. **TIFF to PNG Conversion**
	 - Converts raw `.tiff` images to `.png` format, organizing them by case.
	 - See: `scripts/data_preprocessing.ipynb`
 
 2. **Mask Generation using SAM**
	 - Generates segmentation masks for each image using Meta's Segment Anything Model (SAM).
	 - See: `scripts/01_generate_sam_masks.py`
 
 3. **Annotation of Masks**
	 - Provides an interactive tool to annotate masks with evidence types (e.g., outer boundary, pattern region, unclear region) based on a schema.
	 - See: `scripts/02_annotate_sam_masks.py`
 
 4. **Automated Pre-Annotation**
	 - Suggests evidence regions automatically using mask properties and strict rules (e.g., largest mask as outer boundary).
	 - See: `scripts/03_sam_based_preannotation.py`
 
 5. **Manual Review of Suggestions**
	 - Interactive review tool to accept, reject, or modify pre-annotation suggestions.
	 - See: `scripts/04_review_suggestions.py`
 
 6. **Statistics and Quality Check**
	 - Summarizes evidence type distributions and per-image statistics for pre-annotations.
	 - See: `scripts/check_suggestions.py`
 
 ## Folder Structure
 
 - `data/raw/` - Raw TIFF images
 - `data/processed/` - PNG images (after conversion)
 - `data/suggestions_v3/` - Pre-annotation suggestions
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
 
 ### 3. Annotate Masks
 ```bash
 python scripts/02_annotate_sam_masks.py --data_dir data/processed --masks_dir data/masks --output_dir data/annotations --evidence_schema config/evidence_schema.json
 ```
 
 ### 4. Automated Pre-Annotation
 ```bash
 python scripts/03_sam_based_preannotation.py --masks_dir data/masks --output_dir data/suggestions_v3 --evidence_schema config/evidence_schema.json
 ```
 
 ### 5. Manual Review
 ```bash
 python scripts/04_review_suggestions.py --data_dir data/processed --masks_dir data/masks --suggestions_dir data/suggestions_v3 --output_dir data/reviewed_annotations
 ```
 
 ### 6. Statistics
 ```bash
 python scripts/check_suggestions.py
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
 
