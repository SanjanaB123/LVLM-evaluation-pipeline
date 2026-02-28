import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm

def generate_masks_for_dataset(data_dir, output_dir, sam_checkpoint):
    """
    Generate SAM masks for all footprint images.
    """
    
    # Load SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("⚠️  Warning: Running on CPU - this will be SLOW!")
        print("   Consider requesting a GPU node")
    
    print(f"Loading SAM model from {sam_checkpoint}...")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    print("✓ SAM model loaded")
    
    # Configure SAM for footprint segmentation
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all case directories
    case_dirs = sorted([d for d in Path(data_dir).glob("*") if d.is_dir()])

    # ⭐ TEST MODE: Process only first 10% of cases
    total_cases = len(case_dirs)
    test_size = max(1, total_cases // 10)  # At least 1 case
    case_dirs = case_dirs[:test_size]

    print(f"🧪 TEST MODE: Processing {len(case_dirs)} out of {total_cases} cases (10%)")
    
    if len(case_dirs) == 0:
        print(f"❌ No case directories found in {data_dir}")
        print("   Make sure your images are in data/processed/")
        return
    
    print(f"\nFound {len(case_dirs)} cases to process")
    
    total_images = 0
    processed_images = 0
    skipped_images = 0
    
    # Process each case
    for case_dir in tqdm(case_dirs, desc="Processing cases"):
        case_id = case_dir.name
        
        case_output = Path(output_dir) / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = sorted(list(case_dir.glob("*.png")))
        
        if len(image_files) == 0:
            print(f"\n⚠️  No PNG files found in {case_dir}")
            continue
        
        total_images += len(image_files)
        
        # Process each view
        for img_path in tqdm(image_files, desc=f"  {case_id}", leave=False):
            view_id = img_path.stem
            
            # Check if already processed
            output_file = case_output / f"{view_id}_masks.npz"
            if output_file.exists():
                skipped_images += 1
                continue
            
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    print(f"\n⚠️  Failed to load {img_path}")
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Generate masks
                masks = mask_generator.generate(image_rgb)
                
                if len(masks) == 0:
                    print(f"\n⚠️  No masks generated for {case_id}/{view_id}")
                    continue
                
                # Save masks in compressed format
                np.savez_compressed(
                    output_file,
                    masks=np.array([m['segmentation'] for m in masks]),
                    scores=np.array([m['stability_score'] for m in masks]),
                    areas=np.array([m['area'] for m in masks]),
                    bboxes=np.array([m['bbox'] for m in masks]),
                )
                
                # Save metadata
                metadata = {
                    'case_id': case_id,
                    'view_id': view_id,
                    'num_masks': len(masks),
                    'image_shape': list(image_rgb.shape),
                    'evidence_types_available': [
                        'outer_boundary',
                        'pattern_region', 
                        'unclear_region'
                    ]
                }
                
                with open(case_output / f"{view_id}_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                processed_images += 1
                
            except Exception as e:
                print(f"\n❌ Error processing {case_id}/{view_id}: {e}")
                continue
    
    # Summary
    print("\n" + "="*60)
    print("SAM Mask Generation Complete!")
    print("="*60)
    print(f"Total images found:     {total_images}")
    print(f"Newly processed:        {processed_images}")
    print(f"Previously processed:   {skipped_images}")
    print(f"Output directory:       {output_dir}")
    print("="*60)

if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data/processed"
    OUTPUT_DIR = "data/masks"
    SAM_CHECKPOINT = "models/sam_vit_h_4b8939.pth"
    
    # Check if SAM checkpoint exists
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"❌ SAM checkpoint not found at {SAM_CHECKPOINT}")
        print("Expected location: ~/forensic-vlm-project/models/sam_vit_h_4b8939.pth")
        exit(1)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found at {DATA_DIR}")
        print("Expected location: ~/forensic-vlm-project/data/processed/")
        exit(1)
    
    # Run
    print("Starting SAM mask generation...")
    print(f"Input:  {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Model:  {SAM_CHECKPOINT}")
    print()
    
    generate_masks_for_dataset(DATA_DIR, OUTPUT_DIR, SAM_CHECKPOINT)