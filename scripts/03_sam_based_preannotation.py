import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

class SAMBasedPreannotator:
    """
    VERSION 3: STRICT rules - only ONE outer_boundary per image
    """
    
    def __init__(self, masks_dir, output_dir, evidence_schema):
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        with open(evidence_schema) as f:
            schema = json.load(f)
        self.evidence_types = [et['id'] for et in schema['evidence_types']]
        
    def preannotate_image(self, masks, scores, bboxes):
        """
        Generate suggestions for ONE image
        
        STRICT STRATEGY:
        1. EXACTLY ONE outer_boundary = largest mask
        2. MULTIPLE pattern_regions = medium-sized, high quality
        3. unclear_regions = low quality or very small
        """
        
        # Calculate areas
        areas = np.array([mask.sum() for mask in masks])
        
        # Sort by area (largest first)
        sorted_indices = np.argsort(areas)[::-1]
        
        suggestions = []
        boundary_assigned = False
        
        for rank, idx in enumerate(sorted_indices):
            area = areas[idx]
            score = scores[idx]
            bbox = bboxes[idx]
            
            # Calculate properties
            x, y, w, h = bbox
            compactness = area / (w * h) if (w * h) > 0 else 0
            
            # ==========================================
            # Rule 1: OUTER_BOUNDARY (ONLY FIRST/LARGEST)
            # ==========================================
            if rank == 0 and score > 0.85 and not boundary_assigned:
                suggestions.append({
                    'mask_idx': int(idx),
                    'evidence_type': 'outer_boundary',
                    'confidence': 0.90,
                    'reasoning': f'Largest mask (area: {area:.0f}px, score: {score:.2f})',
                    'properties': {'area': int(area), 'score': float(score), 'bbox': [int(b) for b in bbox]}
                })
                boundary_assigned = True
                continue  # Skip to next mask
            
            # ==========================================
            # Rule 2: PATTERN_REGION
            # ==========================================
            # Skip if already used for boundary
            if boundary_assigned and idx == sorted_indices[0]:
                continue
            
            # Medium to large size (rank 1-30 typically)
            # High quality
            # Reasonable size (not too small, not too large)
            if (score > 0.88 and 
                500 < area < areas[sorted_indices[0]] * 0.8 and  # Smaller than boundary
                rank < 40):  # Top 40 masks
                
                suggestions.append({
                    'mask_idx': int(idx),
                    'evidence_type': 'pattern_region',
                    'confidence': 0.75,
                    'reasoning': f'Pattern area (area: {area:.0f}px, score: {score:.2f}, rank: {rank+1})',
                    'properties': {'area': int(area), 'score': float(score), 'bbox': [int(b) for b in bbox]}
                })
                continue
            
            # Lower threshold for pattern if very high quality
            if (score > 0.92 and 
                300 < area < areas[sorted_indices[0]] * 0.7 and
                rank < 50):
                
                suggestions.append({
                    'mask_idx': int(idx),
                    'evidence_type': 'pattern_region',
                    'confidence': 0.70,
                    'reasoning': f'High-quality pattern (area: {area:.0f}px, score: {score:.2f})',
                    'properties': {'area': int(area), 'score': float(score), 'bbox': [int(b) for b in bbox]}
                })
                continue
            
            # ==========================================
            # Rule 3: UNCLEAR_REGION
            # ==========================================
            # Low quality
            if score < 0.86:
                suggestions.append({
                    'mask_idx': int(idx),
                    'evidence_type': 'unclear_region',
                    'confidence': 0.60,
                    'reasoning': f'Low quality (score: {score:.2f})',
                    'properties': {'area': int(area), 'score': float(score), 'bbox': [int(b) for b in bbox]}
                })
                continue
            
            # Very small fragments
            if area < 400:
                suggestions.append({
                    'mask_idx': int(idx),
                    'evidence_type': 'unclear_region',
                    'confidence': 0.55,
                    'reasoning': f'Small fragment (area: {area:.0f}px)',
                    'properties': {'area': int(area), 'score': float(score), 'bbox': [int(b) for b in bbox]}
                })
                continue
            
            # Very low rank (likely noise)
            if rank > 60:
                suggestions.append({
                    'mask_idx': int(idx),
                    'evidence_type': 'unclear_region',
                    'confidence': 0.50,
                    'reasoning': f'Low-rank mask (rank: {rank+1})',
                    'properties': {'area': int(area), 'score': float(score), 'bbox': [int(b) for b in bbox]}
                })
                continue
        
        # FORCE: If no outer_boundary assigned yet (shouldn't happen), assign to largest
        if not boundary_assigned and len(sorted_indices) > 0:
            idx = sorted_indices[0]
            suggestions.insert(0, {
                'mask_idx': int(idx),
                'evidence_type': 'outer_boundary',
                'confidence': 0.85,
                'reasoning': f'Forced largest mask (area: {areas[idx]:.0f}px)',
                'properties': {'area': int(areas[idx]), 'score': float(scores[idx]), 'bbox': [int(b) for b in bboxes[idx]]}
            })
        
        # Mark all as 'suggested' status
        for s in suggestions:
            s['status'] = 'suggested'
        
        return suggestions
    
    def preannotate_case(self, case_id):
        """Process all views in a case"""
        
        case_dir = self.masks_dir / case_id
        mask_files = sorted(list(case_dir.glob("*_masks.npz")))
        
        for mask_file in mask_files:
            view_id = mask_file.stem.replace('_masks', '')
            
            # Load masks
            masks_data = np.load(mask_file)
            masks = masks_data['masks']
            scores = masks_data['scores']
            bboxes = masks_data['bboxes']
            
            # Generate suggestions
            suggestions = self.preannotate_image(masks, scores, bboxes)
            
            # Save
            output_file = self.output_dir / f"{case_id}_{view_id}_suggestions.json"
            with open(output_file, 'w') as f:
                json.dump(suggestions, f, indent=2)
            
            # Debug first 3 files
            if len(list(self.output_dir.glob("*.json"))) <= 3:
                type_counts = defaultdict(int)
                for s in suggestions:
                    type_counts[s['evidence_type']] += 1
                
                print(f"\n  {case_id}/{view_id}:")
                print(f"    Total masks: {len(masks)}")
                print(f"    Suggestions: {len(suggestions)}")
                for t in ['outer_boundary', 'pattern_region', 'unclear_region']:
                    print(f"      {t}: {type_counts[t]}")
    
    def preannotate_all(self):
        """Process all cases"""
        
        cases = sorted([d.name for d in self.masks_dir.glob("*") if d.is_dir()])
        
        print(f"Generating pre-annotations for {len(cases)} cases...")
        print("(Showing details for first 3 images)\n")
        
        stats = defaultdict(int)
        
        for case_id in tqdm(cases, desc="Processing"):
            self.preannotate_case(case_id)
        
        # Count total suggestions
        for sfile in self.output_dir.glob("*_suggestions.json"):
            with open(sfile) as f:
                suggestions = json.load(f)
            for s in suggestions:
                stats[s['evidence_type']] += 1
        
        # Print statistics
        print("\n" + "="*60)
        print("Pre-annotation Complete!")
        print("="*60)
        print(f"Total files: {len(list(self.output_dir.glob('*.json')))}")
        print(f"Total suggestions: {sum(stats.values())}")
        print(f"\nDistribution:")
        for etype in ['outer_boundary', 'pattern_region', 'unclear_region']:
            count = stats[etype]
            pct = (count / sum(stats.values())) * 100 if sum(stats.values()) > 0 else 0
            print(f"  {etype}: {count} ({pct:.1f}%)")
        
        # Sanity checks
        print(f"\n{'='*60}")
        num_files = len(list(self.output_dir.glob('*.json')))
        if stats['outer_boundary'] > num_files * 1.5:
            print("⚠️  WARNING: Too many outer_boundary!")
            print(f"   Expected ~{num_files}, got {stats['outer_boundary']}")
        
        if stats['pattern_region'] == 0:
            print("⚠️  WARNING: NO pattern_region detected!")
            print("   Your images might not have clear patterns,")
            print("   OR SAM masks don't align with patterns.")
        elif stats['pattern_region'] < num_files:
            print(f"⚠️  Note: Few pattern_region ({stats['pattern_region']} for {num_files} images)")
            print("   This might be normal if patterns are subtle.")
        else:
            print(f"✓ Looks good! {stats['pattern_region']} pattern_regions detected.")
        
        print(f"\nSaved to: {self.output_dir}")
        print("="*60)

def main():
    preannotator = SAMBasedPreannotator(
        masks_dir="data/masks",
        output_dir="data/suggestions_v3",
        evidence_schema="config/evidence_schema.json"
    )
    
    preannotator.preannotate_all()

if __name__ == "__main__":
    main()