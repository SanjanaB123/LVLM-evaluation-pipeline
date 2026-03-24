import numpy as np
import cv2
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt

class VisualCLIReviewer:
    """
    CLI reviewer that generates visualization images
    You view the saved images, then decide in terminal
    """
    
    def __init__(self, data_dir, masks_dir, suggestions_dir, output_dir, viz_dir):
        self.data_dir = Path(data_dir)
        self.masks_dir = Path(masks_dir)
        self.suggestions_dir = Path(suggestions_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.viz_dir = Path(viz_dir)
        self.viz_dir.mkdir(exist_ok=True)
        
        self.suggestion_files = sorted(list(self.suggestions_dir.glob("*_suggestions.json")))
        
        if not self.suggestion_files:
            print(f"❌ No suggestion files found in {self.suggestions_dir}")
            sys.exit(1)
        
        self.stats = {
            'total_reviewed': 0,
            'accepted': 0,
            'rejected': 0,
            'modified': 0,
            'images_completed': 0
        }
        
        print(f"\n{'='*70}")
        print(f"VISUAL CLI ANNOTATION TOOL")
        print(f"{'='*70}")
        print(f"Total images: {len(self.suggestion_files)}")
        print(f"Visualizations will be saved to: {self.viz_dir}")
        print(f"{'='*70}\n")
    
    def create_visualization(self, case_id, view_id, suggestions, masks):
        """Create a grid visualization of all suggestions"""
        
        # Load image
        img_path = self.data_dir / case_id / f"{view_id}.png"
        if not img_path.exists():
            print(f"⚠️  Image not found: {img_path}")
            return None
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Group suggestions by type
        by_type = {
            'outer_boundary': [],
            'pattern_region': [],
            'unclear_region': []
        }
        
        for s in suggestions:
            t = s['evidence_type']
            if t in by_type:
                by_type[t].append(s)
        
        # Create figure with subplots
        total_subs = min(12, len(suggestions))  # Show max 12 suggestions
        rows = 3
        cols = 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        fig.suptitle(f"Suggestions for {case_id}/{view_id}\nTotal: {len(suggestions)} suggestions", 
                    fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        # Color mapping
        colors = {
            'outer_boundary': [255, 0, 0],      # Red
            'pattern_region': [0, 255, 0],      # Green
            'unclear_region': [255, 255, 0]     # Yellow
        }
        
        # Select which suggestions to show
        # Priority: 1 boundary, top patterns, top unclear
        to_show = []
        
        if by_type['outer_boundary']:
            to_show.append(by_type['outer_boundary'][0])
        
        by_type['pattern_region'].sort(key=lambda x: x['confidence'], reverse=True)
        to_show.extend(by_type['pattern_region'][:7])  # Top 7 patterns
        
        by_type['unclear_region'].sort(key=lambda x: x['confidence'], reverse=True)
        to_show.extend(by_type['unclear_region'][:4])  # Top 4 unclear
        
        # Plot each suggestion
        for idx, ax in enumerate(axes):
            if idx >= len(to_show):
                ax.axis('off')
                continue
            
            s = to_show[idx]
            mask_idx = s['mask_idx']
            
            if mask_idx >= len(masks):
                ax.axis('off')
                continue
            
            # Create overlay
            overlay = image.copy()
            mask = masks[mask_idx]
            color = colors[s['evidence_type']]
            overlay[mask] = color
            
            result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
            
            # Plot
            ax.imshow(result)
            title = f"#{idx+1}: {s['evidence_type']}\n"
            title += f"Conf: {s['confidence']:.2f}\n"
            title += f"Area: {mask.sum():.0f}px"
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        
        # Save figure
        viz_file = self.viz_dir / f"{case_id}_{view_id}_viz.png"
        plt.tight_layout()
        plt.savefig(viz_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        return viz_file
    
    def review_image(self, suggestion_file):
        """Review one image with visualization"""
        
        # Parse filename
        filename = suggestion_file.stem.replace('_suggestions', '')
        parts = filename.rsplit('_', 1)
        case_id = parts[0]
        view_id = parts[1] if len(parts) > 1 else '01'
        
        # Load suggestions
        with open(suggestion_file) as f:
            suggestions = json.load(f)
        
        suggestions = [s for s in suggestions if s.get('status') == 'suggested']
        
        if not suggestions:
            print(f"  ⏭️  Already completed, skipping...")
            return
        
        # Load masks
        masks_file = self.masks_dir / case_id / f"{view_id}_masks.npz"
        if masks_file.exists():
            masks_data = np.load(masks_file)
            masks = masks_data['masks']
        else:
            print(f"⚠️  No masks found for {case_id}/{view_id}")
            return
        
        print(f"\n{'='*70}")
        print(f"📸 Image: {case_id}/{view_id}")
        print(f"{'='*70}")
        
        # Create visualization
        print("Creating visualization...")
        viz_file = self.create_visualization(case_id, view_id, suggestions, masks)
        
        if viz_file:
            print(f"✓ Visualization saved to: {viz_file}")
            print(f"\n📌 INSTRUCTIONS:")
            print(f"   1. Open the image in a viewer:")
            print(f"      {viz_file}")
            print(f"   2. Look at the overlaid masks (Red=boundary, Green=pattern, Yellow=unclear)")
            print(f"   3. Return here and choose an option below\n")
        
        # Show summary
        type_counts = {}
        for s in suggestions:
            t = s['evidence_type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print("Summary:")
        for t in ['outer_boundary', 'pattern_region', 'unclear_region']:
            if t in type_counts:
                print(f"  {t}: {type_counts[t]}")
        print()
        
        # Wait for user to view image
        input("Press Enter after viewing the image...")
        
        # Ask for decision
        print("\nReview options:")
        print("  1. Accept ALL suggestions as-is")
        print("  2. Review each suggestion individually (not recommended with so many)")
        print("  3. Smart accept (1 boundary, top 4 patterns, top 2 unclear)")
        print("  4. Custom accept (specify how many of each type)")
        print("  5. Skip entire image")
        print()
        
        choice = input("Choose [1-5]: ").strip()
        
        annotations = []
        
        if choice == '1':
            # Accept all
            for s in suggestions:
                annotations.append(self.create_annotation(case_id, view_id, s, 'accepted'))
                self.stats['accepted'] += 1
            print(f"✓ Accepted all {len(suggestions)} suggestions")
        
        elif choice == '2':
            # Individual review (will be tedious!)
            annotations = self.review_individually(case_id, view_id, suggestions)
        
        elif choice == '3':
            # Smart accept
            annotations = self.smart_accept(case_id, view_id, suggestions, 
                                          n_boundary=1, n_pattern=4, n_unclear=2)
            print(f"✓ Smart accepted: {len(annotations)} annotations")
        
        elif choice == '4':
            # Custom accept
            annotations = self.custom_accept(case_id, view_id, suggestions, type_counts)
        
        elif choice == '5':
            print("⏭️  Skipped")
            return
        
        else:
            print("Invalid choice, skipping...")
            return
        
        # Save annotations
        if annotations:
            ann_file = self.output_dir / f"{case_id}_{view_id}_annotations.json"
            with open(ann_file, 'w') as f:
                json.dump(annotations, f, indent=2)
            print(f"✓ Saved {len(annotations)} annotations")
        
        self.stats['images_completed'] += 1
        self.stats['total_reviewed'] += len(suggestions)
    
    def custom_accept(self, case_id, view_id, suggestions, type_counts):
        """Let user specify how many of each type to accept"""
        
        print("\nCustom accept:")
        
        # Get user input for each type
        n_boundary = 1
        if type_counts.get('outer_boundary', 0) > 0:
            inp = input(f"  How many outer_boundary to accept? (max {type_counts['outer_boundary']}, default 1): ").strip()
            n_boundary = int(inp) if inp.isdigit() else 1
        
        n_pattern = 4
        if type_counts.get('pattern_region', 0) > 0:
            inp = input(f"  How many pattern_region to accept? (max {type_counts['pattern_region']}, default 4): ").strip()
            n_pattern = int(inp) if inp.isdigit() else 4
        
        n_unclear = 2
        if type_counts.get('unclear_region', 0) > 0:
            inp = input(f"  How many unclear_region to accept? (max {type_counts['unclear_region']}, default 2): ").strip()
            n_unclear = int(inp) if inp.isdigit() else 2
        
        return self.smart_accept(case_id, view_id, suggestions, n_boundary, n_pattern, n_unclear)
    
    def smart_accept(self, case_id, view_id, suggestions, n_boundary=1, n_pattern=4, n_unclear=2):
        """Accept top N of each type"""
        
        annotations = []
        
        # Group by type
        by_type = {}
        for s in suggestions:
            t = s['evidence_type']
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(s)
        
        # Sort by confidence
        for t in by_type:
            by_type[t].sort(key=lambda x: x['confidence'], reverse=True)
        
        # Accept top N
        if 'outer_boundary' in by_type:
            for s in by_type['outer_boundary'][:n_boundary]:
                annotations.append(self.create_annotation(case_id, view_id, s, 'accepted'))
                self.stats['accepted'] += 1
        
        if 'pattern_region' in by_type:
            for s in by_type['pattern_region'][:n_pattern]:
                annotations.append(self.create_annotation(case_id, view_id, s, 'accepted'))
                self.stats['accepted'] += 1
        
        if 'unclear_region' in by_type:
            for s in by_type['unclear_region'][:n_unclear]:
                annotations.append(self.create_annotation(case_id, view_id, s, 'accepted'))
                self.stats['accepted'] += 1
        
        return annotations
    
    def review_individually(self, case_id, view_id, suggestions):
        """Review each suggestion (tedious!)"""
        
        annotations = []
        
        print("\n⚠️  This will be tedious with so many suggestions!")
        cont = input("Continue with individual review? [y/N]: ").strip().lower()
        
        if cont != 'y':
            return annotations
        
        for idx, s in enumerate(suggestions[:20]):  # Limit to first 20
            print(f"\nSuggestion {idx+1}: {s['evidence_type']} (conf: {s['confidence']:.2f})")
            action = input("  [a]ccept / [r]eject / [s]kip rest: ").strip().lower()
            
            if action == 'a':
                annotations.append(self.create_annotation(case_id, view_id, s, 'accepted'))
                self.stats['accepted'] += 1
            elif action == 'r':
                self.stats['rejected'] += 1
            elif action == 's':
                break
        
        return annotations
    
    def create_annotation(self, case_id, view_id, suggestion, status):
        """Create annotation dict"""
        return {
            'case_id': case_id,
            'view_id': view_id,
            'mask_idx': suggestion['mask_idx'],
            'evidence_type': suggestion['evidence_type'],
            'confidence': suggestion['confidence'],
            'reasoning': suggestion.get('reasoning', ''),
            'review_status': status,
            'original_type': suggestion.get('evidence_type')
        }
    
    def review_all(self):
        """Review all images"""
        
        for idx, sfile in enumerate(self.suggestion_files):
            print(f"\n{'='*70}")
            print(f"Progress: {idx + 1}/{len(self.suggestion_files)}")
            print(f"{'='*70}")
            
            self.review_image(sfile)
            
            # Show stats every 10 images
            if (idx + 1) % 10 == 0:
                self.show_stats()
        
        # Final stats
        print(f"\n{'='*70}")
        print("REVIEW COMPLETE!")
        print(f"{'='*70}")
        self.show_stats()
        print(f"\nVisualizations saved in: {self.viz_dir}")
    
    def show_stats(self):
        """Show statistics"""
        print(f"\nSession Statistics:")
        print(f"  Images completed: {self.stats['images_completed']}/{len(self.suggestion_files)}")
        print(f"  Total suggestions reviewed: {self.stats['total_reviewed']}")
        print(f"  Accepted: {self.stats['accepted']}")
        print(f"  Rejected: {self.stats['rejected']}")

def main():
    reviewer = VisualCLIReviewer(
        data_dir="data/processed",
        masks_dir="data/masks",
        suggestions_dir="data/suggestions",
        output_dir="data/annotations",
        viz_dir="data/visualizations"
    )
    
    reviewer.review_all()

if __name__ == "__main__":
    main()