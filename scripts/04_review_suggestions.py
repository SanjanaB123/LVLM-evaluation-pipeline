import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import sys

class SuggestionReviewer:
    """
    Manual review tool for SAM-generated suggestions
    Allows accept/reject/modify for each suggestion
    """
    
    def __init__(self, data_dir, masks_dir, suggestions_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.masks_dir = Path(masks_dir)
        self.suggestions_dir = Path(suggestions_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Get all suggestion files
        self.suggestion_files = sorted(list(self.suggestions_dir.glob("*_suggestions.json")))
        
        if not self.suggestion_files:
            print(f"❌ No suggestion files found in {self.suggestions_dir}")
            print(f"   Make sure you ran: python scripts/03_sam_based_preannotation_v3.py")
            sys.exit(1)
        
        self.current_file_idx = 0
        self.current_suggestion_idx = 0
        
        # Statistics
        self.stats = {
            'total_reviewed': 0,
            'accepted': 0,
            'rejected': 0,
            'modified': 0,
            'images_completed': 0
        }
        
        # Load first file
        self.load_suggestion_file(self.suggestion_files[0])
        
        print(f"\n{'='*60}")
        print(f"Manual Review Tool Started")
        print(f"{'='*60}")
        print(f"Total images to review: {len(self.suggestion_files)}")
        print(f"{'='*60}\n")
    
    def load_suggestion_file(self, suggestion_file):
        """Load suggestions for one image"""
        
        # Parse filename: case_id_view_id_suggestions.json
        filename = suggestion_file.stem
        parts = filename.replace('_suggestions', '').rsplit('_', 1)
        
        if len(parts) != 2:
            print(f"⚠️  Warning: Could not parse filename {filename}")
            parts = [filename, '01']
        
        self.current_case = parts[0]
        self.current_view = parts[1]
        
        # Load image
        img_path = self.data_dir / self.current_case / f"{self.current_view}.png"
        
        if not img_path.exists():
            print(f"⚠️  Warning: Image not found at {img_path}")
            # Try alternative path
            alt_cases = list(self.data_dir.glob(f"*{self.current_case[-6:]}*"))
            if alt_cases:
                img_path = alt_cases[0] / f"{self.current_view}.png"
        
        if img_path.exists():
            self.image = cv2.imread(str(img_path))
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            print(f"❌ Could not load image for {self.current_case}/{self.current_view}")
            self.image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Load masks
        masks_file = self.masks_dir / self.current_case / f"{self.current_view}_masks.npz"
        
        if not masks_file.exists():
            print(f"⚠️  Warning: Masks not found at {masks_file}")
            self.masks = []
        else:
            masks_data = np.load(masks_file)
            self.masks = masks_data['masks']
        
        # Load suggestions
        with open(suggestion_file) as f:
            self.suggestions = json.load(f)
        
        # Filter out already-reviewed suggestions
        self.suggestions = [s for s in self.suggestions 
                           if s.get('status') == 'suggested']
        
        self.current_suggestion_idx = 0
        
        # Load existing annotations (to avoid duplicates)
        ann_file = self.output_dir / f"{self.current_case}_{self.current_view}_annotations.json"
        if ann_file.exists():
            with open(ann_file) as f:
                self.existing_annotations = json.load(f)
        else:
            self.existing_annotations = []
        
        print(f"\n→ Loaded: {self.current_case}/{self.current_view}")
        print(f"  Suggestions to review: {len(self.suggestions)}")
        print(f"  Existing annotations: {len(self.existing_annotations)}")
    
    def get_current_suggestion(self):
        """Get current suggestion"""
        if self.current_suggestion_idx < len(self.suggestions):
            return self.suggestions[self.current_suggestion_idx]
        return None
    
    def visualize_current_suggestion(self):
        """Show image with suggested mask highlighted"""
        
        suggestion = self.get_current_suggestion()
        if not suggestion:
            # No more suggestions in this image
            return self.image
        
        image = self.image.copy()
        mask_idx = suggestion['mask_idx']
        
        if mask_idx >= len(self.masks):
            print(f"⚠️  Warning: mask_idx {mask_idx} out of range (have {len(self.masks)} masks)")
            return image
        
        mask = self.masks[mask_idx]
        
        # Color based on suggested type
        colors = {
            'outer_boundary': [255, 0, 0],      # Red
            'pattern_region': [0, 255, 0],      # Green
            'unclear_region': [255, 255, 0]     # Yellow
        }
        color = colors.get(suggestion['evidence_type'], [255, 255, 255])
        
        # Create overlay
        overlay = image.copy()
        overlay[mask] = color
        image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        # Add text info
        text_lines = [
            f"Suggestion {self.current_suggestion_idx + 1}/{len(self.suggestions)}",
            f"Suggested: {suggestion['evidence_type']}",
            f"Confidence: {suggestion['confidence']:.2f}",
            f"Reason: {suggestion.get('reasoning', 'N/A')[:50]}..."
        ]
        
        y_offset = 30
        for line in text_lines:
            cv2.putText(image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, line, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            y_offset += 35
        
        return image
    
    def accept_suggestion(self):
        """Accept current suggestion as-is"""
        suggestion = self.get_current_suggestion()
        if not suggestion:
            return
        
        suggestion['status'] = 'accepted'
        self.save_annotation(suggestion)
        
        self.stats['accepted'] += 1
        self.stats['total_reviewed'] += 1
        
        print(f"✓ Accepted: {suggestion['evidence_type']} (mask {suggestion['mask_idx']})")
        
        self.next_suggestion()
    
    def reject_suggestion(self):
        """Reject current suggestion"""
        suggestion = self.get_current_suggestion()
        if not suggestion:
            return
        
        suggestion['status'] = 'rejected'
        
        self.stats['rejected'] += 1
        self.stats['total_reviewed'] += 1
        
        print(f"✗ Rejected: {suggestion['evidence_type']} (mask {suggestion['mask_idx']})")
        
        self.next_suggestion()
    
    def modify_suggestion(self, new_type):
        """Accept suggestion but change the evidence type"""
        suggestion = self.get_current_suggestion()
        if not suggestion:
            return
        
        old_type = suggestion['evidence_type']
        suggestion['evidence_type'] = new_type
        suggestion['status'] = 'modified'
        suggestion['original_type'] = old_type
        suggestion['confidence'] = 0.95  # High confidence after manual review
        
        self.save_annotation(suggestion)
        
        self.stats['modified'] += 1
        self.stats['total_reviewed'] += 1
        
        print(f"✓ Modified: {old_type} → {new_type} (mask {suggestion['mask_idx']})")
        
        self.next_suggestion()
    
    def save_annotation(self, suggestion):
        """Save accepted/modified annotation"""
        
        ann_file = self.output_dir / f"{self.current_case}_{self.current_view}_annotations.json"
        
        # Create annotation entry
        annotation = {
            'case_id': self.current_case,
            'view_id': self.current_view,
            'mask_idx': suggestion['mask_idx'],
            'evidence_type': suggestion['evidence_type'],
            'confidence': suggestion['confidence'],
            'reasoning': suggestion.get('reasoning', ''),
            'review_status': suggestion['status'],
            'original_type': suggestion.get('original_type', suggestion['evidence_type'])
        }
        
        # Add to existing annotations
        self.existing_annotations.append(annotation)
        
        # Save
        with open(ann_file, 'w') as f:
            json.dump(self.existing_annotations, f, indent=2)
    
    def next_suggestion(self):
        """Move to next suggestion"""
        if self.current_suggestion_idx < len(self.suggestions) - 1:
            self.current_suggestion_idx += 1
        else:
            self.finish_current_image()
    
    def finish_current_image(self):
        """Mark current image as complete and move to next"""
        self.stats['images_completed'] += 1
        
        print(f"\n{'='*60}")
        print(f"✓ Image {self.stats['images_completed']}/{len(self.suggestion_files)} completed!")
        print(f"  Accepted: {len(self.existing_annotations)} annotations")
        print(f"{'='*60}")
        
        self.next_file()
    
    def next_file(self):
        """Move to next image"""
        if self.current_file_idx < len(self.suggestion_files) - 1:
            self.current_file_idx += 1
            self.load_suggestion_file(self.suggestion_files[self.current_file_idx])
        else:
            self.show_final_stats()
            plt.close('all')
    
    def skip_image(self):
        """Skip entire current image"""
        print(f"\n⏭️  Skipping entire image: {self.current_case}/{self.current_view}")
        self.finish_current_image()
    
    def show_final_stats(self):
        """Show final statistics"""
        print(f"\n{'='*60}")
        print(f"REVIEW COMPLETE!")
        print(f"{'='*60}")
        print(f"Images completed: {self.stats['images_completed']}/{len(self.suggestion_files)}")
        print(f"Total reviewed: {self.stats['total_reviewed']}")
        print(f"  Accepted: {self.stats['accepted']}")
        print(f"  Modified: {self.stats['modified']}")
        print(f"  Rejected: {self.stats['rejected']}")
        print(f"{'='*60}")
        print(f"\n✓ Annotations saved to: {self.output_dir}")
        print(f"\nNext step: Check annotation quality with:")
        print(f"  python scripts/check_progress.py")
        print(f"{'='*60}\n")

def create_review_interface():
    """Create interactive matplotlib review interface"""
    
    reviewer = SuggestionReviewer(
        data_dir="data/processed",
        masks_dir="data/masks",
        suggestions_dir="data/suggestions_v3",
        output_dir="data/annotations"
    )
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Main image display
    ax = plt.axes([0.25, 0.15, 0.70, 0.80])
    img_display = ax.imshow(reviewer.visualize_current_suggestion())
    ax.axis('off')
    ax.set_title(f"Review: {reviewer.current_case}/{reviewer.current_view} | "
                f"Image {reviewer.current_file_idx + 1}/{len(reviewer.suggestion_files)}", 
                fontsize=14, fontweight='bold')
    
    # Create buttons
    button_width = 0.18
    button_height = 0.05
    button_left = 0.02
    
    ax_accept = plt.axes([button_left, 0.80, button_width, button_height])
    ax_reject = plt.axes([button_left, 0.73, button_width, button_height])
    
    ax_mod_outer = plt.axes([button_left, 0.62, button_width, button_height*0.8])
    ax_mod_pattern = plt.axes([button_left, 0.55, button_width, button_height*0.8])
    ax_mod_unclear = plt.axes([button_left, 0.48, button_width, button_height*0.8])
    
    ax_skip_sug = plt.axes([button_left, 0.38, button_width, button_height])
    ax_skip_img = plt.axes([button_left, 0.31, button_width, button_height])
    ax_next_img = plt.axes([button_left, 0.24, button_width, button_height])
    
    ax_quit = plt.axes([button_left, 0.10, button_width, button_height])
    
    btn_accept = Button(ax_accept, '✓ Accept (A)', color='lightgreen', hovercolor='green')
    btn_reject = Button(ax_reject, '✗ Reject (R)', color='lightcoral', hovercolor='red')
    
    btn_mod_outer = Button(ax_mod_outer, '→ Outer Boundary (1)', color='lightyellow')
    btn_mod_pattern = Button(ax_mod_pattern, '→ Pattern Region (2)', color='lightyellow')
    btn_mod_unclear = Button(ax_mod_unclear, '→ Unclear Region (3)', color='lightyellow')
    
    btn_skip_sug = Button(ax_skip_sug, 'Skip → (Space)', color='lightgray')
    btn_skip_img = Button(ax_skip_img, 'Skip Entire Image', color='orange')
    btn_next_img = Button(ax_next_img, 'Next Image (N)', color='lightblue')
    
    btn_quit = Button(ax_quit, 'Save & Quit (Q)', color='pink')
    
    # Update display function
    def update_display():
        img_display.set_data(reviewer.visualize_current_suggestion())
        ax.set_title(f"Review: {reviewer.current_case}/{reviewer.current_view} | "
                    f"Image {reviewer.current_file_idx + 1}/{len(reviewer.suggestion_files)} | "
                    f"Progress: {reviewer.stats['images_completed']}/{len(reviewer.suggestion_files)}", 
                    fontsize=14, fontweight='bold')
        fig.canvas.draw_idle()
    
    # Button callbacks
    def on_accept(event):
        reviewer.accept_suggestion()
        update_display()
    
    def on_reject(event):
        reviewer.reject_suggestion()
        update_display()
    
    def on_mod_outer(event):
        reviewer.modify_suggestion('outer_boundary')
        update_display()
    
    def on_mod_pattern(event):
        reviewer.modify_suggestion('pattern_region')
        update_display()
    
    def on_mod_unclear(event):
        reviewer.modify_suggestion('unclear_region')
        update_display()
    
    def on_skip_sug(event):
        reviewer.reject_suggestion()  # Same as reject
        update_display()
    
    def on_skip_img(event):
        reviewer.skip_image()
        update_display()
    
    def on_next_img(event):
        reviewer.finish_current_image()
        update_display()
    
    def on_quit(event):
        reviewer.show_final_stats()
        plt.close('all')
    
    # Keyboard shortcuts
    def on_key(event):
        if event.key == 'a':
            on_accept(event)
        elif event.key == 'r':
            on_reject(event)
        elif event.key == '1':
            on_mod_outer(event)
        elif event.key == '2':
            on_mod_pattern(event)
        elif event.key == '3':
            on_mod_unclear(event)
        elif event.key == ' ':  # spacebar
            on_skip_sug(event)
        elif event.key == 'n':
            on_next_img(event)
        elif event.key == 'q':
            on_quit(event)
    
    # Connect callbacks
    btn_accept.on_clicked(on_accept)
    btn_reject.on_clicked(on_reject)
    btn_mod_outer.on_clicked(on_mod_outer)
    btn_mod_pattern.on_clicked(on_mod_pattern)
    btn_mod_unclear.on_clicked(on_mod_unclear)
    btn_skip_sug.on_clicked(on_skip_sug)
    btn_skip_img.on_clicked(on_skip_img)
    btn_next_img.on_clicked(on_next_img)
    btn_quit.on_clicked(on_quit)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Add instructions
    instructions = """
KEYBOARD SHORTCUTS:

A = Accept suggestion
R = Reject suggestion
1 = Change to Outer Boundary
2 = Change to Pattern Region  
3 = Change to Unclear Region
Space = Skip (same as Reject)
N = Finish image, move to next
Q = Save & Quit

WORKFLOW:
1. Review highlighted mask
2. Press A if correct
3. Press R if wrong/noise
4. Press 2 if unclear→pattern
5. Press 3 if pattern→unclear
6. Press N when done with image

TIPS:
- Focus on quality over speed
- Take breaks every 30-45 min
- Check progress with:
  scripts/check_progress.py
    """
    plt.figtext(0.02, 0.88, instructions, fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='top', family='monospace')
    
    # Statistics display
    stats_text = ax.text(0.02, 0.98, '', transform=fig.transFigure,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def update_stats():
        stats_str = f"Session Stats:\n"
        stats_str += f"Reviewed: {reviewer.stats['total_reviewed']}\n"
        stats_str += f"Accepted: {reviewer.stats['accepted']}\n"
        stats_str += f"Modified: {reviewer.stats['modified']}\n"
        stats_str += f"Rejected: {reviewer.stats['rejected']}"
        stats_text.set_text(stats_str)
    
    # Update stats periodically
    def on_timer(event):
        update_stats()
    
    timer = fig.canvas.new_timer(interval=1000)  # Update every second
    timer.add_callback(on_timer, None)
    timer.start()
    
    plt.show()

if __name__ == "__main__":
    print("="*60)
    print("FOOTPRINT ANNOTATION - MANUAL REVIEW TOOL")
    print("="*60)
    print("\nLoading...")
    create_review_interface()