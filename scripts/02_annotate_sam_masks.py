import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import sys

class FootprintAnnotator:
    def __init__(self, data_dir, masks_dir, output_dir, evidence_schema):
        self.data_dir = Path(data_dir)
        self.masks_dir = Path(masks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load evidence schema
        with open(evidence_schema) as f:
            schema = json.load(f)
        self.evidence_types = {et['id']: et for et in schema['evidence_types']}
        
        # Get all cases
        self.cases = sorted([d.name for d in self.masks_dir.glob("*") if d.is_dir()])
        self.current_case_idx = 0
        self.current_view_idx = 0
        self.current_mask_idx = 0
        
        # Load first case
        self.load_case(self.cases[0])
        
    def load_case(self, case_id):
        """Load all views and masks for a case"""
        self.current_case = case_id
        self.views = []
        
        masks_case_dir = self.masks_dir / case_id
        mask_files = sorted(list(masks_case_dir.glob("*_masks.npz")))
        
        for mask_file in mask_files:
            view_id = mask_file.stem.replace('_masks', '')
            
            # Load image
            img_path = self.data_dir / case_id / f"{view_id}.png"
            if not img_path.exists():
                continue
                
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load masks
            masks_data = np.load(mask_file)
            masks = masks_data['masks']
            scores = masks_data['scores']
            
            # Load existing annotations if any
            ann_file = self.output_dir / f"{case_id}_{view_id}_annotations.json"
            if ann_file.exists():
                with open(ann_file) as f:
                    annotations = json.load(f)
            else:
                annotations = []
            
            self.views.append({
                'view_id': view_id,
                'image': image_rgb,
                'masks': masks,
                'scores': scores,
                'annotations': annotations,
                'ann_file': ann_file
            })
        
        self.current_view_idx = 0
        self.current_mask_idx = 0
        
    def get_current_view(self):
        return self.views[self.current_view_idx]
    
    def visualize_current_mask(self):
        """Create visualization with current mask highlighted"""
        view = self.get_current_view()
        image = view['image'].copy()
        
        if self.current_mask_idx < len(view['masks']):
            mask = view['masks'][self.current_mask_idx]
            score = view['scores'][self.current_mask_idx]
            
            # Create overlay
            overlay = image.copy()
            overlay[mask] = [0, 255, 0]  # Green
            image = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
            
            # Add text
            cv2.putText(image, f"Mask {self.current_mask_idx + 1}/{len(view['masks'])}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Score: {score:.2f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return image
    
    def save_annotation(self, evidence_type, confidence, notes=""):
        """Save current mask annotation"""
        view = self.get_current_view()
        
        annotation = {
            'case_id': self.current_case,
            'view_id': view['view_id'],
            'mask_idx': int(self.current_mask_idx),
            'evidence_type': evidence_type,
            'confidence': float(confidence),
            'notes': notes,
            'mask_score': float(view['scores'][self.current_mask_idx])
        }
        
        view['annotations'].append(annotation)
        
        # Save to file
        with open(view['ann_file'], 'w') as f:
            json.dump(view['annotations'], f, indent=2)
        
        print(f"✓ Saved: {evidence_type} for mask {self.current_mask_idx}")
        
        # Move to next mask
        self.next_mask()
    
    def next_mask(self):
        view = self.get_current_view()
        if self.current_mask_idx < len(view['masks']) - 1:
            self.current_mask_idx += 1
        else:
            self.next_view()
    
    def prev_mask(self):
        if self.current_mask_idx > 0:
            self.current_mask_idx -= 1
    
    def next_view(self):
        if self.current_view_idx < len(self.views) - 1:
            self.current_view_idx += 1
            self.current_mask_idx = 0
        else:
            self.next_case()
    
    def prev_view(self):
        if self.current_view_idx > 0:
            self.current_view_idx -= 1
            self.current_mask_idx = 0
    
    def next_case(self):
        if self.current_case_idx < len(self.cases) - 1:
            self.current_case_idx += 1
            self.load_case(self.cases[self.current_case_idx])
            print(f"\n→ Loaded case: {self.current_case}")
        else:
            print("\n✓ All cases annotated!")
            plt.close('all')
    
    def skip_mask(self):
        """Skip current mask without annotating"""
        print(f"Skipped mask {self.current_mask_idx}")
        self.next_mask()

def create_annotation_interface():
    """Create interactive matplotlib interface"""
    
    # Initialize annotator
    annotator = FootprintAnnotator(
        data_dir="data/processed",
        masks_dir="data/masks",
        output_dir="data/annotations",
        evidence_schema="config/evidence_schema.json"
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.3, bottom=0.15)
    
    # Display first image
    view = annotator.get_current_view()
    img_display = ax.imshow(annotator.visualize_current_mask())
    ax.axis('off')
    ax.set_title(f"Case: {annotator.current_case} | View: {view['view_id']}")
    
    # Create buttons
    ax_outer = plt.axes([0.05, 0.7, 0.15, 0.05])
    ax_pattern = plt.axes([0.05, 0.6, 0.15, 0.05])
    ax_unclear = plt.axes([0.05, 0.5, 0.15, 0.05])
    ax_skip = plt.axes([0.05, 0.4, 0.15, 0.05])
    
    ax_prev = plt.axes([0.3, 0.05, 0.1, 0.05])
    ax_next = plt.axes([0.6, 0.05, 0.1, 0.05])
    
    btn_outer = Button(ax_outer, 'Outer Boundary\n(confidence: 0.9)')
    btn_pattern = Button(ax_pattern, 'Pattern Region\n(confidence: 0.9)')
    btn_unclear = Button(ax_unclear, 'Unclear Region\n(confidence: 0.7)')
    btn_skip = Button(ax_skip, 'Skip Mask')
    
    btn_prev = Button(ax_prev, '← Previous')
    btn_next = Button(ax_next, 'Next →')
    
    # Update display function
    def update_display():
        view = annotator.get_current_view()
        img_display.set_data(annotator.visualize_current_mask())
        ax.set_title(f"Case: {annotator.current_case} | View: {view['view_id']} | "
                    f"Annotations: {len(view['annotations'])}")
        fig.canvas.draw_idle()
    
    # Button callbacks
    def on_outer(event):
        annotator.save_annotation('outer_boundary', confidence=0.9)
        update_display()
    
    def on_pattern(event):
        annotator.save_annotation('pattern_region', confidence=0.9)
        update_display()
    
    def on_unclear(event):
        annotator.save_annotation('unclear_region', confidence=0.7)
        update_display()
    
    def on_skip(event):
        annotator.skip_mask()
        update_display()
    
    def on_prev(event):
        annotator.prev_mask()
        update_display()
    
    def on_next(event):
        annotator.next_mask()
        update_display()
    
    # Keyboard shortcuts
    def on_key(event):
        if event.key == '1':
            on_outer(event)
        elif event.key == '2':
            on_pattern(event)
        elif event.key == '3':
            on_unclear(event)
        elif event.key == ' ':  # spacebar
            on_skip(event)
        elif event.key == 'left':
            on_prev(event)
        elif event.key == 'right':
            on_next(event)
        elif event.key == 'n':
            annotator.next_view()
            update_display()
    
    # Connect callbacks
    btn_outer.on_clicked(on_outer)
    btn_pattern.on_clicked(on_pattern)
    btn_unclear.on_clicked(on_unclear)
    btn_skip.on_clicked(on_skip)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Add instructions
    instructions = """
    KEYBOARD SHORTCUTS:
    1 = Outer Boundary
    2 = Pattern Region  
    3 = Unclear Region
    Space = Skip Mask
    ← → = Navigate masks
    N = Next view
    """
    plt.figtext(0.05, 0.25, instructions, fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.show()

if __name__ == "__main__":
    print("Starting Footprint Annotation Tool...")
    print("=" * 60)
    create_annotation_interface()