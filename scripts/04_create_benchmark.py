"""
create_benchmark.py
Builds benchmark_v1.json from annotations for VLM reasoning evaluation.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Config
ANNOTATIONS_DIR = Path("data/annotations")
MASKS_DIR       = Path("data/masks")
OUTPUT_DIR      = Path("data")
RANDOM_SEED     = 42

HYPOTHESIS_PROMPTS = {
    "footprint_analysis": "Identify and segment the forensic evidence present in this footprint image. Describe what you see and justify your segmentation.",
    "pattern_identification": "What pattern features are visible in this footprint? Segment the relevant regions and explain their forensic significance.",
    "boundary_detection": "Identify the outer boundary of the footprint and any internal pattern regions. What can you infer about the shoe that made this print?",
    "multi_view_consistency": "Across these views of the same footprint, identify consistent forensic features. Are there any regions that appear in multiple views?",
}

def get_case_ids():
    """Extract unique case IDs from annotation filenames e.g. 00101L, 00101R"""
    files = sorted(ANNOTATIONS_DIR.glob("*_annotations.json"))
    case_ids = sorted(set(f.name.split("_")[0] for f in files))
    return case_ids

def get_base_cases(case_ids):
    """Group L/R variants into base cases e.g. 001_01 → [00101L, 00101R]"""
    base_map = defaultdict(list)
    for cid in case_ids:
        # Base case = first 5 digits e.g. 00101
        base = cid[:5]
        base_map[base].append(cid)
    return base_map

def load_annotations_for_case(case_id):
    """Load all annotations for a case across all views"""
    files = sorted(ANNOTATIONS_DIR.glob(f"{case_id}_*_annotations.json"))
    all_annotations = []
    views = []
    for f in files:
        view_id = f.name.replace(f"{case_id}_", "").replace("_annotations.json", "")
        views.append(view_id)
        with open(f) as fp:
            anns = json.load(fp)
        all_annotations.extend(anns)
    return views, all_annotations

def get_gold_masks(annotations):
    """Extract gold mask info grouped by evidence type"""
    gold = defaultdict(list)
    for ann in annotations:
        gold[ann["evidence_type"]].append({
            "view_id":   ann["view_id"],
            "mask_idx":  ann["mask_idx"],
            "confidence": ann["confidence"],
            "reasoning": ann["reasoning"],
        })
    return dict(gold)

def flag_ambiguity(annotations):
    """Flag as ambiguous if many unclear_regions or low average confidence"""
    unclear = [a for a in annotations if a["evidence_type"] == "unclear_region"]
    avg_conf = sum(a["confidence"] for a in annotations) / len(annotations) if annotations else 0
    return len(unclear) > len(annotations) * 0.4 or avg_conf < 0.65

def create_benchmark_item(case_id, views, annotations, prompt_key, split):
    """Create one benchmark item"""
    gold_masks   = get_gold_masks(annotations)
    ambiguous    = flag_ambiguity(annotations)
    evidence_types = list(gold_masks.keys())

    return {
        "benchmark_id":      f"{case_id}_{prompt_key}",
        "case_id":           case_id,
        "split":             split,
        "views":             views,
        "n_views":           len(views),
        "hypothesis_prompt": HYPOTHESIS_PROMPTS[prompt_key],
        "prompt_type":       prompt_key,
        "gold_evidence_types": evidence_types,
        "gold_masks":        gold_masks,
        "total_annotations": len(annotations),
        "ambiguity_flag":    ambiguous,
        "has_outer_boundary": "outer_boundary" in gold_masks,
        "has_pattern_region": "pattern_region" in gold_masks,
        "has_unclear_region": "unclear_region" in gold_masks,
        "image_paths":       [f"data/processed/{case_id}/{v}.png" for v in views],
        "mask_paths":        [f"data/masks/{case_id}/{v}_masks.npz" for v in views],
    }

def split_cases(base_cases, seed=RANDOM_SEED):
    """Split base cases into train/val/test (70/15/15) by case, not by image"""
    bases = sorted(base_cases.keys())
    random.seed(seed)
    random.shuffle(bases)

    n     = len(bases)
    train = bases[:int(n * 0.70)]
    val   = bases[int(n * 0.70):int(n * 0.85)]
    test  = bases[int(n * 0.85):]

    # Expand back to full case IDs (L and R)
    def expand(base_list):
        out = []
        for b in base_list:
            out.extend(base_cases[b])
        return sorted(out)

    return expand(train), expand(val), expand(test)

def main():
    print("Building benchmark_v1.json...")
    print("=" * 60)

    case_ids   = get_case_ids()
    base_cases = get_base_cases(case_ids)

    print(f"Found {len(case_ids)} case variants across {len(base_cases)} base cases")
    print(f"Base cases: {sorted(base_cases.keys())}\n")

    # Split by base case (prevents L/R leakage)
    train_cases, val_cases, test_cases = split_cases(base_cases)

    print(f"Train cases ({len(train_cases)}): {train_cases}")
    print(f"Val cases   ({len(val_cases)}):   {val_cases}")
    print(f"Test cases  ({len(test_cases)}):  {test_cases}")

    split_map = {}
    for c in train_cases: split_map[c] = "train"
    for c in val_cases:   split_map[c] = "val"
    for c in test_cases:  split_map[c] = "test"

    # Save case splits
    case_splits = {"train": train_cases, "val": val_cases, "test": test_cases}
    with open(OUTPUT_DIR / "case_splits.json", "w") as f:
        json.dump(case_splits, f, indent=2)
    print(f"\nCase splits saved to data/case_splits.json")

    # Build benchmark items
    benchmark_items = []

    for case_id in case_ids:
        views, annotations = load_annotations_for_case(case_id)

        if not annotations:
            print(f"  No annotations for {case_id}, skipping")
            continue

        split = split_map.get(case_id, "train")

        # Single view prompts
        for pk in ["footprint_analysis", "pattern_identification", "boundary_detection"]:
            item = create_benchmark_item(case_id, views, annotations, pk, split)
            benchmark_items.append(item)

        # Multi-view prompt (only if more than 1 view)
        if len(views) > 1:
            item = create_benchmark_item(case_id, views, annotations, "multi_view_consistency", split)
            benchmark_items.append(item)

    # Stats
    total     = len(benchmark_items)
    n_train   = sum(1 for i in benchmark_items if i["split"] == "train")
    n_val     = sum(1 for i in benchmark_items if i["split"] == "val")
    n_test    = sum(1 for i in benchmark_items if i["split"] == "test")
    n_ambig   = sum(1 for i in benchmark_items if i["ambiguity_flag"])
    n_multi   = sum(1 for i in benchmark_items if i["prompt_type"] == "multi_view_consistency")

    print(f"\n{'='*60}")
    print(f"Benchmark Statistics:")
    print(f"  Total items:      {total}")
    print(f"  Train:            {n_train}")
    print(f"  Val:              {n_val}")
    print(f"  Test:             {n_test}")
    print(f"  Ambiguous:        {n_ambig}")
    print(f"  Multi-view items: {n_multi}")

    # Save benchmark
    benchmark = {
        "version":     "1.0",
        "description": "Forensic footprint VLM reasoning benchmark",
        "n_cases":     len(case_ids),
        "n_items":     total,
        "splits":      {"train": n_train, "val": n_val, "test": n_test},
        "prompt_types": list(HYPOTHESIS_PROMPTS.keys()),
        "evidence_types": ["outer_boundary", "pattern_region", "unclear_region"],
        "items":       benchmark_items,
    }

    out_path = OUTPUT_DIR / "benchmark_v1.json"
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"\nSaved to {out_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()