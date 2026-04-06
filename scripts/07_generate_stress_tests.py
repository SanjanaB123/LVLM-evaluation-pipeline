"""
07_generate_stress_tests.py
Generates degraded / adversarial benchmark items for RQ5 (Uncertainty & Abstention).

Creates four degradation types from existing test-split images:
  1. heavy_blur      — Gaussian blur, sigma=15  (should trigger low confidence)
  2. low_contrast    — contrast reduced to 10%  (should trigger abstention)
  3. heavy_noise     — Gaussian noise, std=80    (should trigger abstention)
  4. center_crop     — 15% center crop           (partial evidence, low confidence)
  5. blank           — uniform gray image         (model MUST abstain)

Outputs:
  data/stress_test/images/{case_id}/{degradation}/{view}.png
  data/stress_test/benchmark_stress.json   (same format as benchmark_v1.json)

Usage:
  python scripts/07_generate_stress_tests.py
  python scripts/07_generate_stress_tests.py --source-dir data/processed --n-cases 5
"""

from __future__ import annotations

import argparse
import json
import random
import numpy as np
import cv2
from pathlib import Path
from copy import deepcopy

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────

BENCHMARK_PATH = Path("data/benchmark_v1.json")
SPLITS_PATH    = Path("data/case_splits.json")
OUTPUT_DIR     = Path("data/stress_test")
IMAGES_DIR     = OUTPUT_DIR / "images"

# ── Degradation functions ─────────────────────────────────────────────────────

DEGRADATIONS = {
    "heavy_blur": {
        "description": "Gaussian blur sigma=15, destroying fine pattern detail",
        "expected_abstain": False,
        "expected_confidence": "low",
        "expected_evidence": ["unclear_region"],
    },
    "low_contrast": {
        "description": "Contrast reduced to 10%, making patterns nearly invisible",
        "expected_abstain": True,
        "expected_confidence": "very_low",
        "expected_evidence": ["unclear_region"],
    },
    "heavy_noise": {
        "description": "Gaussian noise std=80, obscuring all detail",
        "expected_abstain": True,
        "expected_confidence": "very_low",
        "expected_evidence": ["unclear_region"],
    },
    "center_crop": {
        "description": "Only 15% center region visible, most evidence removed",
        "expected_abstain": False,
        "expected_confidence": "low",
        "expected_evidence": ["pattern_region", "unclear_region"],
    },
    "blank": {
        "description": "Uniform gray image with no forensic content",
        "expected_abstain": True,
        "expected_confidence": "zero",
        "expected_evidence": [],
    },
}


def apply_heavy_blur(img: np.ndarray) -> np.ndarray:
    """Heavy Gaussian blur, sigma=15."""
    return cv2.GaussianBlur(img, (0, 0), sigmaX=15, sigmaY=15)


def apply_low_contrast(img: np.ndarray) -> np.ndarray:
    """Reduce contrast to 10% — flatten toward mean gray."""
    mean_val = img.mean()
    return np.clip(mean_val + (img.astype(np.float32) - mean_val) * 0.10,
                   0, 255).astype(np.uint8)


def apply_heavy_noise(img: np.ndarray) -> np.ndarray:
    """Add Gaussian noise with std=80."""
    noise = np.random.normal(0, 80, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_center_crop(img: np.ndarray) -> np.ndarray:
    """Crop to 15% center region, resize back to original dimensions."""
    h, w = img.shape[:2]
    # 15% area ≈ ~39% of each dimension
    frac = 0.39
    y1 = int(h * (0.5 - frac / 2))
    y2 = int(h * (0.5 + frac / 2))
    x1 = int(w * (0.5 - frac / 2))
    x2 = int(w * (0.5 + frac / 2))
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def apply_blank(img: np.ndarray) -> np.ndarray:
    """Return uniform gray image of same shape."""
    return np.full_like(img, 128)


DEGRADATION_FNS = {
    "heavy_blur":   apply_heavy_blur,
    "low_contrast": apply_low_contrast,
    "heavy_noise":  apply_heavy_noise,
    "center_crop":  apply_center_crop,
    "blank":        apply_blank,
}

# ── Image processing ──────────────────────────────────────────────────────────

def degrade_image(img_path: Path, degradation: str, out_path: Path) -> bool:
    """Apply degradation to an image and save. Returns True on success."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  WARNING: Could not read {img_path}")
        return False

    fn = DEGRADATION_FNS[degradation]
    degraded = fn(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), degraded)
    return True


# ── Benchmark generation ──────────────────────────────────────────────────────

def create_stress_benchmark(source_dir: Path, n_cases: int = 0) -> None:
    """Generate degraded images and stress test benchmark."""

    # Load original benchmark and splits
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    test_cases = splits["test"]

    # Optionally limit number of cases
    if n_cases > 0 and n_cases < len(test_cases):
        test_cases = sorted(random.sample(test_cases, n_cases))

    print(f"Generating stress tests for {len(test_cases)} test cases")
    print(f"Degradation types: {list(DEGRADATIONS.keys())}")
    print(f"Source images: {source_dir}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Get original test benchmark items (footprint_analysis only — 1 per case)
    original_items = {
        item["case_id"]: item
        for item in benchmark["items"]
        if item["split"] == "test" and item["prompt_type"] == "footprint_analysis"
    }

    stress_items = []
    stats = {"total_images": 0, "success": 0, "failed": 0}

    for case_id in sorted(test_cases):
        if case_id not in original_items:
            print(f"  Skipping {case_id} — no benchmark item found")
            continue

        orig = original_items[case_id]
        case_dir = source_dir / case_id
        if not case_dir.exists():
            print(f"  Skipping {case_id} — source dir not found: {case_dir}")
            continue

        # Use only view 01 for stress tests (single-view)
        view = orig["views"][0] if orig["views"] else "01"
        src_path = case_dir / f"{view}.png"
        if not src_path.exists():
            print(f"  Skipping {case_id} — image not found: {src_path}")
            continue

        for deg_name, deg_info in DEGRADATIONS.items():
            out_img_dir = IMAGES_DIR / case_id / deg_name
            out_path = out_img_dir / f"{view}.png"

            stats["total_images"] += 1
            ok = degrade_image(src_path, deg_name, out_path)
            if ok:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                continue

            # Build stress benchmark item
            stress_item = {
                "benchmark_id": f"{case_id}_{deg_name}",
                "case_id": case_id,
                "split": "stress_test",
                "views": [view],
                "n_views": 1,
                "hypothesis_prompt": orig["hypothesis_prompt"],
                "prompt_type": "footprint_analysis",
                "degradation": deg_name,
                "degradation_description": deg_info["description"],
                "expected_abstain": deg_info["expected_abstain"],
                "expected_confidence": deg_info["expected_confidence"],
                "expected_evidence": deg_info["expected_evidence"],
                "gold_evidence_types": orig["gold_evidence_types"],
                "gold_masks": {},  # no valid masks for degraded images
                "total_annotations": 0,
                "ambiguity_flag": True,  # all stress items are ambiguous
                "has_outer_boundary": False,
                "has_pattern_region": deg_name == "center_crop",
                "has_unclear_region": True,
                "image_paths": [str(out_path)],
                "mask_paths": [],
                "source_image": str(src_path),
                "source_case_id": case_id,
            }
            stress_items.append(stress_item)

        print(f"  {case_id}: generated {len(DEGRADATIONS)} degradations")

    # Build final benchmark
    stress_benchmark = {
        "version": "1.0-stress",
        "description": "Stress test benchmark for RQ5 (Uncertainty & Abstention)",
        "purpose": "Test whether models express calibrated uncertainty on degraded/ambiguous inputs",
        "n_cases": len(test_cases),
        "n_items": len(stress_items),
        "degradation_types": {
            name: info["description"] for name, info in DEGRADATIONS.items()
        },
        "expected_behaviors": {
            "heavy_blur": "Lower confidence, more unclear_region labels",
            "low_contrast": "Should abstain — patterns not visible",
            "heavy_noise": "Should abstain — image too noisy",
            "center_crop": "Lower confidence, partial evidence only",
            "blank": "Must abstain — no forensic content present",
        },
        "items": stress_items,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "benchmark_stress.json"
    with open(out_path, "w") as f:
        json.dump(stress_benchmark, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Stress test benchmark saved to {out_path}")
    print(f"Total items: {len(stress_items)}")
    print(f"Images: {stats['success']} created, {stats['failed']} failed")
    print(f"\nBreakdown by degradation:")
    for deg in DEGRADATIONS:
        count = sum(1 for item in stress_items if item["degradation"] == deg)
        print(f"  {deg:20s}: {count} items")
    print(f"\nNext step: run evaluation on stress test benchmark:")
    print(f"  python scripts/05_vlm_evaluation.py --benchmark {out_path} --split all")
    print(f"{'='*60}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate degraded images and stress test benchmark for RQ5"
    )
    parser.add_argument(
        "--source-dir", default="data/processed",
        help="Directory containing processed case images (default: data/processed)"
    )
    parser.add_argument(
        "--n-cases", type=int, default=0,
        help="Limit to N test cases (0 = all, default: all)"
    )
    args = parser.parse_args()
    create_stress_benchmark(Path(args.source_dir), args.n_cases)
