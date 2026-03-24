"""
05_vlm_evaluation.py
GPT-4V evaluation pipeline for forensic footprint reasoning.
Addresses RQ1-RQ6 via a two-call Reasoner → Verifier loop.

Usage:
    python vlm_evaluation.py                      # runs all benchmark items
    python vlm_evaluation.py --split test         # test split only
    python vlm_evaluation.py --dry-run            # prints prompts, no API calls
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import time
from pathlib import Path
from datetime import datetime

from openai import OpenAI

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
MODEL              = "gpt-4o"           # gpt-4o supports multi-image
BENCHMARK_PATH     = Path("data/benchmark_v1.json")
OUTPUT_DIR         = Path("data/vlm_results")
MAX_TOKENS         = 1000
TEMPERATURE        = 0.2               # low = more consistent, less hallucination
RETRY_ATTEMPTS     = 3
RETRY_DELAY        = 5                 # seconds between retries

# ── Prompts ───────────────────────────────────────────────────────────────────
REASONER_SINGLE_PROMPT = """You are a forensic analyst examining footprint evidence.

Analyze this footprint image carefully and respond in JSON format with these exact fields:

{
  "evidence_types_found": ["list of evidence types you can see"],
  "outer_boundary": {
    "present": true/false,
    "description": "describe the boundary if present",
    "confidence": 0.0-1.0
  },
  "pattern_regions": [
    {
      "description": "describe the pattern region",
      "location": "where in the image (top/bottom/left/right/center)",
      "confidence": 0.0-1.0,
      "forensic_significance": "why this is forensically relevant"
    }
  ],
  "unclear_regions": [
    {
      "description": "describe unclear area",
      "reason_unclear": "why it is unclear",
      "confidence": 0.0-1.0
    }
  ],
  "overall_confidence": 0.0-1.0,
  "abstain": true/false,
  "abstain_reason": "if abstain=true, explain why",
  "failure_modes": ["list any issues: blur/occlusion/lighting/unfamiliar_pattern/etc"],
  "hypothesis_supported": true/false,
  "justification": "overall forensic justification for your findings"
}

IMPORTANT:
- Only describe what you can actually see in the image
- If you are unsure, set abstain=true rather than guessing
- Do not invent evidence that is not visually present
- Confidence scores must reflect actual certainty"""

REASONER_MULTI_PROMPT = """You are a forensic analyst examining multiple views of the same footprint.

You are given {n_views} views of the same footprint. Analyze ALL views together and respond in JSON:

{{
  "evidence_types_found": ["list of evidence types visible across views"],
  "outer_boundary": {{
    "present": true/false,
    "description": "describe boundary",
    "consistent_across_views": true/false,
    "confidence": 0.0-1.0
  }},
  "pattern_regions": [
    {{
      "description": "describe pattern region",
      "location": "where in image",
      "views_visible_in": ["list view numbers where this appears e.g. view_1, view_3"],
      "cross_view_consistent": true/false,
      "confidence": 0.0-1.0,
      "forensic_significance": "why forensically relevant"
    }}
  ],
  "unclear_regions": [
    {{
      "description": "describe unclear area",
      "reason_unclear": "why unclear",
      "confidence": 0.0-1.0
    }}
  ],
  "cross_view_consistency": {{
    "score": 0.0-1.0,
    "consistent_features": ["features that appear consistently"],
    "inconsistent_features": ["features that vary across views"],
    "false_positives_reduced": true/false
  }},
  "overall_confidence": 0.0-1.0,
  "abstain": true/false,
  "abstain_reason": "if abstain=true, explain why",
  "failure_modes": ["list any issues across views"],
  "hypothesis_supported": true/false,
  "justification": "overall forensic justification using evidence from multiple views"
}}

IMPORTANT:
- Compare features across views explicitly
- Note which evidence appears in multiple views vs only one
- Cross-view consistency reduces false positives — flag single-view-only findings as lower confidence
- Do not invent evidence not visually present in any view"""

VERIFIER_PROMPT = """You are a forensic evidence verifier. Your job is to check whether the reasoner's 
analysis is supported by what is actually visible in the image(s).

REASONER'S ANALYSIS:
{reasoner_response}

Review the image(s) and verify this analysis. Respond in JSON:

{{
  "verification_verdict": "supported/partially_supported/unsupported",
  "hallucinations_detected": [
    {{
      "claim": "the specific claim that is not supported",
      "reason": "why it lacks visual support"
    }}
  ],
  "overclaiming_detected": [
    {{
      "claim": "claim that goes beyond what image supports",
      "reason": "why this is overclaiming"
    }}
  ],
  "unsupported_regions": ["list of regions reasoner described that are not visible"],
  "missed_evidence": ["list of evidence visible in image that reasoner missed"],
  "confidence_calibration": {{
    "reasoner_overconfident": true/false,
    "reasoner_underconfident": true/false,
    "calibration_notes": "notes on confidence accuracy"
  }},
  "corrected_findings": {{
    "evidence_types_found": ["corrected list after removing hallucinations"],
    "overall_confidence": 0.0-1.0,
    "justification": "verified justification"
  }},
  "verification_confidence": 0.0-1.0
}}"""



# ── Image loading ─────────────────────────────────────────────────────────────
def load_image_b64(img_path: str) -> str:
    """Load image and encode to base64."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_messages(image_paths: list[str]) -> list[dict]:
    """Build list of image content blocks for GPT-4V."""
    blocks = []
    for i, path in enumerate(image_paths):
        if not Path(path).exists():
            log.warning("Image not found: %s", path)
            continue
        b64 = load_image_b64(path)
        blocks.append({
            "type": "text",
            "text": f"View {i+1}:"
        })
        blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64}",
                "detail": "high"
            }
        })
    return blocks


# ── API calls ─────────────────────────────────────────────────────────────────
def call_gpt4v(
    client:         OpenAI,
    system_prompt:  str,
    image_paths:    list[str],
    extra_text:     str = "",
    dry_run:        bool = False,
) -> dict:
    """Call GPT-4V with images. Returns parsed JSON response."""

    image_blocks = build_image_messages(image_paths)

    if extra_text:
        image_blocks.append({"type": "text", "text": extra_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": image_blocks},
    ]

    if dry_run:
        log.info("[DRY RUN] Would call GPT-4V with %d images", len(image_paths))
        return {"dry_run": True, "n_images": len(image_paths)}

    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            log.warning("Attempt %d failed: %s", attempt + 1, e)
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
            else:
                log.error("All retries failed for this item")
                return {"error": str(e)}


# ── Evaluation metrics ────────────────────────────────────────────────────────
def compute_grounding_score(
    reasoner_output: dict,
    gold_evidence_types: list[str],
) -> dict:
    """
    RQ3: Compare predicted evidence types to gold labels.
    Simple set overlap metric.
    """
    predicted = set(reasoner_output.get("evidence_types_found", []))
    gold      = set(gold_evidence_types)

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "predicted":  list(predicted),
        "gold":       list(gold),
        "tp":         tp,
        "fp":         fp,
        "fn":         fn,
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
    }


def compute_hallucination_score(verifier_output: dict) -> dict:
    """RQ4: Count hallucinations from verifier output."""
    hallucinations = verifier_output.get("hallucinations_detected", [])
    overclaiming   = verifier_output.get("overclaiming_detected",   [])
    verdict        = verifier_output.get("verification_verdict", "unknown")

    return {
        "n_hallucinations":    len(hallucinations),
        "n_overclaiming":      len(overclaiming),
        "verdict":             verdict,
        "hallucination_rate":  1.0 if len(hallucinations) > 0 else 0.0,
    }


def compute_uncertainty_score(reasoner_output: dict) -> dict:
    """RQ5: Evaluate calibration and abstention."""
    return {
        "abstained":           reasoner_output.get("abstain", False),
        "abstain_reason":      reasoner_output.get("abstain_reason", ""),
        "overall_confidence":  reasoner_output.get("overall_confidence", -1),
        "failure_modes":       reasoner_output.get("failure_modes", []),
        "n_failure_modes":     len(reasoner_output.get("failure_modes", [])),
    }


# ── Per-item evaluation ───────────────────────────────────────────────────────
def evaluate_item(
    client:   OpenAI,
    item:     dict,
    dry_run:  bool = False,
) -> dict:
    """Run full Reasoner → Verifier loop for one benchmark item."""

    case_id     = item["case_id"]
    prompt_type = item["prompt_type"]
    is_multiview = prompt_type == "multi_view_consistency"
    image_paths  = item["image_paths"]

    log.info("Evaluating %s | %s | %d views",
             case_id, prompt_type, len(image_paths))

    # ── Call 1: Reasoner ──────────────────────────────────────────────────────
    if is_multiview:
        system_prompt = REASONER_MULTI_PROMPT.format(n_views=len(image_paths))
    else:
        system_prompt = REASONER_SINGLE_PROMPT

    reasoner_output = call_gpt4v(
        client, system_prompt, image_paths,
        extra_text=item["hypothesis_prompt"],
        dry_run=dry_run,
    )

    # ── Call 2: Verifier ──────────────────────────────────────────────────────
    verifier_system = VERIFIER_PROMPT.format(
        reasoner_response=json.dumps(reasoner_output, indent=2)
    )
    verifier_output = call_gpt4v(
        client, verifier_system, image_paths,
        dry_run=dry_run,
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    grounding    = compute_grounding_score(
        reasoner_output, item["gold_evidence_types"]
    )
    hallucination = compute_hallucination_score(verifier_output)
    uncertainty   = compute_uncertainty_score(reasoner_output)

    # RQ2: cross-view consistency (multi-view only)
    cross_view = None
    if is_multiview and not dry_run:
        cross_view = reasoner_output.get("cross_view_consistency", {})

    return {
        "benchmark_id":      item["benchmark_id"],
        "case_id":           case_id,
        "split":             item["split"],
        "prompt_type":       prompt_type,
        "is_multiview":      is_multiview,
        "n_views":           len(image_paths),
        "ambiguity_flag":    item["ambiguity_flag"],
        "reasoner_output":   reasoner_output,
        "verifier_output":   verifier_output,
        "grounding_metrics": grounding,        # RQ3
        "hallucination_metrics": hallucination,# RQ4
        "uncertainty_metrics":   uncertainty,  # RQ5
        "cross_view_metrics":    cross_view,   # RQ2
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ── Aggregate results ─────────────────────────────────────────────────────────
def aggregate_results(results: list[dict]) -> dict:
    """Compute aggregate metrics across all evaluated items."""

    def mean(vals):
        vals = [v for v in vals if v is not None and v >= 0]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    # RQ3: grounding
    precisions = [r["grounding_metrics"]["precision"] for r in results]
    recalls    = [r["grounding_metrics"]["recall"]    for r in results]
    f1s        = [r["grounding_metrics"]["f1"]        for r in results]

    # RQ4: hallucination
    hall_rates  = [r["hallucination_metrics"]["hallucination_rate"] for r in results]
    verdicts    = [r["hallucination_metrics"]["verdict"]            for r in results]

    # RQ5: uncertainty
    abstentions = [r["uncertainty_metrics"]["abstained"] for r in results]
    confidences = [r["uncertainty_metrics"]["overall_confidence"] for r in results]

    # RQ2: multi-view vs single-view grounding
    mv_results  = [r for r in results if r["is_multiview"]]
    sv_results  = [r for r in results if not r["is_multiview"]]
    mv_f1 = mean([r["grounding_metrics"]["f1"] for r in mv_results])
    sv_f1 = mean([r["grounding_metrics"]["f1"] for r in sv_results])

    return {
        "n_evaluated": len(results),
        "rq1_domain_shift": {
            "failure_modes_encountered": list(set(
                fm for r in results
                for fm in r["uncertainty_metrics"]["failure_modes"]
            )),
            "avg_failure_modes_per_item": mean([
                r["uncertainty_metrics"]["n_failure_modes"] for r in results
            ]),
        },
        "rq2_multiview": {
            "multiview_avg_f1":  mv_f1,
            "singleview_avg_f1": sv_f1,
            "improvement":       round(mv_f1 - sv_f1, 4),
        },
        "rq3_grounding": {
            "avg_precision": mean(precisions),
            "avg_recall":    mean(recalls),
            "avg_f1":        mean(f1s),
        },
        "rq4_hallucination": {
            "hallucination_rate":    mean(hall_rates),
            "n_hallucinated_items":  sum(1 for r in hall_rates if r > 0),
            "verdict_distribution":  {
                "supported":           verdicts.count("supported"),
                "partially_supported": verdicts.count("partially_supported"),
                "unsupported":         verdicts.count("unsupported"),
            },
        },
        "rq5_uncertainty": {
            "abstention_rate":    mean([1.0 if a else 0.0 for a in abstentions]),
            "avg_confidence":     mean(confidences),
            "n_abstained":        sum(1 for a in abstentions if a),
        },
        "rq6_verification": {
            "verifier_improved_n": sum(
                1 for r in results
                if r["verifier_output"].get("corrected_findings")
            ),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main(split: str = "all", dry_run: bool = False) -> None:
    log.info("=== VLM Forensic Evaluation ===")

    if not OPENAI_API_KEY and not dry_run:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Load benchmark
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    items = benchmark["items"]
    if split != "all":
        items = [i for i in items if i["split"] == split]

    log.info("Evaluating %d items (split=%s)", len(items), split)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, item in enumerate(items):
        log.info("Progress: %d/%d", idx + 1, len(items))

        result = evaluate_item(client, item, dry_run=dry_run)
        results.append(result)

        # Save incrementally in case of crash
        out_path = OUTPUT_DIR / f"{item['benchmark_id']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        # Rate limiting — avoid hitting API limits
        if not dry_run:
            time.sleep(1)

    # Aggregate
    agg = aggregate_results(results)

    log.info("=== Aggregate Results ===")
    log.info("RQ3 Grounding  — P=%.3f  R=%.3f  F1=%.3f",
             agg["rq3_grounding"]["avg_precision"],
             agg["rq3_grounding"]["avg_recall"],
             agg["rq3_grounding"]["avg_f1"])
    log.info("RQ4 Hallucination rate: %.3f",
             agg["rq4_hallucination"]["hallucination_rate"])
    log.info("RQ5 Abstention rate:    %.3f",
             agg["rq5_uncertainty"]["abstention_rate"])
    log.info("RQ2 Multi-view F1 improvement: %.3f",
             agg["rq2_multiview"]["improvement"])

    # Save full results
    final = {
        "model":          MODEL,
        "split":          split,
        "n_items":        len(results),
        "aggregate":      agg,
        "results":        results,
        "timestamp":      datetime.utcnow().isoformat() + "Z",
    }

    results_path = OUTPUT_DIR / f"results_{split}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)

    log.info("Results saved to %s", results_path)
    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   default="all",
                        choices=["all", "train", "val", "test"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without making API calls")
    args = parser.parse_args()
    main(args.split, args.dry_run)