"""
05_vlm_evaluation.py
OpenAI GPT-4o evaluation pipeline for forensic footprint reasoning.
Addresses RQ1-RQ6 via a two-call Reasoner -> Verifier loop.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
from pathlib import Path
from datetime import datetime

from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
MODEL           = "gpt-4o"
BENCHMARK_PATH  = Path("data/benchmark_v1.json")
OUTPUT_DIR      = Path("data/vlm_results")
MAX_TOKENS      = 1500
TEMPERATURE     = 0.2
RETRY_ATTEMPTS  = 3
RETRY_DELAY     = 10

VALID_EVIDENCE_TYPES = ["outer_boundary", "pattern_region", "unclear_region"]
VALID_LABELS_STR     = '["outer_boundary", "pattern_region", "unclear_region"]'

REASONER_SINGLE_PROMPT = f"""You are a shoe print analyst examining impression evidence for academic research.

Analyze this shoe print image carefully and respond in JSON format with these EXACT fields:

{{
  "evidence_types_found": ["<labels from the allowed list only>"],
  "outer_boundary": {{
    "present": true,
    "description": "describe the boundary if present",
    "confidence": 0.9
  }},
  "pattern_regions": [
    {{
      "description": "describe the pattern region",
      "location": "where in the image (top/bottom/left/right/center)",
      "confidence": 0.8,
      "forensic_significance": "why this region is analytically relevant"
    }}
  ],
  "unclear_regions": [
    {{
      "description": "describe unclear area",
      "reason_unclear": "why it is unclear",
      "confidence": 0.5
    }}
  ],
  "overall_confidence": 0.8,
  "abstain": false,
  "abstain_reason": "",
  "failure_modes": [],
  "hypothesis_supported": true,
  "justification": "overall analytical justification for your findings"
}}

CRITICAL RULES:
- "evidence_types_found" should describe what you actually see using your own words.
  Do not force labels — if something doesn't fit a category, describe it naturally.
- Only describe what you can actually see in the image.
- If you are unsure, set abstain to true rather than guessing.
- Do not invent evidence that is not visually present.
- Confidence scores must reflect actual certainty (do not always return 0.9).
- Pay close attention to smudged, partial, or ambiguous areas. If ANY region
  is unclear, degraded, smudged, or only partially visible, you MUST include
  "unclear_region" in evidence_types_found and describe it in unclear_regions.
  Do not skip this even if the unclear area is small.
- Return ONLY valid JSON, no extra text, no markdown fences."""

REASONER_MULTI_PROMPT = f"""You are a shoe print analyst examining multiple views of the same impression for academic research.

You are given {{n_views}} views of the same shoe print. Analyze ALL views together and respond in JSON:

{{{{
  "evidence_types_found": ["<labels from the allowed list only>"],
  "outer_boundary": {{{{
    "present": true,
    "description": "describe boundary",
    "consistent_across_views": true,
    "confidence": 0.9
  }}}},
  "pattern_regions": [
    {{{{
      "description": "describe pattern region",
      "location": "where in image",
      "views_visible_in": ["view_1", "view_2"],
      "cross_view_consistent": true,
      "confidence": 0.8,
      "forensic_significance": "why analytically relevant"
    }}}}
  ],
  "unclear_regions": [
    {{{{
      "description": "describe unclear area",
      "reason_unclear": "why unclear",
      "confidence": 0.5
    }}}}
  ],
  "cross_view_consistency": {{{{
    "score": 0.8,
    "consistent_features": ["features that appear consistently"],
    "inconsistent_features": ["features that vary across views"],
    "false_positives_reduced": true
  }}}},
  "overall_confidence": 0.8,
  "abstain": false,
  "abstain_reason": "",
  "failure_modes": [],
  "hypothesis_supported": true,
  "justification": "overall analytical justification using evidence from multiple views"
}}}}

CRITICAL RULES:
- "evidence_types_found" should describe what you actually see using your own words.
  Do not force labels — if something doesn't fit a category, describe it naturally.
- Compare features across views explicitly.
- Note which evidence appears in multiple views vs only one.
- Do not invent evidence not visually present in any view.
- Return ONLY valid JSON, no extra text, no markdown fences."""

VERIFIER_PROMPT = """You are a shoe print analysis verifier for academic research.
Check whether the analyst's findings are supported by what is actually visible in the image(s).

ANALYST'S FINDINGS:
{reasoner_response}

Review the image(s) and verify this analysis. Respond in JSON:

{{
  "verification_verdict": "supported",
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
  "unsupported_regions": [],
  "missed_evidence": [],
  "confidence_calibration": {{
    "reasoner_overconfident": false,
    "reasoner_underconfident": false,
    "calibration_notes": "notes on confidence accuracy"
  }},
  "corrected_findings": {{
    "evidence_types_found": ["corrected list — only use: outer_boundary, pattern_region, unclear_region"],
    "overall_confidence": 0.8,
    "justification": "verified justification"
  }},
  "verification_confidence": 0.9
}}

IMPORTANT: Return ONLY valid JSON, no extra text, no markdown fences."""


def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_messages(image_paths: list[str], is_multiview: bool = False) -> list[dict]:
    """Build vision content blocks for single-view (1 image, high detail) or multi-view (all images, low detail)."""
    if is_multiview:
        paths_to_use = image_paths
        detail       = "low"
        log.info("Multi-view: sending all %d views at detail=low", len(paths_to_use))
    else:
        paths_to_use = image_paths[:1]
        detail       = "high"
        log.info("Single-view: sending 1 of %d views at detail=high", len(image_paths))

    blocks = []
    for i, path in enumerate(paths_to_use):
        if not Path(path).exists():
            log.warning("Image not found: %s", path)
            continue
        ext  = Path(path).suffix.lower().lstrip(".")
        mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
        b64  = encode_image_b64(path)
        blocks.append({"type": "text",  "text": f"View {i + 1}:"})
        blocks.append({
            "type":      "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": detail},
        })
    return blocks


def parse_json_response(raw: str) -> dict:
    content = re.sub(r"```(?:json)?\s*", "", raw).strip().replace("```", "").strip()
    start, end = content.find("{"), content.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object found: {raw[:200]!r}")
    content = content[start:end]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    try:
        from json_repair import repair_json
        return json.loads(repair_json(content))
    except (ImportError, Exception):
        pass

    raise json.JSONDecodeError(
        f"Failed to parse after all fallbacks.\nRaw: {raw[:300]!r}",
        content, 0,
    )


LABEL_MAP = {
    "boundary":                    "outer_boundary",
    "outer boundary":              "outer_boundary",
    "outer_boundary":              "outer_boundary",
    "print boundary":              "outer_boundary",
    "footprint boundary":          "outer_boundary",
    "shoe print boundary":         "outer_boundary",
    "outline":                     "outer_boundary",
    "print outline":               "outer_boundary",
    "consistent outer boundary":   "outer_boundary",
    "outer sole":                  "outer_boundary",
    "sole outline":                "outer_boundary",
    "shoe outline":                "outer_boundary",
    "tread boundary":              "outer_boundary",

    "pattern":                     "pattern_region",
    "pattern_region":              "pattern_region",
    "pattern_regions":             "pattern_region",
    "clear pattern":               "pattern_region",
    "clear_pattern":               "pattern_region",
    "clear pattern regions":       "pattern_region",
    "clear patterns":              "pattern_region",
    "tread":                       "pattern_region",
    "tread pattern":               "pattern_region",
    "tread patterns":              "pattern_region",
    "patterned regions":           "pattern_region",
    "patterned sole impression":   "pattern_region",
    "patterned tread marks":       "pattern_region",
    "partial impression":          "pattern_region",
    "grid pattern":                "pattern_region",
    "grid-like pattern":           "pattern_region",
    "grid-like structure":         "pattern_region",
    "grid":                        "pattern_region",
    "hexagonal pattern":           "pattern_region",
    "honeycomb pattern":           "pattern_region",
    "circular pattern":            "pattern_region",
    "circular patterns":           "pattern_region",
    "linear elements":             "pattern_region",
    "linear patterns":             "pattern_region",
    "linear grooves":              "pattern_region",
    "block patterns":              "pattern_region",
    "block shapes":                "pattern_region",
    "blocky shapes":               "pattern_region",
    "rectangular patterns":        "pattern_region",
    "rectangular blocks":          "pattern_region",
    "triangular shapes":           "pattern_region",
    "geometric shapes":            "pattern_region",
    "geometric patterns":          "pattern_region",
    "curved line patterns":        "pattern_region",
    "wave-like pattern":           "pattern_region",
    "wavy design elements":        "pattern_region",
    "distinctive shapes":          "pattern_region",
    "tread marks":                 "pattern_region",
    "zigzag lines":                "pattern_region",
    "zigzag patterns":             "pattern_region",
    "textured regions":            "pattern_region",
    "textured areas":              "pattern_region",
    "cross-hatch pattern":         "pattern_region",
    "repeated oval patterns":      "pattern_region",
    "sole pattern":                "pattern_region",

    "unclear":                     "unclear_region",
    "unclear region":              "unclear_region",
    "unclear regions":             "unclear_region",
    "unclear_region":              "unclear_region",
    "unclear_regions":             "unclear_region",
    "ambiguous":                   "unclear_region",
    "degraded area":               "unclear_region",
    "worn areas":                  "unclear_region",
}

def normalise_labels(labels: list[str]) -> list[str]:
    """Map free-text labels to the canonical set; drop unknowns."""
    normalised = []
    for lbl in labels:
        lbl_lower = lbl.lower().strip()
        if lbl_lower in VALID_EVIDENCE_TYPES:
            normalised.append(lbl_lower)
        elif lbl_lower in LABEL_MAP:
            normalised.append(LABEL_MAP[lbl_lower])
        else:
            log.debug("Unknown label dropped during normalisation: %r", lbl)

    seen = set()
    return [x for x in normalised if not (x in seen or seen.add(x))]


def call_openai(
    client:        OpenAI,
    system_prompt: str,
    image_paths:   list[str],
    extra_text:    str = "",
    dry_run:       bool = False,
    is_multiview:  bool = False,
) -> dict:
    """Call GPT-4o with images. Returns parsed JSON or {"error": ...}."""

    if dry_run:
        log.info("[DRY RUN] Would call GPT-4o with %d images", len(image_paths))
        return {
            "dry_run": True,
            "evidence_types_found": ["outer_boundary", "pattern_region"],
            "overall_confidence": 0.8,
            "abstain": False,
            "abstain_reason": "",
            "failure_modes": [],
            "hypothesis_supported": True,
            "justification": "Dry run placeholder.",
        }

    img_blocks = build_image_messages(image_paths, is_multiview=is_multiview)
    if not img_blocks:
        return {"error": "No valid images found"}

    user_content = img_blocks
    if extra_text:
        user_content = user_content + [{"type": "text", "text": extra_text}]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    last_error = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw = response.choices[0].message.content.strip()

            refusal_phrases = [
                "i'm sorry, i can't",
                "i'm unable to assist",
                "i cannot assist",
                "i can't assist",
                "i'm not able to",
            ]
            if any(phrase in raw.lower() for phrase in refusal_phrases):
                log.warning(
                    "Attempt %d: model refused (content policy). Raw: %s",
                    attempt + 1, raw[:120]
                )
                last_error = f"Model refusal: {raw[:120]}"
                if attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                continue

            return parse_json_response(raw)

        except Exception as e:
            log.warning("Attempt %d failed: %s", attempt + 1, e)
            last_error = str(e)
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)

    log.error("All retries failed. Last error: %s", last_error)
    return {"error": last_error or "unknown error"}


def compute_grounding_score(reasoner_output: dict, gold_evidence_types: list[str]) -> dict:
    raw_predicted = reasoner_output.get("evidence_types_found", [])
    predicted = set(normalise_labels(raw_predicted))
    gold      = set(gold_evidence_types)
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {
        "predicted":       list(predicted),
        "predicted_raw":   raw_predicted,
        "gold":            list(gold),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
    }


def compute_hallucination_score(verifier_output: dict) -> dict:
    hallucinations = verifier_output.get("hallucinations_detected", [])
    overclaiming   = verifier_output.get("overclaiming_detected",   [])
    verdict        = verifier_output.get("verification_verdict", "unknown")
    return {
        "n_hallucinations":   len(hallucinations),
        "n_overclaiming":     len(overclaiming),
        "verdict":            verdict,
        "hallucination_rate": 1.0 if (len(hallucinations) > 0 or len(overclaiming) > 0) else 0.0,
    }


def compute_uncertainty_score(reasoner_output: dict) -> dict:
    return {
        "abstained":          reasoner_output.get("abstain", False),
        "abstain_reason":     reasoner_output.get("abstain_reason", ""),
        "overall_confidence": reasoner_output.get("overall_confidence", -1),
        "failure_modes":      reasoner_output.get("failure_modes", []),
        "n_failure_modes":    len(reasoner_output.get("failure_modes", [])),
    }


def evaluate_item(client: OpenAI, item: dict, dry_run: bool = False) -> dict:
    case_id      = item["case_id"]
    prompt_type  = item["prompt_type"]
    is_multiview = prompt_type == "multi_view_consistency"
    image_paths  = item["image_paths"]

    log.info(
        "Evaluating %s | %s | %d views",
        case_id, prompt_type, len(image_paths)
    )

    system_prompt = (
        REASONER_MULTI_PROMPT.format(n_views=len(image_paths))
        if is_multiview else REASONER_SINGLE_PROMPT
    )

    reasoner_output = call_openai(
        client, system_prompt, image_paths,
        extra_text=item["hypothesis_prompt"],
        dry_run=dry_run,
        is_multiview=is_multiview,
    )

    if "error" in reasoner_output:
        log.warning("Skipping item %s — reasoner error: %s",
                    item["benchmark_id"], reasoner_output["error"])
        return {
            "benchmark_id":  item["benchmark_id"],
            "case_id":       case_id,
            "split":         item["split"],
            "prompt_type":   prompt_type,
            "is_multiview":  is_multiview,
            "n_views":       len(image_paths),
            "ambiguity_flag": item["ambiguity_flag"],
            "skipped":       True,
            "skip_reason":   reasoner_output["error"],
            "timestamp":     datetime.now().isoformat() + "Z",
        }

    if not is_multiview and len(image_paths) > 1:
        verifier_image_paths = image_paths[1:2]
        log.info("Verifier using view 2 (independent from reasoner's view 1)")
    else:
        verifier_image_paths = image_paths

    verifier_system = VERIFIER_PROMPT.format(
        reasoner_response=json.dumps(reasoner_output, indent=2)
    )
    verifier_output = call_openai(
        client, verifier_system, verifier_image_paths,
        dry_run=dry_run,
        is_multiview=is_multiview,
    )

    if "error" in verifier_output:
        log.warning("Verifier error for %s — continuing with empty verifier output",
                    item["benchmark_id"])
        verifier_output = {
            "verification_verdict":  "unverified",
            "hallucinations_detected": [],
            "overclaiming_detected":   [],
            "corrected_findings":      {},
            "verification_confidence": 0.0,
            "verifier_error":          verifier_output["error"],
        }

    grounding     = compute_grounding_score(reasoner_output, item["gold_evidence_types"])
    hallucination = compute_hallucination_score(verifier_output)
    uncertainty   = compute_uncertainty_score(reasoner_output)
    cross_view    = reasoner_output.get("cross_view_consistency") if is_multiview else None

    # Verifier-corrected grounding (RQ6: does verification improve grounding?)
    corrected_types = verifier_output.get("corrected_findings", {}).get("evidence_types_found", [])
    if corrected_types:
        grounding_corrected = compute_grounding_score(
            {"evidence_types_found": corrected_types},
            item["gold_evidence_types"],
        )
    else:
        grounding_corrected = grounding  # fallback: same as reasoner

    return {
        "benchmark_id":          item["benchmark_id"],
        "case_id":               case_id,
        "split":                 item["split"],
        "prompt_type":           prompt_type,
        "is_multiview":          is_multiview,
        "n_views":               len(image_paths),
        "ambiguity_flag":        item["ambiguity_flag"],
        "skipped":               False,
        "reasoner_output":       reasoner_output,
        "verifier_output":       verifier_output,
        "grounding_metrics":     grounding,
        "grounding_corrected_metrics": grounding_corrected,
        "hallucination_metrics": hallucination,
        "uncertainty_metrics":   uncertainty,
        "cross_view_metrics":    cross_view,
        "timestamp":             datetime.now().isoformat() + "Z",
    }


def aggregate_results(results: list[dict]) -> dict:
    valid   = [r for r in results if not r.get("skipped", False)]
    skipped = [r for r in results if     r.get("skipped", False)]

    log.info(
        "Aggregating: %d valid, %d skipped out of %d total",
        len(valid), len(skipped), len(results)
    )

    def mean(vals):
        vals = [v for v in vals if v is not None and v >= 0]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    if not valid:
        log.warning("No valid results to aggregate!")
        return {"n_evaluated": 0, "n_skipped": len(skipped), "error": "all items failed"}

    precisions  = [r["grounding_metrics"]["precision"]              for r in valid]
    recalls     = [r["grounding_metrics"]["recall"]                 for r in valid]
    f1s         = [r["grounding_metrics"]["f1"]                     for r in valid]
    precisions_corrected = [r["grounding_corrected_metrics"]["precision"] for r in valid]
    recalls_corrected    = [r["grounding_corrected_metrics"]["recall"]    for r in valid]
    f1s_corrected        = [r["grounding_corrected_metrics"]["f1"]        for r in valid]
    hall_rates  = [r["hallucination_metrics"]["hallucination_rate"] for r in valid]
    verdicts    = [r["hallucination_metrics"]["verdict"]            for r in valid]
    abstentions = [r["uncertainty_metrics"]["abstained"]            for r in valid]
    confidences = [r["uncertainty_metrics"]["overall_confidence"]   for r in valid]

    mv_results = [r for r in valid if r["is_multiview"]]
    sv_results = [r for r in valid if not r["is_multiview"]]
    mv_f1 = mean([r["grounding_metrics"]["f1"] for r in mv_results])
    sv_f1 = mean([r["grounding_metrics"]["f1"] for r in sv_results])

    skip_reasons: dict[str, int] = {}
    for r in skipped:
        reason = r.get("skip_reason", "unknown")
        if "refusal" in reason.lower() or "can't assist" in reason.lower():
            key = "content_policy_refusal"
        elif "image not found" in reason.lower():
            key = "missing_image"
        elif "json" in reason.lower():
            key = "json_parse_error"
        else:
            key = "other"
        skip_reasons[key] = skip_reasons.get(key, 0) + 1

    return {
        "n_evaluated":  len(valid),
        "n_skipped":    len(skipped),
        "skip_reasons": skip_reasons,
        "rq1_domain_shift": {
            "failure_modes_encountered": list(set(
                fm for r in valid for fm in r["uncertainty_metrics"]["failure_modes"]
            )),
            "avg_failure_modes_per_item": mean([
                r["uncertainty_metrics"]["n_failure_modes"] for r in valid
            ]),
        },
        "rq2_multiview": {
            "multiview_avg_f1":  mv_f1,
            "singleview_avg_f1": sv_f1,
            "improvement":       round(mv_f1 - sv_f1, 4),
            "n_multiview_items": len(mv_results),
            "n_singleview_items": len(sv_results),
        },
        "rq3_grounding": {
            "avg_precision": mean(precisions),
            "avg_recall":    mean(recalls),
            "avg_f1":        mean(f1s),
        },
        "rq4_hallucination": {
            "hallucination_rate":   mean(hall_rates),
            "n_hallucinated_items": sum(1 for r in hall_rates if r > 0),
            "verdict_distribution": {
                "supported":           verdicts.count("supported"),
                "partially_supported": verdicts.count("partially_supported"),
                "unsupported":         verdicts.count("unsupported"),
                "unverified":          verdicts.count("unverified"),
            },
        },
        "rq5_uncertainty": {
            "abstention_rate": mean([1.0 if a else 0.0 for a in abstentions]),
            "avg_confidence":  mean(confidences),
            "n_abstained":     sum(1 for a in abstentions if a),
        },
        "rq6_verification": {
            "verifier_improved_n": sum(
                1 for r in valid if r["verifier_output"].get("corrected_findings")
            ),
            "reasoner_avg_f1":        mean(f1s),
            "corrected_avg_f1":       mean(f1s_corrected),
            "f1_improvement":         round(mean(f1s_corrected) - mean(f1s), 4),
            "corrected_avg_precision": mean(precisions_corrected),
            "corrected_avg_recall":    mean(recalls_corrected),
        },
    }


def main(split: str = "all", dry_run: bool = False, benchmark_file: str = None) -> None:
    log.info("=== GPT-4o Forensic Footprint Evaluation ===")

    if not OPENAI_API_KEY and not dry_run:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    client = OpenAI(api_key=OPENAI_API_KEY)

    bench_path = Path(benchmark_file) if benchmark_file else BENCHMARK_PATH
    log.info("Loading benchmark from %s", bench_path)
    with open(bench_path) as f:
        benchmark = json.load(f)

    items = benchmark["items"]
    if split != "all":
        items = [i for i in items if i["split"] == split]
    log.info(
        "Evaluating %d items (split=%s, model=%s)",
        len(items), split, MODEL
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, item in enumerate(items):
        log.info("Progress: %d/%d", idx + 1, len(items))
        result = evaluate_item(client, item, dry_run=dry_run)
        results.append(result)

        out_path = OUTPUT_DIR / f"{item['benchmark_id']}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        if not dry_run:
            time.sleep(5)

    agg = aggregate_results(results)

    log.info("=== Aggregate Results ===")
    log.info("Items evaluated: %d / skipped: %d", agg.get("n_evaluated", 0), agg.get("n_skipped", 0))
    if agg.get("skip_reasons"):
        log.info("Skip breakdown: %s", agg["skip_reasons"])
    if agg.get("n_evaluated", 0) > 0:
        log.info("RQ3 Grounding   — P=%.3f  R=%.3f  F1=%.3f",
                 agg["rq3_grounding"]["avg_precision"],
                 agg["rq3_grounding"]["avg_recall"],
                 agg["rq3_grounding"]["avg_f1"])
        log.info("RQ4 Hallucination rate:     %.3f", agg["rq4_hallucination"]["hallucination_rate"])
        log.info("RQ5 Abstention rate:        %.3f", agg["rq5_uncertainty"]["abstention_rate"])
        log.info("RQ5 Avg confidence:         %.3f", agg["rq5_uncertainty"]["avg_confidence"])
        log.info("RQ2 Multi-view F1:          %.3f  (single-view: %.3f, improvement: %.3f)",
                 agg["rq2_multiview"]["multiview_avg_f1"],
                 agg["rq2_multiview"]["singleview_avg_f1"],
                 agg["rq2_multiview"]["improvement"])

    final = {
        "model":     MODEL,
        "split":     split,
        "n_items":   len(results),
        "aggregate": agg,
        "results":   results,
        "timestamp": datetime.now().isoformat() + "Z",
    }

    tag = "abstention" if benchmark_file else "results"
    results_path = OUTPUT_DIR / f"{tag}.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)

    log.info("Results saved to %s", results_path)
    log.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",   default="all", choices=["all", "train", "val", "test", "stress_test"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark", default=None,
                        help="Path to benchmark JSON (default: data/benchmark_v1.json)")
    args = parser.parse_args()
    main(args.split, args.dry_run, args.benchmark)