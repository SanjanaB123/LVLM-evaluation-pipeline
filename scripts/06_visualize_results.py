"""
06_visualize_results.py
Generates figures from VLM evaluation results for RQ1-RQ6.
Reads vlm_results/results.json and saves plots to results/figures/.
"""

from __future__ import annotations

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from collections import defaultdict

matplotlib.use("Agg")

RESULTS_PATH = Path("vlm_results/results.json")
FIGURES_DIR  = Path("results/figures")

EVIDENCE_TYPES = ["outer_boundary", "pattern_region", "unclear_region"]
EVIDENCE_LABELS = {
    "outer_boundary": "Outer Boundary",
    "pattern_region": "Pattern Region",
    "unclear_region": "Unclear Region",
}
COLORS = {
    "outer_boundary": "#e74c3c",
    "pattern_region": "#2ecc71",
    "unclear_region": "#f39c12",
}


def load_results(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return data


def get_valid_results(data: dict) -> list[dict]:
    return [r for r in data["results"] if not r.get("skipped", False)]


# ── Figure 1: Per-evidence-type Precision / Recall / F1 ──────────────────────

def _count_per_type(results: list[dict], metrics_key: str) -> dict[str, dict[str, float]]:
    """Aggregate TP/FP/FN across items for each evidence type, return P/R/F1."""
    counts: dict[str, dict[str, int]] = {
        et: {"tp": 0, "fp": 0, "fn": 0} for et in EVIDENCE_TYPES
    }
    for r in results:
        gm = r.get(metrics_key, r["grounding_metrics"])
        gold = set(gm["gold"])
        predicted = set(gm["predicted"])
        for et in EVIDENCE_TYPES:
            in_gold = et in gold
            in_pred = et in predicted
            if in_gold and in_pred:
                counts[et]["tp"] += 1
            elif in_pred and not in_gold:
                counts[et]["fp"] += 1
            elif in_gold and not in_pred:
                counts[et]["fn"] += 1
    scores: dict[str, dict[str, float]] = {}
    for et in EVIDENCE_TYPES:
        tp, fp, fn = counts[et]["tp"], counts[et]["fp"], counts[et]["fn"]
        p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1  = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        scores[et] = {"precision": p, "recall": rec, "f1": f1}
    return scores


def fig1_per_type_grounding(results: list[dict], out_dir: Path) -> None:
    """Grouped bar chart: Reasoner vs Verifier-corrected P/R/F1 per evidence type."""
    has_corrected = any("grounding_corrected_metrics" in r for r in results)

    reasoner_scores  = _count_per_type(results, "grounding_metrics")
    corrected_scores = _count_per_type(results, "grounding_corrected_metrics") if has_corrected else None

    fig, axes = plt.subplots(1, 2 if has_corrected else 1,
                             figsize=(18 if has_corrected else 10, 6),
                             sharey=True)
    if not has_corrected:
        axes = [axes]

    metrics = ["precision", "recall", "f1"]
    metric_colors = ["#3498db", "#e74c3c", "#2ecc71"]
    metric_labels = ["Precision", "Recall", "F1"]

    def _draw_bars(ax, scores, title):
        x = np.arange(len(EVIDENCE_TYPES))
        width = 0.25
        for i, (metric, color, label) in enumerate(zip(metrics, metric_colors, metric_labels)):
            vals = [scores[et][metric] for et in EVIDENCE_TYPES]
            bars = ax.bar(x + i * width, vals, width, label=label, color=color, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Evidence Type", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels([EVIDENCE_LABELS[et] for et in EVIDENCE_TYPES])
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    _draw_bars(axes[0], reasoner_scores, "Reasoner (raw labels)")
    if has_corrected:
        _draw_bars(axes[1], corrected_scores, "After Verifier Correction")

    fig.suptitle("RQ3 + RQ6: Grounding Before vs After Verification", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_per_type_grounding.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_per_type_grounding.png")


# ── Figure 2: Multi-view vs Single-view F1 by prompt type ────────────────────

def fig2_mv_vs_sv(results: list[dict], out_dir: Path) -> None:
    """Grouped bar chart: F1 for single-view vs multi-view, by prompt type."""
    prompt_scores: dict[str, dict[str, list]] = defaultdict(lambda: {"sv": [], "mv": []})

    for r in results:
        key = "mv" if r["is_multiview"] else "sv"
        prompt_scores[r["prompt_type"]][key].append(r["grounding_metrics"]["f1"])

    prompt_types = sorted(prompt_scores.keys())
    sv_means = [np.mean(prompt_scores[pt]["sv"]) if prompt_scores[pt]["sv"] else 0.0
                for pt in prompt_types]
    mv_means = [np.mean(prompt_scores[pt]["mv"]) if prompt_scores[pt]["mv"] else 0.0
                for pt in prompt_types]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(prompt_types))
    width = 0.35

    bars_sv = ax.bar(x - width / 2, sv_means, width, label="Single-View", color="#3498db",
                     edgecolor="white")
    bars_mv = ax.bar(x + width / 2, mv_means, width, label="Multi-View", color="#e67e22",
                     edgecolor="white")

    for bars in [bars_sv, bars_mv]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    pretty = [pt.replace("_", " ").title() for pt in prompt_types]
    ax.set_xlabel("Prompt Type", fontsize=12)
    ax.set_ylabel("Average F1", fontsize=12)
    ax.set_title("RQ2: Single-View vs Multi-View Grounding F1", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_mv_vs_sv.png", dpi=150)
    plt.close(fig)
    print("  Saved fig2_mv_vs_sv.png")


# ── Figure 3: Evidence type confusion heatmap ─────────────────────────────────

def fig3_confusion_heatmap(results: list[dict], out_dir: Path) -> None:
    """For each gold type, show hit-rate, miss-rate, and false-positive rate."""
    # Rows = gold types, Cols = [Correctly predicted, Missed, False positive]
    matrix = np.zeros((len(EVIDENCE_TYPES), 3))

    for r in results:
        gm = r["grounding_metrics"]
        gold = set(gm["gold"])
        predicted = set(gm["predicted"])
        for i, et in enumerate(EVIDENCE_TYPES):
            if et in gold and et in predicted:
                matrix[i, 0] += 1  # correct
            elif et in gold and et not in predicted:
                matrix[i, 1] += 1  # missed
            elif et not in gold and et in predicted:
                matrix[i, 2] += 1  # false positive

    # Normalise each row to percentages
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    pct = matrix / row_totals * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pct, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")

    col_labels = ["Correctly\nPredicted", "Missed\n(FN)", "False\nPositive (FP)"]
    ax.set_xticks(range(3))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(EVIDENCE_TYPES)))
    ax.set_yticklabels([EVIDENCE_LABELS[et] for et in EVIDENCE_TYPES], fontsize=11)

    for i in range(len(EVIDENCE_TYPES)):
        for j in range(3):
            count = int(matrix[i, j])
            percent = pct[i, j]
            ax.text(j, i, f"{percent:.0f}%\n(n={count})",
                    ha="center", va="center", fontsize=10,
                    color="white" if percent > 55 else "black")

    ax.set_title("RQ3: Evidence Type Detection Breakdown", fontsize=14)
    fig.colorbar(im, ax=ax, label="Percentage (%)", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_confusion_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved fig3_confusion_heatmap.png")


# ── Figure 4: Confidence calibration (reliability diagram) ───────────────────

def fig4_calibration(results: list[dict], out_dir: Path) -> None:
    """Reliability diagram: binned confidence vs actual accuracy (F1)."""
    confidences = []
    f1s = []
    for r in results:
        conf = r["uncertainty_metrics"].get("overall_confidence", -1)
        if conf < 0:
            continue
        confidences.append(conf)
        f1s.append(r["grounding_metrics"]["f1"])

    if not confidences:
        print("  Skipping fig4 — no confidence data")
        return

    confidences = np.array(confidences)
    f1s = np.array(f1s)

    bin_edges = np.arange(0.0, 1.05, 0.1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        n = mask.sum()
        if n > 0:
            bin_centers.append((lo + hi) / 2)
            bin_accs.append(f1s[mask].mean())
            bin_counts.append(int(n))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color="#3498db",
           edgecolor="white", label="Model")

    for bc, ba, bn in zip(bin_centers, bin_accs, bin_counts):
        ax.text(bc, ba + 0.02, f"n={bn}", ha="center", fontsize=8)

    ax.set_xlabel("Model Confidence", fontsize=12)
    ax.set_ylabel("Actual F1 Score", fontsize=12)
    ax.set_title("RQ5: Confidence Calibration (Reliability Diagram)", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_calibration.png", dpi=150)
    plt.close(fig)
    print("  Saved fig4_calibration.png")


# ── Figure 5: Verifier verdict distribution ───────────────────────────────────

def fig5_verdict_distribution(results: list[dict], agg: dict, out_dir: Path) -> None:
    """Pie/bar chart of verifier verdicts."""
    dist = agg.get("rq4_hallucination", {}).get("verdict_distribution", {})
    if not dist:
        print("  Skipping fig5 — no verdict data")
        return

    labels = []
    sizes = []
    colors_map = {
        "supported": "#2ecc71",
        "partially_supported": "#f39c12",
        "unsupported": "#e74c3c",
        "unverified": "#95a5a6",
    }
    pie_colors = []
    for verdict in ["supported", "partially_supported", "unsupported", "unverified"]:
        count = dist.get(verdict, 0)
        if count > 0:
            labels.append(verdict.replace("_", " ").title())
            sizes.append(count)
            pie_colors.append(colors_map.get(verdict, "#bdc3c7"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=pie_colors,
        startangle=90, textprops={"fontsize": 10}
    )
    ax1.set_title("Verifier Verdict Distribution", fontsize=13)

    # Bar chart with counts
    all_verdicts = ["supported", "partially_supported", "unsupported", "unverified"]
    all_counts = [dist.get(v, 0) for v in all_verdicts]
    all_labels = [v.replace("_", " ").title() for v in all_verdicts]
    bar_colors = [colors_map[v] for v in all_verdicts]
    bars = ax2.bar(all_labels, all_counts, color=bar_colors, edgecolor="white")
    for bar, c in zip(bars, all_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(c), ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("RQ4: Verification Verdicts", fontsize=13)
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("RQ4: Hallucination & Verification Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_verdict_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_verdict_distribution.png")


# ── Figure 6: Hallucination detail table ──────────────────────────────────────

def fig6_hallucination_table(results: list[dict], out_dir: Path) -> None:
    """Table figure showing items where hallucinations or overclaiming were detected."""
    rows = []
    for r in results:
        hm = r["hallucination_metrics"]
        if hm["n_hallucinations"] > 0 or hm["n_overclaiming"] > 0:
            vo = r["verifier_output"]
            hall_claims = [h.get("claim", "—") for h in vo.get("hallucinations_detected", [])]
            over_claims = [o.get("claim", "—") for o in vo.get("overclaiming_detected", [])]
            corrected = vo.get("corrected_findings", {}).get("evidence_types_found", [])
            rows.append([
                r["case_id"],
                r["prompt_type"].replace("_", " "),
                "; ".join(hall_claims) if hall_claims else "—",
                "; ".join(over_claims) if over_claims else "—",
                ", ".join(corrected) if corrected else "—",
            ])

    if not rows:
        print("  Skipping fig6 — no hallucinations found")
        return

    col_labels = ["Case ID", "Prompt Type", "Hallucinations", "Overclaiming", "Corrected Types"]
    fig, ax = plt.subplots(figsize=(16, max(2, 1 + len(rows) * 0.6)))
    ax.axis("off")

    # Truncate long strings for display
    for row in rows:
        for j in range(len(row)):
            if len(str(row[j])) > 60:
                row[j] = str(row[j])[:57] + "..."

    table = ax.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="left", loc="center",
        colWidths=[0.1, 0.12, 0.3, 0.3, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Header styling
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("RQ4: Items with Detected Hallucinations / Overclaiming",
                 fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_hallucination_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_hallucination_table.png")


# ── Figure 7: Overall RQ summary radar ────────────────────────────────────────

def fig7_rq_summary(results: list[dict], agg: dict, out_dir: Path) -> None:
    """Radar chart summarising performance across all RQs on a 0-1 scale.
    Shows both Reasoner and Verifier-corrected F1 if available."""

    rq6 = agg.get("rq6_verification", {})
    has_corrected = "corrected_avg_f1" in rq6

    # Confidence calibration gap: |avg_confidence - avg_f1| (closer to 0 = better)
    avg_conf = agg["rq5_uncertainty"]["avg_confidence"]
    avg_f1 = agg["rq3_grounding"]["avg_f1"]
    calibration_score = max(0, 1 - abs(avg_conf - avg_f1))

    rq_labels = [
        "RQ3\nReasoner F1",
        "RQ4\n1 - Hallucination\nRate",
        "RQ5\nCalibration",
        "RQ2\nSingle-View F1",
        "RQ2\nMulti-View F1",
    ]
    reasoner_values = [
        agg["rq3_grounding"]["avg_f1"],
        1 - agg["rq4_hallucination"]["hallucination_rate"],
        calibration_score,
        agg["rq2_multiview"]["singleview_avg_f1"],
        agg["rq2_multiview"]["multiview_avg_f1"],
    ]

    if has_corrected:
        rq_labels.append("RQ6\nCorrected F1")
        reasoner_values.append(rq6["corrected_avg_f1"])

    n = len(rq_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_plot = reasoner_values + [reasoner_values[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles_plot, values_plot, alpha=0.25, color="#3498db")
    ax.plot(angles_plot, values_plot, "o-", color="#3498db", linewidth=2, label="Performance")

    ax.set_xticks(angles)
    ax.set_xticklabels(rq_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)

    for angle, val in zip(angles, reasoner_values):
        ax.text(angle, val + 0.06, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    title = "Overall Performance Summary"
    if has_corrected:
        delta = rq6["f1_improvement"]
        sign = "+" if delta >= 0 else ""
        title += f"\nVerifier F1 improvement: {sign}{delta:.3f}"
    ax.set_title(title, fontsize=14, pad=25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig7_rq_summary_radar.png", dpi=150)
    plt.close(fig)
    print("  Saved fig7_rq_summary_radar.png")


# ── Figure 8: Stress test / RQ5 uncertainty analysis ──────────────────────────

# Clean-image baselines (from test-split run on undegraded images)
CLEAN_AVG_CONFIDENCE = 0.818
CLEAN_AVG_F1         = 0.727

DEGRADATION_ORDER = ["heavy_blur", "low_contrast", "heavy_noise", "center_crop", "blank"]
DEGRADATION_LABELS = {
    "heavy_blur":   "Heavy\nBlur",
    "low_contrast": "Low\nContrast",
    "heavy_noise":  "Heavy\nNoise",
    "center_crop":  "Center\nCrop",
    "blank":        "Blank\nImage",
}
DEG_COLORS = {
    "heavy_blur":   "#3498db",
    "low_contrast": "#9b59b6",
    "heavy_noise":  "#e67e22",
    "center_crop":  "#1abc9c",
    "blank":        "#e74c3c",
}


def fig8_stress_test(results: list[dict], all_results_incl_skipped: list[dict],
                     out_dir: Path) -> None:
    """Three-panel figure: confidence, abstention rate, and F1 by degradation type."""

    # Extract degradation from benchmark_id (format: {case_id}_{degradation})
    def _get_deg(r: dict) -> str:
        bid = r.get("benchmark_id", "")
        for deg in DEGRADATION_ORDER:
            if bid.endswith(deg):
                return deg
        return ""

    # Check if this is a stress-test result set
    degs_found = [_get_deg(r) for r in all_results_incl_skipped]
    if not any(degs_found):
        print("  Skipping fig8 — no stress test items found")
        return

    # Group valid results by degradation
    deg_results: dict[str, list[dict]] = {d: [] for d in DEGRADATION_ORDER}
    for r in results:
        d = _get_deg(r)
        if d in deg_results:
            deg_results[d].append(r)

    # Count skipped per degradation (includes valid + skipped)
    deg_total: dict[str, int] = {d: 0 for d in DEGRADATION_ORDER}
    deg_skipped: dict[str, int] = {d: 0 for d in DEGRADATION_ORDER}
    for r in all_results_incl_skipped:
        d = _get_deg(r)
        if d in deg_total:
            deg_total[d] += 1
            if r.get("skipped", False):
                deg_skipped[d] += 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(DEGRADATION_ORDER))
    bar_colors = [DEG_COLORS[d] for d in DEGRADATION_ORDER]
    xlabels = [DEGRADATION_LABELS[d] for d in DEGRADATION_ORDER]

    # ── Panel 1: Confidence by degradation ──
    confs = []
    for d in DEGRADATION_ORDER:
        vals = [r["uncertainty_metrics"]["overall_confidence"]
                for r in deg_results[d]
                if r["uncertainty_metrics"]["overall_confidence"] >= 0]
        confs.append(np.mean(vals) if vals else 0.0)

    bars1 = ax1.bar(x, confs, color=bar_colors, edgecolor="white", width=0.6)
    ax1.axhline(y=CLEAN_AVG_CONFIDENCE, color="black", linestyle="--", linewidth=1.5,
                label=f"Clean baseline ({CLEAN_AVG_CONFIDENCE})")
    for bar, v in zip(bars1, confs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=10)
    ax1.set_ylabel("Average Confidence", fontsize=12)
    ax1.set_title("Model Confidence", fontsize=13)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ── Panel 2: Abstention + skip rate ──
    abstain_rates = []
    skip_rates = []
    for d in DEGRADATION_ORDER:
        n_total = deg_total[d]
        n_skip = deg_skipped[d]
        n_abstain = sum(1 for r in deg_results[d] if r["uncertainty_metrics"]["abstained"])
        abstain_rates.append(n_abstain / n_total * 100 if n_total > 0 else 0.0)
        skip_rates.append(n_skip / n_total * 100 if n_total > 0 else 0.0)

    bars2a = ax2.bar(x - 0.15, abstain_rates, 0.3, color=bar_colors, edgecolor="white",
                     label="Abstained")
    bars2b = ax2.bar(x + 0.15, skip_rates, 0.3, color=bar_colors,
                     edgecolor="white", label="Refused / Error", alpha=0.4)
    for bar, v in zip(bars2a, abstain_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, v in zip(bars2b, skip_rates):
        if v > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=10)
    ax2.set_ylabel("Rate (%)", fontsize=12)
    ax2.set_title("Abstention & Refusal Rate", fontsize=13)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # ── Panel 3: F1 by degradation ──
    f1s = []
    for d in DEGRADATION_ORDER:
        vals = [r["grounding_metrics"]["f1"] for r in deg_results[d]]
        f1s.append(np.mean(vals) if vals else 0.0)

    bars3 = ax3.bar(x, f1s, color=bar_colors, edgecolor="white", width=0.6)
    ax3.axhline(y=CLEAN_AVG_F1, color="black", linestyle="--", linewidth=1.5,
                label=f"Clean baseline ({CLEAN_AVG_F1})")
    for bar, v in zip(bars3, f1s):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlabels, fontsize=10)
    ax3.set_ylabel("Average F1", fontsize=12)
    ax3.set_title("Grounding F1", fontsize=13)
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=9)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle("RQ5: Uncertainty Calibration on Degraded Inputs\n"
                 "Model never abstains and maintains high confidence on destroyed images",
                 fontsize=14, y=1.04)
    fig.tight_layout()
    fig.savefig(out_dir / "fig8_stress_test_rq5.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_stress_test_rq5.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def _backfill_corrected_grounding(results: list[dict]) -> list[dict]:
    """For results produced before dual-scoring was added, compute
    grounding_corrected_metrics on the fly from verifier corrected_findings."""
    for r in results:
        if "grounding_corrected_metrics" in r:
            continue  # already has it
        vo = r.get("verifier_output", {})
        corrected_types = vo.get("corrected_findings", {}).get("evidence_types_found", [])
        gold = r.get("grounding_metrics", {}).get("gold", [])
        if corrected_types and gold:
            predicted = set(corrected_types)
            gold_set = set(gold)
            tp = len(predicted & gold_set)
            fp = len(predicted - gold_set)
            fn = len(gold_set - predicted)
            p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1  = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
            r["grounding_corrected_metrics"] = {
                "predicted": list(predicted),
                "predicted_raw": corrected_types,
                "gold": list(gold_set),
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(p, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
            }
        else:
            r["grounding_corrected_metrics"] = r["grounding_metrics"]
    return results


def main(results_path: str = None, output_dir: str = None) -> None:
    path = Path(results_path) if results_path else RESULTS_PATH
    print(f"Loading results from {path}")
    data = load_results(path)
    results = get_valid_results(data)
    results = _backfill_corrected_grounding(results)
    agg = data["aggregate"]

    # Backfill rq6 corrected metrics into aggregate if missing
    rq6 = agg.get("rq6_verification", {})
    if "corrected_avg_f1" not in rq6:
        corr_f1s = [r["grounding_corrected_metrics"]["f1"] for r in results]
        corr_ps  = [r["grounding_corrected_metrics"]["precision"] for r in results]
        corr_rs  = [r["grounding_corrected_metrics"]["recall"] for r in results]
        raw_f1s  = [r["grounding_metrics"]["f1"] for r in results]
        mean_c = sum(corr_f1s) / len(corr_f1s) if corr_f1s else 0.0
        mean_r = sum(raw_f1s) / len(raw_f1s) if raw_f1s else 0.0
        rq6["reasoner_avg_f1"] = round(mean_r, 4)
        rq6["corrected_avg_f1"] = round(mean_c, 4)
        rq6["f1_improvement"] = round(mean_c - mean_r, 4)
        rq6["corrected_avg_precision"] = round(sum(corr_ps) / len(corr_ps), 4) if corr_ps else 0.0
        rq6["corrected_avg_recall"] = round(sum(corr_rs) / len(corr_rs), 4) if corr_rs else 0.0
        agg["rq6_verification"] = rq6

    print(f"  {len(results)} valid results, {len(data['results']) - len(results)} skipped")

    fig_dir = Path(output_dir) if output_dir else FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving figures to {fig_dir}/\n")

    all_results = data["results"]  # includes skipped, needed for fig8

    fig1_per_type_grounding(results, fig_dir)
    fig2_mv_vs_sv(results, fig_dir)
    fig3_confusion_heatmap(results, fig_dir)
    fig4_calibration(results, fig_dir)
    fig5_verdict_distribution(results, agg, fig_dir)
    fig6_hallucination_table(results, fig_dir)
    fig7_rq_summary(results, agg, fig_dir)
    try:
        fig8_stress_test(results, all_results, fig_dir)
    except Exception as e:
        print(f"  ERROR in fig8: {e}")

    print(f"\nDone — figures saved to {fig_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--results", default=None,
                        help="Path to results JSON (default: vlm_results/results.json)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for figures (default: results/figures)")
    args = parser.parse_args()
    main(args.results, args.output_dir)
