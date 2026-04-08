#!/usr/bin/env python3
"""
Compare fine-tuned model (Condition C) against baseline conditions.
Produces decision-gate report for Weekend 2.

Reads:
  - eval/results/finetuned_eval_2026-04-07.json (Condition C)
  - Hardcoded baseline scores from Weekend 1 (STATUS.md)

Usage:
    python3 scripts/5_compare_conditions.py
"""
import json
from pathlib import Path

FINETUNED_PATH = Path("eval/results/finetuned_eval_2026-04-07.json")

# Baseline per-dimension scores from Weekend 1 (STATUS.md)
# These are pass rates from the baseline_v3_2026-04-05.json eval
BASELINE_SCORES = {
    "A_gemma": {
        "D1a_steps": 0.739, "D1b_question": 0.478, "D2_check": 0.261,
        "D3_age": 0.674, "D4_load": 0.130, "D5_prior": 0.196,
        "D6_growth": 0.839, "D7_higher": 0.783, "D8_practice": 0.630,
        "D9_motor": 0.557, "D10_choice": 0.891, "D11_instrument": 1.000,
        "weighted_mean": 0.605,
    },
    "D_opus": {
        "D1a_steps": 0.870, "D1b_question": 0.609, "D2_check": 0.304,
        "D3_age": 1.000, "D4_load": 0.217, "D5_prior": 0.196,
        "D6_growth": 0.652, "D7_higher": 0.565, "D8_practice": 0.587,
        "D9_motor": 0.435, "D10_choice": 1.000, "D11_instrument": 1.000,
        "weighted_mean": 0.639,
    },
    "E_opus_socratic": {
        "weighted_mean": 0.478,
    },
}


def main():
    with open(FINETUNED_PATH) as f:
        ft = json.load(f)

    c_scores = {dim: info["mean"] for dim, info in ft["per_dimension"].items()}
    c_weighted = ft["weighted_mean"]

    a_scores = BASELINE_SCORES["A_gemma"]
    d_scores = BASELINE_SCORES["D_opus"]

    print("=" * 80)
    print("WEEKEND 2 DECISION GATE — FINE-TUNED MODEL EVALUATION")
    print("=" * 80)

    # Per-dimension comparison table
    print(f"\n{'Dimension':<20} {'A (Gemma)':>10} {'C (Tuned)':>10} {'D (Opus)':>10} {'C vs A':>10} {'C vs D':>10}")
    print("-" * 70)

    dims_improved_vs_a = 0
    dims_regressed_vs_a = 0
    dims_unchanged = 0

    all_dims = ["D1a_steps", "D1b_question", "D2_check", "D3_age", "D4_load",
                "D5_prior", "D6_growth", "D7_higher", "D8_practice", "D9_motor",
                "D10_choice", "D11_instrument"]

    for dim in all_dims:
        a = a_scores.get(dim, None)
        c = c_scores.get(dim, None)
        d = d_scores.get(dim, None)

        if a is None or c is None:
            continue

        diff_a = c - a
        diff_d = (c - d) if d else None

        if diff_a > 0.02:
            dims_improved_vs_a += 1
            marker = "↑"
        elif diff_a < -0.02:
            dims_regressed_vs_a += 1
            marker = "↓"
        else:
            dims_unchanged += 1
            marker = "="

        a_str = f"{a:.3f}"
        c_str = f"{c:.3f}"
        d_str = f"{d:.3f}" if d else "N/A"
        diff_a_str = f"{diff_a:+.3f} {marker}"
        diff_d_str = f"{diff_d:+.3f}" if diff_d else "N/A"

        print(f"{dim:<20} {a_str:>10} {c_str:>10} {d_str:>10} {diff_a_str:>10} {diff_d_str:>10}")

    print("-" * 70)
    a_wm = a_scores["weighted_mean"]
    d_wm = d_scores["weighted_mean"]
    print(f"{'Weighted Mean':<20} {a_wm:>10.3f} {c_weighted:>10.3f} {d_wm:>10.3f} {c_weighted - a_wm:>+10.3f} {c_weighted - d_wm:>+10.3f}")

    # Decision gate
    print(f"\n{'=' * 80}")
    print("DECISION GATE CRITERIA")
    print(f"{'=' * 80}")

    total_dims = dims_improved_vs_a + dims_regressed_vs_a + dims_unchanged

    # Criterion 1: C > A on majority of dimensions
    print(f"\n1. Dimensions improved (C > A by >2%): {dims_improved_vs_a}/{total_dims}")
    print(f"   Unchanged: {dims_unchanged}, Regressed: {dims_regressed_vs_a}")
    if dims_improved_vs_a >= 7:
        c1 = "PASS"
    elif dims_improved_vs_a >= 5:
        c1 = "MARGINAL"
    else:
        c1 = "FAIL"
    print(f"   VERDICT: {c1}")

    # Criterion 2: No major regressions
    print(f"\n2. Regressions (C < A by >2%): {dims_regressed_vs_a}")
    if dims_regressed_vs_a == 0:
        c2 = "GREAT"
    elif dims_regressed_vs_a <= 2:
        c2 = "ACCEPTABLE"
    else:
        c2 = "FAIL"
    print(f"   VERDICT: {c2}")

    # Criterion 3: Opus gap narrows
    opus_gap_old = d_wm - a_wm
    opus_gap_new = d_wm - c_weighted
    if opus_gap_old > 0:
        gap_change_pct = (opus_gap_old - opus_gap_new) / opus_gap_old * 100
    else:
        gap_change_pct = 0
    print(f"\n3. Opus gap: {opus_gap_old:.3f} → {opus_gap_new:.3f} ({gap_change_pct:+.0f}% change)")
    if c_weighted > d_wm:
        c3 = "EXCEEDS OPUS"
    elif gap_change_pct > 25:
        c3 = "GREAT"
    elif gap_change_pct > 10:
        c3 = "PASS"
    else:
        c3 = "FAIL"
    print(f"   VERDICT: {c3}")

    # Criterion 4: C > E (fine-tuned beats Socratic prompting)
    e_wm = BASELINE_SCORES["E_opus_socratic"]["weighted_mean"]
    print(f"\n4. C ({c_weighted:.3f}) vs E Socratic ({e_wm:.3f}): {c_weighted - e_wm:+.3f}")
    c4 = "PASS" if c_weighted > e_wm else "FAIL"
    print(f"   VERDICT: {c4}")

    # Critical dimension deep dive
    print(f"\n{'=' * 80}")
    print("CRITICAL DIMENSIONS (highest weight = 1.5x)")
    print(f"{'=' * 80}")
    for dim in ["D2_check", "D4_load", "D5_prior"]:
        a = a_scores.get(dim, 0)
        c = c_scores.get(dim, 0)
        d = d_scores.get(dim, 0)
        print(f"  {dim}: A={a:.3f} → C={c:.3f} (delta {c-a:+.3f}) | Opus={d:.3f}")

    # Overall recommendation
    verdicts = [c1, c2, c3, c4]
    passes = sum(1 for v in verdicts if v in ("PASS", "GREAT", "EXCEEDS OPUS"))
    print(f"\n{'=' * 80}")
    print(f"OVERALL: {passes}/4 criteria met")
    print(f"{'=' * 80}")
    if passes >= 3:
        print("RECOMMENDATION: PROCEED — Fine-tuning shows clear improvement. Next: DPO + data quality.")
    elif passes >= 2:
        print("RECOMMENDATION: MARGINAL — Some improvement but gaps remain. Next: data quality review before DPO.")
    else:
        print("RECOMMENDATION: STOP — Re-examine training data and approach before investing more compute.")

    # Key finding
    print(f"\nKEY FINDING: Model outputs in teacher-guide register (23/23 use 'Here is/are').")
    print("This is a training data quality issue — ConvoLearn + synthetic data was teacher-facing.")
    print("Fix for v2: More direct student conversation examples in training data.")

    # Save report
    report = {
        "timestamp": ft["timestamp"],
        "c_weighted_mean": c_weighted,
        "a_weighted_mean": a_wm,
        "d_weighted_mean": d_wm,
        "e_weighted_mean": e_wm,
        "c_per_dimension": c_scores,
        "dims_improved_vs_a": dims_improved_vs_a,
        "dims_regressed_vs_a": dims_regressed_vs_a,
        "dims_unchanged": dims_unchanged,
        "opus_gap_change_pct": round(gap_change_pct, 1),
        "criteria_met": passes,
        "verdicts": {"c1_improvement": c1, "c2_regressions": c2, "c3_opus_gap": c3, "c4_vs_socratic": c4},
    }
    report_path = Path("eval/results/decision_gate_2026-04-07.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
