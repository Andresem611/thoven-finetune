#!/usr/bin/env python3
"""
Kill criterion check for the fine-tuning experiment.

If Condition B (base Gemma + Socratic prompt) scores >0.75 mean on
D1 (scaffolding) + D5 (prior knowledge) + D6 (growth mindset),
STOP — prompting alone suffices for pedagogy.

Usage:
    python 0_check_kill_criterion.py <promptfoo_results.json>
"""
import json
import sys
from pathlib import Path


KILL_THRESHOLD = 0.75
TARGET_DIMS = {
    "D1_Scaffolding": "Scaffolding",
    "D5_PriorKnowledge": "Prior Knowledge",
    "D6_GrowthMindset": "Growth Mindset",
}
ALL_DIMS = [
    "D1_Scaffolding", "D2_FormativeAssessment", "D3_AgeAppropriate",
    "D4_CognitiveLoad", "D5_PriorKnowledge", "D6_GrowthMindset",
    "D7_CognitiveLevel", "D8_DeliberatePractice", "D9_MotorLearning",
    "D10_StudentAutonomy", "D11_InstrumentRelevance",
]


def extract_scores(results_path: str) -> dict:
    """Extract per-provider, per-dimension scores from promptfoo output."""
    with open(results_path) as f:
        data = json.load(f)

    scores = {}  # {provider_label: {dimension: [scores]}}

    results = data.get("results", data)
    table = results.get("table", {}) if isinstance(results, dict) else {}
    body = table.get("body", [])

    for row in body:
        for output in row.get("outputs", []):
            provider = output.get("provider", output.get("text", "unknown"))
            if provider not in scores:
                scores[provider] = {d: [] for d in ALL_DIMS}

            grading = output.get("gradingResult", {})
            components = grading.get("componentResults", [])

            for i, comp in enumerate(components):
                if i < len(ALL_DIMS):
                    score = comp.get("score", 0)
                    if score is not None:
                        scores[provider][ALL_DIMS[i]].append(score)

    return scores


def check_kill_criterion(results_path: str) -> None:
    scores = extract_scores(results_path)

    # Find Condition B (Socratic prompt)
    b_provider = None
    for provider in scores:
        if "Socratic" in provider or "B:" in provider:
            b_provider = provider
            break

    if not b_provider:
        print("⚠️  Could not find Condition B (Socratic) in results.")
        print(f"   Available providers: {list(scores.keys())}")
        print("   Cannot check kill criterion. Proceeding by default.")
        sys.exit(0)

    # Compute means for target dimensions
    b_scores = scores[b_provider]
    means = {}
    for dim_key, dim_name in TARGET_DIMS.items():
        dim_scores = b_scores.get(dim_key, [])
        means[dim_name] = sum(dim_scores) / len(dim_scores) if dim_scores else 0.0

    overall_mean = sum(means.values()) / len(means) if means else 0.0

    # Print results
    print("=" * 60)
    print("KILL CRITERION CHECK")
    print("=" * 60)
    print(f"\nCondition B ({b_provider}) scores:")
    for name, mean in means.items():
        print(f"  {name:20s}: {mean:.3f}")
    print(f"  {'Mean(D1+D5+D6)':20s}: {overall_mean:.3f}")
    print(f"\n  Threshold:            {KILL_THRESHOLD:.3f}")
    print()

    if overall_mean > KILL_THRESHOLD:
        print("🛑 KILL CRITERION TRIGGERED")
        print("   Prompting alone suffices — fine-tuning adds no value.")
        print("   STOP HERE. Do not proceed to data generation.")
        print("   This is a GOOD outcome — saves training effort.")
        sys.exit(1)
    else:
        print("✅ PROCEED — Fine-tuning has headroom to improve.")
        print(f"   Gap to threshold: {KILL_THRESHOLD - overall_mean:.3f}")

    # Gap analysis across all dimensions for all providers
    print(f"\n{'='*60}")
    print("GAP ANALYSIS — All providers × All dimensions")
    print(f"{'='*60}\n")

    header = f"{'Dimension':25s}"
    for provider in sorted(scores.keys()):
        short = provider[:20]
        header += f" | {short:>20s}"
    print(header)
    print("-" * len(header))

    for dim in ALL_DIMS:
        row = f"{dim:25s}"
        for provider in sorted(scores.keys()):
            dim_scores = scores[provider].get(dim, [])
            mean = sum(dim_scores) / len(dim_scores) if dim_scores else 0.0
            row += f" | {mean:>20.3f}"
        print(row)

    # Training data emphasis recommendations
    print(f"\n{'='*60}")
    print("TRAINING DATA EMPHASIS (based on Condition A floor)")
    print(f"{'='*60}\n")

    a_provider = None
    for provider in scores:
        if "neutral" in provider.lower() and ("A:" in provider or "Base" in provider):
            a_provider = provider
            break

    if a_provider:
        a_scores = scores[a_provider]
        gaps = []
        for dim in ALL_DIMS:
            dim_vals = a_scores.get(dim, [])
            mean = sum(dim_vals) / len(dim_vals) if dim_vals else 0.0
            gaps.append((dim, mean))

        gaps.sort(key=lambda x: x[1])
        print("Weakest dimensions (most training data needed):")
        for dim, mean in gaps[:5]:
            print(f"  {dim:25s}: {mean:.3f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 0_check_kill_criterion.py <results.json>")
        sys.exit(1)
    check_kill_criterion(sys.argv[1])
