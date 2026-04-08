#!/usr/bin/env python3
"""
Run 11 LLM judges on pre-generated fine-tuned model outputs.

Reads: eval/results/finetuned_outputs_2026-04-07.json
Writes: eval/results/finetuned_eval_2026-04-07.json

Usage:
    cd ~/thoven/thoven-finetune
    source .venv/bin/activate && set -a && source .env && set +a
    python3 scripts/4_eval_finetuned.py
"""
import json
import os
import sys
import time
from pathlib import Path

import anthropic

# Import judge prompts and run_judge from validate_judges.py
sys.path.insert(0, str(Path(__file__).parent))
from validate_judges import JUDGE_PROMPTS, DIM_MAP, run_judge

JUDGE_MODEL = "claude-sonnet-4-20250514"
OUTPUTS_PATH = Path("eval/results/finetuned_outputs_2026-04-07.json")
RESULTS_PATH = Path("eval/results/finetuned_eval_2026-04-07.json")

# All LLM-judged dimensions
LLM_DIMENSIONS = list(DIM_MAP.keys())

# D11 is a code-based check
INSTRUMENT_TERMS = {
    "piano": ["piano", "keyboard", "keys", "pedal", "hands"],
    "violin": ["violin", "bow", "string", "fiddle", "vibrato"],
    "guitar": ["guitar", "string", "fret", "chord", "strum", "pick"],
    "voice": ["voice", "sing", "vocal", "breath", "singing"],
    "drums": ["drum", "beat", "stick", "rhythm"],
}


def eval_d11_instrument(test_case_vars, tutor_response):
    """D11: Code-based check — does response mention the student's instrument?"""
    instrument = test_case_vars["instrument"].lower()
    response_lower = tutor_response.lower()
    terms = INSTRUMENT_TERMS.get(instrument, [instrument])
    mentioned = any(term in response_lower for term in terms)
    return {"critique": f"Instrument '{instrument}' mentioned: {mentioned}", "result": "Pass" if mentioned else "Fail"}


def main():
    with open(OUTPUTS_PATH) as f:
        outputs = json.load(f)
    print(f"Loaded {len(outputs)} fine-tuned outputs")
    print(f"Running {len(LLM_DIMENSIONS)} LLM judges + D11 code check\n")

    client = anthropic.Anthropic()
    all_results = []
    dim_scores = {d: [] for d in list(DIM_MAP.keys()) + ["D11_instrument"]}

    total_calls = len(outputs) * len(LLM_DIMENSIONS)
    call_count = 0

    for i, entry in enumerate(outputs):
        vars_ = entry["vars"]
        response = entry["response"]
        print(f"\n[{i+1}/{len(outputs)}] Age {vars_['age']} {vars_['instrument']}: {vars_['student_message'][:50]}...")

        entry_results = {"vars": vars_, "response_preview": response[:200], "judges": {}}

        # Run LLM judges
        for dim in LLM_DIMENSIONS:
            call_count += 1
            judge_prompt = JUDGE_PROMPTS[dim]
            result = run_judge(client, response, vars_, judge_prompt)
            entry_results["judges"][dim] = result

            is_pass = result["result"] == "Pass"
            dim_scores[dim].append(1 if is_pass else 0)
            status = "PASS" if is_pass else "FAIL"
            print(f"  {dim}: {status}  [{call_count}/{total_calls}]")
            time.sleep(0.3)  # Rate limiting

        # Run D11 code check
        d11_result = eval_d11_instrument(vars_, response)
        entry_results["judges"]["D11_instrument"] = d11_result
        is_pass = d11_result["result"] == "Pass"
        dim_scores["D11_instrument"].append(1 if is_pass else 0)
        print(f"  D11_instrument: {'PASS' if is_pass else 'FAIL'}")

        all_results.append(entry_results)

    # Compute summary
    summary = {}
    for dim, scores in dim_scores.items():
        if scores:
            summary[dim] = {
                "mean": round(sum(scores) / len(scores), 3),
                "pass_count": sum(scores),
                "fail_count": len(scores) - sum(scores),
                "total": len(scores),
            }

    # Weighted mean (matching baseline weights)
    weights = {
        "D1a_steps": 1.0, "D1b_question": 1.0, "D2_check": 1.5,
        "D3_age": 1.0, "D4_load": 1.5, "D5_prior": 1.5,
        "D6_growth": 0.5, "D7_higher": 1.0, "D8_practice": 1.0,
        "D9_motor": 0.5, "D10_choice": 0.5, "D11_instrument": 0.5,
    }
    weighted_sum = sum(summary[d]["mean"] * weights.get(d, 1.0) for d in summary if d in weights)
    total_weight = sum(weights.get(d, 1.0) for d in summary if d in weights)
    weighted_mean = round(weighted_sum / total_weight, 3) if total_weight > 0 else 0

    output = {
        "condition": "C: Fine-tuned (bare prompt)",
        "model": "Gemma 4 E4B + LoRA (AndresMartinezThoven/thoven-tutor-v1-lora)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_test_cases": len(outputs),
        "weighted_mean": weighted_mean,
        "per_dimension": summary,
        "detailed_results": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"CONDITION C — FINE-TUNED MODEL RESULTS")
    print(f"{'='*60}")
    print(f"Weighted Mean: {weighted_mean:.3f}")
    print(f"\nPer-Dimension Scores:")
    for dim in ["D1a_steps", "D1b_question", "D2_check", "D3_age", "D4_load",
                "D5_prior", "D6_growth", "D7_higher", "D8_practice", "D9_motor",
                "D10_choice", "D11_instrument"]:
        if dim in summary:
            s = summary[dim]
            print(f"  {dim}: {s['mean']:.3f} ({s['pass_count']}/{s['total']} pass)")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
