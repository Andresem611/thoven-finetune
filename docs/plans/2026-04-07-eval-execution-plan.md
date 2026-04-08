# Fine-Tuned Model Evaluation Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 11-judge binary eval on fine-tuned model outputs, compare against Weekend 1 baseline, produce decision-gate report.

**Architecture:** Fine-tuned outputs were generated in Colab (23 test cases, exact trained weights — no GGUF quantization loss). We create a standalone eval script that runs the 11 LLM judges against these outputs, then compare scores against the existing baseline results. Promptfoo's `config.systemPrompt` bug and GGUF export failure mean we run Condition C (fine-tuned) via pre-generated outputs + direct judge calls, not live Ollama inference.

**Tech Stack:** Python 3, Anthropic API (Claude Sonnet for judging), JSON files, existing `scripts/validate_judges.py` judge prompts.

---

## File Structure

```
eval/
├── results/
│   ├── baseline_v3_2026-04-05.json          # Existing — Conditions A+D
│   ├── opus_socratic_2026-04-05.json         # Existing — Condition E
│   ├── finetuned_outputs_2026-04-07.json     # NEW — Condition C raw outputs (from Colab)
│   └── finetuned_eval_2026-04-07.json        # NEW — Condition C judge scores
scripts/
├── 4_eval_finetuned.py                       # NEW — Run judges on pre-generated outputs
├── 5_compare_conditions.py                   # NEW — Compare C vs A vs D, produce report
```

---

### Task 1: Save Colab Outputs to Eval Results

**Files:**
- Create: `eval/results/finetuned_outputs_2026-04-07.json`

**Context:** User will paste the JSON from `finetuned_outputs.json` generated in Colab. This file has 23 entries, each with `vars` (age, instrument, student_message) and `response` (the fine-tuned model's output).

- [ ] **Step 1: Download `finetuned_outputs.json` from Google Drive**

User downloads from Google Drive to Mac. Then copy to repo:

```bash
cp ~/Downloads/finetuned_outputs.json ~/thoven/thoven-finetune/eval/results/finetuned_outputs_2026-04-07.json
```

- [ ] **Step 2: Verify the file has 23 entries**

```bash
cd ~/thoven/thoven-finetune
python3 -c "
import json
with open('eval/results/finetuned_outputs_2026-04-07.json') as f:
    data = json.load(f)
print(f'Entries: {len(data)}')
print(f'First response preview: {data[0][\"response\"][:100]}...')
print(f'All have vars: {all(\"vars\" in d for d in data)}')
print(f'All have response: {all(\"response\" in d for d in data)}')
"
```

Expected: `Entries: 23`, all checks True.

- [ ] **Step 3: Commit**

```bash
git add eval/results/finetuned_outputs_2026-04-07.json
git commit -m "data: add fine-tuned model outputs from Colab (23 test cases)"
```

---

### Task 2: Create Judge Evaluation Script

**Files:**
- Create: `scripts/4_eval_finetuned.py`

**Context:** This script loads the pre-generated fine-tuned outputs, runs all 11 LLM judges against each response, and saves per-response per-dimension Pass/Fail results. It reuses the same judge prompts from `eval/promptfoo-pedagogy-w2.yaml` (already extracted into `scripts/validate_judges.py` as JUDGE_PROMPTS dict).

- [ ] **Step 1: Create the eval script**

```python
#!/usr/bin/env python3
"""
Run 11 LLM judges on pre-generated fine-tuned model outputs.

Reads: eval/results/finetuned_outputs_2026-04-07.json
Writes: eval/results/finetuned_eval_2026-04-07.json

Usage:
    source .venv/bin/activate && set -a && source .env && set +a
    python3 scripts/4_eval_finetuned.py
"""
import json
import os
import sys
import time
from pathlib import Path

import anthropic

# Import judge prompts from validate_judges.py
sys.path.insert(0, str(Path(__file__).parent))
from validate_judges import JUDGE_PROMPTS, DIM_MAP

JUDGE_MODEL = "claude-sonnet-4-20250514"
OUTPUTS_PATH = Path("eval/results/finetuned_outputs_2026-04-07.json")
RESULTS_PATH = Path("eval/results/finetuned_eval_2026-04-07.json")

# D11 (instrument relevance) is a code-based check, not LLM judge
LLM_DIMENSIONS = [d for d in DIM_MAP.keys() if d != "D11_instrument"]


def run_judge(client, dimension, test_case_vars, tutor_response):
    """Run a single LLM judge on one response. Returns {"pass": bool, "critique": str}."""
    judge_prompt = JUDGE_PROMPTS[dimension]

    # Build the judge input: student context + tutor response
    student_context = (
        f"Student: {test_case_vars['age']}-year-old {test_case_vars['instrument']} student\n"
        f"Student says: {test_case_vars['student_message']}\n\n"
        f"Tutor response:\n{tutor_response}"
    )

    try:
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": f"{judge_prompt}\n\n---\n\nEVALUATE THIS RESPONSE:\n\n{student_context}"}
            ],
        )
        text = response.content[0].text

        # Parse JSON from response
        import re
        json_match = re.search(r'\{[^}]*"pass"\s*:\s*(true|false)[^}]*\}', text, re.IGNORECASE)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "pass": bool(result.get("pass", False)),
                "critique": result.get("critique", text[:200]),
                "score": 1 if result.get("pass", False) else 0,
            }

        # Fallback: look for Pass/Fail keywords
        text_lower = text.lower()
        is_pass = "pass" in text_lower and "fail" not in text_lower
        return {"pass": is_pass, "critique": text[:200], "score": 1 if is_pass else 0}

    except Exception as e:
        print(f"  ERROR on {dimension}: {e}")
        return {"pass": False, "critique": f"Error: {e}", "score": 0, "error": True}


def eval_d11_instrument(test_case_vars, tutor_response):
    """D11: Code-based check — does response mention the student's instrument?"""
    instrument = test_case_vars["instrument"].lower()
    response_lower = tutor_response.lower()

    # Check for instrument name or common synonyms
    instrument_terms = {
        "piano": ["piano", "keyboard", "keys", "pedal"],
        "violin": ["violin", "bow", "string", "fiddle"],
        "guitar": ["guitar", "string", "fret", "chord", "strum"],
        "voice": ["voice", "sing", "vocal", "breath"],
        "drums": ["drum", "beat", "stick", "rhythm"],
    }

    terms = instrument_terms.get(instrument, [instrument])
    mentioned = any(term in response_lower for term in terms)
    return {"pass": mentioned, "critique": f"Instrument '{instrument}' mentioned: {mentioned}", "score": 1 if mentioned else 0}


def main():
    # Load outputs
    with open(OUTPUTS_PATH) as f:
        outputs = json.load(f)
    print(f"Loaded {len(outputs)} fine-tuned outputs")

    client = anthropic.Anthropic()
    all_results = []
    dim_scores = {d: [] for d in list(DIM_MAP.keys())}

    for i, entry in enumerate(outputs):
        vars_ = entry["vars"]
        response = entry["response"]
        print(f"\n[{i+1}/{len(outputs)}] Age {vars_['age']} {vars_['instrument']}: {vars_['student_message'][:50]}...")

        entry_results = {"vars": vars_, "response": response, "judges": {}}

        # Run LLM judges
        for dim in LLM_DIMENSIONS:
            result = run_judge(client, dim, vars_, response)
            entry_results["judges"][dim] = result
            dim_scores[dim].append(result["score"])
            status = "PASS" if result["pass"] else "FAIL"
            print(f"  {dim}: {status}")
            time.sleep(0.5)  # Rate limiting

        # Run D11 code check
        d11_result = eval_d11_instrument(vars_, response)
        entry_results["judges"]["D11_instrument"] = d11_result
        dim_scores.setdefault("D11_instrument", []).append(d11_result["score"])
        print(f"  D11_instrument: {'PASS' if d11_result['pass'] else 'FAIL'}")

        all_results.append(entry_results)

    # Compute summary
    summary = {}
    for dim, scores in dim_scores.items():
        if scores:
            summary[dim] = {
                "mean": sum(scores) / len(scores),
                "pass_count": sum(scores),
                "fail_count": len(scores) - sum(scores),
                "total": len(scores),
            }

    # Weighted mean (same weights as baseline)
    weights = {
        "D1a_steps": 1.0, "D1b_question": 1.0, "D2_check": 1.5,
        "D3_age": 1.0, "D4_load": 1.5, "D5_prior": 1.5,
        "D6_growth": 0.5, "D7_higher": 1.0, "D8_practice": 1.0,
        "D9_motor": 0.5, "D10_choice": 0.5, "D11_instrument": 0.5,
    }
    weighted_sum = sum(summary[d]["mean"] * weights.get(d, 1.0) for d in summary if d in weights)
    total_weight = sum(weights.get(d, 1.0) for d in summary if d in weights)
    weighted_mean = weighted_sum / total_weight if total_weight > 0 else 0

    output = {
        "condition": "C: Fine-tuned (bare prompt)",
        "model": "Gemma 4 E4B + LoRA (AndresMartinezThoven/thoven-tutor-v1-lora)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_test_cases": len(outputs),
        "weighted_mean": round(weighted_mean, 3),
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
    for dim in sorted(summary.keys()):
        s = summary[dim]
        print(f"  {dim}: {s['mean']:.3f} ({s['pass_count']}/{s['total']} pass)")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script loads judge prompts**

```bash
cd ~/thoven/thoven-finetune
source .venv/bin/activate && set -a && source .env && set +a
python3 -c "
from scripts.validate_judges import JUDGE_PROMPTS, DIM_MAP
print(f'Dimensions: {len(DIM_MAP)}')
print(f'Judge prompts loaded: {len(JUDGE_PROMPTS)}')
print(f'Dimensions: {list(DIM_MAP.keys())}')
"
```

Expected: `Dimensions: 11`, `Judge prompts loaded: 11`.

- [ ] **Step 3: Commit**

```bash
git add scripts/4_eval_finetuned.py
git commit -m "feat: add eval script for pre-generated fine-tuned outputs"
```

---

### Task 3: Run the 11-Judge Eval on Fine-Tuned Outputs

**Files:**
- Modify: `eval/results/finetuned_eval_2026-04-07.json` (created by script)

- [ ] **Step 1: Run the eval**

```bash
cd ~/thoven/thoven-finetune
source .venv/bin/activate && set -a && source .env && set +a
python3 scripts/4_eval_finetuned.py
```

Expected: ~23 test cases × 11 judges = 253 API calls. At ~0.5s each with rate limiting = ~3-4 minutes. Cost: ~$2-3 in Sonnet API calls.

Output shows per-dimension Pass/Fail for each test case, then summary.

- [ ] **Step 2: Verify results file**

```bash
python3 -c "
import json
with open('eval/results/finetuned_eval_2026-04-07.json') as f:
    data = json.load(f)
print(f'Weighted Mean: {data[\"weighted_mean\"]}')
for dim, scores in sorted(data['per_dimension'].items()):
    print(f'  {dim}: {scores[\"mean\"]:.3f} ({scores[\"pass_count\"]}/{scores[\"total\"]})')
"
```

- [ ] **Step 3: Commit results**

```bash
git add eval/results/finetuned_eval_2026-04-07.json
git commit -m "data: Condition C eval results — fine-tuned model on 23 test cases"
```

---

### Task 4: Create Comparison Script + Decision Gate Report

**Files:**
- Create: `scripts/5_compare_conditions.py`

**Context:** Compares Condition C (fine-tuned) against A (baseline Gemma), D (Opus ceiling), and E (Opus Socratic). Produces a decision-gate table matching the success criteria from STATUS.md.

- [ ] **Step 1: Create the comparison script**

```python
#!/usr/bin/env python3
"""
Compare fine-tuned model (Condition C) against baseline conditions.
Produces decision-gate report for Weekend 2.

Reads:
  - eval/results/baseline_v3_2026-04-05.json (Conditions A + D)
  - eval/results/opus_socratic_2026-04-05.json (Condition E)
  - eval/results/finetuned_eval_2026-04-07.json (Condition C)

Usage:
    python3 scripts/5_compare_conditions.py
"""
import json
from pathlib import Path

BASELINE_PATH = Path("eval/results/baseline_v3_2026-04-05.json")
OPUS_SOCRATIC_PATH = Path("eval/results/opus_socratic_2026-04-05.json")
FINETUNED_PATH = Path("eval/results/finetuned_eval_2026-04-07.json")

# Baseline per-dimension scores from Weekend 1 (STATUS.md)
BASELINE_SCORES = {
    "A_gemma": {
        "D1a_steps": 0.739, "D2_check": 0.261, "D3_age": 0.674,
        "D4_load": 0.130, "D5_prior": 0.196, "D6_growth": 0.839,
        "D7_higher": 0.783, "D8_practice": 0.630, "D9_motor": 0.557,
        "D10_choice": 0.891, "D11_instrument": 1.000,
        "weighted_mean": 0.605,
    },
    "D_opus": {
        "D1a_steps": 0.870, "D2_check": 0.304, "D3_age": 1.000,
        "D4_load": 0.217, "D5_prior": 0.196, "D6_growth": 0.652,
        "D7_higher": 0.565, "D8_practice": 0.587, "D9_motor": 0.435,
        "D10_choice": 1.000, "D11_instrument": 1.000,
        "weighted_mean": 0.639,
    },
    "E_opus_socratic": {
        "weighted_mean": 0.478,
    },
}


def main():
    # Load fine-tuned results
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
    gap_narrowed_count = 0

    all_dims = sorted(set(list(c_scores.keys()) + [k for k in a_scores.keys() if k != "weighted_mean"]))

    for dim in all_dims:
        a = a_scores.get(dim, None)
        c = c_scores.get(dim, None)
        d = d_scores.get(dim, None)

        if a is None or c is None:
            continue

        diff_a = c - a
        diff_d = (c - d) if d else None

        if diff_a > 0.01:
            dims_improved_vs_a += 1
        elif diff_a < -0.01:
            dims_regressed_vs_a += 1

        if d and a < d:  # Only count gap narrowing where Opus was better
            old_gap = d - a
            new_gap = d - c
            if new_gap < old_gap:
                gap_narrowed_count += 1

        a_str = f"{a:.3f}" if a else "N/A"
        c_str = f"{c:.3f}" if c else "N/A"
        d_str = f"{d:.3f}" if d else "N/A"
        diff_a_str = f"{diff_a:+.3f}" if diff_a else "N/A"
        diff_d_str = f"{diff_d:+.3f}" if diff_d else "N/A"

        print(f"{dim:<20} {a_str:>10} {c_str:>10} {d_str:>10} {diff_a_str:>10} {diff_d_str:>10}")

    print("-" * 70)
    print(f"{'Weighted Mean':<20} {a_scores['weighted_mean']:>10.3f} {c_weighted:>10.3f} {d_scores['weighted_mean']:>10.3f}")

    # Decision gate
    print(f"\n{'=' * 80}")
    print("DECISION GATE")
    print(f"{'=' * 80}")

    total_dims = len([d for d in all_dims if d in c_scores and d in a_scores])

    # Criterion 1: C > A on majority of dimensions
    print(f"\n1. Dimensions improved (C vs A): {dims_improved_vs_a}/{total_dims}")
    if dims_improved_vs_a > 7:
        print("   VERDICT: PASS (>7 = Pass)")
    elif dims_improved_vs_a > 4:
        print("   VERDICT: MARGINAL (5-7)")
    else:
        print("   VERDICT: FAIL (<5)")

    # Criterion 2: No regressions
    print(f"\n2. Dimensions regressed (C vs A): {dims_regressed_vs_a}")
    if dims_regressed_vs_a == 0:
        print("   VERDICT: GREAT (no regressions)")
    elif dims_regressed_vs_a <= 2:
        print("   VERDICT: PASS (minor regressions)")
    else:
        print("   VERDICT: FAIL (significant regressions)")

    # Criterion 3: Opus gap narrows
    opus_gap_old = d_scores["weighted_mean"] - a_scores["weighted_mean"]
    opus_gap_new = d_scores["weighted_mean"] - c_weighted
    gap_change_pct = ((opus_gap_old - opus_gap_new) / opus_gap_old * 100) if opus_gap_old > 0 else 0
    print(f"\n3. Opus gap: {opus_gap_old:.3f} → {opus_gap_new:.3f} ({gap_change_pct:+.0f}% change)")
    if gap_change_pct > 25:
        print("   VERDICT: GREAT (>25% narrowing)")
    elif gap_change_pct > 10:
        print("   VERDICT: PASS (>10% narrowing)")
    else:
        print("   VERDICT: FAIL (gap unchanged)")

    # Criterion 4: C > E (fine-tuned beats Socratic prompting)
    e_weighted = BASELINE_SCORES["E_opus_socratic"]["weighted_mean"]
    print(f"\n4. C ({c_weighted:.3f}) vs E Socratic ({e_weighted:.3f}): {'+' if c_weighted > e_weighted else ''}{c_weighted - e_weighted:.3f}")
    if c_weighted > e_weighted:
        print("   VERDICT: PASS (fine-tuning beats Socratic prompting)")
    else:
        print("   VERDICT: FAIL")

    # Overall recommendation
    print(f"\n{'=' * 80}")
    print("RECOMMENDATION")
    print(f"{'=' * 80}")
    if dims_improved_vs_a >= 7 and dims_regressed_vs_a <= 2 and gap_change_pct > 10:
        print("PROCEED to DPO / data augmentation for v2")
    elif dims_improved_vs_a >= 5:
        print("MARGINAL — consider data quality improvements before DPO")
    else:
        print("STOP — re-examine training data and approach")

    # Save report
    report = {
        "timestamp": ft["timestamp"],
        "c_weighted_mean": c_weighted,
        "a_weighted_mean": a_scores["weighted_mean"],
        "d_weighted_mean": d_scores["weighted_mean"],
        "e_weighted_mean": e_weighted,
        "dims_improved_vs_a": dims_improved_vs_a,
        "dims_regressed_vs_a": dims_regressed_vs_a,
        "opus_gap_change_pct": round(gap_change_pct, 1),
    }
    report_path = Path("eval/results/decision_gate_2026-04-07.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the comparison**

```bash
python3 scripts/5_compare_conditions.py
```

Expected: Prints the full comparison table and decision gate verdicts.

- [ ] **Step 3: Commit**

```bash
git add scripts/5_compare_conditions.py eval/results/decision_gate_2026-04-07.json
git commit -m "feat: add comparison script + decision gate report"
```

---

### Task 5: Update STATUS.md + Paper Outline

**Files:**
- Modify: `STATUS.md`
- Modify: `docs/paper_outline.md`

- [ ] **Step 1: Update STATUS.md with Condition C results**

Add to the Weekend 2 section:

```markdown
### Condition C — Fine-Tuned Model Results (2026-04-07)

| # | Dimension | Gemma (A) | Fine-Tuned (C) | Opus (D) | C vs A |
|---|-----------|:---------:|:-------------:|:--------:|:------:|
| D1a | Scaffolding | 0.739 | [FROM EVAL] | 0.870 | [DELTA] |
| D2 | Comprehension | 0.261 | [FROM EVAL] | 0.304 | [DELTA] |
...

**Weighted Mean:** A=0.605, C=[FROM EVAL], D=0.639

**Decision Gate:** [FROM 5_compare_conditions.py output]

**Key Finding:** Model learned pedagogical structure but outputs in teacher-guide register rather than direct student conversation. Training data quality issue — ConvoLearn + synthetic data was teacher-facing.
```

Values filled in from the eval results JSON after Task 3 completes.

- [ ] **Step 2: Update paper outline with results**

Add Condition C data to the Results section of `docs/paper_outline.md`.

- [ ] **Step 3: Final commit**

```bash
git add STATUS.md docs/paper_outline.md
git commit -m "docs: add Condition C results to STATUS + paper outline"
```

---

## Execution Checklist

| # | Task | Depends On | Est. Time |
|---|------|-----------|-----------|
| 1 | Save Colab outputs to repo | Colab generation completes | 2 min |
| 2 | Create judge eval script | Task 1 | 5 min |
| 3 | Run 11-judge eval | Task 2 + ANTHROPIC_API_KEY | 4 min (~$2-3) |
| 4 | Create comparison + decision gate | Task 3 | 3 min |
| 5 | Update STATUS + paper | Task 4 | 5 min |

**Total:** ~20 min + Colab output download

**Blocking prerequisite:** `finetuned_outputs.json` from Colab (currently generating).
