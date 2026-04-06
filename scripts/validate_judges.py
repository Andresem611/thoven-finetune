#!/usr/bin/env python3
"""
Judge Validation Script — following validate-evaluator skill from hamelsmu/evals-skills.

Steps:
1. Load human labels from eval/labeling/labels.json
2. Split into train (15%) / dev (45%) / test (40%)
3. Run LLM judge on dev set for each dimension
4. Compute TPR/TNR
5. Report disagreements for iteration

Usage:
    python scripts/validate_judges.py --dimension D2_check
    python scripts/validate_judges.py --dimension D4_load
    python scripts/validate_judges.py --all
"""
import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
import numpy as np

JUDGE_MODEL = "claude-sonnet-4-20250514"
LABELS_PATH = Path("eval/labeling/labels.json")
RESULTS_PATH = Path("eval/results/baseline_v3_2026-04-05.json")
VALIDATION_DIR = Path("eval/validation")

# Map dimension short names to labels.json field names
DIM_MAP = {
    "D1a_steps": "D1a_steps",
    "D1b_question": "D1b_question",
    "D2_check": "D2_check",
    "D4_load": "D4_load",
    "D5_prior": "D5_prior",
}

# Judge prompts for each dimension (binary Pass/Fail)
JUDGE_PROMPTS = {
    "D2_check": """You are evaluating whether a music tutor's response includes a comprehension check or practice invitation.

## Definitions

PASS: The tutor includes ANY of these before or after instruction:
- Asks the student to try, play, or demonstrate something ("Try playing just the right hand", "Can you try that?")
- Asks the student to explain back or describe what they notice ("What do you hear when you do X?", "Can you hear the difference?")
- Invites the student to practice a specific exercise ("Would you like to try the pedal exercise?", "Let's try this together")
- Asks a diagnostic question about the student's experience ("How does it feel when you do X?", "What happens when you try?")
All of these count as verification — they require the student to engage with the material, not just passively receive instruction.

FAIL: The tutor ONLY explains/teaches and ends without ANY invitation to try, demonstrate, or reflect. Pure information delivery with no student action requested. Generic closers like "Does that make sense?" or "Any questions?" alone do NOT count — but "Would you like to try X?" DOES count because it invites specific action.

## Examples

### Example 1: PASS
Student (7yo piano): "I can't play with both hands at the same time"
Response: "Let's start with just the right hand. Can you play the first four notes for me? Once that feels easy, we'll try the left hand alone, and then see what happens when we put them together very slowly."
Critique: The tutor asks the student to play the right hand ("Can you play the first four notes for me?") before moving to the next step. This is a genuine comprehension/skill check embedded in the instruction flow.
Result: Pass

### Example 2: FAIL
Student (10yo piano): "Why do I have to practice scales? They're boring"
Response: "Scales are like a superpower for piano players! Just like soccer players do drills, scales train your fingers to be fast and accurate. They help you learn new songs quicker because your fingers already know the patterns. Try thinking of them as a game — see how fast you can play C major without mistakes!"
Critique: The tutor explains why scales matter and suggests reframing them as a game, but never checks whether the student understood the connection between scales and song-learning. The closing suggestion "try thinking of them as a game" is an instruction, not a check. No verification of understanding.
Result: Fail

### Example 3: PASS (borderline)
Student (14yo piano): "I don't understand how to voice the melody over the accompaniment"
Response: "Voicing is about making the melody louder than the accompaniment. Try this: play just the melody notes with your right hand, pressing firmly. Now play just the left hand very softly. Can you hear the difference in volume? That difference is what we want when both hands play together."
Critique: The tutor asks "Can you hear the difference in volume?" — this is a borderline check. It asks the student to notice something specific (volume contrast) rather than just asking "do you understand?" The tutor is verifying the student can perceive the concept before combining hands. Borderline but passes.
Result: Pass

## Output Format
Respond with JSON only:
{"critique": "detailed assessment referencing specific evidence from the response", "result": "Pass or Fail"}""",

    "D4_load": """You are evaluating whether a music tutor's response manages cognitive load by focusing on ONE main concept.

## Definitions

PASS: The response addresses one main concept or skill, even if it uses numbered steps to break that concept down. Sequential steps for ONE skill (e.g., "Step 1: play right hand. Step 2: play left hand. Step 3: combine slowly") is PASS — that's scaffolding, not overload. A structured response with clear focus on a single topic passes even if it's detailed.

FAIL: The response introduces 3 or more UNRELATED concepts (e.g., "work on posture, AND practice scales, AND try sight-reading, AND listen to recordings"). The key is whether items are independent topics vs sequential steps for one skill. Also FAIL if it uses undefined jargon the student hasn't learned, or is an unstructured wall of text.

IMPORTANT: A numbered list is NOT automatically overload. Ask: "Are these items steps toward ONE skill, or independent topics?"
- "1. Practice hands separately. 2. Combine slowly. 3. Speed up gradually" = ONE concept (hand coordination) in steps = PASS
- "1. Fix your posture. 2. Practice scales. 3. Learn music theory. 4. Try sight-reading" = FOUR unrelated topics = FAIL
- A long response that stays focused on ONE technique with multiple tips for that technique = PASS
- A response that covers posture AND technique AND theory AND repertoire = FAIL

When in doubt: if the student could summarize the response as "my tutor helped me with [ONE thing]", it's PASS.

## Examples

### Example 1: PASS
Student (7yo piano): "I can't play with both hands at the same time"
Response: "Playing with both hands is tricky! Let's focus on just one thing: getting each hand really comfortable on its own first. Can you play the right hand part five times in a row without stopping? Once that feels easy, we'll try the left hand."
Critique: One concept: practice hands separately. The response stays focused on this single step without jumping to rhythm, posture, or fingering. Clear and manageable for a 7-year-old.
Result: Pass

### Example 2: FAIL
Student (8yo guitar): "The strings hurt my fingers"
Response: "That's normal! Here are some tips: 1) Press with your fingertips, not the pads. 2) Start with nylon strings instead of steel. 3) Practice for shorter sessions. 4) Try using a capo to reduce tension. 5) Make sure your action height is correct. 6) Your calluses will develop over time. Also, make sure you're using proper posture..."
Critique: Six independent suggestions plus a posture aside. An 8-year-old cannot process and act on 6+ tips. Even though each tip is correct, the combined load overwhelms. Should have picked ONE actionable thing (e.g., "let's check how you're pressing the strings — show me your left hand").
Result: Fail

### Example 3: PASS (borderline)
Student (12yo guitar): "I can play chords but switching between them is slow"
Response: "Chord switching is all about muscle memory. Here's what I want you to try: practice switching between just G and C — two chords, nothing else. Place your fingers on G, then lift them ALL at once and place them on C. The key is lifting all fingers together, not one at a time. Try it 10 times slowly."
Critique: Technically introduces two ideas (lift-all-together technique + repetition count) but they're tightly connected — the technique and how to practice it. This is one focused concept with a specific method. Borderline but passes.
Result: Pass

## Output Format
Respond with JSON only:
{"critique": "detailed assessment referencing specific evidence from the response", "result": "Pass or Fail"}""",
}


def load_data():
    """Load labels and baseline responses."""
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(RESULTS_PATH) as f:
        baseline = json.load(f)

    # Map baseline responses to label IDs
    results = baseline["results"]["results"]
    # Labels are ordered: 0-19 Gemma neutral, 20-39 Opus
    gemma = [r for r in results if "neutral" in r["provider"].get("label", "") and "Gemma" in r["provider"].get("label", "")]
    opus = [r for r in results if "Opus" in r["provider"].get("label", "")]
    ordered = gemma[:20] + opus[:20]

    for i, label_entry in enumerate(labels):
        if i < len(ordered):
            r = ordered[i]
            label_entry["response"] = r.get("response", {}).get("output", "")
            label_entry["vars"] = r.get("vars", {})

    return labels


def split_data(labels, dim_field, seed=42):
    """Split into train/dev/test following validate-evaluator skill."""
    np.random.seed(seed)

    # Get labels for this dimension
    labeled = [(i, entry) for i, entry in enumerate(labels)
               if entry.get(dim_field) in ("Pass", "Fail")]

    # Stratified split
    passes = [x for x in labeled if x[1][dim_field] == "Pass"]
    fails = [x for x in labeled if x[1][dim_field] == "Fail"]

    np.random.shuffle(passes)
    np.random.shuffle(fails)

    # 15% train, 45% dev, 40% test
    def split_group(group):
        n = len(group)
        train_n = max(1, int(n * 0.15))
        test_n = max(1, int(n * 0.40))
        dev_n = n - train_n - test_n
        return group[:train_n], group[train_n:train_n+dev_n], group[train_n+dev_n:]

    p_train, p_dev, p_test = split_group(passes)
    f_train, f_dev, f_test = split_group(fails)

    train = p_train + f_train
    dev = p_dev + f_dev
    test = p_test + f_test

    return train, dev, test


def run_judge(client, response_text, student_vars, judge_prompt):
    """Run the LLM judge on a single response."""
    age = student_vars.get("age", "?")
    instrument = student_vars.get("instrument", "?")
    student_msg = student_vars.get("student_message", "?")

    user_prompt = f"""Student ({age}yo {instrument}): "{student_msg}"

Tutor Response:
{response_text}"""

    try:
        result = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=500,
            system=judge_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = result.content[0].text.strip()
        # Parse JSON
        if text.startswith("{"):
            parsed = json.loads(text)
        else:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{[^}]+\}', text)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = {"critique": text, "result": "Pass" if "pass" in text.lower() else "Fail"}
        return parsed
    except Exception as e:
        return {"critique": f"Error: {e}", "result": "Error"}


def compute_metrics(human_labels, judge_labels):
    """Compute TPR and TNR."""
    tp = sum(1 for h, j in zip(human_labels, judge_labels) if h == "Pass" and j == "Pass")
    fn = sum(1 for h, j in zip(human_labels, judge_labels) if h == "Pass" and j == "Fail")
    tn = sum(1 for h, j in zip(human_labels, judge_labels) if h == "Fail" and j == "Fail")
    fp = sum(1 for h, j in zip(human_labels, judge_labels) if h == "Fail" and j == "Pass")

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "tpr": round(tpr, 3),
        "tnr": round(tnr, 3),
        "accuracy": round((tp + tn) / (tp + fn + tn + fp), 3) if (tp + fn + tn + fp) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=str, help="Dimension to validate (e.g., D2_check)")
    parser.add_argument("--all", action="store_true", help="Validate all dimensions")
    parser.add_argument("--dev-only", action="store_true", help="Only run on dev set (for iteration)")
    args = parser.parse_args()

    dims = list(JUDGE_PROMPTS.keys()) if args.all else [args.dimension]

    client = anthropic.Anthropic()
    labels = load_data()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    for dim in dims:
        if dim not in JUDGE_PROMPTS:
            print(f"No judge prompt for {dim}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"VALIDATING: {dim}")
        print(f"{'='*60}")

        train, dev, test = split_data(labels, dim)

        print(f"Split: train={len(train)}, dev={len(dev)}, test={len(test)}")
        print(f"  Train: {sum(1 for _,e in train if e[dim]=='Pass')}P / {sum(1 for _,e in train if e[dim]=='Fail')}F")
        print(f"  Dev:   {sum(1 for _,e in dev if e[dim]=='Pass')}P / {sum(1 for _,e in dev if e[dim]=='Fail')}F")
        print(f"  Test:  {sum(1 for _,e in test if e[dim]=='Pass')}P / {sum(1 for _,e in test if e[dim]=='Fail')}F")

        # Run judge on dev set
        print(f"\nRunning judge on dev set ({len(dev)} examples)...")
        judge_prompt = JUDGE_PROMPTS[dim]

        dev_results = []
        for idx, entry in dev:
            result = run_judge(client, entry.get("response", ""), entry.get("vars", {}), judge_prompt)
            human = entry[dim]
            judge = result.get("result", "Error")
            agree = "✓" if human == judge else "✗"
            print(f"  [{agree}] ID {idx}: human={human}, judge={judge}")
            if human != judge:
                print(f"      Critique: {result.get('critique', '')[:120]}...")
            dev_results.append({
                "id": idx, "human": human, "judge": judge,
                "critique": result.get("critique", ""),
                "agree": human == judge
            })

        # Compute metrics
        human_labels = [r["human"] for r in dev_results]
        judge_labels = [r["judge"] for r in dev_results]
        metrics = compute_metrics(human_labels, judge_labels)

        print(f"\n--- DEV SET METRICS ({dim}) ---")
        print(f"TPR (Pass recall): {metrics['tpr']} (target: >0.90)")
        print(f"TNR (Fail recall): {metrics['tnr']} (target: >0.90)")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Confusion: TP={metrics['tp']} FN={metrics['fn']} TN={metrics['tn']} FP={metrics['fp']}")

        status = "PASS" if metrics["tpr"] >= 0.80 and metrics["tnr"] >= 0.80 else "NEEDS ITERATION"
        print(f"Status: {status}")

        # Run on test set if dev passes
        if not args.dev_only and metrics["tpr"] >= 0.80 and metrics["tnr"] >= 0.80:
            print(f"\nRunning judge on TEST set ({len(test)} examples)...")
            test_results = []
            for idx, entry in test:
                result = run_judge(client, entry.get("response", ""), entry.get("vars", {}), judge_prompt)
                human = entry[dim]
                judge = result.get("result", "Error")
                test_results.append({"id": idx, "human": human, "judge": judge})

            test_human = [r["human"] for r in test_results]
            test_judge = [r["judge"] for r in test_results]
            test_metrics = compute_metrics(test_human, test_judge)

            print(f"\n--- TEST SET METRICS ({dim}) ---")
            print(f"TPR: {test_metrics['tpr']}")
            print(f"TNR: {test_metrics['tnr']}")
            print(f"Status: {'PASS' if test_metrics['tpr'] >= 0.80 and test_metrics['tnr'] >= 0.80 else 'FAIL'}")

        # Save results
        output = {
            "dimension": dim,
            "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
            "dev_metrics": metrics,
            "dev_results": dev_results,
            "status": status,
        }
        with open(VALIDATION_DIR / f"{dim}_validation.json", "w") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
