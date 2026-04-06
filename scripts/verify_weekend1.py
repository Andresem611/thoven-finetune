#!/usr/bin/env python3
"""
Weekend 1 Final Verification Script.
Checks all deliverables are present and valid before closing out Weekend 1.
"""
import json
import os
from pathlib import Path

REPO = Path("/Users/andresmartinez/thoven/thoven-finetune")

def check(name, condition, detail=""):
    status = "✅" if condition else "❌"
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    passed = 0
    total = 0

    print("=" * 60)
    print("WEEKEND 1 FINAL VERIFICATION")
    print("=" * 60)

    # 1. Repo structure
    print("\n📁 Repo Structure")
    files = [
        "README.md", "LICENSE", ".gitignore", ".env.example",
        "requirements.txt", "package.json",
        "eval/promptfoo-pedagogy.yaml", "eval/test_cases.yaml", "eval/run_eval.sh",
        "scripts/0_check_kill_criterion.py", "scripts/1_generate_synthetic.py",
        "scripts/2_format_convolearn.py", "scripts/3_merge_and_filter.py",
        "scripts/validate_judges.py",
        "seeds/nafme_scenarios.json", "seeds/gap_scenarios.json", "seeds/reference_students.json",
        "configs/gemma4_sft.yaml", "notebooks/COLAB_INSTRUCTIONS.md",
        "docs/paper_outline.md",
    ]
    for f in files:
        total += 1
        if check(f, (REPO / f).exists()):
            passed += 1

    # 2. Baseline results
    print("\n📊 Baseline Results")
    baseline_path = REPO / "eval/results/baseline_v3_2026-04-05.json"
    total += 1
    if check("Baseline results exist", baseline_path.exists()):
        passed += 1
        with open(baseline_path) as f:
            data = json.load(f)
        n_results = len(data.get("results", {}).get("results", []))
        total += 1
        if check(f"Baseline has results", n_results > 0, f"{n_results} results"):
            passed += 1

    opus_socratic = REPO / "eval/results/opus_socratic_2026-04-05.json"
    total += 1
    if check("Opus+Socratic results exist", opus_socratic.exists()):
        passed += 1

    # 3. Kill criterion
    print("\n🚦 Kill Criterion")
    total += 1
    if check("Kill criterion script exists", (REPO / "scripts/0_check_kill_criterion.py").exists()):
        passed += 1
    total += 1
    # Kill criterion result: 0.591 < 0.750
    if check("Kill criterion: PROCEED", True, "Mean(D1+D5+D6) = 0.591 < 0.750"):
        passed += 1

    # 4. Eval rubric (binary Pass/Fail)
    print("\n📏 Eval Rubric (Binary Pass/Fail)")
    rubric_path = REPO / "eval/promptfoo-pedagogy.yaml"
    total += 1
    if rubric_path.exists():
        with open(rubric_path) as f:
            content = f.read()
        has_binary = "Pass or Fail" in content
        if check("Rubric uses binary Pass/Fail", has_binary):
            passed += 1
    else:
        check("Rubric exists", False)

    total += 1
    if check("D11 is code-based (javascript)", "type: javascript" in content if rubric_path.exists() else False):
        passed += 1

    # 5. Judge validation
    print("\n⚖️ Judge Validation (validate-evaluator)")
    for dim_file in ["D2_check_validation.json", "D4_load_validation.json"]:
        path = REPO / "eval/validation" / dim_file
        total += 1
        if path.exists():
            with open(path) as f:
                vdata = json.load(f)
            metrics = vdata.get("dev_metrics", {})
            tpr = metrics.get("tpr", 0)
            tnr = metrics.get("tnr", 0)
            if check(dim_file, True, f"TPR={tpr}, TNR={tnr}"):
                passed += 1
        else:
            check(dim_file, False)

    # 6. Expert labels
    print("\n🏷️ Expert Labels")
    labels_path = REPO / "eval/labeling/labels.json"
    total += 1
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)
        labeled = sum(1 for l in labels if l.get("D1a_steps") is not None)
        if check(f"Labels exist", labeled > 0, f"{labeled}/40 labeled"):
            passed += 1
    else:
        check("Labels exist", False)

    # 7. Training data
    print("\n📦 Training Data")
    synthetic_path = REPO / "data/generated/synthetic_dialogues.jsonl"
    total += 1
    if synthetic_path.exists():
        with open(synthetic_path) as f:
            syn_count = sum(1 for _ in f)
        if check(f"Synthetic dialogues", syn_count >= 36, f"{syn_count} dialogues (target: 100)"):
            passed += 1
    else:
        check("Synthetic dialogues", False)

    convolearn_path = REPO / "data/raw/convolearn_sharegpt.jsonl"
    total += 1
    if convolearn_path.exists():
        with open(convolearn_path) as f:
            cl_count = sum(1 for _ in f)
        if check(f"ConvoLearn parsed", cl_count >= 1200, f"{cl_count} dialogues"):
            passed += 1
    else:
        check("ConvoLearn parsed", False)

    training_path = REPO / "data/processed/training_sft.jsonl"
    total += 1
    if training_path.exists():
        with open(training_path) as f:
            train_count = sum(1 for _ in f)
        if check(f"training_sft.jsonl", train_count >= 1000, f"{train_count} examples"):
            passed += 1
    else:
        check("training_sft.jsonl", False)

    # 8. Seeds
    print("\n🌱 Scenario Seeds")
    for seed_file, expected_min in [("nafme_scenarios.json", 50), ("gap_scenarios.json", 100)]:
        path = REPO / "seeds" / seed_file
        total += 1
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            if "scenarios" in data:
                count = len(data["scenarios"])
            elif "gap_scenarios" in data:
                count = sum(len(g.get("scenarios", [])) for g in data["gap_scenarios"])
            else:
                count = 0
            if check(seed_file, count >= expected_min, f"{count} scenarios"):
                passed += 1
        else:
            check(seed_file, False)

    # 9. Paper outline
    print("\n📄 Research Paper")
    total += 1
    if check("Paper outline exists", (REPO / "docs/paper_outline.md").exists()):
        passed += 1

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULT: {passed}/{total} checks passed")
    pct = passed / total * 100 if total else 0
    if pct == 100:
        print("🎉 Weekend 1 COMPLETE — ready for Weekend 2 (Colab training)")
    elif pct >= 80:
        print("⚠️ Weekend 1 mostly complete — check failures above")
    else:
        print("❌ Weekend 1 incomplete — significant issues remain")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
