# Wave 2: Fine-Tuning with Register-Corrected Data

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate training data with explicit student-facing register control, retrain, and re-eval to beat the 0.605 baseline.

**Architecture:** Same eval pipeline (11 judges, 23 test cases). New data generation approach: persona-locked prompts that force direct student conversation. Model choice to be researched (Gemma 4 E4B vs text-only Gemma 3). AutoResearch for hyperparam optimization if time allows.

**Timeline:** Next week (week of 2026-04-13)

---

## Wave 1 Learnings (DO NOT REPEAT)

| Problem | Root Cause | Fix for Wave 2 |
|---------|-----------|----------------|
| Teacher-guide register (0/4 criteria) | Training data was teacher-facing: ConvoLearn meta-commentary + Sonnet's advisory voice | Persona-locked prompts: "You ARE the tutor. Speak directly TO the student." |
| D1b Questions collapsed (47.8% → 4.3%) | Model learned to describe what questions to ask instead of asking them | Include explicit question requirements in data gen prompts |
| D10 Student Choice collapsed (89.1% → 0%) | Model offers teacher multiple approach options | Train on examples where tutor offers student choices directly |
| D3 Age-appropriate collapsed (67.4% → 8.7%) | Adult-level prose in training data | Age-differentiated data gen: separate prompts for 5-7, 8-10, 11-13, 14-16 |
| GGUF export blocked | Gemma 4 E4B is VLM — vision/audio towers block export | Research text-only alternatives (Gemma 3 1B/4B) |
| ConvoLearn cross-domain transfer failed | Earth science tutoring patterns don't map to direct music instruction | Drop ConvoLearn entirely or use only as negative examples |

---

## Phase 1: Research (Day 1)

### Task 1: Model Selection Research
- [ ] Compare Gemma 4 E4B (8B VLM, GGUF broken) vs Gemma 3 4B (VLM, GGUF works) vs Gemma 3 1B (text-only, GGUF clean)
- [ ] Run capability test: same 5 test cases on each base model, score with judges
- [ ] Decision: which model balances capability + deployability?

### Task 2: Data Strategy Research
- [x] ~~Research NVIDIA NeMo DataDesigner~~ — **EVALUATED: NOT SUITABLE for Wave 2**
  - No native Anthropic support (would need LiteLLM proxy)
  - No pre-built pedagogical rubric
  - Multi-turn requires manual column chaining, not native dialogue loop
  - Value kicks in at 10,000+ rows (we need 200)
  - Register/voice control is prompt-only, same as direct generation
  - **Decision: Use improved two-LLM script (Sonnet tutor + Haiku student) with register-locked prompts**
  - Revisit DataDesigner if we scale to 5,000+ training examples in future waves

### Task 3: AutoResearch Evaluation
- [ ] Clone Karpathy's AutoResearch repo (github.com/karpathy/autoresearch)
- [ ] Evaluate: can we use it for hyperparameter optimization on our training setup?
- [ ] Scope decision: use as-is for LR/warmup/LoRA rank search, OR build custom fork with 3-judge eval metric?
- [ ] If custom fork: estimate effort and API cost per experiment (~$0.50 for 3-judge subset)

---

## Phase 2: Data Generation (Day 2-3)

### Task 4: Design Register-Controlled Prompts

**Tutor persona prompt (Sonnet):**
```
You ARE a music tutor talking directly to a {age}-year-old {instrument} student.

CRITICAL RULES:
1. Speak TO the student using "you" and "your" — never describe what you WOULD say
2. Ask at least one question that checks understanding or prompts the student to try something
3. Match vocabulary to the student's age: {age_guidance}
4. Offer the student a choice ("Would you like to try X or Y?")
5. Keep cognitive load low: one concept at a time, max 3 steps

DO NOT:
- Use "Here is how you can respond" or "Here are some approaches"
- Give multiple numbered options for the teacher to choose from
- Write in teacher-guide format
- Use academic language with young students

The student says: "{student_message}"
```

**Age guidance lookup:**
- Ages 5-7: Simple words, playful metaphors, max 2-3 sentences per step
- Ages 8-10: Concrete language, relatable analogies, short paragraphs
- Ages 11-13: Can handle musical terms with definitions, slightly longer
- Ages 14-16: Near-adult vocabulary, can discuss theory concepts

### Task 5: Generate Student-Facing Data
- [ ] Generate 200 synthetic conversations using register-locked prompts
- [ ] Use ALL 210 scenario seeds (50 NAfME + 160 gap-targeted)
- [ ] Student LLM (Haiku): realistic student responses with follow-ups and confusion
- [ ] Quality gate: sample 20 conversations, manually check register (must be student-facing)

### Task 6: Data Quality Audit
- [ ] Run 3-judge subset (D1b, D3, D10) on 20 random generated responses
- [ ] Threshold: >80% pass rate on all three register-sensitive dimensions
- [ ] If fails: iterate prompts before proceeding to training
- [ ] Drop ConvoLearn entirely — use only new synthetic data

---

## Phase 3: Training (Day 3-4)

### Task 7: Training Config
- [ ] Model: [from Task 1 decision]
- [ ] If AutoResearch viable (Task 3): run overnight hyperparam search first
- [ ] If not: use Wave 1 config (LR=1e-4, LoRA r=16, 3 epochs) as starting point
- [ ] Adjust for dataset size change (200 vs 1,354 examples — fewer epochs or different LR)

### Task 8: Train on Colab
- [ ] Upload register-corrected training data
- [ ] Run training (~1-2 hours on T4)
- [ ] If text-only model chosen: GGUF export should work — verify
- [ ] Push LoRA to HuggingFace Hub (v2-lora)

---

## Phase 4: Evaluation (Day 4-5)

### Task 9: Generate Eval Outputs
- [ ] If GGUF works: Ollama local inference → Promptfoo full 4-condition eval
- [ ] If GGUF blocked: Colab inference → scripts/4_eval_finetuned.py (same as Wave 1)

### Task 10: Run Decision Gate
- [ ] Run scripts/5_compare_conditions.py
- [ ] Compare Wave 2 C vs Wave 1 C vs baseline A vs Opus D
- [ ] Key metrics: did register-sensitive dims (D1b, D3, D10) recover?

### Task 11: Document Results
- [ ] Update STATUS.md with Wave 2 results
- [ ] Update paper outline with comparative analysis
- [ ] If PASS: proceed to paper draft
- [ ] If FAIL: analyze which dims still fail, plan Wave 3

---

## Success Criteria (Wave 2)

| Metric | Fail | Pass | Great |
|--------|------|------|-------|
| Weighted mean vs baseline | C < 0.605 (A) | C > 0.605 | C > 0.639 (D) |
| Register recovery (D1b, D3, D10) | Any still < 0.200 | All > 0.400 | All > baseline |
| Dimensions improved (C vs A) | < 5 of 12 | > 7 of 12 | > 9 of 12 |
| No regression from baseline | > 3 regress | < 2 regress | 0 regress |
| D6/D9 gains preserved | Both drop | One holds | Both hold |

---

## Cost Estimate

| Item | Estimated |
|------|-----------|
| Data generation (200 conversations, Sonnet+Haiku) | $3-5 |
| Data quality audit (60 judge calls) | $1 |
| AutoResearch (if used, 8 hrs × $3/hr H100) | $24 (optional) |
| Colab training | $0 (free T4) |
| Condition C eval (253 judge calls) | $3 |
| **Total (without AutoResearch)** | **$7-9** |
| **Total (with AutoResearch)** | **$31-33** |
