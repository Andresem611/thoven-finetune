# Wave 2 Fine-Tuning Session Prompt

> Paste this into a new Claude Code session to resume the fine-tuning project.

---

## Context

You are continuing a fine-tuning research project for Thoven, a music education platform. The project fine-tunes a small open-weight LLM to be a pedagogically-sound music tutor, evaluated by 11 learning-science-grounded LLM judges.

**Repo:** `~/thoven/thoven-finetune/`
**Wave 1 results:** FAIL (0/4 criteria). Weighted mean dropped from 0.605 (baseline) to 0.231 (fine-tuned).
**Root cause:** Training data register mismatch — model learned pedagogy concepts but outputs them as teacher-guide advice ("Here is how you can respond...") instead of direct student conversation.

### What Wave 1 Proved
- The eval pipeline works (11 judges, 23 test cases, decision gate)
- The model CAN learn from fine-tuning (D6 growth mindset +12%, D9 motor awareness +18%)
- Training data quality is the bottleneck, not model capability or training setup

### What Collapsed in Wave 1
- D10 Student Choice: 89.1% → 0% (gives teacher options, not student choices)
- D3 Age-Appropriate: 67.4% → 8.7% (writes at adult level)
- D7 Higher-Order: 78.3% → 21.7% (explains instead of prompting discovery)
- D1b Questions: 47.8% → 4.3% (talks about questions instead of asking them)

---

## Your Mission (Wave 2)

Execute the plan at `docs/plans/2026-04-07-wave2-finetuning-plan.md`. The plan has 4 phases:

### Phase 1: Research (parallelize these)
1. **Model selection:** Compare Gemma 4 E4B (8B, VLM, GGUF broken) vs Gemma 3 4B (VLM, GGUF works) vs Gemma 3 1B (text-only, clean GGUF). Run 5 test cases on each base model, score with the 3 register-sensitive judges (D1b, D3, D10). Pick the best balance of capability + deployability.

2. **NVIDIA NeMo DataDesigner** (`github.com/NVIDIA-NeMo/DataDesigner`): Evaluate whether it supports persona-locked multi-turn conversation generation for our use case. If yes, use it. If no, iterate on our custom two-LLM approach with register-controlled prompts.

3. **Karpathy's AutoResearch** (`github.com/karpathy/autoresearch`): Evaluate two scopes:
   - **Minimal:** Use as-is for hyperparam optimization (LR, warmup, LoRA rank). Requires GPU (RunPod H100 ~$24 for 8 hrs).
   - **Custom fork:** Replace `val_bpb` with a 3-judge eval metric (D1b, D3, D10 — the register-sensitive dims). ~$0.50/experiment in API costs. Estimate effort to build.
   - **Decision:** Is either scope worth the investment for Wave 2, or should we defer AutoResearch to Wave 3?

### Phase 2: Data Generation
- Drop ConvoLearn entirely (earth science data caused register contamination)
- Generate 200 NEW synthetic conversations with register-locked prompts
- Prompts MUST force: direct student address ("you"/"your"), at least 1 question, age-appropriate vocabulary, student choices
- Quality gate: 3-judge audit on 20 samples before proceeding

### Phase 3: Training
- Use model from Phase 1 decision
- Same Colab T4 free tier
- If GGUF works on chosen model, verify export before committing to training run

### Phase 4: Evaluation
- Same eval pipeline: `scripts/4_eval_finetuned.py` → `scripts/5_compare_conditions.py`
- Compare Wave 2 vs Wave 1 vs baseline
- Key question: did register-sensitive dimensions (D1b, D3, D10) recover?

---

## Key Files

| File | Purpose |
|------|---------|
| `STATUS.md` | Full results history, cost tracking |
| `docs/paper_outline.md` | Research paper structure with Wave 1 findings |
| `docs/plans/2026-04-07-wave2-finetuning-plan.md` | Detailed Wave 2 plan |
| `eval/results/finetuned_eval_2026-04-07.json` | Wave 1 judge results (C=0.231) |
| `eval/results/decision_gate_2026-04-07.json` | Wave 1 decision gate (0/4 FAIL) |
| `scripts/4_eval_finetuned.py` | Run judges on pre-generated outputs |
| `scripts/5_compare_conditions.py` | Compare conditions + decision gate |
| `scripts/1_generate_synthetic.py` | Wave 1 data gen script (needs register fix) |
| `configs/gemma4_sft.yaml` | Training config (may change based on model selection) |
| `.claude/skills/llm-finetuning-music-pedagogy/SKILL.md` | Custom fine-tuning skill |

## Skills to Invoke
- `/fine-tuning-expert` — for training config review
- `/evals-skills:validate-evaluator` — if judge iteration needed
- `/evals-skills:write-judge-prompt` — if new judges needed
- `/edtech-learning-science` — for pedagogy grounding

## Constraints
- This is a **research paper project**, not production deployment
- Budget: ~$10 for data gen + eval, $24 optional for AutoResearch
- Timeline: complete within the week
- User (Andres) is non-technical co-founder — explain ML concepts clearly
- ALWAYS run the eval before claiming success — no qualitative-only assessments
