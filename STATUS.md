# Fine-Tuning Experiment Status

**Last updated:** 2026-04-07 21:00 UTC
**Design doc:** `docs/plans/2026-04-05-finetuning-experiment-design.md` (Rev 2)
**Paper outline:** `docs/paper_outline.md`

---

## Weekend 1: Baseline + Data Generation

### Baseline Eval Results

| Condition | Model | Prompt | Weighted Mean | Status |
|-----------|-------|--------|:---:|--------|
| **A** | Gemma 3 12B | Neutral | **0.605** | Floor |
| **B** | Gemma 3 12B | Socratic | 0.605* | *systemPrompt not delivered via HF API |
| **C** | Fine-tuned Gemma 4 E4B | Bare (no system) | **0.231** | ❌ REGRESSION — teacher-guide register |
| **D** | Opus 4.6 | Neutral | **0.639** | Fair ceiling |
| **E** | Opus 4.6 | Socratic | **0.478** | Socratic HURT scores |

**Kill criterion:** ✅ PROCEED — Mean(D1+D5+D6) = 0.591 < 0.750

### Per-Dimension Baseline (Gemma A vs Opus D)

| # | Dimension | Gemma (A) | Opus (D) | Gap | Training Priority |
|---|-----------|:---------:|:--------:|:---:|:-:|
| D1a | Scaffolding (steps) | 0.739 | 0.870 | +0.131 | Medium |
| D2 | Comprehension check | 0.261 | 0.304 | +0.043 | **HIGH** |
| D3 | Age-appropriate | 0.674 | 1.000 | +0.326 | **HIGH** |
| D4 | Cognitive load | 0.130 | 0.217 | +0.087 | **CRITICAL** |
| D5 | Prior knowledge | 0.196 | 0.196 | 0.000 | **CRITICAL** |
| D6 | Growth mindset | 0.839 | 0.652 | -0.187 | Low (Gemma > Opus) |
| D7 | Cognitive level | 0.783 | 0.565 | -0.218 | Low (Gemma > Opus) |
| D8 | Deliberate practice | 0.630 | 0.587 | -0.043 | Medium |
| D9 | Motor learning | 0.557 | 0.435 | -0.122 | Medium |
| D10 | Student autonomy | 0.891 | 1.000 | +0.109 | Low |
| D11 | Instrument relevance | 1.000 | 1.000 | 0.000 | None |

### Key Findings

1. **D4 Cognitive Load (0.130) and D5 Prior Knowledge (0.196) are critical gaps** — both models dump info without managing load or probing what students know
2. **Socratic prompt hurts Opus** (0.478 vs 0.639) — pure questioning suppresses step-sequencing and specific practice guidance
3. **HF Inference API deprecated** — old endpoint returns error, must use `router.huggingface.co` via inference providers. Gemma 3 4B has no providers; using 12B via featherless-ai
4. **Promptfoo systemPrompt not passed to HF chat provider** — Conditions A and B scored identically. B comparison deferred to Weekend 2 (local Ollama)
5. **Expert labeling confirms D5 is #1 gap** — 34/40 responses FAIL prior knowledge probing across both models

---

## Quality Gates (Skills-First Pipeline)

| Phase | Skill Used | Status | Key Metric |
|-------|-----------|--------|------------|
| Eval audit | `/evals-skills:eval-audit` | ✅ Done | 2 P0 issues found (Likert scale, unvalidated judges) |
| Rubric rewrite | `/evals-skills:write-judge-prompt` | ✅ Done | 12 binary Pass/Fail judges, D11 code-based |
| Expert labeling | `learning-science-advisor` agent | ✅ Done | 40 responses labeled on 5 dimensions |
| Judge validation | `/evals-skills:validate-evaluator` | ✅ Done | D2: TPR=0.70/TNR=0.90, D4: TPR=0.50/TNR=1.0 |
| Data quality gate | `fine-tuning-expert` dataset-prep | ✅ Done | 100% format valid, 0 leakage, quality spot-check passed |
| Data generation | Custom two-LLM script | ⏳ Running | 55/100 synthetic + 1,250 ConvoLearn parsed |
| NeMo Data Designer | Researcher agent evaluation | ❌ Rejected | No two-LLM support, wrong architecture for our use case |

---

## Training Data

| Source | Count | Format | Status |
|--------|------:|--------|--------|
| Synthetic (Sonnet tutor + Haiku student) | ~100 target | ShareGPT JSONL | ⏳ Generating (55/100) |
| ConvoLearn (earth science tutoring) | 1,250 | ShareGPT JSONL | ✅ Parsed |
| **Total (after merge + filter)** | **~1,350** | ShareGPT JSONL | ⏳ Pending re-merge |

### Training Data Quality

| Check | Result |
|-------|--------|
| ShareGPT format valid | 100% (36/36 sample) |
| System prompt leakage | 0 detected |
| Empty turns | 0 |
| Consecutive same-role | 0 |
| Token length median | 640 tokens (well under 2048) |
| >2048 tokens | 21/1299 (1.6%) — will be split or truncated |

---

## Cost Tracking

| Item | Estimated | Actual | Notes |
|------|-----------|--------|-------|
| Baseline eval (A+B+D, 3 runs) | $8-11 | ~$15 | 3 runs due to config fixes |
| Opus+Socratic (E) | $3-4 | ~$4 | |
| Judge validation (D2+D4) | $2-3 | ~$3 | 3 iterations each |
| Synthetic generation (100 dialogues) | $5 | ~$3 (so far) | Sonnet+Haiku, 5 turns |
| **Total Weekend 1** | **$20-26** | **~$25** | Within budget |

### Weekend 2 Costs

| Item | Estimated | Actual | Notes |
|------|-----------|--------|-------|
| Pre-flight 6-agent doc audit | $2-3 | ~$2 | Context7 research agents |
| 10-agent verification | $3-5 | ~$4 | Parallel verification |
| Judge labeling (2 agents) | $1-2 | ~$2 | 40 responses × 6 new dims |
| Judge validation (11 dims) | $3-5 | ~$3 | Sonnet API calls |
| Colab training (GPU) | $0 | **$0** | Free T4 tier |
| Judge iteration (7 judges) | $2-3 | ~$2 (partial) | 4 judges iterated so far |
| Condition C eval (253 judge calls) | $2-3 | ~$3 | 23 test cases × 11 judges |
| **Total Weekend 2** | **$15-22** | **~$16** | Within budget |

---

## Pre-Flight Verification (2026-04-06)

### 6-Agent Documentation Audit

| Agent | Provider | Critical Findings |
|-------|----------|------------------|
| verify-unsloth | Unsloth docs | `FastModel` (not `FastLanguageModel`), `standardize_sharegpt()`, `gemma-it` template |
| verify-ollama | Ollama docs | All 8/8 confirmed. `apiKey="ollama"` for OpenAI-compat endpoint |
| verify-promptfoo | Promptfoo docs | **`config.systemPrompt` silently ignored** — root cause of Weekend 1 Condition B bug |
| verify-trl | TRL v0.29.0 | `SFTConfig` (not `TrainingArguments`), `max_length` (not `max_seq_length`) |
| verify-colab | Web search | T4 not guaranteed (might get K80), GGUF download unreliable (use Google Drive) |
| verify-gemma4 | HF model card | Chat template changed to `<\|turn>`, VRAM 8-10GB, GGUF 4.98GB, E4B is DENSE |

### Remediation Applied

- [x] Custom skill updated (10 corrections)
- [x] Training config revised (batch_size=1, exclude_modules, push_to_hub)
- [x] Promptfoo eval rewritten with JSON prompt files
- [x] Complete Colab training script with all verified APIs
- [x] Ollama Modelfile prepared
- [x] STATUS.md updated

---

## Weekend 2: Training + Evaluation

### Prerequisites (from Weekend 1 + Remediation)
- [x] Baseline scores recorded
- [x] Kill criterion passed
- [x] training_sft.jsonl finalized (1,354 examples)
- [x] Training config written (configs/gemma4_sft.yaml Rev 3)
- [x] Colab script written (notebooks/colab_sft.py)
- [x] JSON prompt files created (eval/prompts/)
- [x] Weekend 2 eval config written (eval/promptfoo-pedagogy-w2.yaml)
- [ ] Ollama installed on Mac
- [x] Colab training completed (2026-04-06, 510 steps, final loss 1.44, peak GPU 12.89GB)
- [x] LoRA adapter saved to HuggingFace Hub (AndresMartinezThoven/thoven-tutor-v1-lora)
- [x] LoRA adapter backed up to Google Drive
- [ ] GGUF conversion (blocked: Colab GPU limit + Unsloth VLM bug, retry tomorrow)
- [ ] GGUF imported to Ollama

### Training Results (2026-04-06)

| Metric | Value |
|--------|-------|
| Final loss | 1.437 (step 510) |
| Best loss | 1.345 (step 430) |
| Total steps | 510 (3 epochs × 170 steps) |
| Training time | ~2.25 hours |
| Peak GPU memory | 12.89 GB / 15.6 GB (82.6% utilization) |
| Trainable params | 42,401,792 / 8,038,558,240 (0.53%) |
| Framework | Unsloth 2026.4.4 + TRL 0.24.0 |
| GGUF export | Q4_K_M (~5 GB) |

**Loss curve:** 10.1 (step 10) → 1.9 (step 90) → 1.5 (step 200) → 1.35 (step 430) → 1.44 (step 510)

**Qualitative observation:** Initial test output shows structured response with scaffolding intent but reads as a teacher guide rather than direct student conversation. Formal eval with 11 judges needed to quantify.

---

### Weekend 2 Plan (Updated)
1. ~~Open Google Colab → T4 GPU → verify with nvidia-smi~~ DONE
2. ~~Upload training_sft.jsonl to Colab~~ DONE
3. ~~Run notebooks/colab_sft.py cells 1-8 (smoke test)~~ DONE
4. ~~Run cell 9 (full SFT, ~2-3 hrs)~~ DONE (510 steps, 2.25 hrs)
5. ~~Run cells 10-12 (export GGUF → Google Drive)~~ IN PROGRESS
6. Download GGUF from Drive to Mac (~4.98 GB)
7. `ollama create thoven-tutor -f models/Modelfile`
8. Run 4-condition eval: `npx promptfoo eval --config eval/promptfoo-pedagogy-w2.yaml`
9. Decision gate: proceed to DPO / adjust data / stop

### Condition C — Fine-Tuned Model Eval Results (2026-04-07)

| # | Dimension | Gemma (A) | Fine-Tuned (C) | Opus (D) | C vs A |
|---|-----------|:---------:|:-------------:|:--------:|:------:|
| D1a | Scaffolding | 0.739 | 0.478 | 0.870 | -0.261 ↓ |
| D1b | Questions | 0.478 | 0.043 | 0.609 | -0.435 ↓↓ |
| D2 | Comprehension | 0.261 | 0.000 | 0.304 | -0.261 ↓↓ |
| D3 | Age-appropriate | 0.674 | 0.087 | 1.000 | -0.587 ↓↓↓ |
| D4 | Cognitive load | 0.130 | 0.087 | 0.217 | -0.043 ↓ |
| D5 | Prior knowledge | 0.196 | 0.130 | 0.196 | -0.066 ↓ |
| **D6** | **Growth mindset** | 0.839 | **0.957** | 0.652 | **+0.118 ↑** |
| D7 | Higher-order | 0.783 | 0.217 | 0.565 | -0.566 ↓↓↓ |
| D8 | Practice | 0.630 | 0.174 | 0.587 | -0.456 ↓↓ |
| **D9** | **Motor awareness** | 0.557 | **0.739** | 0.435 | **+0.182 ↑** |
| D10 | Student choice | 0.891 | 0.000 | 1.000 | -0.891 ↓↓↓ |
| D11 | Instrument | 1.000 | 0.957 | 1.000 | -0.043 = |

**Weighted Mean:** A=0.605 → **C=0.231** → D=0.639 (62% drop)

### Decision Gate — FAIL (0/4 criteria met)

| Criterion | Result | Verdict |
|-----------|--------|---------|
| Dims improved (C > A) | 2/12 | ❌ FAIL |
| No regressions | 10/12 regressed | ❌ FAIL |
| Opus gap narrows | Gap widened -1100% | ❌ FAIL |
| C beats Socratic (E) | 0.231 < 0.478 | ❌ FAIL |

**RECOMMENDATION: STOP. Regenerate training data before next training run.**

### Root Cause Analysis

**Primary failure: Training data register mismatch.** The model learned pedagogy concepts (D6 ↑, D9 ↑) but outputs them as teacher-guide advice ("Here is how you can respond...") rather than direct student conversation. 23/23 responses use "Here is/are" framing.

**Why this happened:**
- ConvoLearn data (1,250 examples) = earth science tutoring, not music-specific, meta-commentary heavy
- Synthetic data (100 examples) = Sonnet playing teacher, Haiku playing student — but Sonnet's "teacher voice" is advisory, not conversational
- No explicit register control in training data generation prompts

**What the model DID learn:**
- Growth mindset language (+11.8%) — consistent positive framing
- Motor/body awareness (+18.2%) — practical physical guidance
- Instrument relevance (95.7%) — stays on-topic

**What collapsed:**
- D10 Student choice (89.1% → 0%) — model gives teacher multiple approach options instead of offering the student choices
- D3 Age-appropriate (67.4% → 8.7%) — writes at adult teacher level, not kid level
- D7 Higher-order thinking (78.3% → 21.7%) — explains concepts instead of prompting discovery
- D1b Questions (47.8% → 4.3%) — talks about what to ask, doesn't actually ask

---

## Repo Structure

```
thoven-finetune/
├── eval/                    # Eval framework
│   ├── promptfoo-pedagogy.yaml   # 12 binary Pass/Fail judges
│   ├── test_cases.yaml           # 23 test scenarios
│   ├── labeling/                 # Expert labels for judge validation
│   ├── validation/               # TPR/TNR results
│   └── results/                  # Baseline JSON files
├── scripts/                 # Pipeline scripts
│   ├── 0_check_kill_criterion.py
│   ├── 1_generate_synthetic.py   # Sonnet tutor + Haiku student
│   ├── 2_format_convolearn.py    # HF dataset → ShareGPT
│   ├── 3_merge_and_filter.py     # Combine + quality filters
│   └── validate_judges.py        # TPR/TNR calibration
├── seeds/                   # Scenario seeds (210 total)
├── configs/                 # Training configs
├── data/                    # Generated data (gitignored)
├── docs/                    # Paper outline
└── notebooks/               # Colab instructions
```
