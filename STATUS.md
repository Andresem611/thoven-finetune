# Fine-Tuning Experiment Status

**Last updated:** 2026-04-05 22:00 UTC
**Design doc:** `docs/plans/2026-04-05-finetuning-experiment-design.md` (Rev 2)
**Paper outline:** `docs/paper_outline.md`

---

## Weekend 1: Baseline + Data Generation

### Baseline Eval Results

| Condition | Model | Prompt | Weighted Mean | Status |
|-----------|-------|--------|:---:|--------|
| **A** | Gemma 3 12B | Neutral | **0.605** | Floor |
| **B** | Gemma 3 12B | Socratic | 0.605* | *systemPrompt not delivered via HF API |
| **C** | Fine-tuned | Neutral | TBD | Weekend 2 |
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
| **Total Weekend 2 (so far)** | **$15-22** | **~$13** | Under budget |

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

Previously:
1. Open Google Colab → T4 GPU → verify with nvidia-smi
2. Upload `training_sft.jsonl` to Colab
3. Run `notebooks/colab_sft.py` cells 1-8 (smoke test)
4. Run cell 9 (full SFT, ~2-3 hrs)
5. Run cells 10-12 (export GGUF → Google Drive)
6. Download GGUF from Drive to Mac (~4.98 GB)
7. `ollama create thoven-tutor -f models/Modelfile`
8. Run 4-condition eval: `npx promptfoo eval --config eval/promptfoo-pedagogy-w2.yaml`
9. Decision gate: proceed to DPO / adjust data / stop

### Success Criteria (Weekend 2)

| Metric | Fail | Pass | Great |
|--------|------|------|-------|
| C > B (fine-tuned+neutral beats base+Socratic) | C < B on scaffolding | C > B on >4 dims | C > B on >7 dims |
| Dimensions improved (C vs A) | < 5 of 11 | > 7 of 11 | > 9 of 11 |
| No regression | Any dimension drops | All hold or improve | All improve |
| Opus gap narrows (C vs D) | Gap unchanged | Gap narrows >10% | Gap narrows >25% |

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
