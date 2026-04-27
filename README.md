# thoven-finetune

**Hypothesis:** Pedagogical reasoning (how to teach) can be encoded into a small model's weights through supervised fine-tuning, while domain knowledge (music theory, repertoire) enters at inference via RAG. A fine-tuned 4–8B model should approach the tutoring quality of a 70B model at a fraction of the inference cost.

**Status:** Wave 1 complete (FAIL — register mismatch). Wave 2 planned with register-corrected data.

**Paper:** "Pedagogy in Weights: Fine-Tuning a Small Language Model for Music Tutoring with Learning-Science-Grounded Evaluation" — `docs/paper_outline.md`

---

## What This Project Is

Thoven is a music education platform for K-12 students. This repo contains the research pipeline to fine-tune a small language model to serve as a direct tutoring agent — one that scaffolds learning, checks comprehension, manages cognitive load, and speaks to a 9-year-old differently than a 15-year-old.

The pipeline is eval-first (backward design): rubrics and test cases were defined before any training data was generated. The evaluation framework covers 11 learning-science dimensions grounded in Vygotsky, Dweck, Sweller, Ericsson, Bloom, and others.

---

## Current State (as of 2026-04-07)

### Wave 1 Results — FAIL

| Condition | Model | Weighted Mean | Delta |
|-----------|-------|:---:|---|
| A (Baseline) | Gemma 3 12B + Neutral | 0.605 | — |
| C (Fine-Tuned) | Gemma 4 E4B + LoRA Wave 1 | 0.231 | **−62%** |
| D (Ceiling) | Claude Opus 4.6 + Neutral | 0.639 | +5.6% |
| E (Ablation) | Claude Opus 4.6 + Socratic | 0.478 | −25% |

**Root cause:** Training data register mismatch. The model learned pedagogical concepts (growth mindset +11.8%, motor awareness +18.2%) but outputs them as teacher-guide advice ("Here is how you can respond...") rather than direct student conversation. 23/23 Wave 1 outputs use "Here is/are" framing.

The training data mixed:
- **ConvoLearn** (1,250 examples of earth science tutoring with meta-commentary) — transfers poorly to direct music instruction
- **Synthetic data** where Claude Sonnet was playing a tutor — Sonnet's "teacher voice" is advisory, not conversational

### Wave 2 Plan

Same eval pipeline, new data. Register-controlled prompts force the tutor LLM to speak directly to the student ("You ARE a tutor talking to a 9-year-old. Use 'you' and 'your'. Never describe what you would say."). Drop ConvoLearn entirely. Age-differentiated data generation for 5-7, 8-10, 11-13, 14-16 age bands.

See `docs/plans/2026-04-07-wave2-finetuning-plan.md` for the full task breakdown.

**Trained LoRA (Wave 1):** `AndresMartinezThoven/thoven-tutor-v1-lora` on HuggingFace Hub

---

## Eval Framework

11 binary Pass/Fail judges, each grounded in a specific learning-science theory. Run via [Promptfoo](https://promptfoo.dev) against 23 test cases (piano, violin, guitar, voice, edge cases; ages 5–16).

| # | Dimension | Theory | Wave 1 Baseline (A) | Wave 1 Fine-Tuned (C) |
|---|-----------|--------|:---:|:---:|
| D1a | Scaffolding | Bloom | 0.739 | 0.478 |
| D1b | Questions | Bloom | 0.478 | 0.043 |
| D2 | Comprehension check | Black & Wiliam | 0.261 | 0.000 |
| D3 | Age-appropriate language | Development | 0.674 | 0.087 |
| D4 | Cognitive load | Sweller | 0.130 | 0.087 |
| D5 | Prior knowledge probe | Ausubel | 0.196 | 0.130 |
| D6 | Growth mindset | Dweck | 0.839 | **0.957 ↑** |
| D7 | Higher-order thinking | Bloom | 0.783 | 0.217 |
| D8 | Deliberate practice | Ericsson | 0.630 | 0.174 |
| D9 | Motor/body awareness | Fitts & Posner | 0.557 | **0.739 ↑** |
| D10 | Student autonomy/choice | Deci & Ryan | 0.891 | 0.000 |
| D11 | Instrument relevance | — | 1.000 | 0.957 |

D6 and D9 improved — the model learned these concepts. The failure was in output register, not model capability.

**Key negative finding:** Socratic prompting without fine-tuning *hurt* Claude Opus by 25% (0.639 → 0.478). Pedagogy requires internalized reasoning in weights, not prompt hacks.

---

## Repo Structure

```
thoven-finetune/
├── scripts/
│   ├── 0_check_kill_criterion.py     # Runs decision gate before training
│   ├── 1_generate_synthetic.py        # Two-LLM data gen: Sonnet tutor + Haiku student
│   ├── 2_format_convolearn.py         # HuggingFace dataset → ShareGPT JSONL
│   ├── 3_merge_and_filter.py          # Combine sources + quality filters
│   ├── 4_eval_finetuned.py            # Run model against 23 test cases
│   ├── 5_compare_conditions.py        # Decision gate: fine-tuned vs baseline
│   └── validate_judges.py             # TPR/TNR calibration for each judge
│
├── eval/
│   ├── promptfoo-pedagogy-w2.yaml     # Main eval config (Wave 2)
│   ├── test_cases.yaml                # 23 test scenarios
│   ├── prompts/                       # Bare / neutral / socratic system prompts
│   ├── labeling/                      # Expert labels used to validate judges
│   ├── validation/                    # TPR/TNR results per dimension
│   └── results/                       # Baseline + fine-tuned eval JSON outputs
│
├── data/
│   ├── raw/convolearn_sharegpt.jsonl  # ConvoLearn earth science (1,250 examples)
│   ├── generated/synthetic_dialogues.jsonl  # Wave 1 synthetic music tutoring
│   └── processed/training_sft.jsonl  # Final merged training set (1,354 examples)
│
├── seeds/
│   ├── nafme_scenarios.json           # 50 NAfME-aligned music scenarios
│   ├── gap_scenarios.json             # 160 gap-targeted scenarios (D2/D4/D5/D10)
│   └── reference_students.json        # Student profiles (age, instrument, level)
│
├── configs/
│   └── gemma4_sft.yaml               # QLoRA training config (Wave 1)
│
├── notebooks/
│   ├── colab_sft.py                   # Full Colab training script (Unsloth + TRL)
│   ├── gguf_conversion.py             # GGUF export (blocked for E4B — see STATUS)
│   └── COLAB_INSTRUCTIONS.md
│
├── docs/
│   ├── paper_outline.md               # Full arXiv-style paper outline with results
│   └── plans/                         # Wave 1 remediation + Wave 2 plan
│
├── STATUS.md                          # Single source of truth — start here
└── requirements.txt
```

---

## Quick Start

### 1. Environment

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY and HF_TOKEN

pip install -r requirements.txt
npm install  # for Promptfoo evals
```

### 2. Run the eval against any model

```bash
# Edit eval/promptfoo-pedagogy-w2.yaml to point at your model
npx promptfoo eval --config eval/promptfoo-pedagogy-w2.yaml
npx promptfoo view
```

### 3. Regenerate training data (Wave 2)

```bash
python scripts/1_generate_synthetic.py   # generates data/generated/synthetic_dialogues.jsonl
python scripts/3_merge_and_filter.py     # merges + filters → data/processed/training_sft.jsonl
```

### 4. Train on Google Colab

Open `notebooks/colab_sft.py` in Colab (T4 GPU, free tier). Takes ~2 hours for 200 examples, ~6 hours for 1,300+. The script handles Unsloth setup, training, and HuggingFace Hub push.

### 5. Evaluate fine-tuned model

```bash
python scripts/4_eval_finetuned.py       # runs 23 test cases through the model
python scripts/5_compare_conditions.py   # decision gate: did we beat baseline 0.605?
```

---

## Training Details (Wave 1)

| Param | Value |
|-------|-------|
| Base model | `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` |
| Method | 4-bit QLoRA |
| LoRA rank / alpha | 16 / 16 |
| Learning rate | 1e-4 (cosine decay) |
| Epochs | 3 |
| Training examples | 1,354 |
| Trainable params | 42.4M / 8.04B (0.53%) |
| Hardware | Google Colab T4 (free) |
| Training time | ~2.25 hours |
| Final loss | 1.437 (best: 1.345 at step 430) |
| LoRA adapter | `AndresMartinezThoven/thoven-tutor-v1-lora` |

**Note:** Gemma 4 E4B is a vision-language model — GGUF export is blocked by the VLM architecture. Wave 2 may switch to Gemma 3 4B (text-only, GGUF works) depending on capability testing.

---

## Wave 2 — What to Fix

1. **Register control in data gen** — prompts must force the tutor to speak directly to the student. See `docs/plans/2026-04-07-wave2-finetuning-plan.md` for the exact prompt template.

2. **Drop ConvoLearn** — earth science tutoring patterns did not transfer to direct student conversation. Use only new synthetic music data.

3. **Age-differentiated prompts** — separate generation prompts for 5-7, 8-10, 11-13, 14-16 age bands.

4. **Data quality gate before training** — run a 3-judge subset (D1b, D3, D10) on 20 random samples. Threshold: >80% pass rate. If it fails, fix the prompts before generating the full set.

5. **Model choice** — evaluate Gemma 3 4B vs Gemma 4 E4B vs Gemma 3 1B on capability + GGUF exportability. Decision gate in Phase 1.

---

## Judge Validation Status

All 11 judges validated to TPR ≥ 0.70 and TNR ≥ 0.80 (most ≥ 0.80/0.90). Labels and validation scripts are in `eval/labeling/` and `eval/validation/`. See `scripts/validate_judges.py` to re-validate or iterate.

---

## License

Apache 2.0 — code, model weights, and evaluation framework.

**Contact:** andres@thoven.co
