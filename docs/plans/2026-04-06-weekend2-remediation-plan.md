# Weekend 2 Remediation + Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 7 critical discrepancies, 3 number corrections, and 5 risks found by the 6-agent verification audit, then execute Weekend 2 (Colab training + 4-condition eval).

**Architecture:** Two-part plan. Part 1 (Remediation) fixes configs, skill docs, Promptfoo setup, and training notebook before touching any GPU. Part 2 (Execution) is the actual Colab training + local eval. Each phase has mandatory skill/subagent gates that MUST be invoked — not optional.

**Tech Stack:** Unsloth (FastModel API), TRL (SFTConfig + SFTTrainer), Promptfoo 0.121.3, Ollama, Google Colab free T4, Gemma 4 E4B (QLoRA 4-bit r=16)

---

## Skill & Agent Audit

### Available Skills (Must Be Invoked as Gates)

| Skill | Where It's Required | Risk if Skipped |
|-------|-------------------|-----------------|
| `/llm-finetuning-music-pedagogy` | Task 1 (skill update), Task 7 (training config) | Outdated guidance causes wrong API calls |
| `/fine-tuning-expert` | Task 7 (training config review), Task 10 (checkpoint strategy) | Hyperparameter misconfiguration |
| `/write-judge-prompt` | Task 5 (judge output format fix) | Judges not parsed → eval scores meaningless |
| `/validate-evaluator` | Task 12 (post-training eval verification) | Judges may have drifted, no TPR/TNR recheck |
| `/eval-audit` | Task 13 (final pipeline audit) | Undetected eval pipeline issues |
| `/edtech-learning-science` | Task 6 (prompt file review) | Pedagogically unsound system prompts |
| `huggingface-skills:hugging-face-model-trainer` | Task 8 (Colab notebook) | Missing HF-specific training patterns |

### Available Agents (Dispatch Points)

| Agent | Where It's Required | Purpose |
|-------|-------------------|---------|
| `learning-science-advisor` | Task 6 (system prompt review) | Validate Socratic prompt is pedagogically sound |
| `code-reviewer` | Task 4, Task 7 (config review) | Catch syntax errors before execution |
| `llm-architect` | Task 1 (skill update) | Validate architecture claims in skill |
| `researcher` | Task 3 (chat template verification) | Verify Gemma 4 Jinja template contents |

---

## File Structure

### Files to Modify (Part 1 — Remediation)

| File | Responsibility | Task |
|------|---------------|------|
| `SKILL.md` (custom skill) | Source of truth for fine-tuning guidance | T1 |
| `configs/gemma4_sft.yaml` | Training hyperparameters | T2 |
| `eval/prompts/neutral.json` | Neutral system prompt (JSON chat format) | T3 |
| `eval/prompts/socratic.json` | Socratic system prompt (JSON chat format) | T3 |
| `eval/promptfoo-pedagogy-w2.yaml` | Weekend 2 eval config (4 conditions) | T4 |
| `notebooks/colab_sft.py` | Complete Colab training script | T5 |
| `notebooks/COLAB_INSTRUCTIONS.md` | Updated instructions | T5 |
| `STATUS.md` | Experiment tracker | T6 |

### Files to Create (Part 2 — Execution)

| File | Responsibility | Task |
|------|---------------|------|
| `eval/results/weekend2_*.json` | 4-condition eval results | T10 |
| `models/Modelfile` | Ollama import config | T9 |
| `docs/plans/2026-04-06-weekend2-remediation-plan.md` | This plan | — |

---

# Part 1: Remediation (Fix Before Training)

## Phase 1: Update Source of Truth

### Task 1: Fix Custom Skill (10 discrepancies)

**Skill gate:** Invoke `/llm-finetuning-music-pedagogy` to load current skill, then dispatch `llm-architect` agent to review architecture claims.

**Files:**
- Modify: `~/.claude/skills/llm-finetuning-music-pedagogy/SKILL.md` (located at `/Users/andresmartinez/thoven/backend-thovie/.claude/skills/llm-finetuning-music-pedagogy/SKILL.md`)

- [ ] **Step 1: Fix total parameter count**

Line ~50, change:
```markdown
# OLD (wrong)
| Architecture | PLE (Parameter-Light Execution), 4.5B effective from 5.1B total |

# NEW (verified)
| Architecture | Dense with PLE (Per-Layer Embeddings), 4.5B effective from 8B total |
```

- [ ] **Step 2: Fix QLoRA contradiction**

Line ~52 and ~399. The skill says "NOT 4-bit QLoRA" in the table then recommends QLoRA in the next paragraph. Remove the contradictory row and update the Common Pitfalls table:

```markdown
# OLD (contradictory)
| Precision | **16-bit (bf16)** -- NOT 4-bit QLoRA |
...
| Using QLoRA (4-bit) | Quantization flattens pedagogical nuance | 16-bit bf16 LoRA, scale compute instead |

# NEW (consistent with our actual approach)
| Precision | **4-bit QLoRA** (start here) or 16-bit bf16 (if quality degrades) |
...
| Skipping smoke test | OOM or format errors waste full training run | Always run max_steps=1 first |
```

- [ ] **Step 3: Add chat template section**

After the Training Configuration code block (~line 82), add:

```markdown
### Chat Template (Gemma 4 — New Format)

**BREAKING CHANGE from Gemma 3:** Gemma 4 uses `<|turn>` / `<turn|>` tokens, NOT `<start_of_turn>` / `<end_of_turn>`.

The chat template is NOT embedded in `tokenizer_config.json`. You must load it manually:

\```python
# REQUIRED — Gemma 4 ships template as separate Jinja file
import requests
jinja = requests.get(
    "https://huggingface.co/google/gemma-4-E4B-it/resolve/main/chat_template.jinja"
).text
tokenizer.chat_template = jinja
\```

For training data formatting, use Unsloth's helper:
\```python
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

tokenizer = get_chat_template(tokenizer, chat_template="gemma-it")
dataset = standardize_sharegpt(dataset)  # converts from/value → role/content
\```
```

- [ ] **Step 4: Add PEFT vision exclusion**

In the Training Configuration code block, add `exclude_modules` to the `get_peft_model` call:

```python
model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    exclude_modules=["vision_tower", "multi_modal_projector", "audio_tower"],
    use_gradient_checkpointing="unsloth",
)
```

- [ ] **Step 5: Fix VRAM numbers**

Update the Key Concepts table and add a VRAM section:

```markdown
### VRAM Requirements (Gemma 4 E4B)

| Use Case | VRAM |
|----------|------|
| Inference, 4-bit | ~5-6 GB |
| QLoRA training (r=16, batch=1) | 8-10 GB |
| QLoRA training + Unsloth optimizations | ~6-8 GB |
| Full bf16 | ~15-16 GB |

**Minimum GPU:** T4 (16GB) for QLoRA. Start with batch_size=1.
**GGUF Q4_K_M file size:** 4.98 GB (not ~3GB — total params are 8B).
```

- [ ] **Step 6: Fix Promptfoo config example**

Lines ~240-264, replace the old string-prompt Promptfoo example with JSON chat format:

```markdown
### Promptfoo Configuration

**CRITICAL:** `config.systemPrompt` is NOT a real Promptfoo field — it is silently ignored by all providers. Use JSON prompt files with `role: system` messages instead.

\```yaml
# promptfoo-pedagogy.yaml
providers:
  - id: ollama:chat:thoven-tutor
    label: "C: Fine-tuned (neutral)"
    prompts: neutral_prompt

prompts:
  file://eval/prompts/neutral.json: neutral_prompt

tests: file://test_cases.yaml
\```

Where `neutral.json` is:
\```json
[
  {"role": "user", "content": "A {{age}}-year-old {{instrument}} student says: {{student_message}}"}
]
\```
```

- [ ] **Step 7: Add SFTConfig note**

In the Training Pipeline Summary or Recommended Stack section:

```markdown
### TRL Configuration (Current API)

Use `SFTConfig` from `trl`, NOT `TrainingArguments` from `transformers`:

\```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./output",
        max_length=2048,          # TRL param (not max_seq_length)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        fp16=True,
        bf16=False,
        optim="adamw_8bit",
        save_steps=100,
        save_total_limit=3,
    ),
    train_dataset=dataset,
)
\```

**Note:** Unsloth uses `max_seq_length` on `FastModel.from_pretrained()`. TRL uses `max_length` on `SFTConfig`. Both must be set and match.
```

- [ ] **Step 8: Update model names for Weekend 1 cost optimization**

Lines ~110-111, update to reflect actual Weekend 1 switch:

```markdown
# OLD
|  Claude Opus 4.6 = Tutor (scaffolding, Socratic questioning)
|  Claude Sonnet 4.5 = Student (age-appropriate responses, realistic mistakes)

# NEW (cost-optimized in Weekend 1)
|  Claude Sonnet 4.6 = Tutor (scaffolding, Socratic questioning)
|  Claude Haiku 4.5 = Student (age-appropriate responses, realistic mistakes)
|  (Original: Opus+Sonnet at ~$55/210 scenarios. Switched to Sonnet+Haiku at ~$2-3/100 scenarios)
```

- [ ] **Step 9: Update changelog**

Add entry at the end:

```markdown
| 2026-04-06 | **Critical fixes from 6-agent verification audit:** total params 8B not 5.1B, QLoRA contradiction removed, Gemma 4 chat template (`<\|turn>` tokens), PEFT exclude_modules, VRAM 8-10GB, Promptfoo JSON prompt files, SFTConfig API, tutor/student model names |
```

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "fix: update custom skill with 10 verified corrections from pre-flight audit"
```

---

## Phase 2: Fix Training Config

### Task 2: Update gemma4_sft.yaml

**Files:**
- Modify: `configs/gemma4_sft.yaml`

- [ ] **Step 1: Update config with verified values**

Replace entire file:

```yaml
# configs/gemma4_sft.yaml (Rev 3 — verified against docs 2026-04-06)
# Verified by: Unsloth docs, TRL v0.29.0, Gemma 4 model card
#
# VRAM budget: ~8-10GB QLoRA training (T4 16GB = minimum)
# Architecture: Dense with PLE, 8B total params, 4.5B effective

model: unsloth/gemma-4-E4B-it-unsloth-bnb-4bit  # Pre-quantized (faster load)
load_in_4bit: true
fp16: true                      # T4 (Turing) does NOT support bf16
bf16: false
max_seq_length: 2048            # Unsloth param — matches TRL max_length

# LoRA config
lora_r: 16
lora_alpha: 16
lora_dropout: 0                 # Unsloth optimized path
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
exclude_modules:                # PEFT bug: Gemma4ClippableLinear in vision layers
  - vision_tower
  - multi_modal_projector
  - audio_tower

# Training
gradient_checkpointing: unsloth # 30% less VRAM than standard
per_device_train_batch_size: 1  # Conservative for T4 (was 2, reduced after VRAM audit)
gradient_accumulation_steps: 8  # Effective batch = 1 * 8 = 8
learning_rate: 2e-4
num_train_epochs: 3
warmup_steps: 5
optimizer: adamw_8bit
weight_decay: 0.01
lr_scheduler_type: cosine
save_steps: 100
save_total_limit: 3

# Checkpoint recovery (Colab session preemption)
push_to_hub: true
hub_strategy: all_checkpoints
hub_model_id: thoven/gemma4-pedagogy-tutor-v1

# GGUF export
gguf_quantization: q4_k_m      # Actual file size: 4.98 GB (not ~3GB)
```

- [ ] **Step 2: Commit**

```bash
git add configs/gemma4_sft.yaml
git commit -m "fix: update training config with verified VRAM, batch size, and checkpoint strategy"
```

---

## Phase 3: Fix Promptfoo Eval Config (ROOT CAUSE of Weekend 1 Bug)

### Task 3: Create JSON Prompt Files

**Skill gate:** Invoke `/edtech-learning-science` to review the Socratic system prompt for pedagogical soundness. Dispatch `learning-science-advisor` agent to validate.

**Files:**
- Create: `eval/prompts/neutral.json`
- Create: `eval/prompts/socratic.json`
- Create: `eval/prompts/bare.json`

- [ ] **Step 1: Create neutral prompt (no system message — user turn only)**

```json
[
  {"role": "user", "content": "You are a music tutor. Help the student.\n\nA {{age}}-year-old {{instrument}} student says: {{student_message}}"}
]
```

Save to `eval/prompts/neutral.json`.

- [ ] **Step 2: Create Socratic prompt (system + user turns)**

```json
[
  {"role": "system", "content": "You are a music tutor for a student. Use Socratic questioning to guide learning.\n\nCORE BEHAVIORS:\n- Use Socratic questioning to scaffold learning\n- Never give answers directly — guide the student to discover\n- Address ONE skill or concept per response\n- When a student raises a new topic, ask what they already know first\n- Praise effort and specific strategies, never talent or natural ability\n- When relevant, reference physical sensations and body awareness\n- Where appropriate, offer choices rather than only directives\n- Check understanding before moving to new material"},
  {"role": "user", "content": "A {{age}}-year-old {{instrument}} student says: {{student_message}}"}
]
```

Save to `eval/prompts/socratic.json`.

- [ ] **Step 3: Create bare prompt (for fine-tuned model — no instructions needed)**

```json
[
  {"role": "user", "content": "A {{age}}-year-old {{instrument}} student says: {{student_message}}"}
]
```

Save to `eval/prompts/bare.json`.

- [ ] **Step 4: Dispatch learning-science-advisor to review Socratic prompt**

Dispatch `learning-science-advisor` agent with:
> "Review this Socratic system prompt for pedagogical soundness. Does it align with Vygotsky's ZPD, Bloom's taxonomy scaffolding, and growth mindset principles? Flag any anti-patterns."

If agent flags issues, fix before proceeding.

- [ ] **Step 5: Commit**

```bash
git add eval/prompts/
git commit -m "feat: add JSON chat-format prompt files (fixes systemPrompt silent ignore bug)"
```

---

### Task 4: Write Weekend 2 Eval Config (4 Conditions)

**Skill gate:** Invoke `/eval-audit` on the new config to catch issues before running.

**Files:**
- Create: `eval/promptfoo-pedagogy-w2.yaml`

- [ ] **Step 1: Write the 4-condition eval config**

```yaml
# eval/promptfoo-pedagogy-w2.yaml
# Weekend 2: 4-condition eval (A, B, C, D)
# Fixed: JSON prompt files instead of config.systemPrompt (which is silently ignored)

description: "Weekend 2 — 4-Condition Pedagogy Eval"

providers:
  # Condition A: Base Gemma 12B + neutral prompt (HF API — same as Weekend 1)
  - id: huggingface:chat:google/gemma-3-12b-it:featherless-ai
    label: "A: Base Gemma 12B (neutral)"
    prompts: neutral_prompt
    config:
      max_new_tokens: 1024
      temperature: 0.7

  # Condition B: Base Gemma 12B + Socratic prompt (Ollama local — fixes Weekend 1 bug)
  - id: ollama:chat:gemma3:12b
    label: "B: Base Gemma 12B (Socratic)"
    prompts: socratic_prompt
    config:
      num_predict: 1024
      temperature: 0.7

  # Condition C: Fine-tuned Gemma 4 E4B + bare prompt (the thing we trained)
  - id: ollama:chat:thoven-tutor
    label: "C: Fine-tuned E4B (neutral)"
    prompts: bare_prompt
    config:
      num_predict: 1024
      temperature: 0.7

  # Condition D: Opus + neutral prompt (ceiling)
  - id: anthropic:messages:claude-opus-4-20250514
    label: "D: Opus 4.6 (ceiling)"
    prompts: neutral_prompt
    config:
      max_tokens: 1024
      temperature: 0.7

prompts:
  file://prompts/neutral.json: neutral_prompt
  file://prompts/socratic.json: socratic_prompt
  file://prompts/bare.json: bare_prompt

tests: file://test_cases.yaml

defaultTest:
  options:
    provider: anthropic:messages:claude-sonnet-4-20250514
  assert:
    # --- Copy all 12 binary judges from promptfoo-pedagogy.yaml ---
    # D1a: Scaffolding Structure
    - type: llm-rubric
      value: |
        You are evaluating a music tutor's response to a student.
        TASK: Determine whether the tutor breaks its response into sequential steps...
        Output ONLY valid JSON: {"critique": "...", "pass": true, "score": 1}
        If FAIL: {"critique": "...", "pass": false, "score": 0}

    # ... (all 12 judges from Weekend 1 config, with output format updated)
    # NOTE TO IMPLEMENTER: Copy all 12 assert blocks from eval/promptfoo-pedagogy.yaml
    # but change the output format in each from:
    #   {"critique": "...", "result": "Pass or Fail"}
    # to:
    #   {"critique": "...", "pass": true, "score": 1}  (for Pass)
    #   {"critique": "...", "pass": false, "score": 0}  (for Fail)
    # This ensures Promptfoo correctly parses the binary verdict.
```

- [ ] **Step 2: Copy all 12 judges from Weekend 1 config**

Read `eval/promptfoo-pedagogy.yaml` and copy all 12 `assert` blocks into the new file. For each judge, update the output instruction from:
```
Output ONLY valid JSON: {"critique": "...", "result": "Pass or Fail"}
```
to:
```
Output ONLY valid JSON: {"critique": "...", "pass": true, "score": 1} for Pass, or {"critique": "...", "pass": false, "score": 0} for Fail.
```

- [ ] **Step 3: Run eval-audit on the new config**

Invoke `/eval-audit` skill against `eval/promptfoo-pedagogy-w2.yaml`. Check for:
- All 12 judges have 4 components (criterion, Pass/Fail defs, few-shots, structured output)
- No Likert scales
- No `config.systemPrompt` usage
- Proper prompt-per-provider assignment

- [ ] **Step 4: Commit**

```bash
git add eval/promptfoo-pedagogy-w2.yaml
git commit -m "feat: Weekend 2 eval config — 4 conditions with JSON prompt files"
```

---

## Phase 4: Write Colab Training Notebook

### Task 5: Create Complete Colab Training Script

**Skill gate:** Invoke `/fine-tuning-expert` to review hyperparameters. Invoke `huggingface-skills:hugging-face-model-trainer` for HF-specific patterns.

**Files:**
- Create: `notebooks/colab_sft.py`
- Modify: `notebooks/COLAB_INSTRUCTIONS.md`

- [ ] **Step 1: Write the complete training script**

```python
#!/usr/bin/env python3
"""
Thoven Weekend 2: SFT Training on Google Colab
Model: Gemma 4 E4B (QLoRA 4-bit, r=16)
Data: 1,354 ShareGPT examples (100 synthetic + 1,250 ConvoLearn)

Run on Google Colab with free T4 GPU.
"""

# === Cell 1: Install dependencies ===
# !pip install unsloth

# === Cell 2: Verify GPU ===
import torch
gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
assert "T4" in gpu_name or "A100" in gpu_name or "L4" in gpu_name, \
    f"Got {gpu_name} — reconnect to get T4/A100/L4"
assert gpu_mem >= 15, f"Only {gpu_mem:.1f}GB VRAM — need 16GB+"

# === Cell 3: Load model ===
from unsloth import FastModel

MAX_SEQ_LENGTH = 2048

model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    full_finetuning=False,
)

# === Cell 4: Load and assign Gemma 4 chat template ===
# CRITICAL: Gemma 4 ships template as separate Jinja file (not in tokenizer_config.json)
import requests
jinja_template = requests.get(
    "https://huggingface.co/google/gemma-4-E4B-it/resolve/main/chat_template.jinja"
).text
tokenizer.chat_template = jinja_template
print(f"Chat template loaded ({len(jinja_template)} chars)")

# === Cell 5: Apply LoRA ===
model = FastModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    exclude_modules=["vision_tower", "multi_modal_projector", "audio_tower"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# === Cell 6: Load training data ===
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

# Upload training_sft.jsonl to Colab first (drag into Files panel)
dataset = load_dataset("json", data_files="/content/training_sft.jsonl", split="train")
print(f"Loaded {len(dataset)} examples")

# Convert ShareGPT from/value → messages/role/content (TRL requirement)
dataset = standardize_sharegpt(dataset)

# Apply Gemma chat template
tokenizer = get_chat_template(tokenizer, chat_template="gemma-it")

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print(f"Formatted. Sample:\n{dataset[0]['text'][:500]}")

# === Cell 7: Configure trainer ===
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="/content/checkpoints",
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_steps=5,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    save_steps=100,
    save_total_limit=3,
    logging_steps=10,
    seed=3407,
    # Checkpoint recovery for Colab preemption
    push_to_hub=True,
    hub_strategy="all_checkpoints",
    hub_model_id="thoven/gemma4-pedagogy-tutor-v1",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
)

# === Cell 8: Smoke test (1 step) ===
print("=== SMOKE TEST (1 step) ===")
trainer.args.max_steps = 1
trainer.train()
print("Smoke test PASSED — no OOM, no format errors")
trainer.args.max_steps = -1  # Reset to use num_train_epochs

# === Cell 9: Full training run ===
print("=== FULL SFT RUN ===")
trainer.train()
print("Training complete!")

# === Cell 10: Save and export ===
# Save adapter
model.save_pretrained("/content/thoven-tutor-adapter")
tokenizer.save_pretrained("/content/thoven-tutor-adapter")

# Export GGUF Q4_K_M (actual size: ~4.98 GB)
model.save_pretrained_gguf(
    "/content/thoven-tutor-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
print("GGUF exported to /content/thoven-tutor-gguf/")

# === Cell 11: Copy to Google Drive (DO NOT use files.download for 5GB) ===
import shutil
from google.colab import drive

drive.mount("/content/drive")
shutil.copytree(
    "/content/thoven-tutor-gguf",
    "/content/drive/MyDrive/thoven-tutor-gguf",
)
print("Copied to Google Drive. Download from Drive, not Colab.")
```

- [ ] **Step 2: Update COLAB_INSTRUCTIONS.md**

Replace entire file with updated instructions referencing the new script, verified GPU requirements, Google Drive download strategy, and checkpoint recovery.

- [ ] **Step 3: Invoke /fine-tuning-expert to review training config**

Dispatch the skill to review:
- LoRA rank vs dataset size ratio
- Learning rate for QLoRA
- Epoch count risk (overfitting at 3 epochs on 1,354 examples?)
- Gradient accumulation math (effective batch = 8)

- [ ] **Step 4: Invoke huggingface-skills:hugging-face-model-trainer**

Verify the `push_to_hub` + `hub_strategy` pattern works for Colab checkpoint recovery.

- [ ] **Step 5: Commit**

```bash
git add notebooks/
git commit -m "feat: complete Colab SFT training script with all verified fixes"
```

---

## Phase 5: Prepare Local Eval Infrastructure

### Task 6: Install Ollama + Create Modelfile

**Files:**
- Create: `models/Modelfile`

- [ ] **Step 1: Install Ollama on Mac**

```bash
brew install ollama
ollama serve  # Start the server (runs in background)
```

- [ ] **Step 2: Pull base Gemma 3 12B for Condition B**

```bash
ollama pull gemma3:12b
```

Verify: `ollama run gemma3:12b "Hello"` should respond.

- [ ] **Step 3: Create Modelfile for fine-tuned model**

```dockerfile
# models/Modelfile
# Import fine-tuned Gemma 4 E4B from GGUF
FROM ./thoven-tutor.gguf
```

Note: The GGUF file path will be filled in after downloading from Google Drive in Part 2.

- [ ] **Step 4: Commit**

```bash
git add models/Modelfile
git commit -m "feat: add Ollama Modelfile for fine-tuned model import"
```

---

## Phase 6: Update Documentation

### Task 7: Update STATUS.md

**Files:**
- Modify: `STATUS.md`

- [ ] **Step 1: Add remediation section to STATUS.md**

Add after the Weekend 1 section:

```markdown
---

## Pre-Flight Verification (2026-04-06)

### 6-Agent Documentation Audit

| Agent | Provider | Critical Findings |
|-------|----------|------------------|
| verify-unsloth | Unsloth docs | FastModel (not FastLanguageModel), standardize_sharegpt(), gemma-it template |
| verify-ollama | Ollama docs | All 8/8 confirmed. apiKey="ollama" for OpenAI-compat endpoint |
| verify-promptfoo | Promptfoo docs | **config.systemPrompt silently ignored** — root cause of Weekend 1 Condition B bug |
| verify-trl | TRL v0.29.0 | SFTConfig (not TrainingArguments), max_length (not max_seq_length) |
| verify-colab | Web search | T4 not guaranteed (might get K80), GGUF download unreliable (use Google Drive) |
| verify-gemma4 | HF model card | Chat template changed to <\|turn>, VRAM 8-10GB, GGUF 4.98GB, E4B is DENSE |

### Remediation Applied

- [x] Custom skill updated (10 corrections)
- [x] Training config revised (batch_size=1, exclude_modules, push_to_hub)
- [x] Promptfoo eval rewritten with JSON prompt files
- [x] Complete Colab training script with all verified APIs
- [x] Ollama infrastructure prepared
```

- [ ] **Step 2: Commit**

```bash
git add STATUS.md
git commit -m "docs: add pre-flight verification results and remediation status"
```

---

# Part 2: Execution (Training + Eval)

## Phase 7: Colab Training

### Task 8: Upload Data + Smoke Test

- [ ] **Step 1: Open Google Colab**

Go to `colab.research.google.com` → New Notebook → Runtime → Change runtime type → T4 GPU.

- [ ] **Step 2: Verify GPU**

Run Cell 2 from `notebooks/colab_sft.py`. Must see "T4" with >=15GB. If K80, disconnect and reconnect.

- [ ] **Step 3: Upload training data**

Drag `data/processed/training_sft.jsonl` into Colab's Files panel (left sidebar).

- [ ] **Step 4: Run Cells 1-8 (install through smoke test)**

Run sequentially. Cell 8 (smoke test with max_steps=1) must complete without OOM.

Expected output: "Smoke test PASSED — no OOM, no format errors"

- [ ] **Step 5: Check HF Hub push works**

After smoke test, verify the checkpoint appeared at `huggingface.co/thoven/gemma4-pedagogy-tutor-v1`.

---

### Task 9: Full SFT Training Run

- [ ] **Step 1: Run Cell 9 (full training)**

Monitor loss curve. Expected:
- Loss should decrease smoothly from ~2.5 to ~0.8-1.2
- No sudden spikes (would indicate data issues)
- ~508 steps total (~2-3 hours on T4)

- [ ] **Step 2: Run Cells 10-11 (export + Drive copy)**

After training completes, export GGUF and copy to Google Drive.

Expected: `thoven-tutor-gguf/` folder in Google Drive (~5GB).

- [ ] **Step 3: Download GGUF from Google Drive to Mac**

Download `thoven-tutor-gguf/` from Google Drive to `~/thoven/thoven-finetune/models/`.

---

## Phase 8: Local Model Setup

### Task 10: Import to Ollama

- [ ] **Step 1: Update Modelfile with actual GGUF path**

```dockerfile
FROM /Users/andresmartinez/thoven/thoven-finetune/models/thoven-tutor-gguf/unsloth.Q4_K_M.gguf
```

- [ ] **Step 2: Create the model in Ollama**

```bash
cd ~/thoven/thoven-finetune
ollama create thoven-tutor -f models/Modelfile
```

- [ ] **Step 3: Sanity check**

```bash
ollama run thoven-tutor "A 7-year-old piano student says: I can't play with both hands at the same time"
```

Verify: Response should show scaffolding behavior (breaking into steps, asking guiding questions).

---

## Phase 9: 4-Condition Eval

### Task 11: Run Full Eval

**Skill gate:** Invoke `/validate-evaluator` to spot-check judges still work. Invoke `/eval-audit` on final config.

- [ ] **Step 1: Verify Ollama is running**

```bash
ollama list  # Should show thoven-tutor and gemma3:12b
```

- [ ] **Step 2: Run the eval**

```bash
cd ~/thoven/thoven-finetune
npx promptfoo eval --config eval/promptfoo-pedagogy-w2.yaml -o eval/results/weekend2_$(date +%Y-%m-%d).json
```

Expected: ~20-30 minutes for 23 test cases × 4 providers × 12 judges.

- [ ] **Step 3: View results**

```bash
npx promptfoo view
```

Opens browser with results comparison table.

- [ ] **Step 4: Extract scores per condition per dimension**

Run the kill criterion / scoring script adapted for 4 conditions. Record:
- Condition A weighted mean
- Condition B weighted mean (first real B score!)
- Condition C weighted mean (the trained model!)
- Condition D weighted mean

---

### Task 12: Decision Gate

**Skill gate:** Invoke `/edtech-learning-science` to interpret pedagogical implications of score changes.

- [ ] **Step 1: Compute success criteria**

| Metric | Fail | Pass | Great |
|--------|------|------|-------|
| C > B (fine-tuned+neutral > base+Socratic) | C < B on scaffolding | C > B on >4 dims | C > B on >7 dims |
| Dimensions improved (C vs A) | < 5 of 11 | > 7 of 11 | > 9 of 11 |
| No regression | Any dimension drops >0.1 | All hold or improve | All improve |
| Opus gap narrows (C vs D) | Gap unchanged | Gap narrows >10% | Gap narrows >25% |

- [ ] **Step 2: Record decision**

Update STATUS.md with results table and decision: PROCEED to DPO / ADJUST data / STOP.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: Weekend 2 complete — 4-condition eval results"
```

---

## Skill Invocation Map (Anti-Skip Reference)

This table lists every phase where a skill MUST be invoked. If executing inline or via subagent, the skill check happens BEFORE the phase's code work begins.

| Phase | Skill | What It Checks | Skip Risk |
|-------|-------|---------------|-----------|
| Phase 1 (Task 1) | `/llm-finetuning-music-pedagogy` | Load current skill to see what needs fixing | Edit wrong file |
| Phase 1 (Task 1) | `llm-architect` agent | Architecture claims in skill are valid | Wrong VRAM/param numbers |
| Phase 3 (Task 3) | `/edtech-learning-science` | Socratic prompt pedagogically sound | Bad system prompt |
| Phase 3 (Task 3) | `learning-science-advisor` agent | Validate prompt against learning science | Anti-patterns in prompt |
| Phase 4 (Task 4) | `/eval-audit` | New eval config has no P0 issues | Broken eval pipeline |
| Phase 5 (Task 5) | `/fine-tuning-expert` | Training hyperparameters validated | Overfitting, wrong LR |
| Phase 5 (Task 5) | `huggingface-skills:hugging-face-model-trainer` | HF push_to_hub pattern correct | Lost checkpoints on Colab crash |
| Phase 9 (Task 11) | `/validate-evaluator` | Judges still calibrated | Scores meaningless |
| Phase 9 (Task 11) | `/eval-audit` | Final pipeline audit | Undetected issues |
| Phase 9 (Task 12) | `/edtech-learning-science` | Interpret score changes pedagogically | Misread results |

---

## Self-Review Checklist

1. **Spec coverage:** All 7 critical issues (C1-C7) have corresponding tasks. All 3 number corrections (N1-N3) addressed. All 5 risks (R1-R5) mitigated.
2. **Placeholder scan:** No TBD/TODO. All code blocks are complete. Task 4 Step 2 says "copy all 12 judges" — the implementer must read the source file but the instruction is actionable.
3. **Type consistency:** `FastModel` used consistently (not `FastLanguageModel`). `SFTConfig` used (not `TrainingArguments`). `max_length` on SFTConfig, `max_seq_length` on FastModel.from_pretrained. JSON prompt files referenced consistently across Tasks 3 and 4.
