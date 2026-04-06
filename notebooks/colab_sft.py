"""
Thovie Fine-Tuning: Gemma 4 E4B QLoRA SFT
==========================================
Model  : unsloth/gemma-4-E4B-it-unsloth-bnb-4bit
Data   : /content/training_sft.jsonl  (1,354 ShareGPT examples)
Target : thoven/gemma4-pedagogy-tutor-v1
GPU    : T4 (free Colab tier), also validated on A100 / L4

Each "# === Cell N ===" block is one Colab cell.
Copy-paste cells top-to-bottom. Run the smoke test (Cell 6) before Cell 7.
"""

# === Cell 1: Install dependencies ===
# Runtime: ~3-5 min on first install, ~30 s on cached runtimes
# Restart runtime after this cell if prompted.

import subprocess, sys

def _run(cmd: str) -> None:
    result = subprocess.run(cmd, shell=True, check=True,
                            capture_output=False, text=True)

_run(
    "pip install --quiet "
    "'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git' "
    "unsloth_zoo "
    "trl>=0.15.0 "
    "peft>=0.14.0 "
    "bitsandbytes>=0.43.0 "
    "datasets "
    "huggingface_hub "
    "accelerate "
    "xformers "
    "triton"
)

print("Dependencies installed. If prompted to restart runtime, do so now.")


# === Cell 2: GPU verification ===
# Rejects K80 (too slow / too little VRAM for 4-bit Gemma 4).

import subprocess

def _get_gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        return out
    except Exception:
        return "NO_GPU"

gpu_name = _get_gpu_name()
print(f"Detected GPU: {gpu_name}")

SUPPORTED = ("T4", "A100", "L4", "A10", "V100")
BLOCKED    = ("K80",)

if any(b in gpu_name for b in BLOCKED):
    raise RuntimeError(
        f"GPU '{gpu_name}' is not supported. "
        "K80 has insufficient VRAM and compute for 4-bit Gemma 4. "
        "Upgrade runtime to at least T4."
    )

if not any(s in gpu_name for s in SUPPORTED):
    print(
        f"WARNING: GPU '{gpu_name}' is unrecognised. "
        "Proceeding anyway — monitor VRAM carefully."
    )
else:
    print(f"GPU check passed: {gpu_name}")


# === Cell 3: Authentication (HuggingFace Hub) ===
# Paste your HF token with WRITE access when prompted.
# The token must be authorised for the `thoven` org.

from huggingface_hub import login

# Option A: Interactive prompt (recommended for free Colab)
login()

# Option B: Colab Secrets (uncomment if you use Colab Pro secrets)
# from google.colab import userdata
# login(token=userdata.get("HF_TOKEN"))

print("HuggingFace authentication complete.")


# === Cell 4: Load model and tokenizer ===
# Uses Unsloth's FastModel (required for Gemma 4 — FastLanguageModel does NOT
# support multimodal architectures with Gemma4ClippableLinear layers).

from unsloth import FastModel
import torch

MAX_SEQ_LENGTH = 2048   # Context window during training
DTYPE          = None   # Auto-detect: bf16 on A100/L4, fp16 on T4
LOAD_IN_4BIT   = True   # QLoRA — keeps peak VRAM ≤ 10 GB on T4

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

print(f"Loading model: {MODEL_ID}")
print(f"max_seq_length={MAX_SEQ_LENGTH}, dtype={DTYPE}, 4bit={LOAD_IN_4BIT}")

model, tokenizer = FastModel.from_pretrained(
    model_name       = MODEL_ID,
    max_seq_length   = MAX_SEQ_LENGTH,
    dtype            = DTYPE,
    load_in_4bit     = LOAD_IN_4BIT,
)

print("Model loaded successfully.")
print(f"Model dtype: {model.dtype}")
print(f"Trainable params before PEFT: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


# === Cell 5: Apply LoRA adapter ===
# exclude_modules is REQUIRED: PEFT has a known bug with Gemma4ClippableLinear
# (vision_tower, multi_modal_projector, audio_tower).  Omitting this causes
# "Expected all tensors to be on the same device" errors during adapter init.

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,   # Text-only fine-tune
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,

    r           = 16,
    lora_alpha  = 16,
    lora_dropout= 0,
    bias        = "none",

    # Exclude multimodal projection layers — PEFT bug workaround
    exclude_modules = [
        "vision_tower",
        "multi_modal_projector",
        "audio_tower",
    ],

    use_gradient_checkpointing = "unsloth",  # Unsloth's custom implementation
    random_state               = 42,
    use_rslora                 = False,
    loftq_config               = None,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,}  ({100 * trainable / total:.2f}%)")


# === Cell 6: Load chat template (Gemma 4 requires manual fetch) ===
# Gemma 4's chat template is NOT embedded in tokenizer_config.json.
# It ships as a separate Jinja file in the model repo.  We fetch it directly
# so TRL's SFTTrainer can apply the correct prompt structure.

import requests
from transformers import AutoTokenizer

TEMPLATE_URL = (
    "https://huggingface.co/google/gemma-4-it/resolve/main/"
    "tokenizer/chat_template.jinja"
)

print(f"Fetching Gemma 4 chat template from: {TEMPLATE_URL}")

response = requests.get(TEMPLATE_URL, timeout=30)
response.raise_for_status()
chat_template_str = response.text

tokenizer.chat_template = chat_template_str
print("Chat template applied to tokenizer.")

# Quick sanity-check: verify the template renders without error
_test_msg = [{"role": "user", "content": "Hello"}]
_rendered  = tokenizer.apply_chat_template(
    _test_msg, tokenize=False, add_generation_prompt=True
)
print(f"Template smoke test passed. Sample output:\n{_rendered[:120]!r}")


# === Cell 7: Load and prepare dataset ===
# training_sft.jsonl uses ShareGPT format:
#   {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
#
# TRL's SFTTrainer only reads the OpenAI messages format:
#   {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
#
# standardize_sharegpt() performs the conversion automatically.

from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt

DATASET_PATH = "/content/training_sft.jsonl"

print(f"Loading dataset from: {DATASET_PATH}")
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
print(f"Raw dataset size: {len(raw_dataset)} examples")
print(f"Columns: {raw_dataset.column_names}")

# Convert ShareGPT → OpenAI messages format
dataset = standardize_sharegpt(raw_dataset)
print(f"Dataset after standardize_sharegpt: {len(dataset)} examples")
print(f"Columns after standardization: {dataset.column_names}")

# Inspect one example to confirm format
sample = dataset[0]
print("\nSample conversation (first example):")
for turn in sample["messages"][:3]:
    role    = turn.get("role", "?")
    content = turn.get("content", "")[:120]
    print(f"  [{role}] {content!r}{'...' if len(turn.get('content','')) > 120 else ''}")


# === Cell 8: Smoke test — 1 step (run BEFORE full training) ===
# Verifies the full pipeline (model + data + trainer) without committing hours.
# Expected: loss prints, no CUDA OOM, no shape errors.
# If this fails, debug before running Cell 9.

from trl import SFTTrainer, SFTConfig

print("=" * 60)
print("SMOKE TEST: max_steps=1")
print("=" * 60)

smoke_trainer = SFTTrainer(
    model     = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        output_dir          = "/content/smoke_test_output",
        max_steps           = 1,
        per_device_train_batch_size     = 1,
        gradient_accumulation_steps     = 1,
        learning_rate       = 1e-4,          # fine-tuning-expert: 2e-4 is QLoRA ceiling; 1e-4 safer
        fp16                = not torch.cuda.is_bf16_supported(),
        bf16                = torch.cuda.is_bf16_supported(),
        logging_steps       = 1,
        optim               = "adamw_8bit",
        # max_length is the SFTConfig parameter (NOT max_seq_length)
        max_length          = MAX_SEQ_LENGTH,
        report_to           = "none",
    ),
)

smoke_stats = smoke_trainer.train()
print(f"Smoke test PASSED. Loss at step 1: {smoke_stats.training_loss:.4f}")
print("Proceeding to full training is safe.\n")

# Free smoke trainer before full run
del smoke_trainer
import gc
gc.collect()
torch.cuda.empty_cache()


# === Cell 9: Full SFT training ===
# ~2-3 hours on T4 with 1,354 examples x 3 epochs.
# Checkpoints saved to /content/gemma4-pedagogy-tutor-v1/ every 100 steps.
# HF Hub push on every checkpoint for crash recovery.
#
# VRAM budget (T4, 16 GB):
#   4-bit weights  : ~6.0 GB
#   LoRA adapters  : ~0.3 GB
#   Activations    : ~2.5 GB  (batch=1, grad_accum=8, seq=2048)
#   Optimizer      : ~0.8 GB  (adamw_8bit)
#   Buffer         : ~1.4 GB
#   Total estimate : ~11 GB   ← safe margin on T4

import math

OUTPUT_DIR    = "/content/gemma4-pedagogy-tutor-v1"
HUB_MODEL_ID  = "thoven/gemma4-pedagogy-tutor-v1"
NUM_EPOCHS    = 3

# Compute total steps for reference
steps_per_epoch = math.ceil(len(dataset) / (1 * 8))  # batch=1, grad_accum=8
total_steps     = steps_per_epoch * NUM_EPOCHS
print(f"Dataset size      : {len(dataset)} examples")
print(f"Steps per epoch   : {steps_per_epoch}")
print(f"Total steps       : {total_steps}")
print(f"Output dir        : {OUTPUT_DIR}")
print(f"Hub model id      : {HUB_MODEL_ID}")
print()

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        # --- Output & Hub ---
        output_dir          = OUTPUT_DIR,
        hub_model_id        = HUB_MODEL_ID,
        push_to_hub         = True,
        hub_strategy        = "all_checkpoints",  # Push every checkpoint (crash recovery)

        # --- Schedule ---
        num_train_epochs    = NUM_EPOCHS,
        warmup_ratio        = 0.03,          # fine-tuning-expert: ~15 steps (not 5)
        lr_scheduler_type   = "cosine",
        learning_rate       = 1e-4,          # fine-tuning-expert: 2e-4 is QLoRA ceiling; 1e-4 safer

        # --- Batch & Memory ---
        per_device_train_batch_size  = 1,
        gradient_accumulation_steps  = 8,

        # --- Precision ---
        fp16                = not torch.cuda.is_bf16_supported(),
        bf16                = torch.cuda.is_bf16_supported(),

        # --- Optimizer ---
        optim               = "adamw_8bit",
        weight_decay        = 0.01,

        # --- Checkpointing ---
        save_steps          = 100,
        save_total_limit    = 3,

        # --- Logging ---
        logging_steps       = 10,
        report_to           = "none",     # Set to "wandb" if you have W&B

        # --- Sequence length ---
        # NOTE: SFTConfig uses max_length, NOT max_seq_length
        max_length          = MAX_SEQ_LENGTH,

        # --- Reproducibility ---
        seed                = 42,
    ),
)

print("Starting full SFT run...")
train_result = trainer.train()

print("\nTraining complete.")
print(f"Final loss          : {train_result.training_loss:.4f}")
print(f"Total steps trained : {train_result.global_step}")
print(f"Runtime             : {train_result.metrics.get('train_runtime', 0):.0f}s  "
      f"({train_result.metrics.get('train_runtime', 0)/3600:.2f}h)")


# === Cell 10: Save final adapter locally and push to HF Hub ===

print("Saving final LoRA adapter to disk...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to: {OUTPUT_DIR}")

print(f"Pushing final model to Hub: {HUB_MODEL_ID}")
model.push_to_hub(HUB_MODEL_ID, tokenizer=tokenizer)
print("Hub push complete.")


# === Cell 11: Export GGUF (Q4_K_M) ===
# Saves a quantised GGUF to /content/gemma4-pedagogy-tutor-v1-gguf/
# then copies it to Google Drive for persistent storage.
#
# We use Google Drive, NOT files.download(), because:
#   - GGUF files are 4-5 GB
#   - files.download() times out / silently fails on large files in Colab
#   - Drive persists across sessions and disconnect events

import os
from pathlib import Path

GGUF_OUTPUT_DIR = "/content/gemma4-pedagogy-tutor-v1-gguf"
GGUF_QUANT      = "q4_k_m"

print(f"Exporting GGUF ({GGUF_QUANT.upper()}) to: {GGUF_OUTPUT_DIR}")

model.save_pretrained_gguf(
    GGUF_OUTPUT_DIR,
    tokenizer,
    quantization_method = GGUF_QUANT,
)

# List produced files
gguf_files = sorted(Path(GGUF_OUTPUT_DIR).glob("*.gguf"))
print(f"\nGGUF files produced:")
for f in gguf_files:
    size_gb = f.stat().st_size / 1e9
    print(f"  {f.name}  ({size_gb:.2f} GB)")


# === Cell 12: Copy GGUF to Google Drive ===
# Mount Drive once, then copy.  If Drive is already mounted, the mount call
# is a no-op.

from google.colab import drive
import shutil

DRIVE_MOUNT  = "/content/drive"
DRIVE_TARGET = f"{DRIVE_MOUNT}/MyDrive/thovie-training/gguf"

print("Mounting Google Drive...")
drive.mount(DRIVE_MOUNT, force_remount=False)

os.makedirs(DRIVE_TARGET, exist_ok=True)
print(f"Drive target directory: {DRIVE_TARGET}")

for gguf_file in sorted(Path(GGUF_OUTPUT_DIR).glob("*.gguf")):
    dest = Path(DRIVE_TARGET) / gguf_file.name
    print(f"Copying {gguf_file.name} → Drive... ", end="", flush=True)
    shutil.copy2(str(gguf_file), str(dest))
    size_gb = dest.stat().st_size / 1e9
    print(f"done ({size_gb:.2f} GB)")

print("\nAll GGUF files safely written to Google Drive.")
print(f"Location: {DRIVE_TARGET}")
print("\nYou can now safely close Colab — files are in Drive.")


# === Cell 13: (Optional) Push GGUF to HF Hub ===
# Pushes the GGUF as a separate Hub repository so you can pull it directly
# into Ollama on any machine with `ollama pull`.

HUB_GGUF_ID = f"{HUB_MODEL_ID}-GGUF"

print(f"Pushing GGUF to Hub: {HUB_GGUF_ID}")

model.push_to_hub_gguf(
    HUB_GGUF_ID,
    tokenizer,
    quantization_method = GGUF_QUANT,
)

print(f"GGUF Hub push complete: https://huggingface.co/{HUB_GGUF_ID}")


# === Cell 14: Final summary ===

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"Base model    : {MODEL_ID}")
print(f"LoRA adapter  : https://huggingface.co/{HUB_MODEL_ID}")
print(f"GGUF (Q4_K_M) : https://huggingface.co/{HUB_GGUF_ID}")
print(f"Drive backup  : {DRIVE_TARGET}")
print()
print("Next steps (Weekend 2 eval):")
print("  1. Pull GGUF to local machine:")
print(f"       ollama pull hf.co/{HUB_GGUF_ID}")
print("  2. Create Modelfile and register with Ollama:")
print("       ollama create thoven-tutor -f Modelfile")
print("  3. Run promptfoo eval (all 4 conditions including C: fine-tuned):")
print("       npx promptfoo eval --config eval/promptfoo-pedagogy-full.yaml")
print()
print("Fine-tuning dimensions to re-score:")
print("  A=Socratic questioning  D=Diagnosis accuracy  E=Explanation quality")
print("  (Baseline: A=0.605, D=0.639, E=0.478)")
