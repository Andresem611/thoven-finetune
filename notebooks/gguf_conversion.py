# === GGUF Conversion Notebook ===
# Run in Google Colab with T4 GPU (or Kaggle with T4)
# Requires: GPU quota available (free tier resets daily)
#
# This loads the LoRA adapter from HuggingFace Hub and exports to GGUF.
# The training is already done — this is just a format conversion.

# === Cell 1: Install + Load + Export ===
!pip install unsloth

from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    "AndresMartinezThoven/thoven-tutor-v1-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
print("Model loaded from HuggingFace!")

model.save_pretrained_gguf(
    "thoven-tutor-gguf",
    tokenizer,
    quantization_method="q4_k_m",
)
print("GGUF exported!")

# === Cell 2: Copy to Google Drive ===
import shutil, glob, os
from google.colab import drive
drive.mount("/content/drive")

gguf_files = glob.glob("/content/thoven-tutor-gguf/*.gguf") + glob.glob("/content/*.gguf")
for f in gguf_files:
    dest = f"/content/drive/MyDrive/{os.path.basename(f)}"
    print(f"Copying {f} ({os.path.getsize(f)/1e9:.1f} GB) to Drive...")
    shutil.copy2(f, dest)
print("GGUF on Google Drive! Download to ~/thoven/thoven-finetune/models/")

# === If GGUF export fails (known VLM bug), try manual llama.cpp: ===
# !git clone https://github.com/ggerganov/llama.cpp /content/llama_cpp
# !pip install -r /content/llama_cpp/requirements/requirements-convert_hf_to_gguf.txt
# First merge the model:
# model.save_pretrained_merged("/content/merged", tokenizer, save_method="merged_4bit")
# Then convert:
# !python /content/llama_cpp/convert_hf_to_gguf.py /content/merged --outfile /content/model.gguf --outtype f16
# !cd /content/llama_cpp && make -j llama-quantize
# !/content/llama_cpp/llama-quantize /content/model.gguf /content/thoven-tutor-q4km.gguf Q4_K_M
