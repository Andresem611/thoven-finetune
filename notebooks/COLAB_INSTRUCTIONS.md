# Colab Training Instructions (Weekend 2)

## Prerequisites
- `data/processed/training_sft.jsonl` from Weekend 1
- HuggingFace token with write access
- Baseline results from Weekend 1 (kill criterion passed)

## Steps

1. Open Unsloth official Gemma 4 notebook (link TBD when available)
2. Upload `training_sft.jsonl` to Colab session storage
3. Apply config overrides from `configs/gemma4_sft.yaml`:
   - `load_in_4bit=True`
   - `max_seq_length=2048`
   - `fp16=True, bf16=False`
   - `save_steps=100, save_total_limit=3`
4. **Smoke test:** Set `max_steps=1`, run training to verify no crashes
5. **Full SFT run:** ~2-3 hours on T4 with ~1,500 examples
6. Monitor loss curves — should decrease smoothly, watch for overfitting spikes
7. Export to GGUF Q4_K_M for local inference via Ollama
8. Push adapter to HuggingFace Hub: `thoven/gemma4-pedagogy-tutor-v1`

## After Training

```bash
# Import to Ollama for local eval
ollama create thoven-tutor -f Modelfile

# Run Weekend 2 eval (all 4 conditions including C: fine-tuned)
npx promptfoo eval --config eval/promptfoo-pedagogy-full.yaml
```

## Checkpoint Recovery

If Colab session disconnects:
- Checkpoints saved every 100 steps to session storage
- Push to HF Hub in training callback as backup
- Re-mount and resume from latest checkpoint
