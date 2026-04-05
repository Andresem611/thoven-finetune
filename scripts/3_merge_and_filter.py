#!/usr/bin/env python3
"""
Script 3: Merge, Filter, and Produce Final Training Set

Merges synthetic dialogues + ConvoLearn data, applies quality filters, and
outputs a clean training set for SFT fine-tuning.

Pipeline steps:
  1. Split long dialogues at turn boundaries (>2048 tokens)
  2. Deduplicate via MD5 hash
  3. Token length statistics
  4. LLM quality filters (Sonnet) -- pedagogical coherence, naturalness, plausibility
  5. System prompt leakage detection
  6. Output final training JSONL with stats

Usage:
    python scripts/3_merge_and_filter.py
    python scripts/3_merge_and_filter.py --skip-llm-filter    # Skip step 4
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SONNET_MODEL = "claude-sonnet-4-20250514"
SONNET_INPUT_COST_PER_M = 3.0
SONNET_OUTPUT_COST_PER_M = 15.0

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_FILE = REPO_ROOT / "data" / "generated" / "synthetic_dialogues.jsonl"
CONVOLEARN_FILE = REPO_ROOT / "data" / "raw" / "convolearn_sharegpt.jsonl"
OUTPUT_DIR = REPO_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "training_sft.jsonl"

MAX_TOKENS = 2048
FILTER_TRUNCATION_CHARS = 2000

# Leakage phrases that should never appear in gpt turns
LEAKAGE_PHRASES = [
    "CORE BEHAVIORS",
    "SOCRATIC QUESTIONING:",
    "NEVER GIVE ANSWERS DIRECTLY:",
    "ONE CONCEPT AT A TIME:",
    "PROBE PRIOR KNOWLEDGE",
    "PRAISE EFFORT NOT TALENT:",
    "REFERENCE PHYSICAL SENSATIONS:",
    "OFFER CHOICES:",
    "CHECK UNDERSTANDING BEFORE MOVING ON:",
    "SECONDARY FOCUS FOR THIS LESSON",
]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_tokenizer = None


def get_tokenizer():
    """Lazy-load tiktoken tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def count_tokens(text: str) -> int:
    """Count tokens in a string using cl100k_base."""
    return len(get_tokenizer().encode(text))


def conversation_token_count(conversation: list[dict[str, str]]) -> int:
    """Count total tokens across all turns in a conversation."""
    total = 0
    for turn in conversation:
        total += count_tokens(turn.get("value", ""))
    return total


# ---------------------------------------------------------------------------
# Step 1: Split long dialogues
# ---------------------------------------------------------------------------


def flatten_conversation(conversation: list[dict[str, str]]) -> str:
    """Flatten a conversation into a single string for hashing/display."""
    parts = []
    for turn in conversation:
        parts.append(f"{turn['from']}: {turn['value']}")
    return "\n".join(parts)


def split_long_dialogue(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a dialogue exceeding MAX_TOKENS at turn boundaries.

    Each chunk:
    - Preserves the system turn
    - Has at least 1 human + 1 gpt turn
    - Stays under MAX_TOKENS
    """
    conversation = entry["conversations"]
    total_tokens = conversation_token_count(conversation)

    if total_tokens <= MAX_TOKENS:
        return [entry]

    # Separate system turn from content turns
    system_turn = None
    content_turns = []
    for turn in conversation:
        if turn["from"] == "system":
            system_turn = turn
        else:
            content_turns.append(turn)

    if not content_turns:
        return [entry]

    system_tokens = count_tokens(system_turn["value"]) if system_turn else 0
    budget = MAX_TOKENS - system_tokens

    chunks = []
    current_chunk: list[dict[str, str]] = []
    current_tokens = 0

    for turn in content_turns:
        turn_tokens = count_tokens(turn.get("value", ""))

        if current_tokens + turn_tokens > budget and current_chunk:
            # Check if current chunk has at least 1 human + 1 gpt
            has_human = any(t["from"] == "human" for t in current_chunk)
            has_gpt = any(t["from"] == "gpt" for t in current_chunk)

            if has_human and has_gpt:
                chunk_conv = ([system_turn] if system_turn else []) + current_chunk
                chunk_entry = {
                    "conversations": chunk_conv,
                    "metadata": {
                        **entry.get("metadata", {}),
                        "is_chunk": True,
                        "chunk_index": len(chunks),
                    },
                }
                chunks.append(chunk_entry)

            current_chunk = []
            current_tokens = 0

        current_chunk.append(turn)
        current_tokens += turn_tokens

    # Handle remaining turns
    if current_chunk:
        has_human = any(t["from"] == "human" for t in current_chunk)
        has_gpt = any(t["from"] == "gpt" for t in current_chunk)

        if has_human and has_gpt:
            chunk_conv = ([system_turn] if system_turn else []) + current_chunk
            chunk_entry = {
                "conversations": chunk_conv,
                "metadata": {
                    **entry.get("metadata", {}),
                    "is_chunk": True,
                    "chunk_index": len(chunks),
                },
            }
            chunks.append(chunk_entry)
        elif chunks:
            # Not enough turns for standalone chunk -- merge into last chunk
            last_conv = chunks[-1]["conversations"]
            last_conv.extend(current_chunk)

    # If splitting produced nothing valid, keep original (will be flagged in stats)
    if not chunks:
        return [entry]

    # Update total chunk count in metadata
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = len(chunks)

    return chunks


# ---------------------------------------------------------------------------
# Step 2: Deduplication
# ---------------------------------------------------------------------------


def dialogue_hash(conversation: list[dict[str, str]]) -> str:
    """MD5 hash of flattened dialogue text for dedup."""
    flat = flatten_conversation(conversation)
    return hashlib.md5(flat.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Step 4: LLM quality filter
# ---------------------------------------------------------------------------

QUALITY_FILTER_PROMPT = """\
You are a quality reviewer for AI tutoring dialogue training data. Evaluate this dialogue on three dimensions.

DIALOGUE (may be truncated):
{dialogue}

EVALUATION CRITERIA:

1. PEDAGOGICAL COHERENCE (1-5): Does the tutor scaffold learning through questions rather than giving answers directly? Does the tutor check understanding before moving on?
   - 5: Exemplary Socratic method, never gives answers, checks understanding
   - 3: Mixed -- sometimes scaffolds, sometimes lectures
   - 1: Tutor lectures, gives answers, no scaffolding

2. DIALOGUE NATURALNESS (1-5): Does the conversation flow realistically? Are responses appropriate lengths? Does it feel like a real lesson?
   - 5: Completely natural, realistic flow
   - 3: Somewhat stilted but acceptable
   - 1: Robotic, repetitive, or incoherent

3. INSTRUMENT PLAUSIBILITY (1-5, synthetic only): Are instrument-specific details accurate? No cross-instrument confusion?
   - 5: All instrument details are accurate
   - 3: Minor inaccuracies that don't affect pedagogy
   - 1: Major instrument confusion (e.g., telling a violinist to use pedal)
   - N/A: Not a music-specific dialogue

Respond in EXACTLY this JSON format (no other text):
{{"pedagogical_coherence": <int>, "dialogue_naturalness": <int>, "instrument_plausibility": <int or null>, "pass": <bool>, "reason": "<brief reason if fail>"}}

A dialogue PASSES if pedagogical_coherence >= 3 AND dialogue_naturalness >= 3 AND (instrument_plausibility is null OR instrument_plausibility >= 3)."""


def run_llm_quality_filter(
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Run LLM quality filter on entries using Sonnet.

    Returns:
        Tuple of (passed entries, failed entries, token usage).
    """
    try:
        import anthropic
    except ImportError:
        print("  WARNING: anthropic package not available. Skipping LLM filter.")
        return entries, [], {"input": 0, "output": 0}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  WARNING: ANTHROPIC_API_KEY not set. Skipping LLM filter.")
        return entries, [], {"input": 0, "output": 0}

    client = anthropic.Anthropic(api_key=api_key)

    passed = []
    failed = []
    token_usage = {"input": 0, "output": 0}
    errors = 0

    for idx, entry in enumerate(entries):
        if (idx + 1) % 50 == 0:
            print(f"    LLM filter: {idx + 1}/{len(entries)}...")

        # Truncate dialogue for cost control
        flat = flatten_conversation(entry["conversations"])
        if len(flat) > FILTER_TRUNCATION_CHARS:
            flat = flat[:FILTER_TRUNCATION_CHARS] + "\n[...truncated...]"

        prompt = QUALITY_FILTER_PROMPT.format(dialogue=flat)

        try:
            response = client.messages.create(
                model=SONNET_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            token_usage["input"] += response.usage.input_tokens
            token_usage["output"] += response.usage.output_tokens

            result_text = response.content[0].text.strip()

            # Parse JSON response
            # Handle potential markdown code block wrapping
            if result_text.startswith("```"):
                result_text = result_text.split("\n", 1)[-1]
                if result_text.endswith("```"):
                    result_text = result_text[:-3]
                result_text = result_text.strip()

            result = json.loads(result_text)

            if result.get("pass", True):
                entry["metadata"]["quality_scores"] = {
                    "pedagogical_coherence": result.get("pedagogical_coherence"),
                    "dialogue_naturalness": result.get("dialogue_naturalness"),
                    "instrument_plausibility": result.get("instrument_plausibility"),
                }
                passed.append(entry)
            else:
                entry["metadata"]["filter_reason"] = result.get("reason", "quality threshold")
                failed.append(entry)

        except json.JSONDecodeError:
            # On parse error, pass through (don't lose data)
            passed.append(entry)
            errors += 1
        except Exception:
            # On any API/network error, pass through
            passed.append(entry)
            errors += 1

        # Light rate limiting for filter calls
        time.sleep(0.2)

    if errors > 0:
        print(f"    LLM filter errors (passed through): {errors}")

    return passed, failed, token_usage


# ---------------------------------------------------------------------------
# Step 5: Leakage detection
# ---------------------------------------------------------------------------


def check_leakage(entry: dict[str, Any]) -> bool:
    """Check if any gpt turn contains system prompt leakage.

    Returns True if leakage found.
    """
    for turn in entry["conversations"]:
        if turn["from"] == "gpt":
            text_upper = turn["value"].upper()
            for phrase in LEAKAGE_PHRASES:
                if phrase.upper() in text_upper:
                    return True
    return False


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def print_token_stats(entries: list[dict[str, Any]]) -> None:
    """Print token length statistics."""
    if not entries:
        print("  No entries to compute stats.")
        return

    token_counts = []
    for entry in entries:
        tc = conversation_token_count(entry["conversations"])
        token_counts.append(tc)

    arr = np.array(token_counts)

    print(f"\n  Token Length Statistics:")
    print(f"    Count:    {len(arr)}")
    print(f"    Min:      {int(arr.min())}")
    print(f"    Max:      {int(arr.max())}")
    print(f"    Mean:     {arr.mean():.1f}")
    print(f"    Median:   {np.median(arr):.1f}")
    print(f"    Std:      {arr.std():.1f}")
    print(f"    >2048:    {int((arr > 2048).sum())}")
    print(f"    >1024:    {int((arr > 1024).sum())}")
    print(f"    <256:     {int((arr < 256).sum())}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_jsonl(filepath: Path, source_label: str) -> list[dict[str, Any]]:
    """Load entries from a JSONL file."""
    entries = []
    if not filepath.exists():
        print(f"  WARNING: {filepath} not found. Skipping {source_label}.")
        return entries

    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"  WARNING: JSON parse error in {filepath} line {line_num}: {e}")

    print(f"  Loaded {len(entries)} entries from {source_label} ({filepath.name})")
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge, filter, and produce final SFT training set."
    )
    parser.add_argument(
        "--skip-llm-filter",
        action="store_true",
        help="Skip LLM quality filter (step 4) for faster testing.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MERGE AND FILTER PIPELINE")
    print("=" * 60)

    # --- Load sources ---
    print("\nStep 0: Loading source data...")
    synthetic = load_jsonl(SYNTHETIC_FILE, "synthetic")
    convolearn = load_jsonl(CONVOLEARN_FILE, "convolearn")

    all_entries = synthetic + convolearn
    if not all_entries:
        print("ERROR: No data loaded from either source.")
        sys.exit(1)

    source_counts = {
        "synthetic_loaded": len(synthetic),
        "convolearn_loaded": len(convolearn),
        "total_loaded": len(all_entries),
    }
    print(f"  Total loaded: {len(all_entries)} ({len(synthetic)} synthetic + {len(convolearn)} convolearn)")

    # --- Step 1: Split long dialogues ---
    print("\nStep 1: Splitting long dialogues (>{} tokens)...".format(MAX_TOKENS))
    split_entries = []
    splits_performed = 0
    for entry in all_entries:
        chunks = split_long_dialogue(entry)
        if len(chunks) > 1:
            splits_performed += 1
        split_entries.extend(chunks)
    print(f"  Dialogues split: {splits_performed}")
    print(f"  Entries after splitting: {len(split_entries)}")

    # --- Step 2: Deduplication ---
    print("\nStep 2: Deduplicating...")
    seen_hashes = set()
    deduped = []
    dupes_removed = 0
    for entry in split_entries:
        h = dialogue_hash(entry["conversations"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(entry)
        else:
            dupes_removed += 1
    print(f"  Duplicates removed: {dupes_removed}")
    print(f"  Entries after dedup: {len(deduped)}")

    # --- Step 3: Token stats ---
    print("\nStep 3: Token length statistics...")
    print_token_stats(deduped)

    # --- Step 4: LLM quality filter ---
    failed_entries: list[dict[str, Any]] = []
    filter_token_usage = {"input": 0, "output": 0}

    if args.skip_llm_filter:
        print("\nStep 4: LLM quality filter SKIPPED (--skip-llm-filter)")
        filtered = deduped
    else:
        print(f"\nStep 4: LLM quality filter ({len(deduped)} entries)...")
        filtered, failed_entries, filter_token_usage = run_llm_quality_filter(deduped)
        print(f"  Passed: {len(filtered)}")
        print(f"  Failed: {len(failed_entries)}")

        filter_cost = (
            filter_token_usage["input"] / 1_000_000 * SONNET_INPUT_COST_PER_M
            + filter_token_usage["output"] / 1_000_000 * SONNET_OUTPUT_COST_PER_M
        )
        print(f"  Filter cost: ${filter_cost:.4f} ({filter_token_usage['input']:,} in / {filter_token_usage['output']:,} out)")

    # --- Step 5: Leakage detection ---
    print("\nStep 5: Checking for system prompt leakage...")
    clean = []
    leakage_count = 0
    for entry in filtered:
        if check_leakage(entry):
            leakage_count += 1
            entry["metadata"]["leakage_detected"] = True
            # Remove the entry rather than include leaked system prompts in training data
        else:
            clean.append(entry)
    print(f"  Leakage detected and removed: {leakage_count}")
    print(f"  Entries after leakage check: {len(clean)}")

    # --- Step 6: Write output ---
    print(f"\nStep 6: Writing output to {OUTPUT_FILE}...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as out_f:
        for entry in clean:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # --- Final stats ---
    # Count sources in final output
    final_synthetic = sum(
        1 for e in clean
        if e.get("metadata", {}).get("source") == "synthetic_two_llm"
    )
    final_convolearn = sum(
        1 for e in clean
        if e.get("metadata", {}).get("source") == "convolearn"
    )

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  Source Breakdown:")
    print(f"    Synthetic (loaded):    {source_counts['synthetic_loaded']}")
    print(f"    ConvoLearn (loaded):   {source_counts['convolearn_loaded']}")
    print(f"    Synthetic (final):     {final_synthetic}")
    print(f"    ConvoLearn (final):    {final_convolearn}")
    print(f"\n  Pipeline Stats:")
    print(f"    Total loaded:          {source_counts['total_loaded']}")
    print(f"    After splitting:       {len(split_entries)}")
    print(f"    After dedup:           {len(deduped)}")
    print(f"    After LLM filter:      {len(filtered)}")
    print(f"    After leakage check:   {len(clean)}")
    print(f"    Final training set:    {len(clean)}")
    print(f"\n  Output: {OUTPUT_FILE}")

    # Final token stats on clean set
    print("\n  Final Training Set Token Stats:")
    print_token_stats(clean)

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
