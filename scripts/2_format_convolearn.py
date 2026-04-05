#!/usr/bin/env python3
"""
Script 2: ConvoLearn Dataset Formatter

Downloads the masharma/convolearn dataset from HuggingFace (MIT license, ~1,250
earth science tutoring dialogues) and converts it into ShareGPT JSONL format
for merging with synthetic music pedagogy data.

The ConvoLearn dialogues teach general tutoring patterns (Socratic questioning,
scaffolding, checking understanding) which transfer to music pedagogy even though
the domain is earth science. This is intentional: we are fine-tuning HOW to teach,
not WHAT to teach.

Usage:
    python scripts/2_format_convolearn.py
    python scripts/2_format_convolearn.py --inspect    # Print schema and sample values only
    python scripts/2_format_convolearn.py --limit 100  # Process only first 100 entries
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_NAME = "masharma/convolearn"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_FILE = OUTPUT_DIR / "convolearn_sharegpt.jsonl"

MIN_CONVERSATION_LENGTH = 50  # Skip entries with <50 chars in cleaned_conversation

SYSTEM_PROMPT = (
    "You are a tutor. Help the student learn through Socratic questioning. "
    "Guide the student to discover answers themselves rather than giving answers directly. "
    "Praise effort, check understanding, and adapt to the student's pace."
)

# Regex to split on speaker labels. Handles variations like:
# "Tutor:", "Student:", "Tutor :", "TUTOR:", etc.
SPEAKER_PATTERN = re.compile(
    r"(?:^|\n)\s*(Tutor|Student)\s*:\s*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_conversation(raw_text: str) -> list[dict[str, str]]:
    """Parse a ConvoLearn cleaned_conversation string into turn dicts.

    The raw format is typically:
        Tutor: Hello! Today we'll...
        Student: Hi! I was wondering...
        Tutor: Great question! ...

    Returns a list of {"speaker": "tutor"|"student", "text": "..."} dicts.
    """
    if not raw_text or not raw_text.strip():
        return []

    # Split by speaker labels, keeping the label as a capture group
    parts = SPEAKER_PATTERN.split(raw_text.strip())

    turns = []
    # parts[0] is text before the first speaker label (usually empty)
    # Then alternating: speaker_label, text, speaker_label, text, ...
    i = 1  # skip pre-label text
    while i + 1 < len(parts):
        speaker = parts[i].strip().lower()
        text = parts[i + 1].strip()
        if text:
            turns.append({"speaker": speaker, "text": text})
        i += 2

    return turns


def turns_to_sharegpt(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Convert parsed turns into ShareGPT format.

    Maps: Tutor -> "gpt", Student -> "human"
    Prepends system turn.
    Ensures conversation starts with a "human" turn.
    """
    if not turns:
        return []

    conversation = [{"from": "system", "value": SYSTEM_PROMPT}]

    # If conversation starts with tutor, prepend a generic student opener
    if turns[0]["speaker"] == "tutor":
        conversation.append({
            "from": "human",
            "value": "Hi! I'm ready for today's lesson.",
        })

    for turn in turns:
        role = "gpt" if turn["speaker"] == "tutor" else "human"

        # Avoid consecutive same-role turns -- merge them
        if conversation and conversation[-1]["from"] == role:
            conversation[-1]["value"] += " " + turn["text"]
        else:
            conversation.append({"from": role, "value": turn["text"]})

    return conversation


def build_metadata(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Extract metadata fields from a ConvoLearn row."""
    metadata: dict[str, Any] = {
        "source": "convolearn",
        "domain": "earth_science",
        "convolearn_index": idx,
    }

    # Preserve available metadata fields
    for field in [
        "effectiveness_consensus",
        "completeness_consensus",
        "num_exchanges",
        "earthscience_topic",
        "kb_dim",
    ]:
        if field in row and row[field] is not None:
            metadata[field] = row[field]

    return metadata


# ---------------------------------------------------------------------------
# Schema inspection
# ---------------------------------------------------------------------------


def inspect_dataset(dataset) -> None:
    """Print column names, types, and sample values for the first few rows."""
    print("\n" + "=" * 60)
    print("DATASET SCHEMA INSPECTION")
    print("=" * 60)

    print(f"\nDataset size: {len(dataset)} rows")
    print(f"Columns: {dataset.column_names}")
    print(f"Features: {dataset.features}")

    if len(dataset) > 0:
        print("\n--- Sample row (index 0) ---")
        row = dataset[0]
        for key, value in row.items():
            val_str = str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"  {key}: {val_str}")

    if len(dataset) > 1:
        print("\n--- Sample row (index 1) ---")
        row = dataset[1]
        for key, value in row.items():
            val_str = str(value)
            if len(val_str) > 200:
                val_str = val_str[:200] + "..."
            print(f"  {key}: {val_str}")

    # Print value distributions for key categorical fields
    for field in ["effectiveness_consensus", "completeness_consensus", "earthscience_topic", "kb_dim"]:
        if field in dataset.column_names:
            try:
                values = dataset[field]
                unique = set(str(v) for v in values if v is not None)
                print(f"\n  {field}: {len(unique)} unique values")
                if len(unique) <= 15:
                    for v in sorted(unique):
                        count = sum(1 for x in values if str(x) == v)
                        print(f"    {v}: {count}")
            except Exception:
                pass

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and format ConvoLearn dataset into ShareGPT JSONL."
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print schema and sample values only, do not write output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N entries.",
    )
    args = parser.parse_args()

    # --- Download dataset ---
    print(f"Downloading dataset: {DATASET_NAME}")
    start_time = time.time()

    try:
        ds = load_dataset(DATASET_NAME)
    except Exception as e:
        print(f"ERROR: Failed to download dataset: {e}")
        sys.exit(1)

    # ConvoLearn typically has a single 'train' split
    if "train" in ds:
        dataset = ds["train"]
    else:
        # Use whichever split is available
        split_name = list(ds.keys())[0]
        print(f"  Using split: {split_name}")
        dataset = ds[split_name]

    download_time = time.time() - start_time
    print(f"  Downloaded {len(dataset)} rows in {download_time:.1f}s")

    # --- Schema inspection ---
    inspect_dataset(dataset)

    if args.inspect:
        print("Inspect-only mode. Exiting.")
        return

    # --- Determine conversation field ---
    # ConvoLearn uses 'cleaned_conversation' but fall back to alternatives
    conv_field = None
    for candidate in ["cleaned_conversation", "conversation", "text", "dialogue"]:
        if candidate in dataset.column_names:
            conv_field = candidate
            break

    if conv_field is None:
        print(f"ERROR: Could not find conversation field. Available columns: {dataset.column_names}")
        sys.exit(1)

    print(f"  Using conversation field: '{conv_field}'")

    # --- Process entries ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    if args.limit is not None:
        total = min(total, args.limit)

    written = 0
    skipped_empty = 0
    skipped_short = 0
    skipped_parse = 0

    with open(OUTPUT_FILE, "w") as out_f:
        for idx in range(total):
            row = dataset[idx]
            raw_text = row.get(conv_field, "")

            # Skip empty
            if not raw_text or not raw_text.strip():
                skipped_empty += 1
                continue

            # Skip too short
            if len(raw_text.strip()) < MIN_CONVERSATION_LENGTH:
                skipped_short += 1
                continue

            # Parse into turns
            turns = parse_conversation(raw_text)
            if len(turns) < 2:
                skipped_parse += 1
                continue

            # Convert to ShareGPT
            conversation = turns_to_sharegpt(turns)

            # Validate: need at least 1 human + 1 gpt turn (beyond system)
            gpt_count = sum(1 for t in conversation if t["from"] == "gpt")
            human_count = sum(1 for t in conversation if t["from"] == "human")
            if gpt_count < 1 or human_count < 1:
                skipped_parse += 1
                continue

            # Build entry
            metadata = build_metadata(row, idx)
            metadata["num_turns"] = gpt_count + human_count

            entry = {
                "conversations": conversation,
                "metadata": metadata,
            }

            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{total} rows ({written} written)...")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("CONVOLEARN FORMATTING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total rows in dataset: {len(dataset)}")
    print(f"  Rows processed:        {total}")
    print(f"  Written to output:     {written}")
    print(f"  Skipped (empty):       {skipped_empty}")
    print(f"  Skipped (too short):   {skipped_short}")
    print(f"  Skipped (parse error): {skipped_parse}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
