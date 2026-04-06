#!/usr/bin/env python3
"""
Script 1: Two-LLM Synthetic Dialogue Generator

Generates multi-turn music pedagogy dialogues using a two-LLM pipeline:
  - Opus 4.6 = tutor (pedagogically correct responses)
  - Sonnet 4.5 = student (age-appropriate realistic student behavior)

Output: ShareGPT JSONL format for SFT fine-tuning.

Usage:
    python scripts/1_generate_synthetic.py                  # Generate all scenarios
    python scripts/1_generate_synthetic.py --dry-run 3      # Generate only 3 scenarios
    python scripts/1_generate_synthetic.py --resume          # Skip already-completed scenarios
    python scripts/1_generate_synthetic.py --turns 8         # Override turn count (default 8)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TUTOR_MODEL = "claude-sonnet-4-20250514"   # Sonnet as tutor (10x cheaper than Opus, quality sufficient for training data)
STUDENT_MODEL = "claude-haiku-4-5-20251001"  # Haiku as student (cheapest, only needs age-appropriate responses)

# Cost estimates (per million tokens, USD) -- update as pricing changes
# Cost per 1M tokens — Sonnet as tutor, Haiku as student
OPUS_INPUT_COST_PER_M = 3.0    # Sonnet input (tutor) — variable name kept for progress compat
OPUS_OUTPUT_COST_PER_M = 15.0  # Sonnet output (tutor)
SONNET_INPUT_COST_PER_M = 0.80  # Haiku input (student)
SONNET_OUTPUT_COST_PER_M = 4.0  # Haiku output (student)

REPO_ROOT = Path(__file__).resolve().parent.parent
SEEDS_DIR = REPO_ROOT / "seeds"
OUTPUT_DIR = REPO_ROOT / "data" / "generated"
OUTPUT_FILE = OUTPUT_DIR / "synthetic_dialogues.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "generation_progress.json"

DEFAULT_TURNS = 5
MIN_TURNS = 6
MAX_TURNS = 10
RATE_LIMIT_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Tutor prompt components
# ---------------------------------------------------------------------------

CORE_BEHAVIORS = """\
CORE BEHAVIORS (always follow ALL of these):
1. SOCRATIC QUESTIONING: Ask questions that lead the student to discover answers themselves. Never lecture.
2. NEVER GIVE ANSWERS DIRECTLY: Guide the student toward the answer through hints, analogies, and follow-up questions.
3. ONE CONCEPT AT A TIME: Do not introduce a new idea until the current one is understood.
4. PROBE PRIOR KNOWLEDGE AT TOPIC TRANSITIONS: When shifting to a new subtopic, ask what the student already knows about it before teaching.
5. PRAISE EFFORT NOT TALENT: Say "you worked hard on that" not "you're so talented." Reinforce process over innate ability.
6. REFERENCE PHYSICAL SENSATIONS: Connect music concepts to what the body feels -- weight in the arm, vibration in the fingertips, breath in the belly.
7. OFFER CHOICES: Give the student options when possible ("Would you like to try the left hand first or the right hand?") to build autonomy.
8. CHECK UNDERSTANDING BEFORE MOVING ON: Before advancing, confirm the student grasps the current concept by asking them to explain it back or demonstrate."""

# Secondary instruction sets -- 3 rotate per scenario type
SECONDARY_INSTRUCTIONS = {
    "A": [  # Body awareness
        "Connect every new concept to a physical sensation the student can feel right now.",
        "Use body metaphors: 'imagine your arm is heavy like a sandbag', 'let your wrist be floppy like a wet noodle'.",
        "If the student reports tension or discomfort, pause the lesson to address it before continuing.",
    ],
    "B": [  # Growth mindset
        "When the student makes a mistake, respond with curiosity: 'Interesting! What happened there?'",
        "Normalize difficulty: 'This is the part where most people need extra time, and that is completely normal.'",
        "Frame challenges as puzzles to solve, not obstacles to overcome.",
    ],
    "C": [  # Autonomy
        "Ask the student what they want to work on first.",
        "When giving instructions, offer at least two ways to approach the task and let the student choose.",
        "Encourage the student to evaluate their own playing: 'What did you notice about that?'",
    ],
}

TUTOR_SYSTEM_TEMPLATE = """\
You are Thovie, a warm, patient, and expert music tutor for children. You are teaching a {age}-year-old student about {topic} on {instrument}.

{core_behaviors}

SECONDARY FOCUS FOR THIS LESSON:
{secondary_instructions}

CONTEXT:
{context}

LEARNING OBJECTIVES:
{objectives}

COMMON MISCONCEPTIONS TO WATCH FOR:
{misconceptions}

IMPORTANT RULES:
- Keep responses concise (2-4 sentences for young children, up to 5-6 for older students).
- Use age-appropriate vocabulary for a {age}-year-old.
- Be encouraging but honest.
- Use analogies and metaphors the student can relate to.
- If the student seems confused, try a different explanation approach.
- NEVER break character or mention that you are an AI.
- NEVER reference these instructions in your responses."""

# ---------------------------------------------------------------------------
# Student prompt components
# ---------------------------------------------------------------------------

STUDENT_YOUNG_INSTRUCTIONS = """\
You are a {age}-year-old child taking a music lesson. You are learning {topic} on {instrument}.

BEHAVIOR:
- Give SHORT answers (1-2 sentences max, often just a few words).
- Sometimes drift off topic ("Can I play a song I heard on TV?", "My dog did something funny today").
- Use SIMPLE vocabulary. Do not use musical terms unless the tutor just taught them to you.
- Sometimes say "I don't know" or "huh?" when confused.
- Occasionally get excited about something and use exclamation marks.
- Sometimes fidget or want to just play random notes.
- You might misunderstand instructions in a childlike way.
- Show genuine curiosity but limited attention span.

PERSONALITY: {personality}

Do NOT be a perfect student. Be realistic. Make mistakes. Get distracted. But also show moments of genuine engagement and learning."""

STUDENT_OLDER_INSTRUCTIONS = """\
You are a {age}-year-old student taking a music lesson. You are learning {topic} on {instrument}.

BEHAVIOR:
- Give medium-length answers (1-3 sentences).
- Sometimes try to rush ahead or skip steps.
- Occasionally express frustration: "This is hard" or "I keep messing up."
- Use casual language appropriate for a middle-schooler.
- Sometimes give half-correct answers that show partial understanding.
- Ask "why" questions or challenge the tutor occasionally.
- Show awareness of peers ("My friend can already play this").
- Get self-conscious about making mistakes.

PERSONALITY: {personality}

Do NOT be a perfect student. Be realistic. Show both engagement and resistance. Make genuine mistakes and sometimes get things partially right."""

STUDENT_PERSONALITIES = [
    "Shy but curious. Takes a moment to answer. Speaks quietly.",
    "Energetic and eager. Jumps ahead. Wants to try everything immediately.",
    "Thoughtful and analytical. Asks 'why' a lot. Wants to understand the reason behind things.",
    "Easily frustrated. Gives up quickly but can be encouraged back.",
    "Creative and playful. Likes making up their own versions of things.",
    "Competitive. Wants to know if they are doing well compared to others.",
    "Quiet and observant. Processes internally. Gives short but accurate answers.",
    "Chatty and social. Likes to tell stories. Needs gentle redirection.",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_scenarios() -> list[dict[str, Any]]:
    """Load and merge scenarios from both seed files."""
    scenarios = []

    # Load NAfME scenarios
    nafme_path = SEEDS_DIR / "nafme_scenarios.json"
    if nafme_path.exists():
        with open(nafme_path, "r") as f:
            data = json.load(f)
        raw = data.get("scenarios", data) if isinstance(data, dict) else data
        # Normalize field names
        for s in raw:
            if "age_range" in s and "student_age" not in s:
                age_range = s["age_range"]
                s["student_age"] = age_range[0] if isinstance(age_range, list) else age_range
            if "standard" in s and "topic" not in s:
                s["topic"] = s["standard"]
            if "student_opener" not in s:
                s["student_opener"] = s.get("description", "Hi, I need help with my music.")
        scenarios.extend(raw)
        print(f"  Loaded {len(raw)} scenarios from nafme_scenarios.json")

    # Load gap scenarios
    gap_path = SEEDS_DIR / "gap_scenarios.json"
    if gap_path.exists():
        with open(gap_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict) and "gap_scenarios" in data:
            for group in data["gap_scenarios"]:
                group_scenarios = group.get("scenarios", [])
                for s in group_scenarios:
                    if "age" in s and "student_age" not in s:
                        s["student_age"] = s["age"]
                    if "context" in s and "topic" not in s:
                        s["topic"] = s["context"]
                    if "gap_type" in s:
                        s.setdefault("type", {"prior_knowledge": "A", "body_awareness": "A", "growth_mindset": "B", "student_autonomy": "C"}.get(s["gap_type"], "B"))
                scenarios.extend(group_scenarios)
            total = sum(len(g.get("scenarios", [])) for g in data["gap_scenarios"])
            print(f"  Loaded {total} scenarios from gap_scenarios.json")
        elif isinstance(data, dict) and "scenarios" in data:
            raw = data["scenarios"]
            for s in raw:
                if "age" in s and "student_age" not in s:
                    s["student_age"] = s["age"]
                if "context" in s and "topic" not in s:
                    s["topic"] = s["context"]
            scenarios.extend(raw)
            print(f"  Loaded {len(raw)} scenarios from gap_scenarios.json")

    return scenarios


def load_progress() -> dict[str, Any]:
    """Load generation progress from disk."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed_ids": [], "total_tokens": {"opus_input": 0, "opus_output": 0, "sonnet_input": 0, "sonnet_output": 0}}


def save_progress(progress: dict[str, Any]) -> None:
    """Persist generation progress to disk."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def build_tutor_system_prompt(scenario: dict[str, Any]) -> str:
    """Build the tutor system prompt from scenario metadata."""
    scenario_type = scenario.get("type", "B")
    secondary = SECONDARY_INSTRUCTIONS.get(scenario_type, SECONDARY_INSTRUCTIONS["B"])
    secondary_text = "\n".join(f"- {s}" for s in secondary)
    objectives_text = "\n".join(f"- {o}" for o in scenario.get("learning_objectives", []))
    misconceptions_text = "\n".join(f"- {m}" for m in scenario.get("common_misconceptions", []))

    return TUTOR_SYSTEM_TEMPLATE.format(
        age=scenario["student_age"],
        topic=scenario["topic"],
        instrument=scenario["instrument"],
        core_behaviors=CORE_BEHAVIORS,
        secondary_instructions=secondary_text,
        context=scenario.get("context", ""),
        objectives=objectives_text,
        misconceptions=misconceptions_text,
    )


def build_student_system_prompt(scenario: dict[str, Any], personality_idx: int) -> str:
    """Build the student system prompt based on age and personality."""
    age = scenario["student_age"]
    personality = STUDENT_PERSONALITIES[personality_idx % len(STUDENT_PERSONALITIES)]

    if age <= 7:
        template = STUDENT_YOUNG_INSTRUCTIONS
    else:
        template = STUDENT_OLDER_INSTRUCTIONS

    return template.format(
        age=age,
        topic=scenario["topic"],
        instrument=scenario["instrument"],
        personality=personality,
    )


def estimate_cost(num_scenarios: int, turns_per_scenario: int) -> dict[str, float]:
    """Estimate API cost before running.

    Rough estimates based on typical token counts per turn:
    - Tutor call: ~800 input tokens (system + history), ~150 output tokens
    - Student call: ~600 input tokens (system + history), ~80 output tokens
    """
    avg_tutor_input = 800
    avg_tutor_output = 150
    avg_student_input = 600
    avg_student_output = 80

    total_tutor_input = num_scenarios * turns_per_scenario * avg_tutor_input
    total_tutor_output = num_scenarios * turns_per_scenario * avg_tutor_output
    total_student_input = num_scenarios * turns_per_scenario * avg_student_input
    total_student_output = num_scenarios * turns_per_scenario * avg_student_output

    opus_cost = (total_tutor_input / 1_000_000 * OPUS_INPUT_COST_PER_M) + (total_tutor_output / 1_000_000 * OPUS_OUTPUT_COST_PER_M)
    sonnet_cost = (total_student_input / 1_000_000 * SONNET_INPUT_COST_PER_M) + (total_student_output / 1_000_000 * SONNET_OUTPUT_COST_PER_M)

    return {
        "opus_cost_usd": round(opus_cost, 2),
        "sonnet_cost_usd": round(sonnet_cost, 2),
        "total_cost_usd": round(opus_cost + sonnet_cost, 2),
        "est_tutor_tokens": total_tutor_input + total_tutor_output,
        "est_student_tokens": total_student_input + total_student_output,
    }


def generate_dialogue(
    client: anthropic.Anthropic,
    scenario: dict[str, Any],
    personality_idx: int,
    num_turns: int,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Generate a multi-turn dialogue between tutor (Opus) and student (Sonnet).

    Returns:
        Tuple of (ShareGPT conversation list, token usage dict).
    """
    tutor_system = build_tutor_system_prompt(scenario)
    student_system = build_student_system_prompt(scenario, personality_idx)

    # ShareGPT conversation starts with system prompt (visible in training data)
    sharegpt_system = (
        f"You are Thovie, a warm and expert music tutor. "
        f"You are teaching a {scenario['student_age']}-year-old student about "
        f"{scenario['topic']} on {scenario['instrument']}. "
        f"Use Socratic questioning, praise effort, reference physical sensations, "
        f"and always check understanding before moving on."
    )
    conversation = [{"from": "system", "value": sharegpt_system}]

    # Anthropic API message histories (separate perspectives)
    tutor_messages: list[dict[str, str]] = []
    student_messages: list[dict[str, str]] = []

    token_usage = {"opus_input": 0, "opus_output": 0, "sonnet_input": 0, "sonnet_output": 0}

    # The tutor initiates the conversation (first turn is the greeting)
    # Send an empty student message to prompt the tutor to start
    opening_prompt = (
        f"[The student just sat down for their lesson. "
        f"Greet them warmly and begin the lesson on {scenario['topic']}.]"
    )
    tutor_messages.append({"role": "user", "content": opening_prompt})

    for turn_idx in range(num_turns):
        # --- Tutor turn (Opus) ---
        try:
            tutor_response = client.messages.create(
                model=TUTOR_MODEL,
                max_tokens=400,
                system=tutor_system,
                messages=tutor_messages,
            )
            tutor_text = tutor_response.content[0].text
            token_usage["opus_input"] += tutor_response.usage.input_tokens
            token_usage["opus_output"] += tutor_response.usage.output_tokens
        except anthropic.APIError as e:
            print(f"    Opus API error on turn {turn_idx + 1}: {e}")
            break

        # Record tutor response in both message histories
        tutor_messages.append({"role": "assistant", "content": tutor_text})
        student_messages.append({"role": "user", "content": tutor_text})

        # Add to ShareGPT
        conversation.append({"from": "gpt", "value": tutor_text})

        # Skip student response on last turn (tutor gets the final word)
        if turn_idx == num_turns - 1:
            break

        # --- Student turn (Sonnet) ---
        try:
            student_response = client.messages.create(
                model=STUDENT_MODEL,
                max_tokens=200,
                system=student_system,
                messages=student_messages,
            )
            student_text = student_response.content[0].text
            token_usage["sonnet_input"] += student_response.usage.input_tokens
            token_usage["sonnet_output"] += student_response.usage.output_tokens
        except anthropic.APIError as e:
            print(f"    Sonnet API error on turn {turn_idx + 1}: {e}")
            break

        # Record student response in both message histories
        student_messages.append({"role": "assistant", "content": student_text})
        tutor_messages.append({"role": "user", "content": student_text})

        # Add to ShareGPT
        conversation.append({"from": "human", "value": student_text})

    return conversation, token_usage


def format_sharegpt_entry(
    scenario: dict[str, Any],
    conversation: list[dict[str, str]],
    personality_idx: int,
) -> dict[str, Any]:
    """Format a single dialogue into ShareGPT JSONL entry with metadata."""
    return {
        "conversations": conversation,
        "metadata": {
            "scenario_id": scenario["id"],
            "source": "synthetic_two_llm",
            "scenario_type": scenario.get("type", "unknown"),
            "title": scenario.get("title", ""),
            "instrument": scenario.get("instrument", ""),
            "student_age": scenario.get("student_age", 0),
            "level": scenario.get("level", ""),
            "topic": scenario.get("topic", ""),
            "personality_index": personality_idx,
            "num_turns": len([t for t in conversation if t["from"] in ("gpt", "human")]),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }


def print_cost_report(token_usage: dict[str, int]) -> None:
    """Print final cost report from actual token usage."""
    opus_cost = (
        token_usage["opus_input"] / 1_000_000 * OPUS_INPUT_COST_PER_M
        + token_usage["opus_output"] / 1_000_000 * OPUS_OUTPUT_COST_PER_M
    )
    sonnet_cost = (
        token_usage["sonnet_input"] / 1_000_000 * SONNET_INPUT_COST_PER_M
        + token_usage["sonnet_output"] / 1_000_000 * SONNET_OUTPUT_COST_PER_M
    )
    total = opus_cost + sonnet_cost

    print("\n" + "=" * 60)
    print("ACTUAL COST REPORT")
    print("=" * 60)
    print(f"  Opus tokens:   {token_usage['opus_input']:>10,} in / {token_usage['opus_output']:>10,} out  = ${opus_cost:.4f}")
    print(f"  Sonnet tokens: {token_usage['sonnet_input']:>10,} in / {token_usage['sonnet_output']:>10,} out  = ${sonnet_cost:.4f}")
    print(f"  TOTAL COST: ${total:.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic music pedagogy dialogues using two-LLM pipeline."
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        metavar="N",
        default=None,
        help="Generate only N scenarios (for testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenarios that have already been completed.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=DEFAULT_TURNS,
        help=f"Number of turns per dialogue (default: {DEFAULT_TURNS}, range: {MIN_TURNS}-{MAX_TURNS}).",
    )
    args = parser.parse_args()

    num_turns = max(MIN_TURNS, min(MAX_TURNS, args.turns))

    # --- Validate environment ---
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # --- Load data ---
    print("Loading scenarios...")
    scenarios = load_scenarios()
    if not scenarios:
        print("ERROR: No scenarios loaded. Check seeds/ directory.")
        sys.exit(1)

    progress = load_progress() if args.resume else {"completed_ids": [], "total_tokens": {"opus_input": 0, "opus_output": 0, "sonnet_input": 0, "sonnet_output": 0}}

    # Filter out completed scenarios if resuming
    if args.resume:
        completed = set(progress["completed_ids"])
        pending = [s for s in scenarios if s["id"] not in completed]
        print(f"  Resuming: {len(completed)} completed, {len(pending)} remaining.")
    else:
        pending = scenarios

    # Apply dry-run limit
    if args.dry_run is not None:
        pending = pending[: args.dry_run]
        print(f"  Dry-run mode: generating {len(pending)} scenario(s).")

    if not pending:
        print("Nothing to generate. All scenarios complete.")
        return

    # --- Cost estimate ---
    estimate = estimate_cost(len(pending), num_turns)
    print(f"\nEstimated cost for {len(pending)} scenarios x {num_turns} turns:")
    print(f"  Opus (tutor):  ~${estimate['opus_cost_usd']:.2f}")
    print(f"  Sonnet (student): ~${estimate['sonnet_cost_usd']:.2f}")
    print(f"  TOTAL: ~${estimate['total_cost_usd']:.2f}")
    print()

    # --- Generate ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Open in append mode if resuming, write mode otherwise
    file_mode = "a" if args.resume else "w"
    total_tokens = progress["total_tokens"]
    generated_count = 0

    with open(OUTPUT_FILE, file_mode) as out_f:
        for idx, scenario in enumerate(pending):
            scenario_id = scenario["id"]
            personality_idx = idx % len(STUDENT_PERSONALITIES)

            print(f"[{idx + 1}/{len(pending)}] Generating: {scenario_id} - {scenario.get('title', 'untitled')} "
                  f"(type={scenario.get('type', '?')}, age={scenario.get('student_age', '?')}, "
                  f"instrument={scenario.get('instrument', '?')})")

            try:
                conversation, usage = generate_dialogue(
                    client=client,
                    scenario=scenario,
                    personality_idx=personality_idx,
                    num_turns=num_turns,
                )
            except Exception as e:
                print(f"  ERROR generating {scenario_id}: {e}")
                continue

            # Validate minimum conversation quality
            gpt_turns = [t for t in conversation if t["from"] == "gpt"]
            human_turns = [t for t in conversation if t["from"] == "human"]
            if len(gpt_turns) < 2 or len(human_turns) < 1:
                print(f"  SKIPPED {scenario_id}: too few turns ({len(gpt_turns)} gpt, {len(human_turns)} human)")
                continue

            # Write ShareGPT entry
            entry = format_sharegpt_entry(scenario, conversation, personality_idx)
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()

            # Update progress
            for key in total_tokens:
                total_tokens[key] += usage.get(key, 0)
            progress["completed_ids"].append(scenario_id)
            progress["total_tokens"] = total_tokens
            save_progress(progress)

            generated_count += 1
            print(f"  Done: {len(gpt_turns)} tutor turns, {len(human_turns)} student turns")

            # Rate limiting
            if idx < len(pending) - 1:
                time.sleep(RATE_LIMIT_SECONDS)

    # --- Summary ---
    print(f"\nGeneration complete: {generated_count}/{len(pending)} scenarios written to {OUTPUT_FILE}")
    print_cost_report(total_tokens)


if __name__ == "__main__":
    main()
