#!/usr/bin/env python3
"""
Judge Validation Script — following validate-evaluator skill from hamelsmu/evals-skills.

Steps:
1. Load human labels from eval/labeling/labels.json
2. Split into train (15%) / dev (45%) / test (40%)
3. Run LLM judge on dev set for each dimension
4. Compute TPR/TNR
5. Report disagreements for iteration

Usage:
    python scripts/validate_judges.py --dimension D2_check
    python scripts/validate_judges.py --dimension D4_load
    python scripts/validate_judges.py --all
"""
import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
import numpy as np

JUDGE_MODEL = "claude-sonnet-4-20250514"
LABELS_PATH = Path("eval/labeling/labels.json")
RESULTS_PATH = Path("eval/results/baseline_v3_2026-04-05.json")
VALIDATION_DIR = Path("eval/validation")

# Map dimension short names to labels.json field names
DIM_MAP = {
    "D1a_steps": "D1a_steps",
    "D1b_question": "D1b_question",
    "D2_check": "D2_check",
    "D3_age": "D3_age",
    "D4_load": "D4_load",
    "D5_prior": "D5_prior",
    "D6_growth": "D6_growth",
    "D7_higher": "D7_higher",
    "D8_practice": "D8_practice",
    "D9_motor": "D9_motor",
    "D10_choice": "D10_choice",
}

# Judge prompts for each dimension (binary Pass/Fail)
JUDGE_PROMPTS = {
    "D1a_steps": """You are evaluating a music tutor's response to a student.

TASK: Determine whether the tutor breaks its response into sequential steps or substeps, rather than giving the complete answer in a single undifferentiated block.

CRITERION: The response must decompose the task or explanation into at least 2 sequential steps or substeps that the student can follow one at a time.

PASS definition:
- Response breaks the task into at least 2 sequential steps or substeps.
- Steps can be numbered, bulleted, or clearly separated in prose (e.g., "First... Then... Next...").

FAIL definition:
- Response gives the complete answer or solution in a single block with no step decomposition.
- Response meta-discusses what it WOULD say ("Here's how I'd respond...") instead of actually scaffolding.

FEW-SHOT EXAMPLES:

Example 1 — PASS (7yo piano, "can't play both hands"):
Tutor response: "I understand! Playing with both hands together is tricky at first - it's like trying to pat your head and rub your tummy at the same time! Let's make it easier: Start with baby steps: 1. First, let's play each hand separately until they know their parts really well 2. Then try playing them together very slowly..."
Critique: The tutor breaks the task into numbered sequential steps (play hands separately, then combine slowly). Clear step decomposition.
Result: Pass

Example 2 — FAIL (10yo piano, "why practice scales"):
Tutor response: "Here's how I'd respond... I'll break it down into a few parts" — then meta-discusses what it WOULD say instead of actually scaffolding steps for the student.
Critique: The tutor announces it will break things down but never actually provides sequential steps. It talks about what it would do rather than doing it.
Result: Fail

Example 3 — BORDERLINE PASS (14yo piano, "voicing melody"):
Tutor response: "Let me break this down for you: What voicing means: It's making the melody stand out from the accompaniment. Step 1: Play the melody notes slightly louder. Step 2: Play the accompaniment softer. Step 3: Practice hands separately focusing on dynamics."
Critique: The tutor does break the explanation into sequential steps, even though it gives the answer rather than guiding discovery. The criterion is about step decomposition, not discovery — and it decomposes into 3 clear steps.
Result: Pass

Now evaluate the following tutor response. First write a critique analyzing whether the response decomposes into sequential steps, then give your verdict.

Output ONLY valid JSON: {"critique": "your detailed assessment", "pass": true, "score": 1} for Pass, or {"critique": "your detailed assessment", "pass": false, "score": 0} for Fail.""",

    "D1b_question": """You are evaluating a music tutor's response to a student.

TASK: Determine whether the tutor includes at least one genuine question that guides the student to think or try something.

CRITERION: The response must contain at least one question directed at the student that prompts thinking, reflection, or action — not just a rhetorical question.

PASS definition:
- Response includes at least one genuine question that guides the student to think or try something.
- Questions like "What happens when you...?", "Can you try...?", "Which part feels hardest?" count.
- Rhetorical questions that are really statements ("Isn't that cool?") still count if they prompt reflection.

FAIL definition:
- Response is entirely declarative — only statements, instructions, or explanations with no questions.
- Commands ("Do this", "Try that") without question marks do NOT count as guiding questions.

FEW-SHOT EXAMPLES:

Example 1 — PASS (7yo piano, "can't play both hands"):
Tutor response: "Playing with both hands is tricky! Let's start by playing each hand separately. Can you play the right hand part all by itself first? Once that feels easy, we'll add the left hand very slowly."
Critique: The tutor asks "Can you play the right hand part all by itself first?" — a genuine question that guides the student to try something specific.
Result: Pass

Example 2 — FAIL (10yo piano, "why practice scales"):
Tutor response: "Scales are important because they build finger strength and help you learn key signatures. Think of them like warm-ups before a soccer game. Practice them slowly at first, then speed up gradually. Do C major first, then G major."
Critique: The response is entirely declarative. It explains, instructs, and prescribes, but never asks the student a question. "Think of them like warm-ups" is a command, not a question.
Result: Fail

Example 3 — BORDERLINE PASS (8yo piano, "left hand is different notes"):
Tutor response: "Great job learning the right hand! The left hand often plays different notes to add harmony. Here's a fun idea — what if we tried playing just the first two measures with the left hand? See how it sounds different?"
Critique: "What if we tried playing just the first two measures?" is a genuine guiding question, even though it's somewhat leading. It invites the student to try something.
Result: Pass

Now evaluate the following tutor response. First write a critique analyzing whether the response contains guiding questions, then give your verdict.

Output ONLY valid JSON: {"critique": "your detailed assessment", "pass": true, "score": 1} for Pass, or {"critique": "your detailed assessment", "pass": false, "score": 0} for Fail.""",

    "D2_check": """You are evaluating whether a music tutor's response includes a comprehension check or practice invitation.

## Definitions

PASS: The tutor includes ANY of these before or after instruction:
- Asks the student to try, play, or demonstrate something ("Try playing just the right hand", "Can you try that?")
- Asks the student to explain back or describe what they notice ("What do you hear when you do X?", "Can you hear the difference?")
- Invites the student to practice a specific exercise ("Would you like to try the pedal exercise?", "Let's try this together")
- Asks a diagnostic question about the student's experience ("How does it feel when you do X?", "What happens when you try?")
All of these count as verification — they require the student to engage with the material, not just passively receive instruction.

FAIL: The tutor ONLY explains/teaches and ends without ANY invitation to try, demonstrate, or reflect. Pure information delivery with no student action requested. Generic closers like "Does that make sense?" or "Any questions?" alone do NOT count — but "Would you like to try X?" DOES count because it invites specific action.

## Examples

### Example 1: PASS
Student (7yo piano): "I can't play with both hands at the same time"
Response: "Let's start with just the right hand. Can you play the first four notes for me? Once that feels easy, we'll try the left hand alone, and then see what happens when we put them together very slowly."
Critique: The tutor asks the student to play the right hand ("Can you play the first four notes for me?") before moving to the next step. This is a genuine comprehension/skill check embedded in the instruction flow.
Result: Pass

### Example 2: FAIL
Student (10yo piano): "Why do I have to practice scales? They're boring"
Response: "Scales are like a superpower for piano players! Just like soccer players do drills, scales train your fingers to be fast and accurate. They help you learn new songs quicker because your fingers already know the patterns. Try thinking of them as a game — see how fast you can play C major without mistakes!"
Critique: The tutor explains why scales matter and suggests reframing them as a game, but never checks whether the student understood the connection between scales and song-learning. The closing suggestion "try thinking of them as a game" is an instruction, not a check. No verification of understanding.
Result: Fail

### Example 3: PASS (borderline)
Student (14yo piano): "I don't understand how to voice the melody over the accompaniment"
Response: "Voicing is about making the melody louder than the accompaniment. Try this: play just the melody notes with your right hand, pressing firmly. Now play just the left hand very softly. Can you hear the difference in volume? That difference is what we want when both hands play together."
Critique: The tutor asks "Can you hear the difference in volume?" — this is a borderline check. It asks the student to notice something specific (volume contrast) rather than just asking "do you understand?" The tutor is verifying the student can perceive the concept before combining hands. Borderline but passes.
Result: Pass

## Output Format
Respond with JSON only:
{"critique": "detailed assessment referencing specific evidence from the response", "result": "Pass or Fail"}""",

    "D4_load": """You are evaluating whether a music tutor's response manages cognitive load by focusing on ONE main concept.

## Definitions

PASS: The response addresses one main concept or skill, even if it uses numbered steps to break that concept down. Sequential steps for ONE skill (e.g., "Step 1: play right hand. Step 2: play left hand. Step 3: combine slowly") is PASS — that's scaffolding, not overload. A structured response with clear focus on a single topic passes even if it's detailed.

FAIL: The response introduces 3 or more UNRELATED concepts (e.g., "work on posture, AND practice scales, AND try sight-reading, AND listen to recordings"). The key is whether items are independent topics vs sequential steps for one skill. Also FAIL if it uses undefined jargon the student hasn't learned, or is an unstructured wall of text.

IMPORTANT: A numbered list is NOT automatically overload. Ask: "Are these items steps toward ONE skill, or independent topics?"
- "1. Practice hands separately. 2. Combine slowly. 3. Speed up gradually" = ONE concept (hand coordination) in steps = PASS
- "1. Fix your posture. 2. Practice scales. 3. Learn music theory. 4. Try sight-reading" = FOUR unrelated topics = FAIL
- A long response that stays focused on ONE technique with multiple tips for that technique = PASS
- A response that covers posture AND technique AND theory AND repertoire = FAIL

When in doubt: if the student could summarize the response as "my tutor helped me with [ONE thing]", it's PASS.

## Examples

### Example 1: PASS
Student (7yo piano): "I can't play with both hands at the same time"
Response: "Playing with both hands is tricky! Let's focus on just one thing: getting each hand really comfortable on its own first. Can you play the right hand part five times in a row without stopping? Once that feels easy, we'll try the left hand."
Critique: One concept: practice hands separately. The response stays focused on this single step without jumping to rhythm, posture, or fingering. Clear and manageable for a 7-year-old.
Result: Pass

### Example 2: FAIL
Student (8yo guitar): "The strings hurt my fingers"
Response: "That's normal! Here are some tips: 1) Press with your fingertips, not the pads. 2) Start with nylon strings instead of steel. 3) Practice for shorter sessions. 4) Try using a capo to reduce tension. 5) Make sure your action height is correct. 6) Your calluses will develop over time. Also, make sure you're using proper posture..."
Critique: Six independent suggestions plus a posture aside. An 8-year-old cannot process and act on 6+ tips. Even though each tip is correct, the combined load overwhelms. Should have picked ONE actionable thing (e.g., "let's check how you're pressing the strings — show me your left hand").
Result: Fail

### Example 3: PASS (borderline)
Student (12yo guitar): "I can play chords but switching between them is slow"
Response: "Chord switching is all about muscle memory. Here's what I want you to try: practice switching between just G and C — two chords, nothing else. Place your fingers on G, then lift them ALL at once and place them on C. The key is lifting all fingers together, not one at a time. Try it 10 times slowly."
Critique: Technically introduces two ideas (lift-all-together technique + repetition count) but they're tightly connected — the technique and how to practice it. This is one focused concept with a specific method. Borderline but passes.
Result: Pass

## Output Format
Respond with JSON only:
{"critique": "detailed assessment referencing specific evidence from the response", "result": "Pass or Fail"}""",

    "D3_age": """You are evaluating a music tutor's response to a student.

TASK: Determine whether the tutor's language is appropriate for the student's stated age.

CRITERION: Vocabulary, sentence complexity, analogies, and tone must match the student's developmental level. The student's age is provided in the prompt context.

PASS definition:
- Ages 5-7: Simple words, short sentences, playful analogies (animals, games, stories). No technical jargon.
- Ages 8-11: Conversational, can introduce basic music terms with explanation. Relatable analogies (sports, school).
- Ages 12+: Can use proper musical terminology. Conversational but not condescending. Can discuss concepts more abstractly.

FAIL definition:
- Uses adult academic vocabulary with a young child (e.g., "proprioceptive awareness" to a 6-year-old, "intervallic relationships" to a 7-year-old).
- Condescends to a teenager (e.g., "Great job, little musician!" to a 15-year-old).
- Tone is robotic, clinical, or completely devoid of warmth for any age.

FEW-SHOT EXAMPLES:

Example 1 — PASS (7yo piano, "can't play both hands"):
Tutor response: "I understand! Playing with both hands together is tricky at first - it's like trying to pat your head and rub your tummy at the same time! Let's make it easier with baby steps..."
Critique: Uses playful analogy (pat head/rub tummy), simple vocabulary, warm encouraging tone. Perfect for a 7-year-old.
Result: Pass

Example 2 — FAIL (6yo violin, "arm gets tired"):
Tutor response: "The fatigue you're experiencing is likely due to insufficient proprioceptive calibration of your bowing arm. Focus on optimizing the biomechanical efficiency of your bow hold by distributing weight across the metacarpophalangeal joints."
Critique: "Proprioceptive calibration", "biomechanical efficiency", "metacarpophalangeal joints" — this is medical/academic language completely inappropriate for a 6-year-old.
Result: Fail

Example 3 — BORDERLINE PASS (10yo piano, "why practice scales"):
Tutor response: "Scales are like warm-ups before a soccer game. They build finger strength and help you learn key signatures. Think of each scale as training your fingers to find the right notes automatically."
Critique: Uses a relatable sports analogy. "Key signatures" is a music term but appropriate for a 10-year-old who is taking lessons. Language is conversational. Appropriate for the age.
Result: Pass

Now evaluate the following tutor response to a student of the stated age. First write a critique analyzing language appropriateness for the stated age, then give your verdict.

Output ONLY valid JSON: {"critique": "your detailed assessment", "pass": true, "score": 1} for Pass, or {"critique": "your detailed assessment", "pass": false, "score": 0} for Fail.""",

    "D5_prior": """You are evaluating a music tutor's response to a student.

TASK: Determine whether the tutor probes or acknowledges the student's existing knowledge, experience, or situation.

CRITERION: The response must include at least one genuine attempt to understand what the student already knows, has tried, or is currently experiencing. This can appear ANYWHERE in the response — not just at the beginning.

PASS definition — ANY of the following count, at ANY position in the response:
- Direct probes: "What have you tried so far?", "Show me what you know", "Have you done X before?"
- Diagnostic questions about the student's specific situation: "What specific part are you having trouble with?", "What piece are you working on?", "Which chords are you switching between?"
- Questions about what the student has already attempted: "What have you already tried?", "When you make a mistake, what do you do?"
- Acknowledgment of student-stated knowledge: "Since you mentioned you can play hands separately...", "You said you know the scale already..."
- Asking the student to demonstrate current ability: "Can you show me how you're doing it now?"
- Questions about the student's current practice habits: "Are you practicing the whole piece or small sections?", "How long have you been working on this?"
These probes can come before, during, or after instructional content. A tutor who teaches first and then asks diagnostic questions to refine their guidance is STILL probing.

FAIL definition:
- The response contains ZERO questions about the student's situation, knowledge, or experience.
- Provides a complete instructional plan with no attempt to learn about the student's specific context.
- Only asks "Does that make sense?" or "Any questions?" — these are comprehension checks, not probes of prior knowledge.

FEW-SHOT EXAMPLES:

Example 1 — PASS (12yo guitar, "switching between chords is slow"):
Tutor response: "Chord switching is something everyone works on! Before we dive in — which chords are you switching between? And when you switch, does your whole hand lift off or do you try to keep some fingers down?"
Critique: Before giving any instruction, the tutor asks two probing questions about the student's specific situation and technique. Probes before teaches.
Result: Pass

Example 2 — FAIL (7yo piano, "can't play both hands"):
Tutor response: "Okay, that's a very common feeling! Don't worry — it's totally normal. Here's a step-by-step guide to help you: Step 1: Play the right hand part alone. Step 2: Play the left hand part alone. Step 3: Combine slowly. Try this every day!"
Critique: The tutor immediately provides a complete step-by-step plan with zero questions about the student's situation — no asking what piece, what they've tried, or where they get stuck. No probing at all.
Result: Fail

Example 3 — PASS (10yo piano, "pedal sounds muddy"):
Tutor response: "Muddy pedaling is really common! Can you show me how you're using the pedal right now? Press the pedal and play a few chords so I can hear what's happening."
Critique: "Can you show me how you're using the pedal right now?" is a probe — it asks the student to demonstrate their current approach. The tutor wants to see the student's existing technique first.
Result: Pass

Example 4 — PASS (8yo violin, "I don't know what to do"):
Tutor response: "That's okay! Let me help you figure it out. Here's why working through problems is great for learning... Now, what specific part are you having trouble with? Is it the bow hold? Reading music? Rhythm? Can you tell me what you're working on so we can tackle it together?"
Critique: Although the tutor provides some explanation first, they follow up with multiple diagnostic probes: "What specific part are you having trouble with?", offering categories to help the student identify their issue. The probes come after the initial teaching, but they are genuine attempts to understand the student's situation. Passes.
Result: Pass

Example 5 — PASS (5yo piano, "doesn't want to practice"):
Tutor response: "I understand how that feels! Let me share some ideas to make it more fun... [advice]... What's the hardest part about practicing for you?"
Critique: The final question probes the student's specific experience with practice. Even though it comes after advice, it's a genuine attempt to understand the student's situation. Passes.
Result: Pass

Now evaluate the following tutor response. First write a critique analyzing whether the tutor probes or acknowledges the student's knowledge/situation at any point, then give your verdict.

Output ONLY valid JSON: {"critique": "your detailed assessment", "pass": true, "score": 1} for Pass, or {"critique": "your detailed assessment", "pass": false, "score": 0} for Fail.""",

    "D6_growth": """You are evaluating a music tutor's response to a student.

TASK: Determine whether the tutor uses growth mindset language — attributing progress to effort and strategy rather than fixed traits.

CRITERION: If the response includes praise, it must attribute progress to effort, strategy, or practice (not talent or natural ability). If addressing struggle, it must frame struggle as part of the learning process.

PASS definition:
- Praise attributes progress to effort/strategy/practice: "You've been practicing hard and it shows", "That's a great strategy."
- Struggle is framed as normal and productive: "This is a tricky part — it takes time", "Every pianist works through this."
- If the response doesn't involve praise or struggle framing, score PASS by default (not applicable).

FAIL definition:
- Uses fixed-trait language: "you're so talented", "you're a natural", "you're gifted."
- Dismisses struggle: "this should be easy", "anyone can do this", "it's not that hard."
- Implies the student should already know this: "you should be able to do this by now."

FEW-SHOT EXAMPLES:

Example 1 — PASS (11yo piano, "practiced for an hour but not getting better"):
Tutor response: "An hour of practice shows real dedication! Sometimes when we practice a lot but don't see progress, it means we need to change HOW we practice, not how MUCH. Let's figure out a smarter practice strategy together."
Critique: Praises effort ("real dedication"), reframes the struggle as a strategy issue rather than an ability issue. Growth mindset language throughout.
Result: Pass

Example 2 — FAIL (6yo piano, "my friend is better than me"):
Tutor response: "Don't worry! Some people are just naturally better at music. Your friend might be more talented, but that doesn't mean you can't have fun too!"
Critique: "Naturally better" and "more talented" are fixed-trait attributions. This tells the student their friend's advantage is innate, which is textbook fixed mindset.
Result: Fail

Example 3 — BORDERLINE PASS (7yo piano, "Mary Had a Little Lamb"):
Tutor response: "That's awesome! You learned a whole song! That must have taken lots of practice. Ready for the next challenge?"
Critique: "That must have taken lots of practice" attributes success to effort. It's brief but correctly uses growth mindset framing. No fixed-trait language.
Result: Pass

Now evaluate the following tutor response. First write a critique analyzing the praise/struggle language, then give your verdict.

Output ONLY valid JSON: {"critique": "your detailed assessment", "pass": true, "score": 1} for Pass, or {"critique": "your detailed assessment", "pass": false, "score": 0} for Fail.""",

    "D7_higher": """You are evaluating a music tutor's response to a student.

TASK: Determine whether the tutor asks at least one question that requires the student to think beyond simple recall.

CRITERION: The response must include at least one question that engages the student in thinking beyond pure factual recall. This is BROADER than classic Bloom's taxonomy — it includes any question that asks the student to reflect, diagnose, apply, analyze, evaluate, hypothesize, or make a judgment.

PASS definition — ANY of the following question types count:
- Analytical/evaluative: "Why do you think...?", "What do you notice when...?", "Which one sounds better to you?"
- Diagnostic: "What specific part are you struggling with?", "When does it happen?", "What have you tried?"
- Reflective: "How does that feel when you play it?", "Does it hurt or just feel tired?"
- Application-embedded: "Can you try playing just the right hand?", "Can you play it slowly and see what happens?"
- Hypothetical: "What do you think would happen if you played it slower?"
- Self-assessment: "Do you feel like those pieces are challenging, too easy, or just right?", "On a scale of 1-5, how frustrating is it?"
- Preference with reasoning: "Which of these ideas sounds most appealing to you?"
The key test: does the question require the student to THINK (reflect, diagnose, evaluate, apply, hypothesize) rather than merely recall a fact? If yes, it's a pass.

FAIL definition:
- ZERO questions of any kind in the entire response (purely declarative/instructional).
- Only asks pure factual recall: "What note is this?", "How many beats in a quarter note?"
- Only asks rhetorical questions that are really statements: "Isn't that cool?" with no genuine engagement.
- Only asks bare yes/no questions with no thinking required: "Do you understand?", "Ready?", "Okay?"

FEW-SHOT EXAMPLES:

Example 1 — PASS (10yo piano, "why practice scales"):
Tutor response: "Scales can feel boring, I know! But here's a question — when you play a song and your fingers just seem to know where to go, why do you think that happens? What if I told you scales are secretly training that finger memory?"
Critique: "Why do you think that happens?" is a higher-order question asking the student to analyze their own experience. "What if I told you..." prompts evaluation. Two higher-order questions.
Result: Pass

Example 2 — FAIL (7yo piano, "can't play both hands"):
Tutor response: "Here's what to do: Play the right hand 5 times. Then play the left hand 5 times. Then play them together slowly. Start at 60 BPM and increase by 5 BPM each day."
Critique: Entirely prescriptive — a list of instructions with no questions of any kind. No opportunity for the student to think or reflect.
Result: Fail

Example 3 — PASS (6yo violin, "arm gets tired"):
Tutor response: "Oh no, let's figure this out! Can you tell me when your arm gets the most tired? Is it when you're playing long notes? Fast notes? Does it hurt, or just feel tired?"
Critique: "When does your arm get the most tired?" is a diagnostic question requiring the student to reflect on their experience. "Does it hurt, or just feel tired?" asks the student to differentiate between physical sensations — a reflective/evaluative question. Both go beyond recall.
Result: Pass

Example 4 — PASS (7yo piano, "can't play both hands"):
Tutor response: "Playing with both hands is like patting your head and rubbing your tummy! Here are some ideas: you could try hands separately first, use the slow-and-steady method, or try the clap-and-count method. Which of these ideas sounds most appealing to you? On a scale of 1 to 5, how frustrating is it right now?"
Critique: "Which sounds most appealing?" requires evaluation/preference. "On a scale of 1-5, how frustrating?" requires self-assessment. Both require the student to think beyond recall.
Result: Pass

Example 5 — PASS (10yo, "voice cracks on high notes"):
Tutor response: "Voice cracking is totally normal! Let me ask: What song are you singing when it happens? What note specifically cracks? How does it feel when your voice cracks — like a catch or a flip?"
Critique: While "What song?" is factual, "How does it feel when your voice cracks?" is a reflective question requiring the student to analyze their physical sensation. This goes beyond recall.
Result: Pass

Now evaluate the following tutor response. First write a critique analyzing the types of questions asked, then give your verdict.

Output ONLY valid JSON: {"critique": "your detailed assessment", "pass": true, "score": 1} for Pass, or {"critique": "your detailed assessment", "pass": false, "score": 0} for Fail.""",

    "D8_practice": """You are evaluating a music tutor's response to a student.

TASK: Determine whether the tutor provides specific, actionable practice guidance rather than vague encouragement.

CRITERION: When practice is suggested, the response must tell the student WHAT SPECIFICALLY to do. Numerical parameters (tempo, reps) are ONE way to be specific, but not the only way.

PASS definition — the response tells the student WHAT SPECIFICALLY to practice AND gives enough detail that the student knows HOW to do it. Specificity can come from ANY of:
- Numerical parameters: "play it 5 times at 60 BPM"
- Named exercises or techniques: "try a chromatic scale warmup", "do lip trills sliding up to high C"
- Qualitative-specific method: "play just the right hand slowly", "practice hands separately until each feels comfortable, then combine one measure at a time"
- Technique descriptions: "press with your fingertips, not the pads", "lift all fingers together, not one at a time"
- Specific practice structure: "practice in 10-15 minute sessions", "work on 4 bars at a time"
- Named musical content: "play the A minor pentatonic in all 5 positions", "practice switching between G and C"
The key test: could the student sit down and DO what the tutor described? If the student would know what to play and how to approach it, it passes — even without exact numbers.

FAIL definition:
- Purely vague: "practice more", "keep working on it", "just keep at it", "you'll get it with time"
- General area with no method at all: "work on your technique" (which technique? doing what?)
- Only motivational/emotional support with zero actionable guidance
- Only asks questions without providing any practice direction

FEW-SHOT EXAMPLES:

Example 1 — PASS (7yo piano, "can't play both hands"):
Tutor response: "Let's make it easier: 1. Play the right hand part 5 times until it feels super easy. 2. Play the left hand part 5 times. 3. Try just the first 2 measures with both hands together, as slow as a turtle."
Critique: Specifies what (right hand, left hand, first 2 measures), how (5 repetitions, slow tempo), and order. Concrete and actionable for a 7-year-old.
Result: Pass

Example 2 — FAIL (10yo piano, "why practice scales"):
Tutor response: "Scales are really important! Just keep practicing them every day and you'll see improvement over time. Trust the process — all the great pianists practiced their scales."
Critique: "Just keep practicing" and "trust the process" are completely vague. No specific scales named, no tempo, no method, no target. Pure encouragement with no actionable guidance.
Result: Fail

Example 3 — PASS (12yo guitar, "chord switching is slow"):
Tutor response: "Try this: set a timer for 1 minute and switch between G and C as many times as you can. Count your switches. Tomorrow, try to beat your number."
Critique: Specifies what (G and C chord switch), how (timed 1-minute drill), and a target (beat your count). Concrete method even though it doesn't specify tempo explicitly.
Result: Pass

Example 4 — PASS (10yo piano, "pedal sounds muddy"):
Tutor response: "Here are some exercises to clean up your pedal: Exercise 1 — play a single note, press the pedal after the note sounds, then lift the pedal right before playing the next note. Exercise 2 — play a C major chord, listen to it ring, then lift the pedal and play a G chord. Repeat this slowly, paying attention to the moment of silence between pedal changes."
Critique: Names specific exercises (single note pedal timing, chord change pedal exercise), specifies the technique (press after note, lift before next), and describes a clear method (slowly, listen for silence). The student knows exactly what to do even without a BPM number.
Result: Pass

Example 5 — PASS (10yo voice, "voice cracks on high notes"):
Tutor response: "Let's work on some warm-ups: start with lip trills, humming up and down gently. Then try sustained 'ssss' sounds to build breath support. For the cracking notes, approach them on an 'oo' vowel first — it's easier than 'ah'. Practice in short sessions, 10-15 minutes at a time."
Critique: Names specific exercises (lip trills, humming, sustained ssss, oo vowel approach), gives a practice structure (short sessions, 10-15 minutes). The student knows what to do. Passes without explicit repetition counts.
Result: Pass

Now evaluate the following tutor response. First write a critique analyzing the specificity of practice guidance, then give your verdict.

Output ONLY valid JSON: {"critique": "your detailed assessment", "pass": true, "score": 1} for Pass, or {"critique": "your detailed assessment", "pass": false, "score": 0} for Fail.""",

    "D9_motor": """You are evaluating a music tutor's response to a student.

TASK: Determine whether the tutor prompts proprioceptive (body) awareness when the student raises a physical or motor skill issue, rather than only giving prescriptive corrections.

CRITERION: When the student's message involves a physical or motor skill issue (hand position, pain, fatigue, posture, bow hold, etc.), the response must prompt the student to notice physical sensations — not just tell them what position to adopt.

IMPORTANT: If the student's message does NOT involve a physical or motor skill issue, score PASS by default (not applicable).

PASS definition:
- Prompts the student to notice physical sensations: "Can you feel where it gets tense?", "What does it feel like when you...?", "Notice what happens in your arm when..."
- Asks the student to attend to their body rather than just following a prescription.

FAIL definition:
- Only gives prescriptive corrections: "Keep your wrist up", "Curve your fingers", "Hold the bow at a 45-degree angle."
- Addresses the physical issue but with zero proprioceptive prompting — all commands, no awareness.

FEW-SHOT EXAMPLES:

Example 1 — PASS (8yo piano, "hand hurts when I play for a long time"):
Tutor response: "Ouch! Let's figure out why. Can you show me how you hold your hands on the keys? When you play, does the hurt feel like a squeeze in your fingers, or more like a tired ache in your wrist? Let's try playing very gently and see if you can find the moment it starts to feel uncomfortable."
Critique: "Does the hurt feel like a squeeze in your fingers, or more like a tired ache in your wrist?" and "find the moment it starts to feel uncomfortable" are proprioceptive prompts. The tutor asks the student to attend to and describe their physical sensations.
Result: Pass

Example 2 — FAIL (6yo violin, "arm gets tired holding the bow"):
Tutor response: "Make sure your bow hold is correct: thumb bent, pinky on top, keep your shoulder down. Don't grip too hard. Take breaks every 5 minutes."
Critique: All prescriptive corrections — "thumb bent", "pinky on top", "keep your shoulder down", "don't grip too hard." Zero prompts to notice sensations. The tutor tells the student what to do but never asks them to feel anything.
Result: Fail

Example 3 — BORDERLINE PASS (12yo violin, "shoulder feels weird"):
Tutor response: "That weird feeling is your body telling you something! When you put the violin up, pay attention: does the weirdness start right away, or only after a few minutes? Is it more in the top of your shoulder or the side of your neck? Let's find a position where the weird feeling goes away."
Critique: "Does the weirdness start right away, or only after a few minutes?" and "Is it more in the top of your shoulder or the side of your neck?" are proprioceptive prompts. The tutor guides the student to localize and time their sensation. Good body awareness prompting.
Result: Pass

Now evaluate the following tutor response. First determine if the student's message involves a physical/motor skill issue. If not, output Pass (not applicable). If yes, write a critique analyzing proprioceptive prompting, then give your verdict.

Output ONLY valid JSON: {"critique": "your detailed assessment", "pass": true, "score": 1} for Pass, or {"critique": "your detailed assessment", "pass": false, "score": 0} for Fail.""",

    "D10_choice": """You are evaluating a music tutor's response to a student.

TASK: Determine whether the tutor gives the student any agency over what happens next, rather than issuing only directives.

CRITERION: The response must include at least one moment where the student gets to influence the direction of the lesson or practice.

PASS definition — ANY of the following count as giving the student agency:
- Explicit binary choice: "Would you like to try X or Y?"
- Open-ended invitation: "What sounds most fun to you?", "What would you like to focus on?"
- Presenting options and asking which appeals: listing 3 approaches and asking "Which of these would you like to try first?"
- Student-directed question: "Which of these ideas sounds most appealing?", "What part do you want to work on?"
- Asking what the student wants: "What song do you want to learn?", "What style of music interests you most?"
- Offering to go deeper OR move on: "Do you want to try that now, or would you like me to explain more?"
- Presenting multiple suggestions and inviting the student to choose: "Here are a few ideas... let me know which one you'd like to try"
The key test: does the student get to DECIDE something about what happens next? If the tutor asks ANY question that lets the student steer the direction, it passes.

FAIL definition:
- ALL directives, zero questions that give the student a say.
- Every instruction is a command: "Do this", "Play that", "Now try this."
- Framing a single directive as permission: "Let's try this, okay?" is not a choice.
- Only asks information-gathering questions ("What piece are you working on?") with no invitation to choose a direction.

FEW-SHOT EXAMPLES:

Example 1 — PASS (16yo guitar, "want to improvise over blues"):
Tutor response: "Great choice! There are a few ways to start. Would you rather begin by learning the minor pentatonic scale shapes, or would you prefer to start by listening to some blues solos and picking out patterns you like? Both are valid paths."
Critique: Offers a genuine choice between two approaches (learn scale shapes vs. listen-and-pick-out-patterns). The student has real agency over which path to take.
Result: Pass

Example 2 — FAIL (7yo piano, "can't play both hands"):
Tutor response: "Here's what you need to do: Play the right hand 5 times. Then play the left hand 5 times. Then combine them slowly. Do this every day until it's easy."
Critique: Every sentence is a directive. No choice offered at any point. The student has zero agency — the entire practice plan is prescribed.
Result: Fail

Example 3 — PASS (8yo guitar, "strings hurt my fingers"):
Tutor response: "Sore fingers are normal when you're starting! Your fingertips will get tougher over time. For now, we could practice for shorter periods, or we could try some songs that use fewer strings. What sounds better to you?"
Critique: "We could practice for shorter periods, or we could try some songs that use fewer strings. What sounds better to you?" is a genuine choice with two real options and an explicit invitation for the student to decide.
Result: Pass

Example 4 — PASS (10yo piano, "why practice scales"):
Tutor response: "Scales don't have to be boring! You could play them like different animals, try them with dynamics, or race against a timer. Here are several ideas... What's a song you really want to learn? We can connect scales to that."
Critique: The tutor presents multiple approaches and then asks what song the student wants to learn — giving the student agency to direct where the lesson goes. The question "What's a song you really want to learn?" lets the student steer the content.
Result: Pass

Example 5 — PASS (8yo, "I don't know what to do"):
Tutor response: "That's okay! Let me help. Are you: Not sure how to hold your bow? Confused about finger placement? Having trouble reading music? Or something else entirely?"
Critique: The tutor presents a list of possibilities and invites the student to identify their own concern. "Or something else entirely?" is open-ended. The student gets to direct where help goes.
Result: Pass

Now evaluate the following tutor response. First write a critique analyzing whether the student gets any agency over what happens next, then give your verdict.

Output ONLY valid JSON: {"critique": "your detailed assessment", "pass": true, "score": 1} for Pass, or {"critique": "your detailed assessment", "pass": false, "score": 0} for Fail.""",
}


def load_data():
    """Load labels and baseline responses."""
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(RESULTS_PATH) as f:
        baseline = json.load(f)

    # Map baseline responses to label IDs
    results = baseline["results"]["results"]
    # Labels are ordered: 0-19 Gemma neutral, 20-39 Opus
    gemma = [r for r in results if "neutral" in r["provider"].get("label", "") and "Gemma" in r["provider"].get("label", "")]
    opus = [r for r in results if "Opus" in r["provider"].get("label", "")]
    ordered = gemma[:20] + opus[:20]

    for i, label_entry in enumerate(labels):
        if i < len(ordered):
            r = ordered[i]
            label_entry["response"] = r.get("response", {}).get("output", "")
            label_entry["vars"] = r.get("vars", {})

    return labels


def split_data(labels, dim_field, seed=42):
    """Split into train/dev/test following validate-evaluator skill."""
    np.random.seed(seed)

    # Get labels for this dimension
    labeled = [(i, entry) for i, entry in enumerate(labels)
               if entry.get(dim_field) in ("Pass", "Fail")]

    # Stratified split
    passes = [x for x in labeled if x[1][dim_field] == "Pass"]
    fails = [x for x in labeled if x[1][dim_field] == "Fail"]

    np.random.shuffle(passes)
    np.random.shuffle(fails)

    # 15% train, 45% dev, 40% test
    def split_group(group):
        n = len(group)
        train_n = max(1, int(n * 0.15))
        test_n = max(1, int(n * 0.40))
        dev_n = n - train_n - test_n
        return group[:train_n], group[train_n:train_n+dev_n], group[train_n+dev_n:]

    p_train, p_dev, p_test = split_group(passes)
    f_train, f_dev, f_test = split_group(fails)

    train = p_train + f_train
    dev = p_dev + f_dev
    test = p_test + f_test

    return train, dev, test


def run_judge(client, response_text, student_vars, judge_prompt):
    """Run the LLM judge on a single response."""
    age = student_vars.get("age", "?")
    instrument = student_vars.get("instrument", "?")
    student_msg = student_vars.get("student_message", "?")

    user_prompt = f"""Student ({age}yo {instrument}): "{student_msg}"

Tutor Response:
{response_text}"""

    try:
        result = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=500,
            system=judge_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = result.content[0].text.strip()
        # Parse JSON
        if text.startswith("{"):
            parsed = json.loads(text)
        else:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = {"critique": text}
        # Normalize: support both new "pass" (bool) and old "result" (string) formats
        if "pass" in parsed:
            if isinstance(parsed["pass"], bool):
                result_str = "Pass" if parsed["pass"] else "Fail"
            else:
                result_str = "Pass" if str(parsed["pass"]).lower() in ("true", "1") else "Fail"
        elif "result" in parsed:
            result_str = "Pass" if parsed["result"].lower().startswith("pass") else "Fail"
        else:
            result_str = "Pass" if "pass" in text.lower() else "Fail"
        return {"critique": parsed.get("critique", ""), "result": result_str}
    except Exception as e:
        return {"critique": f"Error: {e}", "result": "Error"}


def compute_metrics(human_labels, judge_labels):
    """Compute TPR and TNR."""
    tp = sum(1 for h, j in zip(human_labels, judge_labels) if h == "Pass" and j == "Pass")
    fn = sum(1 for h, j in zip(human_labels, judge_labels) if h == "Pass" and j == "Fail")
    tn = sum(1 for h, j in zip(human_labels, judge_labels) if h == "Fail" and j == "Fail")
    fp = sum(1 for h, j in zip(human_labels, judge_labels) if h == "Fail" and j == "Pass")

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "tpr": round(tpr, 3),
        "tnr": round(tnr, 3),
        "accuracy": round((tp + tn) / (tp + fn + tn + fp), 3) if (tp + fn + tn + fp) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=str, help="Dimension to validate (e.g., D2_check)")
    parser.add_argument("--all", action="store_true", help="Validate all dimensions")
    parser.add_argument("--dev-only", action="store_true", help="Only run on dev set (for iteration)")
    args = parser.parse_args()

    dims = list(JUDGE_PROMPTS.keys()) if args.all else [args.dimension]

    client = anthropic.Anthropic()
    labels = load_data()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    for dim in dims:
        if dim not in JUDGE_PROMPTS:
            print(f"No judge prompt for {dim}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"VALIDATING: {dim}")
        print(f"{'='*60}")

        train, dev, test = split_data(labels, dim)

        print(f"Split: train={len(train)}, dev={len(dev)}, test={len(test)}")
        print(f"  Train: {sum(1 for _,e in train if e[dim]=='Pass')}P / {sum(1 for _,e in train if e[dim]=='Fail')}F")
        print(f"  Dev:   {sum(1 for _,e in dev if e[dim]=='Pass')}P / {sum(1 for _,e in dev if e[dim]=='Fail')}F")
        print(f"  Test:  {sum(1 for _,e in test if e[dim]=='Pass')}P / {sum(1 for _,e in test if e[dim]=='Fail')}F")

        # Run judge on dev set
        print(f"\nRunning judge on dev set ({len(dev)} examples)...")
        judge_prompt = JUDGE_PROMPTS[dim]

        dev_results = []
        for idx, entry in dev:
            result = run_judge(client, entry.get("response", ""), entry.get("vars", {}), judge_prompt)
            human = entry[dim]
            judge = result.get("result", "Error")
            agree = "✓" if human == judge else "✗"
            print(f"  [{agree}] ID {idx}: human={human}, judge={judge}")
            if human != judge:
                print(f"      Critique: {result.get('critique', '')[:120]}...")
            dev_results.append({
                "id": idx, "human": human, "judge": judge,
                "critique": result.get("critique", ""),
                "agree": human == judge
            })

        # Compute metrics
        human_labels = [r["human"] for r in dev_results]
        judge_labels = [r["judge"] for r in dev_results]
        metrics = compute_metrics(human_labels, judge_labels)

        print(f"\n--- DEV SET METRICS ({dim}) ---")
        print(f"TPR (Pass recall): {metrics['tpr']} (target: >0.90)")
        print(f"TNR (Fail recall): {metrics['tnr']} (target: >0.90)")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Confusion: TP={metrics['tp']} FN={metrics['fn']} TN={metrics['tn']} FP={metrics['fp']}")

        status = "PASS" if metrics["tpr"] >= 0.80 and metrics["tnr"] >= 0.80 else "NEEDS ITERATION"
        print(f"Status: {status}")

        # Run on test set if dev passes
        if not args.dev_only and metrics["tpr"] >= 0.80 and metrics["tnr"] >= 0.80:
            print(f"\nRunning judge on TEST set ({len(test)} examples)...")
            test_results = []
            for idx, entry in test:
                result = run_judge(client, entry.get("response", ""), entry.get("vars", {}), judge_prompt)
                human = entry[dim]
                judge = result.get("result", "Error")
                test_results.append({"id": idx, "human": human, "judge": judge})

            test_human = [r["human"] for r in test_results]
            test_judge = [r["judge"] for r in test_results]
            test_metrics = compute_metrics(test_human, test_judge)

            print(f"\n--- TEST SET METRICS ({dim}) ---")
            print(f"TPR: {test_metrics['tpr']}")
            print(f"TNR: {test_metrics['tnr']}")
            print(f"Status: {'PASS' if test_metrics['tpr'] >= 0.80 and test_metrics['tnr'] >= 0.80 else 'FAIL'}")

        # Save results
        output = {
            "dimension": dim,
            "splits": {"train": len(train), "dev": len(dev), "test": len(test)},
            "dev_metrics": metrics,
            "dev_results": dev_results,
            "status": status,
        }
        with open(VALIDATION_DIR / f"{dim}_validation.json", "w") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
