"""Grounded conversational Q&A over the tracker's already-computed numbers.

Same non-negotiable rule as insights.py: the model narrates, it never
computes. Every number it can reference is handed to it in the grounding
payload; it is instructed to use only those numbers and to say so honestly
when something isn't in the payload.
"""
from __future__ import annotations

import json

from src import config, insights

SYSTEM = (
    "You are a Q&A assistant embedded in the Brent-WTI Spread Tracker dashboard. "
    "You will receive a JSON payload of already-computed numbers (spread, z-score, "
    "recent annotations, correlations, news sentiment, options positioning) and the "
    "recent conversation. Rules: use ONLY the numbers in the payload, never compute "
    "or infer new figures, and never invent values. If the answer isn't in the "
    "payload, say so honestly instead of guessing. Mark uncertainty. Keep answers "
    "short, a few sentences. Do not use em dashes. Plain text only, no markdown fences."
)


def build_payload(con, extra: dict | None = None) -> dict:
    """Assemble the same grounding payload used for the insight note, plus extras."""
    payload = insights.build_payload(con)
    if extra:
        payload.update(extra)
    return payload


def answer(con, question: str, history: list[dict], extra: dict | None = None) -> str:
    """Answer one chat turn, grounded in fresh computed numbers plus prior turns.

    `history` is a list of {"role": "user"|"assistant", "content": str} dicts for
    prior turns in this session (not including the current question).
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set; cannot answer.")

    payload = build_payload(con, extra=extra)

    from google import genai
    from google.genai import types

    contents = []
    for turn in history:
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=turn["content"])]))
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=f"Grounding data: {json.dumps(payload)}\n\nQuestion: {question}")],
    ))

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.2,
            max_output_tokens=400,
        ),
    )
    return (response.text or "").strip()
