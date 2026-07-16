"""Grounded LLM insight via the Google Gemini API.

The model narrates, it never computes. Every number it speaks is one Python has
already calculated and passed into its context, and it is asked to return JSON
pairing each claim with the field that supports it so the output is verifiable.
"""
from __future__ import annotations

import json

import pandas as pd

from src import config, store

SYSTEM = (
    "You are a markets analyst writing a short note on the Brent-WTI crude spread. "
    "You will receive a JSON payload of already-computed numbers and recent annotations. "
    "Rules: use ONLY the numbers in the payload, never compute or infer new figures, and "
    "never invent values. Mark uncertainty honestly. Do not use em dashes. "
    "Return ONLY valid JSON, with no preamble and no markdown fences, in this exact shape: "
    '{"summary": string, "claims": [{"claim": string, "support": string}], '
    '"caveats": [string]}. Each claim.support must name a field and value from the payload.'
)


def build_payload(con, extra: dict | None = None) -> dict:
    """Assemble the grounding context from the latest stored spread and annotations.

    `extra` carries optional pre-computed context (for example a Marketaux news
    sentiment snapshot). Every value in it must already be a finished number or
    label; the model is still forbidden from doing arithmetic on it.
    """
    spread = store.read_spread(con)
    if spread.empty:
        return {}
    valid = spread.dropna(subset=["zscore"])
    if valid.empty:
        return {}
    latest = valid.iloc[-1]
    recent = store.read_annotations(con).sort_values("ts").tail(5)

    def num(x):
        return None if pd.isna(x) else round(float(x), 3)

    payload = {
        "as_of": str(latest["ts"]),
        "spread": num(latest["spread"]),
        "zscore": num(latest["zscore"]),
        "roll_mean": num(latest["roll_mean"]),
        "roll_std": num(latest["roll_std"]),
        "pct_range": num(latest["pct_range"]),
        "recent_annotations": recent[["ts", "text"]].astype(str).to_dict("records"),
    }
    if extra:
        payload.update(extra)
    return payload


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```"))
    if "{" in text and "}" in text:
        return text[text.find("{"): text.rfind("}") + 1]
    return text


def generate(con, extra: dict | None = None) -> dict:
    """Produce and persist a grounded insight note for the latest spread state."""
    payload = build_payload(con, extra=extra)
    if not payload:
        return {"summary": "No spread data available yet.", "claims": [], "caveats": []}
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set; cannot generate insight.")

    from google import genai
    from google.genai import types

    # The system prompt carries the grounding rules; the payload is the only data
    # the model may reference. Temperature is kept low so the note stays factual.
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=json.dumps(payload),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0.2,
            max_output_tokens=800,
            response_mime_type="application/json",
        ),
    )
    text = response.text or ""
    try:
        note = json.loads(_extract_json(text))
    except json.JSONDecodeError:
        note = {"summary": text.strip(), "claims": [],
                "caveats": ["Model returned unparseable JSON."]}

    store.write_insight(
        con, payload["as_of"], json.dumps(payload),
        json.dumps(note), config.GEMINI_MODEL,
    )
    return note


NARRATE_SYSTEM = (
    "You are writing a short plain-English caption for a panel on a financial "
    "dashboard, aimed at a reader who isn't a quant. You will receive an "
    "instruction describing what the panel shows and a JSON payload of "
    "already-computed numbers. Rules: use ONLY the numbers in the payload, "
    "never compute or infer new figures, never invent values. If the payload "
    "doesn't support a clear statement, say so briefly instead of guessing. "
    "Two to three sentences, plain text, no markdown, no bullet points, no "
    "em dashes."
)


def narrate_metric(payload: dict, instruction: str) -> str:
    """Produce a short, plain-English explanation of an already-computed metric.

    Same non-negotiable rule as generate(): the model narrates numbers it is
    handed, it never computes them. Returns plain text, not JSON, for inline
    captions under charts. Raises if GEMINI_API_KEY isn't set; callers should
    catch and degrade gracefully.
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set; cannot narrate.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=f"Instruction: {instruction}\n\nData: {json.dumps(payload)}",
        config=types.GenerateContentConfig(
            system_instruction=NARRATE_SYSTEM,
            temperature=0.2,
            max_output_tokens=150,
        ),
    )
    return (response.text or "").strip()
