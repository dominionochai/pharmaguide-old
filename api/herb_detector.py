"""Optional image-to-herb identification with safe fuzzy database lookup."""
from __future__ import annotations

import base64
import difflib
import mimetypes
import os
from pathlib import Path

import requests

from pharmaguide.herbs_db import ALIASES, HERBS_DB, normalize_name


def lookup(name: str):
    query = normalize_name(name)
    key = ALIASES.get(query, query)
    if key in HERBS_DB and key != "unknown herb":
        return HERBS_DB[key], 1.0
    candidates = [item for item in HERBS_DB if item != "unknown herb"]
    folded = {item: normalize_name(item) for item in candidates}
    close = difflib.get_close_matches(query, list(folded.values()), n=1, cutoff=0.55)
    if close:
        selected = next(item for item, value in folded.items() if value == close[0])
        score = difflib.SequenceMatcher(None, query, close[0]).ratio()
        return HERBS_DB[selected], round(min(score, 0.99), 2)
    for item, value in folded.items():
        if query and (query in value or value in query):
            return HERBS_DB[item], 0.75
    return None, 0.0


def identify_from_image(image_path: str | Path, api_key: str | None = None) -> str | None:
    """Ask an optional OpenAI-compatible/Gemini-compatible endpoint for a name only."""
    key = api_key or os.getenv("HERB_VISION_API_KEY")
    if not key:
        return None
    endpoint = os.getenv("HERB_VISION_API_URL", "https://api.openai.com/v1/chat/completions")
    try:
        path = Path(image_path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        prompt = "Identify the medicinal herb, plant, or pack label in this image. Return the common name only, with no explanation."
        payload = {"model": os.getenv("HERB_VISION_MODEL", "gpt-4o-mini"), "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}]}], "max_tokens": 20}
        response = requests.post(endpoint, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=payload, timeout=30)
        response.raise_for_status()
        body = response.json()
        text = body.get("choices", [{}])[0].get("message", {}).get("content")
        if not text and body.get("candidates"):
            text = body["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text")
        return str(text).strip().splitlines()[0] if text else None
    except (OSError, requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None
