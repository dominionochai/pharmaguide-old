"""Optional image-to-herb identification with safe fuzzy database lookup."""
from __future__ import annotations

import base64
import difflib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

from pharmaguide.herbs_db import ALIASES, HERBS_DB, normalize_name


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "herb_classifier.pth"
LABEL_MAP_PATH = ROOT / "models" / "herb_label_map.json"
_TRAINED_RUNTIME: tuple[Any, Any, dict[int, str]] | None = None


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


def _load_label_map() -> dict[int, str] | None:
    try:
        raw = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            labels = {index: str(name) for index, name in enumerate(raw)}
        elif isinstance(raw, dict):
            labels = {int(index): str(name) for index, name in raw.items()}
        else:
            return None
        if not labels or set(labels) != set(range(len(labels))) or any(not name.strip() for name in labels.values()):
            return None
        return labels
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _load_trained_runtime():
    """Load the optional classifier only when its artifacts are present."""
    global _TRAINED_RUNTIME
    if _TRAINED_RUNTIME is not None:
        return _TRAINED_RUNTIME
    if not MODEL_PATH.is_file() or not LABEL_MAP_PATH.is_file():
        return None

    labels = _load_label_map()
    if labels is None:
        return None
    try:
        # These imports intentionally stay inside the loader: the API must boot
        # when the optional, heavyweight training dependencies are unavailable.
        import torch
        import torchvision

        try:
            checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        except TypeError:  # Older torch versions do not accept weights_only.
            checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        if not isinstance(checkpoint, dict):
            return None
        architecture = str(checkpoint.get("architecture", "mobilenet_v3_small")).lower().replace("-", "_")
        if architecture not in {"mobilenet_v3_small", "mobilenetv3_small"}:
            return None
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
        if not isinstance(state_dict, dict):
            return None
        state_dict = {
            (key[7:] if str(key).startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        num_classes = int(checkpoint.get("num_classes", len(labels)))
        if num_classes != len(labels):
            return None

        try:
            model = torchvision.models.mobilenet_v3_small(weights=None)
        except (TypeError, AttributeError):
            model = torchvision.models.mobilenet_v3_small(pretrained=False)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        _TRAINED_RUNTIME = (torch, model, transform, labels)
        return _TRAINED_RUNTIME
    except Exception:
        # A corrupt/incompatible artifact, missing optional package, or model
        # construction failure must never prevent the API from serving.
        return None


def _identify_with_trained_model(image_path: str | Path) -> dict[str, Any] | None:
    runtime = _load_trained_runtime()
    if runtime is None:
        return None
    torch, model, transform, labels = runtime
    try:
        from PIL import Image

        with Image.open(image_path) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        position = int(torch.argmax(probabilities).item())
        herb_name = labels.get(position)
        if not herb_name:
            return None
        return {
            "identified": True,
            "herb": herb_name,
            "confidence": float(probabilities[position].item()),
            "method": "trained_model",
        }
    except Exception:
        return None


def _identify_with_vision_api(image_path: str | Path, api_key: str | None) -> str | None:
    key = api_key or os.getenv("HERB_VISION_API_KEY")
    if not key:
        return None
    endpoint = os.getenv("HERB_VISION_API_URL", "https://api.openai.com/v1/chat/completions")
    try:
        path = Path(image_path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        prompt = (
            "Identify the medicinal herb, plant, or pack label in this image. "
            "Return the common name only, with no explanation."
        )
        payload = {
            "model": os.getenv("HERB_VISION_MODEL", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ],
                }
            ],
            "max_tokens": 20,
        }
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        text = body.get("choices", [{}])[0].get("message", {}).get("content")
        if not text and body.get("candidates"):
            text = body["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text")
        return str(text).strip().splitlines()[0] if text else None
    except (OSError, requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None


def identify_from_image(image_path: str | Path, api_key: str | None = None) -> dict[str, Any] | None:
    """Identify an image with the local model first, then the optional vision API."""
    trained = _identify_with_trained_model(image_path)
    if trained is not None:
        return trained
    vision_name = _identify_with_vision_api(image_path, api_key)
    if vision_name:
        return {"identified": True, "herb": vision_name, "method": "vision_api"}
    return None
