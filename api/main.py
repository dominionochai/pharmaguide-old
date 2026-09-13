"""FastAPI service for medicine and herbal interaction flags."""
from __future__ import annotations

import hashlib
import json
import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any

from api.demo_sheet import tell_your_doctor_sheet
from api.herb_detector import identify_from_image, lookup
from pharmaguide.herbs_db import HERBS_DB, medication_flags
from pharmaguide.medicines_db import MEDICINES_DB, medicine_count, resolve_medicine

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "interaction_model.joblib"
LABEL_PATH = ROOT / "models" / "severity_labels.json"
DEFAULT_LABELS = ["minor", "moderate", "major", "contraindicated"]

app = FastAPI(title="pharmaguide")
_model: Any = None
_labels = DEFAULT_LABELS


class MedicationRequest(BaseModel):
    medications: list[str] = Field(..., min_length=2, max_length=100)


def _load_model() -> Any:
    global _model, _labels
    if _model is not None:
        return _model
    if not MODEL_PATH.is_file():
        return None
    try:
        _model = joblib.load(MODEL_PATH)
        if LABEL_PATH.is_file():
            loaded = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and loaded:
                _labels = [str(item) for item in loaded]
        return _model
    except Exception:
        _model = None
        return None


def _stable_fingerprint(name: str) -> np.ndarray:
    """Create a stable 1,024-bit representation for model inference."""
    bits = np.zeros(1024, dtype=np.uint8)
    digest = hashlib.sha256(name.casefold().encode("utf-8")).digest()
    for offset in range(0, len(digest), 2):
        bits[int.from_bytes(digest[offset:offset + 2], "little") % 1024] = 1
    return bits


def _model_flag(first: str, second: str, model: Any) -> dict | None:
    if model is None:
        return None
    try:
        features = np.bitwise_xor(
            _stable_fingerprint(first), _stable_fingerprint(second)
        ).reshape(1, -1)
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features))[0]
            index = int(np.argmax(probabilities))
            confidence = round(float(probabilities[index]), 2)
        else:
            prediction = np.asarray(model.predict(features)).reshape(-1)
            index = int(prediction[0])
            confidence = 1.0
        classes = list(getattr(model, "classes_", range(len(_labels))))
        if classes and 0 <= index < len(classes):
            label_index = int(classes[index]) if isinstance(classes[index], (int, np.integer)) else index
        else:
            label_index = index
        severity = _labels[label_index] if 0 <= label_index < len(_labels) else _labels[0]
        return {
            "drug_a": first,
            "drug_b": second,
            "severity": severity,
            "confidence": confidence,
            "reason": "Severity estimated by the trained interaction model.",
            "source": "trained_model",
        }
    except Exception:
        return None


def _database_flag(first: str, second: str) -> dict | None:
    record = MEDICINES_DB.get(first)
    if record is None:
        return None
    for flag in record.get("known_interactions", []):
        if flag.get("with") == second:
            return {
                "drug_a": first,
                "drug_b": second,
                "severity": flag["severity"],
                "confidence": round(float(flag.get("confidence", 0.95)), 2),
                "reason": flag["reason"],
                "source": "medicine_database",
            }
    return None


def _resolve_items(names: list[str]) -> tuple[list[dict], list[str]]:
    matched: list[dict] = []
    unknown: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw in names:
        supplied = str(raw).strip()
        if not supplied:
            continue
        medicine = resolve_medicine(supplied)
        if medicine:
            key = ("medicine", medicine)
            if key not in seen:
                matched.append({"name": medicine, "type": "medicine", "details": MEDICINES_DB[medicine]})
                seen.add(key)
            continue
        herb, confidence = lookup(supplied)
        if herb is not None:
            herb_name = herb["common_name"]
            key = ("herb", herb_name)
            if key not in seen:
                matched.append({"name": herb_name, "type": "herb", "confidence": confidence, "details": herb})
                seen.add(key)
        else:
            unknown.append(supplied)
    return matched, unknown


def _flags_for_items(items: list[dict], model: Any) -> list[dict]:
    flags: list[dict] = []
    for index, first in enumerate(items):
        for second in items[index + 1:]:
            first_name, second_name = first["name"], second["name"]
            if first["type"] == "medicine" and second["type"] == "medicine":
                database_flag = _database_flag(first_name, second_name) or _database_flag(second_name, first_name)
                if database_flag:
                    flags.append(database_flag)
                model_flag = _model_flag(first_name, second_name, model)
                if model_flag:
                    flags.append(model_flag)
            elif first["type"] == "herb" and second["type"] == "medicine":
                flags.extend(_herb_flags(first, second_name))
            elif first["type"] == "medicine" and second["type"] == "herb":
                flags.extend(_herb_flags(second, first_name))
    return flags


def _herb_flags(herb_item: dict, medicine: str) -> list[dict]:
    flags = medication_flags(herb_item["details"], [medicine])
    return [
        {
            "drug_a": herb_item["name"],
            "drug_b": item["medication"],
            "severity": item["severity"],
            "confidence": 0.8,
            "reason": item["note"],
            "source": "herb_database",
        }
        for item in flags
    ]


def _sort_flags(flags: list[dict]) -> list[dict]:
    order = {"contraindicated": 4, "major": 3, "moderate": 2, "minor": 1}
    return sorted(flags, key=lambda item: order.get(item.get("severity", ""), 0), reverse=True)


@app.get("/health")
def health() -> dict:
    model = _load_model()
    return {"status": "ok", "model_loaded": model is not None, "medicine_count": medicine_count()}


@app.post("/detect")
def detect(request: MedicationRequest) -> dict:
    if len({str(item).strip().casefold() for item in request.medications if str(item).strip()}) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two distinct medicine or herb names")
    items, unknown = _resolve_items(request.medications)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unrecognized medicine or herb: {', '.join(unknown)}")
    if len(items) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two recognized medicine or herb names")
    model = _load_model()
    flags = _sort_flags(_flags_for_items(items, model))
    return {
        "interactions": flags,
        "summary": f"{len(flags)} interaction flags found",
        "matched": items,
        "mode": "model_and_database" if model is not None else "database_only",
        "model_loaded": model is not None,
        "tell_your_doctor": tell_your_doctor_sheet(flags),
    }


def _medication_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (TypeError, ValueError, json.JSONDecodeError):
        pass
    return [part.strip() for part in value.split(",") if part.strip()]


@app.post("/detect-herb")
async def detect_herb(
    image: UploadFile | None = File(default=None),
    name: str | None = Form(default=None),
    medications: str | None = Form(default=None),
) -> dict:
    identified_name = name.strip() if name and name.strip() else None
    temporary: Path | None = None
    method = "manual"
    trained_confidence: float | None = None
    try:
        if not identified_name and image is not None:
            suffix = Path(image.filename or "herb.jpg").suffix or ".jpg"
            temporary = Path(__import__("tempfile").NamedTemporaryFile(delete=False, suffix=suffix).name)
            temporary.write_bytes(await image.read())
            identification = identify_from_image(temporary)
            if identification:
                identified_name = identification.get("herb") or identification.get("name")
                method = identification.get("method", "image")
                trained_confidence = identification.get("confidence")
        if not identified_name:
            return {"identified": False, "method": method, "message": "could not identify — please type the herb name", "herbs": list(HERBS_DB)}
        record, match_confidence = lookup(identified_name)
        if record is None:
            return {"identified": False, "method": method, "message": "could not identify — please type the herb name", "herbs": list(HERBS_DB)}
        meds = _medication_list(medications)
        canonical_meds = [resolve_medicine(item) or item.casefold() for item in meds]
        flags = _herb_flags({"name": record["common_name"], "details": record}, canonical_meds)
        confidence = trained_confidence if method == "trained_model" and trained_confidence is not None else match_confidence
        return {
            "identified": True,
            "herb": {**record, "confidence": float(confidence)},
            "interactions": flags,
            "method": method,
            "tell_your_doctor": tell_your_doctor_sheet(flags),
        }
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


@app.get("/herbs")
def herbs() -> list[dict]:
    return list(HERBS_DB.values())
