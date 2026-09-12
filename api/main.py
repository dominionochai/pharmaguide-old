"""FastAPI service for pairwise medication and herbal interaction flags."""
from __future__ import annotations

import json
import os
import tempfile
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.herb_detector import identify_from_image, lookup
from data.chem import canonical_smiles, fingerprint_array, load_cache, save_cache
from pharmaguide.herbs_db import HERBS_DB, medication_flags


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "interaction_model.joblib"
LABEL_PATH = ROOT / "models" / "severity_labels.json"
DEFAULT_LABELS = ["minor", "moderate", "major", "contraindicated"]
app = FastAPI(title="pharmaguide")
_model = None
_labels = DEFAULT_LABELS


def _load_model():
    global _model, _labels
    if _model is None and MODEL_PATH.exists():
        try:
            _model = joblib.load(MODEL_PATH)
            if LABEL_PATH.exists():
                loaded = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, list) and loaded:
                    _labels = loaded
        except Exception:
            _model = None
    return _model


class MedicationRequest(BaseModel):
    medications: list[str] = Field(..., min_length=2, max_length=100)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _load_model() is not None}


@app.post("/detect")
def detect(request: MedicationRequest):
    model = _load_model()
    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Interaction model is not trained yet; run train.py first",
        )

    medications = [name.strip() for name in request.medications if name.strip()]
    if len(medications) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two medication names")
    if len(set(medications)) < 2:
        raise HTTPException(
            status_code=400, detail="Provide at least two distinct medication names"
        )

    cache = load_cache()
    smiles = {
        name: canonical_smiles(name, cache)
        for name in dict.fromkeys(medications)
    }
    save_cache(cache)
    missing = [name for name, value in smiles.items() if not value]
    if missing:
        raise HTTPException(
            status_code=400, detail=f"PubChem could not resolve: {', '.join(missing)}"
        )

    results = []
    for drug_a, drug_b in combinations(medications, 2):
        try:
            first = np.asarray(fingerprint_array(smiles[drug_a]), dtype=np.uint8)
            second = np.asarray(fingerprint_array(smiles[drug_b]), dtype=np.uint8)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        probabilities = model.predict_proba(
            np.bitwise_xor(first, second).reshape(1, -1)
        )[0]
        position = int(np.argmax(probabilities))
        classes = getattr(model, "classes_", [])
        class_id = int(classes[position]) if len(classes) > position else position
        severity = (
            _labels[class_id] if 0 <= class_id < len(_labels) else DEFAULT_LABELS[position]
        )
        results.append(
            {
                "drug_a": drug_a,
                "drug_b": drug_b,
                "severity": severity,
                "confidence": round(float(probabilities[position]), 2),
            }
        )

    order = {label: index for index, label in enumerate(DEFAULT_LABELS)}
    results.sort(
        key=lambda item: order.get(item["severity"], -1), reverse=True
    )
    return {
        "interactions": results,
        "summary": f"{len(results)} risky interactions found",
    }


def _unknown_herb(method: str = "manual"):
    return {
        "identified": False,
        "method": method if method in {"trained_model", "vision_api", "manual"} else "manual",
        "message": "could not identify — please type the herb name",
        "herbs": list(HERBS_DB.keys()),
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
):
    identified_name = name.strip() if name and name.strip() else None
    temporary = None
    method = "manual"
    trained_confidence = None
    try:
        if not identified_name and image is not None:
            contents = await image.read()
            if contents:
                suffix = Path(image.filename or "herb.jpg").suffix or ".jpg"
                temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temporary.write(contents)
                temporary.close()
                try:
                    identification = identify_from_image(
                        temporary.name, os.getenv("HERB_VISION_API_KEY")
                    )
                except Exception:
                    identification = None
                if isinstance(identification, dict):
                    identified_name = identification.get("herb") or identification.get("name")
                    candidate_method = identification.get("method")
                    if candidate_method in {"trained_model", "vision_api"}:
                        method = candidate_method
                    if candidate_method == "trained_model":
                        try:
                            trained_confidence = float(identification.get("confidence"))
                        except (TypeError, ValueError):
                            trained_confidence = None
        if not identified_name:
            return _unknown_herb(method)

        record, match_confidence = lookup(identified_name)
        if record is None:
            return _unknown_herb(method)
        meds = _medication_list(medications)
        flags = medication_flags(record, meds)
        confidence = (
            trained_confidence
            if method == "trained_model" and trained_confidence is not None
            else match_confidence
        )
        return {
            "identified": True,
            "herb": {**record, "confidence": float(confidence)},
            "interactions": flags,
            "method": method,
            "tell_your_doctor": f"Oga doctor: dis one na {record['common_name']}, e fit interact with some of my medicines — abeg check am.",
        }
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary.name)
            except OSError:
                pass


@app.get("/herbs")
def herbs():
    return list(HERBS_DB.values())
