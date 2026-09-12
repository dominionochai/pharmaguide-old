"""FastAPI service for pairwise medication interaction flags."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from data.chem import canonical_smiles, fingerprint_array, load_cache, save_cache

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "interaction_model.joblib"
LABEL_PATH = ROOT / "models" / "severity_labels.json"
DEFAULT_LABELS = ["minor", "moderate", "major", "contraindicated"]
SEVERITY_ORDER = {label: index for index, label in enumerate(DEFAULT_LABELS)}

app = FastAPI(title="pharmaguide")
_model = None
_labels = DEFAULT_LABELS


def _load_model():
    global _model, _labels
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        if LABEL_PATH.exists():
            _labels = json.loads(LABEL_PATH.read_text())
    return _model


class MedicationRequest(BaseModel):
    medications: list[str] = Field(..., min_length=2)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _load_model() is not None}


@app.post("/detect")
def detect(request: MedicationRequest):
    model = _load_model()
    if model is None:
        raise HTTPException(status_code=400, detail="Interaction model is not trained yet; run train.py first")
    medications = [name.strip() for name in request.medications if name.strip()]
    if len(medications) < 2:
        raise HTTPException(status_code=400, detail="Provide at least two medication names")

    cache = load_cache()
    smiles = {name: canonical_smiles(name, cache) for name in dict.fromkeys(medications)}
    save_cache(cache)
    missing = [name for name, value in smiles.items() if not value]
    if missing:
        raise HTTPException(status_code=400, detail=f"PubChem could not resolve: {', '.join(missing)}")

    results = []
    for drug_a, drug_b in combinations(dict.fromkeys(medications), 2):
        try:
            first = np.asarray(fingerprint_array(smiles[drug_a]), dtype=np.uint8)
            second = np.asarray(fingerprint_array(smiles[drug_b]), dtype=np.uint8)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        vector = np.bitwise_xor(first, second).reshape(1, -1)
        probabilities = model.predict_proba(vector)[0]
        position = int(np.argmax(probabilities))
        class_number = int(model.classes_[position])
        severity = _labels[class_number]
        results.append({"drug_a": drug_a, "drug_b": drug_b, "severity": severity, "confidence": round(float(probabilities[position]), 2)})
    results.sort(key=lambda item: SEVERITY_ORDER.get(item["severity"], -1), reverse=True)
    return {"interactions": results, "summary": f"{len(results)} risky interactions found"}
