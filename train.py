"""Train the CPU-only DDInter severity classifier, quickly and defensively."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from data.chem import canonical_smiles, fingerprint_array, load_cache, save_cache

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "ddinter.csv"
MODELS = ROOT / "models"
LABELS = ["minor", "moderate", "major", "contraindicated"]


def _fallback_frame() -> pd.DataFrame:
    return pd.DataFrame([
        ("aspirin", "warfarin", "major"), ("metformin", "alcohol", "major"),
        ("ibuprofen", "aspirin", "moderate"), ("sildenafil", "nitroglycerin", "contraindicated"),
        ("amlodipine", "simvastatin", "moderate"), ("digoxin", "furosemide", "major"),
        ("paracetamol", "warfarin", "moderate"), ("lisinopril", "potassium", "moderate"),
    ], columns=["drug_a", "drug_b", "severity"])


def _features(frame: pd.DataFrame, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    for row in frame.itertuples(index=False):
        left, right = canonical_smiles(row.drug_a, cache), canonical_smiles(row.drug_b, cache)
        if not left or not right or row.severity not in LABELS:
            continue
        try:
            features.append(np.bitwise_xor(np.asarray(fingerprint_array(left), dtype=np.uint8), np.asarray(fingerprint_array(right), dtype=np.uint8)))
            targets.append(LABELS.index(row.severity))
        except ValueError:
            continue
    return np.asarray(features, dtype=np.uint8), np.asarray(targets, dtype=np.int64)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", default=True, help="quick mode (the default): at most 1000 pairs and 100 trees")
    args = parser.parse_args(argv)
    del args
    MODELS.mkdir(parents=True, exist_ok=True)
    try:
        frame = pd.read_csv(DATA).head(1000)
        if not {"drug_a", "drug_b", "severity"}.issubset(frame.columns):
            raise ValueError("dataset must contain drug_a, drug_b, severity")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"dataset unavailable ({exc}); using built-in training examples")
        frame = _fallback_frame()
    cache = load_cache()
    X, y = _features(frame, cache)
    save_cache(cache)
    if len(X) == 0:
        frame = _fallback_frame()
        X, y = _features(frame, cache)
        save_cache(cache)
    if len(X) == 0:
        raise SystemExit("no usable chemical pairs; artifacts could not be trained")

    X_train, X_test, y_train, y_test = X, X, y, y
    if len(X) >= 8 and len(set(y)) >= 2:
        counts = np.bincount(y)
        stratify = y if len(counts) and np.all(counts[counts > 0] >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(f"accuracy: {accuracy_score(y_test, predicted):.3f}")
    print(f"weighted F1: {f1_score(y_test, predicted, average='weighted', zero_division=0):.3f}")
    print(classification_report(y_test, predicted, labels=range(4), target_names=LABELS, zero_division=0))
    matrix = confusion_matrix(y_test, predicted, labels=range(4))
    print("confusion matrix (rows=true, cols=pred): " + "; ".join(f"{LABELS[i]}={','.join(map(str, row))}" for i, row in enumerate(matrix)))
    import joblib
    joblib.dump(model, MODELS / "interaction_model.joblib")
    (MODELS / "severity_labels.json").write_text(json.dumps(LABELS, indent=2) + "\n")
    print(f"saved {MODELS / 'interaction_model.joblib'} and {MODELS / 'severity_labels.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
