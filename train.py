"""Train the single DDInter interaction-severity classifier."""
from __future__ import annotations

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


def main() -> int:
    if not DATA.exists():
        raise SystemExit("data/ddinter.csv is missing; run python data/fetch_ddinter.py first")
    frame = pd.read_csv(DATA).head(2000)
    required = {"drug_a", "drug_b", "severity"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"DDInter data must contain columns {sorted(required)}")

    cache = load_cache()
    features, targets = [], []
    for row in frame.itertuples(index=False):
        label = str(row.severity).strip().lower()
        if label not in LABELS:
            continue
        left = canonical_smiles(str(row.drug_a), cache)
        right = canonical_smiles(str(row.drug_b), cache)
        if not left or not right:
            continue
        try:
            a = np.asarray(fingerprint_array(left), dtype=np.uint8)
            b = np.asarray(fingerprint_array(right), dtype=np.uint8)
        except ValueError:
            continue
        features.append(np.bitwise_xor(a, b))
        targets.append(LABELS.index(label))
    save_cache(cache)
    if len(features) < 4 or len(set(targets)) < 2:
        raise SystemExit("Not enough resolvable, multi-class DDInter pairs to train")

    X, y = np.asarray(features), np.asarray(targets)
    counts = np.bincount(y, minlength=len(LABELS))
    stratify = y if np.all(counts[counts > 0] >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print(f"accuracy: {accuracy_score(y_test, predicted):.3f}")
    print(f"weighted F1: {f1_score(y_test, predicted, average='weighted', zero_division=0):.3f}")
    print(classification_report(y_test, predicted, labels=range(4), target_names=LABELS, zero_division=0))
    matrix = confusion_matrix(y_test, predicted, labels=range(4))
    print("confusion matrix (rows=true, cols=pred):")
    print("; ".join(f"{LABELS[i]}={','.join(map(str, row))}" for i, row in enumerate(matrix)))

    MODELS.mkdir(exist_ok=True)
    import joblib
    joblib.dump(model, MODELS / "interaction_model.joblib")
    (MODELS / "severity_labels.json").write_text(json.dumps(LABELS, indent=2) + "\n")
    print(f"saved {MODELS / 'interaction_model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
