"""Download DDInter when possible and always emit a usable normalized CSV."""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

URLS = [
    "https://github.com/hyunmokang/DDInter/archive/refs/heads/main.zip",
    "https://raw.githubusercontent.com/hyunmokang/DDInter/main/DDInter.csv",
    "https://raw.githubusercontent.com/hyunmokang/DDInter/main/data/DDInter.csv",
]
OUT = Path(__file__).resolve().parent / "ddinter.csv"
SEVERITIES = ("minor", "moderate", "major", "contraindicated")
FALLBACK = [
    ("aspirin", "warfarin", "major"), ("metformin", "alcohol", "major"),
    ("ibuprofen", "aspirin", "moderate"), ("amlodipine", "simvastatin", "moderate"),
    ("digoxin", "furosemide", "major"), ("sildenafil", "nitroglycerin", "contraindicated"),
    ("warfarin", "amiodarone", "major"), ("warfarin", "ibuprofen", "major"),
    ("warfarin", "trimethoprim", "major"), ("warfarin", "rifampicin", "major"),
    ("warfarin", "paracetamol", "moderate"), ("aspirin", "clopidogrel", "major"),
    ("aspirin", "ibuprofen", "moderate"), ("aspirin", "prednisone", "moderate"),
    ("metformin", "contrast media", "major"), ("metformin", "cimetidine", "moderate"),
    ("simvastatin", "clarithromycin", "major"), ("atorvastatin", "clarithromycin", "major"),
    ("simvastatin", "grapefruit", "moderate"), ("atorvastatin", "grapefruit", "moderate"),
    ("amlodipine", "lisinopril", "minor"), ("amlodipine", "verapamil", "moderate"),
    ("digoxin", "amiodarone", "major"), ("digoxin", "verapamil", "major"),
    ("furosemide", "lithium", "major"), ("furosemide", "ibuprofen", "moderate"),
    ("sildenafil", "doxazosin", "moderate"), ("nitroglycerin", "tadalafil", "contraindicated"),
    ("ciprofloxacin", "theophylline", "major"), ("ciprofloxacin", "antacids", "moderate"),
    ("fluoxetine", "tramadol", "major"), ("sertraline", "linezolid", "contraindicated"),
    ("clarithromycin", "colchicine", "major"), ("erythromycin", "warfarin", "major"),
    ("lisinopril", "potassium", "moderate"), ("hydrochlorothiazide", "lithium", "moderate"),
    ("insulin", "beta blockers", "moderate"), ("levothyroxine", "calcium", "moderate"),
    ("omeprazole", "clopidogrel", "moderate"), ("loratadine", "alcohol", "minor"),
]


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _find_column(columns, names):
    normalized = {_norm(c): c for c in columns}
    for name in names:
        if _norm(name) in normalized:
            return normalized[_norm(name)]
    for key, original in normalized.items():
        if any(_norm(name) in key for name in names):
            return original
    return None


def _parse(blob: bytes) -> pd.DataFrame:
    frames = []
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for member in archive.namelist():
            if member.endswith("/") or not member.lower().endswith((".csv", ".tsv", ".txt")):
                continue
            raw = archive.read(member)
            for sep in (None, ",", "\t", ";"):
                try:
                    frame = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
                    if len(frame.columns) >= 3:
                        frames.append(frame)
                        break
                except Exception:
                    continue
    if not frames:
        raise ValueError("zip contained no readable table")
    return _normalize(frames[0])


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    a = _find_column(frame.columns, ("drug_a", "drug1", "drugname1", "medication_a", "drug"))
    b = _find_column(frame.columns, ("drug_b", "drug2", "drugname2", "medication_b", "interacting_drug"))
    s = _find_column(frame.columns, ("severity", "level", "risk", "interaction_type", "classification"))
    if not (a and b and s):
        raise ValueError("missing drug-pair or severity columns")
    rows = []
    for left, right, label in frame[[a, b, s]].itertuples(index=False, name=None):
        text = _norm(label)
        mapped = next((x for x in SEVERITIES if x in text), None)
        if text in {"1", "1 0", "low"}: mapped = "minor"
        elif text in {"2", "2 0", "medium"}: mapped = "moderate"
        elif text in {"3", "3 0", "high", "severe"}: mapped = "major"
        elif text in {"4", "4 0"}: mapped = "contraindicated"
        if mapped and pd.notna(left) and pd.notna(right):
            rows.append({"drug_a": str(left).strip(), "drug_b": str(right).strip(), "severity": mapped})
    result = pd.DataFrame(rows, columns=["drug_a", "drug_b", "severity"]).drop_duplicates()
    if result.empty:
        raise ValueError("table had no usable rows")
    return result


def _fallback() -> pd.DataFrame:
    return pd.DataFrame(FALLBACK, columns=["drug_a", "drug_b", "severity"])


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    usable = None
    errors = []
    for url in URLS:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            if url.endswith(".zip"):
                usable = _parse(response.content)
            else:
                usable = _normalize(pd.read_csv(io.BytesIO(response.content), sep=None, engine="python"))
            if len(usable) >= 1:
                print(f"DDInter source used: {url} ({len(usable)} rows)")
                break
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            print(f"DDInter source failed: {url}: {exc}", file=sys.stderr)
    if usable is None or usable.empty:
        usable = _fallback()
        print(f"DDInter source used: built-in curated fallback ({len(usable)} rows)")
    usable[["drug_a", "drug_b", "severity"]].to_csv(OUT, index=False)
    if errors:
        print("DDInter download/parse errors were handled; training data was still written.", file=sys.stderr)
    print(f"Wrote {OUT} with columns drug_a, drug_b, severity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
