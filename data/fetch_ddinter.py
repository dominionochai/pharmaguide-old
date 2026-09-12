"""Download and normalize DDInter interaction data."""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://github.com/hyunmokang/DDInter/archive/refs/heads/main.zip"
OUT = Path(__file__).resolve().parent / "ddinter.csv"
SEVERITIES = ("minor", "moderate", "major", "contraindicated")


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_column(columns, candidates):
    normalized = {_norm(c): c for c in columns}
    for candidate in candidates:
        for key, original in normalized.items():
            if candidate in key:
                return original
    return None


def _severity(value: object) -> str | None:
    text = _norm(value)
    if "contra" in text or text in {"4", "4.0"}:
        return "contraindicated"
    if "major" in text or text in {"3", "3.0", "severe", "high"}:
        return "major"
    if "moderate" in text or text in {"2", "2.0", "medium"}:
        return "moderate"
    if "minor" in text or text in {"1", "1.0", "low"}:
        return "minor"
    return None


def _tables(blob: bytes):
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".csv", ".tsv", ".txt")):
                continue
            raw = archive.read(info)
            for sep in (None, ",", "\t", ";"):
                try:
                    frame = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python")
                    if len(frame.columns) >= 3:
                        yield info.filename, frame
                        break
                except (UnicodeDecodeError, pd.errors.ParserError, ValueError):
                    continue


def normalize(blob: bytes) -> pd.DataFrame:
    for filename, frame in _tables(blob):
        a = _find_column(frame.columns, ("druga", "drug1", "drugname1", "medicationa", "drug"))
        b = _find_column(frame.columns, ("drugb", "drug2", "drugname2", "medicationb"))
        severity = _find_column(frame.columns, ("severity", "level", "risk", "interactiontype", "classification"))
        if not (a and b and severity):
            continue
        rows = []
        for left, right, label in frame[[a, b, severity]].itertuples(index=False, name=None):
            mapped = _severity(label)
            if mapped and pd.notna(left) and pd.notna(right):
                rows.append({"drug_a": str(left).strip(), "drug_b": str(right).strip(), "severity": mapped})
        result = pd.DataFrame(rows, columns=["drug_a", "drug_b", "severity"])
        if not result.empty:
            return result.drop_duplicates()
        raise ValueError(f"Found table {filename!r}, but it contained no supported severity labels")
    raise ValueError("No supported DDInter CSV/TSV table found: expected drug-pair and severity columns")


def main() -> int:
    try:
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        result = normalize(response.content)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(OUT, index=False)
        print(f"Saved {len(result)} normalized DDInter rows to {OUT}")
        return 0
    except (requests.RequestException, zipfile.BadZipFile, ValueError, OSError) as exc:
        print(f"DDInter download/normalization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
