"""Best-effort DDInter downloader that always writes usable fallback data."""
from __future__ import annotations
import io, sys, zipfile
from pathlib import Path
import pandas as pd
import requests
URLS = ["https://github.com/hyunmokan/DDInter/archive/refs/heads/main.zip", "https://raw.githubusercontent.com/hyunmokan/DDInter/main/DDInter.csv", "https://raw.githubusercontent.com/hyunmokan/DDInter/main/data/DDInter.csv"]
OUT = Path(__file__).resolve().parent / "ddinter.csv"
FALLBACK = [
    ("aspirin", "warfarin", "major"), ("metformin", "alcohol", "major"), ("ibuprofen", "aspirin", "moderate"),
    ("sildenafil", "nitroglycerin", "contraindicated"), ("amlodipine", "simvastatin", "moderate"), ("digoxin", "furosemide", "major"),
    ("paracetamol", "warfarin", "moderate"), ("warfarin", "amiodarone", "major"), ("warfarin", "trimethoprim", "major"),
    ("warfarin", "rifampicin", "major"), ("aspirin", "clopidogrel", "major"), ("metformin", "contrast media", "major"),
    ("metformin", "cimetidine", "moderate"), ("simvastatin", "clarithromycin", "major"), ("atorvastatin", "clarithromycin", "major"),
    ("simvastatin", "grapefruit", "moderate"), ("amlodipine", "lisinopril", "minor"), ("digoxin", "amiodarone", "major"),
    ("digoxin", "verapamil", "major"), ("furosemide", "lithium", "major"), ("furosemide", "ibuprofen", "moderate"),
    ("sildenafil", "doxazosin", "moderate"), ("nitroglycerin", "tadalafil", "contraindicated"), ("ciprofloxacin", "theophylline", "major"),
    ("fluoxetine", "tramadol", "major"), ("sertraline", "linezolid", "contraindicated"), ("clarithromycin", "colchicine", "major"),
    ("erythromycin", "warfarin", "major"), ("lisinopril", "potassium", "moderate"), ("hydrochlorothiazide", "lithium", "moderate"),
    ("insulin", "beta blockers", "moderate"), ("levothyroxine", "calcium", "moderate"), ("omeprazole", "clopidogrel", "moderate"),
    ("loratadine", "alcohol", "minor"),
]

def fallback_frame():
    return pd.DataFrame(FALLBACK, columns=["drug_a", "drug_b", "severity"])

def _normalise(frame):
    columns = {str(c).lower().replace(" ", "_"): c for c in frame.columns}
    def find(*names):
        for name in names:
            if name in columns: return columns[name]
        return None
    a, b, s = find("drug_a", "drug1", "drug_name1", "medicine_a"), find("drug_b", "drug2", "drug_name2", "interacting_drug"), find("severity", "level", "risk", "classification")
    if not (a and b and s): raise ValueError("DDInter table has no recognisable columns")
    result = frame[[a, b, s]].copy(); result.columns = ["drug_a", "drug_b", "severity"]
    result["severity"] = result["severity"].astype(str).str.lower().replace({"low": "minor", "medium": "moderate", "high": "major", "severe": "major", "4": "contraindicated"})
    return result[result["severity"].isin(["minor", "moderate", "major", "contraindicated"])].dropna().drop_duplicates()

def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True); frame = None
    for url in URLS:
        try:
            response = requests.get(url, timeout=8); response.raise_for_status()
            if url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                    names = [n for n in archive.namelist() if n.lower().endswith((".csv", ".tsv", ".txt"))]
                    if not names: raise ValueError("archive contained no table")
                    frame = _normalise(pd.read_csv(io.BytesIO(archive.read(names[0])), sep=None, engine="python"))
            else:
                frame = _normalise(pd.read_csv(io.BytesIO(response.content), sep=None, engine="python"))
            if len(frame): break
        except Exception as exc:
            print(f"DDInter source failed: {url}: {exc}", file=sys.stderr)
    if frame is None or frame.empty:
        frame = fallback_frame(); print(f"DDInter unavailable; wrote {len(frame)} built-in curated pairs.", file=sys.stderr)
    else:
        frame = pd.concat([frame, fallback_frame()], ignore_index=True).drop_duplicates(); print(f"DDInter source used; wrote {len(frame)} rows.")
    frame[["drug_a", "drug_b", "severity"]].to_csv(OUT, index=False)
    print(f"Wrote {OUT} with columns drug_a, drug_b, severity")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
