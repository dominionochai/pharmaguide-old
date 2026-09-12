"""Shared chemistry lookup and fingerprint helpers for training and the API."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import requests
from rdkit import DataStructs
from rdkit.Chem import AllChem, MolFromSmiles
from rdkit.Chem.rdchem import ExplicitBitVect

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "smiles_cache.json"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"


def load_cache(path: Path = CACHE_PATH) -> dict[str, str | None]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict[str, str | None], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True))


def canonical_smiles(name: str, cache: dict[str, str | None] | None = None) -> str | None:
    cache = cache if cache is not None else load_cache()
    key = name.strip().lower()
    if not key:
        return None
    if key in cache:
        return cache[key]
    url = PUBCHEM_URL.format(quote(name.strip(), safe=""))
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        value = response.json()["PropertyTable"]["Properties"][0]["ConnectivitySMILES"]
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        value = None
    cache[key] = value
    return value


def fingerprint(smiles: str) -> ExplicitBitVect:
    molecule = MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024)


def fingerprint_array(smiles: str):
    bits = [0] * 1024
    DataStructs.ConvertToNumpyArray(fingerprint(smiles), bits)
    return bits
