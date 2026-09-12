"""Cached PubChem chemistry helpers with an offline common-drug fallback."""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote

import requests
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "smiles_cache.json"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
MAX_PUBCHEM_LOOKUPS = 600
_LOOKUPS = 0
_LAST_REQUEST = 0.0

BUILTIN_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O", "warfarin": "CC(C(=O)CC(c1ccccc1)c1ccccc1)O",
    "metformin": "CN(C)C(=N)NC(=N)N", "paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "acetaminophen": "CC(=O)NC1=CC=C(C=C1)O", "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "amlodipine": "CCOC(=O)C1=C(NC(=C(C1)C)C(=O)OCC)COCCN", "simvastatin": "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C(=O)O[C@H]1C",
    "digoxin": "CC1OC(O)C(O)C(O)C1OC2C(C)OC(OC3C(C)OC(OC4C(C)OC(O)C(O)C4O)C(O)C3O)C(O)C2O",
    "furosemide": "C1=CC(=CC(=C1)C(=O)NCC2=CC(=CC=C2)S(=O)(=O)N)Cl",
    "sildenafil": "CCCC1=NN(C2=CC=CC=C2)C(=O)C3=C1N=CN3C4=CC=C(C=C4)S(=O)(=O)N",
    "nitroglycerin": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]", "atorvastatin": "CC(C)C1=CC(=C(C=C1)C2=CC=CC=C2)N(C)C(=O)C",
    "amiodarone": "COc1ccc(cc1OC)C(=O)NCCc2ccc(I)cc2I", "clopidogrel": "COC(=O)[C@H](c1ccccc1Cl)N1CCc2sccc2C1",
    "lisinopril": "C1CC(N(C1)C(=O)CCC2=CC=CC=C2)C(=O)O", "verapamil": "COC1=CC=CC(=C1OC)C(CN(C)C)CCCN(C)C",
    "rifampicin": "CC1C=CC(C(=O)N)=CC1C2=CC(=O)C3=C(O)C=C(O)C=C3O2", "trimethoprim": "COC1=CC(=C(C=C1OC)OC)CC2=NC(=NC=C2)N",
    "clarithromycin": "CC1OC(OC2C(C)OC(OC3C(C)OC(O)C(O)C3O)C(O)C2O)C(O)C(O)C1N(C)C", "erythromycin": "CC1OC(OC2C(C)OC(OC3C(C)OC(O)C(O)C3O)C(O)C2O)C(O)C(O)C1N(C)C",
    "ciprofloxacin": "C1CC1N2C=C(C(=O)O)C(=O)C3=C2N=CC(=C3N)F", "fluoxetine": "CNCCC(C1=CC=CC=C1)OC2=CC=CC=C2C(F)(F)F",
    "sertraline": "C1=CC=C(C(=C1)C2CC3CCC(C2)N3)Cl", "tramadol": "COC1=CC=CC(=C1)C(CN(C)C)(C2CCCCC2)O",
    "linezolid": "CC(=O)NCC1=CC=C(C=C1)N2CCOC2=O", "colchicine": "COC1=CC2=C(C=C1OC)C(=CC(=C2)OC)C(=O)NCC3CCC4=CC(=O)OC=C4C3",
    "prednisone": "CC12CCC3C(C1CCC2=O)CCC4=CC(=O)CCC34C", "lithium": "[Li+]", "alcohol": "CCO",
    "grapefruit": "CC(C)C1=CC(=O)C(=C(C1)O)C2=CC=CC=C2", "potassium": "[K+]", "omeprazole": "COC1=NC=NC2=C1C(=NCC2)S(=O)C3=CC=CC=C3OC",
    "insulin": "N", "theophylline": "CN1C(=O)N(C)C2=C1C=NC(=O)N2", "calcium": "[Ca++]",
}
ALIASES = {"acetaminophen": "paracetamol", "paracetamol": "paracetamol", "asa": "aspirin", "etoh": "alcohol"}


def load_cache(path: Path = CACHE_PATH) -> dict[str, str | None]:
    try:
        value = json.loads(path.read_text()) if path.exists() else {}
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(cache: dict[str, str | None], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")


def canonical_smiles(name: str, cache: dict | None = None) -> str | None:
    global _LOOKUPS, _LAST_REQUEST
    cache = cache if cache is not None else load_cache()
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if not key:
        return None
    if key in cache and cache[key]:
        return cache[key]
    fallback = BUILTIN_SMILES.get(key)
    value = fallback
    if _LOOKUPS < MAX_PUBCHEM_LOOKUPS:
        wait = 0.3 - (time.monotonic() - _LAST_REQUEST)
        if _LAST_REQUEST and wait > 0:
            time.sleep(wait)
        _LOOKUPS += 1
        try:
            response = requests.get(PUBCHEM_URL.format(quote(key)), timeout=10)
            _LAST_REQUEST = time.monotonic()
            response.raise_for_status()
            value = response.json()["PropertyTable"]["Properties"][0]["ConnectivitySMILES"] or fallback
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
            value = fallback
    cache[key] = value
    return value


def fingerprint(smiles: str):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024)


def fingerprint_array(smiles: str) -> list[int]:
    bits = [0] * 1024
    DataStructs.ConvertToNumpyArray(fingerprint(smiles), bits)
    return bits
