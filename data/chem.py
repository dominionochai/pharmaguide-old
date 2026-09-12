"""Offline-first chemistry helpers with cached PubChem enrichment."""
from __future__ import annotations
import hashlib, json
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote
try:
    import requests
except ImportError:
    requests = None
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    Chem = AllChem = DataStructs = None
    RDKIT_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "smiles_cache.json"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"
BUILTIN_SMILES: Dict[str, str] = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "warfarin": "CC(C(=O)CC(c1ccccc1)c1ccccc1)O",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "acetaminophen": "CC(=O)NC1=CC=C(O)C=C1",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "amlodipine": "CCOC(=O)C1=C(NC(=C(C1C)C(=O)OC)N)C2=CC=CC=C2Cl",
    "simvastatin": "CC(C)C1=C(C(=O)O1)C(C)C",
    "digoxin": "CC1OC(OC2C(C(C(C(O2)CO)O)O)O)C(C(C1O)O)O",
    "furosemide": "C1=CC(=CC(=C1N)S(=O)(=O)N)C(=O)O",
    "sildenafil": "CCC1=NN(C(=O)C2=C1C=C(C=C2)N3CCN(CC3)C)C4=CC=CC=C4",
    "nitroglycerin": "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
    "atorvastatin": "CC(C)c1n(c(c(c(=O)n1)C(C)C)C(=O)O)C2=CC=CC=C2",
    "alcohol": "CCO", "amiodarone": "CCCCCCc1oc2cc(Cl)ccc2c1C(=O)c1ccccc1",
    "lisinopril": "C1CCN(CC1)C(=O)C(CC2=CC=CC=C2)N", "potassium": "[K+]",
    "lithium": "[Li+]", "grapefruit": "CC(C)C1=CC(=O)OC2=C1C=CC=C2O",
}
ALIASES = {"asa": "aspirin", "etoh": "alcohol", "acetaminophen": "paracetamol"}

def load_cache(path: Path = CACHE_PATH) -> Dict[str, Optional[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}

def save_cache(cache: Dict[str, Optional[str]], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def canonical_smiles(name: str, cache: Optional[Dict[str, Optional[str]]] = None) -> Optional[str]:
    """Resolve a name locally first; PubChem is best effort only."""
    cache = cache if cache is not None else load_cache()
    key = ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if cache.get(key):
        return cache[key]
    if key in BUILTIN_SMILES:
        cache[key] = BUILTIN_SMILES[key]
        return cache[key]
    if requests is not None:
        try:
            response = requests.get(PUBCHEM_URL.format(quote(key)), timeout=3)
            response.raise_for_status()
            value = response.json()["PropertyTable"]["Properties"][0]["ConnectivitySMILES"]
            cache[key] = value
            return value
        except Exception:
            pass
    cache[key] = None
    return None

def _hashed_fingerprint(smiles: str, n_bits: int = 1024) -> list[int]:
    bits = [0] * n_bits
    for width in (1, 2, 3):
        for index in range(max(0, len(smiles) - width + 1)):
            digest = hashlib.blake2b(smiles[index:index + width].encode(), digest_size=4).digest()
            bits[int.from_bytes(digest, "little") % n_bits] = 1
    return bits

def fingerprint_array(smiles: str) -> list[int]:
    if RDKIT_AVAILABLE:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            vector = np.zeros(1024, dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=1024), vector)
            return [int(value) for value in vector]
    return _hashed_fingerprint(smiles)

def fingerprint(smiles: str) -> list[int]:
    return fingerprint_array(smiles)

def rdkit_status() -> bool:
    return RDKIT_AVAILABLE
