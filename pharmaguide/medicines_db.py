"""Structured medicine names, uses, and known pairwise interaction flags."""
from __future__ import annotations

from typing import Dict, Iterable, List


def _drug(name: str, category: str, *uses: str) -> dict:
    return {
        "name": name,
        "category": category,
        "common_uses": list(uses),
        "known_interactions": [],
    }


MEDICINES_DB: Dict[str, dict] = {
    "amlodipine": _drug("amlodipine", "calcium-channel blocker", "hypertension", "angina"),
    "lisinopril": _drug("lisinopril", "ACE inhibitor", "hypertension", "heart failure"),
    "atenolol": _drug("atenolol", "beta blocker", "hypertension", "angina"),
    "losartan": _drug("losartan", "angiotensin receptor blocker", "hypertension", "kidney protection"),
    "nifedipine": _drug("nifedipine", "calcium-channel blocker", "hypertension", "angina"),
    "metformin": _drug("metformin", "antidiabetic", "type 2 diabetes"),
    "glibenclamide": _drug("glibenclamide", "sulfonylurea", "type 2 diabetes"),
    "gliclazide": _drug("gliclazide", "sulfonylurea", "type 2 diabetes"),
    "insulin": _drug("insulin", "antidiabetic hormone", "diabetes"),
    "artemether/lumefantrine": _drug("artemether/lumefantrine", "antimalarial", "uncomplicated malaria"),
    "chloroquine": _drug("chloroquine", "antimalarial", "malaria", "some autoimmune conditions"),
    "quinine": _drug("quinine", "antimalarial", "severe malaria", "leg cramps"),
    "doxycycline": _drug("doxycycline", "tetracycline antibiotic", "bacterial infections", "malaria prophylaxis"),
    "paracetamol": _drug("paracetamol", "analgesic", "pain", "fever"),
    "ibuprofen": _drug("ibuprofen", "NSAID", "pain", "inflammation", "fever"),
    "diclofenac": _drug("diclofenac", "NSAID", "pain", "inflammation"),
    "aspirin": _drug("aspirin", "antiplatelet and analgesic", "pain", "fever", "cardiovascular prevention"),
    "tramadol": _drug("tramadol", "opioid analgesic", "moderate pain"),
    "amoxicillin": _drug("amoxicillin", "penicillin antibiotic", "bacterial infections"),
    "ciprofloxacin": _drug("ciprofloxacin", "fluoroquinolone antibiotic", "bacterial infections"),
    "azithromycin": _drug("azithromycin", "macrolide antibiotic", "bacterial infections"),
    "metronidazole": _drug("metronidazole", "antimicrobial", "anaerobic infections", "parasitic infections"),
    "co-trimoxazole": _drug("co-trimoxazole", "sulfonamide antibiotic", "bacterial infections"),
    "multivitamins": _drug("multivitamins", "vitamin supplement", "vitamin supplementation"),
    "ors": _drug("ors", "oral rehydration solution", "dehydration from diarrhoea or vomiting"),
    "antacids": _drug("antacids", "antacid", "heartburn", "indigestion"),
    "albendazole": _drug("albendazole", "anthelmintic", "intestinal worm infections"),
    "prednisolone": _drug("prednisolone", "corticosteroid", "inflammation", "allergic disease"),
    "omeprazole": _drug("omeprazole", "proton-pump inhibitor", "acid reflux", "ulcer disease"),
    "digoxin": _drug("digoxin", "cardiac glycoside", "heart failure", "atrial fibrillation"),
    "warfarin": _drug("warfarin", "anticoagulant", "prevention of blood clots"),
    "sildenafil": _drug("sildenafil", "PDE5 inhibitor", "erectile dysfunction", "pulmonary hypertension"),
    "simvastatin": _drug("simvastatin", "statin", "high cholesterol", "cardiovascular risk reduction"),
    "atorvastatin": _drug("atorvastatin", "statin", "high cholesterol", "cardiovascular risk reduction"),
    "diazepam": _drug("diazepam", "benzodiazepine", "anxiety", "muscle spasm", "seizures"),
    "codeine": _drug("codeine", "opioid analgesic", "pain", "cough"),
    "hydrochlorothiazide": _drug("hydrochlorothiazide", "thiazide diuretic", "hypertension", "oedema"),
    "furosemide": _drug("furosemide", "loop diuretic", "oedema", "heart failure"),
    "amiodarone": _drug("amiodarone", "antiarrhythmic", "irregular heart rhythm"),
    "clopidogrel": _drug("clopidogrel", "antiplatelet", "prevention of blood clots"),
    "rifampicin": _drug("rifampicin", "rifamycin antibiotic", "tuberculosis", "other bacterial infections"),
    "nitroglycerin": _drug("nitroglycerin", "nitrate", "angina"),
    "fluoxetine": _drug("fluoxetine", "SSRI antidepressant", "depression", "anxiety"),
    "alcohol": _drug("alcohol", "substance", "beverage or solvent exposure"),
}


def _add_flag(first: str, second: str, severity: str, reason: str, confidence: float = 0.95) -> None:
    """Add a symmetric, database-derived flag for a pair in the catalog."""
    flag = {
        "with": second,
        "severity": severity,
        "confidence": confidence,
        "reason": reason,
    }
    reverse = {**flag, "with": first}
    MEDICINES_DB[first]["known_interactions"].append(flag)
    MEDICINES_DB[second]["known_interactions"].append(reverse)


_add_flag("warfarin", "aspirin", "major", "Combined blood-thinning effects may increase bleeding risk.")
_add_flag("warfarin", "ibuprofen", "major", "An anticoagulant combined with an NSAID can substantially increase gastrointestinal bleeding risk.")
_add_flag("warfarin", "diclofenac", "major", "An anticoagulant combined with an NSAID can increase bleeding risk.")
_add_flag("warfarin", "paracetamol", "moderate", "Regular or high-dose use may alter anticoagulant effect; monitoring may be needed.")
_add_flag("warfarin", "metronidazole", "major", "Metronidazole can increase anticoagulant effect and bleeding risk.")
_add_flag("warfarin", "co-trimoxazole", "major", "Co-trimoxazole can increase anticoagulant effect and bleeding risk.")
_add_flag("warfarin", "rifampicin", "major", "Rifampicin can reduce anticoagulant effect and destabilize clot protection.")
_add_flag("warfarin", "clopidogrel", "major", "Two antithrombotic medicines can increase bleeding risk.")
_add_flag("warfarin", "amiodarone", "major", "Amiodarone can increase anticoagulant exposure and bleeding risk.")
_add_flag("warfarin", "doxycycline", "moderate", "Doxycycline may increase anticoagulant effect; monitor for bleeding.")
_add_flag("aspirin", "ibuprofen", "moderate", "Combined NSAID and antiplatelet effects can increase stomach irritation and bleeding.")
_add_flag("aspirin", "diclofenac", "moderate", "Combined antiplatelet and NSAID effects can increase stomach irritation and bleeding.")
_add_flag("aspirin", "clopidogrel", "major", "Dual antiplatelet therapy increases bleeding risk.")
_add_flag("aspirin", "prednisolone", "moderate", "The combination can increase gastrointestinal irritation and bleeding risk.")
_add_flag("metformin", "alcohol", "major", "Alcohol can increase the risk of metformin-associated lactic acidosis and alter glucose control.")
_add_flag("metformin", "prednisolone", "moderate", "Prednisolone can raise blood glucose and make diabetes control more difficult.")
_add_flag("ciprofloxacin", "antacids", "moderate", "Minerals in antacids can reduce ciprofloxacin absorption; separate administration.")
_add_flag("ciprofloxacin", "multivitamins", "moderate", "Mineral supplements can reduce ciprofloxacin absorption; separate administration.")
_add_flag("amlodipine", "simvastatin", "moderate", "Amlodipine can increase simvastatin exposure; the combination may require a dose limit and review.")
_add_flag("sildenafil", "nitroglycerin", "contraindicated", "PDE5 inhibitors with nitrates can cause a dangerous fall in blood pressure.")
_add_flag("sildenafil", "alcohol", "moderate", "Alcohol may worsen dizziness and blood-pressure lowering effects.")
_add_flag("digoxin", "furosemide", "major", "Diuretic-related electrolyte changes can increase digoxin toxicity risk.")
_add_flag("digoxin", "amiodarone", "major", "Amiodarone can increase digoxin exposure and toxicity risk.")
_add_flag("insulin", "glibenclamide", "moderate", "Two glucose-lowering medicines can cause hypoglycaemia.")
_add_flag("insulin", "gliclazide", "moderate", "Two glucose-lowering medicines can cause hypoglycaemia.")
_add_flag("insulin", "alcohol", "moderate", "Alcohol can make blood glucose harder to control and may contribute to hypoglycaemia.")
_add_flag("diazepam", "codeine", "major", "Combined sedative effects can cause profound drowsiness and breathing problems.")
_add_flag("tramadol", "fluoxetine", "major", "The combination can increase serotonin toxicity and seizure risk.")
_add_flag("tramadol", "alcohol", "major", "Combined central nervous system depression can impair breathing and alertness.")
_add_flag("codeine", "alcohol", "major", "Combined central nervous system depression can impair breathing and alertness.")
_add_flag("atorvastatin", "amiodarone", "moderate", "Amiodarone may increase statin exposure and muscle-toxicity risk.")
_add_flag("simvastatin", "amiodarone", "major", "Amiodarone may substantially increase simvastatin exposure and muscle-toxicity risk.")
_add_flag("lisinopril", "losartan", "major", "Dual renin-angiotensin blockade can increase kidney injury and high potassium risk.")
_add_flag("lisinopril", "potassium", "major", "Potassium exposure can increase the risk of high blood potassium.") if "potassium" in MEDICINES_DB else None


ALIASES = {
    "oral rehydration salts": "ors",
    "oral rehydration solution": "ors",
    "artemether lumefantrine": "artemether/lumefantrine",
    "co trimoxazole": "co-trimoxazole",
    "co-trimoxazole": "co-trimoxazole",
    "acetaminophen": "paracetamol",
}

# Public aliases make the catalog easy to consume without exposing mutable setup details.
medicines_db = MEDICINES_DB


def normalize_medicine_name(name: str) -> str:
    return " ".join(str(name).casefold().replace("_", " ").split())


def resolve_medicine(name: str) -> str | None:
    normalized = normalize_medicine_name(name)
    canonical = ALIASES.get(normalized, normalized)
    return canonical if canonical in MEDICINES_DB else None


def medicine_count() -> int:
    return len(MEDICINES_DB)
