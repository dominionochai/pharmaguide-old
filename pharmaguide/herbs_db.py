"""Curated, non-diagnostic herbal interaction knowledge for the photo layer."""
from __future__ import annotations

HERBS_DB = {
    "bitter leaf": {"common_name": "bitter leaf", "scientific_name": "Vernonia amygdalina", "interaction_notes": "May add to hypoglycemia when combined with antidiabetic medicines; monitor glucose.", "severity": "moderate", "affected_drug_classes": ["antidiabetics", "insulin"]},
    "neem": {"common_name": "neem", "scientific_name": "Azadirachta indica", "interaction_notes": "May lower glucose and may add liver stress with hepatotoxic medicines.", "severity": "moderate", "affected_drug_classes": ["antidiabetics", "hepatotoxic medicines", "warfarin"]},
    "garlic": {"common_name": "garlic", "scientific_name": "Allium sativum", "interaction_notes": "May increase bleeding risk with warfarin, aspirin, and antiplatelet medicines.", "severity": "moderate", "affected_drug_classes": ["warfarin", "aspirin", "antiplatelets", "anticoagulants"]},
    "ginger": {"common_name": "ginger", "scientific_name": "Zingiber officinale", "interaction_notes": "May increase bleeding risk with anticoagulants and antiplatelet medicines.", "severity": "moderate", "affected_drug_classes": ["anticoagulants", "antiplatelets", "warfarin"]},
    "moringa": {"common_name": "moringa", "scientific_name": "Moringa oleifera", "interaction_notes": "May enhance the effects of antihypertensive and antidiabetic medicines.", "severity": "moderate", "affected_drug_classes": ["antihypertensives", "antidiabetics"]},
    "turmeric": {"common_name": "turmeric/curcumin", "scientific_name": "Curcuma longa", "interaction_notes": "May add bleeding risk with warfarin and other blood thinners.", "severity": "moderate", "affected_drug_classes": ["warfarin", "anticoagulants", "antiplatelets"]},
    "st john's wort": {"common_name": "St John's wort", "scientific_name": "Hypericum perforatum", "interaction_notes": "Strong enzyme inducer, including CYP3A4; can reduce levels of many medicines.", "severity": "major", "affected_drug_classes": ["CYP3A4 substrates", "oral contraceptives", "antidepressants", "anticoagulants"]},
    "ginseng": {"common_name": "ginseng", "scientific_name": "Panax ginseng", "interaction_notes": "May affect bleeding risk and blood pressure; use caution with anticoagulants and BP medicines.", "severity": "moderate", "affected_drug_classes": ["anticoagulants", "warfarin", "antihypertensives"]},
    "kola nut": {"common_name": "kola nut/caffeine sources", "scientific_name": "Cola acuminata", "interaction_notes": "Stimulant caffeine sources may worsen palpitations or oppose some blood-pressure medicines.", "severity": "minor", "affected_drug_classes": ["antihypertensives", "stimulants", "cardiac medicines"]},
    "papaya leaf": {"common_name": "papaya leaf", "scientific_name": "Carica papaya", "interaction_notes": "May affect platelets and bleeding; discuss use with anticoagulant or antiplatelet medicines.", "severity": "moderate", "affected_drug_classes": ["anticoagulants", "antiplatelets", "warfarin"]},
    "aloe vera": {"common_name": "aloe vera", "scientific_name": "Aloe barbadensis", "interaction_notes": "Oral use may lower potassium and change the effect of diuretics or cardiac medicines.", "severity": "moderate", "affected_drug_classes": ["diuretics", "digoxin", "cardiac medicines"]},
    "ginkgo": {"common_name": "ginkgo", "scientific_name": "Ginkgo biloba", "interaction_notes": "May increase bleeding risk with anticoagulant or antiplatelet medicines.", "severity": "moderate", "affected_drug_classes": ["warfarin", "anticoagulants", "antiplatelets"]},
    "unknown herb": {"common_name": "unknown herb", "scientific_name": "unknown", "interaction_notes": "The plant could not be identified; composition and interactions are unknown.", "severity": "major", "affected_drug_classes": []},
}

ALIASES = {
    "bitterleaf": "bitter leaf", "vernonia amygdalina": "bitter leaf", "azadirachta indica": "neem",
    "allium sativum": "garlic", "zingiber officinale": "ginger", "moringa oleifera": "moringa",
    "curcumin": "turmeric", "turmeric curcumin": "turmeric", "st johns wort": "st john's wort",
    "hypericum perforatum": "st john's wort", "panax ginseng": "ginseng", "caffeine": "kola nut",
    "kola nut caffeine": "kola nut", "carica papaya": "papaya leaf", "papaya": "papaya leaf",
}

# Public lower-case alias retained for callers that want the database directly.
herbs_db = HERBS_DB

def normalize_name(name: str) -> str:
    return " ".join(str(name).casefold().replace("’", "'").replace("-", " ").split())


def medication_flags(record: dict, medications: list[str]) -> list[dict]:
    flags = []
    classes = [normalize_name(x) for x in record.get("affected_drug_classes", [])]
    for medication in medications:
        med = normalize_name(medication)
        if not med:
            continue
        matched = [drug_class for drug_class in classes if drug_class in med or med in drug_class]
        if matched or any(token in med for token in ("warfarin", "aspirin", "clopidogrel", "insulin", "metformin", "glibenclamide", "lisinopril", "amlodipine", "digoxin")) and classes:
            flags.append({"medication": medication, "severity": record["severity"], "matched_classes": matched or classes, "note": record["interaction_notes"]})
    return flags
