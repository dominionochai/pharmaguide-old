"""Plain-language tell-your-doctor sheet snippets."""
from __future__ import annotations


def tell_your_doctor_sheet(drug_a: str, drug_b: str, severity: str) -> str:
    """Render one flag in English and simple pidgin for a clinic conversation."""
    english = f"Please tell my doctor that {drug_a} and {drug_b} may have a {severity} interaction."
    pidgin = "Oga doctor, this one go affect heart rhythm; abeg check am." if severity in {"major", "contraindicated"} else "Oga doctor, abeg check how these two medicines fit work together."
    return f"English: {english}\nSimple pidgin: {pidgin}"
