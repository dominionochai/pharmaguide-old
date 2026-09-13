"""Plain-language tell-your-doctor sheets for detected interaction flags."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _as_flags(
    flags_or_drug_a: Any,
    drug_b: str | None = None,
    severity: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize the current flag list, while retaining the old call shape."""
    if drug_b is not None or severity is not None:
        return [
            {
                "drug_a": str(flags_or_drug_a),
                "drug_b": str(drug_b or "unknown medicine"),
                "severity": str(severity or "unknown"),
            }
        ]

    if isinstance(flags_or_drug_a, Mapping):
        return [dict(flags_or_drug_a)]
    if flags_or_drug_a is None or isinstance(flags_or_drug_a, str):
        return []
    if isinstance(flags_or_drug_a, Iterable):
        return [dict(flag) for flag in flags_or_drug_a if isinstance(flag, Mapping)]
    return []


def _confidence_text(value: Any) -> str:
    try:
        return f"{float(value):.0%} confidence"
    except (TypeError, ValueError):
        return "confidence not reported"


def tell_your_doctor_sheet(
    flags_or_drug_a: Any,
    drug_b: str | None = None,
    severity: str | None = None,
) -> str:
    """Render the interaction flags from the current request in English and Pidgin."""
    flags = _as_flags(flags_or_drug_a, drug_b, severity)
    if not flags:
        return (
            "Tell your doctor sheet\n"
            "English: No interaction flags were found for the recognized items.\n"
            "Simple pidgin: Abeg tell your doctor say no interaction flag dey available."
        )

    english_lines: list[str] = []
    pidgin_lines: list[str] = []
    for flag in flags:
        drug_a = str(flag.get("drug_a") or "unknown medicine")
        drug_b_value = str(flag.get("drug_b") or "unknown medicine")
        level = str(flag.get("severity") or "unspecified").lower()
        reason = str(flag.get("reason") or "No additional reason was recorded.")
        confidence = _confidence_text(flag.get("confidence"))
        english_lines.append(
            f"- Tell my doctor that {drug_a} and {drug_b_value} may have a "
            f"{level} interaction ({confidence}). Reason: {reason}"
        )
        if level in {"major", "contraindicated"}:
            pidgin_lines.append(
                f"- Oga doctor, abeg check {drug_a} with {drug_b_value}; "
                f"the interaction fit serious ({confidence}). Reason: {reason}"
            )
        else:
            pidgin_lines.append(
                f"- Oga doctor, abeg check how {drug_a} and {drug_b_value} "
                f"fit work together ({level}, {confidence}). Reason: {reason}"
            )

    return (
        "Tell your doctor sheet\n"
        "English:\n"
        + "\n".join(english_lines)
        + "\nSimple pidgin:\n"
        + "\n".join(pidgin_lines)
    )
