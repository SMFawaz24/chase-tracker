"""Validators that keep LLM output honest.

The LLM is only allowed to *phrase* facts that SQL has already verified. These checks reject any text that
introduces a number that is not in the facts, leaks the answer, or reveals a player's name.
"""
import re

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def normalize_number(text):
    text = text.replace(",", "")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def extract_numbers(text):
    return {normalize_number(m) for m in _NUM.findall(text)}


def allowed_from(values):
    """Set of number strings a text may mention, built from the verified fact values."""
    out = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            out.add(str(v))
        elif isinstance(v, float):
            out.update({normalize_number(f"{v:.2f}"), normalize_number(f"{v:.1f}"), str(int(round(v)))})
        else:
            out |= extract_numbers(str(v))
    return out


def numbers_grounded(text, allowed):
    """True when every number in `text` appears in `allowed`."""
    return extract_numbers(text) <= set(allowed)


def leaks_answer(text, answer):
    return answer.strip().lower() in text.lower()


def _name_tokens(name):
    return [t for t in re.split(r"[\s\-\.]+", name) if len(t) >= 3]


def leaks_name(text, name):
    """True when any part of a player's name (3+ letters) appears as a word in `text`."""
    return any(re.search(rf"\b{re.escape(t)}\b", text, re.IGNORECASE) for t in _name_tokens(name))


def name_pattern(name):
    """'V Kohli' -> 'V K____'; 'AB de Villiers' -> 'AB d_ V_______'."""
    parts = name.split()
    out = []
    for i, p in enumerate(parts):
        is_initials = p.isupper() and len(p) <= 3 and i < len(parts) - 1
        out.append(p if is_initials else p[0] + "_" * (len(p) - 1))
    return " ".join(out)
