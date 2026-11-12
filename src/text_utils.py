"""text normalisation shared by grounding checks and requirement matching."""

import re
import unicodedata


def normalize_for_match(text: str) -> str:
    """lowercase, drop punctuation, collapse spaces; padded so ' term ' matches whole tokens."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9+#.]+", " ", text)
    text = re.sub(r"\.+(?=\s|$)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return f" {text} " if text else " "


def contains_term(haystack_norm: str, term: str) -> bool:
    """whole-token containment between an already-normalised haystack and a raw term."""
    term_norm = normalize_for_match(term)
    if not term_norm.strip():
        return False
    return term_norm in haystack_norm
