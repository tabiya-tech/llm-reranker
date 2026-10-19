from __future__ import annotations

import re

REMOTE_KEY = "remote"

# Separators between several regions in one field, e.g. "Nairobi, Busia, Kisumu" or "National/Remote".
_PART_SEPARATORS = re.compile(r"[,;/]")
# Punctuation left around a region name by the scraper, e.g. "CABA)" or "Lusaka.".
_EDGE_PUNCTUATION = " \t\n.()[]\"'"
_WHITESPACE = re.compile(r"\s+")


def canonical_location(value: str | None) -> str:
    """One region name as a key: punctuation-trimmed, whitespace-collapsed, casefolded."""
    if not value:
        return ""
    return _WHITESPACE.sub(" ", value.strip(_EDGE_PUNCTUATION)).casefold()


def location_keys_for(*values: str | None) -> tuple[str, ...]:
    """The sorted, de-duplicated location keys of a place given as one or more strings."""
    keys: set[str] = set()
    for value in values:
        if not value:
            continue
        if REMOTE_KEY in value.casefold():
            keys.add(REMOTE_KEY)
        for part in _PART_SEPARATORS.split(value):
            key = canonical_location(part)
            if key:
                keys.add(key)
    return tuple(sorted(keys))
