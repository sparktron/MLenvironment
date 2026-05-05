from __future__ import annotations

from typing import Any


def set_nested(d: dict[str, Any], key: str, value: Any, *, strict: bool = True) -> None:
    """Set a value in a nested dict using a dotted key path.

    Parameters
    ----------
    strict:
        When True (default) raises KeyError if any part of the key path is
        missing in *d*.  When False, missing intermediate dicts are created
        via ``setdefault`` so new keys can be introduced.
    """
    keys = key.split(".")
    cur = d
    for k in keys[:-1]:
        if strict:
            if not isinstance(cur, dict) or k not in cur:
                raise KeyError(f"Key path '{key}': intermediate key '{k}' not found in config")
            cur = cur[k]
        else:
            cur = cur.setdefault(k, {})
    if strict and (not isinstance(cur, dict) or keys[-1] not in cur):
        raise KeyError(f"Key path '{key}': leaf key '{keys[-1]}' not found in config")
    cur[keys[-1]] = value
