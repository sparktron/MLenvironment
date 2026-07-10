from __future__ import annotations

from typing import Any


def get_section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    """Return ``cfg[key]`` as a dict, treating a missing key and an explicit
    ``null`` value the same way: as an empty section.

    The GUI wizard has historically written ``key: null`` for an empty nested
    config group instead of omitting the key. ``cfg.get(key, {})`` does not
    help there — a *present* key whose value is ``None`` is returned as
    ``None``, not the default, so every ``.get(...) or {}`` /
    ``.setdefault(...)`` callsite across the env and callback modules has to
    remember this separately (and several didn't). Centralizing the
    normalization here — and writing the resolved dict back into *cfg* — lets
    callers treat "missing" and "null" identically and upgrades the stored
    config in place so later direct reads of ``cfg[key]`` are safe too.
    """
    value = cfg.get(key)
    if value is None:
        value = {}
        cfg[key] = value
    return value


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
                raise KeyError(
                    f"Key path '{key}': intermediate key '{k}' not found in config"
                )
            cur = cur[k]
        else:
            # get_section also normalizes an explicit `k: null` intermediate
            # to {} — plain setdefault would leave it None and crash the
            # next iteration/assignment.
            cur = get_section(cur, k)
    if strict and (not isinstance(cur, dict) or keys[-1] not in cur):
        raise KeyError(f"Key path '{key}': leaf key '{keys[-1]}' not found in config")
    cur[keys[-1]] = value
