"""Declarative live (mid-run tunable) parameters, shared by every task.

An env declares *which* config fields can change while it runs, and *when*
the change takes effect, as data. This module turns those declarations into
the accepted-key set, the value coercion and the apply loop, so no env needs
its own per-key ``if/elif`` ladder and the key set cannot drift from the
fields it describes.

Two timings exist:

``immediate``
    Applied as soon as ``update_live_params`` is called.
``reset``
    Staged and applied at the next ``reset()`` (for example terrain, which can
    only be rebuilt between episodes).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal

When = Literal["immediate", "reset"]
IMMEDIATE: When = "immediate"
AT_RESET: When = "reset"

# Dataclass field metadata key used by ``live_field``.
_LIVE_META = "live"


def cast_pair(value: Any) -> list[float]:
    """Coerce a ``[low, high]`` pair, e.g. a randomisation range."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("must be a pair of numbers")
    return [float(item) for item in value]


def cast_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("must be a mapping")
    return value


@dataclass(frozen=True)
class LiveParam:
    """One tunable key.

    ``cast`` of ``None`` means "keep the type the field currently has", which
    is how reward and termination dataclasses (all-live, mixed types) work.
    """

    key: str
    cast: Callable[[Any], Any] | None = None
    when: When = IMMEDIATE
    # Used when ``cast`` is None and the field has no current value yet.
    fallback: Callable[[Any], Any] | None = None

    def coerce(self, value: Any, current: Any = None) -> Any:
        if self.cast is not None:
            try:
                return self.cast(value)
            except ValueError as exc:
                raise ValueError(f"{self.key}: {exc}") from exc
        if current is None:
            return self.fallback(value) if self.fallback is not None else value
        return type(current)(value)


class LiveParamSet:
    """The set of live parameters one env accepts."""

    def __init__(self, params: Iterable[LiveParam]):
        self._params: dict[str, LiveParam] = {}
        for param in params:
            if param.key in self._params:
                raise ValueError(f"Duplicate live parameter {param.key!r}")
            self._params[param.key] = param

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(self._params)

    def __contains__(self, key: object) -> bool:
        return key in self._params

    def __getitem__(self, key: str) -> LiveParam:
        return self._params[key]

    def check(self, requested: Iterable[str]) -> None:
        unknown = set(requested) - self._params.keys()
        if unknown:
            raise ValueError(
                f"Unknown live parameters {sorted(unknown)}; "
                f"valid keys: {sorted(self._params)}"
            )

    def split(self, params: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Partition ``params`` into (immediate, at-reset) after validating keys."""
        self.check(params)
        now: dict[str, Any] = {}
        later: dict[str, Any] = {}
        for key, value in params.items():
            (later if self._params[key].when == AT_RESET else now)[key] = value
        return now, later


def live_field(
    default: Any,
    *,
    cast: Callable[[Any], Any] | None = None,
    when: When = IMMEDIATE,
):
    """A dataclass field that is tunable at run time.

    Use in place of ``field(default=...)``. ``cast`` defaults to the field's
    default's type.
    """
    from copy import copy
    from dataclasses import field

    meta = {_LIVE_META: {"cast": cast or type(default), "when": when}}
    if isinstance(default, (list, dict, set)):
        # Mutable defaults need a factory so instances do not share state.
        return field(default_factory=lambda: copy(default), metadata=meta)
    return field(default=default, metadata=meta)


def params_from_dataclass(section: str, cls: type, *, all_live: bool = False):
    """Derive ``LiveParam`` entries for a config dataclass.

    With ``all_live`` every field is tunable and keeps its runtime type
    (reward, termination). Otherwise only ``live_field`` fields are.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    out: list[LiveParam] = []
    for f in fields(cls):
        meta = f.metadata.get(_LIVE_META)
        if meta is not None:
            out.append(LiveParam(f"{section}.{f.name}", meta["cast"], meta["when"]))
        elif all_live:
            out.append(LiveParam(f"{section}.{f.name}"))
    return out


__all__ = [
    "AT_RESET",
    "IMMEDIATE",
    "LiveParam",
    "LiveParamSet",
    "cast_mapping",
    "cast_pair",
    "live_field",
    "params_from_dataclass",
]
