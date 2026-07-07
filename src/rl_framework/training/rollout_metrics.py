"""Read training metrics reliably at rollout boundaries.

SB3's ``Logger.name_to_value`` is a ``defaultdict(float)`` that ``dump()``
clears, and ``OnPolicyAlgorithm.learn`` orders one iteration as
``collect_rollouts`` (which ends with ``callback.on_rollout_end()``) →
``dump_logs`` (which records ``rollout/ep_rew_mean`` / ``rollout/ep_len_mean``
and immediately dumps) → ``train``. So at ``on_rollout_end`` the ``rollout/*``
keys are never present: an indexed read silently yields the defaultdict's 0.0
*and* inserts the key, polluting the next TensorBoard dump. Callbacks must
instead take episode stats from ``model.ep_info_buffer`` (the same source
``dump_logs`` reads) and treat any other logger key as absent unless it is
actually in the mapping.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Metric name -> ep_info_buffer field ("r" = episode reward, "l" = length).
_EP_INFO_FIELDS = {"rollout/ep_rew_mean": "r", "rollout/ep_len_mean": "l"}


def ep_info_mean(model: Any, key: str) -> float | None:
    """Mean episode stat for *key* from ``model.ep_info_buffer``, or ``None``.

    Returns ``None`` when the buffer is missing or empty (no completed
    episodes yet) — callers must treat that as "no data", never as 0.0.
    """
    field = _EP_INFO_FIELDS.get(key)
    buffer = getattr(model, "ep_info_buffer", None)
    if field is None or not buffer:
        return None
    values = [info[field] for info in buffer if field in info]
    if not values:
        return None
    return float(np.mean(values))


def logger_value(logger: Any, key: str) -> float | None:
    """Value of *key* from the SB3 logger, or ``None`` when absent.

    The membership check matters: ``name_to_value`` is a ``defaultdict``, so a
    plain index would fabricate (and store) 0.0 for a missing key.
    """
    try:
        values = logger.name_to_value
    except AttributeError:
        return None
    if key not in values:
        return None
    try:
        return float(values[key])
    except (TypeError, ValueError):
        return None


def rollout_metric(model: Any, logger: Any, key: str) -> float | None:
    """Resolve *key* at a rollout boundary, or ``None`` when unavailable.

    ``rollout/ep_rew_mean`` / ``rollout/ep_len_mean`` come from the model's
    ``ep_info_buffer``; everything else (e.g. ``arena/agent_0_win_rate``,
    ``train/*``) from the logger, absent-safe.
    """
    if key in _EP_INFO_FIELDS:
        return ep_info_mean(model, key)
    return logger_value(logger, key)
