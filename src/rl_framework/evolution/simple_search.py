from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EvolutionResult:
    best_params: dict[str, Any]
    best_score: float


class RandomMorphologySearch:
    """Optional evolutionary hook to mutate morphology parameters across trials."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def mutate(self, morphology_cfg: dict[str, Any]) -> dict[str, Any]:
        out = dict(morphology_cfg)
        out["base_size"] = float(np.clip(out.get("base_size", 1.0) + self.rng.normal(0, 0.1), 0.5, 2.0))
        out["health"] = float(np.clip(out.get("health", 1.0) + self.rng.normal(0, 0.1), 0.5, 3.0))
        return out
