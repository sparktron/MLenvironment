from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DomainRandomizationConfig:
    mass_scale_range: tuple[float, float] = (1.0, 1.0)
    friction_range: tuple[float, float] = (0.8, 1.2)
    sensor_noise_std: float = 0.0
    action_latency_steps: int = 0


@dataclass
class CurriculumConfig:
    enabled: bool = False
    level: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EnvContext:
    seed: int = 0
    sim_config: Dict[str, Any] = field(default_factory=dict)
    reward_config: Dict[str, Any] = field(default_factory=dict)
    termination_config: Dict[str, Any] = field(default_factory=dict)
    domain_randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
