from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WalkerTermination:
    min_height: float = 0.38
    max_tilt_radians: float = 0.9
    max_steps: int = 1000

    def check(self, z_height: float, roll: float, pitch: float, step_count: int) -> tuple[bool, bool]:
        terminated = z_height < self.min_height or abs(roll) > self.max_tilt_radians or abs(pitch) > self.max_tilt_radians
        truncated = step_count >= self.max_steps
        return terminated, truncated
