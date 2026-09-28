from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .vectors import as_vec, normalized


@dataclass(frozen=True)
class Ray:
    origin: np.ndarray
    direction: np.ndarray
    wavelength_um: float = 0.5876
    intensity: float = 1.0
    ray_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", as_vec(self.origin))
        object.__setattr__(self, "direction", normalized(self.direction))
        if self.wavelength_um <= 0.0:
            raise ValueError("wavelength must be positive")
        if self.intensity < 0.0:
            raise ValueError("intensity cannot be negative")


@dataclass(frozen=True)
class RayRecord:
    """A complete ray outcome, including failed or incomplete paths."""

    ray_id: str
    status: str
    events: list[dict[str, Any]]
    final_point: np.ndarray
    final_direction: np.ndarray
    final_intensity: float
    path_length: float
    message: str = ""

    @property
    def reached_image(self) -> bool:
        return self.status == "reached_image"

    def event_kinds(self) -> list[str]:
        return [event["kind"] for event in self.events]

