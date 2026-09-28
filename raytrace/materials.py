"""Wavelength-dependent refractive-index rules.

The tracer is scalar and geometric. Dispersion is represented by refractive
indices only; absorption, polarization and bulk scattering are not modeled.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


class Material(Protocol):
    name: str

    def refractive_index(self, wavelength_um: float) -> float: ...


@dataclass(frozen=True)
class Air:
    """Vacuum-like unity index; adequate for the simplified model."""

    name: str = "air"

    def refractive_index(self, wavelength_um: float) -> float:
        return 1.0


@dataclass(frozen=True)
class ConstantMaterial:
    name: str
    index: float

    def __post_init__(self) -> None:
        if self.index <= 0.0:
            raise ValueError("refractive index must be positive")

    def refractive_index(self, wavelength_um: float) -> float:
        return float(self.index)


@dataclass(frozen=True)
class SellmeierMaterial:
    """Sellmeier equation using wavelengths in micrometres."""

    name: str
    b1: float
    b2: float
    b3: float
    c1: float
    c2: float
    c3: float

    def refractive_index(self, wavelength_um: float) -> float:
        w = float(wavelength_um)
        if w <= 0.0:
            raise ValueError("wavelength must be positive")
        w2 = w * w
        value = (
            1.0
            + self.b1 * w2 / (w2 - self.c1)
            + self.b2 * w2 / (w2 - self.c2)
            + self.b3 * w2 / (w2 - self.c3)
        )
        if value <= 0.0:
            raise ValueError(f"invalid Sellmeier index for {self.name}")
        return math.sqrt(value)


BK7 = SellmeierMaterial(
    "N-BK7",
    1.03961217,
    0.231792344,
    1.01046945,
    0.00600069867,
    0.0200179144,
    103.560653,
)

