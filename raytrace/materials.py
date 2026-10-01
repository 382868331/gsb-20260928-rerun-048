"""Material and refractive-index rules.

The initial model uses scalar refractive indices.  Wavelength is carried by
each ray so dispersive materials can be added without changing the tracer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping


class UnknownMaterialResponse(ValueError):
    """Raised when a material has no verified response at a wavelength."""


@dataclass(frozen=True)
class Material:
    name: str
    index: float = 1.0
    spectral_indices: Mapping[float, float] = field(default_factory=dict)

    def refractive_index(self, wavelength_nm: float | None = None) -> float:
        if self.index <= 0:
            raise ValueError(f"material {self.name!r} must have a positive index")
        if wavelength_nm is not None:
            if wavelength_nm <= 0:
                raise ValueError("wavelength_nm must be positive")
            spectral = dict(self.spectral_indices)
            for value in spectral.values():
                if value <= 0:
                    raise ValueError(f"material {self.name!r} must have positive spectral indices")
            if spectral:
                try:
                    return spectral[float(wavelength_nm)]
                except KeyError:
                    raise UnknownMaterialResponse(
                        f"material {self.name!r} has no verified response at {wavelength_nm:g} nm"
                    ) from None
        return self.index


class MaterialLibrary:
    def __init__(self, materials: Mapping[str, Material] | None = None):
        self.materials: dict[str, Material] = dict(materials or {})
        if "air" not in self.materials:
            self.add(Material("air", 1.0))

    def add(self, material: Material) -> None:
        self.materials[material.name] = material

    def get(self, name: str) -> Material:
        try:
            return self.materials[name]
        except KeyError as exc:
            raise KeyError(f"unknown material {name!r}") from exc

    def index(self, name: str, wavelength_nm: float | None = None) -> float:
        return self.get(name).refractive_index(wavelength_nm)

    def to_dict(self) -> dict:
        data = {}
        for name, material in self.materials.items():
            item = vars(material).copy()
            item["spectral_indices"] = {
                str(wavelength): index for wavelength, index in material.spectral_indices.items()
            }
            data[name] = item
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialLibrary":
        materials = {}
        for name, values in data.items():
            item = dict(values)
            spectral = {
                float(wavelength): float(index)
                for wavelength, index in item.pop("spectral_indices", {}).items()
            }
            materials[name] = Material(spectral_indices=spectral, **item)
        return cls(materials)


def fresnel_coefficients(n1: float, n2: float, cos_i: float) -> tuple[float, float]:
    """Return unpolarized reflected and transmitted power fractions.

    Absorption is not included.  The caller is responsible for detecting total
    internal reflection before calling this function.
    """

    cos_i = min(1.0, abs(cos_i))
    sin_i2 = 1.0 - cos_i * cos_i
    cos_t2 = 1.0 - (n1 / n2) ** 2 * sin_i2
    if cos_t2 <= 0:
        return 1.0, 0.0
    cos_t = math.sqrt(cos_t2)
    r_s = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    r_p = ((n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)) ** 2
    reflected = 0.5 * (r_s + r_p)
    return reflected, 1.0 - reflected
