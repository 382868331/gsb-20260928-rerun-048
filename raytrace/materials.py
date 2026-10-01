"""Material and wavelength-dependent refractive-index rules.

A material may be achromatic (a single scalar index) or spectral: an
explicit, measured ``index_by_wavelength`` table, optionally with a
``loss_by_wavelength`` table of bulk loss fractions applied on
transmission into the material.  Wavelength response is *never*
interpolated or inferred from neighbouring values: rays whose wavelength
is missing from a spectral table are rejected with
:class:`UnknownWavelengthError` so an unverified wavelength can never be
silently treated as a validated one.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping


class UnknownWavelengthError(ValueError):
    """A ray wavelength has no verified response table entry.

    Subclassing :class:`ValueError` keeps callers that already reject bad
    numeric material parameters working unchanged.
    """


_WAVELENGTH_REL_TOLERANCE = 1e-9
_WAVELENGTH_ABS_TOLERANCE_NM = 1e-6


def _wavelength_key(wavelength_nm: float | None, table: Mapping[float, float], material: str, response: str) -> float:
    """Find an exact (within rounding) measured wavelength; never interpolate."""

    if wavelength_nm is None:
        raise UnknownWavelengthError(
            f"material {material!r} has a wavelength-dependent {response} table but the ray "
            "carries no wavelength_nm"
        )
    for tabulated in table:
        if math.isclose(
            float(tabulated),
            float(wavelength_nm),
            rel_tol=_WAVELENGTH_REL_TOLERANCE,
            abs_tol=_WAVELENGTH_ABS_TOLERANCE_NM,
        ):
            return float(tabulated)
    verified = ", ".join(f"{value:g} nm" for value in sorted(table))
    raise UnknownWavelengthError(
        f"material {material!r} has no verified {response} at {wavelength_nm:g} nm "
        f"(verified wavelengths: {verified}); refusing to use a neighbouring value"
    )


@dataclass(frozen=True)
class Material:
    name: str
    index: float = 1.0
    index_by_wavelength: Dict[float, float] | None = None
    loss_by_wavelength: Dict[float, float] | None = None

    def __post_init__(self) -> None:
        if self.index_by_wavelength:
            for wavelength, value in self.index_by_wavelength.items():
                if not float(value) > 0:
                    raise ValueError(
                        f"material {self.name!r} must have a positive index at {float(wavelength):g} nm"
                    )
        elif not self.index > 0:
            raise ValueError(f"material {self.name!r} must have a positive index")
        if self.loss_by_wavelength:
            for wavelength, value in self.loss_by_wavelength.items():
                if not 0.0 <= float(value) <= 1.0:
                    raise ValueError(
                        f"material {self.name!r} loss at {float(wavelength):g} nm must be within [0, 1]"
                    )
            if self.index_by_wavelength and set(self.loss_by_wavelength) - set(self.index_by_wavelength):
                raise ValueError(
                    f"material {self.name!r} loss wavelengths must be a subset of verified index wavelengths"
                )

    @property
    def spectral(self) -> bool:
        return bool(self.index_by_wavelength)

    def refractive_index(self, wavelength_nm: float | None = None) -> float:
        if not self.index_by_wavelength:
            return self.index
        key = _wavelength_key(wavelength_nm, self.index_by_wavelength, self.name, "refractive index")
        return float(self.index_by_wavelength[key])

    def loss_fraction(self, wavelength_nm: float | None = None) -> float:
        """Bulk loss applied when transmitting into this material.

        Achromatic materials (the original model) have no bulk loss.  A
        spectral material without a loss table still requires the
        wavelength to be present in its index table: a missing wavelength
        must be rejected, not silently treated as lossless.
        """

        if not self.loss_by_wavelength:
            if not self.index_by_wavelength:
                return 0.0
            # Validate that the wavelength is verified for this material
            # even when no explicit loss response was measured.
            self.refractive_index(wavelength_nm)
            return 0.0
        key = _wavelength_key(wavelength_nm, self.loss_by_wavelength, self.name, "loss")
        return float(self.loss_by_wavelength[key])

    def verified_wavelengths(self) -> tuple[float, ...]:
        if not self.index_by_wavelength:
            return ()
        return tuple(sorted(self.index_by_wavelength))

    def to_dict(self) -> dict:
        if not self.index_by_wavelength and not self.loss_by_wavelength:
            return {"name": self.name, "index": self.index}
        data: dict[str, Any] = {"name": self.name}
        if self.index_by_wavelength:
            data["index_by_wavelength"] = {str(k): v for k, v in self.index_by_wavelength.items()}
        else:
            data["index"] = self.index
        if self.loss_by_wavelength:
            data["loss_by_wavelength"] = {str(k): v for k, v in self.loss_by_wavelength.items()}
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Material":
        values = dict(data)
        values["index_by_wavelength"] = _coerce_table(values.get("index_by_wavelength"))
        values["loss_by_wavelength"] = _coerce_table(values.get("loss_by_wavelength"))
        return cls(**values)


def _coerce_table(table: Mapping[str, float] | None) -> Dict[float, float] | None:
    if table is None:
        return None
    return {float(wavelength): float(value) for wavelength, value in table.items()}


class MaterialLibrary:
    def __init__(self, materials: Dict[str, Material] | None = None):
        self.materials: Dict[str, Material] = dict(materials or {})
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

    def loss(self, name: str, wavelength_nm: float | None = None) -> float:
        return self.get(name).loss_fraction(wavelength_nm)

    def to_dict(self) -> dict:
        return {name: material.to_dict() for name, material in self.materials.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialLibrary":
        return cls({name: Material.from_dict(values) for name, values in data.items()})


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
