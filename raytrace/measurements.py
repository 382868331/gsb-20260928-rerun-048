"""Measured interval responses and correlated uncertainty arithmetic.

Measurement crews provide response *intervals* per optical element,
wavelength and polarization channel instead of deterministic constants.
Each measured interval becomes one uncertainty symbol; propagation uses
affine arithmetic so that two beams derived from the same measurement keep
their correlation when they recombine (the shared symbol cancels instead of
being added as an independent error).

A channel that has not been measured yields no interval at all; callers
must treat it as unknown rather than defaulting to lossless transmission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class AffineQuantity:
    """constant + sum(coeff * symbol), every symbol ranging over [-1, 1].

    Symbols name shared uncertainty sources (one per measured interval, or
    one per nonlinear product term).  Adding two quantities that share a
    symbol adds their coefficients, so correlated contributions recombine
    exactly instead of widening like independent errors.
    """

    constant: float
    coeffs: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coeffs",
            {symbol: value for symbol, value in self.coeffs.items() if value != 0.0},
        )

    @property
    def is_certain(self) -> bool:
        return not self.coeffs

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.coeffs))

    def interval(self) -> tuple[float, float]:
        radius = sum(abs(value) for value in self.coeffs.values())
        return (self.constant - radius, self.constant + radius)

    def _combine(self, other: "AffineQuantity", sign: float) -> "AffineQuantity":
        coeffs = dict(self.coeffs)
        for symbol, value in other.coeffs.items():
            coeffs[symbol] = coeffs.get(symbol, 0.0) + sign * value
        return AffineQuantity(self.constant + sign * other.constant, coeffs)

    def __add__(self, other: "AffineQuantity") -> "AffineQuantity":
        return self._combine(other, 1.0)

    def __sub__(self, other: "AffineQuantity") -> "AffineQuantity":
        return self._combine(other, -1.0)

    def scale(self, factor: float) -> "AffineQuantity":
        return AffineQuantity(
            self.constant * factor,
            {symbol: value * factor for symbol, value in self.coeffs.items()},
        )

    def mul(self, other: "AffineQuantity", noise_symbol: str | None = None) -> "AffineQuantity":
        """Affine-arithmetic product.

        Linear terms keep the original symbols (correlation preserved); the
        quadratic residual is bounded by one fresh noise symbol so the
        result stays a sound outer approximation.
        """

        coeffs: dict[str, float] = {}
        for symbol, value in self.coeffs.items():
            coeffs[symbol] = coeffs.get(symbol, 0.0) + value * other.constant
        for symbol, value in other.coeffs.items():
            coeffs[symbol] = coeffs.get(symbol, 0.0) + value * self.constant
        residual = sum(
            abs(a) * abs(b)
            for a in self.coeffs.values()
            for b in other.coeffs.values()
        )
        if residual > 0.0:
            if noise_symbol is None:
                raise ValueError("a noise symbol is required for nonlinear products")
            coeffs[noise_symbol] = coeffs.get(noise_symbol, 0.0) + residual
        return AffineQuantity(self.constant * other.constant, coeffs)

    def to_dict(self) -> dict[str, Any]:
        return {"constant": self.constant, "coeffs": dict(self.coeffs)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "AffineQuantity | None":
        if data is None:
            return None
        return cls(float(data["constant"]), dict(data.get("coeffs", {})))


_QUANTITIES = ("reflectivity", "transmissivity", "absorptivity")


class InfeasibleMeasurementError(ValueError):
    """Raised when measured response intervals cannot conserve energy."""


@dataclass(frozen=True)
class ResponseMeasurement:
    """One measured response interval for an element/channel."""

    surface: str
    quantity: str  # "reflectivity" | "transmissivity" | "absorptivity"
    lower: float
    upper: float
    wavelength_nm: float | None = None  # None = broadband (applies to any wavelength)
    polarization: str | None = None  # "s" | "p" | None (scalar/unpolarized)
    note: str = ""

    def __post_init__(self) -> None:
        if self.quantity not in _QUANTITIES:
            raise ValueError(f"quantity must be one of {_QUANTITIES}, got {self.quantity!r}")
        if self.polarization is not None and self.polarization not in ("s", "p"):
            raise ValueError("polarization must be 's', 'p' or None")
        if self.wavelength_nm is not None and self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")
        if not 0.0 <= self.lower <= self.upper <= 1.0:
            raise ValueError("measurement interval must satisfy 0 <= lower <= upper <= 1")

    @property
    def key(self) -> tuple:
        return (self.surface, self.quantity, self.wavelength_nm, self.polarization)

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    def symbol(self, batch_id: str) -> str:
        wavelength = "any" if self.wavelength_nm is None else f"{self.wavelength_nm:g}"
        polarization = self.polarization or "scalar"
        return f"{batch_id}|{self.surface}|{self.quantity}|{wavelength}|{polarization}"

    def to_affine(self, batch_id: str) -> AffineQuantity:
        return AffineQuantity(
            self.midpoint,
            {self.symbol(batch_id): 0.5 * (self.upper - self.lower)},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "quantity": self.quantity,
            "lower": self.lower,
            "upper": self.upper,
            "wavelength_nm": self.wavelength_nm,
            "polarization": self.polarization,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResponseMeasurement":
        return cls(
            surface=data["surface"],
            quantity=data["quantity"],
            lower=float(data["lower"]),
            upper=float(data["upper"]),
            wavelength_nm=data.get("wavelength_nm"),
            polarization=data.get("polarization"),
            note=data.get("note", ""),
        )


@dataclass
class MeasurementBatch:
    """A named, provenance-tracked set of measured response intervals.

    Batches are mutually exclusive when they disagree (disjoint intervals)
    on the same element/channel; only one batch is active for a trace, and
    revoking a batch never rewrites the provenance recorded in older
    projects or trace batches.
    """

    batch_id: str
    measurements: tuple[ResponseMeasurement, ...] = ()
    label: str = ""
    revoked: bool = False

    def __post_init__(self) -> None:
        self.measurements = tuple(self.measurements)
        seen: set[tuple] = set()
        for measurement in self.measurements:
            if measurement.key in seen:
                raise ValueError(f"duplicate measurement for {measurement.key!r}")
            seen.add(measurement.key)
        conflicts = self.energy_conflicts(self.measurements)
        if conflicts:
            conflict = conflicts[0]
            channels = ", ".join(conflict["quantities"])
            raise InfeasibleMeasurementError(
                f"infeasible measurements for {conflict['surface']!r} "
                f"({conflict['wavelength_description']}, "
                f"{conflict['polarization_description']}): conflicting channels "
                f"{channels} have lower bounds summing to {conflict['lower_sum']:g} > 1"
            )

    @staticmethod
    def energy_conflicts(measurements: tuple[ResponseMeasurement, ...]) -> list[dict[str, Any]]:
        """Return response channels whose measured intervals make R+T+A impossible."""

        conflicts: list[dict[str, Any]] = []
        surfaces = sorted({measurement.surface for measurement in measurements})
        wavelengths = [None] + sorted({
            measurement.wavelength_nm
            for measurement in measurements
            if measurement.wavelength_nm is not None
        })
        polarizations = [None, "s", "p"]
        reported: set[frozenset[tuple]] = set()

        for surface in surfaces:
            surface_measurements = [
                measurement for measurement in measurements if measurement.surface == surface
            ]
            for wavelength_nm in wavelengths:
                for polarization in polarizations:
                    quantities: dict[str, ResponseMeasurement] = {}
                    for measurement in surface_measurements:
                        if measurement.wavelength_nm is not None and (
                            wavelength_nm is None
                            or float(measurement.wavelength_nm) != float(wavelength_nm)
                        ):
                            continue
                        if measurement.polarization is not None and (
                            polarization is None or measurement.polarization != polarization
                        ):
                            continue
                        previous = quantities.get(measurement.quantity)
                        if previous is None:
                            quantities[measurement.quantity] = measurement
                            continue
                        specificity = (
                            (measurement.wavelength_nm is not None)
                            + (measurement.polarization is not None)
                        )
                        previous_specificity = (
                            (previous.wavelength_nm is not None)
                            + (previous.polarization is not None)
                        )
                        if specificity > previous_specificity:
                            quantities[measurement.quantity] = measurement

                    if len(quantities) < 2:
                        continue
                    selected_keys = frozenset(measurement.key for measurement in quantities.values())
                    if selected_keys in reported:
                        continue
                    reported.add(selected_keys)
                    lower_sum = sum(quantity.lower for quantity in quantities.values())
                    upper_sum = sum(quantity.upper for quantity in quantities.values())
                    measured_all_three = set(quantities) == set(_QUANTITIES)
                    infeasible = lower_sum > 1.0 + 1e-12 or (
                        measured_all_three and upper_sum < 1.0 - 1e-12
                    )
                    if not infeasible:
                        continue
                    conflicts.append({
                        "surface": surface,
                        "wavelength_nm": wavelength_nm,
                        "polarization": polarization,
                        "wavelength_description": (
                            "broadband" if wavelength_nm is None else f"{wavelength_nm:g} nm"
                        ),
                        "polarization_description": polarization or "scalar",
                        "quantities": [
                            quantity
                            for quantity in _QUANTITIES
                            if quantity in quantities
                        ],
                        "intervals": {
                            name: [measurement.lower, measurement.upper]
                            for name, measurement in quantities.items()
                        },
                        "lower_sum": lower_sum,
                        "upper_sum": upper_sum,
                    })
        return conflicts

    def surfaces_declared(self) -> set[str]:
        return {measurement.surface for measurement in self.measurements}

    def _candidate(self, surface: str, quantity: str, wavelength_nm: float | None,
                   polarization: str | None) -> ResponseMeasurement | None:
        """Most specific applicable measurement, or None when unmeasured.

        Exact wavelength beats broadband; an exact polarization entry beats
        the scalar one.  A scalar entry may serve a polarization query (it
        is still measured data), but a missing channel never falls back to
        a fabricated lossless value.
        """

        best: tuple[int, ResponseMeasurement] | None = None
        for measurement in self.measurements:
            if measurement.surface != surface or measurement.quantity != quantity:
                continue
            if measurement.wavelength_nm is not None:
                if wavelength_nm is None or float(measurement.wavelength_nm) != float(wavelength_nm):
                    continue
            if measurement.polarization is not None and measurement.polarization != polarization:
                continue
            specificity = (measurement.wavelength_nm is not None) + (measurement.polarization is not None)
            if best is None or specificity > best[0]:
                best = (specificity, measurement)
        return best[1] if best else None

    def interval(self, surface: str, quantity: str, wavelength_nm: float | None,
                 polarization: str | None) -> AffineQuantity | None:
        measurement = self._candidate(surface, quantity, wavelength_nm, polarization)
        if measurement is None:
            return None
        return measurement.to_affine(self.batch_id)

    def conflicts_with(self, other: "MeasurementBatch") -> list[dict[str, Any]]:
        """Mutually exclusive conditions: same key, disjoint intervals."""

        conflicts: list[dict[str, Any]] = []
        by_key: dict[tuple, ResponseMeasurement] = {m.key: m for m in self.measurements}
        for measurement in other.measurements:
            mine = by_key.get(measurement.key)
            if mine is None:
                continue
            if mine.upper < measurement.lower or measurement.upper < mine.lower:
                surface, quantity, wavelength_nm, polarization = measurement.key
                conflicts.append({
                    "surface": surface,
                    "quantity": quantity,
                    "wavelength_nm": wavelength_nm,
                    "polarization": polarization,
                    "batches": [self.batch_id, other.batch_id],
                    "intervals": [[mine.lower, mine.upper], [measurement.lower, measurement.upper]],
                })
        return conflicts

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "label": self.label,
            "revoked": self.revoked,
            "measurements": [measurement.to_dict() for measurement in self.measurements],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MeasurementBatch":
        return cls(
            batch_id=data["batch_id"],
            label=data.get("label", ""),
            revoked=bool(data.get("revoked", False)),
            measurements=tuple(
                ResponseMeasurement.from_dict(item) for item in data.get("measurements", [])
            ),
        )
