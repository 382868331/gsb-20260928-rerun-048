"""Measured optical-element responses as energy-constrained intervals.

Metrology provides per-wavelength, per-polarization-channel response
*intervals* for an element (reflectivity / transmissivity / absorptivity)
instead of deterministic constants.  Intervals of one channel must satisfy
the energy constraint jointly -- R + T + A <= 1 -- so the upper bounds are
tightened against each other instead of allowing an impossible total.

Uncertain quantities propagate as affine forms: every measured quantity is
a shared symbol, so when a beam splits and later recombines the correlated
uncertainty cancels instead of being summed as independent error.

Several measurement batches may cover the same element with mutually
inconsistent values.  A MeasurementBook keeps them all, resolves coverage
by active-batch priority, and supports revoking a batch; channels left
without any active measurement stay *unknown* and are never silently
treated as lossless.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping


class InconsistentMeasurement(ValueError):
    """Raised when a measured channel cannot satisfy the energy constraint."""


@dataclass(frozen=True)
class Interval:
    """Closed interval [low, high] with 0 <= low <= high <= 1."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.low <= self.high <= 1.0 + 1e-12:
            raise ValueError(
                f"response interval must satisfy 0 <= low <= high <= 1, "
                f"got [{self.low}, {self.high}]"
            )

    @property
    def center(self) -> float:
        return 0.5 * (self.low + self.high)

    @property
    def radius(self) -> float:
        return 0.5 * (self.high - self.low)

    @property
    def is_definite(self) -> bool:
        return self.high - self.low <= 1e-15

    def tightened_high(self, cap: float) -> "Interval":
        high = min(self.high, cap)
        if high < self.low - 1e-12:
            raise InconsistentMeasurement(
                f"energy constraint caps this channel at {cap:g}, "
                f"below its measured lower bound {self.low:g}"
            )
        return Interval(self.low, max(self.low, high))

    def overlaps(self, other: "Interval") -> bool:
        return self.low <= other.high + 1e-12 and other.low <= self.high + 1e-12

    def to_dict(self) -> dict:
        return {"low": self.low, "high": self.high}

    @classmethod
    def from_dict(cls, data) -> "Interval":
        if isinstance(data, (int, float)):
            return cls(float(data), float(data))
        return cls(float(data["low"]), float(data["high"]))


class AffineForm:
    """center + sum(coefficient * symbol); symbols share epsilon in [-1, 1].

    Every symbol names one measured quantity of one measurement batch, so
    two branches that inherit the same symbol stay correlated: summing
    their affine forms (e.g. recombined beams on one detector) cancels the
    shared uncertainty instead of widening it like independent intervals.
    """

    __slots__ = ("center", "terms")

    def __init__(self, center: float, terms: Mapping[str, float] | None = None):
        self.center = float(center)
        self.terms = {symbol: float(coef) for symbol, coef in (terms or {}).items() if coef != 0.0}

    @classmethod
    def constant(cls, value: float) -> "AffineForm":
        return cls(value)

    @classmethod
    def from_interval(cls, interval: Interval, symbol: str) -> "AffineForm":
        if interval.is_definite:
            return cls(interval.center)
        return cls(interval.center, {symbol: interval.radius})

    def __add__(self, other: "AffineForm") -> "AffineForm":
        terms = dict(self.terms)
        for symbol, coef in other.terms.items():
            terms[symbol] = terms.get(symbol, 0.0) + coef
        return AffineForm(self.center + other.center, terms)

    def __radd__(self, other) -> "AffineForm":
        return self if other == 0 else self.__add__(other)

    def negate(self) -> "AffineForm":
        return AffineForm(-self.center, {s: -c for s, c in self.terms.items()})

    def __sub__(self, other: "AffineForm") -> "AffineForm":
        return self.__add__(other.negate())

    def scale(self, factor: float) -> "AffineForm":
        return AffineForm(self.center * factor, {s: c * factor for s, c in self.terms.items()})

    def multiply(self, other: "AffineForm", noise_symbol: str | None = None) -> "AffineForm":
        """Affine product; the second-order residual becomes a fresh symbol.

        The residual radius rad(self) * rad(other) is conservatively enclosed
        under noise_symbol (a unique symbol per multiplication site), so
        first-order correlation is preserved exactly and only genuinely
        nonlinear coupling is treated as new uncertainty.
        """

        center = self.center * other.center
        terms: dict[str, float] = {}
        for symbol, coef in self.terms.items():
            terms[symbol] = terms.get(symbol, 0.0) + coef * other.center
        for symbol, coef in other.terms.items():
            terms[symbol] = terms.get(symbol, 0.0) + coef * self.center
        residual = self.radius() * other.radius()
        if residual > 0.0:
            symbol = noise_symbol or f"nonlinear:{id(self):x}:{id(other):x}"
            terms[symbol] = terms.get(symbol, 0.0) + residual
        return AffineForm(center, terms)

    def radius(self) -> float:
        return sum(abs(coef) for coef in self.terms.values())

    @property
    def is_definite(self) -> bool:
        return not self.terms

    def to_interval(self) -> tuple[float, float]:
        radius = self.radius()
        return self.center - radius, self.center + radius

    def to_dict(self) -> dict:
        return {"center": self.center, "terms": dict(self.terms)}

    @classmethod
    def from_dict(cls, data: dict) -> "AffineForm":
        return cls(data["center"], data.get("terms"))


@dataclass(frozen=True)
class ChannelResponse:
    """Measured R/T/A intervals for one (element, wavelength, axis) channel.

    Any subset of the three quantities may be measured as long as at least
    two are present; the remaining quantity is derived as the energy
    complement.  Upper bounds are tightened so that R + T + A <= 1 always
    holds jointly -- taking every channel's upper bound independently can
    never yield an impossible total energy.
    """

    reflectivity: Interval | None = None
    transmissivity: Interval | None = None
    absorptivity: Interval | None = None

    def __post_init__(self) -> None:
        measured = [
            ("reflectivity", self.reflectivity),
            ("transmissivity", self.transmissivity),
            ("absorptivity", self.absorptivity),
        ]
        present = [(name, interval) for name, interval in measured if interval is not None]
        if len(present) < 2:
            raise InconsistentMeasurement(
                "a channel needs at least two of reflectivity/transmissivity/"
                "absorptivity measured; the remaining quantity is then fixed "
                "by the energy constraint"
            )
        low_sum = sum(interval.low for _, interval in present)
        if low_sum > 1.0 + 1e-12:
            raise InconsistentMeasurement(
                f"measured lower bounds sum to {low_sum:g} > 1: "
                "the channel violates energy conservation"
            )
        tightened = {}
        for name, interval in measured:
            if interval is None:
                tightened[name] = None
                continue
            others_low = sum(
                other.low for other_name, other in present if other_name != name
            )
            tightened[name] = interval.tightened_high(1.0 - others_low)
        if all(tightened[name] is not None for name, _ in measured):
            derived = self._derive(tightened["reflectivity"], tightened["transmissivity"])
            if not derived.overlaps(tightened["absorptivity"]):
                raise InconsistentMeasurement(
                    f"measured absorptivity {tightened['absorptivity']} is "
                    f"incompatible with 1 - R - T = {derived}"
                )
        object.__setattr__(self, "reflectivity", tightened["reflectivity"])
        object.__setattr__(self, "transmissivity", tightened["transmissivity"])
        object.__setattr__(self, "absorptivity", tightened["absorptivity"])

    @staticmethod
    def _derive(first: Interval, second: Interval) -> Interval:
        low = max(0.0, 1.0 - first.high - second.high)
        high = max(low, 1.0 - first.low - second.low)
        return Interval(low, min(1.0, high))

    def resolved(self) -> tuple[Interval, Interval, Interval]:
        """(R, T, A) intervals that jointly satisfy R + T + A <= 1."""

        reflectivity = self.reflectivity
        transmissivity = self.transmissivity
        absorptivity = self.absorptivity
        if absorptivity is None:
            absorptivity = self._derive(reflectivity, transmissivity)
        elif transmissivity is None:
            transmissivity = self._derive(reflectivity, absorptivity)
        elif reflectivity is None:
            reflectivity = self._derive(transmissivity, absorptivity)
        return reflectivity, transmissivity, absorptivity

    def to_dict(self) -> dict:
        return {
            name: getattr(self, name).to_dict() if getattr(self, name) is not None else None
            for name in ("reflectivity", "transmissivity", "absorptivity")
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChannelResponse":
        return cls(
            **{
                name: Interval.from_dict(value) if value is not None else None
                for name, value in data.items()
            }
        )


def channel_key(surface_id: str, wavelength_nm: float | None, axis: str | None) -> str:
    wavelength = "unspecified" if wavelength_nm is None else f"{float(wavelength_nm):g}"
    return f"{surface_id}|{wavelength}|{axis or 'unpolarized'}"


@dataclass
class MeasurementBatch:
    """One named measurement campaign over a set of element channels.

    entries maps 'surface|wavelength|axis' channel keys to ChannelResponse.
    Batches may be mutually inconsistent; the book resolves conflicts by
    active-batch priority and records which batch supplied every value.
    """

    name: str
    entries: Mapping[str, ChannelResponse] = field(default_factory=dict)
    condition: str = ""

    def covers(self, surface_id: str, wavelength_nm: float | None, axis: str) -> bool:
        return channel_key(surface_id, wavelength_nm, axis) in self.entries

    def covers_surface(self, surface_id: str) -> bool:
        prefix = f"{surface_id}|"
        return any(key.startswith(prefix) for key in self.entries)

    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "condition": self.condition,
            "entries": {key: response.to_dict() for key, response in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MeasurementBatch":
        return cls(
            name=data["name"],
            condition=data.get("condition", ""),
            entries={
                key: ChannelResponse.from_dict(value)
                for key, value in data.get("entries", {}).items()
            },
        )


class MeasurementBook:
    """All measurement batches of a project plus the active resolution order.

    active lists batch names in priority order: the first active batch
    covering a channel supplies its response.  Revoking a batch removes it
    from the active set; its data stays archived so saved projects and old
    trace batches still point at the original measurement basis.
    """

    def __init__(self, batches: list[MeasurementBatch] | None = None,
                 active: list[str] | None = None):
        self.batches: dict[str, MeasurementBatch] = {
            batch.name: batch for batch in (batches or [])
        }
        self.active: list[str] = [
            name for name in (active if active is not None else list(self.batches))
            if name in self.batches
        ]

    def add(self, batch: MeasurementBatch, activate: bool = True) -> None:
        self.batches[batch.name] = batch
        if activate and batch.name not in self.active:
            self.active.append(batch.name)

    def revoke(self, name: str) -> None:
        """Deactivate a batch; its archived data and provenance remain."""

        if name not in self.batches:
            raise KeyError(f"unknown measurement batch {name!r}")
        self.active = [active for active in self.active if active != name]

    def activate(self, name: str, first: bool = False) -> None:
        if name not in self.batches:
            raise KeyError(f"unknown measurement batch {name!r}")
        self.active = [active for active in self.active if active != name]
        if first:
            self.active.insert(0, name)
        else:
            self.active.append(name)

    def covers_surface(self, surface_id: str) -> bool:
        """True when any batch (active or revoked) ever measured this element."""

        return any(batch.covers_surface(surface_id) for batch in self.batches.values())

    def response_for(self, surface_id: str, wavelength_nm: float | None,
                     axis: str) -> tuple[str, ChannelResponse] | None:
        """(batch name, response) from the highest-priority active batch."""

        key = channel_key(surface_id, wavelength_nm, axis)
        for name in self.active:
            batch = self.batches[name]
            if key in batch.entries:
                return name, batch.entries[key]
        return None

    def basis(self) -> dict:
        """Provenance snapshot: which batches a new trace relies on."""

        return {
            "active": list(self.active),
            "digests": {name: self.batches[name].digest() for name in self.active},
            "revoked": [name for name in self.batches if name not in self.active],
        }

    def to_dict(self) -> dict:
        return {
            "batches": [self.batches[name].to_dict() for name in self.batches],
            "active": list(self.active),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "MeasurementBook":
        if not data:
            return cls()
        batches = [MeasurementBatch.from_dict(item) for item in data.get("batches", [])]
        return cls(batches, active=list(data.get("active", [])))

