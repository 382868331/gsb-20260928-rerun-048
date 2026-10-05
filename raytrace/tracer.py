"""Branching geometric ray tracing.

Refractive interfaces split an incoming ray into a transmitted and a
reflected child beam.  Both children carry the remaining power of their
parent and are traced independently; every leaf keeps its ancestry so that
detector hits and energy accounts can be grouped by source ray.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
from typing import Any

from .materials import MaterialLibrary, UnknownMaterialResponse, fresnel_coefficients
from .materials import fresnel_amplitudes
from .measurements import AffineQuantity, MeasurementBatch
from .surfaces import Surface, add, cross, dot, normalize, scale


def _reference_basis(direction: tuple[float, float, float]):
    """Stable transverse basis (u, v) for a propagation direction."""

    reference = (0.0, 0.0, 1.0)
    if abs(dot(direction, reference)) > 0.9:
        reference = (0.0, 1.0, 0.0)
    u = normalize(cross(direction, reference))
    v = normalize(cross(direction, u))
    return u, v


def _linear_state(direction: tuple[float, float, float], angle_deg: float) -> tuple[float, float, float]:
    """Unit linear-polarization vector at an angle in the transverse plane."""

    u, v = _reference_basis(direction)
    theta = math.radians(angle_deg)
    return normalize(add(scale(u, math.cos(theta)), scale(v, math.sin(theta))))


@dataclass(frozen=True)
class Ray:
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    wavelength_nm: float | None = None
    weight: float = 1.0
    label: str | None = None
    spectral_weights: tuple[tuple[float, float], ...] | None = None
    polarization: tuple[float, float, float] | None = None
    polarization_angle_deg: float | None = None
    polarization_weights: tuple[tuple[float, float], ...] | None = None

    def normalized(self) -> "Ray":
        return Ray(
            self.origin,
            normalize(self.direction),
            self.wavelength_nm,
            self.weight,
            self.label,
            self.spectral_weights,
            self.polarization,
            self.polarization_angle_deg,
            self.polarization_weights,
        )

    def spectral_components(self) -> tuple[tuple[float, float], ...]:
        """Return explicit wavelength components with absolute power."""

        if not self.spectral_weights:
            return ((self.wavelength_nm, self.weight),)
        total_fraction = sum(weight for _, weight in self.spectral_weights)
        if total_fraction <= 0:
            raise ValueError("spectral component weights must have a positive sum")
        wavelengths = [wavelength for wavelength, _ in self.spectral_weights]
        if len(set(wavelengths)) != len(wavelengths):
            raise ValueError("spectral component wavelengths must be unique")
        components = []
        for wavelength, fraction in self.spectral_weights:
            if wavelength <= 0:
                raise ValueError("wavelength_nm must be positive")
            if fraction < 0:
                raise ValueError("spectral component weights must be non-negative")
            if fraction > 0:
                components.append((float(wavelength), self.weight * fraction / total_fraction))
        if not components:
            raise ValueError("a spectral ray requires at least one positive component")
        return tuple(components)

    def polarization_components(self):
        """Return (angle_deg, field vector, absolute power) per component.

        Incoherent components split the input power by fraction, so the
        parts always sum to the input weight and never to multiple full
        copies of it.  A ray without polarization_weights is a single
        component: either its explicit field vector or the legacy
        unpolarized interpretation.
        """

        direction = normalize(self.direction)
        if not self.polarization_weights:
            return ((self.polarization_angle_deg, self.polarization, self.weight),)
        total_fraction = sum(fraction for _, fraction in self.polarization_weights)
        if total_fraction <= 0:
            raise ValueError("polarization component weights must have a positive sum")
        angles = [angle for angle, _ in self.polarization_weights]
        if len(set(angles)) != len(angles):
            raise ValueError("polarization component angles must be unique")
        components = []
        for angle, fraction in self.polarization_weights:
            if fraction < 0:
                raise ValueError("polarization component weights must be non-negative")
            if fraction > 0:
                components.append((
                    float(angle),
                    _linear_state(direction, float(angle)),
                    self.weight * fraction / total_fraction,
                ))
        if not components:
            raise ValueError("a polarized ray requires at least one positive component")
        return tuple(components)

    @classmethod
    def polarized(cls, origin, direction, angles, *,
                  fractions=None, weight=1.0,
                  label=None, wavelength_nm=None,
                  spectral_weights=None) -> "Ray":
        """Input split into mutually incoherent linear-polarization parts."""

        if not angles:
            raise ValueError("a polarized ray requires at least one angle")
        if fractions is None:
            fractions = [1.0 / len(angles)] * len(angles)
        if len(fractions) != len(angles):
            raise ValueError("angles and fractions must have the same length")
        return cls(origin, direction, wavelength_nm, weight, label, spectral_weights,
                   None, None, tuple(zip(angles, fractions)))

    @classmethod
    def spectral(cls, origin, direction, wavelengths: list[float], *,
                 fractions: list[float] | None = None, weight: float = 1.0,
                 label: str | None = None) -> "Ray":
        if not wavelengths:
            raise ValueError("a spectral ray requires at least one wavelength")
        if fractions is None:
            fractions = [1.0 / len(wavelengths)] * len(wavelengths)
        if len(fractions) != len(wavelengths):
            raise ValueError("wavelengths and fractions must have the same length")
        return cls(origin, direction, None, weight, label, tuple(zip(wavelengths, fractions)))


@dataclass(frozen=True)
class TraceSettings:
    max_events: int = 64
    ray_offset: float = 1e-9
    grazing_cosine: float = 1e-6
    max_branches: int = 100_000
    min_branch_weight: float = 1e-12


@dataclass
class RayResult:
    ray: Ray
    status: str
    endpoint: tuple[float, float, float]
    weight: float
    events: list[dict[str, Any]] = field(default_factory=list)
    source_label: str = "root"
    branch_id: str = "root"
    parent_branch_id: str | None = None
    detector: str | None = None
    depth: int = 0
    wavelength_nm: float | None = None
    polarization: float | None = None
    # Serialized AffineQuantity ({constant, coeffs}) when the terminal power
    # depends on measured response intervals; None means fully determined.
    weight_uncertainty: dict | None = None


class RayTracer:
    def __init__(self, surfaces: list[Surface], materials: MaterialLibrary, settings: TraceSettings | None = None,
                 measurements: MeasurementBatch | None = None,
                 revoked_batches: tuple[MeasurementBatch, ...] = ()):
        self.surfaces = surfaces
        self.materials = materials
        self.settings = settings or TraceSettings()
        self.measurements = measurements
        # Batches whose measurement basis was withdrawn.  A surface a
        # revoked batch declared must stay unknown when the active batch
        # does not cover it; the revoked defaults never fill in.
        self.revoked_batches = tuple(revoked_batches)
        self._noise_counter = 0

    def _noise(self) -> str:
        """Fresh symbol bounding one nonlinear product term."""

        self._noise_counter += 1
        return f"nonlinear:{self._noise_counter}"

    def trace(self, ray: Ray) -> RayResult:
        leaves = self.trace_branches([ray])
        return self._primary_leaf(ray, leaves)

    @staticmethod
    def _primary_leaf(ray: Ray, leaves: list["RayResult"]) -> RayResult:
        """Compatibility view: the all-transmission leaf of one source ray."""

        transmitted_only = re.compile(r"^root(?:\.\d+\.t)*$")
        candidates = [leaf for leaf in leaves if transmitted_only.match(leaf.branch_id)]
        if not candidates:
            candidates = leaves
        primary = max(candidates, key=lambda leaf: (leaf.depth, -len(leaf.branch_id)))
        return RayResult(
            ray=ray,
            status=primary.status,
            endpoint=primary.endpoint,
            weight=primary.weight,
            events=[dict(event) for event in primary.events],
            source_label=primary.source_label,
            branch_id=primary.branch_id,
            parent_branch_id=primary.parent_branch_id,
            detector=primary.detector,
            depth=primary.depth,
            wavelength_nm=primary.wavelength_nm,
            polarization=primary.polarization,
        )

    def trace_branches(self, rays: list[Ray]) -> list[RayResult]:
        """Trace every source ray and return all terminal sub-beams."""

        self._noise_counter = 0
        leaves: list[RayResult] = []
        for source_index, source_ray in enumerate(rays):
            normalized = source_ray.normalized()
            source_label = (
                normalized.label
                if normalized.label is not None
                else f"ray-{source_index}"
            )
            spectral_components = (
                normalized.spectral_components()
                if normalized.spectral_weights
                else ((normalized.wavelength_nm, normalized.weight),)
            )
            for wavelength, spectral_weight in spectral_components:
                scale_factor = spectral_weight / normalized.weight if normalized.weight else 0.0
                polarized_components = normalized.polarization_components()
                for angle, field_vector, polarization_weight in polarized_components:
                    seed = Ray(
                        normalized.origin,
                        normalized.direction,
                        wavelength,
                        polarization_weight * scale_factor,
                        source_label,
                        None,
                        field_vector,
                        angle,
                    )
                    branch_root = "root"
                    if normalized.spectral_weights:
                        branch_root += f"[wavelength={wavelength:g}]"
                    if normalized.polarization_weights:
                        branch_root += f"[pol={angle:g}]"
                    leaves.extend(self._trace_one(seed, branch_root))
        return leaves

    def _trace_one(self, seed: Ray, branch_root: str = "root") -> list[RayResult]:
        settings = self.settings
        pending: list[dict[str, Any]] = [
            {
                "ray": seed,
                "branch_id": branch_root,
                "parent_branch_id": None,
                "depth": 0,
                "events": [],
                "last_surface": None,
                "saw_tir": False,
                "weight_affine": AffineQuantity(seed.weight),
            }
        ]
        leaves: list[RayResult] = []
        branch_count = 0
        event_counter = 0

        while pending:
            branch_count += 1
            if branch_count > settings.max_branches:
                for overflow in pending:
                    leaves.append(self._leaf(seed, overflow, "max_branches", overflow["ray"].origin))
                break

            state = pending.pop()
            current: Ray = state["ray"]
            weight = current.weight
            events = state["events"]

            if state["depth"] >= settings.max_events:
                leaves.append(self._leaf(seed, state, "max_events", current.origin))
                continue

            hit = self._nearest_hit(current, state["last_surface"])
            if hit is None:
                leaves.append(self._leaf(seed, state, "no_image", current.origin))
                continue

            t, point, hit_normal, surface = hit
            # Orient the normal toward the incident half-space so reflection
            # offsets and the Snell-geometry sign agree for both travel
            # directions through any surface orientation.
            normal = scale(hit_normal, -1.0 if dot(current.direction, hit_normal) > 0 else 1.0)
            if weight < settings.min_branch_weight:
                leaves.append(self._leaf(seed, state, "stopped", point))
                continue
            incident = tuple(current.direction)
            edge_hit = surface.at_aperture_edge(point)
            cosine = abs(dot(current.direction, normal))
            event: dict[str, Any] = {
                "event_uid": event_counter,
                "surface": surface.identifier,
                "kind": surface.kind,
                "distance": t,
                "point": point,
                "normal": normal,
                "incident_direction": incident,
                "incident_weight": weight,
                "aperture_edge": edge_hit,
                "wavelength_nm": current.wavelength_nm,
            }

            # Bulk (volume) loss along the propagated segment, attributed to
            # the medium on the incident side of the hit surface.  Missing
            # verified attenuation data ends the branch as unknown instead
            # of silently assuming a lossless medium.
            relative = add(current.origin, scale(surface.vertex, -1.0))
            medium_name = (
                surface.back_material
                if dot(relative, surface.axis) > 0
                else surface.front_material
            )
            try:
                attenuation = self.materials.bulk_attenuation(medium_name, current.wavelength_nm)
            except UnknownMaterialResponse as exc:
                event.update({
                    "action": "unknown_material_response",
                    "medium": medium_name,
                    "reason": str(exc),
                    "final_direction": incident,
                    "absorbed_power": 0.0,
                    "bulk_loss_power": 0.0,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                continue
            propagation_loss = (
                weight * (1.0 - math.exp(-attenuation * t)) if attenuation > 0.0 else 0.0
            )
            if propagation_loss > 0.0:
                weight -= propagation_loss
                current = replace(current, weight=weight)
                state = {
                    **state,
                    "ray": current,
                    "weight_affine": state["weight_affine"].scale(math.exp(-attenuation * t)),
                }
                event["incident_weight"] = weight
            event["medium"] = medium_name
            event["propagation_distance"] = t
            event["bulk_loss_power"] = propagation_loss

            if edge_hit or cosine < settings.grazing_cosine:
                event.update({
                    "action": "grazing",
                    "cosine_incidence": cosine,
                    "final_direction": incident,
                    "absorbed_power": 0.0,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                leaves.append(self._finish(seed, state, event, "grazing", point, surface.identifier))
                continue

            if surface.kind == "detector":
                event.update({
                    "action": "land",
                    "final_direction": incident,
                    "absorbed_power": 0.0,
                })
                if surface.analyzer_angle_deg is not None:
                    fraction = self._analyzer_fraction(surface, current)
                    transmitted = weight * fraction
                    event.update({
                        "analyzer_angle_deg": surface.analyzer_angle_deg,
                        "analyzer_transmitted_power": transmitted,
                        "analyzer_rejected_power": weight - transmitted,
                    })
                    current = replace(current, weight=transmitted)
                    state = {
                        **state,
                        "ray": current,
                        "weight_affine": state["weight_affine"].scale(fraction),
                    }
                event["event_uid"] = event_counter
                event_counter += 1
                leaves.append(self._finish(seed, state, event, "reached_image", point, surface.identifier))
                continue

            if surface.kind == "aperture_stop":
                event.update({
                    "action": "pass_aperture",
                    "absorbed_power": 0.0,
                    "transmitted_power": weight,
                    "reflected_power": 0.0,
                    "outgoing_direction": incident,
                    "outgoing_weight": weight,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                outgoing = replace(
                    current,
                    origin=self._offset(point, incident),
                    direction=incident,
                    weight=weight,
                )
                pending.append(self._advance(state, event, outgoing, surface.identifier, state["saw_tir"], event_counter - 1))
                continue

            if surface.kind == "beamsplitter":
                reflected = self._reflect(current.direction, scale(normal, -1.0))
                transmitted = tuple(incident)
                measured_split = self._measured_split(surface, current)
                if measured_split is None and not surface.has_engineering_response():
                    revoked = self._revoked_declarations(surface.identifier)
                    if revoked:
                        reason = (
                            f"beamsplitter {surface.identifier!r} lost its measurement basis "
                            f"(revoked batch(es): {', '.join(batch_id for batch_id, _ in revoked)}) "
                            "and has no active measurement or independent engineering response; "
                            "the response stays unknown instead of falling back to default values"
                        )
                    else:
                        reason = (
                            f"beamsplitter {surface.identifier!r} has no active measurement "
                            "batch or independent engineering response"
                        )
                    event.update({
                        "action": "unknown_engineering_response",
                        "unknown_channels": ["reflectivity", "transmissivity"],
                        "reason": reason,
                        "final_direction": incident,
                        "absorbed_power": 0.0,
                    })
                    if revoked:
                        event["revoked_batches"] = [batch_id for batch_id, _ in revoked]
                    event["event_uid"] = event_counter
                    event_counter += 1
                    leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                    continue
                if measured_split is not None:
                    # Measured interval responses: fractions are affine
                    # quantities over shared uncertainty symbols, so beams
                    # recombined later keep their correlation.  A channel
                    # the batch does not cover ends as unknown instead of
                    # defaulting to lossless transmission.
                    (
                        reflected_fraction_aff,
                        transmitted_fraction_aff,
                        reflected_field,
                        transmitted_field,
                        s_fraction,
                        unknown_channels,
                    ) = self._polarized_split_affine(surface, incident, normal, current, measured_split)
                    weight_affine = state["weight_affine"]
                    reflected_affine = (
                        weight_affine.mul(reflected_fraction_aff, self._noise())
                        if reflected_fraction_aff is not None else None
                    )
                    transmitted_affine = (
                        weight_affine.mul(transmitted_fraction_aff, self._noise())
                        if transmitted_fraction_aff is not None else None
                    )
                    known_affine = AffineQuantity(0.0)
                    if reflected_affine is not None:
                        known_affine = known_affine + reflected_affine
                    if transmitted_affine is not None:
                        known_affine = known_affine + transmitted_affine
                    remainder_affine = weight_affine - known_affine
                    if unknown_channels:
                        absorbed_affine = None
                        absorbed_power = 0.0
                    else:
                        absorbed_affine = remainder_affine
                        absorbed_power = remainder_affine.constant
                    reflected_weight = reflected_affine.constant if reflected_affine is not None else 0.0
                    transmitted_weight = transmitted_affine.constant if transmitted_affine is not None else 0.0
                    symbols: set[str] = set()
                    for quantity in (reflected_affine, transmitted_affine, remainder_affine):
                        if quantity is not None:
                            symbols.update(quantity.symbols)
                    event.update({
                        "action": "split",
                        "cosine_incidence": cosine,
                        "reflectivity": reflected_fraction_aff.constant if reflected_fraction_aff else None,
                        "transmissivity": transmitted_fraction_aff.constant if transmitted_fraction_aff else None,
                        "polarization_s_fraction": s_fraction,
                        "measurement_batch": self.measurements.batch_id,
                        "reflectivity_interval": (
                            list(reflected_fraction_aff.interval()) if reflected_fraction_aff else None
                        ),
                        "transmissivity_interval": (
                            list(transmitted_fraction_aff.interval()) if transmitted_fraction_aff else None
                        ),
                        "unknown_channels": unknown_channels,
                        "reflected_power": reflected_weight,
                        "transmitted_power": transmitted_weight,
                        "absorbed_power": absorbed_power,
                        "reflected_power_interval": list(reflected_affine.interval()) if reflected_affine else None,
                        "transmitted_power_interval": list(transmitted_affine.interval()) if transmitted_affine else None,
                        "absorbed_power_interval": (
                            list(absorbed_affine.interval())
                            if absorbed_affine is not None else None
                        ),
                        "uncertainty_symbols": sorted(symbols),
                        "transmitted_direction": transmitted,
                        "reflected_direction": reflected,
                        "outgoing_direction": transmitted,
                        "outgoing_weight": transmitted_weight,
                    })
                    event["event_uid"] = event_counter
                    event_counter += 1
                    base_events = self._extended_events(events, event, state["branch_id"], event_counter - 1)
                    next_depth = state["depth"] + 1
                    common = {
                        "last_surface": surface.identifier,
                        "depth": next_depth,
                        "saw_tir": state["saw_tir"],
                    }
                    children = []
                    if reflected_affine is not None and reflected_weight > 0.0:
                        children.append({
                            **common,
                            "events": list(base_events),
                            "ray": replace(
                                current,
                                origin=self._offset(point, reflected),
                                direction=reflected,
                                weight=reflected_weight,
                                polarization=reflected_field,
                            ),
                            "weight_affine": reflected_affine,
                            "branch_id": f"{state['branch_id']}.{len(base_events)}.r",
                            "parent_branch_id": state["branch_id"],
                        })
                    if transmitted_affine is not None and transmitted_weight > 0.0:
                        children.append({
                            **common,
                            "events": list(base_events),
                            "ray": replace(
                                current,
                                origin=self._offset(point, transmitted),
                                direction=transmitted,
                                weight=transmitted_weight,
                                polarization=transmitted_field,
                            ),
                            "weight_affine": transmitted_affine,
                            "branch_id": f"{state['branch_id']}.{len(base_events)}.t",
                            "parent_branch_id": state["branch_id"],
                        })
                    pending.extend(children)
                    if unknown_channels:
                        unknown_state = {
                            **common,
                            "events": list(base_events),
                            "ray": replace(current, weight=max(0.0, remainder_affine.constant)),
                            "weight_affine": remainder_affine,
                            "branch_id": f"{state['branch_id']}.{len(base_events)}.u",
                            "parent_branch_id": state["branch_id"],
                        }
                        leaves.append(self._leaf(seed, unknown_state, "unknown_response", point, None))
                    elif not children:
                        leaves.append(self._finish(seed, state, event, "absorbed", point, None))
                    continue
                # reflectivity always means the reflected power fraction,
                # matching mirrors; transmission defaults to the lossless
                # complement and may be set explicitly.  Polarization
                # tables, when declared, override the scalars per axis.
                decomposition = self._polarized_split(surface, incident, normal, current)
                if decomposition is None:
                    event.update({
                        "action": "unknown_polarization_response",
                        "reason": (
                            f"beamsplitter {surface.identifier!r} declares a polarization"
                            " table without a verified entry for every s/p axis"
                        ),
                        "final_direction": incident,
                        "absorbed_power": 0.0,
                    })
                    event["event_uid"] = event_counter
                    event_counter += 1
                    leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                    continue
                (
                    reflected_fraction,
                    transmitted_fraction,
                    reflected_field,
                    transmitted_field,
                    s_fraction,
                ) = decomposition
                reflected_weight = weight * reflected_fraction
                transmitted_weight = weight * transmitted_fraction
                absorbed_power = max(0.0, weight - reflected_weight - transmitted_weight)
                event.update({
                    "action": "split",
                    "cosine_incidence": cosine,
                    "reflectivity": reflected_fraction,
                    "transmissivity": transmitted_fraction,
                    "polarization_s_fraction": s_fraction,
                    "reflected_power": reflected_weight,
                    "transmitted_power": transmitted_weight,
                    "absorbed_power": absorbed_power,
                    "transmitted_direction": transmitted,
                    "reflected_direction": reflected,
                    "outgoing_direction": transmitted,
                    "outgoing_weight": transmitted_weight,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                base_events = self._extended_events(events, event, state["branch_id"], event_counter - 1)
                next_depth = state["depth"] + 1
                common = {
                    "last_surface": surface.identifier,
                    "depth": next_depth,
                    "saw_tir": state["saw_tir"],
                }
                children: list[dict[str, Any]] = []
                if reflected_weight > 0.0:
                    children.append({
                        **common,
                        "events": list(base_events),
                        "ray": replace(
                            current,
                            origin=self._offset(point, reflected),
                            direction=reflected,
                            weight=reflected_weight,
                            polarization=reflected_field,
                        ),
                        "weight_affine": state["weight_affine"].scale(reflected_fraction),
                        "branch_id": f"{state['branch_id']}.{len(base_events)}.r",
                        "parent_branch_id": state["branch_id"],
                    })
                if transmitted_weight > 0.0:
                    children.append({
                        **common,
                        "events": list(base_events),
                        "ray": replace(
                            current,
                            origin=self._offset(point, transmitted),
                            direction=transmitted,
                            weight=transmitted_weight,
                            polarization=transmitted_field,
                        ),
                        "weight_affine": state["weight_affine"].scale(transmitted_fraction),
                        "branch_id": f"{state['branch_id']}.{len(base_events)}.t",
                        "parent_branch_id": state["branch_id"],
                    })
                if children:
                    pending.extend(children)
                else:
                    # Fully absorbing splitter: keep the event on a terminal
                    # leaf so the lost power stays in the energy ledger.
                    leaves.append(self._finish(seed, state, event, "absorbed", point, None))
                continue

            if surface.kind == "mirror":
                reflected = self._reflect(current.direction, scale(normal, -1.0))
                measured_reflectivity = None
                if self.measurements is not None and surface.identifier in self.measurements.surfaces_declared():
                    measured_reflectivity = self.measurements.interval(
                        surface.identifier, "reflectivity", current.wavelength_nm, None
                    )
                    if measured_reflectivity is None:
                        event.update({
                            "action": "unknown_measured_response",
                            "measurement_batch": self.measurements.batch_id,
                            "reason": (
                                f"measurement batch {self.measurements.batch_id!r} declares"
                                f" {surface.identifier!r} without a reflectivity interval"
                            ),
                            "final_direction": incident,
                            "absorbed_power": 0.0,
                        })
                        event["event_uid"] = event_counter
                        event_counter += 1
                        leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                        continue
                if measured_reflectivity is None:
                    # A mirror whose only response basis was revoked must not
                    # silently reuse the default reflectivity; name the
                    # element, the undetermined channel and the revoked
                    # batches, then end the branch as unknown.
                    revoked = self._revoked_declarations(surface.identifier)
                    if revoked:
                        event.update({
                            "action": "revoked_measurement_basis",
                            "revoked_batches": [batch_id for batch_id, _ in revoked],
                            "unknown_channels": ["reflectivity"],
                            "reason": (
                                f"mirror {surface.identifier!r} lost its measurement basis "
                                f"(revoked batch(es): {', '.join(batch_id for batch_id, _ in revoked)}) "
                                "and no active measurement covers its reflectivity; "
                                "the response stays unknown instead of falling back to "
                                "the default reflectivity"
                            ),
                            "final_direction": incident,
                            "absorbed_power": 0.0,
                        })
                        event["event_uid"] = event_counter
                        event_counter += 1
                        leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                        continue
                if measured_reflectivity is not None:
                    new_affine = state["weight_affine"].mul(measured_reflectivity, self._noise())
                    absorbed_affine = state["weight_affine"] - new_affine
                    absorbed_interval = absorbed_affine.interval()
                    new_weight = new_affine.constant
                    absorbed = max(0.0, absorbed_affine.constant)
                    event.update({
                        "measurement_batch": self.measurements.batch_id,
                        "reflectivity_interval": list(measured_reflectivity.interval()),
                        "reflected_power_interval": list(new_affine.interval()),
                        "absorbed_power_interval": [max(0.0, absorbed_interval[0]), absorbed_interval[1]],
                        "uncertainty_symbols": sorted(new_affine.symbols),
                    })
                else:
                    new_affine = state["weight_affine"].scale(surface.reflectivity)
                    absorbed = weight * (1.0 - surface.reflectivity)
                    new_weight = weight * surface.reflectivity
                event.update({
                    "action": "reflect",
                    "cosine_incidence": cosine,
                    "reflected_power": new_weight,
                    "absorbed_power": absorbed,
                    "transmitted_power": 0.0,
                    "outgoing_direction": reflected,
                    "outgoing_weight": new_weight,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                if new_weight > 0.0:
                    outgoing = replace(
                        current,
                        origin=self._offset(point, reflected),
                        direction=reflected,
                        weight=new_weight,
                        polarization=self._reflect_field(current, normal),
                    )
                    pending.append(self._advance(
                        state, event, outgoing, surface.identifier, state["saw_tir"],
                        event_counter - 1, weight_affine=new_affine,
                    ))
                else:
                    leaves.append(self._finish(seed, state, event, "absorbed", point, None))
                continue

            # The incident medium is the half-space containing the ray
            # origin, not the sign of the axial direction component; this
            # stays correct for tilted surfaces and returning beams.
            relative = add(current.origin, scale(surface.vertex, -1.0))
            origin_on_back_side = dot(relative, surface.axis) > 0
            if origin_on_back_side:
                n1_name, n2_name = surface.back_material, surface.front_material
            else:
                n1_name, n2_name = surface.front_material, surface.back_material
            try:
                n1 = self.materials.index(n1_name, current.wavelength_nm)
                n2 = self.materials.index(n2_name, current.wavelength_nm)
            except UnknownMaterialResponse as exc:
                event.update({
                    "action": "unknown_material_response",
                    "incident_material": n1_name,
                    "transmitted_material": n2_name,
                    "reason": str(exc),
                    "final_direction": incident,
                    "absorbed_power": 0.0,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                continue
            # ``normal`` already points into the incident half-space.
            against_normal = normal
            cos_i = -dot(current.direction, against_normal)
            eta = n1 / n2
            k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)

            if k < 0:
                reflected = self._reflect(current.direction, scale(normal, -1.0))
                event.update({
                    "action": "total_internal_reflection",
                    "n_incident": n1,
                    "n_transmitted": n2,
                    "cosine_incidence": cos_i,
                    "critical_angle_rad": math.asin(n2 / n1) if n1 > n2 else None,
                    "critical_cosine": n2 / n1 if n1 > n2 else None,
                    "reflected_power": weight,
                    "transmitted_power": 0.0,
                    "absorbed_power": 0.0,
                    "outgoing_direction": reflected,
                    "outgoing_weight": weight,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                outgoing = replace(
                    current,
                    origin=self._offset(point, reflected),
                    direction=reflected,
                    weight=weight,
                    polarization=self._reflect_field(current, normal),
                )
                pending.append(self._advance(state, event, outgoing, surface.identifier, True, event_counter - 1))
                continue

            refracted = normalize(tuple(
                eta * current.direction[i] + (eta * cos_i - math.sqrt(k)) * against_normal[i]
                for i in range(3)
            ))
            reflected = self._reflect(current.direction, scale(normal, -1.0))
            reflected_field = None
            transmitted_field = None
            s_fraction = None
            if current.polarization is not None:
                (
                    reflected_fraction,
                    transmitted_fraction,
                    reflected_field,
                    transmitted_field,
                    s_fraction,
                ) = self._polarized_fresnel(n1, n2, cos_i, incident, normal, refracted, reflected, current)
            else:
                reflected_fraction, transmitted_fraction = fresnel_coefficients(n1, n2, cos_i)
            reflected_weight = weight * reflected_fraction
            transmitted_weight = weight * transmitted_fraction
            absorbed_power = max(0.0, weight - reflected_weight - transmitted_weight)
            event.update({
                "action": "refract",
                "n_incident": n1,
                "n_transmitted": n2,
                "cosine_incidence": cos_i,
                "polarization_s_fraction": s_fraction,
                "reflected_power": reflected_weight,
                "transmitted_power": transmitted_weight,
                "absorbed_power": absorbed_power,
                "transmitted_direction": refracted,
                "reflected_direction": reflected,
                "outgoing_direction": refracted,
                "outgoing_weight": transmitted_weight,
            })

            event["event_uid"] = event_counter
            event_counter += 1
            base_events = self._extended_events(events, event, state["branch_id"], event_counter - 1)
            next_depth = state["depth"] + 1
            common = {
                "last_surface": surface.identifier,
                "depth": next_depth,
                "events": base_events,
                "saw_tir": state["saw_tir"],
            }
            if reflected_weight > 0.0:
                reflected_ray = replace(
                    current,
                    origin=self._offset(point, reflected),
                    direction=reflected,
                    weight=reflected_weight,
                    polarization=reflected_field,
                )
                pending.append({
                    **common,
                    "events": list(base_events),
                    "ray": reflected_ray,
                    "weight_affine": state["weight_affine"].scale(reflected_fraction),
                    "branch_id": f"{state['branch_id']}.{len(base_events)}.r",
                    "parent_branch_id": state["branch_id"],
                })
            transmitted_ray = replace(
                current,
                origin=self._offset(point, refracted),
                direction=refracted,
                weight=transmitted_weight,
                polarization=transmitted_field,
            )
            pending.append({
                **common,
                "events": list(base_events),
                "ray": transmitted_ray,
                "weight_affine": state["weight_affine"].scale(transmitted_fraction),
                "branch_id": f"{state['branch_id']}.{len(base_events)}.t",
                "parent_branch_id": state["branch_id"],
            })

        return leaves

    @staticmethod
    def _extended_events(events: list[dict[str, Any]], event: dict[str, Any], branch_id: str, event_uid: int) -> list[dict[str, Any]]:
        recorded = dict(event)
        recorded["event_uid"] = event_uid
        recorded["branch_id"] = branch_id
        return [*events, recorded]

    @staticmethod
    def _advance(state: dict[str, Any], event: dict[str, Any], ray: Ray, surface_identifier: str, saw_tir: bool, event_uid: int,
                 weight_affine: "AffineQuantity | None" = None) -> dict[str, Any]:
        return {
            "ray": ray,
            "branch_id": state["branch_id"],
            "parent_branch_id": state["parent_branch_id"],
            "depth": state["depth"] + 1,
            "events": RayTracer._extended_events(state["events"], event, state["branch_id"], event_uid),
            "last_surface": surface_identifier,
            "saw_tir": saw_tir,
            "weight_affine": weight_affine if weight_affine is not None else state.get("weight_affine"),
        }

    @staticmethod
    def _leaf(seed: Ray, state: dict[str, Any], status: str, endpoint, detector: str | None = None) -> RayResult:
        affine = state.get("weight_affine")
        return RayResult(
            ray=seed,
            status=status,
            endpoint=tuple(endpoint),
            weight=state["ray"].weight,
            events=state["events"],
            source_label=seed.label or "root",
            branch_id=state["branch_id"],
            parent_branch_id=state["parent_branch_id"],
            detector=detector,
            depth=state["depth"],
            wavelength_nm=seed.wavelength_nm,
            polarization=seed.polarization_angle_deg,
            weight_uncertainty=(
                affine.to_dict() if (affine is not None and not affine.is_certain) else None
            ),
        )

    def _finish(self, seed: Ray, state: dict[str, Any], event: dict[str, Any], status: str, point, detector: str | None) -> RayResult:
        advanced = self._advance(
            state,
            event,
            state["ray"],
            state["last_surface"],
            state["saw_tir"],
            event["event_uid"],
        )
        return self._leaf(seed, advanced, status, point, detector)

    def _nearest_hit(self, ray: Ray, last_surface: str | None):
        candidates = []
        for surface in self.surfaces:
            intersection = surface.intersect(ray.origin, ray.direction)
            if intersection is None:
                continue
            t, point, normal = intersection
            if surface.identifier == last_surface and t <= self.settings.ray_offset * 100:
                continue
            if t <= 0:
                continue
            if not surface.accepts(point):
                continue
            candidates.append((t, point, normal, surface))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])

    @staticmethod
    def _reflect(direction, normal) -> tuple[float, float, float]:
        factor = 2.0 * dot(direction, normal)
        return normalize(tuple(direction[i] - factor * normal[i] for i in range(3)))

    @staticmethod
    def _sp_basis(direction, normal):
        """Local s/p unit vectors for an incident direction and normal."""

        s_axis = cross(direction, normal)
        length = math.sqrt(dot(s_axis, s_axis))
        if length < 1e-12:
            s_axis, _ = _reference_basis(direction)
        else:
            s_axis = scale(s_axis, 1.0 / length)
        p_axis = normalize(cross(s_axis, direction))
        return s_axis, p_axis

    def _polarized_split(self, surface, incident, normal, ray: Ray):
        """Beamsplitter fractions and child fields for the ray polarization.

        Returns None when the surface declares polarization tables but has
        no verified response for an axis the ray actually carries; the
        caller must end the branch as unknown instead of substituting the
        scalar (unpolarized) numbers.
        """

        split_s = surface.polarized_split("s")
        split_p = surface.polarized_split("p")
        if split_s is None or split_p is None:
            return None
        (r_s, t_s), (r_p, t_p) = split_s, split_p
        if ray.polarization is None:
            s_fraction = 0.5
            reflected_fraction = 0.5 * (r_s + r_p)
            transmitted_fraction = 0.5 * (t_s + t_p)
            return reflected_fraction, transmitted_fraction, None, None, s_fraction
        s_axis, p_axis = self._sp_basis(incident, normal)
        e_s = dot(ray.polarization, s_axis)
        e_p = dot(ray.polarization, p_axis)
        s_fraction = e_s * e_s
        p_fraction = e_p * e_p
        reflected_fraction = s_fraction * r_s + p_fraction * r_p
        transmitted_fraction = s_fraction * t_s + p_fraction * t_p
        reflected = self._reflect(incident, scale(normal, -1.0))
        reflected_p_axis = normalize(cross(s_axis, reflected))
        reflected_field = self._combine_field(
            e_s * math.sqrt(r_s), s_axis,
            e_p * math.sqrt(r_p), reflected_p_axis,
            ray.polarization,
        )
        transmitted_field = self._combine_field(
            e_s * math.sqrt(t_s), s_axis,
            e_p * math.sqrt(t_p), p_axis,
            ray.polarization,
        )
        return reflected_fraction, transmitted_fraction, reflected_field, transmitted_field, s_fraction

    def _revoked_declarations(self, surface_identifier: str) -> list[tuple[str, list[str]]]:
        """(batch_id, quantities) of revoked batches declaring this surface."""

        declarations: list[tuple[str, list[str]]] = []
        for batch in self.revoked_batches:
            quantities = sorted({
                measurement.quantity
                for measurement in batch.measurements
                if measurement.surface == surface_identifier
            })
            if quantities:
                declarations.append((batch.batch_id, quantities))
        return declarations

    def _measured_split(self, surface: Surface, ray: Ray):
        """Per-axis measured interval split, or None when nothing is declared.

        When the active batch measures two of reflectivity/transmissivity/
        absorptivity for a channel, the third follows from energy
        conservation over the *same* symbols, keeping the correlation.
        """

        batch = self.measurements
        if batch is None or surface.identifier not in batch.surfaces_declared():
            return None
        one = AffineQuantity(1.0)
        split = {}
        for axis in ("s", "p"):
            reflectivity = batch.interval(surface.identifier, "reflectivity", ray.wavelength_nm, axis)
            transmissivity = batch.interval(surface.identifier, "transmissivity", ray.wavelength_nm, axis)
            absorptivity = batch.interval(surface.identifier, "absorptivity", ray.wavelength_nm, axis)
            if transmissivity is None and reflectivity is not None and absorptivity is not None:
                transmissivity = one - reflectivity - absorptivity
            if reflectivity is None and transmissivity is not None and absorptivity is not None:
                reflectivity = one - transmissivity - absorptivity
            split[axis] = (reflectivity, transmissivity)
        return split

    def _polarized_split_affine(self, surface, incident, normal, ray: Ray, measured_split):
        """Affine beamsplitter fractions from measured intervals.

        A fraction is None when any polarization axis the ray actually
        carries has no measured response for it; the caller terminates
        that share as unknown rather than assuming lossless transmission.
        """

        (r_s, t_s), (r_p, t_p) = measured_split["s"], measured_split["p"]

        def combine(first, second, w_s, w_p):
            if first is None or second is None:
                return None
            return first.scale(w_s) + second.scale(w_p)

        if ray.polarization is None:
            s_fraction = 0.5
            reflected_fraction = combine(r_s, r_p, 0.5, 0.5)
            transmitted_fraction = combine(t_s, t_p, 0.5, 0.5)
            unknown_channels = []
            if reflected_fraction is None:
                unknown_channels.append("reflectivity")
            if transmitted_fraction is None:
                unknown_channels.append("transmissivity")
            return reflected_fraction, transmitted_fraction, None, None, s_fraction, unknown_channels

        s_axis, p_axis = self._sp_basis(incident, normal)
        e_s = dot(ray.polarization, s_axis)
        e_p = dot(ray.polarization, p_axis)
        s_fraction = e_s * e_s
        p_fraction = e_p * e_p
        reflected_fraction = combine(r_s, r_p, s_fraction, p_fraction)
        transmitted_fraction = combine(t_s, t_p, s_fraction, p_fraction)
        unknown_channels = []
        if reflected_fraction is None:
            unknown_channels.append("reflectivity")
        if transmitted_fraction is None:
            unknown_channels.append("transmissivity")
        reflected = self._reflect(incident, scale(normal, -1.0))
        reflected_p_axis = normalize(cross(s_axis, reflected))
        reflected_field = None
        if reflected_fraction is not None:
            reflected_field = self._combine_field(
                e_s * math.sqrt(r_s.constant), s_axis,
                e_p * math.sqrt(r_p.constant), reflected_p_axis,
                ray.polarization,
            )
        transmitted_field = None
        if transmitted_fraction is not None:
            transmitted_field = self._combine_field(
                e_s * math.sqrt(t_s.constant), s_axis,
                e_p * math.sqrt(t_p.constant), p_axis,
                ray.polarization,
            )
        return (
            reflected_fraction,
            transmitted_fraction,
            reflected_field,
            transmitted_field,
            s_fraction,
            unknown_channels,
        )

    def _polarized_fresnel(self, n1, n2, cos_i, incident, normal, refracted, reflected, ray: Ray):
        """Fresnel split for a linearly polarized ray (lossless interface)."""

        r_s, r_p, t_s, t_p = fresnel_amplitudes(n1, n2, cos_i)
        s_axis, p_axis = self._sp_basis(incident, normal)
        e_s = dot(ray.polarization, s_axis)
        e_p = dot(ray.polarization, p_axis)
        s_fraction = e_s * e_s
        p_fraction = e_p * e_p
        reflected_fraction = s_fraction * r_s * r_s + p_fraction * r_p * r_p
        transmitted_fraction = 1.0 - reflected_fraction
        reflected_p_axis = normalize(cross(s_axis, reflected))
        transmitted_p_axis = normalize(cross(s_axis, refracted))
        reflected_field = self._combine_field(
            e_s * r_s, s_axis, e_p * r_p, reflected_p_axis, ray.polarization,
        )
        transmitted_field = self._combine_field(
            e_s * t_s, s_axis, e_p * t_p, transmitted_p_axis, ray.polarization,
        )
        return reflected_fraction, transmitted_fraction, reflected_field, transmitted_field, s_fraction

    @staticmethod
    def _combine_field(s_amplitude, s_axis, p_amplitude, p_axis, fallback):
        vector = add(scale(s_axis, s_amplitude), scale(p_axis, p_amplitude))
        if math.sqrt(dot(vector, vector)) < 1e-12:
            return fallback
        return normalize(vector)

    def _reflect_field(self, ray: Ray, normal):
        if ray.polarization is None:
            return None
        return self._reflect(ray.polarization, normal)

    @staticmethod
    def _analyzer_fraction(surface: Surface, ray: Ray) -> float:
        """Power fraction passing the detector analyzer."""

        if ray.polarization is None:
            return 0.5
        u_axis, v_axis = _reference_basis(surface.axis)
        theta = math.radians(surface.analyzer_angle_deg)
        analyzer = add(scale(u_axis, math.cos(theta)), scale(v_axis, math.sin(theta)))
        projection = dot(ray.polarization, analyzer)
        return projection * projection

    def _offset(self, point, direction) -> tuple[float, float, float]:
        return add(point, scale(direction, self.settings.ray_offset))
