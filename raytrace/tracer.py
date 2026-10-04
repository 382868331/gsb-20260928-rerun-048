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
from .measurements import AffineForm, MeasurementBook, channel_key
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
    # Uncertain measured responses propagate as an affine form over shared
    # measurement symbols.  weight stays the center value; weight_range is
    # the enclosing interval and weight_terms the serializable symbol
    # coefficients (None when the leaf is fully definite).
    weight_range: tuple[float, float] | None = None
    weight_terms: dict[str, float] | None = None

    def weight_affine(self) -> AffineForm:
        return AffineForm(self.weight, self.weight_terms)


class RayTracer:
    def __init__(self, surfaces: list[Surface], materials: MaterialLibrary, settings: TraceSettings | None = None,
                 measurements: MeasurementBook | None = None):
        self.surfaces = surfaces
        self.materials = materials
        self.settings = settings or TraceSettings()
        self.measurements = measurements if measurements is not None else MeasurementBook()

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
            weight_range=primary.weight_range,
            weight_terms=dict(primary.weight_terms) if primary.weight_terms else None,
        )

    def trace_branches(self, rays: list[Ray]) -> list[RayResult]:
        """Trace every source ray and return all terminal sub-beams."""

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
                "weight_aff": None,
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
                state = {**state, "ray": current}
                if state.get("weight_aff") is not None:
                    state["weight_aff"] = state["weight_aff"].scale(math.exp(-attenuation * t))
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
                    state = {**state, "ray": current}
                    if state.get("weight_aff") is not None:
                        state["weight_aff"] = state["weight_aff"].scale(fraction)
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
                # reflectivity always means the reflected power fraction,
                # matching mirrors; transmission defaults to the lossless
                # complement and may be set explicitly.  Polarization
                # tables, when declared, override the scalars per axis.
                reflected_aff = None
                transmitted_aff = None
                absorbed_aff = None
                if self.measurements is not None and self.measurements.covers_surface(surface.identifier):
                    measured_split = self._measured_split(
                        surface, current, incident, normal, state, event_counter
                    )
                    if measured_split is None:
                        event.update({
                            "action": "unknown_measured_response",
                            "reason": (
                                f"element {surface.identifier!r} has no active measured"
                                f" response at wavelength {current.wavelength_nm} nm"
                                " for every s/p channel; missing channels stay unknown"
                                " instead of defaulting to lossless transmission"
                            ),
                            "final_direction": incident,
                            "absorbed_power": 0.0,
                        })
                        event["event_uid"] = event_counter
                        event_counter += 1
                        leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                        continue
                    (
                        reflected_aff,
                        transmitted_aff,
                        absorbed_aff,
                        reflected_field,
                        transmitted_field,
                        s_fraction,
                        measured_info,
                    ) = measured_split
                    reflected_weight = reflected_aff.center
                    transmitted_weight = transmitted_aff.center
                    absorbed_power = max(0.0, absorbed_aff.center)
                    event.update(measured_info)
                else:
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
                    if state.get("weight_aff") is not None:
                        reflected_aff = state["weight_aff"].scale(reflected_fraction)
                        transmitted_aff = state["weight_aff"].scale(transmitted_fraction)
                        absorbed_aff = (
                            state["weight_aff"] - reflected_aff - transmitted_aff
                        )
                    event.update({
                        "reflectivity": reflected_fraction,
                        "transmissivity": transmitted_fraction,
                    })
                event.update({
                    "action": "split",
                    "cosine_incidence": cosine,
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
                        "weight_aff": reflected_aff,
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
                        "weight_aff": transmitted_aff,
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
                reflected_aff = None
                if self.measurements is not None and self.measurements.covers_surface(surface.identifier):
                    found = self.measurements.response_for(
                        surface.identifier, current.wavelength_nm, "unpolarized"
                    )
                    if found is None:
                        event.update({
                            "action": "unknown_measured_response",
                            "reason": (
                                f"mirror {surface.identifier!r} has no active measured"
                                f" response at wavelength {current.wavelength_nm} nm;"
                                " a missing channel stays unknown instead of"
                                " defaulting to lossless reflection"
                            ),
                            "final_direction": incident,
                            "absorbed_power": 0.0,
                        })
                        event["event_uid"] = event_counter
                        event_counter += 1
                        leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                        continue
                    batch_name, response = found
                    reflectivity_aff, _, absorptivity_aff = self._channel_forms(
                        surface.identifier, current.wavelength_nm, "unpolarized",
                        batch_name, response,
                    )
                    base_aff = state.get("weight_aff") or AffineForm.constant(weight)
                    reflected_aff = base_aff.multiply(
                        reflectivity_aff,
                        noise_symbol=(
                            f"nonlinear:{surface.identifier}:{state['branch_id']}:{event_counter}"
                        ),
                    )
                    absorbed_aff = base_aff - reflected_aff
                    new_weight = reflected_aff.center
                    absorbed = max(0.0, absorbed_aff.center)
                    event.update({
                        "measured": True,
                        "measurement_batch": batch_name,
                        "reflectivity_interval": reflectivity_aff.to_interval(),
                        "absorptivity_interval": absorptivity_aff.to_interval(),
                        "reflected_power_range": reflected_aff.to_interval(),
                        "absorbed_power_range": absorbed_aff.to_interval(),
                    })
                else:
                    absorbed = weight * (1.0 - surface.reflectivity)
                    new_weight = weight * surface.reflectivity
                    if state.get("weight_aff") is not None:
                        reflected_aff = state["weight_aff"].scale(surface.reflectivity)
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
                    advanced = self._advance(state, event, outgoing, surface.identifier, state["saw_tir"], event_counter - 1)
                    advanced["weight_aff"] = reflected_aff
                    pending.append(advanced)
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
            weight_aff = state.get("weight_aff")
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
                    "weight_aff": (
                        weight_aff.scale(reflected_fraction) if weight_aff is not None else None
                    ),
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
                "weight_aff": (
                    weight_aff.scale(transmitted_fraction) if weight_aff is not None else None
                ),
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
    def _advance(state: dict[str, Any], event: dict[str, Any], ray: Ray, surface_identifier: str, saw_tir: bool, event_uid: int) -> dict[str, Any]:
        return {
            "ray": ray,
            "branch_id": state["branch_id"],
            "parent_branch_id": state["parent_branch_id"],
            "depth": state["depth"] + 1,
            "events": RayTracer._extended_events(state["events"], event, state["branch_id"], event_uid),
            "last_surface": surface_identifier,
            "saw_tir": saw_tir,
            "weight_aff": state.get("weight_aff"),
        }

    @staticmethod
    def _leaf(seed: Ray, state: dict[str, Any], status: str, endpoint, detector: str | None = None) -> RayResult:
        weight_aff = state.get("weight_aff")
        weight_range = None
        weight_terms = None
        if weight_aff is not None and not weight_aff.is_definite:
            weight_range = weight_aff.to_interval()
            weight_terms = dict(weight_aff.terms)
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
            weight_range=weight_range,
            weight_terms=weight_terms,
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

    def _channel_forms(self, surface_id: str, wavelength_nm, axis: str, batch_name: str, response):
        """Correlated affine forms for one measured channel.

        Measured quantities become shared symbols named by batch and channel;
        the third quantity is derived as the exact energy complement so
        R + T + A closes symbolically instead of adding independent error.
        """

        prefix = f"meas:{batch_name}:{channel_key(surface_id, wavelength_nm, axis)}"
        affs = {}
        for quantity, attribute in (
            ("R", "reflectivity"),
            ("T", "transmissivity"),
            ("A", "absorptivity"),
        ):
            interval = getattr(response, attribute)
            if interval is not None:
                affs[quantity] = AffineForm.from_interval(interval, f"{prefix}|{quantity}")
        if len(affs) == 3:
            # All three measured: consistency was validated on construction;
            # derive A so energy closure stays exact and correlated.
            del affs["A"]
        if len(affs) == 2:
            missing = next(quantity for quantity in "RTA" if quantity not in affs)
            first, second = (affs[quantity] for quantity in "RTA" if quantity != missing)
            affs[missing] = AffineForm.constant(1.0) - first - second
        return affs["R"], affs["T"], affs["A"]

    def _measured_split(self, surface: Surface, ray: Ray, incident, normal, state, event_uid: int):
        """Beamsplitter split from measured interval responses.

        Returns None when any required s/p channel lacks an active
        measurement; the caller terminates the branch as unknown instead of
        falling back to a lossless default.
        """

        per_axis = {}
        batches = {}
        for axis in ("s", "p"):
            found = self.measurements.response_for(surface.identifier, ray.wavelength_nm, axis)
            if found is None:
                return None
            batch_name, response = found
            batches[axis] = batch_name
            per_axis[axis] = self._channel_forms(
                surface.identifier, ray.wavelength_nm, axis, batch_name, response
            )
        (r_s, t_s, _), (r_p, t_p, _) = per_axis["s"], per_axis["p"]
        s_axis = p_axis = None
        e_s = e_p = None
        if ray.polarization is None:
            s_fraction = 0.5
            p_fraction = 0.5
        else:
            s_axis, p_axis = self._sp_basis(incident, normal)
            e_s = dot(ray.polarization, s_axis)
            e_p = dot(ray.polarization, p_axis)
            s_fraction = e_s * e_s
            p_fraction = e_p * e_p
        r_combined = r_s.scale(s_fraction) + r_p.scale(p_fraction)
        t_combined = t_s.scale(s_fraction) + t_p.scale(p_fraction)
        base = state.get("weight_aff")
        if base is None:
            base = AffineForm.constant(state["ray"].weight)
        noise = f"nonlinear:{surface.identifier}:{state['branch_id']}:{event_uid}"
        reflected_aff = base.multiply(r_combined, noise_symbol=f"{noise}:r")
        transmitted_aff = base.multiply(t_combined, noise_symbol=f"{noise}:t")
        absorbed_aff = base - reflected_aff - transmitted_aff
        if ray.polarization is None:
            reflected_field = None
            transmitted_field = None
        else:
            reflected = self._reflect(incident, scale(normal, -1.0))
            reflected_p_axis = normalize(cross(s_axis, reflected))
            reflected_field = self._combine_field(
                e_s * math.sqrt(max(0.0, r_s.center)), s_axis,
                e_p * math.sqrt(max(0.0, r_p.center)), reflected_p_axis,
                ray.polarization,
            )
            transmitted_field = self._combine_field(
                e_s * math.sqrt(max(0.0, t_s.center)), s_axis,
                e_p * math.sqrt(max(0.0, t_p.center)), p_axis,
                ray.polarization,
            )
        info = {
            "measured": True,
            "measurement_batches": dict(batches),
            "reflectivity_interval": r_combined.to_interval(),
            "transmissivity_interval": t_combined.to_interval(),
            "reflected_power_range": reflected_aff.to_interval(),
            "transmitted_power_range": transmitted_aff.to_interval(),
            "absorbed_power_range": absorbed_aff.to_interval(),
        }
        return (
            reflected_aff,
            transmitted_aff,
            absorbed_aff,
            reflected_field,
            transmitted_field,
            s_fraction,
            info,
        )

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
