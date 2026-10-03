"""Branching geometric ray tracing.

Refractive interfaces split an incoming ray into a transmitted and a
reflected child beam.  Both children carry the remaining power of their
parent and are traced independently; every leaf keeps its ancestry so that
detector hits and energy accounts can be grouped by source ray.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any

from .materials import (
    MaterialLibrary,
    UnknownMaterialResponse,
    UnknownPolarizationResponse,
    polarized_fresnel_coefficients,
)
from .surfaces import Surface, add, cross, dot, normalize, scale


@dataclass(frozen=True)
class Ray:
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    wavelength_nm: float | None = None
    weight: float = 1.0
    label: str | None = None
    spectral_weights: tuple[tuple[float, float], ...] | None = None
    polarization: str | None = None
    polarization_weights: tuple[tuple[str, float], ...] | None = None
    electric_direction: tuple[float, float, float] | None = None

    def normalized(self) -> "Ray":
        return Ray(
            self.origin,
            normalize(self.direction),
            self.wavelength_nm,
            self.weight,
            self.label,
            self.spectral_weights,
            self.polarization,
            self.polarization_weights,
            None if self.electric_direction is None else normalize(self.electric_direction),
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
        return cls(
            origin,
            direction,
            None,
            weight,
            label,
            tuple(zip(wavelengths, fractions)),
        )

    def polarization_components(self) -> tuple[tuple[str, float], ...]:
        """Return incoherent s/p components with absolute power."""

        if not self.polarization_weights:
            return ((self.polarization, self.weight),)
        total_fraction = sum(weight for _, weight in self.polarization_weights)
        if total_fraction <= 0:
            raise ValueError("polarization component weights must have a positive sum")
        channels = [channel for channel, _ in self.polarization_weights]
        if set(channels) - {"s", "p"} or len(channels) != len(set(channels)):
            raise ValueError("polarization components must be unique 's' and/or 'p' channels")
        components = []
        for channel, fraction in self.polarization_weights:
            if fraction < 0:
                raise ValueError("polarization component weights must be non-negative")
            if fraction > 0:
                components.append((channel, self.weight * fraction / total_fraction))
        if not components:
            raise ValueError("a polarized ray requires at least one positive component")
        return tuple(components)

    @classmethod
    def polarized(cls, origin, direction, components: dict[str, float] | tuple[tuple[str, float], ...],
                  *, wavelength_nm: float | None = None, weight: float = 1.0,
                  label: str | None = None) -> "Ray":
        items = tuple(components.items()) if isinstance(components, dict) else tuple(components)
        return cls(origin, direction, wavelength_nm, weight, label, None, None, tuple(items))

    @classmethod
    def linear_polarized(cls, origin, direction, electric_direction,
                         *, wavelength_nm: float | None = None, weight: float = 1.0,
                         label: str | None = None) -> "Ray":
        propagation = normalize(direction)
        field = normalize(electric_direction)
        if abs(dot(propagation, field)) > 1e-10:
            raise ValueError("electric field direction must be perpendicular to propagation")
        return cls(origin, direction, wavelength_nm, weight, label, None, None, None, field)


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
    polarization: str | None = None


class RayTracer:
    def __init__(self, surfaces: list[Surface], materials: MaterialLibrary, settings: TraceSettings | None = None):
        self.surfaces = surfaces
        self.materials = materials
        self.settings = settings or TraceSettings()

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

        leaves: list[RayResult] = []
        for source_index, source_ray in enumerate(rays):
            normalized = source_ray.normalized()
            source_label = (
                normalized.label
                if normalized.label is not None
                else f"ray-{source_index}"
            )
            wavelength_components = normalized.spectral_components()
            for wavelength, wavelength_weight in wavelength_components:
                polarization_components = self._initial_polarization_components(normalized, wavelength)
                for polarization, polarization_weight in polarization_components:
                    component_weight = wavelength_weight * polarization_weight / normalized.weight
                    seed = Ray(
                        normalized.origin,
                        normalized.direction,
                        wavelength,
                        component_weight,
                        source_label,
                        None,
                        polarization,
                    )
                    root_parts = []
                    if normalized.spectral_weights:
                        root_parts.append(f"wavelength={wavelength:g}")
                    if polarization is not None:
                        root_parts.append(f"polarization={polarization}")
                    branch_root = "root[" + ",".join(root_parts) + "]" if root_parts else "root"
                    leaves.extend(self._trace_one(seed, branch_root))
        return leaves

    def _initial_polarization_components(
        self,
        ray: Ray,
        wavelength_nm: float | None,
    ) -> tuple[tuple[str | None, float], ...]:
        if ray.polarization_weights:
            return ray.polarization_components()
        if ray.electric_direction is None:
            return ((ray.polarization, ray.weight),)

        probe = Ray(ray.origin, ray.direction, wavelength_nm, ray.weight, ray.label)
        hit = self._nearest_hit(probe, None)
        if hit is None:
            return ((None, ray.weight),)
        _, _, hit_normal, surface = hit
        normal = scale(hit_normal, -1.0 if dot(ray.direction, hit_normal) > 0 else 1.0)
        s_direction = self._s_direction(ray.direction, normal, surface.axis)
        p_direction = normalize(cross(ray.direction, s_direction))
        s_weight = dot(ray.electric_direction, s_direction) ** 2
        p_weight = dot(ray.electric_direction, p_direction) ** 2
        total = s_weight + p_weight
        if total <= 0:
            raise ValueError("electric field direction has no projection onto the surface polarization basis")
        return ("s", ray.weight * s_weight / total), ("p", ray.weight * p_weight / total)

    @staticmethod
    def _s_direction(direction, normal, fallback_axis):
        candidate = cross(normal, direction)
        if dot(candidate, candidate) > 1e-20:
            return normalize(candidate)
        for axis in (fallback_axis, (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)):
            tangent = cross(axis, direction)
            if dot(tangent, tangent) > 1e-20:
                return normalize(tangent)
        raise ValueError("cannot construct an s-polarization basis")

    @staticmethod
    def _ray_with_weight(current: Ray, origin, direction, weight: float, label: str,
                         electric_direction=None, normal=None, surface_axis=(1.0, 0.0, 0.0),
                         reflected=False) -> Ray:
        if electric_direction is None and current.polarization in {"s", "p"} and normal is not None:
            s_direction = RayTracer._s_direction(direction, normal, surface_axis)
            s_sign = -1.0 if reflected and current.polarization == "s" else 1.0
            electric_direction = (
                scale(s_direction, s_sign)
                if current.polarization == "s"
                else normalize(cross(direction, s_direction))
            )
        if electric_direction is None:
            electric_direction = current.electric_direction
        return Ray(
            origin,
            direction,
            current.wavelength_nm,
            weight,
            label,
            None,
            current.polarization,
            None,
            electric_direction,
        )

    @staticmethod
    def _analyzer_fractions(current: Ray, surface: Surface, normal, direction) -> tuple[float, float]:
        axis = surface.analyzer_axis
        if axis is None:
            raise UnknownPolarizationResponse(f"analyzer {surface.identifier!r} requires an analyzer axis")
        projected = tuple(axis[i] - dot(axis, normal) * normal[i] for i in range(3))
        if dot(projected, projected) <= 1e-24:
            raise UnknownPolarizationResponse("analyzer axis cannot be parallel to the surface normal")
        axis_in_plane = normalize(projected)
        if current.electric_direction is not None:
            field = current.electric_direction
        elif current.polarization in {"s", "p"}:
            s_direction = RayTracer._s_direction(direction, normal, surface.axis)
            field = s_direction if current.polarization == "s" else normalize(cross(direction, s_direction))
        else:
            if current.polarization is None:
                raise UnknownPolarizationResponse("an analyzer cannot substitute an unpolarized average for a known channel")
            raise UnknownPolarizationResponse(f"unsupported polarization {current.polarization!r}")
        transmitted = max(0.0, min(1.0, dot(field, axis_in_plane) ** 2))
        return transmitted, 1.0 - transmitted

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
                "medium": "air",
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
            pre_volume_weight = weight
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
            try:
                absorption_coefficient = self.materials.get(state["medium"]).absorption_coefficient(
                    current.wavelength_nm
                )
            except UnknownMaterialResponse as exc:
                event.update({
                    "action": "unknown_material_response",
                    "incident_material": state["medium"],
                    "transmitted_material": state["medium"],
                    "reason": str(exc),
                    "final_direction": incident,
                    "absorbed_power": 0.0,
                    "volume_loss_power": 0.0,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                continue
            volume_fraction = math.exp(-absorption_coefficient * t)
            volume_loss_power = weight * (1.0 - volume_fraction)
            weight *= volume_fraction
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
                "incident_weight": pre_volume_weight,
                "aperture_edge": edge_hit,
                "wavelength_nm": current.wavelength_nm,
                "polarization": current.polarization,
                "volume_loss_power": volume_loss_power,
                "post_volume_weight": weight,
            }

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

            if surface.kind == "detector" and surface.analyzer_axis is not None:
                try:
                    transmitted_fraction, rejected_fraction = self._analyzer_fractions(
                        current, surface, normal, incident
                    )
                except UnknownPolarizationResponse as exc:
                    event.update({
                        "action": "unknown_polarization_response",
                        "reason": str(exc),
                        "final_direction": incident,
                        "absorbed_power": 0.0,
                        "analyzer_rejected_power": 0.0,
                    })
                    event["event_uid"] = event_counter
                    event_counter += 1
                    leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                    continue
                detected_weight = weight * transmitted_fraction
                rejected_weight = weight * rejected_fraction
                event.update({
                    "action": "analyze",
                    "final_direction": incident,
                    "transmissivity": transmitted_fraction,
                    "analyzer_rejectivity": rejected_fraction,
                    "transmitted_power": detected_weight,
                    "analyzer_rejected_power": rejected_weight,
                    "absorbed_power": 0.0,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                if detected_weight > 0.0:
                    leaves.append(self._finish(seed, state, event, "reached_image", point, surface.identifier))
                if rejected_weight > 0.0:
                    rejected_state = dict(state)
                    rejected_state["ray"] = self._ray_with_weight(current, point, incident, rejected_weight, seed.label)
                    rejected_state["branch_id"] = f"{state['branch_id']}.{len(events) + 1}.x"
                    rejected_state["parent_branch_id"] = state["branch_id"]
                    rejected_state["depth"] = state["depth"] + 1
                    leaves.append(self._finish(seed, rejected_state, event, "analyzer_blocked", point, None))
                if detected_weight == 0.0 and rejected_weight == 0.0:
                    leaves.append(self._finish(seed, state, event, "analyzer_blocked", point, None))
                continue

            if surface.kind == "detector":
                event.update({
                    "action": "land",
                    "final_direction": incident,
                    "absorbed_power": 0.0,
                })
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
                outgoing = self._ray_with_weight(
                    current, self._offset(point, incident), incident, weight, seed.label
                )
                pending.append(self._advance(
                    state, event, outgoing, surface.identifier, state["saw_tir"], event_counter - 1,
                    medium=state["medium"],
                ))
                continue

            if surface.kind == "analyzer":
                try:
                    transmitted_fraction, rejected_fraction = self._analyzer_fractions(
                        current, surface, normal, incident
                    )
                except UnknownPolarizationResponse as exc:
                    event.update({
                        "action": "unknown_polarization_response",
                        "reason": str(exc),
                        "final_direction": incident,
                        "absorbed_power": 0.0,
                    })
                    event["event_uid"] = event_counter
                    event_counter += 1
                    leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                    continue
                transmitted_weight = weight * transmitted_fraction
                rejected_weight = weight * rejected_fraction
                event.update({
                    "action": "analyze",
                    "cosine_incidence": cosine,
                    "transmissivity": transmitted_fraction,
                    "analyzer_rejectivity": rejected_fraction,
                    "transmitted_power": transmitted_weight,
                    "analyzer_rejected_power": rejected_weight,
                    "reflected_power": 0.0,
                    "absorbed_power": 0.0,
                    "outgoing_direction": incident,
                    "outgoing_weight": transmitted_weight,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                base_events = self._extended_events(events, event, state["branch_id"], event_counter - 1)
                if rejected_weight > 0.0:
                    rejected_ray = self._ray_with_weight(current, point, incident, rejected_weight, seed.label)
                    leaves.append(RayResult(
                        ray=seed,
                        status="analyzer_blocked",
                        endpoint=point,
                        weight=rejected_weight,
                        events=list(base_events),
                        source_label=seed.label or "root",
                        branch_id=f"{state['branch_id']}.{len(base_events)}.x",
                        parent_branch_id=state["branch_id"],
                        detector=None,
                        depth=state["depth"] + 1,
                        wavelength_nm=seed.wavelength_nm,
                        polarization=seed.polarization,
                    ))
                if transmitted_weight > 0.0:
                    outgoing = self._ray_with_weight(
                        current, self._offset(point, incident), incident, transmitted_weight, seed.label
                    )
                    pending.append(self._advance(
                        state, event, outgoing, surface.identifier, state["saw_tir"], event_counter - 1,
                        events=list(base_events), medium=state["medium"],
                    ))
                elif rejected_weight == 0.0:
                    leaves.append(self._finish(seed, state, event, "analyzer_blocked", point, None))
                continue

            if surface.kind == "beamsplitter":
                reflected = self._reflect(current.direction, scale(normal, -1.0))
                transmitted = tuple(incident)
                # ``reflectivity`` always means the reflected power
                # fraction, matching mirrors; transmission defaults to the
                # lossless complement and may be set explicitly.
                try:
                    if current.polarization is not None:
                        reflected_fraction, transmitted_fraction = surface.polarized_fractions(
                            current.polarization, current.wavelength_nm
                        )
                    else:
                        reflected_fraction, transmitted_fraction = (
                            surface.reflectivity, surface.transmission_fraction()
                        )
                except UnknownPolarizationResponse as exc:
                    event.update({"action": "unknown_polarization_response", "reason": str(exc),
                                  "final_direction": incident, "absorbed_power": 0.0})
                    event["event_uid"] = event_counter
                    event_counter += 1
                    leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                    continue
                reflected_weight = weight * reflected_fraction
                transmitted_weight = weight * transmitted_fraction
                absorbed_power = max(0.0, weight - reflected_weight - transmitted_weight)
                event.update({
                    "action": "split",
                    "cosine_incidence": cosine,
                    "reflectivity": reflected_fraction,
                    "transmissivity": transmitted_fraction,
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
                    "medium": state["medium"],
                }
                children: list[dict[str, Any]] = []
                if reflected_weight > 0.0:
                    children.append({
                        **common,
                        "events": list(base_events),
                        "ray": self._ray_with_weight(
                            current, self._offset(point, reflected), reflected,
                            reflected_weight, seed.label, normal=normal,
                            surface_axis=surface.axis, reflected=True,
                        ),
                        "branch_id": f"{state['branch_id']}.{len(base_events)}.r",
                        "parent_branch_id": state["branch_id"],
                    })
                if transmitted_weight > 0.0:
                    children.append({
                        **common,
                        "events": list(base_events),
                        "ray": self._ray_with_weight(
                            current, self._offset(point, transmitted), transmitted,
                            transmitted_weight, seed.label, normal=normal,
                            surface_axis=surface.axis,
                        ),
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
                try:
                    if current.polarization is not None:
                        reflected_fraction, transmitted_fraction = surface.polarized_fractions(
                            current.polarization, current.wavelength_nm
                        )
                        absorbed_fraction = 1.0 - reflected_fraction - transmitted_fraction
                    else:
                        reflected_fraction = surface.reflectivity
                        absorbed_fraction = 1.0 - surface.reflectivity
                except UnknownPolarizationResponse as exc:
                    event.update({"action": "unknown_polarization_response", "reason": str(exc),
                                  "final_direction": incident, "absorbed_power": 0.0})
                    event["event_uid"] = event_counter
                    event_counter += 1
                    leaves.append(self._finish(seed, state, event, "unknown_response", point, None))
                    continue
                absorbed = weight * absorbed_fraction
                new_weight = weight * reflected_fraction
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
                    outgoing = self._ray_with_weight(
                        current, self._offset(point, reflected), reflected, new_weight,
                        seed.label, normal=normal, surface_axis=surface.axis, reflected=True,
                    )
                    pending.append(self._advance(
                        state, event, outgoing, surface.identifier, state["saw_tir"], event_counter - 1,
                        medium=state["medium"],
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
                outgoing = self._ray_with_weight(
                    current,
                    self._offset(point, reflected),
                    reflected,
                    weight,
                    seed.label,
                    normal=normal,
                    surface_axis=surface.axis,
                    reflected=True,
                )
                pending.append(self._advance(
                    state, event, outgoing, surface.identifier, True, event_counter - 1,
                    medium=state["medium"],
                ))
                continue

            refracted = normalize(tuple(
                eta * current.direction[i] + (eta * cos_i - math.sqrt(k)) * against_normal[i]
                for i in range(3)
            ))
            reflected = self._reflect(current.direction, scale(normal, -1.0))
            if current.polarization is None:
                reflected_fraction, transmitted_fraction = 0.5 * (
                    polarized_fresnel_coefficients(n1, n2, cos_i)["s"][0]
                    + polarized_fresnel_coefficients(n1, n2, cos_i)["p"][0]
                ), 0.0
                transmitted_fraction = 1.0 - reflected_fraction
            else:
                reflected_fraction, transmitted_fraction = polarized_fresnel_coefficients(
                    n1, n2, cos_i
                )[current.polarization]
            reflected_weight = weight * reflected_fraction
            transmitted_weight = weight * transmitted_fraction
            absorbed_power = max(0.0, weight - reflected_weight - transmitted_weight)
            event.update({
                "action": "refract",
                "n_incident": n1,
                "n_transmitted": n2,
                "cosine_incidence": cos_i,
                "reflected_power": reflected_weight,
                "transmitted_power": transmitted_weight,
                "absorbed_power": absorbed_power,
                "transmitted_direction": refracted,
                "reflected_direction": reflected,
                "outgoing_direction": refracted,
                "outgoing_weight": transmitted_weight,
                "reflectivity": reflected_fraction,
                "transmissivity": transmitted_fraction,
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
                "medium": n2_name,
            }
            if reflected_weight > 0.0:
                reflected_ray = self._ray_with_weight(
                    current, self._offset(point, reflected), reflected, reflected_weight,
                    seed.label, normal=normal, surface_axis=surface.axis, reflected=True,
                )
                pending.append({
                    **common,
                    "events": list(base_events),
                    "ray": reflected_ray,
                    "branch_id": f"{state['branch_id']}.{len(base_events)}.r",
                    "parent_branch_id": state["branch_id"],
                    "medium": n1_name,
                })
            transmitted_ray = self._ray_with_weight(
                current, self._offset(point, refracted), refracted, transmitted_weight,
                seed.label, normal=against_normal, surface_axis=surface.axis,
            )
            pending.append({
                **common,
                "events": list(base_events),
                "ray": transmitted_ray,
                "branch_id": f"{state['branch_id']}.{len(base_events)}.t",
                "parent_branch_id": state["branch_id"],
                "medium": n2_name,
            })

        return leaves

    @staticmethod
    def _extended_events(events: list[dict[str, Any]], event: dict[str, Any], branch_id: str, event_uid: int) -> list[dict[str, Any]]:
        recorded = dict(event)
        recorded["event_uid"] = event_uid
        recorded["branch_id"] = branch_id
        return [*events, recorded]

    @staticmethod
    def _advance(state: dict[str, Any], event: dict[str, Any], ray: Ray, surface_identifier: str,
                 saw_tir: bool, event_uid: int, events=None, medium: str | None = None) -> dict[str, Any]:
        advanced = {
            "ray": ray,
            "branch_id": state["branch_id"],
            "parent_branch_id": state["parent_branch_id"],
            "depth": state["depth"] + 1,
            "events": RayTracer._extended_events(
                state["events"] if events is None else events, event, state["branch_id"], event_uid
            ),
            "last_surface": surface_identifier,
            "saw_tir": saw_tir,
            "medium": state["medium"] if medium is None else medium,
        }
        return advanced

    @staticmethod
    def _leaf(seed: Ray, state: dict[str, Any], status: str, endpoint, detector: str | None = None) -> RayResult:
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
            polarization=seed.polarization,
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

    def _offset(self, point, direction) -> tuple[float, float, float]:
        return add(point, scale(direction, self.settings.ray_offset))
