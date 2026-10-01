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
from typing import Any, Mapping, Sequence

from .materials import MaterialLibrary, fresnel_coefficients
from .surfaces import Surface, add, dot, normalize, scale


@dataclass(frozen=True)
class Ray:
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    wavelength_nm: float | None = None
    weight: float = 1.0
    label: str | None = None

    def normalized(self) -> "Ray":
        return Ray(self.origin, normalize(self.direction), self.wavelength_nm, self.weight, self.label)


@dataclass(frozen=True)
class Beam:
    """One input beam carrying several spectral components.

    Each wavelength is traced as its own source ray: branches, directions,
    energies and terminal reasons are never averaged across wavelengths.
    """

    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    wavelengths: Sequence[float]
    weights: Sequence[float] | None = None
    label: str | None = None

    def rays(self) -> list[Ray]:
        wavelengths = list(self.wavelengths)
        if not wavelengths:
            raise ValueError("a beam must contain at least one wavelength")
        if len(set(float(w) for w in wavelengths)) != len(wavelengths):
            raise ValueError("beam wavelengths must be distinct")
        weights = [1.0] * len(wavelengths) if self.weights is None else list(self.weights)
        if len(weights) != len(wavelengths):
            raise ValueError("beam weights must match wavelengths one for one")
        return [
            Ray(self.origin, self.direction, float(wavelength), float(weight), self.label)
            for wavelength, weight in zip(wavelengths, weights)
        ]

    @classmethod
    def equal(cls, origin, direction, wavelengths: Sequence[float], label: str | None = None, weight: float = 1.0) -> "Beam":
        return cls(origin, direction, list(wavelengths), [weight] * len(list(wavelengths)), label)


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

        transmitted_only = re.compile(r"^root(?:@[^.]+)?(?:\.\d+\.t)*$")
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
        )

    def trace_branches(self, sources: Sequence[Ray | Beam]) -> list[RayResult]:
        """Trace every source ray/beam and return all terminal sub-beams.

        Different wavelengths of one beam are expanded into independent
        source rays; they never merge into a single polychromatic branch.
        Global event numbering still spans every wavelength so per-source
        energy ledgers cannot double-count a shared event id.
        """

        leaves: list[RayResult] = []
        seed_rays = list(_expand_sources(sources))
        event_counter = 0
        for source_index, source_ray in enumerate(seed_rays):
            normalized = source_ray.normalized()
            source_label = (
                normalized.label
                if normalized.label is not None
                else f"ray-{source_index}"
            )
            seed = Ray(
                normalized.origin,
                normalized.direction,
                normalized.wavelength_nm,
                normalized.weight,
                source_label,
            )
            source_leaves, event_counter = self._trace_one(seed, event_counter)
            leaves.extend(source_leaves)
        return leaves


    def _trace_one(self, seed: Ray, event_counter: int = 0) -> tuple[list[RayResult], int]:
        settings = self.settings
        root_branch_id = "root" if seed.wavelength_nm is None else f"root@{float(seed.wavelength_nm):g}nm"
        pending: list[dict[str, Any]] = [
            {
                "ray": seed,
                "branch_id": root_branch_id,
                "parent_branch_id": None,
                "depth": 0,
                "events": [],
                "last_surface": None,
                "saw_tir": False,
            }
        ]
        leaves: list[RayResult] = []
        branch_count = 0

        while pending:
            branch_count += 1
            if branch_count > settings.max_branches:
                for overflow in pending:
                    leaves.append(self._leaf(seed, overflow, "max_branches", overflow["ray"].origin))
                return leaves, event_counter

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
                "wavelength_nm": current.wavelength_nm,
                "surface": surface.identifier,
                "kind": surface.kind,
                "distance": t,
                "point": point,
                "normal": normal,
                "incident_direction": incident,
                "incident_weight": weight,
                "aperture_edge": edge_hit,
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
                outgoing = Ray(
                    self._offset(point, incident),
                    incident,
                    current.wavelength_nm,
                    weight,
                    seed.label,
                )
                pending.append(self._advance(state, event, outgoing, surface.identifier, state["saw_tir"], event_counter - 1))
                continue

            if surface.kind == "beamsplitter":
                reflected = self._reflect(current.direction, scale(normal, -1.0))
                transmitted = tuple(incident)
                # ``reflectivity`` always means the reflected power
                # fraction, matching mirrors; transmission defaults to the
                # lossless complement and may be set explicitly.
                reflected_fraction, transmitted_fraction, spectral_response = surface.spectral_response(
                    current.wavelength_nm
                )
                reflected_weight = weight * reflected_fraction
                transmitted_weight = weight * transmitted_fraction
                absorbed_power = max(0.0, weight - reflected_weight - transmitted_weight)
                event.update({
                    "action": "split",
                    "cosine_incidence": cosine,
                    "reflectivity": reflected_fraction,
                    "transmissivity": transmitted_fraction,
                    "spectral_response": spectral_response,
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
                        "ray": Ray(self._offset(point, reflected), reflected,
                                  current.wavelength_nm, reflected_weight, seed.label),
                        "branch_id": f"{state['branch_id']}.{len(base_events)}.r",
                        "parent_branch_id": state["branch_id"],
                    })
                if transmitted_weight > 0.0:
                    children.append({
                        **common,
                        "events": list(base_events),
                        "ray": Ray(self._offset(point, transmitted), transmitted,
                                  current.wavelength_nm, transmitted_weight, seed.label),
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
                reflected_fraction, _, spectral_response = surface.spectral_response(current.wavelength_nm)
                absorbed = weight * (1.0 - reflected_fraction)
                new_weight = weight * reflected_fraction
                event.update({
                    "action": "reflect",
                    "cosine_incidence": cosine,
                    "reflectivity": reflected_fraction,
                    "spectral_response": spectral_response,
                    "reflected_power": new_weight,
                    "absorbed_power": absorbed,
                    "transmitted_power": 0.0,
                    "outgoing_direction": reflected,
                    "outgoing_weight": new_weight,
                })
                event["event_uid"] = event_counter
                event_counter += 1
                if new_weight > 0.0:
                    outgoing = Ray(
                        self._offset(point, reflected),
                        reflected,
                        current.wavelength_nm,
                        new_weight,
                        seed.label,
                    )
                    pending.append(self._advance(state, event, outgoing, surface.identifier, state["saw_tir"], event_counter - 1))
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
            n1 = self.materials.index(n1_name, current.wavelength_nm)
            n2 = self.materials.index(n2_name, current.wavelength_nm)
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
                outgoing = Ray(
                    self._offset(point, reflected),
                    reflected,
                    current.wavelength_nm,
                    weight,
                    seed.label,
                )
                pending.append(self._advance(state, event, outgoing, surface.identifier, True, event_counter - 1))
                continue

            refracted = normalize(tuple(
                eta * current.direction[i] + (eta * cos_i - math.sqrt(k)) * against_normal[i]
                for i in range(3)
            ))
            reflected = self._reflect(current.direction, scale(normal, -1.0))
            reflected_fraction, transmitted_fraction = fresnel_coefficients(n1, n2, cos_i)
            # Bulk loss belongs to the medium the transmitted branch enters;
            # it is kept separate from Fresnel/surface absorption and from
            # aperture clipping in the energy ledger.
            transmitted_medium = n2_name
            bulk_loss_fraction = self.materials.loss(transmitted_medium, current.wavelength_nm)
            reflected_weight = weight * reflected_fraction
            transmitted_weight = weight * transmitted_fraction * (1.0 - bulk_loss_fraction)
            surface_absorbed_power = max(0.0, weight - weight * reflected_fraction - weight * transmitted_fraction)
            bulk_absorbed_power = weight * transmitted_fraction * bulk_loss_fraction
            absorbed_power = surface_absorbed_power + bulk_absorbed_power
            event.update({
                "action": "refract",
                "n_incident": n1,
                "n_transmitted": n2,
                "cosine_incidence": cos_i,
                "reflected_power": reflected_weight,
                "transmitted_power": transmitted_weight,
                "absorbed_power": absorbed_power,
                "surface_absorbed_power": surface_absorbed_power,
                "bulk_absorbed_power": bulk_absorbed_power,
                "bulk_loss_fraction": bulk_loss_fraction,
                "transmitted_material": transmitted_medium,
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
                reflected_ray = Ray(
                    self._offset(point, reflected),
                    reflected,
                    current.wavelength_nm,
                    reflected_weight,
                    seed.label,
                )
                pending.append({
                    **common,
                    "events": list(base_events),
                    "ray": reflected_ray,
                    "branch_id": f"{state['branch_id']}.{len(base_events)}.r",
                    "parent_branch_id": state["branch_id"],
                })
            transmitted_ray = Ray(
                self._offset(point, refracted),
                refracted,
                current.wavelength_nm,
                transmitted_weight,
                seed.label,
            )
            pending.append({
                **common,
                "events": list(base_events),
                "ray": transmitted_ray,
                "branch_id": f"{state['branch_id']}.{len(base_events)}.t",
                "parent_branch_id": state["branch_id"],
            })

        return leaves, event_counter

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
        }

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


def _expand_sources(sources: Sequence[Ray | Beam]) -> list[Ray]:
    rays: list[Ray] = []
    for source in sources:
        if isinstance(source, Beam):
            rays.extend(source.rays())
        else:
            rays.append(source)
    return rays
