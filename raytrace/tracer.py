"""Sequential geometric ray tracing with beam splitting."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .materials import MaterialLibrary, fresnel_coefficients
from .surfaces import Surface, add, dot, normalize, scale


@dataclass(frozen=True)
class Ray:
    origin: tuple[float, float, float]
    direction: tuple[float, float, float]
    wavelength_nm: float | None = None

    def normalized(self) -> "Ray":
        return Ray(self.origin, normalize(self.direction), self.wavelength_nm)


@dataclass(frozen=True)
class TraceSettings:
    max_events: int = 64
    ray_offset: float = 1e-9
    grazing_cosine: float = 1e-6
    min_weight: float = 1e-6
    max_generations: int = 32


@dataclass
class RayResult:
    ray: Ray
    status: str
    endpoint: tuple[float, float, float]
    weight: float
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BeamResult:
    """One branch of a split ray tree.

    beam_id is unique within one trace_beams call; parent_id links the beam
    to the beam it split from (None for the root).  branch records how the
    beam was created: "root", "transmitted" or "reflected".
    """

    beam_id: int
    parent_id: int | None
    branch: str
    generation: int
    ray: Ray
    status: str
    endpoint: tuple[float, float, float]
    weight: float
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_ray_result(self) -> RayResult:
        return RayResult(self.ray, self.status, self.endpoint, self.weight, self.events)


class RayTracer:
    def __init__(self, surfaces: list[Surface], materials: MaterialLibrary, settings: TraceSettings | None = None):
        self.surfaces = surfaces
        self.materials = materials
        self.settings = settings or TraceSettings()

    def trace(self, ray: Ray) -> RayResult:
        """Trace only the primary (root) path, for backward compatibility."""
        for beam in self.trace_beams(ray):
            if beam.parent_id is None:
                return beam.to_ray_result()
        raise RuntimeError("trace_beams did not return a root beam")

    def trace_beams(self, ray: Ray) -> list[BeamResult]:
        """Trace a ray and every child beam split off by reflection/refraction.

        Beams are returned in creation order (breadth-first).  A beam ends
        with one of the statuses "reached_image", "no_image", "grazing",
        "max_events", "min_weight" or "max_generations".  The residual weight
        of beams stopped by "min_weight"/"max_events"/"max_generations" is
        NOT absorbed energy; callers must book it separately (see
        TraceBatch.energy_report).
        """
        beams: list[BeamResult] = []
        queue: list[tuple[Ray, float, int | None, str, int]] = [(ray.normalized(), 1.0, None, "root", 0)]
        while queue:
            current, weight, parent_id, branch, generation = queue.pop(0)
            beam_id = len(beams)
            if weight < self.settings.min_weight:
                beams.append(BeamResult(beam_id, parent_id, branch, generation, current,
                                        "min_weight", current.origin, weight, []))
                continue
            if generation > self.settings.max_generations:
                beams.append(BeamResult(beam_id, parent_id, branch, generation, current,
                                        "max_generations", current.origin, weight, []))
                continue
            beam, children = self._trace_single(current, weight, beam_id, parent_id, branch, generation)
            beams.append(beam)
            for child_ray, child_weight, child_branch in children:
                queue.append((child_ray, child_weight, beam_id, child_branch, generation + 1))
        return beams

    def _trace_single(self, ray: Ray, weight: float, beam_id: int,
                      parent_id: int | None, branch: str, generation: int):
        current = ray
        events: list[dict[str, Any]] = []
        last_surface: str | None = None
        children: list[tuple[Ray, float, str]] = []

        def finish(status: str, endpoint):
            return BeamResult(beam_id, parent_id, branch, generation, ray,
                              status, endpoint, weight, events), children

        for _ in range(self.settings.max_events):
            if weight < self.settings.min_weight:
                return finish("min_weight", current.origin)
            hit = self._nearest_hit(current, last_surface)
            if hit is None:
                return finish("no_image", current.origin)

            t, point, normal, surface = hit
            incident = tuple(current.direction)
            edge_hit = surface.at_aperture_edge(point)
            cosine = abs(dot(current.direction, normal))
            event: dict[str, Any] = {
                "index": len(events),
                "surface": surface.identifier,
                "kind": surface.kind,
                "distance": t,
                "point": point,
                "normal": normal,
                "incident_direction": incident,
                "incident_weight": weight,
                "aperture_edge": edge_hit,
            }

            if edge_hit or cosine < self.settings.grazing_cosine:
                event.update({
                    "action": "grazing",
                    "cosine_incidence": cosine,
                    "final_direction": incident,
                })
                events.append(event)
                return finish("grazing", point)

            if surface.kind == "detector":
                event.update({"action": "land", "final_direction": incident})
                events.append(event)
                return finish("reached_image", point)

            if surface.kind == "mirror":
                reflected = self._reflect(current.direction, normal)
                absorbed = weight * (1.0 - surface.reflectivity)
                new_weight = weight * surface.reflectivity
                event.update({
                    "action": "reflect",
                    "cosine_incidence": cosine,
                    "reflected_power": weight - absorbed,
                    "absorbed_power": absorbed,
                    "transmitted_power": 0.0,
                    "outgoing_direction": reflected,
                    "outgoing_weight": new_weight,
                })
                events.append(event)
                current = Ray(self._offset(point, reflected), reflected, current.wavelength_nm)
                last_surface = surface.identifier
                weight = new_weight
                continue

            traveling_along_axis = dot(current.direction, surface.axis) > 0
            if traveling_along_axis:
                n1_name, n2_name = surface.front_material, surface.back_material
            else:
                n1_name, n2_name = surface.back_material, surface.front_material
            n1 = self.materials.index(n1_name, current.wavelength_nm)
            n2 = self.materials.index(n2_name, current.wavelength_nm)
            against_normal = scale(normal, -1.0 if dot(current.direction, normal) > 0 else 1.0)
            cos_i = -dot(current.direction, against_normal)
            eta = n1 / n2
            k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)

            if k < 0:
                reflected = self._reflect(current.direction, normal)
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
                events.append(event)
                current = Ray(self._offset(point, reflected), reflected, current.wavelength_nm)
                last_surface = surface.identifier
                continue

            refracted = normalize(tuple(
                eta * current.direction[i] + (eta * cos_i - math.sqrt(k)) * against_normal[i]
                for i in range(3)
            ))
            reflected_dir = self._reflect(current.direction, normal)
            reflected_fraction, transmitted_fraction = fresnel_coefficients(n1, n2, cos_i)
            reflected_power = weight * reflected_fraction
            transmitted_weight = weight * transmitted_fraction
            absorbed_power = max(0.0, weight - reflected_power - transmitted_weight)
            children.append((
                Ray(self._offset(point, reflected_dir), reflected_dir, current.wavelength_nm),
                reflected_power,
                "reflected",
            ))
            event.update({
                "action": "refract",
                "n_incident": n1,
                "n_transmitted": n2,
                "cosine_incidence": cos_i,
                "reflected_power": reflected_power,
                "transmitted_power": transmitted_weight,
                "absorbed_power": absorbed_power,
                "outgoing_direction": refracted,
                "outgoing_weight": transmitted_weight,
                "split": True,
            })
            events.append(event)
            current = Ray(self._offset(point, refracted), refracted, current.wavelength_nm)
            last_surface = surface.identifier
            weight = transmitted_weight

        return finish("max_events", current.origin)

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
