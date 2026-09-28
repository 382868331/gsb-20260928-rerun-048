"""Sequential geometric ray tracing."""

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


@dataclass
class RayResult:
    ray: Ray
    status: str
    endpoint: tuple[float, float, float]
    weight: float
    events: list[dict[str, Any]] = field(default_factory=list)


class RayTracer:
    def __init__(self, surfaces: list[Surface], materials: MaterialLibrary, settings: TraceSettings | None = None):
        self.surfaces = surfaces
        self.materials = materials
        self.settings = settings or TraceSettings()

    def trace(self, ray: Ray) -> RayResult:
        current = ray.normalized()
        weight = 1.0
        events: list[dict[str, Any]] = []
        last_surface: str | None = None
        saw_tir = False

        for _ in range(self.settings.max_events):
            hit = self._nearest_hit(current, last_surface)
            if hit is None:
                return RayResult(ray, "no_image", current.origin, weight, events)

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
                return RayResult(ray, "grazing", point, weight, events)

            if surface.kind == "detector":
                event.update({"action": "land", "final_direction": incident})
                events.append(event)
                return RayResult(ray, "reached_image", point, weight, events)

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
                saw_tir = True
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
            reflected_fraction, transmitted_fraction = fresnel_coefficients(n1, n2, cos_i)
            reflected_power = weight * reflected_fraction
            transmitted_weight = weight * transmitted_fraction
            absorbed_power = max(0.0, weight - reflected_power - transmitted_weight)
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
            })
            events.append(event)
            current = Ray(self._offset(point, refracted), refracted, current.wavelength_nm)
            last_surface = surface.identifier
            weight = transmitted_weight

        return RayResult(ray, "max_events" if saw_tir else "no_image", current.origin, weight, events)

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
