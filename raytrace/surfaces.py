"""Surface intersection and interaction rules.

Coordinates use z as the optical axis. Refracting surfaces explicitly store
the material on each geometric side, so rays may propagate in either +z or -z.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .materials import Material
from .vectors import normalized

INF = float("inf")


@dataclass
class Hit:
    surface: "Surface"
    t: float
    point: np.ndarray
    geometric_normal: np.ndarray


@dataclass
class Surface:
    name: str
    aperture_radius: float = INF

    def intersect(
        self, origin: np.ndarray, direction: np.ndarray, t_min: float
    ) -> Hit | None:
        raise NotImplementedError

    def interaction(
        self, direction: np.ndarray, hit: Hit, wavelength_um: float
    ) -> dict[str, Any]:
        raise NotImplementedError


def within_aperture(point: np.ndarray, aperture_radius: float) -> bool:
    return float(np.hypot(point[0], point[1])) <= aperture_radius + 1.0e-9


def oriented_normal(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return normal if float(np.dot(direction, normal)) < 0.0 else -normal


def intersect_plane(
    surface: Surface,
    origin: np.ndarray,
    direction: np.ndarray,
    t_min: float,
    z: float,
    geometric_normal: np.ndarray | None = None,
) -> Hit | None:
    if abs(float(direction[2])) <= 1.0e-12:
        return None
    t = (z - float(origin[2])) / float(direction[2])
    if t <= t_min:
        return None
    point = origin + direction * t
    if not within_aperture(point, surface.aperture_radius):
        return None
    return Hit(surface, t, point, geometric_normal or np.array([0.0, 0.0, 1.0]))


def intersect_sphere(
    surface: Surface,
    origin: np.ndarray,
    direction: np.ndarray,
    t_min: float,
    vertex_z: float,
    curvature_radius: float,
) -> Hit | None:
    center = np.array([0.0, 0.0, vertex_z - curvature_radius])
    offset = origin - center
    b = float(np.dot(direction, offset))
    c = float(np.dot(offset, offset) - curvature_radius * curvature_radius)
    discriminant = b * b - c
    if discriminant < -1.0e-10:
        return None
    root = math.sqrt(max(0.0, discriminant))
    for t in (-b - root, -b + root):
        if t <= t_min:
            continue
        point = origin + direction * t
        if within_aperture(point, surface.aperture_radius):
            normal = normalized((point - center) / abs(curvature_radius))
            return Hit(surface, t, point, normal)
    return None


@dataclass
class PlaneInterface(Surface):
    z: float = 0.0
    material_normal_side: Material | None = None
    material_opposite_side: Material | None = None

    def intersect(self, origin, direction, t_min) -> Hit | None:
        return intersect_plane(self, origin, direction, t_min, self.z)

    def interaction(self, direction, hit, wavelength_um) -> dict[str, Any]:
        return refract_interaction(
            direction,
            hit.geometric_normal,
            self.material_normal_side,
            self.material_opposite_side,
            wavelength_um,
        )


@dataclass
class SphericalInterface(Surface):
    """Spherical refracting surface with a signed curvature radius."""

    z: float = 0.0
    curvature_radius: float = 1.0
    material_normal_side: Material | None = None
    material_opposite_side: Material | None = None

    def intersect(self, origin, direction, t_min) -> Hit | None:
        return intersect_sphere(
            self, origin, direction, t_min, self.z, self.curvature_radius
        )

    def interaction(self, direction, hit, wavelength_um) -> dict[str, Any]:
        return refract_interaction(
            direction,
            hit.geometric_normal,
            self.material_normal_side,
            self.material_opposite_side,
            wavelength_um,
        )


def refract_interaction(
    direction,
    geometric_normal,
    material_normal_side,
    material_opposite_side,
    wavelength_um,
) -> dict[str, Any]:
    if material_normal_side is None or material_opposite_side is None:
        raise ValueError("refracting surface requires materials on both sides")
    normal = oriented_normal(direction, geometric_normal)
    if float(np.dot(geometric_normal, normal)) > 0.0:
        n_in = material_normal_side.refractive_index(wavelength_um)
        n_out = material_opposite_side.refractive_index(wavelength_um)
    else:
        n_in = material_opposite_side.refractive_index(wavelength_um)
        n_out = material_normal_side.refractive_index(wavelength_um)
    cos_i = -float(np.dot(direction, normal))
    eta = n_in / n_out
    sin2_t = eta * eta * max(0.0, 1.0 - cos_i * cos_i)
    if sin2_t > 1.0 + 1.0e-11:
        return {
            "kind": "total_internal_reflection",
            "normal": normal,
            "n_in": n_in,
            "n_out": n_out,
        }
    sin2_t = min(1.0, sin2_t)
    cos_t = math.sqrt(1.0 - sin2_t)
    transmitted = eta * direction + (eta * cos_i - cos_t) * normal
    return {
        "kind": "refract",
        "normal": normal,
        "n_in": n_in,
        "n_out": n_out,
        "transmitted_direction": normalized(transmitted),
    }


@dataclass
class Mirror(Surface):
    z: float = 0.0
    curvature_radius: float | None = None

    def intersect(self, origin, direction, t_min) -> Hit | None:
        if self.curvature_radius is None:
            return intersect_plane(self, origin, direction, t_min, self.z)
        return intersect_sphere(
            self, origin, direction, t_min, self.z, self.curvature_radius
        )

    def interaction(self, direction, hit, wavelength_um) -> dict[str, Any]:
        normal = oriented_normal(direction, hit.geometric_normal)
        reflected = direction - 2.0 * float(np.dot(direction, normal)) * normal
        return {
            "kind": "reflect",
            "normal": normal,
            "reflected_direction": normalized(reflected),
        }


@dataclass
class ImagePlane(Surface):
    z: float = 0.0

    def intersect(self, origin, direction, t_min) -> Hit | None:
        return intersect_plane(self, origin, direction, t_min, self.z)

    def interaction(self, direction, hit, wavelength_um) -> dict[str, Any]:
        return {"kind": "image", "normal": oriented_normal(direction, hit.geometric_normal)}


@dataclass
class PlaneBoundary(Surface):
    """Absorbing finite plane, optionally with a circular clear hole."""

    z: float = 0.0
    clear_radius: float | None = None

    def intersect(self, origin, direction, t_min) -> Hit | None:
        if abs(float(direction[2])) <= 1.0e-12:
            return None
        t = (self.z - float(origin[2])) / float(direction[2])
        if t <= t_min:
            return None
        point = origin + direction * t
        if not within_aperture(point, self.aperture_radius):
            return None
        if self.clear_radius is not None and float(np.hypot(point[0], point[1])) < self.clear_radius:
            return None
        return Hit(self, t, point, np.array([0.0, 0.0, 1.0]))

    def interaction(self, direction, hit, wavelength_um) -> dict[str, Any]:
        return {"kind": "absorb", "normal": oriented_normal(direction, hit.geometric_normal)}
