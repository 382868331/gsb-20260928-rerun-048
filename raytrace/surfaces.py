"""Surface shapes, apertures, mirrors and refractive interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

Vector = tuple[float, float, float]


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vector) -> float:
    return math.sqrt(dot(a, a))


def normalize(a: Vector) -> Vector:
    length = norm(a)
    if length == 0:
        raise ValueError("cannot normalize a zero vector")
    return (a[0] / length, a[1] / length, a[2] / length)


def add(a: Vector, b: Vector) -> Vector:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: Vector, value: float) -> Vector:
    return (a[0] * value, a[1] * value, a[2] * value)


@dataclass(frozen=True)
class CircularAperture:
    radius: float
    edge_tolerance: float = 1e-8

    def contains(self, point: Vector, vertex: Vector, axis: Vector) -> bool:
        delta = (point[0] - vertex[0], point[1] - vertex[1], point[2] - vertex[2])
        axial = dot(delta, axis)
        radial_vec = add(delta, scale(axis, -axial))
        radial = norm(radial_vec)
        return radial <= self.radius + self.edge_tolerance

    def on_edge(self, point: Vector, vertex: Vector, axis: Vector) -> bool:
        delta = (point[0] - vertex[0], point[1] - vertex[1], point[2] - vertex[2])
        axial = dot(delta, axis)
        radial = norm(add(delta, scale(axis, -axial)))
        return abs(radial - self.radius) <= self.edge_tolerance


@dataclass(frozen=True)
class Surface:
    identifier: str
    kind: str  # "refract", "mirror", "detector", "beamsplitter", or "aperture_stop"
    shape: str  # "plane" or "sphere"
    vertex: Vector
    axis: Vector
    aperture: Optional[CircularAperture] = None
    radius: Optional[float] = None
    front_material: str = "air"
    back_material: str = "air"
    reflectivity: float = 1.0
    reflectivity_convention: str = "reflection"
    legacy_reflectivity: Optional[float] = None
    legacy_reflectivity_convention: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in {"refract", "mirror", "detector", "beamsplitter", "aperture_stop"}:
            raise ValueError(f"unsupported surface kind {self.kind!r}")
        if self.shape not in {"plane", "sphere"}:
            raise ValueError(f"unsupported surface shape {self.shape!r}")
        if self.shape == "sphere" and self.radius is None:
            raise ValueError("a spherical surface requires a radius")
        if not 0.0 <= self.reflectivity <= 1.0:
            raise ValueError("reflectivity must be between 0 and 1")
        if self.reflectivity_convention != "reflection":
            raise ValueError("reflectivity must always use the physical reflection convention")
        if self.legacy_reflectivity is not None and not 0.0 <= self.legacy_reflectivity <= 1.0:
            raise ValueError("legacy reflectivity must be between 0 and 1")
        object.__setattr__(self, "axis", normalize(self.axis))

    def intersect(self, origin: Vector, direction: Vector) -> Optional[tuple[float, Vector, Vector]]:
        if self.shape == "plane":
            return self._intersect_plane(origin, direction)
        return self._intersect_sphere(origin, direction)

    def _intersect_plane(self, origin: Vector, direction: Vector) -> Optional[tuple[float, Vector, Vector]]:
        denominator = dot(direction, self.axis)
        if abs(denominator) < 1e-14:
            return None
        t = dot(add(self.vertex, scale(origin, -1.0)), self.axis) / denominator
        if t <= 0:
            return None
        point = add(origin, scale(direction, t))
        return t, point, self.axis

    def _intersect_sphere(self, origin: Vector, direction: Vector) -> Optional[tuple[float, Vector, Vector]]:
        # The radius uses the optical sign convention: positive places the
        # center after the vertex (+axis), negative before it.
        center = add(self.vertex, scale(self.axis, float(self.radius)))
        offset = add(origin, scale(center, -1.0))
        b = dot(offset, direction)
        c = dot(offset, offset) - float(self.radius) ** 2
        discriminant = b * b - c
        if discriminant < -1e-12:
            return None
        root = math.sqrt(max(0.0, discriminant))
        for t in (-b - root, -b + root):
            if t <= 0:
                continue
            point = add(origin, scale(direction, t))
            surface_normal = normalize(scale(add(point, scale(center, -1.0)), 1.0 / abs(float(self.radius))))
            expected_sign = 1.0 if self.radius < 0 else -1.0
            if dot(surface_normal, self.axis) * expected_sign < 0.0:
                continue  # Reject the opposite hemisphere / far spherical cap.
            return t, point, surface_normal
        return None

    def accepts(self, point: Vector) -> bool:
        return self.aperture is None or self.aperture.contains(point, self.vertex, self.axis)

    def at_aperture_edge(self, point: Vector) -> bool:
        return self.aperture is not None and self.aperture.on_edge(point, self.vertex, self.axis)


def PlaneSurface(identifier: str, vertex: Vector, axis: Vector, **kwargs) -> Surface:
    return Surface(identifier, shape="plane", vertex=vertex, axis=axis, **kwargs)


def SphericalSurface(identifier: str, vertex: Vector, axis: Vector, radius: float, **kwargs) -> Surface:
    return Surface(identifier, shape="sphere", vertex=vertex, axis=axis, radius=radius, **kwargs)
