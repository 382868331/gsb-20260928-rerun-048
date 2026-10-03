"""Surface shapes, apertures, mirrors and refractive interfaces."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional

Vector = tuple[float, float, float]


def dot(a: Vector, b: Vector) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector, b: Vector) -> Vector:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


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
    transmissivity: Optional[float] = None
    polarization_reflectivity: Optional[Mapping[str, float]] = None
    polarization_transmissivity: Optional[Mapping[str, float]] = None
    analyzer_angle_deg: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in {"refract", "mirror", "detector", "beamsplitter", "aperture_stop"}:
            raise ValueError(f"unsupported surface kind {self.kind!r}")
        if self.shape not in {"plane", "sphere"}:
            raise ValueError(f"unsupported surface shape {self.shape!r}")
        if self.shape == "sphere" and self.radius is None:
            raise ValueError("a spherical surface requires a radius")
        if not 0.0 <= self.reflectivity <= 1.0:
            raise ValueError("reflectivity must be between 0 and 1")
        if self.transmissivity is not None and not 0.0 <= self.transmissivity <= 1.0:
            raise ValueError("transmissivity must be between 0 and 1")
        if self.kind == "beamsplitter":
            transmission = 1.0 - self.reflectivity if self.transmissivity is None else self.transmissivity
            if self.reflectivity + transmission > 1.0 + 1e-12:
                raise ValueError("beamsplitter reflectivity + transmissivity must not exceed 1")
        for table_name, table in (
            ("polarization_reflectivity", self.polarization_reflectivity),
            ("polarization_transmissivity", self.polarization_transmissivity),
        ):
            if table is None:
                continue
            if self.kind != "beamsplitter":
                raise ValueError(f"{table_name} is only meaningful on a beamsplitter")
            for axis, value in table.items():
                if axis not in ("s", "p"):
                    raise ValueError(f"{table_name} keys must be 's' or 'p', got {axis!r}")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{table_name} values must be between 0 and 1")
        if self.polarization_reflectivity or self.polarization_transmissivity:
            axes = set(self.polarization_reflectivity or {}) & set(self.polarization_transmissivity or {})
            for axis in axes:
                total = self.polarization_reflectivity[axis] + self.polarization_transmissivity[axis]
                if total > 1.0 + 1e-12:
                    raise ValueError(f"polarized reflectivity + transmissivity for {axis!r} must not exceed 1")
        if self.analyzer_angle_deg is not None and self.kind != "detector":
            raise ValueError("an analyzer angle is only meaningful on a detector")
        object.__setattr__(self, "axis", normalize(self.axis))

    def polarized_split(self, axis: str) -> Optional[tuple[float, float]]:
        """(reflectivity, transmissivity) for one polarization axis.

        Returns None when the surface declares a polarization table but has
        no verified entry for that axis; the caller must treat that as an
        unknown response rather than falling back to the scalar values.
        """

        if self.polarization_reflectivity is None and self.polarization_transmissivity is None:
            return self.reflectivity, self.transmission_fraction()
        reflectivity = (self.polarization_reflectivity or {}).get(axis)
        transmissivity = (self.polarization_transmissivity or {}).get(axis)
        if reflectivity is None and transmissivity is None:
            return None
        if reflectivity is None:
            reflectivity = 1.0 - transmissivity
        if transmissivity is None:
            transmissivity = 1.0 - reflectivity
        return reflectivity, transmissivity

    def transmission_fraction(self) -> float:
        """Forward-transmitted power fraction; loss is ``1 - R - T``."""

        if self.kind != "beamsplitter":
            return 1.0 - self.reflectivity
        if self.transmissivity is None:
            return 1.0 - self.reflectivity
        return self.transmissivity

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
