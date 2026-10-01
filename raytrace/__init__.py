"""Minimal geometric ray-tracing core for educational optical models."""

from .materials import Material, MaterialLibrary, UnknownWavelengthError
from .project import Project, TraceBatch, load_batch, load_project
from .surfaces import (
    CircularAperture,
    PlaneSurface,
    SphericalSurface,
    Surface,
)
from .tracer import Beam, Ray, RayResult, RayTracer, TraceSettings

__all__ = [
    "CircularAperture",
    "Beam",
    "Material",
    "MaterialLibrary",
    "PlaneSurface",
    "Project",
    "Ray",
    "RayResult",
    "RayTracer",
    "SphericalSurface",
    "Surface",
    "TraceBatch",
    "TraceSettings",
    "UnknownWavelengthError",
    "load_batch",
    "load_project",
]
