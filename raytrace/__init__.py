"""Minimal geometric ray-tracing core for educational optical models."""

from .materials import Material, MaterialLibrary
from .project import Project, TraceBatch, load_batch, load_project
from .surfaces import (
    CircularAperture,
    PlaneSurface,
    SphericalSurface,
    Surface,
)
from .tracer import BeamResult, Ray, RayTracer, TraceSettings

__all__ = [
    "BeamResult",
    "CircularAperture",
    "Material",
    "MaterialLibrary",
    "PlaneSurface",
    "Project",
    "Ray",
    "RayTracer",
    "SphericalSurface",
    "Surface",
    "TraceBatch",
    "TraceSettings",
    "load_batch",
    "load_project",
]
