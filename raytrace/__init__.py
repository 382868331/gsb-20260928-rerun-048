"""Minimal geometric ray-tracing core for educational optical models."""

from .materials import (
    Material,
    MaterialLibrary,
    UnknownMaterialResponse,
    UnknownPolarizationResponse,
)
from .project import Project, TraceBatch, load_batch, load_project
from .surfaces import (
    CircularAperture,
    PlaneSurface,
    SphericalSurface,
    Surface,
)
from .tracer import Ray, RayResult, RayTracer, TraceSettings

__all__ = [
    "CircularAperture",
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
    "UnknownMaterialResponse",
    "UnknownPolarizationResponse",
    "load_batch",
    "load_project",
]
