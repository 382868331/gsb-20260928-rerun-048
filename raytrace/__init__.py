"""Minimal geometric ray-tracing core for educational optical models."""

from .materials import Material, MaterialLibrary, UnknownMaterialResponse
from .measurements import AffineQuantity, MeasurementBatch, ResponseMeasurement
from .project import Project, TraceBatch, load_batch, load_project
from .surfaces import (
    CircularAperture,
    PlaneSurface,
    SphericalSurface,
    Surface,
)
from .tracer import Ray, RayResult, RayTracer, TraceSettings

__all__ = [
    "AffineQuantity",
    "CircularAperture",
    "Material",
    "MaterialLibrary",
    "MeasurementBatch",
    "PlaneSurface",
    "Project",
    "Ray",
    "RayResult",
    "RayTracer",
    "ResponseMeasurement",
    "SphericalSurface",
    "Surface",
    "TraceBatch",
    "TraceSettings",
    "UnknownMaterialResponse",
    "load_batch",
    "load_project",
]
