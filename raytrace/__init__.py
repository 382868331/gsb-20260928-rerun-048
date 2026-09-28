"""A small, explicit geometric ray-tracing model."""

from .materials import Air, ConstantMaterial, Material, SellmeierMaterial
from .rays import Ray, RayRecord
from .surfaces import (
    ImagePlane,
    Mirror,
    PlaneBoundary,
    PlaneInterface,
    SphericalInterface,
    Surface,
)
from .system import Engineering, OpticalSystem
from .tracer import TraceConfig, load_records_json, trace_ray, trace_batch

__all__ = [
    "Air",
    "ConstantMaterial",
    "Material",
    "SellmeierMaterial",
    "Ray",
    "RayRecord",
    "ImagePlane",
    "Mirror",
    "PlaneBoundary",
    "PlaneInterface",
    "SphericalInterface",
    "Surface",
    "Engineering",
    "OpticalSystem",
    "TraceConfig",
    "trace_ray",
    "trace_batch",
    "load_records_json",
]
