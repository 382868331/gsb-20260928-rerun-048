"""Serializable projects and trace batches."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .materials import MaterialLibrary
from .surfaces import CircularAperture, SphericalSurface, PlaneSurface, Surface
from .tracer import Ray, RayResult, RayTracer, TraceSettings


def _surface_to_dict(surface: Surface) -> dict[str, Any]:
    data = vars(surface).copy()
    data["aperture"] = vars(surface.aperture) if surface.aperture else None
    return data


def _surface_from_dict(data: dict[str, Any]) -> Surface:
    aperture_data = data.get("aperture")
    aperture = CircularAperture(**aperture_data) if aperture_data else None
    common = {
        "identifier": data["identifier"],
        "kind": data["kind"],
        "vertex": tuple(data["vertex"]),
        "axis": tuple(data["axis"]),
        "aperture": aperture,
        "front_material": data.get("front_material", "air"),
        "back_material": data.get("back_material", "air"),
        "reflectivity": data.get("reflectivity", 1.0),
    }
    if data["shape"] == "plane":
        return PlaneSurface(**common)
    return SphericalSurface(radius=data["radius"], **common)


@dataclass
class Project:
    name: str
    surfaces: list[Surface]
    materials: MaterialLibrary = field(default_factory=MaterialLibrary)
    settings: TraceSettings = field(default_factory=TraceSettings)
    description: str = "Scalar-index, geometric-optics educational model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "settings": vars(self.settings),
            "materials": self.materials.to_dict(),
            "surfaces": [_surface_to_dict(surface) for surface in self.surfaces],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            settings=TraceSettings(**data.get("settings", {})),
            materials=MaterialLibrary.from_dict(data["materials"]),
            surfaces=[_surface_from_dict(item) for item in data["surfaces"]],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def trace_rays(self, rays: list[Ray]) -> "TraceBatch":
        tracer = RayTracer(self.surfaces, self.materials, self.settings)
        return TraceBatch(project_name=self.name, rays=rays, results=[tracer.trace(ray) for ray in rays])


def _ray_to_dict(ray: Ray) -> dict[str, Any]:
    return {"origin": ray.origin, "direction": ray.direction, "wavelength_nm": ray.wavelength_nm}


def _ray_from_dict(data: dict[str, Any]) -> Ray:
    return Ray(tuple(data["origin"]), tuple(data["direction"]), data.get("wavelength_nm"))


@dataclass
class TraceBatch:
    project_name: str
    rays: list[Ray]
    results: list[RayResult]

    def status_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for result in self.results:
            summary[result.status] = summary.get(result.status, 0) + 1
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "rays": [_ray_to_dict(ray) for ray in self.rays],
            "results": [
                {
                    **vars(result),
                    "ray": _ray_to_dict(result.ray),
                }
                for result in self.results
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceBatch":
        rays = [_ray_from_dict(item) for item in data["rays"]]
        results = [
            RayResult(
                ray=_ray_from_dict(item["ray"]),
                status=item["status"],
                endpoint=tuple(item["endpoint"]),
                weight=item["weight"],
                events=item["events"],
            )
            for item in data["results"]
        ]
        return cls(data["project_name"], rays, results)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_project(path: str | Path) -> Project:
    return Project.load(path)


def load_batch(path: str | Path) -> TraceBatch:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TraceBatch.from_dict(data)
