"""Serializable projects and trace batches."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .materials import MaterialLibrary
from .surfaces import CircularAperture, SphericalSurface, PlaneSurface, Surface
from .tracer import BeamResult, Ray, RayResult, RayTracer, TraceSettings


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
        beams = [tracer.trace_beams(ray) for ray in rays]
        results = [
            next(beam for beam in ray_beams if beam.parent_id is None).to_ray_result()
            for ray_beams in beams
        ]
        return TraceBatch(project_name=self.name, rays=rays, results=results, beams=beams)


def _ray_to_dict(ray: Ray) -> dict[str, Any]:
    return {"origin": ray.origin, "direction": ray.direction, "wavelength_nm": ray.wavelength_nm}


def _ray_from_dict(data: dict[str, Any]) -> Ray:
    return Ray(tuple(data["origin"]), tuple(data["direction"]), data.get("wavelength_nm"))


def _beam_to_dict(beam: BeamResult) -> dict[str, Any]:
    return {
        "beam_id": beam.beam_id,
        "parent_id": beam.parent_id,
        "branch": beam.branch,
        "generation": beam.generation,
        "ray": _ray_to_dict(beam.ray),
        "status": beam.status,
        "endpoint": beam.endpoint,
        "weight": beam.weight,
        "events": beam.events,
    }


def _beam_from_dict(data: dict[str, Any]) -> BeamResult:
    return BeamResult(
        beam_id=data["beam_id"],
        parent_id=data["parent_id"],
        branch=data["branch"],
        generation=data["generation"],
        ray=_ray_from_dict(data["ray"]),
        status=data["status"],
        endpoint=tuple(data["endpoint"]),
        weight=data["weight"],
        events=data["events"],
    )


@dataclass
class TraceBatch:
    project_name: str
    rays: list[Ray]
    results: list[RayResult]
    beams: list[list[BeamResult]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.beams:
            self.beams = [
                [BeamResult(0, None, "root", 0, result.ray, result.status,
                            result.endpoint, result.weight, result.events)]
                for result in self.results
            ]

    def status_summary(self) -> dict[str, int]:
        summary: dict[str, int] = {}
        for ray_beams in self.beams:
            for beam in ray_beams:
                summary[beam.status] = summary.get(beam.status, 0) + 1
        return summary

    def energy_report(self) -> dict[str, Any]:
        """Bookkeep every watt: detected, absorbed, residual, escaped, grazing.

        Residual energy belongs to beams stopped by explicit termination
        conditions (min_weight / max_events / max_generations) and is kept
        separate from absorbed energy.
        """
        detectors: dict[str, dict[str, Any]] = {}
        absorbed = 0.0
        residual = 0.0
        escaped = 0.0
        grazing = 0.0
        for source_index, ray_beams in enumerate(self.beams):
            for beam in ray_beams:
                for event in beam.events:
                    absorbed += event.get("absorbed_power", 0.0)
                if beam.status == "reached_image":
                    surface_id = beam.events[-1]["surface"]
                    entry = detectors.setdefault(surface_id, {
                        "energy": 0.0, "hits": 0, "points": [], "by_source": {},
                    })
                    entry["energy"] += beam.weight
                    entry["hits"] += 1
                    entry["points"].append(beam.endpoint)
                    key = str(source_index)
                    entry["by_source"][key] = entry["by_source"].get(key, 0.0) + beam.weight
                elif beam.status == "no_image":
                    escaped += beam.weight
                elif beam.status == "grazing":
                    grazing += beam.weight
                else:  # min_weight, max_events, max_generations
                    residual += beam.weight
        detected = sum(entry["energy"] for entry in detectors.values())
        input_energy = float(len(self.rays))
        accounted = detected + absorbed + residual + escaped + grazing
        return {
            "input_energy": input_energy,
            "detected_energy": detected,
            "detectors": detectors,
            "absorbed_energy": absorbed,
            "residual_energy": residual,
            "escaped_energy": escaped,
            "grazing_energy": grazing,
            "unaccounted_energy": input_energy - accounted,
        }

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
            "beams": [[_beam_to_dict(beam) for beam in ray_beams] for ray_beams in self.beams],
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
        beams = [[_beam_from_dict(item) for item in ray_beams]
                 for ray_beams in data.get("beams", [])]
        return cls(data["project_name"], rays, results, beams)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_project(path: str | Path) -> Project:
    return Project.load(path)


def load_batch(path: str | Path) -> TraceBatch:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TraceBatch.from_dict(data)
