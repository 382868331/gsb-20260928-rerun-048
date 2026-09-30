"""Serializable projects and trace batches."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import warnings

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
        "transmissivity": data.get("transmissivity"),
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
    # Current convention: ``reflectivity`` is always the reflected power
    # fraction.  Projects saved before this convention stored the forward
    # transmission fraction of a beamsplitter in ``reflectivity``; such files
    # have no ``convention`` marker (or the explicit legacy marker below) and
    # are migrated audibly instead of being reinterpreted silently.
    convention: str = "physical-reflected-fraction-v1"
    migration_notes: list[str] = field(default_factory=list)

    LEGACY_CONVENTION = "legacy-beamsplitter-reflectivity-is-transmission"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "convention": self.convention,
            "migration_notes": self.migration_notes,
            "settings": vars(self.settings),
            "materials": self.materials.to_dict(),
            "surfaces": [_surface_to_dict(surface) for surface in self.surfaces],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Project":
        project, _ = cls._from_dict_with_migration(data)
        return project

    @classmethod
    def _from_dict_with_migration(cls, data: dict[str, Any]) -> tuple["Project", list[str]]:
        surface_data = [dict(item) for item in data["surfaces"]]
        stored_convention = data.get("convention")
        legacy = stored_convention in (None, cls.LEGACY_CONVENTION)
        new_notes: list[str] = []
        if legacy:
            for item in surface_data:
                if item.get("kind") != "beamsplitter" or item.get("transmissivity") is not None:
                    continue
                old_value = float(item.get("reflectivity", 1.0))
                item["transmissivity"] = old_value
                item["reflectivity"] = 1.0 - old_value
                new_notes.append(
                    f"Surface {item['identifier']!r} used the legacy beamsplitter convention: "
                    f"reflectivity={old_value:g} actually stored the forward transmission fraction. "
                    f"Migrated to transmissivity={old_value:g}, reflectivity={1.0 - old_value:g} "
                    "so the original physical result is preserved."
                )
        project = cls(
            name=data["name"],
            description=data.get("description", ""),
            settings=TraceSettings(**data.get("settings", {})),
            materials=MaterialLibrary.from_dict(data["materials"]),
            surfaces=[_surface_from_dict(item) for item in surface_data],
            convention="physical-reflected-fraction-v1",
            migration_notes=new_notes + list(data.get("migration_notes", [])),
        )
        return project, new_notes

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        project, new_notes = cls._from_dict_with_migration(json.loads(Path(path).read_text(encoding="utf-8")))
        for note in new_notes:
            warnings.warn(f"{path}: {note}", stacklevel=2)
        return project

    def trace_rays(self, rays: list[Ray]) -> "TraceBatch":
        tracer = RayTracer(self.surfaces, self.materials, self.settings)
        return TraceBatch(project_name=self.name, rays=rays, results=tracer.trace_branches(rays))


def _ray_to_dict(ray: Ray) -> dict[str, Any]:
    return {
        "origin": ray.origin,
        "direction": ray.direction,
        "wavelength_nm": ray.wavelength_nm,
        "weight": ray.weight,
        "label": ray.label,
    }


def _ray_from_dict(data: dict[str, Any]) -> Ray:
    return Ray(
        tuple(data["origin"]),
        tuple(data["direction"]),
        data.get("wavelength_nm"),
        data.get("weight", 1.0),
        data.get("label"),
    )


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

    def detector_hits(self) -> dict[str, list[dict[str, Any]]]:
        """Terminal detector hits grouped by detector identifier."""

        hits: dict[str, list[dict[str, Any]]] = {}
        for result in self.results:
            if result.status != "reached_image" or result.detector is None:
                continue
            hits.setdefault(result.detector, []).append({
                "source_label": result.source_label,
                "branch_id": result.branch_id,
                "parent_branch_id": result.parent_branch_id,
                "depth": result.depth,
                "path": [event["action"] for event in result.events],
                "point": result.endpoint,
                "power": result.weight,
            })
        return hits

    def hits_by_source(self) -> dict[str, list[dict[str, Any]]]:
        """Detector power and landing positions grouped by source ray."""

        grouped: dict[str, list[dict[str, Any]]] = {}
        for detector, hits in self.detector_hits().items():
            for hit in hits:
                grouped.setdefault(hit["source_label"], []).append({**hit, "detector": detector})
        return grouped

    def detector_summary(self) -> dict[str, dict[str, Any]]:
        summaries: dict[str, dict[str, Any]] = {}
        for detector, hits in self.detector_hits().items():
            total_power = sum(hit["power"] for hit in hits)
            by_source: dict[str, float] = {}
            for hit in hits:
                by_source[hit["source_label"]] = by_source.get(hit["source_label"], 0.0) + hit["power"]
            summaries[detector] = {
                "power": total_power,
                "power_by_source": by_source,
                "count": len(hits),
                "branches": sorted(hits, key=lambda hit: (hit["source_label"], hit["branch_id"])),
                "mean_point": tuple(
                    sum(hit["point"][axis] * hit["power"] for hit in hits) / total_power
                    if total_power > 0
                    else sum(hit["point"][axis] for hit in hits) / len(hits)
                    for axis in range(3)
                ),
            }
        return summaries

    def branch_energy_flow(self) -> list[dict[str, Any]]:
        """Traceable per-branch energy flow for every terminal sub-beam.

        Each entry preserves the source ray, full branch id and ancestry,
        every reflection/transmission/loss event with its surface and
        powers, and the terminal state.  Hits from different branches keep
        separate entries even when they land on the same detector region.
        """

        flows: list[dict[str, Any]] = []
        for result in self.results:
            steps = []
            for event in result.events:
                step: dict[str, Any] = {
                    "event_uid": event["event_uid"],
                    "surface": event["surface"],
                    "action": event["action"],
                    "point": event["point"],
                    "incident_power": event.get("incident_weight"),
                    "reflected_power": event.get("reflected_power", 0.0),
                    "transmitted_power": event.get("transmitted_power", 0.0),
                    "absorbed_power": event.get("absorbed_power", 0.0),
                }
                if "reflectivity" in event:
                    step["reflectivity"] = event["reflectivity"]
                    step["transmissivity"] = event["transmissivity"]
                steps.append(step)
            flows.append({
                "source_label": result.source_label,
                "branch_id": result.branch_id,
                "parent_branch_id": result.parent_branch_id,
                "depth": result.depth,
                "status": result.status,
                "detector": result.detector,
                "endpoint": result.endpoint,
                "terminal_power": result.weight,
                "steps": steps,
            })
        return sorted(flows, key=lambda item: (item["source_label"], item["branch_id"]))

    def _absorbed_by_source(self) -> dict[str, float]:
        absorbed: dict[str, float] = {}
        seen: set[tuple[str, int]] = set()
        for result in self.results:
            for event in result.events:
                key = (result.source_label, event["event_uid"])
                if key in seen:
                    continue
                seen.add(key)
                absorbed[result.source_label] = absorbed.get(result.source_label, 0.0) + event.get("absorbed_power", 0.0)
        return absorbed

    def energy_ledger(self, input_power: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
        """Power account per source.

        ``detected`` reached a detector, ``absorbed`` was lost on surfaces,
        ``grazing`` remains on beams stopped at an aperture edge and
        ``unreached``/``stopped`` are beams that escaped or hit the iteration
        limits.  Stopped residuals are never reported as absorbed.
        """

        absorbed = self._absorbed_by_source()
        accounts: dict[str, dict[str, Any]] = {}
        for result in self.results:
            account = accounts.setdefault(result.source_label, {
                "input": 0.0,
                "detected": 0.0,
                "absorbed": 0.0,
                "grazing": 0.0,
                "unreached": 0.0,
                "stopped": 0.0,
            })
            bucket = {
                "reached_image": "detected",
                "grazing": "grazing",
                "no_image": "unreached",
                "max_events": "stopped",
                "max_branches": "stopped",
                # A leaf terminated by a fully absorbing surface: its power
                # is already booked as surface absorption below.
                "absorbed": None,
            }.get(result.status, "stopped")
            if bucket is not None:
                account[bucket] += result.weight

        for source, account in accounts.items():
            account["absorbed"] = absorbed.get(source, 0.0)
            account["input"] = (input_power or {}).get(
                source,
                account["detected"] + account["absorbed"] + account["grazing"] + account["unreached"] + account["stopped"],
            )
            closed = account["detected"] + account["absorbed"] + account["grazing"] + account["unreached"] + account["stopped"]
            account["unaccounted"] = account["input"] - closed
        return accounts

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
                source_label=item.get("source_label", "root"),
                branch_id=item.get("branch_id", "root"),
                parent_branch_id=item.get("parent_branch_id"),
                detector=item.get("detector"),
                depth=item.get("depth", len(item["events"])),
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
