"""Serializable projects and trace batches."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import warnings

from .materials import MaterialLibrary
from .surfaces import CircularAperture, SphericalSurface, PlaneSurface, Surface
from .tracer import Beam, Ray, RayResult, RayTracer, TraceSettings


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
        "reflectivity_by_wavelength": {
            float(wavelength): value
            for wavelength, value in (data.get("reflectivity_by_wavelength") or {}).items()
        } or None,
        "transmissivity_by_wavelength": {
            float(wavelength): value
            for wavelength, value in (data.get("transmissivity_by_wavelength") or {}).items()
        } or None,
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
    description: str = "Wavelength-aware, geometric-optics educational model"
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

    def trace_rays(self, rays: list[Ray] | list[Ray | Beam]) -> "TraceBatch":
        tracer = RayTracer(self.surfaces, self.materials, self.settings)
        expanded = _expand_sources(rays)
        return TraceBatch(project_name=self.name, rays=expanded, results=tracer.trace_branches(rays))


def _expand_sources(sources) -> list[Ray]:
    rays: list[Ray] = []
    for source in sources:
        if isinstance(source, Beam):
            rays.extend(source.rays())
        else:
            rays.append(source)
    return rays


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


def _wavelength_label(wavelength_nm: float | None) -> str:
    return "broadband" if wavelength_nm is None else f"{float(wavelength_nm):g}"


def _wavelength_sort(wavelength_nm: float | None) -> float:
    return float("inf") if wavelength_nm is None else float(wavelength_nm)


def _wavelength_from_label(label: str) -> float | None:
    return None if label == "broadband" else float(label)


def _input_lookup(input_power: dict[str, Any] | None, source: str) -> Any:
    return None if input_power is None else input_power.get(source)


def _wavelength_input(provided: Any, label: str, wavelength_nm: float | None, closed: float) -> float:
    if isinstance(provided, dict):
        if wavelength_nm is not None and wavelength_nm in provided:
            return float(provided[wavelength_nm])
        if label in provided:
            return float(provided[label])
        return closed
    return closed


def _source_input(provided: Any, default: float) -> float:
    if isinstance(provided, dict):
        return sum(float(value) for value in provided.values())
    return default if provided is None else float(provided)


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

    def status_summary_by_wavelength(self) -> dict[str, dict[str, int]]:
        """Terminal-reason counts kept separate for each wavelength."""

        summaries: dict[str, dict[str, Any]] = {}
        for result in self.results:
            key = _wavelength_label(result.wavelength_nm)
            summary = summaries.setdefault(key, {"wavelength_nm": result.wavelength_nm})
            summary[result.status] = summary.get(result.status, 0) + 1
        return summaries

    def detector_hits(self) -> dict[str, list[dict[str, Any]]]:
        """Terminal detector hits grouped by detector identifier."""

        hits: dict[str, list[dict[str, Any]]] = {}
        for result in self.results:
            if result.status != "reached_image" or result.detector is None:
                continue
            hits.setdefault(result.detector, []).append({
                "source_label": result.source_label,
                "wavelength_nm": result.wavelength_nm,
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
            by_wavelength: dict[str, float] = {}
            by_source_wavelength: dict[str, dict[str, float]] = {}
            branches_by_wavelength: dict[str, list[dict[str, Any]]] = {}
            for hit in hits:
                by_source[hit["source_label"]] = by_source.get(hit["source_label"], 0.0) + hit["power"]
                wavelength_key = _wavelength_label(hit["wavelength_nm"])
                by_wavelength[wavelength_key] = by_wavelength.get(wavelength_key, 0.0) + hit["power"]
                source_bucket = by_source_wavelength.setdefault(hit["source_label"], {})
                source_bucket[wavelength_key] = source_bucket.get(wavelength_key, 0.0) + hit["power"]
                branches_by_wavelength.setdefault(wavelength_key, []).append(hit)
            summaries[detector] = {
                "power": total_power,
                "power_by_source": by_source,
                "power_by_wavelength": by_wavelength,
                "power_by_source_wavelength": by_source_wavelength,
                "branches_by_wavelength": branches_by_wavelength,
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
                    "wavelength_nm": event.get("wavelength_nm"),
                    "surface": event["surface"],
                    "action": event["action"],
                    "point": event["point"],
                    "incident_power": event.get("incident_weight"),
                    "reflected_power": event.get("reflected_power", 0.0),
                    "transmitted_power": event.get("transmitted_power", 0.0),
                    "absorbed_power": event.get("absorbed_power", 0.0),
                    "surface_absorbed_power": event.get("surface_absorbed_power", event.get("absorbed_power", 0.0)),
                    "bulk_absorbed_power": event.get("bulk_absorbed_power", 0.0),
                }
                if "reflectivity" in event:
                    step["reflectivity"] = event["reflectivity"]
                    step["transmissivity"] = event["transmissivity"]
                steps.append(step)
            flows.append({
                "source_label": result.source_label,
                "wavelength_nm": result.wavelength_nm,
                "branch_id": result.branch_id,
                "parent_branch_id": result.parent_branch_id,
                "depth": result.depth,
                "status": result.status,
                "detector": result.detector,
                "endpoint": result.endpoint,
                "terminal_power": result.weight,
                "steps": steps,
            })
        return sorted(
            flows,
            key=lambda item: (
                item["source_label"],
                _wavelength_sort(item["wavelength_nm"]),
                item["branch_id"],
            ),
        )

    def _losses_by_source(self) -> dict[str, dict[str, dict[str, float]]]:
        losses: dict[str, dict[str, dict[str, float]]] = {}
        seen: set[tuple[str, int]] = set()
        for result in self.results:
            for event in result.events:
                key = (result.source_label, event["event_uid"])
                if key in seen:
                    continue
                seen.add(key)
                wavelength_key = _wavelength_label(result.wavelength_nm)
                bucket = losses.setdefault(result.source_label, {}).setdefault(
                    wavelength_key, {"surface": 0.0, "bulk": 0.0, "absorbed": 0.0}
                )
                surface_loss = (
                    event.get("surface_absorbed_power")
                    if "surface_absorbed_power" in event
                    else event.get("absorbed_power", 0.0) - event.get("bulk_absorbed_power", 0.0)
                )
                bulk_loss = event.get("bulk_absorbed_power", 0.0)
                bucket["surface"] += surface_loss
                bucket["bulk"] += bulk_loss
                bucket["absorbed"] += event.get("absorbed_power", 0.0)
        return losses

    def energy_ledger(self, input_power: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
        """Power account per source.

        ``detected`` reached a detector, ``absorbed`` was lost on surfaces,
        ``grazing`` remains on beams stopped at an aperture edge and
        ``unreached``/``stopped`` are beams that escaped or hit the iteration
        limits.  Stopped residuals are never reported as absorbed.
        """

        losses = self._losses_by_source()
        accounts: dict[str, dict[str, Any]] = {}
        for result in self.results:
            account = accounts.setdefault(result.source_label, {
                "input": 0.0,
                "detected": 0.0,
                "absorbed": 0.0,
                "grazing": 0.0,
                "unreached": 0.0,
                "stopped": 0.0,
                "surface_absorbed": 0.0,
                "bulk_absorbed": 0.0,
                "by_wavelength": {},
            })
            wavelength_key = _wavelength_label(result.wavelength_nm)
            spectral = account["by_wavelength"].setdefault(wavelength_key, {
                "wavelength_nm": result.wavelength_nm,
                "input": 0.0,
                "detected": 0.0,
                "absorbed": 0.0,
                "surface_absorbed": 0.0,
                "bulk_absorbed": 0.0,
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
                spectral[bucket] += result.weight

        for source, account in accounts.items():
            for wavelength_key, loss_bucket in losses.get(source, {}).items():
                spectral = account["by_wavelength"].setdefault(wavelength_key, {
                    "wavelength_nm": _wavelength_from_label(wavelength_key),
                    "input": 0.0,
                    "detected": 0.0,
                    "absorbed": 0.0,
                    "surface_absorbed": 0.0,
                    "bulk_absorbed": 0.0,
                    "grazing": 0.0,
                    "unreached": 0.0,
                    "stopped": 0.0,
                })
                spectral["surface_absorbed"] += loss_bucket["surface"]
                spectral["bulk_absorbed"] += loss_bucket["bulk"]
                spectral["absorbed"] += loss_bucket["absorbed"]
            provided = _input_lookup(input_power, source)
            for wavelength_key, spectral in account["by_wavelength"].items():
                closed_spectral = (
                    spectral["detected"] + spectral["absorbed"] + spectral["grazing"]
                    + spectral["unreached"] + spectral["stopped"]
                )
                spectral_input = _wavelength_input(provided, wavelength_key, spectral["wavelength_nm"], closed_spectral)
                spectral["input"] = spectral_input
                spectral["unaccounted"] = spectral_input - closed_spectral
                account["surface_absorbed"] += spectral["surface_absorbed"]
                account["bulk_absorbed"] += spectral["bulk_absorbed"]
            account["absorbed"] = account["surface_absorbed"] + account["bulk_absorbed"]
            default_input = (
                account["detected"] + account["absorbed"] + account["grazing"]
                + account["unreached"] + account["stopped"]
            )
            account["input"] = _source_input(provided, default_input)
            closed = default_input
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
                wavelength_nm=item.get("wavelength_nm", item["ray"].get("wavelength_nm")),
            )
            for item in data["results"]
        ]
        return cls(data["project_name"], rays, results)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    def append_batch(self, other: "TraceBatch") -> "TraceBatch":
        """Append a later wavelength run without rewriting an old batch.

        The returned copy keeps every old leaf and every new leaf; frozen
        batches on disk are therefore safe to reopen and extend.
        """

        # Event uids are only unique inside one batch; offset the appended
        # batch while deep-copying event records so the saved old batch is
        # left byte-for-byte untouched and ledger de-duplication still works.
        offset = 1 + max(
            (event["event_uid"] for result in self.results for event in result.events),
            default=-1,
        )
        rays = [*self.rays, *other.rays]
        results = [
            *self.results,
            *[
                RayResult(
                    ray=result.ray,
                    status=result.status,
                    endpoint=result.endpoint,
                    weight=result.weight,
                    events=[
                        {**event, "event_uid": event["event_uid"] + offset}
                        for event in result.events
                    ],
                    source_label=result.source_label,
                    branch_id=result.branch_id,
                    parent_branch_id=result.parent_branch_id,
                    detector=result.detector,
                    depth=result.depth,
                    wavelength_nm=result.wavelength_nm,
                )
                for result in other.results
            ],
        ]
        name = self.project_name if self.project_name == other.project_name else f"{self.project_name}+{other.project_name}"
        return TraceBatch(project_name=name, rays=rays, results=results)

    def detector_power_comparison(self, other: "TraceBatch") -> dict[str, dict[str, Any]]:
        """Compare detector totals before/after a material response change.

        Every detector reports total delta plus a per-wavelength delta and
        the contributing branches on each side, so gains and losses can be
        traced back to source, wavelength and path rather than averaged.
        """

        before = self.detector_summary()
        after = other.detector_summary()
        comparison: dict[str, dict[str, Any]] = {}
        for detector in sorted(set(before) | set(after)):
            left = before.get(detector, {"power": 0.0, "power_by_wavelength": {}, "branches": []})
            right = after.get(detector, {"power": 0.0, "power_by_wavelength": {}, "branches": []})
            wavelengths = sorted(
                set(left.get("power_by_wavelength", {})) | set(right.get("power_by_wavelength", {})),
                key=lambda label: _wavelength_from_label(label) if label != "broadband" else float("inf"),
            )
            comparison[detector] = {
                "before": left["power"],
                "after": right["power"],
                "delta": right["power"] - left["power"],
                "by_wavelength": {
                    label: {
                        "before": left["power_by_wavelength"].get(label, 0.0),
                        "after": right["power_by_wavelength"].get(label, 0.0),
                        "delta": right["power_by_wavelength"].get(label, 0.0) - left["power_by_wavelength"].get(label, 0.0),
                    }
                    for label in wavelengths
                },
                "branches_before": left.get("branches", []),
                "branches_after": right.get("branches", []),
            }
        return comparison

    def energy_ledger_comparison(self, other: "TraceBatch", input_power: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reconcile aperture clipping and material loss separately."""

        before_ledger = self.energy_ledger(input_power)
        after_ledger = other.energy_ledger(input_power)
        buckets = (
            "detected",
            "surface_absorbed",
            "bulk_absorbed",
            "absorbed",
            "grazing",
            "unreached",
            "stopped",
        )
        comparison: dict[str, Any] = {}
        for source in sorted(set(before_ledger) | set(after_ledger)):
            left = before_ledger.get(source, {bucket: 0.0 for bucket in buckets})
            right = after_ledger.get(source, {bucket: 0.0 for bucket in buckets})
            comparison[source] = {
                bucket: {"before": left.get(bucket, 0.0), "after": right.get(bucket, 0.0),
                         "delta": right.get(bucket, 0.0) - left.get(bucket, 0.0)}
                for bucket in buckets
            }
        return comparison


def load_project(path: str | Path) -> Project:
    return Project.load(path)


def load_batch(path: str | Path) -> TraceBatch:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TraceBatch.from_dict(data)
