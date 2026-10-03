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
        "analyzer_axis": tuple(data["analyzer_axis"]) if data.get("analyzer_axis") else None,
        "polarization_response": data.get("polarization_response"),
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
        "spectral_weights": list(ray.spectral_weights) if ray.spectral_weights else None,
        "polarization": ray.polarization,
        "polarization_weights": list(ray.polarization_weights) if ray.polarization_weights else None,
        "electric_direction": ray.electric_direction,
    }


def _ray_from_dict(data: dict[str, Any]) -> Ray:
    return Ray(
        tuple(data["origin"]),
        tuple(data["direction"]),
        data.get("wavelength_nm"),
        data.get("weight", 1.0),
        data.get("label"),
        tuple(tuple(item) for item in data["spectral_weights"]) if data.get("spectral_weights") else None,
        data.get("polarization"),
        tuple(tuple(item) for item in data["polarization_weights"]) if data.get("polarization_weights") else None,
        tuple(data["electric_direction"]) if data.get("electric_direction") else None,
    )


def _wavelength_key(wavelength_nm: float | None) -> str:
    return "unspecified" if wavelength_nm is None else f"{wavelength_nm:g} nm"


def _add_power(groups: dict[str, dict[str, float]], source: str, wavelength_nm: float | None, power: float) -> None:
    groups.setdefault(source, {})
    key = _wavelength_key(wavelength_nm)
    groups[source][key] = groups[source].get(key, 0.0) + power


def _polarization_key(polarization: str | None) -> str:
    return "unpolarized" if polarization is None else polarization


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
                "wavelength_nm": result.wavelength_nm,
                "polarization": result.polarization,
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
            by_polarization: dict[str, float] = {}
            by_source_wavelength: dict[str, dict[str, float]] = {}
            by_source_polarization: dict[str, dict[str, float]] = {}
            for hit in hits:
                by_source[hit["source_label"]] = by_source.get(hit["source_label"], 0.0) + hit["power"]
                wavelength_key = _wavelength_key(hit["wavelength_nm"])
                polarization_key = _polarization_key(hit.get("polarization"))
                by_wavelength[wavelength_key] = by_wavelength.get(wavelength_key, 0.0) + hit["power"]
                by_polarization[polarization_key] = (
                    by_polarization.get(polarization_key, 0.0) + hit["power"]
                )
                by_source_wavelength.setdefault(hit["source_label"], {})
                by_source_wavelength[hit["source_label"]][wavelength_key] = (
                    by_source_wavelength[hit["source_label"]].get(wavelength_key, 0.0) + hit["power"]
                )
                by_source_polarization.setdefault(hit["source_label"], {})
                by_source_polarization[hit["source_label"]][polarization_key] = (
                    by_source_polarization[hit["source_label"]].get(polarization_key, 0.0) + hit["power"]
                )
            summaries[detector] = {
                "power": total_power,
                "power_by_source": by_source,
                "power_by_wavelength": by_wavelength,
                "power_by_polarization": by_polarization,
                "power_by_source_wavelength": by_source_wavelength,
                "power_by_source_polarization": by_source_polarization,
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
                    "volume_loss_power": event.get("volume_loss_power", 0.0),
                    "analyzer_rejected_power": event.get("analyzer_rejected_power", 0.0),
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
                "wavelength_nm": result.wavelength_nm,
                "polarization": result.polarization,
                "steps": steps,
            })
        return sorted(flows, key=lambda item: (
            item["source_label"],
            -1.0 if item["wavelength_nm"] is None else item["wavelength_nm"],
            str(item.get("polarization")),
            item["branch_id"],
        ))

    def _absorbed_by_source(self) -> dict[str, float]:
        absorbed: dict[str, dict[str, float]] = {}
        seen: set[tuple[str, float | None, str | None, str, int]] = set()
        for result in self.results:
            for event in result.events:
                key = (
                    result.source_label,
                    result.wavelength_nm,
                    result.polarization,
                    result.branch_id.split(".", 1)[0],
                    event["event_uid"],
                )
                if key in seen:
                    continue
                seen.add(key)
                account = absorbed.setdefault(result.source_label, {"surface": 0.0, "volume": 0.0, "analyzer": 0.0})
                account["surface"] += event.get("absorbed_power", 0.0)
                account["volume"] += event.get("volume_loss_power", 0.0)
                account["analyzer"] += event.get("analyzer_rejected_power", 0.0)
        return absorbed

    def _losses_by_source_wavelength_polarization(self):
        absorbed: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        seen: set[tuple[str, float | None, str | None, str, int]] = set()
        for result in self.results:
            for event in result.events:
                key = (
                    result.source_label,
                    result.wavelength_nm,
                    result.polarization,
                    result.branch_id.split(".", 1)[0],
                    event["event_uid"],
                )
                if key in seen:
                    continue
                seen.add(key)
                account = (
                    absorbed.setdefault(result.source_label, {})
                    .setdefault(_wavelength_key(result.wavelength_nm), {})
                    .setdefault(_polarization_key(result.polarization), {
                        "surface": 0.0,
                        "volume": 0.0,
                        "analyzer": 0.0,
                    })
                )
                account["surface"] += event.get("absorbed_power", 0.0)
                account["volume"] += event.get("volume_loss_power", 0.0)
                account["analyzer"] += event.get("analyzer_rejected_power", 0.0)
        return absorbed

    def _initial_inputs_by_source_wavelength_polarization(self):
        roots: dict[tuple[str, float | None, str | None, str], float] = {}
        for result in self.results:
            root_branch = result.branch_id.split(".", 1)[0]
            key = (result.source_label, result.wavelength_nm, result.polarization, root_branch)
            if result.events:
                first_event = result.events[0]
                initial_power = (
                    first_event.get("incident_weight", result.weight)
                    + first_event.get("volume_loss_power", 0.0)
                )
            else:
                initial_power = result.weight
            roots[key] = max(roots.get(key, 0.0), initial_power)

        inputs: dict[str, dict[str, dict[str, float]]] = {}
        for (source, wavelength, polarization, _root), power in roots.items():
            wavelength_key = _wavelength_key(wavelength)
            polarization_key = _polarization_key(polarization)
            inputs.setdefault(source, {}).setdefault(wavelength_key, {})
            inputs[source][wavelength_key][polarization_key] = (
                inputs[source][wavelength_key].get(polarization_key, 0.0) + power
            )
        return inputs

    def energy_ledger(self, input_power: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
        """Power account per source.

        ``detected`` reached a detector, ``absorbed`` was lost on surfaces,
        ``grazing`` remains on beams stopped at an aperture edge and
        ``unreached``/``stopped`` are beams that escaped or hit the iteration
        limits.  Stopped residuals are never reported as absorbed.
        """

        absorbed_detail = self._absorbed_by_source()
        losses = self._losses_by_source_wavelength_polarization()
        accounts: dict[str, dict[str, Any]] = {}
        for result in self.results:
            accounts.setdefault(result.source_label, {
                "input": 0.0,
                "detected": 0.0,
                "absorbed": 0.0,
                "surface_absorption": 0.0,
                "material_loss": 0.0,
                "volume_loss": 0.0,
                "grazing": 0.0,
                "aperture_grazing": 0.0,
                "analyzer_blocked": 0.0,
                "unreached": 0.0,
                "stopped": 0.0,
                "unknown_response": 0.0,
                "by_wavelength": {},
                "by_polarization": {},
            })
            account = accounts[result.source_label]
            wavelength_account = account["by_wavelength"].setdefault(
                _wavelength_key(result.wavelength_nm),
                {
                    "input": 0.0,
                    "detected": 0.0,
                    "absorbed": 0.0,
                    "surface_absorption": 0.0,
                    "material_loss": 0.0,
                    "volume_loss": 0.0,
                    "grazing": 0.0,
                    "analyzer_blocked": 0.0,
                    "unreached": 0.0,
                    "stopped": 0.0,
                    "unknown_response": 0.0,
                    "by_polarization": {},
                },
            )
            bucket = {
                "reached_image": "detected",
                "grazing": "grazing",
                "no_image": "unreached",
                "max_events": "stopped",
                "max_branches": "stopped",
                "unknown_response": "unknown_response",
                "analyzer_blocked": "analyzer_blocked",
                # A leaf terminated by a fully absorbing surface: its power
                # is already booked as surface absorption below.
                "absorbed": None,
            }.get(result.status, "stopped")
            if bucket is not None:
                account[bucket] += result.weight
                wavelength_account[bucket] += result.weight
                polarization_key = _polarization_key(result.polarization)
                polarization_account = account["by_polarization"].setdefault(polarization_key, {
                    "input": 0.0, "detected": 0.0, "absorbed": 0.0,
                    "surface_absorption": 0.0, "material_loss": 0.0, "volume_loss": 0.0,
                    "grazing": 0.0, "aperture_grazing": 0.0, "analyzer_blocked": 0.0,
                    "unreached": 0.0, "stopped": 0.0, "unknown_response": 0.0,
                })
                polarization_account[bucket] += result.weight
                polarization_account["input"] += result.weight
                wavelength_pol_account = wavelength_account["by_polarization"].setdefault(polarization_key, {
                    "input": 0.0, "detected": 0.0, "absorbed": 0.0,
                    "surface_absorption": 0.0, "material_loss": 0.0, "volume_loss": 0.0,
                    "grazing": 0.0, "aperture_grazing": 0.0, "analyzer_blocked": 0.0,
                    "unreached": 0.0, "stopped": 0.0, "unknown_response": 0.0,
                })
                wavelength_pol_account[bucket] += result.weight

        for source, account in accounts.items():
            initial_inputs = self._initial_inputs_by_source_wavelength_polarization().get(source, {})
            detail = absorbed_detail.get(source, {"surface": 0.0, "volume": 0.0, "analyzer": 0.0})
            account["surface_absorption"] = detail["surface"]
            account["volume_loss"] = detail["volume"]
            account["absorbed"] = account["surface_absorption"] + account["volume_loss"]
            account["material_loss"] = account["volume_loss"]
            account["aperture_grazing"] = account["grazing"]
            for key, wavelength_account in account["by_wavelength"].items():
                wavelength_input = initial_inputs.get(key, {})
                wavelength_account["input"] = sum(wavelength_input.values())
                wavelength_detail = {"surface": 0.0, "volume": 0.0, "analyzer": 0.0}
                for pol_detail in losses.get(source, {}).get(key, {}).values():
                    for category in wavelength_detail:
                        wavelength_detail[category] = wavelength_detail[category] + pol_detail[category]
                wavelength_account["surface_absorption"] = wavelength_detail["surface"]
                wavelength_account["volume_loss"] = wavelength_detail["volume"]
                wavelength_account["absorbed"] = wavelength_detail["surface"] + wavelength_detail["volume"]
                wavelength_account["material_loss"] = wavelength_detail["volume"]
                wavelength_account["aperture_grazing"] = wavelength_account["grazing"]
                for pol_key, pol_account in wavelength_account["by_polarization"].items():
                    pol_account["input"] = wavelength_input.get(pol_key, 0.0)
                    detail = losses.get(source, {}).get(key, {}).get(
                        pol_key, {"surface": 0.0, "volume": 0.0, "analyzer": 0.0}
                    )
                    pol_account["surface_absorption"] = detail["surface"]
                    pol_account["volume_loss"] = detail["volume"]
                    pol_account["absorbed"] = detail["surface"] + detail["volume"]
                    pol_account["material_loss"] = detail["volume"]
                    pol_account["aperture_grazing"] = pol_account["grazing"]
                    closed_pol = (
                        pol_account["detected"] + pol_account["absorbed"] + pol_account["grazing"]
                        + pol_account["analyzer_blocked"] + pol_account["unreached"]
                        + pol_account["stopped"] + pol_account["unknown_response"]
                    )
                    pol_account["unaccounted"] = pol_account["input"] - closed_pol
                wavelength_closed = (
                    wavelength_account["detected"]
                    + wavelength_account["absorbed"]
                    + wavelength_account["grazing"]
                    + wavelength_account["analyzer_blocked"]
                    + wavelength_account["unreached"]
                    + wavelength_account["stopped"]
                    + wavelength_account["unknown_response"]
                )
                wavelength_account["unaccounted"] = wavelength_account["input"] - wavelength_closed
            for pol_key, pol_account in account["by_polarization"].items():
                pol_account["input"] = sum(
                    wavelength_inputs.get(pol_key, 0.0)
                    for wavelength_inputs in initial_inputs.values()
                )
                pol_detail = {"surface": 0.0, "volume": 0.0, "analyzer": 0.0}
                for wavelength_detail in losses.get(source, {}).values():
                    detail = wavelength_detail.get(pol_key)
                    if detail:
                        for category in pol_detail:
                            pol_detail[category] = pol_detail[category] + detail[category]
                pol_account["surface_absorption"] = pol_detail["surface"]
                pol_account["volume_loss"] = pol_detail["volume"]
                pol_account["absorbed"] = pol_detail["surface"] + pol_detail["volume"]
                pol_account["material_loss"] = pol_detail["volume"]
                pol_account["aperture_grazing"] = pol_account["grazing"]
                closed_pol = (
                    pol_account["detected"] + pol_account["absorbed"] + pol_account["grazing"]
                    + pol_account["analyzer_blocked"] + pol_account["unreached"]
                    + pol_account["stopped"] + pol_account["unknown_response"]
                )
                pol_account["unaccounted"] = pol_account["input"] - closed_pol
            inferred_input = sum(
                sum(wavelength_inputs.values())
                for wavelength_inputs in initial_inputs.values()
            )
            account["input"] = (input_power or {}).get(source, inferred_input)
            closed = (
                account["detected"] + account["absorbed"] + account["grazing"]
                + account["analyzer_blocked"] + account["unreached"]
                + account["stopped"] + account["unknown_response"]
            )
            account["unaccounted"] = account["input"] - closed
        return accounts

    def detector_power_by_wavelength(self) -> dict[str, dict[str, float]]:
        return {
            detector: summary["power_by_wavelength"]
            for detector, summary in self.detector_summary().items()
        }

    def compare_detector_power(self, other: "TraceBatch") -> dict[str, dict[str, Any]]:
        """Compare detector totals before and after a material-response change."""

        before = self.detector_summary()
        after = other.detector_summary()
        comparison: dict[str, dict[str, Any]] = {}
        for detector in sorted(set(before) | set(after)):
            before_summary = before.get(detector, {"power": 0.0, "power_by_wavelength": {}})
            after_summary = after.get(detector, {"power": 0.0, "power_by_wavelength": {}})
            wavelengths = sorted(
                set(before_summary.get("power_by_wavelength", {}))
                | set(after_summary.get("power_by_wavelength", {}))
            )
            comparison[detector] = {
                "before": before_summary.get("power", 0.0),
                "after": after_summary.get("power", 0.0),
                "delta": after_summary.get("power", 0.0) - before_summary.get("power", 0.0),
                "by_wavelength": {
                    wavelength: {
                        "before": before_summary.get("power_by_wavelength", {}).get(wavelength, 0.0),
                        "after": after_summary.get("power_by_wavelength", {}).get(wavelength, 0.0),
                        "delta": (
                            after_summary.get("power_by_wavelength", {}).get(wavelength, 0.0)
                            - before_summary.get("power_by_wavelength", {}).get(wavelength, 0.0)
                        ),
                    }
                    for wavelength in wavelengths
                },
            }
        return comparison

    def compare_energy_ledger(
        self,
        other: "TraceBatch",
        input_power: dict[str, float] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Compare material losses, aperture clipping and terminal residuals."""

        before_ledger = self.energy_ledger(input_power)
        after_ledger = other.energy_ledger(input_power)
        comparison: dict[str, dict[str, Any]] = {}
        categories = (
            "detected",
            "absorbed",
            "material_loss",
            "surface_absorption",
            "volume_loss",
            "analyzer_blocked",
            "grazing",
            "aperture_grazing",
            "unreached",
            "stopped",
            "unknown_response",
            "unaccounted",
        )
        for source in sorted(set(before_ledger) | set(after_ledger)):
            before_account = before_ledger.get(source, {})
            after_account = after_ledger.get(source, {})
            delta = {
                category: after_account.get(category, 0.0) - before_account.get(category, 0.0)
                for category in categories
            }
            comparison[source] = {
                "before": before_account,
                "after": after_account,
                "delta": delta,
            }
        return comparison

    def compare_wavelength_response(
        self,
        other: "TraceBatch",
        input_power: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return {
            "detectors": self.compare_detector_power(other),
            "energy_ledger": self.compare_energy_ledger(other, input_power),
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
                polarization=item.get("polarization", item["ray"].get("polarization")),
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
