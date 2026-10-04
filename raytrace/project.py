"""Serializable projects and trace batches."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import warnings

from .materials import MaterialLibrary
from .measurements import AffineQuantity, MeasurementBatch
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
        "polarization_reflectivity": data.get("polarization_reflectivity"),
        "polarization_transmissivity": data.get("polarization_transmissivity"),
        "analyzer_angle_deg": data.get("analyzer_angle_deg"),
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
    # Measured interval responses, keyed by batch id.  Only one batch is
    # active for new traces; revoked batches stay recorded so older
    # projects and trace batches keep pointing at their original
    # measurement basis.
    measurement_batches: dict[str, MeasurementBatch] = field(default_factory=dict)
    active_measurement_batch: str | None = None

    LEGACY_CONVENTION = "legacy-beamsplitter-reflectivity-is-transmission"

    def add_measurement_batch(self, batch: MeasurementBatch, activate: bool = True) -> None:
        self.measurement_batches[batch.batch_id] = batch
        if activate:
            if batch.revoked:
                raise ValueError(f"measurement batch {batch.batch_id!r} is revoked")
            self.active_measurement_batch = batch.batch_id

    def revoke_measurement_batch(self, batch_id: str) -> None:
        """Withdraw a batch; existing provenance references stay intact."""

        try:
            batch = self.measurement_batches[batch_id]
        except KeyError:
            raise KeyError(f"unknown measurement batch {batch_id!r}") from None
        batch.revoked = True
        if self.active_measurement_batch == batch_id:
            self.active_measurement_batch = None

    def measurement_conflicts(self) -> list[dict[str, Any]]:
        """Mutually exclusive conditions among the non-revoked batches."""

        batches = [batch for batch in self.measurement_batches.values() if not batch.revoked]
        conflicts: list[dict[str, Any]] = []
        for index, first in enumerate(batches):
            for second in batches[index + 1:]:
                conflicts.extend(first.conflicts_with(second))
        return conflicts

    def active_measurements(self) -> MeasurementBatch | None:
        if self.active_measurement_batch is None:
            return None
        batch = self.measurement_batches.get(self.active_measurement_batch)
        if batch is None or batch.revoked:
            return None
        return batch

    def trace_with_measurements(self, batch_id: str, rays: list[Ray]) -> "TraceBatch":
        """Trace with one specific batch, even a revoked one (for audits)."""

        try:
            batch = self.measurement_batches[batch_id]
        except KeyError:
            raise KeyError(f"unknown measurement batch {batch_id!r}") from None
        tracer = RayTracer(self.surfaces, self.materials, self.settings, measurements=batch)
        return TraceBatch(
            project_name=self.name,
            rays=rays,
            results=tracer.trace_branches(rays),
            measurement_batch_id=batch_id,
        )

    def compare_measurement_batches(
        self,
        first_id: str,
        second_id: str,
        rays: list[Ray],
        input_power: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compare two measurement batches on identical inputs."""

        first = self.trace_with_measurements(first_id, rays)
        second = self.trace_with_measurements(second_id, rays)
        first_uncertainty = first.uncertainty_summary(input_power)
        second_uncertainty = second.uncertainty_summary(input_power)
        uncertainty: dict[str, Any] = {}
        for source in sorted(set(first_uncertainty) | set(second_uncertainty)):
            before = first_uncertainty.get(source, {})
            after = second_uncertainty.get(source, {})
            uncertainty[source] = {
                "before": {
                    "certain": before.get("certain", 0.0),
                    "possible": before.get("possible"),
                    "unknown": before.get("unknown", 0.0),
                },
                "after": {
                    "certain": after.get("certain", 0.0),
                    "possible": after.get("possible"),
                    "unknown": after.get("unknown", 0.0),
                },
            }
        return {
            "measurement_batches": [first_id, second_id],
            "detectors": first.compare_detector_power(second),
            "uncertainty": uncertainty,
            "energy_ledger": first.compare_energy_ledger(second, input_power),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "convention": self.convention,
            "migration_notes": self.migration_notes,
            "settings": vars(self.settings),
            "materials": self.materials.to_dict(),
            "surfaces": [_surface_to_dict(surface) for surface in self.surfaces],
            "measurement_batches": [batch.to_dict() for batch in self.measurement_batches.values()],
            "active_measurement_batch": self.active_measurement_batch,
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
            measurement_batches={
                batch.batch_id: batch
                for batch in (
                    MeasurementBatch.from_dict(item)
                    for item in data.get("measurement_batches", [])
                )
            },
            active_measurement_batch=data.get("active_measurement_batch"),
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
        measurements = self.active_measurements()
        tracer = RayTracer(self.surfaces, self.materials, self.settings, measurements=measurements)
        return TraceBatch(
            project_name=self.name,
            rays=rays,
            results=tracer.trace_branches(rays),
            measurement_batch_id=measurements.batch_id if measurements is not None else None,
        )


def _ray_to_dict(ray: Ray) -> dict[str, Any]:
    return {
        "origin": ray.origin,
        "direction": ray.direction,
        "wavelength_nm": ray.wavelength_nm,
        "weight": ray.weight,
        "label": ray.label,
        "spectral_weights": list(ray.spectral_weights) if ray.spectral_weights else None,
        "polarization": list(ray.polarization) if ray.polarization else None,
        "polarization_angle_deg": ray.polarization_angle_deg,
        "polarization_weights": list(ray.polarization_weights) if ray.polarization_weights else None,
    }


def _ray_from_dict(data: dict[str, Any]) -> Ray:
    return Ray(
        tuple(data["origin"]),
        tuple(data["direction"]),
        data.get("wavelength_nm"),
        data.get("weight", 1.0),
        data.get("label"),
        tuple(tuple(item) for item in data["spectral_weights"]) if data.get("spectral_weights") else None,
        tuple(data["polarization"]) if data.get("polarization") else None,
        data.get("polarization_angle_deg"),
        tuple(tuple(item) for item in data["polarization_weights"]) if data.get("polarization_weights") else None,
    )


def _polarization_key(polarization: float | None) -> str:
    return "unpolarized" if polarization is None else f"{polarization:g} deg"


def _wavelength_key(wavelength_nm: float | None) -> str:
    return "unspecified" if wavelength_nm is None else f"{wavelength_nm:g} nm"


def _add_power(groups: dict[str, dict[str, float]], source: str, wavelength_nm: float | None, power: float) -> None:
    groups.setdefault(source, {})
    key = _wavelength_key(wavelength_nm)
    groups[source][key] = groups[source].get(key, 0.0) + power


@dataclass
class TraceBatch:
    project_name: str
    rays: list[Ray]
    results: list[RayResult]
    # Provenance: which measurement batch produced these results.  Saved
    # with the batch so reopened results always point at their original
    # measurement basis, even after that batch is revoked.
    measurement_batch_id: str | None = None

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
                "power_uncertainty": result.weight_uncertainty,
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
            total_affine = AffineQuantity(0.0)
            by_source: dict[str, float] = {}
            by_wavelength: dict[str, float] = {}
            by_source_wavelength: dict[str, dict[str, float]] = {}
            by_polarization: dict[str, float] = {}
            for hit in hits:
                affine = AffineQuantity.from_dict(hit.get("power_uncertainty"))
                if affine is None:
                    affine = AffineQuantity(hit["power"])
                total_affine = total_affine + affine
                by_source[hit["source_label"]] = by_source.get(hit["source_label"], 0.0) + hit["power"]
                wavelength_key = _wavelength_key(hit["wavelength_nm"])
                by_wavelength[wavelength_key] = by_wavelength.get(wavelength_key, 0.0) + hit["power"]
                polarization_key = _polarization_key(hit["polarization"])
                by_polarization[polarization_key] = by_polarization.get(polarization_key, 0.0) + hit["power"]
                by_source_wavelength.setdefault(hit["source_label"], {})
                by_source_wavelength[hit["source_label"]][wavelength_key] = (
                    by_source_wavelength[hit["source_label"]].get(wavelength_key, 0.0) + hit["power"]
                )
            summaries[detector] = {
                "power": total_power,
                "power_interval": (
                    max(0.0, total_affine.interval()[0]),
                    total_affine.interval()[1],
                ),
                "power_by_source": by_source,
                "power_by_wavelength": by_wavelength,
                "power_by_polarization": by_polarization,
                "power_by_source_wavelength": by_source_wavelength,
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
                    "bulk_loss_power": event.get("bulk_loss_power", 0.0),
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
                "steps": steps,
            })
        return sorted(flows, key=lambda item: (
            item["source_label"],
            -1.0 if item["wavelength_nm"] is None else item["wavelength_nm"],
            item["branch_id"],
        ))

    def _absorbed_by_source(self) -> dict[str, float]:
        absorbed: dict[str, float] = {}
        seen: set[tuple[str, float | None, float | None, int]] = set()
        for result in self.results:
            for event in result.events:
                key = (result.source_label, result.wavelength_nm, result.polarization, event["event_uid"])
                if key in seen:
                    continue
                seen.add(key)
                absorbed[result.source_label] = absorbed.get(result.source_label, 0.0) + event.get("absorbed_power", 0.0)
        return absorbed

    def _absorbed_by_source_wavelength(self) -> dict[str, dict[str, float]]:
        absorbed: dict[str, dict[str, float]] = {}
        seen: set[tuple[str, float | None, float | None, int]] = set()
        for result in self.results:
            for event in result.events:
                key = (result.source_label, result.wavelength_nm, result.polarization, event["event_uid"])
                if key in seen:
                    continue
                seen.add(key)
                power = event.get("absorbed_power", 0.0)
                if power:
                    _add_power(absorbed, result.source_label, result.wavelength_nm, power)
        return absorbed

    def _event_power_by_source(self, field_name: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        """Sum an event power field once per (source, wavelength, polarization, event)."""

        by_source: dict[str, float] = {}
        by_source_wavelength: dict[str, dict[str, float]] = {}
        seen: set[tuple[str, float | None, float | None, int]] = set()
        for result in self.results:
            for event in result.events:
                key = (result.source_label, result.wavelength_nm, result.polarization, event["event_uid"])
                if key in seen:
                    continue
                seen.add(key)
                power = event.get(field_name, 0.0)
                if not power:
                    continue
                by_source[result.source_label] = by_source.get(result.source_label, 0.0) + power
                _add_power(by_source_wavelength, result.source_label, result.wavelength_nm, power)
        return by_source, by_source_wavelength

    def energy_ledger(self, input_power: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
        """Power account per source.

        ``detected`` reached a detector, ``absorbed`` was lost on surfaces,
        ``grazing`` remains on beams stopped at an aperture edge and
        ``unreached``/``stopped`` are beams that escaped or hit the iteration
        limits.  Stopped residuals are never reported as absorbed.
        """

        absorbed = self._absorbed_by_source()
        absorbed_by_wavelength = self._absorbed_by_source_wavelength()
        bulk_loss, bulk_loss_by_wavelength = self._event_power_by_source("bulk_loss_power")
        analyzer_rejected, analyzer_rejected_by_wavelength = self._event_power_by_source("analyzer_rejected_power")
        accounts: dict[str, dict[str, Any]] = {}
        for result in self.results:
            accounts.setdefault(result.source_label, {
                "input": 0.0,
                "detected": 0.0,
                "absorbed": 0.0,
                "material_loss": 0.0,
                "bulk_loss": 0.0,
                "analyzer_rejected": 0.0,
                "grazing": 0.0,
                "aperture_grazing": 0.0,
                "unreached": 0.0,
                "stopped": 0.0,
                "unknown_response": 0.0,
                "by_wavelength": {},
            })
            account = accounts[result.source_label]
            wavelength_account = account["by_wavelength"].setdefault(
                _wavelength_key(result.wavelength_nm),
                {
                    "input": 0.0,
                    "detected": 0.0,
                    "absorbed": 0.0,
                    "material_loss": 0.0,
                    "bulk_loss": 0.0,
                    "analyzer_rejected": 0.0,
                    "grazing": 0.0,
                    "unreached": 0.0,
                    "stopped": 0.0,
                    "unknown_response": 0.0,
                },
            )
            bucket = {
                "reached_image": "detected",
                "grazing": "grazing",
                "no_image": "unreached",
                "max_events": "stopped",
                "max_branches": "stopped",
                "unknown_response": "unknown_response",
                # A leaf terminated by a fully absorbing surface: its power
                # is already booked as surface absorption below.
                "absorbed": None,
            }.get(result.status, "stopped")
            if bucket is not None:
                account[bucket] += result.weight
                wavelength_account[bucket] += result.weight
            wavelength_account["input"] += result.weight

        for source, account in accounts.items():
            account["absorbed"] = absorbed.get(source, 0.0)
            account["material_loss"] = account["absorbed"]
            account["bulk_loss"] = bulk_loss.get(source, 0.0)
            account["analyzer_rejected"] = analyzer_rejected.get(source, 0.0)
            account["aperture_grazing"] = account["grazing"]
            for key, wavelength_account in account["by_wavelength"].items():
                wavelength_account["absorbed"] = absorbed_by_wavelength.get(source, {}).get(key, 0.0)
                wavelength_account["material_loss"] = wavelength_account["absorbed"]
                wavelength_account["bulk_loss"] = bulk_loss_by_wavelength.get(source, {}).get(key, 0.0)
                wavelength_account["analyzer_rejected"] = analyzer_rejected_by_wavelength.get(source, {}).get(key, 0.0)
                wavelength_closed = (
                    wavelength_account["detected"]
                    + wavelength_account["absorbed"]
                    + wavelength_account["bulk_loss"]
                    + wavelength_account["analyzer_rejected"]
                    + wavelength_account["grazing"]
                    + wavelength_account["unreached"]
                    + wavelength_account["stopped"]
                    + wavelength_account["unknown_response"]
                )
                wavelength_account["unaccounted"] = wavelength_account["input"] - wavelength_closed
            account["input"] = (input_power or {}).get(
                source,
                account["detected"] + account["absorbed"] + account["grazing"]
                + account["bulk_loss"] + account["analyzer_rejected"]
                + account["unreached"] + account["stopped"] + account["unknown_response"],
            )
            closed = (
                account["detected"] + account["absorbed"] + account["grazing"]
                + account["bulk_loss"] + account["analyzer_rejected"]
                + account["unreached"] + account["stopped"] + account["unknown_response"]
            )
            account["unaccounted"] = account["input"] - closed
        return accounts

    def detector_power_by_wavelength(self) -> dict[str, dict[str, float]]:
        return {
            detector: summary["power_by_wavelength"]
            for detector, summary in self.detector_summary().items()
        }

    def uncertainty_summary(self, input_power: dict[str, float] | None = None) -> dict[str, dict[str, Any]]:
        """Certain power, possible range and undeterminable share per source.

        'certain' is detected power that does not depend on any measured
        interval.  'possible' is the affine sum of detected powers that
        do: beams sharing a measurement symbol recombine with their
        correlation intact instead of adding as independent errors, and
        the reported interval is clipped to the energy the source can
        physically supply.  'unknown' is power that terminated on
        unmeasured channels and cannot be determined at all.
        """

        inputs: dict[str, float] = {}
        for index, ray in enumerate(self.rays):
            label = ray.label if ray.label is not None else f"ray-{index}"
            inputs[label] = inputs.get(label, 0.0) + ray.weight
        collected: dict[str, dict[str, Any]] = {}
        for result in self.results:
            entry = collected.setdefault(result.source_label, {
                "certain": 0.0,
                "unknown": 0.0,
                "possible_affine": AffineQuantity(0.0),
                "unknown_affine": AffineQuantity(0.0),
                "symbol_branches": {},
            })
            affine = AffineQuantity.from_dict(result.weight_uncertainty)
            if affine is None:
                affine = AffineQuantity(result.weight)
            if result.status == "reached_image":
                if affine.is_certain:
                    entry["certain"] += result.weight
                else:
                    entry["possible_affine"] = entry["possible_affine"] + affine
            elif result.status == "unknown_response":
                entry["unknown"] += result.weight
                entry["unknown_affine"] = entry["unknown_affine"] + affine
            for symbol in affine.symbols:
                entry["symbol_branches"].setdefault(symbol, set()).add(result.branch_id)
        summary: dict[str, dict[str, Any]] = {}
        for source, entry in collected.items():
            source_input = (input_power or {}).get(source, inputs.get(source, 0.0))
            possible = entry["possible_affine"]
            low, high = possible.interval()
            ceiling = max(0.0, source_input - entry["certain"])
            clipped = high > ceiling + 1e-12 or low < -1e-12
            unknown_affine = entry["unknown_affine"]
            summary[source] = {
                "input": source_input,
                "certain": entry["certain"],
                "possible": {
                    "nominal": possible.constant,
                    "interval": (max(0.0, low), min(high, ceiling)),
                    "clipped_by_energy_constraint": clipped,
                },
                "unknown": entry["unknown"],
                "unknown_interval": (
                    unknown_affine.interval()
                    if not unknown_affine.is_certain
                    else (entry["unknown"], entry["unknown"])
                ),
                "shared_symbols": {
                    symbol: sorted(branches)
                    for symbol, branches in entry["symbol_branches"].items()
                    if len(branches) > 1
                },
            }
        return summary

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
            "bulk_loss",
            "analyzer_rejected",
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

    def compare_polarization_response(
        self,
        other: "TraceBatch",
        input_power: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Compare two batches that differ only in polarization handling.

        Reports per-detector totals, the per-polarization breakdown and the
        full energy-ledger delta so a rotated input or analyzer can be
        audited path by path.
        """

        before = self.detector_summary()
        after = other.detector_summary()
        by_polarization: dict[str, dict[str, Any]] = {}
        for detector in sorted(set(before) | set(after)):
            before_pol = before.get(detector, {}).get("power_by_polarization", {})
            after_pol = after.get(detector, {}).get("power_by_polarization", {})
            by_polarization[detector] = {
                key: {
                    "before": before_pol.get(key, 0.0),
                    "after": after_pol.get(key, 0.0),
                    "delta": after_pol.get(key, 0.0) - before_pol.get(key, 0.0),
                }
                for key in sorted(set(before_pol) | set(after_pol))
            }
        return {
            "detectors": self.compare_detector_power(other),
            "detectors_by_polarization": by_polarization,
            "energy_ledger": self.compare_energy_ledger(other, input_power),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "measurement_batch_id": self.measurement_batch_id,
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
                polarization=item.get("polarization"),
                weight_uncertainty=item.get("weight_uncertainty"),
            )
            for item in data["results"]
        ]
        return cls(data["project_name"], rays, results, data.get("measurement_batch_id"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_project(path: str | Path) -> Project:
    return Project.load(path)


def load_batch(path: str | Path) -> TraceBatch:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TraceBatch.from_dict(data)
