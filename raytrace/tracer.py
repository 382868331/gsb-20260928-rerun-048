from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .rays import Ray, RayRecord
from .surfaces import Hit, Surface


@dataclass(frozen=True)
class TraceConfig:
    max_path_length: float = 100.0
    max_interactions: int = 64
    grazing_cos_threshold: float = 1.0e-4
    surface_epsilon: float = 1.0e-8


def _event(kind: str, hit: Hit, data: dict[str, Any], path_length: float) -> dict[str, Any]:
    event = {
        "kind": kind,
        "surface": hit.surface.name,
        "t": hit.t,
        "path_length": path_length,
        "point": hit.point.tolist(),
    }
    event.update(data)
    return event


def trace_ray(ray: Ray, surfaces: list[Surface], config: TraceConfig) -> RayRecord:
    point = ray.origin.copy()
    direction = ray.direction.copy()
    intensity = float(ray.intensity)
    path_length = 0.0
    events: list[dict[str, Any]] = []
    ray_id = ray.ray_id or f"ray-{id(ray):x}"

    for interaction_count in range(config.max_interactions):
        remaining = config.max_path_length - path_length
        if remaining <= 0.0:
            return _record(
                ray_id,
                "no_image",
                events,
                point,
                direction,
                intensity,
                path_length,
                "maximum path length reached",
            )

        hits: list[Hit] = []
        for surface in surfaces:
            hit = surface.intersect(
                point, direction, config.surface_epsilon
            )
            if hit is not None and hit.t <= remaining + 1.0e-9:
                hits.append(hit)
        if not hits:
            point = point + direction * remaining
            path_length = config.max_path_length
            return _record(
                ray_id,
                "no_image",
                events,
                point,
                direction,
                intensity,
                path_length,
                "no surface or image plane reached within the path budget",
            )

        hit = min(hits, key=lambda item: item.t)
        result = hit.surface.interaction(direction, hit, ray.wavelength_um)
        kind = result["kind"]
        path_length += hit.t
        normal = result["normal"]
        cos_incidence = abs(float(np.dot(direction, normal)))
        intensity_before = intensity
        intensity_after = 0.0 if kind == "absorb" else intensity
        event_data = {
            "cos_incidence": cos_incidence,
            "intensity_before": intensity_before,
            "intensity_after": intensity_after,
        }
        if "n_in" in result:
            event_data["n_in"] = result["n_in"]
            event_data["n_out"] = result["n_out"]
        events.append(
            _event(
                kind,
                hit,
                event_data,
                path_length,
            )
        )
        point = hit.point
        intensity = intensity_after

        if kind == "image":
            return _record(
                ray_id,
                "reached_image",
                events,
                point,
                direction,
                intensity,
                path_length,
            )
        if kind == "absorb":
            return _record(
                ray_id,
                "absorbed",
                events,
                point,
                direction,
                0.0,
                path_length,
                "absorbed by a boundary",
            )
        if kind == "total_internal_reflection":
            return _record(
                ray_id,
                "total_internal_reflection",
                events,
                point,
                direction,
                intensity,
                path_length,
                "ray retained at TIR event; split reflected beams are not modeled",
            )
        if cos_incidence <= config.grazing_cos_threshold:
            events[-1]["kind"] = "grazing"
            return _record(
                ray_id,
                "grazing",
                events,
                point,
                direction,
                intensity,
                path_length,
                "incidence is too close to tangential for a stable transmitted ray",
            )
        if kind == "refract":
            direction = result["transmitted_direction"]
        elif kind == "reflect":
            direction = result["reflected_direction"]
        else:
            return _record(
                ray_id,
                "no_image",
                events,
                point,
                direction,
                intensity,
                path_length,
                f"unsupported interaction {kind}",
            )

    return _record(
        ray_id,
        "no_image",
        events,
        point,
        direction,
        intensity,
        path_length,
        "maximum interaction count reached",
    )


def _record(
    ray_id,
    status,
    events,
    point,
    direction,
    intensity,
    path_length,
    message="",
) -> RayRecord:
    return RayRecord(
        ray_id=ray_id,
        status=status,
        events=events,
        final_point=point,
        final_direction=direction,
        final_intensity=float(intensity),
        path_length=float(path_length),
        message=message,
    )


def trace_batch(
    rays: list[Ray], surfaces: list[Surface], config: TraceConfig
) -> list[RayRecord]:
    return [trace_ray(ray, surfaces, config) for ray in rays]


def records_to_dict(records: list[RayRecord]) -> dict[str, Any]:
    return {
        "records": [
            {
                "ray_id": record.ray_id,
                "status": record.status,
                "message": record.message,
                "events": record.events,
                "final_point": record.final_point.tolist(),
                "final_direction": record.final_direction.tolist(),
                "final_intensity": record.final_intensity,
                "path_length": record.path_length,
            }
            for record in records
        ]
    }


def save_records_json(records: list[RayRecord], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(records_to_dict(records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def records_from_dict(data: dict[str, Any]) -> list[RayRecord]:
    return [
        RayRecord(
            ray_id=item["ray_id"],
            status=item["status"],
            events=item["events"],
            final_point=np.asarray(item["final_point"], dtype=float),
            final_direction=np.asarray(item["final_direction"], dtype=float),
            final_intensity=float(item["final_intensity"]),
            path_length=float(item["path_length"]),
            message=item.get("message", ""),
        )
        for item in data["records"]
    ]


def load_records_json(path: str | Path) -> list[RayRecord]:
    return records_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
