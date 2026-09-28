from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from .rays import Ray


@dataclass
class Engineering:
    """A re-openable source batch and its trace settings."""

    name: str
    wavelength_um: float = 0.5876
    rays: list[Ray] = field(default_factory=list)
    max_path_length: float = 100.0
    max_interactions: int = 64
    grazing_cos_threshold: float = 1.0e-4
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wavelength_um": self.wavelength_um,
            "max_path_length": self.max_path_length,
            "max_interactions": self.max_interactions,
            "grazing_cos_threshold": self.grazing_cos_threshold,
            "notes": self.notes,
            "rays": [
                {
                    "ray_id": ray.ray_id,
                    "origin": ray.origin.tolist(),
                    "direction": ray.direction.tolist(),
                    "wavelength_um": ray.wavelength_um,
                    "intensity": ray.intensity,
                    "metadata": ray.metadata,
                }
                for ray in self.rays
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Engineering":
        rays = [
            Ray(
                origin=item["origin"],
                direction=item["direction"],
                wavelength_um=item.get("wavelength_um", data.get("wavelength_um", 0.5876)),
                intensity=item.get("intensity", 1.0),
                ray_id=item.get("ray_id"),
                metadata=item.get("metadata", {}),
            )
            for item in data.get("rays", [])
        ]
        return cls(
            name=data["name"],
            wavelength_um=float(data.get("wavelength_um", 0.5876)),
            rays=rays,
            max_path_length=float(data.get("max_path_length", 100.0)),
            max_interactions=int(data.get("max_interactions", 64)),
            grazing_cos_threshold=float(data.get("grazing_cos_threshold", 1.0e-4)),
            notes=data.get("notes", ""),
        )

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "Engineering":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class OpticalSystem:
    name: str
    surfaces: list[Any]
    notes: str = ""

    def __post_init__(self) -> None:
        names = [surface.name for surface in self.surfaces]
        if len(names) != len(set(names)):
            raise ValueError("surface names must be unique")


def fan_rays(
    origins_y: np.ndarray,
    angles_rad: np.ndarray,
    wavelength_um: float = 0.5876,
    x: float = 0.0,
    z: float = 0.0,
) -> list[Ray]:
    rays: list[Ray] = []
    for yi, y in enumerate(np.asarray(origins_y, dtype=float)):
        for ai, angle in enumerate(np.asarray(angles_rad, dtype=float)):
            rays.append(
                Ray(
                    origin=(x, float(y), z),
                    direction=(0.0, float(np.sin(angle)), float(np.cos(angle))),
                    wavelength_um=wavelength_um,
                    ray_id=f"r{yi:03d}-{ai:03d}",
                )
            )
    return rays

