from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raytrace import (
    Air,
    ConstantMaterial,
    Engineering,
    ImagePlane,
    OpticalSystem,
    SphericalInterface,
    TraceConfig,
)
from raytrace.system import fan_rays
from raytrace.tracer import save_records_json, trace_batch


def main() -> None:
    air = Air()
    glass = ConstantMaterial("acrylic-like", 1.491)
    front = SphericalInterface(
        "front",
        aperture_radius=1.0,
        z=0.0,
        curvature_radius=8.0,
        material_normal_side=glass,
        material_opposite_side=air,
    )
    back = SphericalInterface(
        "back",
        aperture_radius=1.0,
        z=0.35,
        curvature_radius=-8.0,
        material_normal_side=air,
        material_opposite_side=glass,
    )
    image = ImagePlane("image", aperture_radius=5.0, z=8.0)
    system = OpticalSystem(
        "symmetric-lens-demo",
        [front, back, image],
        notes="Ideal surfaces; no Fresnel loss, coating, diffraction or manufacturing tolerance.",
    )
    rays = fan_rays(
        origins_y=np.linspace(-0.7, 0.7, 5),
        angles_rad=np.linspace(-0.03, 0.03, 3),
        z=-2.0,
    )
    config = TraceConfig(
        max_path_length=20.0,
        max_interactions=32,
        grazing_cos_threshold=1e-4,
    )
    records = trace_batch(rays, system.surfaces, config)
    engineering = Engineering("demo-fan", rays=rays, notes=system.notes)
    engineering.save_json("engineering.json")
    save_records_json(records, "trace-results.json")

    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    reached = [record for record in records if record.reached_image]
    ys = np.array([record.final_point[1] for record in reached])
    print(f"system: {system.name}")
    print(f"rays: {len(records)}; status counts: {counts}")
    if len(ys):
        print(f"image y range: {ys.min():+.6f} to {ys.max():+.6f}")


if __name__ == "__main__":
    main()
