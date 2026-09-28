import math

import numpy as np

from raytrace import (
    Air,
    ConstantMaterial,
    Engineering,
    ImagePlane,
    Mirror,
    OpticalSystem,
    PlaneBoundary,
    PlaneInterface,
    Ray,
    SphericalInterface,
    TraceConfig,
    trace_ray,
)
from raytrace.tracer import load_records_json, save_records_json, trace_batch


AIR = Air()
GLASS = ConstantMaterial("glass", 1.5)


def vec(value):
    return np.asarray(value, dtype=float)


def test_reverse_ray_reconstructs_original_direction():
    enter = PlaneInterface(
        "enter", 2.0, 0.0,
        material_normal_side=GLASS, material_opposite_side=AIR,
    )
    exit_surface = PlaneInterface(
        "exit", 2.0, 1.0,
        material_normal_side=GLASS, material_opposite_side=AIR,
    )
    forward = Ray((0, 0, -1), (0, 0.3, 1))
    forward_record = trace_ray(forward, [enter, exit_surface], TraceConfig(max_path_length=5))
    second = forward_record.events[1]
    reverse_origin = vec(second["point"]) + vec(forward_record.final_direction) * 1e-6
    reverse = Ray(reverse_origin, -vec(forward_record.final_direction))
    reverse_record = trace_ray(reverse, [enter, exit_surface], TraceConfig(max_path_length=5))
    np.testing.assert_allclose(reverse_record.final_direction, -forward.direction, atol=1e-9)


def test_nearest_surface_is_selected():
    near = PlaneBoundary("near", z=1.0)
    far = PlaneBoundary("far", z=2.0)
    record = trace_ray(Ray((0, 0, 0), (0, 0, 1)), [far, near], TraceConfig())
    assert record.status == "absorbed"
    assert [event["surface"] for event in record.events] == ["near"]


def test_flat_mirror_reflects_direction():
    mirror = Mirror("mirror", z=-1.0)
    direction = vec((0, 0.3, -1))
    direction /= np.linalg.norm(direction)
    record = trace_ray(Ray((0, 0, 0), direction), [mirror], TraceConfig())
    assert record.events[0]["kind"] == "reflect"
    assert np.isclose(record.final_direction[2], 1.0 / math.hypot(0.3, 1), atol=1e-12)
    assert np.isclose(record.final_direction[1], 0.3 / math.hypot(0.3, 1), atol=1e-12)


def test_total_internal_reflection_is_retained_as_distinct_status():
    surface = PlaneInterface(
        "glass-air", 3.0, 1.0,
        material_normal_side=AIR, material_opposite_side=GLASS,
    )
    ray = Ray((0, 0, 0), (0, 1, 1), wavelength_um=0.5876)
    record = trace_ray(ray, [surface], TraceConfig(max_path_length=5.0))
    assert record.status == "total_internal_reflection"
    assert record.final_intensity == 1.0
    assert record.events[-1]["kind"] == "total_internal_reflection"


def test_grazing_ray_is_not_forced_into_refraction():
    surface = PlaneInterface(
        "grazed", 200.0, 1.0e-5,
        material_normal_side=GLASS, material_opposite_side=AIR,
    )
    ray = Ray((0, 0, 0), (0, 1.0, 1.0e-7))
    record = trace_ray(
        ray, [surface], TraceConfig(grazing_cos_threshold=1e-4, max_path_length=1000.0)
    )
    assert record.status == "grazing"
    assert record.events[-1]["kind"] == "grazing"


def test_ray_missing_clear_hole_is_absorbed_by_stop():
    surface = PlaneBoundary(
        "stop-with-hole",
        aperture_radius=0.1,
        clear_radius=0.05,
        z=1.0,
    )
    image = ImagePlane("image", z=2.0)
    record = trace_ray(Ray((0, 0.08, 0), (0, 0, 1)), [surface, image], TraceConfig())
    assert record.status == "absorbed"

    passed = trace_ray(Ray((0, 0.0, 0), (0, 0, 1)), [surface, image], TraceConfig())
    assert passed.status == "reached_image"


def test_symmetric_lens_preserves_mirror_symmetry():
    front = SphericalInterface(
        "front", 1.0, z=0.0, curvature_radius=10.0,
        material_normal_side=GLASS, material_opposite_side=AIR,
    )
    back = SphericalInterface(
        "back", 1.0, z=0.4, curvature_radius=-10.0,
        material_normal_side=AIR, material_opposite_side=GLASS,
    )
    image = ImagePlane("image", z=10.0)
    records = trace_batch(
        [
            Ray((0, 0.2, -1), (0, 0, 1), ray_id="upper"),
            Ray((0, -0.2, -1), (0, 0, 1), ray_id="lower"),
        ],
        [front, back, image],
        TraceConfig(),
    )
    assert all(record.status == "reached_image" for record in records)
    assert np.isclose(records[0].final_point[1], -records[1].final_point[1], atol=1e-12)
    assert np.isclose(records[0].final_direction[1], -records[1].final_direction[1], atol=1e-12)


def test_engineering_batch_round_trip(tmp_path):
    engineering = Engineering(
        "batch",
        rays=[Ray((0, 0, 0), (0, 0, 1), ray_id="one")],
        notes="reopen me",
    )
    path = tmp_path / "batch.json"
    engineering.save_json(path)
    reopened = Engineering.load_json(path)
    assert reopened.notes == "reopen me"
    assert reopened.rays[0].ray_id == "one"
    np.testing.assert_allclose(reopened.rays[0].direction, [0, 0, 1])


def test_trace_result_batch_round_trip(tmp_path):
    image = ImagePlane("image", z=1.0)
    record = trace_ray(Ray((0, 0, 0), (0, 0, 1), ray_id="x"), [image], TraceConfig())
    path = tmp_path / "records.json"
    save_records_json([record], path)
    reopened = load_records_json(path)
    assert reopened[0].ray_id == "x"
    assert reopened[0].status == "reached_image"


def test_optical_system_requires_unique_surface_names():
    import pytest

    with pytest.raises(ValueError):
        OpticalSystem("duplicate", [PlaneBoundary("a", z=1), PlaneBoundary("a", z=2)])
