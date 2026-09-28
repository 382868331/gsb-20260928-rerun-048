import math
import tempfile
import unittest

from raytrace import (
    CircularAperture,
    Material,
    MaterialLibrary,
    PlaneSurface,
    Project,
    Ray,
    SphericalSurface,
    TraceSettings,
)


class RayTraceTests(unittest.TestCase):
    def setUp(self):
        self.materials = MaterialLibrary()
        self.materials.add(Material("glass", 1.5))

    def test_plane_refraction_direction_and_energy(self):
        surfaces = [
            PlaneSurface(
                "front",
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="refract",
                aperture=CircularAperture(2.0),
                back_material="glass",
            ),
            PlaneSurface(
                "back",
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="refract",
                aperture=CircularAperture(2.0),
                front_material="glass",
            ),
            PlaneSurface(
                "image",
                (2.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="detector",
                aperture=CircularAperture(2.0),
            ),
        ]
        result = Project("slab", surfaces, self.materials).trace_rays([Ray((-1, 0, 0), (1, 0, 0))]).results[0]

        self.assertEqual(result.status, "reached_image")
        self.assertEqual(result.endpoint, (2.0, 0.0, 0.0))
        self.assertAlmostEqual(result.weight, 0.9216, places=10)
        self.assertTrue(all(event["outgoing_direction"] == (1.0, 0.0, 0.0) for event in result.events[:2]))

    def test_symmetric_lens_bends_ray_toward_axis(self):
        surfaces = [
            SphericalSurface(
                "front",
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                radius=10.0,
                kind="refract",
                aperture=CircularAperture(2.0),
                back_material="glass",
            ),
            SphericalSurface(
                "back",
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                radius=-10.0,
                kind="refract",
                aperture=CircularAperture(2.0),
                front_material="glass",
            ),
            PlaneSurface(
                "image",
                (5.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="detector",
                aperture=CircularAperture(3.0),
            ),
        ]
        result = Project("lens", surfaces, self.materials).trace_rays([Ray((-1, 0.5, 0), (1, 0, 0))]).results[0]

        self.assertEqual(result.status, "reached_image")
        self.assertLess(result.endpoint[1], 0.5)
        self.assertLess(result.events[1]["outgoing_direction"][1], 0.0)
        self.assertLessEqual(result.weight, 1.0)

    def test_mirror_reflection(self):
        surfaces = [
            PlaneSurface(
                "mirror",
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="mirror",
                aperture=CircularAperture(1.0),
                reflectivity=0.8,
            ),
            PlaneSurface(
                "return_detector",
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="detector",
                aperture=CircularAperture(1.0),
            ),
        ]
        result = Project("mirror", surfaces).trace_rays([Ray((0, 0, 0), (1, 0, 0))]).results[0]

        self.assertEqual(result.status, "reached_image")
        self.assertAlmostEqual(result.endpoint[0], -1.0)
        self.assertEqual(result.endpoint[1:], (0.0, 0.0))
        self.assertAlmostEqual(result.weight, 0.8)
        self.assertEqual(result.events[0]["outgoing_direction"], (-1.0, 0.0, 0.0))

    def test_total_internal_reflection_is_retained_without_image(self):
        surfaces = [
            PlaneSurface(
                "back",
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="refract",
                aperture=CircularAperture(3.0),
                front_material="glass",
                back_material="air",
            ),
        ]
        direction = (1.0 / math.sqrt(5.0), 2.0 / math.sqrt(5.0), 0.0)
        result = Project("tir", surfaces, self.materials).trace_rays([Ray((0.1, 0, 0), direction)]).results[0]

        self.assertEqual(result.status, "no_image")
        self.assertEqual(result.events[-1]["action"], "total_internal_reflection")
        self.assertAlmostEqual(result.weight, 1.0)
        self.assertLess(result.events[-1]["outgoing_direction"][0], 0.0)
        self.assertGreater(result.events[-1]["outgoing_direction"][1], 0.0)

    def test_aperture_edge_is_classified_as_grazing(self):
        surfaces = [
            PlaneSurface(
                "edge",
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="refract",
                aperture=CircularAperture(1.0),
                back_material="glass",
            ),
            PlaneSurface("image", (2.0, 0.0, 0.0), (1, 0, 0), kind="detector", aperture=CircularAperture(2)),
        ]
        result = Project("edge", surfaces, self.materials).trace_rays([Ray((0, 1, 0), (1, 0, 0))]).results[0]

        self.assertEqual(result.status, "grazing")
        self.assertTrue(result.events[-1]["aperture_edge"])

    def test_project_and_batch_roundtrip(self):
        surfaces = [
            PlaneSurface("image", (1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(1)),
        ]
        project = Project("roundtrip", surfaces, self.materials, TraceSettings(max_events=8))
        batch = project.trace_rays([Ray((0, 0, 0), (1, 0, 0))])

        with tempfile.TemporaryDirectory() as directory:
            project_path = f"{directory}/project.json"
            batch_path = f"{directory}/batch.json"
            project.save(project_path)
            batch.save(batch_path)
            reopened_project = Project.load(project_path)
            reopened_batch = Project.load(project_path).trace_rays([Ray((0, 0, 0), (1, 0, 0))])

        self.assertEqual(reopened_project.settings.max_events, 8)
        self.assertEqual(reopened_project.surfaces[0].identifier, "image")
        self.assertEqual(batch.status_summary(), reopened_batch.status_summary())


if __name__ == "__main__":
    unittest.main()
