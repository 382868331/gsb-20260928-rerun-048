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
    TraceSettings,
    load_batch,
)
from raytrace.materials import fresnel_coefficients


def make_materials():
    materials = MaterialLibrary()
    materials.add(Material("glass", 1.5))
    return materials


def splitter_project(settings=None):
    """Plane beam splitter at 45 deg incidence plus two detectors."""
    surfaces = [
        PlaneSurface(
            "splitter",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            kind="refract",
            aperture=CircularAperture(5.0),
            back_material="glass",
        ),
        PlaneSurface(
            "forward_detector",
            (3.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            kind="detector",
            aperture=CircularAperture(5.0),
        ),
        PlaneSurface(
            "return_detector",
            (0.0, 3.0, 0.0),
            (0.0, 1.0, 0.0),
            kind="detector",
            aperture=CircularAperture(5.0),
        ),
    ]
    return Project("splitter", surfaces, make_materials(), settings or TraceSettings())


class BeamSplittingTests(unittest.TestCase):
    def test_reflected_and_transmitted_branches_both_traced(self):
        project = splitter_project()
        direction = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
        batch = project.trace_rays([Ray((-1.0, 0.0, 0.0), direction)])
        beams = batch.beams[0]

        self.assertEqual(len(beams), 2)
        root, child = beams
        self.assertEqual(root.branch, "root")
        self.assertIsNone(root.parent_id)
        self.assertEqual(child.branch, "reflected")
        self.assertEqual(child.parent_id, root.beam_id)
        self.assertEqual(child.generation, 1)

        self.assertEqual(root.status, "reached_image")
        self.assertEqual(root.events[-1]["surface"], "forward_detector")
        self.assertEqual(child.status, "reached_image")
        self.assertEqual(child.events[-1]["surface"], "return_detector")

        # Reflected direction is mirrored about the surface normal (x axis).
        reflected_dir = child.events[-1]["final_direction"]
        self.assertLess(reflected_dir[0], 0.0)
        self.assertGreater(reflected_dir[1], 0.0)
        self.assertAlmostEqual(child.endpoint[1], 3.0)
        self.assertLess(child.endpoint[0], 0.0)

        reflected_fraction, transmitted_fraction = fresnel_coefficients(1.0, 1.5, math.sqrt(0.5))
        self.assertAlmostEqual(child.weight, reflected_fraction)
        self.assertAlmostEqual(root.weight, transmitted_fraction)

    def test_energy_report_books_everything_by_source(self):
        project = splitter_project()
        direction = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
        rays = [Ray((-1.0, 0.0, 0.0), direction), Ray((-1.0, -0.5, 0.0), direction)]
        report = project.trace_rays(rays).energy_report()

        self.assertAlmostEqual(report["input_energy"], 2.0)
        self.assertAlmostEqual(report["absorbed_energy"], 0.0)
        self.assertAlmostEqual(report["residual_energy"], 0.0)
        self.assertAlmostEqual(report["escaped_energy"], 0.0)
        self.assertAlmostEqual(report["grazing_energy"], 0.0)
        self.assertAlmostEqual(report["unaccounted_energy"], 0.0)
        self.assertAlmostEqual(report["detected_energy"], 2.0)

        forward = report["detectors"]["forward_detector"]
        returned = report["detectors"]["return_detector"]
        self.assertEqual(forward["hits"], 2)
        self.assertEqual(returned["hits"], 2)
        self.assertEqual(len(forward["points"]), 2)
        for source in ("0", "1"):
            self.assertIn(source, forward["by_source"])
            self.assertIn(source, returned["by_source"])
            total = forward["by_source"][source] + returned["by_source"][source]
            self.assertAlmostEqual(total, 1.0)

    def test_tir_does_not_split_and_keeps_full_weight(self):
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
        batch = Project("tir", surfaces, make_materials()).trace_rays([Ray((0.1, 0, 0), direction)])

        self.assertEqual(len(batch.beams[0]), 1)
        beam = batch.beams[0][0]
        self.assertEqual(beam.events[-1]["action"], "total_internal_reflection")
        self.assertAlmostEqual(beam.weight, 1.0)
        report = batch.energy_report()
        self.assertAlmostEqual(report["escaped_energy"], 1.0)
        self.assertAlmostEqual(report["absorbed_energy"], 0.0)

    def test_aperture_edge_goes_to_grazing_bucket_not_absorbed(self):
        surfaces = [
            PlaneSurface(
                "edge",
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="refract",
                aperture=CircularAperture(1.0),
                back_material="glass",
            ),
        ]
        batch = Project("edge", surfaces, make_materials()).trace_rays([Ray((0, 1, 0), (1, 0, 0))])
        report = batch.energy_report()

        self.assertEqual(batch.beams[0][0].status, "grazing")
        self.assertAlmostEqual(report["grazing_energy"], 1.0)
        self.assertAlmostEqual(report["absorbed_energy"], 0.0)
        self.assertAlmostEqual(report["unaccounted_energy"], 0.0)

    def test_repeated_reflection_stops_at_min_weight_without_absorbing_residual(self):
        surfaces = [
            PlaneSurface("left", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="mirror",
                         aperture=CircularAperture(1.0), reflectivity=0.5),
            PlaneSurface("right", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), kind="mirror",
                         aperture=CircularAperture(1.0), reflectivity=0.5),
        ]
        settings = TraceSettings(min_weight=0.1)
        batch = Project("cavity", surfaces, MaterialLibrary(), settings).trace_rays(
            [Ray((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))])
        beam = batch.beams[0][0]

        self.assertEqual(beam.status, "min_weight")
        self.assertAlmostEqual(beam.weight, 0.0625)
        report = batch.energy_report()
        self.assertAlmostEqual(report["absorbed_energy"], 0.9375)
        self.assertAlmostEqual(report["residual_energy"], 0.0625)
        self.assertAlmostEqual(report["unaccounted_energy"], 0.0)

    def test_max_events_residual_is_not_absorbed(self):
        surfaces = [
            PlaneSurface("left", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="mirror",
                         aperture=CircularAperture(1.0), reflectivity=0.99),
            PlaneSurface("right", (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), kind="mirror",
                         aperture=CircularAperture(1.0), reflectivity=0.99),
        ]
        settings = TraceSettings(max_events=6, min_weight=0.0)
        batch = Project("cavity", surfaces, MaterialLibrary(), settings).trace_rays(
            [Ray((0.5, 0.0, 0.0), (1.0, 0.0, 0.0))])
        beam = batch.beams[0][0]

        self.assertEqual(beam.status, "max_events")
        report = batch.energy_report()
        self.assertAlmostEqual(report["residual_energy"], beam.weight)
        self.assertAlmostEqual(report["absorbed_energy"], 1.0 - beam.weight)
        self.assertAlmostEqual(report["unaccounted_energy"], 0.0)

    def test_saved_project_recomputes_identical_beam_tree(self):
        settings = TraceSettings(min_weight=0.05)
        project = splitter_project(settings)
        direction = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
        rays = [Ray((-1.0, 0.0, 0.0), direction)]
        batch = project.trace_rays(rays)

        with tempfile.TemporaryDirectory() as directory:
            project_path = f"{directory}/project.json"
            batch_path = f"{directory}/batch.json"
            project.save(project_path)
            batch.save(batch_path)
            recomputed = Project.load(project_path).trace_rays(rays)
            reloaded = load_batch(batch_path)

        for other in (recomputed, reloaded):
            self.assertEqual(len(other.beams), 1)
            self.assertEqual(len(other.beams[0]), len(batch.beams[0]))
            for expected, actual in zip(batch.beams[0], other.beams[0]):
                self.assertEqual(expected.beam_id, actual.beam_id)
                self.assertEqual(expected.parent_id, actual.parent_id)
                self.assertEqual(expected.branch, actual.branch)
                self.assertEqual(expected.status, actual.status)
                self.assertEqual(expected.generation, actual.generation)
                for a, b in zip(expected.endpoint, actual.endpoint):
                    self.assertAlmostEqual(a, b)
                self.assertAlmostEqual(expected.weight, actual.weight)
                self.assertEqual(len(expected.events), len(actual.events))


if __name__ == "__main__":
    unittest.main()
