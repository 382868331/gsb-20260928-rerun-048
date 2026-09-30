import math
import json
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
    load_batch,
)


def _combo_surfaces(materials):
    splitter_axis = (math.sqrt(3.0) / 2.0, -0.5, 0.0)
    wedge = (0.5, math.sqrt(3.0) / 2.0, 0.0)
    edge_radius = 2.0 / math.sqrt(3.0)
    return [
        PlaneSurface(
            "splitter",
            (0.0, 0.0, 0.0),
            splitter_axis,
            kind="beamsplitter",
            aperture=CircularAperture(5.0),
            reflectivity=0.04,
        ),
        PlaneSurface(
            "clip",
            (0.0, 2.0, 0.0),
            (0.0, 1.0, 0.0),
            kind="aperture_stop",
            aperture=CircularAperture(edge_radius, edge_tolerance=1e-10),
        ),
        PlaneSurface(
            "enter",
            (2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            kind="refract",
            aperture=CircularAperture(5.0),
            back_material="glass",
        ),
        PlaneSurface(
            "tir",
            (3.0, 0.0, 0.0),
            wedge,
            kind="refract",
            aperture=CircularAperture(5.0),
            front_material="glass",
        ),
        PlaneSurface(
            "det_tir",
            (4.0, -math.sqrt(3.0), 0.0),
            (0.5, -math.sqrt(3.0) / 2.0, 0.0),
            kind="detector",
            aperture=CircularAperture(1.0),
        ),
    ]


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

    def test_beamsplitter_traces_both_children_with_lineage(self):
        surfaces = [
            PlaneSurface("split", (0, 0, 0), (1, 0, 0), kind="beamsplitter",
                         aperture=CircularAperture(2.0), reflectivity=0.3),
            PlaneSurface("det_t", (1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
            PlaneSurface("det_r", (-1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
        ]
        batch = Project("split", surfaces).trace_rays([Ray((-0.5, 0, 0), (1, 0, 0), label="src")])
        leaves = {leaf.branch_id: leaf for leaf in batch.results}

        self.assertEqual(set(leaves), {"root.1.t", "root.1.r"})
        transmitted = leaves["root.1.t"]
        reflected = leaves["root.1.r"]
        split_event = transmitted.events[0]
        self.assertAlmostEqual(split_event["reflected_fraction"], 0.3)
        self.assertAlmostEqual(split_event["transmitted_fraction"], 0.7)
        self.assertAlmostEqual(transmitted.weight, 0.7)
        self.assertAlmostEqual(reflected.weight, 0.3)
        self.assertEqual(transmitted.parent_branch_id, "root")
        self.assertEqual(reflected.source_label, "src")
        self.assertEqual(transmitted.events[-1]["action"], "land")
        self.assertEqual(reflected.events[0]["action"], "split")
        ledger = batch.energy_ledger({"src": 1.0})
        self.assertAlmostEqual(ledger["src"]["detected"], 1.0)
        self.assertAlmostEqual(ledger["src"]["unaccounted"], 0.0)

    def test_combined_split_tir_and_aperture_grazing(self):
        batch = Project("combo", _combo_surfaces(self.materials), self.materials).trace_rays(
            [Ray((-2.0, 0.0, 0.0), (1.0, 0.0, 0.0), label="src")]
        )
        by_branch = {leaf.branch_id: leaf for leaf in batch.results}

        grazing = by_branch["root.1.r"]
        self.assertEqual(grazing.status, "grazing")
        self.assertEqual(grazing.detector, "clip")
        self.assertAlmostEqual(grazing.weight, 0.04)
        self.assertTrue(grazing.events[-1]["aperture_edge"])
        self.assertAlmostEqual(grazing.endpoint[0], -2.0 / math.sqrt(3.0), places=8)

        tir_leaf = by_branch["root.1.t.2.t"]
        self.assertEqual(tir_leaf.status, "reached_image")
        self.assertEqual(tir_leaf.detector, "det_tir")
        self.assertAlmostEqual(tir_leaf.weight, 0.96 * 0.96, places=10)
        self.assertEqual(tir_leaf.events[2]["action"], "total_internal_reflection")
        outgoing = tir_leaf.events[2]["outgoing_direction"]
        self.assertAlmostEqual(outgoing[0], 0.5)
        self.assertAlmostEqual(outgoing[1], -math.sqrt(3.0) / 2.0)
        self.assertEqual(outgoing[2], 0.0)
        self.assertAlmostEqual(tir_leaf.endpoint[0], 4.0)
        self.assertAlmostEqual(tir_leaf.endpoint[1], -math.sqrt(3.0))
        self.assertEqual(tir_leaf.endpoint[2], 0.0)

        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["detected"] + ledger["absorbed"] + ledger["aperture_clipped"] + ledger["grazing"]
                               + ledger["unreached"] + ledger["stopped"], 1.0, places=9)
        self.assertAlmostEqual(ledger["detected"], 0.9216, places=6)
        self.assertGreaterEqual(ledger["grazing"], 0.04 - 1e-9)
        self.assertGreater(ledger["unreached"], 0.0)
        self.assertAlmostEqual(ledger["absorbed"], 0.0)
        detector = batch.detector_summary()["det_tir"]
        self.assertAlmostEqual(detector["power_by_source"]["src"], detector["power"])

    def test_multiple_sources_are_grouped_independently(self):
        surfaces = [
            PlaneSurface("split", (0, 0, 0), (1, 0, 0), kind="beamsplitter",
                         aperture=CircularAperture(3.0), reflectivity=0.5),
            PlaneSurface("det_t", (1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(3.0)),
            PlaneSurface("det_r", (-1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(3.0)),
        ]
        rays = [
            Ray((-0.5, 0.2, 0), (1, 0, 0), weight=2.0, label="a"),
            Ray((-0.5, -0.2, 0), (1, 0, 0), weight=0.5, label="b"),
        ]
        batch = Project("two", surfaces).trace_rays(rays)
        hits = batch.hits_by_source()
        self.assertEqual({hit["detector"] for hit in hits["a"]}, {"det_t", "det_r"})
        self.assertEqual({hit["detector"] for hit in hits["b"]}, {"det_t", "det_r"})
        ledger = batch.energy_ledger({"a": 2.0, "b": 0.5})
        self.assertAlmostEqual(ledger["a"]["detected"], 2.0)
        self.assertAlmostEqual(ledger["b"]["detected"], 0.5)

    def test_beamsplitter_extremes_keep_only_existing_branch(self):
        surfaces = [
            PlaneSurface("split", (0, 0, 0), (1, 0, 0), kind="beamsplitter",
                         aperture=CircularAperture(2.0), reflectivity=0.0),
            PlaneSurface("det_t", (1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
            PlaneSurface("det_r", (-1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
        ]
        batch = Project("zero-reflection", surfaces).trace_rays([Ray((-0.5, 0, 0), (1, 0, 0), label="src")])
        self.assertEqual([leaf.branch_id for leaf in batch.results], ["root.1.t"])
        leaf = batch.results[0]
        self.assertAlmostEqual(leaf.weight, 1.0)
        self.assertEqual(leaf.events[0]["reflected_power"], 0.0)
        self.assertEqual(leaf.events[0]["transmitted_power"], 1.0)

        surfaces[0] = PlaneSurface("split", (0, 0, 0), (1, 0, 0), kind="beamsplitter",
                                   aperture=CircularAperture(2.0), reflectivity=1.0)
        batch = Project("full-reflection", surfaces).trace_rays([Ray((-0.5, 0, 0), (1, 0, 0), label="src")])
        self.assertEqual([leaf.branch_id for leaf in batch.results], ["root.1.r"])
        leaf = batch.results[0]
        self.assertAlmostEqual(leaf.weight, 1.0)
        self.assertEqual(leaf.events[0]["reflected_power"], 1.0)
        self.assertEqual(leaf.events[0]["transmitted_power"], 0.0)

    def test_zero_reflection_mirror_is_accounted_as_absorbed(self):
        surfaces = [
            PlaneSurface("mirror", (1, 0, 0), (1, 0, 0), kind="mirror",
                         aperture=CircularAperture(1.0), reflectivity=0.0),
        ]
        batch = Project("mirror", surfaces).trace_rays([Ray((0, 0, 0), (1, 0, 0), label="src")])
        self.assertEqual([leaf.status for leaf in batch.results], ["absorbed"])
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["absorbed"], 1.0)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0)

    def test_outside_aperture_is_clipped_and_traced_in_energy_flow(self):
        surfaces = [
            PlaneSurface("small_stop", (1, 0, 0), (1, 0, 0), kind="aperture_stop",
                         aperture=CircularAperture(0.1)),
        ]
        batch = Project("clip", surfaces).trace_rays([Ray((0, 1, 0), (1, 0, 0), label="src")])
        leaf = batch.results[0]
        self.assertEqual(leaf.status, "aperture_clipped")
        self.assertEqual(leaf.detector, "small_stop")
        self.assertEqual(leaf.events[-1]["action"], "aperture_clip")
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["aperture_clipped"], 1.0)
        self.assertAlmostEqual(ledger["detected"], 0.0)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0)
        flow = batch.energy_flows()[0]
        self.assertEqual(flow["path"][-1]["lost_power"], 1.0)

    def test_branches_to_same_detector_remain_separate(self):
        first_axis = (0.5, -math.sqrt(3.0) / 2.0, 0.0)
        second_axis = (0.5, -math.sqrt(3.0) / 2.0, 0.0)
        surfaces = [
            PlaneSurface("s1", (0, 0, 0), first_axis, kind="beamsplitter",
                         aperture=CircularAperture(0.2), reflectivity=0.25),
            PlaneSurface("s2", (2, 0, 0), second_axis, kind="beamsplitter",
                         aperture=CircularAperture(0.2), reflectivity=0.5),
            PlaneSurface("shared", (1, 3, 0), (0, 1, 0), kind="detector",
                         aperture=CircularAperture(4.0)),
        ]
        batch = Project("shared", surfaces).trace_rays([
            Ray((-1, 0, 0), (1, 0, 0), label="src"),
        ])
        expected = {"root.1.r": 0.25, "root.1.t.2.r": 0.375}
        flows = {(flow["source_label"], flow["branch_id"]): flow for flow in batch.detector_flows()["shared"]}
        self.assertEqual(set(flows), {("src", branch) for branch in expected})
        for branch, power in expected.items():
            self.assertAlmostEqual(flows[("src", branch)]["power"], power)
        by_branch = batch.detector_summary()["shared"]["power_by_branch"]["src"]
        self.assertEqual(set(by_branch), set(expected))
        self.assertAlmostEqual(sum(by_branch.values()), 0.625)

    def test_legacy_beamsplitter_project_migration_is_explicit(self):
        data = {
            "name": "legacy",
            "surfaces": [{
                "identifier": "split",
                "kind": "beamsplitter",
                "shape": "plane",
                "vertex": [0, 0, 0],
                "axis": [1, 0, 0],
                "aperture": {"radius": 1.0, "edge_tolerance": 1e-8},
                "front_material": "air",
                "back_material": "air",
                "reflectivity": 0.8,
            }],
            "materials": {"air": {"name": "air", "index": 1.0}},
            "settings": {},
        }
        project = Project.from_dict(data)
        surface = project.surfaces[0]
        self.assertAlmostEqual(surface.reflectivity, 0.2)
        self.assertAlmostEqual(surface.legacy_reflectivity, 0.8)
        self.assertEqual(surface.legacy_reflectivity_convention, "transmission")
        self.assertTrue(any("split" in note for note in project.compatibility_notes))
        saved = json.loads(json.dumps(project.to_dict()))
        self.assertEqual(saved["schema_version"], 2)
        self.assertAlmostEqual(saved["surfaces"][0]["reflectivity"], 0.2)
        self.assertAlmostEqual(saved["surfaces"][0]["legacy_reflectivity"], 0.8)

    def test_legacy_batch_records_are_preserved_without_silent_recalculation(self):
        surfaces = [
            PlaneSurface("split", (0, 0, 0), (1, 0, 0), kind="beamsplitter",
                         aperture=CircularAperture(2.0), reflectivity=0.8),
        ]
        batch = Project("modern", surfaces).trace_rays([Ray((-0.5, 0, 0), (1, 0, 0), label="src")])
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/batch.json"
            batch.save(path)
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["schema_version"] = 1
            payload["results"][0]["weight"] = 0.8
            payload["results"][0]["branch_id"] = "root.1.t"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload))
            old_batch = load_batch(path)
        self.assertEqual(old_batch.schema_version, 1)
        self.assertAlmostEqual(old_batch.results[0].weight, 0.8)
        self.assertTrue(old_batch.compatibility_notes)

    def test_repeated_reflection_stop_is_not_counted_as_absorbed(self):
        surfaces = [
            PlaneSurface("m1", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="mirror",
                         aperture=CircularAperture(2.0), reflectivity=0.9),
            PlaneSurface("m2", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="mirror",
                         aperture=CircularAperture(2.0), reflectivity=0.9),
        ]
        settings = TraceSettings(max_events=10)
        batch = Project("bounce", surfaces, self.materials, settings=settings).trace_rays(
            [Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), label="src")]
        )
        stopped = [leaf for leaf in batch.results if leaf.status == "max_events"]
        self.assertTrue(stopped)
        leaf = stopped[0]
        self.assertEqual(leaf.depth, 10)
        self.assertAlmostEqual(leaf.weight, 0.9**10)
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["stopped"], 0.9**10)
        self.assertAlmostEqual(ledger["absorbed"], 1.0 - 0.9**10)
        self.assertAlmostEqual(ledger["stopped"] + ledger["absorbed"], 1.0)

    def test_saved_project_reproduces_paths_and_terminal_reasons(self):
        project = Project("combo", _combo_surfaces(self.materials), self.materials, TraceSettings(max_events=20))
        rays = [Ray((-2.0, 0.0, 0.0), (1.0, 0.0, 0.0), label="src")]
        batch = project.trace_rays(rays)

        with tempfile.TemporaryDirectory() as directory:
            project_path = f"{directory}/project.json"
            project.save(project_path)
            reopened = Project.load(project_path)
            batch_path = f"{directory}/batch.json"
            batch.save(batch_path)
            recalculated = reopened.trace_rays(rays)

        original = sorted((leaf.branch_id, leaf.status, leaf.detector, leaf.endpoint, leaf.weight)
                          for leaf in batch.results)
        repeated = sorted((leaf.branch_id, leaf.status, leaf.detector, leaf.endpoint, leaf.weight)
                          for leaf in recalculated.results)
        self.assertEqual(original, repeated)
        self.assertEqual(batch.energy_ledger(), recalculated.energy_ledger())


if __name__ == "__main__":
    unittest.main()
