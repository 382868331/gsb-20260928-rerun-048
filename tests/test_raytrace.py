import math
import json
import tempfile
import unittest
from pathlib import Path

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


def _combo_surfaces(materials):
    c = math.cos(math.radians(30))
    s = math.sin(math.radians(30))
    wedge = (1 / math.sqrt(2), 1 / math.sqrt(2), 0.0)
    edge_radius = 2.0 / math.sqrt(3.0)
    return [
        PlaneSurface(
            "splitter",
            (0.0, 0.0, 0.0),
            (c, s, 0.0),
            kind="beamsplitter",
            aperture=CircularAperture(5.0),
            reflectivity=0.04,
        ),
        PlaneSurface(
            "clip",
            (0.0, -2.0, 0.0),
            (0.0, -1.0, 0.0),
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
            (3.0, -2.0, 0.0),
            (0.0, -1.0, 0.0),
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
        # reflectivity is the reflected fraction, same meaning as a mirror
        self.assertAlmostEqual(transmitted.weight, 0.7)
        self.assertAlmostEqual(reflected.weight, 0.3)
        self.assertEqual(transmitted.parent_branch_id, "root")
        self.assertEqual(reflected.source_label, "src")
        self.assertEqual(transmitted.events[-1]["action"], "land")
        self.assertEqual(reflected.events[0]["action"], "split")
        split_event = reflected.events[0]
        self.assertAlmostEqual(split_event["reflectivity"], 0.3)
        self.assertAlmostEqual(split_event["transmissivity"], 0.7)
        self.assertAlmostEqual(split_event["absorbed_power"], 0.0)
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
        self.assertEqual(tuple(round(v, 9) for v in tir_leaf.events[2]["outgoing_direction"]), (0.0, -1.0, 0.0))
        self.assertEqual(tuple(round(v, 9) for v in tir_leaf.endpoint), (3.0, -2.0, 0.0))

        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["detected"] + ledger["absorbed"] + ledger["grazing"]
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

    def test_zero_reflectivity_splitter_is_transparent(self):
        surfaces = [
            PlaneSurface("split", (0, 0, 0), (1, 0, 0), kind="beamsplitter",
                         aperture=CircularAperture(2.0), reflectivity=0.0),
            PlaneSurface("image", (1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
        ]
        batch = Project("split", surfaces).trace_rays([Ray((-0.5, 0, 0), (1, 0, 0), label="src")])
        self.assertEqual(set(leaf.branch_id for leaf in batch.results), {"root.1.t"})
        leaf = batch.results[0]
        self.assertAlmostEqual(leaf.weight, 1.0)
        split_event = leaf.events[0]
        self.assertEqual(split_event["action"], "split")
        self.assertAlmostEqual(split_event["reflected_power"], 0.0)
        self.assertAlmostEqual(split_event["transmitted_power"], 1.0)
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["detected"], 1.0)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0)

    def test_full_reflectivity_splitter_reflects_all(self):
        surfaces = [
            PlaneSurface("split", (0, 0, 0), (1, 0, 0), kind="beamsplitter",
                         aperture=CircularAperture(2.0), reflectivity=1.0),
            PlaneSurface("det_t", (1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
            PlaneSurface("det_r", (-1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
        ]
        batch = Project("split", surfaces).trace_rays([Ray((-0.5, 0, 0), (1, 0, 0), label="src")])
        self.assertEqual(set(leaf.branch_id for leaf in batch.results), {"root.1.r"})
        leaf = batch.results[0]
        self.assertEqual(leaf.detector, "det_r")
        self.assertAlmostEqual(leaf.weight, 1.0)
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["detected"], 1.0)
        self.assertAlmostEqual(ledger["absorbed"], 0.0)

    def test_partial_splitter_loss_closes_the_energy_account(self):
        surfaces = [
            PlaneSurface("split", (0, 0, 0), (1, 0, 0), kind="beamsplitter",
                         aperture=CircularAperture(2.0),
                         reflectivity=0.6, transmissivity=0.3),
            PlaneSurface("det_t", (1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
            PlaneSurface("det_r", (-1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
        ]
        batch = Project("lossy", surfaces).trace_rays([Ray((-0.5, 0, 0), (1, 0, 0), label="src")])
        leaves = {leaf.branch_id: leaf for leaf in batch.results}
        self.assertAlmostEqual(leaves["root.1.t"].weight, 0.3)
        self.assertAlmostEqual(leaves["root.1.r"].weight, 0.6)
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["detected"], 0.9)
        self.assertAlmostEqual(ledger["absorbed"], 0.1)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0)

    def test_zero_reflectivity_mirror_absorbs_everything(self):
        surfaces = [
            PlaneSurface("m", (1, 0, 0), (1, 0, 0), kind="mirror",
                         aperture=CircularAperture(2.0), reflectivity=0.0),
        ]
        batch = Project("black", surfaces).trace_rays([Ray((0, 0, 0), (1, 0, 0), label="src")])
        leaf = batch.results[0]
        self.assertEqual(leaf.status, "absorbed")
        self.assertAlmostEqual(leaf.weight, 1.0)
        self.assertAlmostEqual(leaf.events[-1]["absorbed_power"], 1.0)
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["absorbed"], 1.0)
        self.assertAlmostEqual(ledger["detected"], 0.0)
        self.assertAlmostEqual(ledger["stopped"], 0.0)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0)

    def test_legacy_project_convention_is_migrated_audibly(self):
        surfaces = [
            PlaneSurface("split", (0, 0, 0), (1, 0, 0), kind="beamsplitter",
                         aperture=CircularAperture(2.0), reflectivity=0.8),
            PlaneSurface("det_t", (1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
            PlaneSurface("det_r", (-1, 0, 0), (1, 0, 0), kind="detector", aperture=CircularAperture(2.0)),
        ]
        project = Project("legacy", surfaces)
        legacy_dict = project.to_dict()
        # Simulate a file saved before the convention unification.
        del legacy_dict["convention"]
        migrated = Project.from_dict(legacy_dict)
        splitter = next(s for s in migrated.surfaces if s.identifier == "split")
        self.assertAlmostEqual(splitter.reflectivity, 0.2)
        self.assertAlmostEqual(splitter.transmissivity, 0.8)
        self.assertTrue(any("split" in note for note in migrated.migration_notes))
        # The original experiment interpretation is preserved: forward beam
        # still carries 0.8 of the energy, reflected beam 0.2.
        batch = migrated.trace_rays([Ray((-0.5, 0, 0), (1, 0, 0), label="src")])
        leaves = {leaf.branch_id: leaf for leaf in batch.results}
        self.assertAlmostEqual(leaves["root.1.t"].weight, 0.8)
        self.assertAlmostEqual(leaves["root.1.r"].weight, 0.2)
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["detected"], 1.0)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0)

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/legacy.json"
            Path(path).write_text(json.dumps(legacy_dict), encoding="utf-8")
            with self.assertWarns(UserWarning):
                Project.load(path)

    def test_branch_energy_flow_traces_every_direction_and_merging_sources(self):
        # Two splitters at 45 degrees send opposite branches onto one shared
        # detector area: merged hits must remain distinguishable by source
        # and branch id with full event ancestry.
        diagonal = (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0)
        surfaces = [
            PlaneSurface("s1", (0.0, 0.0, 0.0), diagonal, kind="beamsplitter",
                         aperture=CircularAperture(5.0), reflectivity=0.3),
            PlaneSurface("s2", (2.0, 0.0, 0.0), diagonal, kind="beamsplitter",
                         aperture=CircularAperture(5.0), reflectivity=0.3),
            PlaneSurface("det_merge", (1.0, -1.0, 0.0), (0.0, -1.0, 0.0),
                         kind="detector", aperture=CircularAperture(5.0)),
            PlaneSurface("det_end", (4.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         kind="detector", aperture=CircularAperture(5.0)),
        ]
        rays = [
            Ray((-1.0, 0.2, 0.0), (1.0, 0.0, 0.0), label="a"),
            Ray((-1.0, -0.2, 0.0), (1.0, 0.0, 0.0), label="b"),
        ]
        batch = Project("merge", surfaces).trace_rays(rays)
        flows = batch.branch_energy_flow()
        by_id = {(flow["source_label"], flow["branch_id"]): flow for flow in flows}

        flow_r = by_id[("a", "root.1.r")]
        self.assertEqual(flow_r["detector"], "det_merge")
        self.assertAlmostEqual(flow_r["terminal_power"], 0.3)
        split_step = flow_r["steps"][0]
        self.assertEqual(split_step["surface"], "s1")
        self.assertAlmostEqual(split_step["reflected_power"], 0.3)
        self.assertAlmostEqual(split_step["transmitted_power"], 0.7)

        flow_tt = by_id[("a", "root.1.t.2.t")]
        self.assertEqual(flow_tt["detector"], "det_end")
        self.assertAlmostEqual(flow_tt["terminal_power"], 0.7 * 0.7, places=12)
        self.assertEqual([step["surface"] for step in flow_tt["steps"][:2]], ["s1", "s2"])

        flow_tr = by_id[("a", "root.1.t.2.r")]
        self.assertEqual(flow_tr["detector"], "det_merge")
        self.assertAlmostEqual(flow_tr["terminal_power"], 0.7 * 0.3, places=12)

        merged = batch.detector_summary()["det_merge"]
        branch_keys = {(hit["source_label"], hit["branch_id"]) for hit in merged["branches"]}
        self.assertIn(("a", "root.1.r"), branch_keys)
        self.assertIn(("a", "root.1.t.2.r"), branch_keys)
        self.assertIn(("b", "root.1.r"), branch_keys)
        self.assertIn(("b", "root.1.t.2.r"), branch_keys)
        self.assertEqual(merged["count"], len(merged["branches"]))
        self.assertAlmostEqual(
            merged["power"],
            sum(hit["power"] for hit in merged["branches"]),
        )

        ledger_a = batch.energy_ledger({"a": 1.0, "b": 1.0})["a"]
        total_a = (ledger_a["detected"] + ledger_a["absorbed"] + ledger_a["grazing"]
                   + ledger_a["unreached"] + ledger_a["stopped"])
        self.assertAlmostEqual(total_a, 1.0, places=9)
        self.assertAlmostEqual(ledger_a["unaccounted"], 0.0)

    def test_spectral_ray_keeps_independent_wavelength_branches(self):
        surfaces = [
            PlaneSurface("enter", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="refract",
                         aperture=CircularAperture(5.0), back_material="glass"),
            PlaneSurface("back", (3.0, 0.0, 0.0),
                         (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0.0),
                         kind="refract", aperture=CircularAperture(5.0),
                         front_material="glass"),
            PlaneSurface("det_reflected", (3.0, -2.0, 0.0), (0.0, -1.0, 0.0),
                         kind="detector", aperture=CircularAperture(2.0)),
            PlaneSurface("det_transmitted", (5.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         kind="detector", aperture=CircularAperture(2.0)),
        ]
        glass = Material("glass", 1.5, {450.0: 1.5, 650.0: 1.1})
        library = MaterialLibrary()
        library.add(glass)
        ray = Ray.spectral((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [450.0, 650.0], label="src")
        batch = Project("dispersive", surfaces, library).trace_rays([ray])

        wavelengths = {leaf.wavelength_nm for leaf in batch.results}
        self.assertEqual(wavelengths, {450.0, 650.0})
        self.assertTrue(
            all(leaf.branch_id.startswith("root[wavelength=450]")
                for leaf in batch.results if leaf.wavelength_nm == 450.0)
        )
        tir = next(
            leaf for leaf in batch.results
            if leaf.wavelength_nm == 450.0
            and any(event.get("action") == "total_internal_reflection" for event in leaf.events)
        )
        transmitted = next(
            leaf for leaf in batch.results
            if leaf.wavelength_nm == 650.0 and leaf.detector == "det_transmitted"
        )
        tir_event = next(event for event in tir.events if event["action"] == "total_internal_reflection")
        transmission_event = next(event for event in transmitted.events if event["surface"] == "back")
        self.assertEqual(tir_event["n_incident"], 1.5)
        self.assertEqual(transmission_event["n_incident"], 1.1)
        self.assertNotEqual(tir.branch_id, transmitted.branch_id)

        summary = batch.detector_summary()
        self.assertIn("450 nm", summary["det_reflected"]["power_by_wavelength"])
        self.assertIn("650 nm", summary["det_transmitted"]["power_by_wavelength"])
        self.assertIn("450 nm", summary["det_reflected"]["power_by_source_wavelength"]["src"])
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["unaccounted"], 0.0)
        self.assertIn("450 nm", ledger["by_wavelength"])
        self.assertIn("650 nm", ledger["by_wavelength"])

    def test_unverified_wavelength_is_marked_unknown_without_interpolation(self):
        surfaces = [
            PlaneSurface("enter", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="refract",
                         aperture=CircularAperture(2.0), back_material="glass"),
            PlaneSurface("image", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         kind="detector", aperture=CircularAperture(2.0)),
        ]
        library = MaterialLibrary()
        library.add(Material("glass", 1.5, {450.0: 1.48}))
        ray = Ray.spectral((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [450.0, 650.0], label="src")
        batch = Project("strict", surfaces, library).trace_rays([ray])

        unknown = [leaf for leaf in batch.results if leaf.status == "unknown_response"]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0].wavelength_nm, 650.0)
        self.assertEqual(unknown[0].events[-1]["action"], "unknown_material_response")
        self.assertAlmostEqual(unknown[0].weight, 0.5)
        ledger = batch.energy_ledger({"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["unknown_response"], 0.5)
        self.assertAlmostEqual(ledger["by_wavelength"]["650 nm"]["unknown_response"], 0.5)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0)

    def test_wavelength_response_comparison_keeps_loss_categories_separate(self):
        surfaces = [
            PlaneSurface("split", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="beamsplitter",
                         aperture=CircularAperture(2.0), reflectivity=0.2,
                         transmissivity=0.7),
            PlaneSurface("enter", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="refract",
                         aperture=CircularAperture(1.0), back_material="glass"),
            PlaneSurface("image", (3.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         kind="detector", aperture=CircularAperture(2.0)),
        ]
        before_library = MaterialLibrary()
        before_library.add(Material("glass", 1.5))
        after_library = MaterialLibrary()
        after_library.add(Material("glass", 1.5, {450.0: 1.5, 650.0: 1.1}))
        ray = Ray.spectral((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [450.0, 650.0], label="src")
        before = Project("before", surfaces, before_library).trace_rays([ray])
        after = Project("after", surfaces, after_library).trace_rays([ray])

        comparison = before.compare_wavelength_response(after, {"src": 1.0})
        self.assertIn("image", comparison["detectors"])
        self.assertIn("650 nm", comparison["detectors"]["image"]["by_wavelength"])
        ledger_delta = comparison["energy_ledger"]["src"]["delta"]
        self.assertIn("material_loss", ledger_delta)
        self.assertIn("aperture_grazing", ledger_delta)

    def test_old_batch_remains_unchanged_when_wavelength_scheme_is_added(self):
        surfaces = [
            PlaneSurface("image", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         kind="detector", aperture=CircularAperture(2.0)),
        ]
        project = Project("archive", surfaces, self.materials)
        old_batch = project.trace_rays([Ray((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), label="old")])
        with tempfile.TemporaryDirectory() as directory:
            old_path = Path(directory) / "old_batch.json"
            old_batch.save(old_path)
            reloaded_old = type(old_batch).from_dict(json.loads(old_path.read_text(encoding="utf-8")))
            spectral_ray = Ray.spectral((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), [450.0, 650.0], label="new")
            spectral_batch = project.trace_rays([spectral_ray])
            new_path = Path(directory) / "wavelength_batch.json"
            spectral_batch.save(new_path)

        self.assertEqual(reloaded_old.energy_ledger(), old_batch.energy_ledger())
        self.assertEqual(
            [leaf.branch_id for leaf in reloaded_old.results],
            [leaf.branch_id for leaf in old_batch.results],
        )


class PolarizationTests(unittest.TestCase):
    def setUp(self):
        self.materials = MaterialLibrary()
        self.materials.add(Material("glass", 1.5))

    @staticmethod
    def _polarizing_splitter_project(**splitter_kwargs):
        a = math.sqrt(0.5)
        materials = MaterialLibrary()
        materials.add(Material("glass", 1.5))
        surfaces = [
            PlaneSurface(
                "splitter",
                (0.0, 0.0, 0.0),
                (a, a, 0.0),
                kind="beamsplitter",
                aperture=CircularAperture(5.0),
                **splitter_kwargs,
            ),
            PlaneSurface(
                "det_transmitted",
                (2.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="detector",
                aperture=CircularAperture(3.0),
            ),
            PlaneSurface(
                "det_reflected",
                (0.0, -2.0, 0.0),
                (0.0, -1.0, 0.0),
                kind="detector",
                aperture=CircularAperture(3.0),
            ),
        ]
        return Project("pol-splitter", surfaces, materials)

    def test_incoherent_components_share_not_double_the_input_power(self):
        project = self._polarizing_splitter_project(
            polarization_reflectivity={"s": 0.8, "p": 0.2},
        )
        ray = Ray.polarized((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [0.0, 90.0], label="src")
        batch = project.trace_rays([ray])
        ledger = batch.energy_ledger(input_power={"src": 1.0})["src"]

        self.assertAlmostEqual(ledger["input"], 1.0)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0, places=12)
        detected = batch.detector_summary()
        total_detected = sum(summary["power"] for summary in detected.values())
        self.assertLessEqual(total_detected, 1.0 + 1e-12)
        self.assertAlmostEqual(
            total_detected + ledger["unreached"] + ledger["absorbed"], 1.0, places=12,
        )
        roots = sorted({leaf.branch_id.split(".")[0] for leaf in batch.results})
        self.assertEqual(roots, ["root[pol=0]", "root[pol=90]"])

    def test_rotating_input_polarization_moves_power_between_paths(self):
        project = self._polarizing_splitter_project(
            polarization_reflectivity={"s": 0.9, "p": 0.1},
        )
        # angle 0 lies in the plane of incidence (p), 90 is perpendicular (s)
        batch_p = project.trace_rays([
            Ray.polarized((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [0.0], label="src")
        ])
        batch_s = project.trace_rays([
            Ray.polarized((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [90.0], label="src")
        ])

        summary_p = batch_p.detector_summary()
        summary_s = batch_s.detector_summary()
        self.assertAlmostEqual(summary_p["det_transmitted"]["power"], 0.9, places=12)
        self.assertAlmostEqual(summary_p["det_reflected"]["power"], 0.1, places=12)
        self.assertAlmostEqual(summary_s["det_transmitted"]["power"], 0.1, places=12)
        self.assertAlmostEqual(summary_s["det_reflected"]["power"], 0.9, places=12)

        comparison = batch_p.compare_polarization_response(
            batch_s, input_power={"src": 1.0}
        )
        self.assertAlmostEqual(
            comparison["detectors"]["det_reflected"]["delta"], 0.8, places=12
        )
        self.assertAlmostEqual(
            comparison["detectors"]["det_transmitted"]["delta"], -0.8, places=12
        )
        self.assertAlmostEqual(
            comparison["energy_ledger"]["src"]["delta"]["unaccounted"], 0.0, places=12
        )

    def test_analyzer_rejected_energy_is_reported_separately(self):
        materials = MaterialLibrary()
        surfaces = [
            PlaneSurface(
                "image",
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="detector",
                aperture=CircularAperture(2.0),
                analyzer_angle_deg=0.0,
            ),
        ]
        project = Project("analyzer", surfaces, materials)
        aligned = project.trace_rays([
            Ray.polarized((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), [0.0], label="src")
        ])
        crossed = project.trace_rays([
            Ray.polarized((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), [90.0], label="src")
        ])
        diagonal = project.trace_rays([
            Ray.polarized((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), [45.0], label="src")
        ])

        self.assertAlmostEqual(
            aligned.energy_ledger(input_power={"src": 1.0})["src"]["detected"], 1.0
        )
        crossed_ledger = crossed.energy_ledger(input_power={"src": 1.0})["src"]
        self.assertAlmostEqual(crossed_ledger["detected"], 0.0)
        self.assertAlmostEqual(crossed_ledger["analyzer_rejected"], 1.0)
        self.assertAlmostEqual(crossed_ledger["unaccounted"], 0.0, places=12)
        diagonal_ledger = diagonal.energy_ledger(input_power={"src": 1.0})["src"]
        self.assertAlmostEqual(diagonal_ledger["detected"], 0.5)
        self.assertAlmostEqual(diagonal_ledger["analyzer_rejected"], 0.5)
        self.assertAlmostEqual(diagonal_ledger["unaccounted"], 0.0, places=12)

    def test_tir_and_partial_transmission_coexist_across_wavelengths(self):
        materials = MaterialLibrary()
        materials.add(Material("glass", 1.5, spectral_indices={450.0: 1.8, 650.0: 1.4}))
        cos40 = math.cos(math.radians(40.0))
        sin40 = math.sin(math.radians(40.0))
        cos80 = math.cos(math.radians(80.0))
        sin80 = math.sin(math.radians(80.0))
        surfaces = [
            PlaneSurface(
                "exit",
                (1.0, 0.0, 0.0),
                (cos40, sin40, 0.0),
                kind="refract",
                aperture=CircularAperture(5.0),
                front_material="glass",
            ),
            PlaneSurface(
                "det_tir",
                (1.0 - cos80, -sin80, 0.0),
                (cos80, sin80, 0.0),
                kind="detector",
                aperture=CircularAperture(3.0),
            ),
            PlaneSurface(
                "det_transmitted",
                (3.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                kind="detector",
                aperture=CircularAperture(5.0),
            ),
        ]
        project = Project("spectral-tir", surfaces, materials)
        ray = Ray.spectral(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), [450.0, 650.0], label="src"
        )
        batch = project.trace_rays([ray])

        actions_450 = [
            event["action"]
            for leaf in batch.results
            if leaf.wavelength_nm == 450.0
            for event in leaf.events
        ]
        actions_650 = [
            event["action"]
            for leaf in batch.results
            if leaf.wavelength_nm == 650.0
            for event in leaf.events
        ]
        self.assertIn("total_internal_reflection", actions_450)
        self.assertNotIn("refract", actions_450)
        self.assertIn("refract", actions_650)
        self.assertNotIn("total_internal_reflection", actions_650)

        summary = batch.detector_summary()
        # det_tir also collects the 650 nm Fresnel reflection, which follows
        # the same geometric path as the 450 nm total internal reflection
        cos_i = math.cos(math.radians(40.0))
        cos_t = math.sqrt(1.0 - 1.4 ** 2 * (1.0 - cos_i ** 2))
        r_s = ((1.4 * cos_i - cos_t) / (1.4 * cos_i + cos_t)) ** 2
        r_p = ((1.4 * cos_t - cos_i) / (1.4 * cos_t + cos_i)) ** 2
        expected_tir = 0.5 + 0.5 * 0.5 * (r_s + r_p)
        self.assertAlmostEqual(summary["det_tir"]["power"], expected_tir, places=12)
        self.assertGreater(summary["det_transmitted"]["power"], 0.4)
        ledger = batch.energy_ledger(input_power={"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["unaccounted"], 0.0, places=12)
        for wavelength_account in ledger["by_wavelength"].values():
            self.assertAlmostEqual(wavelength_account["unaccounted"], 0.0, places=12)

    def test_missing_polarization_response_stays_unknown(self):
        project = self._polarizing_splitter_project(
            polarization_reflectivity={"s": 0.5},
        )
        ray = Ray.polarized((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [0.0], label="src")
        batch = project.trace_rays([ray])

        self.assertEqual(batch.status_summary(), {"unknown_response": 1})
        self.assertEqual(batch.detector_hits(), {})
        ledger = batch.energy_ledger(input_power={"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["unknown_response"], 1.0)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0, places=12)
        self.assertEqual(
            batch.results[0].events[-1]["action"], "unknown_polarization_response"
        )

    def test_recombined_branches_feed_one_detector_without_double_counting(self):
        a = math.sqrt(0.5)
        surfaces = [
            PlaneSurface("splitter1", (0.0, 0.0, 0.0), (a, a, 0.0),
                         kind="beamsplitter", aperture=CircularAperture(0.5),
                         reflectivity=0.5),
            PlaneSurface("mirror_top", (3.0, 0.0, 0.0), (a, a, 0.0),
                         kind="mirror", aperture=CircularAperture(0.5)),
            PlaneSurface("mirror_side", (0.0, -2.0, 0.0), (a, a, 0.0),
                         kind="mirror", aperture=CircularAperture(0.5)),
            PlaneSurface("splitter2", (3.0, -2.0, 0.0), (a, a, 0.0),
                         kind="beamsplitter", aperture=CircularAperture(0.5),
                         reflectivity=0.5),
            PlaneSurface("detector", (5.0, -2.0, 0.0), (1.0, 0.0, 0.0),
                         kind="detector", aperture=CircularAperture(3.0)),
        ]
        project = Project("mach-zehnder", surfaces, self.materials)
        batch = project.trace_rays([Ray((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), label="src")])

        summary = batch.detector_summary()["detector"]
        self.assertAlmostEqual(summary["power"], 0.5, places=12)
        self.assertEqual(summary["count"], 2)
        paths = sorted(tuple(hit["path"]) for hit in summary["branches"])
        self.assertEqual(len(paths), 2)
        ledger = batch.energy_ledger(input_power={"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["detected"], 0.5, places=12)
        self.assertAlmostEqual(ledger["unreached"], 0.5, places=12)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0, places=12)

    def test_bulk_loss_is_tracked_separately_from_surface_loss(self):
        materials = MaterialLibrary()
        materials.add(Material("glass", 1.5, bulk_attenuation=0.5))
        surfaces = [
            PlaneSurface("front", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         kind="refract", aperture=CircularAperture(2.0),
                         back_material="glass"),
            PlaneSurface("back", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         kind="refract", aperture=CircularAperture(2.0),
                         front_material="glass"),
            PlaneSurface("image", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         kind="detector", aperture=CircularAperture(2.0)),
        ]
        project = Project("lossy-slab", surfaces, materials)
        batch = project.trace_rays([Ray((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), label="src")])
        ledger = batch.energy_ledger(input_power={"src": 1.0})["src"]

        # each glass pass loses 1 - exp(-0.5); the 4% Fresnel reflection at
        # each exit bounces back for another pass, forming a geometric series
        q = math.exp(-0.5)
        expected_bulk = 0.96 * (1.0 - q) / (1.0 - 0.04 * q)
        self.assertAlmostEqual(ledger["bulk_loss"], expected_bulk, places=7)
        self.assertGreater(ledger["bulk_loss"], 0.0)
        self.assertAlmostEqual(ledger["unaccounted"], 0.0, places=12)
        flow = batch.branch_energy_flow()
        refract_steps = [
            step for entry in flow for step in entry["steps"]
            if step["action"] == "refract"
        ]
        self.assertTrue(any(step["bulk_loss_power"] > 0.0 for step in refract_steps))

    def test_spectral_and_polarized_components_combine_without_extra_energy(self):
        project = self._polarizing_splitter_project(reflectivity=0.3)
        ray = Ray.polarized(
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            [0.0, 90.0],
            label="src",
            spectral_weights=((450.0, 1.0), (650.0, 1.0)),
        )
        batch = project.trace_rays([ray])

        roots = sorted({leaf.branch_id.split(".")[0] for leaf in batch.results})
        self.assertEqual(
            roots,
            [
                "root[wavelength=450][pol=0]",
                "root[wavelength=450][pol=90]",
                "root[wavelength=650][pol=0]",
                "root[wavelength=650][pol=90]",
            ],
        )
        ledger = batch.energy_ledger(input_power={"src": 1.0})["src"]
        self.assertAlmostEqual(ledger["unaccounted"], 0.0, places=12)
        summary = batch.detector_summary()
        # each polarization key aggregates both wavelengths: 2 x 0.25 x R or T
        transmitted = summary["det_transmitted"]["power_by_polarization"]
        reflected = summary["det_reflected"]["power_by_polarization"]
        for key in ("0 deg", "90 deg"):
            self.assertAlmostEqual(transmitted[key], 0.35, places=12)
            self.assertAlmostEqual(reflected[key], 0.15, places=12)

    def test_unpolarized_batch_reopens_with_original_interpretation(self):
        project = self._polarizing_splitter_project(reflectivity=0.3)
        old_batch = project.trace_rays([
            Ray((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), label="old")
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.json"
            old_batch.save(path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            for key in ("polarization", "polarization_angle_deg", "polarization_weights"):
                for ray_dict in raw["rays"]:
                    ray_dict.pop(key, None)
                for result in raw["results"]:
                    result.pop("polarization", None)
                    result["ray"].pop(key, None)
            reloaded = type(old_batch).from_dict(raw)

        self.assertEqual(reloaded.detector_summary(), old_batch.detector_summary())
        self.assertEqual(reloaded.energy_ledger(), old_batch.energy_ledger())
        summary = reloaded.detector_summary()
        self.assertAlmostEqual(
            summary["det_transmitted"]["power_by_polarization"]["unpolarized"],
            0.7,
            places=12,
        )

if __name__ == "__main__":
    unittest.main()
