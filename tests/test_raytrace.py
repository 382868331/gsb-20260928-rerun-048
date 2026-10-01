import math
import json
import tempfile
import unittest
from pathlib import Path

from raytrace import (
    Beam,
    CircularAperture,
    Material,
    MaterialLibrary,
    PlaneSurface,
    Project,
    Ray,
    SphericalSurface,
    TraceBatch,
    TraceSettings,
    UnknownWavelengthError,
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


class SpectralTracingTests(unittest.TestCase):
    def _dispersive_project(self, low_index=1.60, high_index=1.45):
        materials = MaterialLibrary()
        materials.add(Material(
            "glass",
            index_by_wavelength={450.0: low_index, 650.0: high_index},
            loss_by_wavelength={450.0: 0.10, 650.0: 0.02},
        ))
        surfaces = [
            PlaneSurface("front", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="refract",
                          aperture=CircularAperture(2.0), back_material="glass"),
            PlaneSurface("back", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="refract",
                          aperture=CircularAperture(2.0), front_material="glass"),
            PlaneSurface("image", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          kind="detector", aperture=CircularAperture(3.0)),
        ]
        return Project("spectral", surfaces, materials)

    def test_same_beam_keeps_independent_wavelength_branches_and_energy(self):
        project = self._dispersive_project()
        beam = Beam((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [450.0, 650.0], label="beam")
        batch = project.trace_rays([beam])

        self.assertEqual({ray.wavelength_nm for ray in batch.rays}, {450.0, 650.0})
        detected = batch.detector_summary()["image"]
        self.assertAlmostEqual(detected["power"], detected["power_by_wavelength"]["450"]
                               + detected["power_by_wavelength"]["650"])
        self.assertNotAlmostEqual(
            detected["power_by_wavelength"]["450"],
            detected["power_by_wavelength"]["650"],
        )
        self.assertTrue(all(hit["wavelength_nm"] in {450.0, 650.0} for hit in detected["branches"]))
        wavelengths_on_paths = {
            (flow["wavelength_nm"], tuple(step["action"] for step in flow["steps"]))
            for flow in batch.branch_energy_flow()
        }
        self.assertTrue(any(wavelength == 450.0 for wavelength, _ in wavelengths_on_paths))
        self.assertTrue(any(wavelength == 650.0 for wavelength, _ in wavelengths_on_paths))

        ledger = batch.energy_ledger({"beam": {450.0: 1.0, 650.0: 1.0}})["beam"]
        by_wavelength = ledger["by_wavelength"]
        self.assertGreater(by_wavelength["450"]["bulk_absorbed"], by_wavelength["650"]["bulk_absorbed"])
        for spectral in by_wavelength.values():
            self.assertAlmostEqual(spectral["unaccounted"], 0.0)
            self.assertAlmostEqual(
                spectral["surface_absorbed"] + spectral["bulk_absorbed"],
                spectral["absorbed"],
            )

    def test_unknown_wavelength_is_rejected_not_interpolated(self):
        project = self._dispersive_project()
        with self.assertRaises(UnknownWavelengthError) as context:
            project.trace_rays([Ray((-1, 0, 0), (1, 0, 0), wavelength_nm=550.0, label="beam")])
        message = str(context.exception)
        self.assertIn("550 nm", message)
        self.assertIn("450 nm", message)
        self.assertIn("650 nm", message)
        self.assertIn("refusing", message)

        with self.assertRaises(UnknownWavelengthError):
            project.trace_rays([Beam((-1, 0, 0), (1, 0, 0), [550.0], label="beam")])

    def test_tir_branch_is_retained_for_only_one_wavelength(self):
        materials = MaterialLibrary()
        materials.add(Material("glass", index_by_wavelength={450.0: 1.6, 650.0: 1.45}))
        surfaces = [
            PlaneSurface("exit", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="refract",
                          aperture=CircularAperture(5.0), front_material="glass"),
            PlaneSurface("image", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          kind="detector", aperture=CircularAperture(5.0)),
        ]
        project = Project("split-tir", surfaces, materials)
        incident = (math.cos(math.radians(40.0)), math.sin(math.radians(40.0)), 0.0)
        batch = project.trace_rays([Beam((0.1, 0.0, 0.0), incident, [450.0, 650.0], label="beam")])

        terminal = {
            (flow["wavelength_nm"], flow["branch_id"], flow["status"], flow["steps"][-1]["action"])
            for flow in batch.branch_energy_flow()
        }
        self.assertIn((450.0, "root@450nm", "no_image", "total_internal_reflection"), terminal)
        self.assertTrue(any(
            wavelength == 650.0 and status == "reached_image" and action == "land"
            for wavelength, _, status, action in terminal
        ))
        self.assertTrue(any(
            wavelength == 650.0 and action == "refract"
            for wavelength, _, _, action in terminal
        ))

    def test_response_change_comparison_separates_detector_gain_and_losses(self):
        before = self._dispersive_project(low_index=1.62, high_index=1.48).trace_rays(
            [Beam((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [450.0, 650.0], label="beam")]
        )
        after = self._dispersive_project(low_index=1.60, high_index=1.45).trace_rays(
            [Beam((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), [450.0, 650.0], label="beam")]
        )
        detector_delta = before.detector_power_comparison(after)["image"]
        self.assertAlmostEqual(
            detector_delta["delta"],
            sum(item["delta"] for item in detector_delta["by_wavelength"].values()),
        )
        self.assertEqual(set(detector_delta["by_wavelength"]), {"450", "650"})

        ledger_delta = before.energy_ledger_comparison(
            after, {"beam": {450.0: 1.0, 650.0: 1.0}}
        )["beam"]
        self.assertIn("grazing", ledger_delta)
        self.assertIn("bulk_absorbed", ledger_delta)
        self.assertIn("surface_absorbed", ledger_delta)

    def test_spectral_surface_responses_and_project_roundtrip(self):
        materials = MaterialLibrary()
        materials.add(Material("glass", index_by_wavelength={450.0: 1.6, 650.0: 1.45}))
        surfaces = [
            PlaneSurface("split", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="beamsplitter",
                          aperture=CircularAperture(3.0),
                          reflectivity_by_wavelength={450.0: 0.4, 650.0: 0.1}),
            PlaneSurface("det_t", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          kind="detector", aperture=CircularAperture(3.0)),
            PlaneSurface("det_r", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          kind="detector", aperture=CircularAperture(3.0)),
        ]
        project = Project("coated", surfaces, materials)
        beam = Beam((-0.5, 0.0, 0.0), (1.0, 0.0, 0.0), [450.0, 650.0], label="beam")
        batch = project.trace_rays([beam])

        powers = batch.detector_summary()
        self.assertAlmostEqual(powers["det_r"]["power_by_wavelength"]["450"], 0.4)
        self.assertAlmostEqual(powers["det_t"]["power_by_wavelength"]["450"], 0.6)
        self.assertAlmostEqual(powers["det_r"]["power_by_wavelength"]["650"], 0.1)
        self.assertAlmostEqual(powers["det_t"]["power_by_wavelength"]["650"], 0.9)

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/project.json"
            project.save(path)
            reloaded = Project.load(path)
            recalculated = reloaded.trace_rays([beam])

        self.assertEqual(batch.detector_summary(), recalculated.detector_summary())

    def test_old_batch_energy_explanation_is_preserved_when_appending(self):
        old_surfaces = [
            PlaneSurface("image", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          kind="detector", aperture=CircularAperture(2.0)),
        ]
        old_batch = Project("old", old_surfaces).trace_rays([Ray((0, 0, 0), (1, 0, 0), label="old")])
        materials = MaterialLibrary()
        materials.add(Material("glass", index_by_wavelength={450.0: 1.6}))
        spectral_surfaces = [
            PlaneSurface("image", (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), kind="refract",
                          aperture=CircularAperture(2.0), back_material="glass"),
            PlaneSurface("detector", (2.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                          kind="detector", aperture=CircularAperture(2.0)),
        ]
        spectral_batch = Project("new", spectral_surfaces, materials).trace_rays([
            Beam((0, 0, 0), (1, 0, 0), [450.0], label="new")
        ])
        combined = old_batch.append_batch(spectral_batch)

        old_leaf = next(result for result in combined.results if result.source_label == "old")
        self.assertEqual(old_leaf.status, "reached_image")
        self.assertEqual(old_leaf.detector, "image")
        self.assertAlmostEqual(old_leaf.weight, 1.0)
        self.assertEqual(old_leaf.wavelength_nm, None)
        self.assertEqual(combined.energy_ledger({"old": 1.0})["old"]["detected"], 1.0)
        self.assertGreater(combined.energy_ledger()["new"]["by_wavelength"]["450"]["detected"], 0.9)

        with tempfile.TemporaryDirectory() as directory:
            old_path = f"{directory}/old.json"
            new_path = f"{directory}/combined.json"
            old_batch.save(old_path)
            combined.save(new_path)
            reloaded_old = TraceBatch.from_dict(json.loads(Path(old_path).read_text(encoding="utf-8")))
        self.assertEqual(reloaded_old.status_summary(), old_batch.status_summary())


if __name__ == "__main__":
    unittest.main()
