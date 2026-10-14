"""Regression checks for the paper's finite A-canopy topology."""

import unittest

from canopytda.a_canopy import ACanopy
from canopytda.complex import SimplicialComplex
from canopytda.filtration import Filtration
from canopytda.stratification import Stratification
from canopytda.topology import CanopyTopology
from canopytda.topology import build_canopy_topology
from canopytda.topology import canopy_points_separable
from canopytda.topology import constructible_closure
from canopytda.topology import validate_canopy_topology

from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS
from data.paper_complexes import TWO_DISJOINT_EDGES_FILTRATION
from data.paper_complexes import TWO_DISJOINT_EDGES_PAIRS


def _two_vertex_crossing():
    """Return a two-sheet crossing with a structural point at t=1/2."""
    _complex = SimplicialComplex(((0,), (1,)))
    _first = Filtration(_complex, {(0,): 0.0, (1,): 1.0})
    _last = Filtration(_complex, {(0,): 1.0, (1,): 0.0})
    return ACanopy(Stratification((_first, _last), (0.0, 1.0)),
                   _homology_dimension=0)


class FiniteTopologyTests(unittest.TestCase):
    """Test trajectories, relative-open windows and whole-cell closure."""

    @classmethod
    def setUpClass(cls):
        cls.canopy = _two_vertex_crossing()
        cls.topology = CanopyTopology(cls.canopy)

    def test_class_and_record_adapter_have_the_same_cells(self):
        """Expose the class API without modifying its A-canopy input."""
        _record = build_canopy_topology(self.canopy)
        self.assertEqual(self.topology.cells, _record["cells"])
        self.assertEqual(self.topology.relations, _record["relations"])
        self.assertEqual(_record["cell_count"], 6)
        self.assertEqual(_record["canopy"], self.canopy.to_record())
        self.assertIs(validate_canopy_topology(_record), _record)
        self.assertIs(validate_canopy_topology(self.topology), self.topology)
        self.assertFalse(_record["continuum_enumerated"])

    def test_structural_event_has_two_distinct_nonseparable_points(self):
        """Two event points share local neighborhoods without merging IDs."""
        _first, _second = (tuple(_point["pair"]
                                 for _point in self.canopy.fiber(0.5)))
        _local = self.topology.local_components(0.5)
        _certificate = self.topology.separation_certificate(
            0.5, _first, 0.5, _second)
        self.assertEqual(_local["component_count"], 1)
        self.assertEqual(len(_local["components"][0]["fiber_cell_ids"]), 2)
        self.assertTrue(_certificate["distinct"])
        self.assertFalse(_certificate["separable"])
        self.assertEqual(self.topology.hausdorff_classes()["class_count"], 1)
        self.assertEqual(self.topology.basis_neighborhood(
            0.5, _first, (0.49, 0.51))["stratum_ids"], (0, 1, 2))

    def test_regular_fiber_and_different_parameters_separate(self):
        """Shrinking the window excludes the distant crossing."""
        _left, _right = (tuple(_point["pair"]
                              for _point in self.canopy.fiber(0.25)))
        self.assertTrue(self.topology.points_separable(
            0.25, _left, 0.25, _right))
        self.assertTrue(self.topology.points_separable(
            0.25, _left, 0.75, self.canopy.fiber(0.75)[0]["pair"]))
        self.assertEqual(self.topology.basis_neighborhood(
            0.25, _left, (0.2, 0.3))["stratum_ids"], (0,))

    def test_closure_adds_event_boundary_but_no_opposite_interval(self):
        """Closing one whole sheet adds both structural event points."""
        _result = self.topology.closure(((0, 0),))
        self.assertEqual(_result["boundary_cell_ids"], ((1, 0), (1, 1)))
        self.assertFalse(any(_id[0] == 2
                             for _id in _result["closure_cell_ids"]))
        self.assertEqual(self.topology.closure(((1, 0),))[
            "boundary_cell_ids"], ((1, 1),))
        self.assertEqual(self.topology.closure(())[
            "closure_cell_ids"], ())
        with self.assertRaises(ValueError):
            self.topology.closure(((99, 0),))

    def test_validation_rejects_changed_cell_or_relation(self):
        """Check cell and relation data against the original continuations."""
        _altered_cell = self.topology.to_record()
        _altered_cell["cells"][0]["pair"] = ((99,), None)
        with self.assertRaises(ValueError):
            validate_canopy_topology(_altered_cell)
        _altered_relation = self.topology.to_record()
        _altered_relation["relations"][0]["kind"] = "fake"
        with self.assertRaises(ValueError):
            validate_canopy_topology(_altered_relation)

    def test_class_record_is_detached(self):
        """Public reads cannot mutate the stored cells or A-canopy."""
        _copy = self.topology.to_record()
        _copy["cells"][0]["pair"] = ((99,), None)
        self.assertNotEqual(self.topology.cells[0]["pair"], ((99,), None))
        _copy = self.topology["canopy"]
        _copy["fiber_cardinality"] = -1
        self.assertEqual(self.topology.canopy["fiber_cardinality"], 2)
        with self.assertRaises(AttributeError):
            self.topology.cells = ()


class PaperTopologyTests(unittest.TestCase):
    """Check structural, nonstructural and coincidental point behavior."""

    @classmethod
    def setUpClass(cls):
        _family = Stratification(
            PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)
        cls.canopy = ACanopy(_family, _homology_dimension=1)
        cls.topology = CanopyTopology(cls.canopy)

    def test_example_3_12_has_one_then_two_local_components(self):
        """Only the t=1/3 event has nonseparable H1 points."""
        _structural = self.topology.local_components(1 / 3)
        _ordinary = self.topology.local_components(3 / 4)
        self.assertEqual(_structural["component_count"], 1)
        self.assertEqual(len(_structural["components"][0][
            "fiber_cell_ids"]), 2)
        self.assertEqual(_ordinary["component_count"], 2)
        self.assertTrue(all(len(_component["fiber_cell_ids"]) == 1
                            for _component in _ordinary["components"]))
        self.assertEqual(self.topology.hausdorff_classes()["class_count"], 1)
        _agreement = self.topology.structurality_certificate()
        self.assertTrue(_agreement["independent_construction"])
        self.assertTrue(_agreement["all_events_agree"])
        self.assertEqual(tuple(_event["parameter"]
                               for _event in _agreement["events"]),
                         (1 / 3, 3 / 4))

    def test_matching_diagram_points_can_still_be_separable(self):
        """Example 3.16 has equal finite coordinates on unrelated edges."""
        _canopy = ACanopy(Stratification((TWO_DISJOINT_EDGES_FILTRATION,)),
                          _homology_dimension=0)
        _topology = CanopyTopology(_canopy)
        _first, _second = TWO_DISJOINT_EDGES_PAIRS
        self.assertEqual(_canopy.diagram(0, 0).count((1.0, 2.0)), 2)
        self.assertTrue(_topology.points_separable(
            0.0, _first, 0.0, _second))


if __name__ == "__main__":
    unittest.main()
