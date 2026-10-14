"""Local structurality certificates from stratification and equivalence."""

import unittest

from canopytda.filtration import Filtration
from canopytda.stratification import Stratification
from canopytda.structurality import Structurality
from canopytda.structurality import adjacent_stratum_equivalences
from canopytda.structurality import classify_structural_events
from canopytda.structurality import verified_stratum_adjacency

from data.paper_complexes import A_SIMPLEX
from data.paper_complexes import B_SIMPLEX
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS
from data.paper_complexes import TWO_DISJOINT_EDGES_COMPLEX


class StructuralityTests(unittest.TestCase):
    """Distinguish structural and coincidental paper-style events."""

    def setUp(self):
        """Build the fixed L example once per test."""
        self.stratification = Stratification(
            PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)

    def test_paper_example_312_has_one_structural_event(self):
        """The one-third event branches, while three-quarters does not."""
        _analysis = Structurality(self.stratification)
        _result = _analysis.classify_events(1)
        _events = _result["events"]
        self.assertEqual(len(_events), 2)
        self.assertAlmostEqual(_events[0]["parameter"], 1.0 / 3.0)
        self.assertEqual(_events[0]["classification"], "structural")
        self.assertEqual(_events[0]["structural_points"][0]["multiplicity"], 2)
        self.assertEqual({_pair[0] for _pair in _events[0][
            "structural_points"][0]["non_hausdorff_pairs"]},
                         {A_SIMPLEX, B_SIMPLEX})
        self.assertEqual(_events[1]["parameter"], 0.75)
        self.assertEqual(_events[1]["classification"], "nonstructural")
        self.assertEqual(_result["structural_parameters"], (1.0 / 3.0,))
        self.assertTrue(_result["complete"])

    def test_both_dependencies_connect_through_adjacency(self):
        """Every incident edge carries complete combinatorial maps."""
        _analysis = Structurality(self.stratification)
        _graph = _analysis.verified_adjacency()
        self.assertEqual(len(_graph["edges"]), 4)
        self.assertTrue(_graph["all_edges_compatible"])
        _event_id = self.stratification.events[0]["event_stratum_id"]
        _neighbor = self.stratification.events[0]["left_stratum_id"]
        _maps = _analysis.adjacent_equivalences(_event_id, _neighbor)
        self.assertTrue(_maps["complete"])
        self.assertTrue(_maps["equivalences"])
        self.assertEqual(_maps["source_stratum_id"], _event_id)
        self.assertEqual(_maps["target_stratum_id"], _neighbor)
        self.assertEqual(verified_stratum_adjacency(
            self.stratification.to_record())["edges"], _graph["edges"])
        self.assertEqual(len(adjacent_stratum_equivalences(
            self.stratification, _event_id, _neighbor)["equivalences"]),
            len(_maps["equivalences"]))
        self.assertEqual(classify_structural_events(
            self.stratification.to_record(), _homology_dimension=1)[
                "structural_parameters"], (1.0 / 3.0,))

    def test_disconnected_coincidence_is_not_structural(self):
        """Equal H0 intervals in separate components remain distinct."""
        _complex = TWO_DISJOINT_EDGES_COMPLEX
        _base = {(0,): -2.0, (2,): -1.0,
                 (0, 1): 2.0, (2, 3): 2.0}
        _first = Filtration(_complex, {**_base, (1,): 0.0, (3,): 1.0})
        _last = Filtration(_complex, {**_base, (1,): 1.0, (3,): 0.0})
        _family = Stratification((_first, _last), (0.0, 1.0))
        _result = Structurality(_family).classify_events(0)
        _at_middle = tuple(_event for _event in _result["events"]
                           if abs(_event["parameter"] - 0.5) < 1e-12)
        self.assertEqual(len(_at_middle), 1)
        self.assertEqual(_at_middle[0]["classification"], "coincidental")
        self.assertEqual(_at_middle[0]["multiplicity_points"][0][
            "multiplicity"], 2)
        self.assertEqual(_result["structural_parameters"], ())

    def test_limited_search_does_not_claim_nonstructural(self):
        """Equivalence truncation stops the structurality computation."""
        with self.assertRaises(RuntimeError):
            Structurality(self.stratification,
                          _max_equivalences=1).classify_events(1)
        with self.assertRaises(RuntimeError):
            Structurality(self.stratification,
                          _max_common_refinements=1).classify_events(1)

    def test_invalid_strata_and_dimension(self):
        """Reject nonadjacent IDs and invalid homological dimensions."""
        _analysis = Structurality(self.stratification)
        with self.assertRaises(ValueError):
            _analysis.adjacent_equivalences(0, 4)
        with self.assertRaises(TypeError):
            _analysis.classify_events(True)
        _record = self.stratification.to_record()
        _record["events"] = ()
        with self.assertRaises(ValueError):
            Structurality(_record)


if __name__ == "__main__":
    unittest.main()
