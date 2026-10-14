"""Paper examples and finite invariants for the public A-canopy API."""

from copy import deepcopy
import math
import unittest

from canopytda.a_canopy import ACanopy
from canopytda.a_canopy import a_canopy_diagram
from canopytda.a_canopy import a_canopy_fiber
from canopytda.a_canopy import adjacent_continuations
from canopytda.a_canopy import build_a_canopy
from canopytda.a_canopy import validate_a_canopy
from canopytda.complex import SimplicialComplex
from canopytda.filtration import Filtration
from canopytda.pairing import PersistencePairing
from canopytda.stratification import Stratification

from data.paper_complexes import A_SIMPLEX
from data.paper_complexes import B_SIMPLEX
from data.paper_complexes import C_SIMPLEX
from data.paper_complexes import D_SIMPLEX
from data.paper_complexes import FILLED_TRIANGLE_FILTRATION
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS


class ACanopyTests(unittest.TestCase):
    """Check distinct sheets, all incidences, and correct eta images."""

    def test_two_essential_vertices_keep_both_continuations(self):
        """At a value crossing, neither continuation branch is discarded."""
        _complex = SimplicialComplex(((0,), (1,)))
        _first = Filtration(_complex, {(0,): 0.0, (1,): 1.0})
        _last = Filtration(_complex, {(0,): 1.0, (1,): 0.0})
        _family = Stratification((_first, _last), (0.0, 1.0))
        _canopy = ACanopy(_family, _homology_dimension=0)
        _record = _canopy.to_record()
        self.assertEqual(_canopy.fiber_cardinality, 2)
        self.assertEqual(len(_record["representative_fibers"]), 3)
        self.assertEqual(len(_record["incidences"]), 2)
        self.assertIs(validate_a_canopy(_record), _record)
        _forward = _canopy.continuations(0, 1)
        _backward = _canopy.continuations(1, 0)
        self.assertEqual(len(_forward), 2)
        for _first_map, _reverse_map in zip(_forward, _backward):
            self.assertEqual(
                set(_reverse_map["mapping"]),
                {(_target, _source)
                 for _source, _target in _first_map["mapping"]})

    def test_eta_image_matches_persistence_at_arbitrary_parameters(self):
        """The fiber image is the augmented multiset, including duplicates."""
        _family = Stratification(
            PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)
        _canopy = ACanopy(_family, _homology_dimension=1)
        for _parameter in (0.0, 0.25, 1 / 3, 0.5, 0.75, 0.9, 1.0):
            with self.subTest(parameter=_parameter):
                _direct = PersistencePairing(
                    _family.filtration_at(_parameter)).augmented_diagram(1)
                self.assertEqual(_canopy.diagram(_parameter, 1), _direct)
                self.assertEqual(len(_canopy.fiber(_parameter)), 2)
                self.assertEqual(tuple(_point["projection"]
                                       for _point in _canopy.fiber(_parameter)),
                                 (_parameter, _parameter))
        _event_points = _canopy.fiber(1 / 3)
        self.assertEqual(_event_points[0]["eta"], _event_points[1]["eta"])
        self.assertNotEqual(_event_points[0]["pair"],
                            _event_points[1]["pair"])

    def test_paper_event_retains_swapped_self_continuation(self):
        """Keep both paper-labeled pair points at the structural event."""
        _family = Stratification(
            PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)
        _canopy = ACanopy(_family, _homology_dimension=1)
        _event_id = _family.events[0]["event_stratum_id"]
        _fiber = next(_item for _item in _canopy["representative_fibers"]
                      if _item["stratum_id"] == _event_id)
        _swap = {((A_SIMPLEX, D_SIMPLEX), (B_SIMPLEX, C_SIMPLEX)),
                 ((B_SIMPLEX, C_SIMPLEX), (A_SIMPLEX, D_SIMPLEX))}
        self.assertTrue(any(set(_continuation["pair_mapping"]) == _swap
                            for _continuation in _fiber["self_continuations"]))
        self.assertEqual(_canopy.structural_parameters, (1 / 3,))
        self.assertEqual(len(_canopy["incidences"]), 4)

    def test_diagonal_and_essential_points_remain_in_augmented_fiber(self):
        """Retain zero-length intervals and essential infinite endpoints."""
        _family = Stratification((FILLED_TRIANGLE_FILTRATION,))
        _canopy = ACanopy(_family)
        _diagram = _canopy.diagram(0.0)
        _direct = PersistencePairing(FILLED_TRIANGLE_FILTRATION)
        for _dimension, _points in _diagram.items():
            self.assertEqual(_points, _direct.augmented_diagram(_dimension))
        self.assertIn((1.0, 1.0), _diagram[1])
        self.assertIn((0.0, math.inf), _diagram[0])
        self.assertEqual(_canopy["structural_parameters"], ())

    def test_empty_selected_degree_has_empty_bijections(self):
        """A degree absent from every fiber has cardinality zero."""
        _family = Stratification((FILLED_TRIANGLE_FILTRATION,))
        _canopy = ACanopy(_family, _homology_dimension=3)
        self.assertEqual(_canopy.fiber_cardinality, 0)
        self.assertEqual(_canopy.diagram(0.0, 3), ())
        self.assertEqual(_canopy.fiber(0.0), ())

    def test_bounded_search_never_picks_one_branch(self):
        """Construction refuses incomplete common-order enumeration."""
        _complex = SimplicialComplex(((0,), (1,)))
        _first = Filtration(_complex, {(0,): 0.0, (1,): 1.0})
        _last = Filtration(_complex, {(0,): 1.0, (1,): 0.0})
        _family = Stratification((_first, _last), (0.0, 1.0))
        with self.assertRaises(RuntimeError):
            ACanopy(_family, _homology_dimension=0,
                    _max_common_refinements=1)

    def test_public_record_adapter_and_validation(self):
        """Build from a record and reject a truncated incidence."""
        _family = Stratification(
            PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)
        _record = build_a_canopy(_family.to_record(), _homology_dimension=1)
        self.assertEqual(a_canopy_diagram(_record, 1 / 3)[1],
                         PersistencePairing(
                             _family.filtration_at(1 / 3)).augmented_diagram(1))
        self.assertEqual(len(a_canopy_fiber(_record, 0.2)), 2)
        _edge = _record["incidences"][0]
        self.assertEqual(len(adjacent_continuations(
            _record, _edge["source_stratum_id"],
            _edge["target_stratum_id"])),
            _edge["continuation_count"])
        _broken = deepcopy(_record)
        _broken["incidences"][0]["continuations"] = ()
        with self.assertRaises(ValueError):
            validate_a_canopy(_broken)

    def test_public_object_is_immutable(self):
        """Returned record values can be edited without changing the class."""
        _canopy = ACanopy(Stratification((FILLED_TRIANGLE_FILTRATION,)))
        _original = _canopy.fiber_cardinality
        _copy = _canopy.to_record()
        _copy["fiber_cardinality"] = -1
        self.assertEqual(_canopy.fiber_cardinality, _original)
        with self.assertRaises(AttributeError):
            _canopy.fiber_cardinality = 7
        with self.assertRaises(TypeError):
            _canopy.diagram(0.0, True)


if __name__ == "__main__":
    unittest.main()
