"""Regression tests for finite diminished canopies (paper Section 4)."""

from copy import deepcopy
import unittest

from canopytda.a_canopy import ACanopy
from canopytda.complex import SimplicialComplex
from canopytda.d_canopy import DELTA
from canopytda.d_canopy import DCanopy
from canopytda.d_canopy import build_d_canopy
from canopytda.d_canopy import validate_d_canopy
from canopytda.filtration import Filtration
from canopytda.pairing import PersistencePairing
from canopytda.stratification import Stratification
from canopytda.topology import CanopyTopology

from data.paper_complexes import D_ALWAYS_DIAGONAL_PAIR
from data.paper_complexes import D_DIAGONAL_FILTRATIONS
from data.paper_complexes import D_DIAGONAL_PARAMETERS
from data.paper_complexes import D_FINITE_PAIR
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS
from data.paper_complexes import TWO_DISJOINT_EDGES_FILTRATION


def _crossing_canopy():
    """Return two essential points that coincide structurally at 1/2."""
    _complex = SimplicialComplex(((0,), (1,)))
    _first = Filtration(_complex, {(0,): 0.0, (1,): 1.0})
    _last = Filtration(_complex, {(0,): 1.0, (1,): 0.0})
    return ACanopy(Stratification((_first, _last), (0.0, 1.0)),
                   _homology_dimension=0)


def _interpolated_diagram(_parameter):
    """Compute the independent reduced diagram of the example family."""
    _parameters = D_DIAGONAL_PARAMETERS
    _samples = D_DIAGONAL_FILTRATIONS
    _index = 0 if _parameter <= 0.5 else 1
    _left, _right = _parameters[_index:_index + 2]
    _weight = (_parameter - _left) / (_right - _left)
    _values = {
        _simplex: ((1 - _weight) * _samples[_index].value(_simplex) +
                   _weight * _samples[_index + 1].value(_simplex))
        for _simplex in _samples[0].complex.simplices
    }
    return PersistencePairing(Filtration(_samples[0].complex, _values))\
        .diminished_diagram(0)


class DCanopyTests(unittest.TestCase):
    """Exercise closure, diagonal values, structural loci and public API."""

    @classmethod
    def setUpClass(cls):
        _family = Stratification(D_DIAGONAL_FILTRATIONS,
                                 D_DIAGONAL_PARAMETERS)
        cls.canopy = ACanopy(_family, _homology_dimension=0)
        cls.topology = CanopyTopology(cls.canopy)
        cls.d_canopy = DCanopy(cls.canopy, cls.topology)

    def test_refinement_reaches_diagonal_without_an_order_event(self):
        """An unchanged order must still yield a retained middle point."""
        self.assertEqual(self.canopy["stratification"]["events"], ())
        self.assertIn(0.5, self.d_canopy["base"]["refinement_breakpoints"])
        _boundary, = self.d_canopy.boundary_cells()
        self.assertEqual((_boundary["pair"], _boundary["start"]),
                         (D_FINITE_PAIR, 0.5))
        self.assertEqual(_boundary["representative_eta_prime"], DELTA)
        self.assertEqual(next(_point["eta_prime"] for _point in
                              self.d_canopy.fiber(0.5) if
                              _point["pair"] == D_FINITE_PAIR), DELTA)

    def test_always_diagonal_is_removed(self):
        """Diagonal-only cells have no off-diagonal points in their closure."""
        _ambient = self.d_canopy["ambient_refinement"]["cells"]
        _removed = set(self.d_canopy["removed_diagonal_cell_ids"])
        self.assertIn(D_ALWAYS_DIAGONAL_PAIR,
                      {_cell["pair"] for _cell in _ambient
                       if _cell["cell_id"] in _removed})
        self.assertNotIn(D_ALWAYS_DIAGONAL_PAIR,
                         {_cell["pair"] for _cell in self.d_canopy.cells})
        self.assertEqual(len(self.d_canopy.sheet_domains()), 3)

    def test_diminished_diagrams_match_independent_pairing(self):
        """Match direct reduction at sample, interior and diagonal values."""
        for _parameter in (0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0):
            with self.subTest(parameter=_parameter):
                self.assertEqual(self.d_canopy.diagram(_parameter, 0),
                                 _interpolated_diagram(_parameter))
        self.assertEqual(self.d_canopy.diagram(0.5, 1), ())
        self.assertEqual(self.d_canopy.diagram(0.5)[0],
                         ((0.0, float("inf")), (2.0, float("inf"))))

    def test_class_and_record_inputs_and_detachment(self):
        """The functional entry point agrees and public reads are isolated."""
        _record = build_d_canopy(self.canopy.to_record(),
                                 self.topology.to_record())
        self.assertEqual(self.d_canopy.to_record(), _record)
        self.assertIs(validate_d_canopy(self.d_canopy), self.d_canopy)
        _copy = self.d_canopy.to_record()
        _copy["cells"][0]["pair"] = ((99,), None)
        self.assertNotEqual(self.d_canopy.cells[0]["pair"], ((99,), None))
        with self.assertRaises(AttributeError):
            self.d_canopy.cells = ()

    def test_mismatched_topology_and_invalid_arguments(self):
        """Reject topology from another canopy and invalid query values."""
        with self.assertRaises(ValueError):
            DCanopy(self.canopy, CanopyTopology(_crossing_canopy()))
        with self.assertRaises(ValueError):
            DCanopy(self.canopy, _atol=-1)
        with self.assertRaises(TypeError):
            DCanopy(self.canopy, _max_refinement_events=True)
        with self.assertRaises(ValueError):
            self.d_canopy.fiber(2.0)
        with self.assertRaises(ValueError):
            self.d_canopy.diagram(float("nan"))

    def test_validation_detects_tampered_adjacency(self):
        """Stored topology must agree with witnessed retained relations."""
        _record = deepcopy(self.d_canopy.to_record())
        _relation = next(_item for _item in _record["relations"]
                         if _item["source_cell_id"] != _item["target_cell_id"])
        _record["adjacency"][_relation["source_cell_id"]] = (
            _relation["source_cell_id"],)
        with self.assertRaises(ValueError):
            validate_d_canopy(_record)

    def test_validation_detects_tampered_diagonal_boundary(self):
        """A purported boundary must actually be a limit of the open locus."""
        _record = self.d_canopy.to_record()
        _record["diagonal_boundary_cell_ids"] = ()
        with self.assertRaises(ValueError):
            validate_d_canopy(_record)

    def test_essential_structural_point_is_excluded_from_sheet_domains(self):
        """The doubled essential event yields an approximate D-canopy."""
        _d_canopy = DCanopy(_crossing_canopy())
        self.assertTrue(_d_canopy.approximate)
        self.assertEqual(_d_canopy["excluded_structural_parameters"], (0.5,))
        self.assertTrue(_d_canopy["local_chart_certificate"][
            "all_charts_valid"])

    def test_coincidental_finite_multiplicity_remains_separable(self):
        """Example 3.16 has two identical diagram values on disjoint edges."""
        _canopy = ACanopy(Stratification((TWO_DISJOINT_EDGES_FILTRATION,)),
                          _homology_dimension=0)
        _d_canopy = DCanopy(_canopy)
        self.assertEqual(_d_canopy.diagram(0, 0).count((1.0, 2.0)), 2)
        _groups = _d_canopy["structurality"]["events"][0][
            "multiplicity_points"]
        _finite, = (_group for _group in _groups if
                    _group["eta_prime"] == (1.0, 2.0))
        self.assertEqual(_finite["classification"], "coincidental")

    def test_paper_example_3_12_preserves_h1_diagrams(self):
        """Off-diagonal L-example pairs survive, including the event."""
        _family = Stratification(PAPER_3_7_FILTRATIONS,
                                 PAPER_3_7_PARAMETERS)
        _d_canopy = DCanopy(ACanopy(_family, _homology_dimension=1))
        self.assertEqual(_d_canopy["removed_diagonal_cell_ids"], ())
        self.assertEqual(_d_canopy["diagonal_boundary_cell_ids"], ())
        self.assertEqual(_d_canopy["excluded_structural_parameters"],
                         (1 / 3,))
        for _parameter, _filtration in zip(PAPER_3_7_PARAMETERS,
                                            PAPER_3_7_FILTRATIONS):
            self.assertEqual(_d_canopy.diagram(_parameter, 1),
                             PersistencePairing(_filtration)
                             .diminished_diagram(1))


if __name__ == "__main__":
    unittest.main()
