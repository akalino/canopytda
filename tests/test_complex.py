"""Tests for the fixed complex, its F2 boundary, and GUDHI conversion.

Run after installing the package and GUDHI::

    python -m unittest tests.test_complex -v
"""

import math
import unittest

import gudhi as gd

from canopytda.complex import boundary
from canopytda.complex import boundary_columns
from canopytda.complex import SimplicialComplex
from canopytda.complex import ordered_simplices
from canopytda.complex import simplex_tree_from_point_cloud
from canopytda.complex import simplex_tree_from_simplices


PAPER_COMPLEXES = {"single_edge": [[0], [1], [0, 1]],
                   "filled_triangle": [[0], [1], [2], [0, 1], [0, 2],
                                       [1, 2], [0, 1, 2]],
                   "l_complex": [[0], [1], [2], [3], [0, 1], [1, 3], [2, 3],
                                 [0, 2], [1, 2], [0, 1, 2], [1, 2, 3]],
                   "essential_cycle": [[0], [1], [2], [0, 1], [0, 2], [1, 2]]}


class ComplexTests(unittest.TestCase):
    """Check the fixed simplex domain and its GUDHI compatibility."""

    def test_paper_complexes(self):
        """Accept every distinct complex in the initial paper fixtures."""

        _expected = {"single_edge": (2, 3, 1), "filled_triangle": (3, 7, 2),
                     "l_complex": (4, 11, 2), "essential_cycle": (3, 6, 1)}

        for _name, _simplices in PAPER_COMPLEXES.items():
            with self.subTest(name=_name):
                _tree = simplex_tree_from_simplices(_simplices)
                self.assertEqual(_tree.num_vertices(), _expected[_name][0])
                self.assertEqual(_tree.num_simplices(), _expected[_name][1])
                self.assertEqual(_tree.dimension(), _expected[_name][2])

    def test_facets_are_completed(self):
        """Allow maximal simplices without explicitly listing faces."""

        _tree = simplex_tree_from_simplices([[0, 1, 2]])
        self.assertEqual(_tree.num_simplices(), 7)

    def test_fixed_complex_closes_faces_and_exposes_incidences(self):
        """Keep one labeled simplex domain with a canonical face order."""

        _complex = SimplicialComplex([[2, 0, 1], [4]])
        self.assertEqual(_complex.vertices, (0, 1, 2, 4))
        self.assertEqual(_complex.dimension, 2)
        self.assertEqual(_complex.facets, ((4,), (0, 1, 2)))
        self.assertEqual(len(_complex), 8)
        self.assertEqual(_complex.faces((1, 0, 2)),
                         ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)))
        self.assertEqual(_complex.cofaces((0, 1)), ((0, 1, 2),))
        self.assertIn((2, 1, 0), _complex)
        self.assertEqual(ordered_simplices(_complex), _complex.simplices)

    def test_fixed_complex_is_immutable_and_validated(self):
        """Reject invalid simplices and prevent changes to stored labels."""

        _complex = SimplicialComplex([[0, 1]])
        with self.assertRaises(AttributeError):
            _complex._simplices = ()
        with self.assertRaises(ValueError):
            SimplicialComplex([])
        with self.assertRaises(ValueError):
            SimplicialComplex([[0, 0]])
        with self.assertRaises(ValueError):
            _complex.faces((2,))

    def test_fixed_complex_gudhi_round_trip_discards_filtration(self):
        """Copy simplex identities without transferring filtration values."""

        _complex = SimplicialComplex([[0, 1, 2]])
        _first_tree = _complex.to_simplex_tree()
        _first_tree.assign_filtration([0, 1, 2], 4.0)
        _recovered = SimplicialComplex.from_simplex_tree(_first_tree)
        self.assertEqual(_recovered, _complex)
        self.assertEqual(
            _complex.to_simplex_tree().filtration([0, 1, 2]), 0.0)
        self.assertEqual(boundary_columns(_complex),
                         boundary_columns(_complex.to_simplex_tree()))

    def test_fixed_complex_matches_gudhi_on_paper_examples(self):
        """Compare facet completion with an independently assembled tree."""

        for _name, _simplices in PAPER_COMPLEXES.items():
            with self.subTest(name=_name):
                _complex = SimplicialComplex(_simplices)
                _reference = gd.SimplexTree()
                for _simplex in _simplices:
                    _reference.insert(_simplex, filtration=0.0)
                _reference_simplices = {
                    tuple(sorted(_simplex))
                    for _simplex, _value in _reference.get_simplices()
                }
                self.assertEqual(set(_complex.simplices), _reference_simplices)
                self.assertEqual(_complex.dimension, _reference.dimension())
                self.assertEqual(len(_complex.vertices), _reference.num_vertices())

    def test_boundary_squared_is_zero(self):
        """Verify the chain-complex identity on a tetrahedron and an edge."""

        _complex = SimplicialComplex([[0, 1, 2, 3], [3, 4]])
        _columns = boundary_columns(_complex)
        for _simplex, _column in zip(_complex.simplices, _columns):
            with self.subTest(simplex=_simplex):
                _second_boundary = 0
                for _index, _face_column in enumerate(_columns):
                    if _column & (1 << _index):
                        _second_boundary ^= _face_column
                self.assertEqual(_second_boundary, 0)

    def test_boundary(self):
        """Return the expected triangle boundary."""

        self.assertEqual(boundary([0, 1, 2]), ((0, 1), (0, 2), (1, 2)))

    def test_boundary_columns(self):
        """Encode triangle boundaries as integer bitsets."""

        _order = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]
        _tree = simplex_tree_from_simplices(_order)
        self.assertEqual(boundary_columns(_tree, _order), (0, 0, 0, 3, 5, 6, 56))

    def test_invalid_boundary_order(self):
        """Reject an order that places a coface before its faces."""

        _tree = simplex_tree_from_simplices([[0, 1]])
        with self.assertRaises(ValueError):
            boundary_columns(_tree, [[0, 1], [0], [1]])

    def test_point_cloud(self):
        """Build one filled Rips triangle."""

        _points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        _tree = simplex_tree_from_point_cloud(_points, 1.5, 2)

        self.assertEqual(_tree.num_vertices(), 3)
        self.assertEqual(_tree.num_simplices(), 7)
        self.assertEqual(_tree.dimension(), 2)
        self.assertAlmostEqual(_tree.filtration([1, 2]), math.sqrt(2.0))


if __name__ == "__main__":
    unittest.main()
