"""Public behavior checks for the attached ``canopytda/filtration.py``.

Put this file in the package's ``tests`` directory and run::

    python -m unittest discover -s tests -p 'test_filtration.py' -v

The tests require the package and its GUDHI dependency to be installed.
"""

import math
import unittest
from unittest.mock import Mock, patch

import gudhi as gd

from canopytda.complex import SimplicialComplex
from canopytda.filtration import Filtration
from canopytda.filtration import alpha_filtration
from canopytda.filtration import available_filtrations
from canopytda.filtration import build_filtration
from canopytda.filtration import custom_filtration
from canopytda.filtration import delaunay_cech_filtration
from canopytda.filtration import explicit_filtration
from canopytda.filtration import filtration_values
from canopytda.filtration import lower_star_filtration
from canopytda.filtration import rips_filtration


class FiltrationTests(unittest.TestCase):
    """Check one filtration function on a fixed simplex domain."""

    def setUp(self):
        self.complex = SimplicialComplex([[2, 0, 1]])
        self.values = {
            (0,): -1, (1,): 0, (2,): 1,
            (0, 1): 2, (0, 2): 2, (1, 2): 2,
            (0, 1, 2): 3,
        }

    def test_explicit_values_include_every_face_and_canonicalize_keys(self):
        _values = dict(self.values)
        _values[(1, 0)] = _values.pop((0, 1))
        _filtration = Filtration(self.complex, _values)

        self.assertIs(_filtration.complex, self.complex)
        self.assertEqual(_filtration.value((1, 0)), 2.0)
        self.assertEqual(_filtration.value((2, 0, 1)), 3.0)
        self.assertEqual(dict(_filtration.values), {
            _simplex: float(_value)
            for _simplex, _value in self.values.items()
        })
        with self.assertRaises(KeyError):
            _filtration.value((9,))

    def test_filtration_does_not_share_mutable_values_with_caller(self):
        _values = dict(self.values)
        _filtration = Filtration(self.complex, _values)
        _values[(0,)] = 100

        self.assertEqual(_filtration.value((0,)), -1.0)
        with self.assertRaises(TypeError):
            _filtration.values[(0,)] = 100
        with self.assertRaises(AttributeError):
            _filtration._values = {}
        with self.assertRaises(AttributeError):
            del _filtration._complex

    def test_missing_and_unknown_simplices_are_rejected(self):
        for _edit in ("missing", "unknown"):
            with self.subTest(edit=_edit):
                _values = dict(self.values)
                if _edit == "missing":
                    del _values[(0, 1)]
                else:
                    _values[(8,)] = 0
                with self.assertRaises(ValueError):
                    Filtration(self.complex, _values)

    def test_duplicate_simplex_keys_in_different_orders_are_rejected(self):
        _values = dict(self.values)
        _values[(1, 0)] = 2
        with self.assertRaisesRegex(ValueError, "repeats a simplex"):
            Filtration(self.complex, _values)

    def test_face_monotonicity_and_finite_values_are_required(self):
        for _simplex, _value, _error in (
            ((0, 1), -2, ValueError),
            ((0, 1, 2), 1, ValueError),
            ((0,), math.nan, ValueError),
            ((0,), math.inf, ValueError),
            ((0,), "not a number", TypeError),
        ):
            with self.subTest(simplex=_simplex, value=_value):
                _values = dict(self.values)
                _values[_simplex] = _value
                with self.assertRaises(_error):
                    Filtration(self.complex, _values)

    def test_tree_round_trip_preserves_values_and_independence(self):
        _first = Filtration(self.complex, self.values)
        _tree = _first.to_simplex_tree()
        _second_tree = _first.to_simplex_tree()

        self.assertIsInstance(_tree, gd.SimplexTree)
        self.assertIsNot(_tree, _second_tree)
        self.assertEqual(filtration_values(_tree), dict(_first.values))
        self.assertEqual(Filtration.from_simplex_tree(_tree).complex, self.complex)
        self.assertEqual(dict(Filtration.from_simplex_tree(_tree).values),
                         dict(_first.values))
        self.assertEqual(dict(Filtration(_tree, self.values).values),
                         dict(_first.values))

        _tree.assign_filtration([0, 1, 2], 9)
        self.assertEqual(_first.value((0, 1, 2)), 3.0)
        self.assertEqual(_second_tree.filtration([0, 1, 2]), 3.0)

    def test_filtration_values_returns_a_detached_mapping(self):
        _filtration = Filtration(self.complex, self.values)
        _values = filtration_values(_filtration)
        _values[(0,)] = 100
        self.assertEqual(_filtration.value((0,)), -1.0)

    def test_lower_star_class_and_tree_adapter_agree(self):
        _vertex_values = {0: -1, 1: 2, 2: 1}
        _filtration = Filtration.lower_star(self.complex, _vertex_values)
        _tree = lower_star_filtration(self.complex, _vertex_values)

        self.assertEqual(_filtration.value((0, 2)), 1.0)
        self.assertEqual(_filtration.value((0, 1, 2)), 2.0)
        self.assertEqual(filtration_values(_tree), dict(_filtration.values))

    def test_lower_star_requires_exactly_the_fixed_vertices(self):
        for _values, _error in (
            ({0: 0, 1: 1}, ValueError),
            ({0: 0, 1: 1, 2: 2, 3: 3}, ValueError),
            ({0: 0, 1: 1, 2: math.nan}, ValueError),
            ({0: 0, True: 2, 2: 2}, TypeError),
        ):
            with self.subTest(values=_values):
                with self.assertRaises(_error):
                    Filtration.lower_star(self.complex, _values)

    def test_tree_adapters_preserve_the_input_tree(self):
        _tree = self.complex.to_simplex_tree()
        _explicit = explicit_filtration(_tree, self.values)
        _star = lower_star_filtration(_tree, {0: 0, 1: 1, 2: 2})

        self.assertIsNot(_explicit, _tree)
        self.assertIsNot(_star, _tree)
        self.assertEqual(_tree.filtration([0, 1, 2]), 0.0)
        self.assertEqual(_explicit.filtration([0, 1, 2]), 3.0)
        self.assertEqual(_star.filtration([0, 1, 2]), 2.0)

    def test_custom_callback_receives_each_canonical_simplex(self):
        _seen = []

        def _function(_simplex):
            _seen.append(_simplex)
            return len(_simplex) + 0.5

        _filtration = custom_filtration(self.complex, _function)
        self.assertIsInstance(_filtration, Filtration)
        self.assertIs(_filtration.complex, self.complex)
        self.assertEqual(_seen, list(self.complex.simplices))
        self.assertEqual(_filtration.value((0, 1)), 2.5)
        self.assertEqual(_filtration.value((0, 1, 2)), 3.5)

    def test_custom_callback_must_be_callable_and_face_monotone(self):
        with self.assertRaises(TypeError):
            custom_filtration(object(), lambda _simplex: 0)
        with self.assertRaises(TypeError):
            custom_filtration(self.complex, None)
        for _function, _error in (
            (lambda _simplex: -len(_simplex), ValueError),
            (lambda _simplex: math.inf, ValueError),
            (lambda _simplex: None, TypeError),
        ):
            with self.subTest(function=_function):
                with self.assertRaises(_error):
                    custom_filtration(self.complex, _function)


class ConstructorTests(unittest.TestCase):
    """Check dispatch and the GUDHI construction adapters."""

    def setUp(self):
        self.points = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

    def test_dispatcher_normalizes_names_and_passes_arguments(self):
        _complex = SimplicialComplex([[0, 1]])
        _values = {(0,): 0, (1,): 1, (0, 1): 2}
        _tree = build_filtration(" EXPLICIT ", _complex, _values=_values)
        self.assertEqual(_tree.filtration([0, 1]), 2.0)

        with patch("canopytda.filtration.simplex_tree_from_point_cloud") as _rips:
            _result = build_filtration(
                "Vietoris-Rips", self.points,
                _max_edge_length=2, _max_dimension=1,
            )
        self.assertIs(_result, _rips.return_value)
        _rips.assert_called_once_with(self.points, 2, 1)

    def test_dispatcher_lists_supported_names_and_rejects_unknown(self):
        self.assertEqual(available_filtrations(), (
            "explicit", "lower_star", "rips", "alpha", "delaunay_cech",
        ))
        with self.assertRaises(TypeError):
            build_filtration(None, self.points)
        with self.assertRaises(ValueError):
            build_filtration("unknown", self.points)

    def test_rips_builder_forwards_point_cloud_and_default_dimension(self):
        with patch("canopytda.filtration.simplex_tree_from_point_cloud") as _rips:
            self.assertIs(rips_filtration(self.points, 1.5), _rips.return_value)
        _rips.assert_called_once_with(self.points, 1.5, 2)

    def test_alpha_builder_forwards_squared_threshold_and_prunes(self):
        _tree = Mock()
        with patch("canopytda.filtration.gd.AlphaComplex") as _alpha:
            _alpha.return_value.create_simplex_tree.return_value = _tree
            _result = alpha_filtration(
                self.points, _max_alpha_square=4, _max_dimension=1,
                _precision="exact", _output_squared_values=False,
            )

        self.assertIs(_result, _tree)
        _alpha.assert_called_once_with(points=self.points, precision="exact")
        _alpha.return_value.create_simplex_tree.assert_called_once_with(
            max_alpha_square=4.0, output_squared_values=False,
        )
        _tree.prune_above_dimension.assert_called_once_with(1)

    def test_cech_alias_uses_the_delaunay_constructor(self):
        _tree = Mock()
        with patch("canopytda.filtration.gd.DelaunayCechComplex") as _cech:
            _cech.return_value.create_simplex_tree.return_value = _tree
            _result = build_filtration(
                "CECH", self.points, _max_radius_square=9,
                _max_dimension=2, _output_squared_values=True,
            )

        self.assertIs(_result, _tree)
        _cech.assert_called_once_with(points=self.points, precision="safe")
        _cech.return_value.create_simplex_tree.assert_called_once_with(
            max_alpha_square=9.0, output_squared_values=True,
        )
        _tree.prune_above_dimension.assert_called_once_with(2)

    def test_unbounded_cech_keeps_all_dimensions(self):
        _tree = Mock()
        with patch("canopytda.filtration.gd.DelaunayCechComplex") as _cech:
            _cech.return_value.create_simplex_tree.return_value = _tree
            self.assertIs(delaunay_cech_filtration(self.points), _tree)
        _cech.return_value.create_simplex_tree.assert_called_once_with(
            max_alpha_square=math.inf, output_squared_values=True,
        )
        _tree.prune_above_dimension.assert_not_called()

    def test_geometric_builders_reject_empty_or_unsized_clouds(self):
        for _constructor in (alpha_filtration, delaunay_cech_filtration):
            with self.subTest(constructor=_constructor.__name__):
                with self.assertRaises(ValueError):
                    _constructor([])
                with self.assertRaises(TypeError):
                    _constructor(iter(self.points))

    def test_geometric_builders_reject_bad_thresholds(self):
        for _constructor, _keyword in (
            (alpha_filtration, "_max_alpha_square"),
            (delaunay_cech_filtration, "_max_radius_square"),
        ):
            for _value, _error in ((-1, ValueError), (math.nan, ValueError),
                                   ("bad", TypeError)):
                with self.subTest(builder=_constructor.__name__, value=_value):
                    with self.assertRaises(_error):
                        _constructor(self.points, **{_keyword: _value})

    def test_geometric_builders_reject_invalid_max_dimension(self):
        for _constructor in (alpha_filtration, delaunay_cech_filtration):
            for _value in (-1, 1.2, True):
                with self.subTest(builder=_constructor.__name__, value=_value):
                    with patch("canopytda.filtration.gd." + (
                        "AlphaComplex" if _constructor is alpha_filtration
                        else "DelaunayCechComplex"
                    )):
                        with self.assertRaises(ValueError):
                            _constructor(self.points, _max_dimension=_value)


if __name__ == "__main__":
    unittest.main()
