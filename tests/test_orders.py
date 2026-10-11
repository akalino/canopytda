"""Public API checks for the attached complex, filtration and orders modules.

Run after placing the modules in ``src/canopytda``::

    python -m unittest discover -s tests -p 'test_order_public.py' -v
"""

import math
import unittest

from canopytda.complex import SimplicialComplex
from canopytda.filtration import Filtration
from canopytda.orders import MinimalConsistentOrder
from canopytda.orders import canonical_refinement
from canopytda.orders import is_consistent_refinement
from canopytda.orders import minimal_consistent_covers
from canopytda.orders import minimal_consistent_order
from canopytda.orders import total_order_refinements
from canopytda.orders import validate_refinement


class PublicOrderTests(unittest.TestCase):
    """Check fixed-complex order semantics and legacy tree entry points."""

    def setUp(self):
        self.complex = SimplicialComplex([[0, 1]])
        self.filtration = Filtration(
            self.complex, {(0,): 0, (1,): 0, (0, 1): 0}
        )

    def test_class_exposes_minimal_order_without_breaking_ties(self):
        _order = MinimalConsistentOrder(self.filtration)
        _expected = (((0,), (0, 1)), ((1,), (0, 1)))

        self.assertIs(_order._filtration, self.filtration)
        self.assertIs(_order.complex, self.complex)
        self.assertEqual(_order.relations, _expected)
        self.assertEqual(_order.covers, _expected)
        self.assertNotIn(((0,), (1,)), _order.relations)
        self.assertEqual(minimal_consistent_order(_order), _order.relations)
        self.assertEqual(minimal_consistent_covers(_order), _order.covers)

    def test_class_is_immutable_including_explicit_reinitialization(self):
        _order = MinimalConsistentOrder(self.filtration)
        with self.assertRaises(AttributeError):
            _order._relations = ()
        with self.assertRaises(AttributeError):
            del _order._filtration
        with self.assertRaises(AttributeError):
            _order.__init__(self.filtration)

    def test_canonical_and_all_total_refinements_agree_with_functions(self):
        _order = MinimalConsistentOrder(self.filtration)
        _first = ((0,), (1,), (0, 1))
        _second = ((1,), (0,), (0, 1))

        self.assertEqual(_order.canonical_refinement(), _first)
        self.assertEqual(canonical_refinement(_order), _first)
        self.assertEqual(tuple(_order.total_refinements()), (_first, _second))
        self.assertEqual(tuple(total_order_refinements(_order, _limit=1)),
                         (_first,))
        self.assertEqual(tuple(_order.total_refinements(_limit=0)), ())

    def test_refinement_validation_respects_faces_and_filtration_values(self):
        _order = MinimalConsistentOrder(self.filtration)
        _valid = ((1,), (0,), (1, 0))
        self.assertEqual(_order.validate_refinement(_valid),
                         ((1,), (0,), (0, 1)))
        self.assertTrue(is_consistent_refinement(_order, _valid))

        for _candidate in (
            ((0, 1), (0,), (1,)),
            ((0,), (0,), (0, 1)),
            ((0,), (1,)),
        ):
            with self.subTest(candidate=_candidate):
                with self.assertRaises(ValueError):
                    validate_refinement(_order, _candidate)
                self.assertFalse(_order.is_consistent_refinement(_candidate))

        _strict = Filtration(self.complex, {
            (0,): 0, (1,): 1, (0, 1): 2,
        })
        self.assertFalse(is_consistent_refinement(
            _strict, ((1,), (0,), (0, 1))
        ))

    def test_tree_conversion_captures_values_before_mutation(self):
        _tree = self.filtration.to_simplex_tree()
        _order = MinimalConsistentOrder(_tree)
        self.assertEqual(_order._filtration.value((0, 1)), 0.0)
        _tree.assign_filtration([0, 1], 2)

        self.assertEqual(_order._filtration.value((0, 1)), 0.0)
        self.assertEqual(_order.relations,
                         minimal_consistent_order(self.filtration))
        self.assertEqual(minimal_consistent_order(_tree),
                         (((0,), (0, 1)), ((1,), (0, 1))))

    def test_rejects_invalid_types_and_invalid_raw_trees(self):
        with self.assertRaises(TypeError):
            MinimalConsistentOrder(self.complex)
        with self.assertRaises(TypeError):
            canonical_refinement({})
        _tree = self.complex.to_simplex_tree()
        _tree.assign_filtration([0], 2)
        with self.assertRaises(ValueError):
            MinimalConsistentOrder(_tree)
        _tree.assign_filtration([0], math.inf)
        with self.assertRaises(ValueError):
            minimal_consistent_order(_tree)

    def test_limit_validation(self):
        _order = MinimalConsistentOrder(self.filtration)
        for _limit in (True, -1, 1.5):
            with self.subTest(limit=_limit):
                with self.assertRaises(ValueError):
                    tuple(_order.total_refinements(_limit=_limit))


if __name__ == "__main__":
    unittest.main()
