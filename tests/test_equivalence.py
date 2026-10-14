"""Paper examples and boundary cases for pairing equivalence."""

import unittest

from canopytda.equivalence import CombinatorialEquivalences
from canopytda.equivalence import FiltrationEquivalences
from canopytda.equivalence import common_refinement
from canopytda.equivalence import common_refinements
from canopytda.equivalence import compatibility_conflicts
from canopytda.equivalence import compose_equivalences
from canopytda.equivalence import filtrations_compatible
from canopytda.filtration import Filtration
from canopytda.orders import validate_refinement
from canopytda.pairing import PersistencePairing

from data.paper_complexes import A_SIMPLEX
from data.paper_complexes import B_SIMPLEX
from data.paper_complexes import C_SIMPLEX
from data.paper_complexes import D_SIMPLEX
from data.paper_complexes import L_COMPLEX
from data.paper_complexes import L_EARLY_SIMPLICES
from data.paper_complexes import L_F_FILTRATION
from data.paper_complexes import L_G_FILTRATION
from data.paper_complexes import L_G_ORDER_ONE
from data.paper_complexes import L_G_ORDER_TWO
from data.paper_complexes import TWO_DISJOINT_EDGES_FILTRATION


class EquivalenceTests(unittest.TestCase):
    """Check definitions 2.16, 2.18, and 2.21 on fixed complexes."""

    def test_compatible_filtrations_have_shared_orders(self):
        """The paper's f and g admit at least one shared refinement."""
        self.assertTrue(filtrations_compatible(L_F_FILTRATION, L_G_FILTRATION))
        _order = common_refinement(L_F_FILTRATION, L_G_FILTRATION)
        self.assertEqual(_order, validate_refinement(L_F_FILTRATION, _order))
        self.assertEqual(_order, validate_refinement(L_G_FILTRATION, _order))
        self.assertEqual(next(common_refinements(
            L_F_FILTRATION, L_G_FILTRATION, _limit=1))[-4:],
                         (A_SIMPLEX, B_SIMPLEX, C_SIMPLEX, D_SIMPLEX))

    def test_incompatible_strict_orders_and_different_complex(self):
        """Reject strict reversals and changes to the labeled simplex set."""
        _values = dict(L_F_FILTRATION.values)
        _values[A_SIMPLEX], _values[B_SIMPLEX] = 2.0, 1.0
        _values[C_SIMPLEX] = _values[D_SIMPLEX] = 3.0
        _opposite = Filtration(L_COMPLEX, _values)
        self.assertIn((A_SIMPLEX, B_SIMPLEX),
                      compatibility_conflicts(L_G_FILTRATION, _opposite))
        self.assertFalse(filtrations_compatible(L_G_FILTRATION, _opposite))
        with self.assertRaises(ValueError):
            common_refinement(L_G_FILTRATION, _opposite)
        with self.assertRaises(ValueError):
            compatibility_conflicts(L_F_FILTRATION,
                                    TWO_DISJOINT_EDGES_FILTRATION)

    def test_g_has_only_the_named_birth_preserving_map(self):
        """Distinct birth values prohibit swapping the named H1 classes."""
        _result = FiltrationEquivalences(L_G_FILTRATION, L_G_ORDER_ONE,
                                         L_G_ORDER_TWO)
        self.assertTrue(_result.complete)
        _named = {(_source, _target)
                  for _source, _target in _result.mappings[0]
                  if _source[0] in (A_SIMPLEX, B_SIMPLEX)}
        self.assertEqual(_named, {
            ((A_SIMPLEX, D_SIMPLEX), (A_SIMPLEX, C_SIMPLEX)),
            ((B_SIMPLEX, C_SIMPLEX), (B_SIMPLEX, D_SIMPLEX)),
        })
        self.assertEqual(len({frozenset(
            (_source, _target) for _source, _target in _mapping
            if _source[0] in (A_SIMPLEX, B_SIMPLEX))
            for _mapping in _result.mappings}), 1)

    def test_f_admits_supported_swapped_self_map(self):
        """The crossed pairs in Example 2.19 have an actual order witness."""
        _order = L_EARLY_SIMPLICES + (
            B_SIMPLEX, A_SIMPLEX, D_SIMPLEX, C_SIMPLEX)
        _result = FiltrationEquivalences(L_F_FILTRATION, _order, _order)
        self.assertTrue(_result.complete)
        _named_maps = set()
        for _witness in _result.witnesses:
            _named_maps.add(frozenset(
                (_source, _target)
                for _source, _target in _witness["mapping"]
                if _source[0] in (A_SIMPLEX, B_SIMPLEX)))
        self.assertEqual(len(_named_maps), 2)
        self.assertTrue(any(
            any(_order is not None for _order in _witness["support_orders"])
            for _witness in _result.witnesses))
        for _witness in _result.witnesses:
            for (_source, _target), _support_order in zip(
                    _witness["mapping"], _witness["support_orders"]):
                if _support_order is None:
                    continue
                _support_pairs = set(PersistencePairing(
                    L_F_FILTRATION, _support_order).pairs)
                self.assertIn((_source[0], _target[1]), _support_pairs)
                self.assertIn((_target[0], _source[1]), _support_pairs)

    def test_separate_components_with_identical_diagram_values(self):
        """Identical intervals do not imply equivalence across components."""
        _order = common_refinement(TWO_DISJOINT_EDGES_FILTRATION,
                                   TWO_DISJOINT_EDGES_FILTRATION)
        _result = FiltrationEquivalences(
            TWO_DISJOINT_EDGES_FILTRATION, _order, _order)
        _finite = {((1,), (0, 1)), ((3,), (2, 3))}
        for _mapping in _result.mappings:
            self.assertEqual({(_source, _target) for _source, _target in _mapping
                              if _source in _finite},
                             {(_pair, _pair) for _pair in _finite})

    def test_example_222_both_compositions(self):
        """The two named continuations through a shared order both occur."""
        _first = L_EARLY_SIMPLICES + (
            B_SIMPLEX, A_SIMPLEX, C_SIMPLEX, D_SIMPLEX)
        _result = CombinatorialEquivalences(
            L_F_FILTRATION, L_G_FILTRATION, _first, L_G_ORDER_TWO)
        self.assertTrue(_result.complete)
        _named_maps = {frozenset(
            (_source, _target) for _source, _target in _mapping
            if _source[0] in (A_SIMPLEX, B_SIMPLEX))
            for _mapping in _result.mappings}
        self.assertEqual(len(_named_maps), 2)

    def test_bounds_and_composition(self):
        """A bounded search never claims a complete map enumeration."""
        _order = L_EARLY_SIMPLICES + (
            B_SIMPLEX, A_SIMPLEX, D_SIMPLEX, C_SIMPLEX)
        with self.assertRaises(RuntimeError):
            FiltrationEquivalences(L_F_FILTRATION, _order, _order,
                                   _max_refinements=1)
        _result = FiltrationEquivalences(
            L_F_FILTRATION, _order, _order, _max_equivalences=1)
        self.assertFalse(_result.complete)
        self.assertEqual(len(_result.mappings), 1)
        _identity = (("a", "b"), ("c", "d"))
        self.assertEqual(compose_equivalences(
            _identity, (("d", "y"), ("b", "x"))),
            (("a", "x"), ("c", "y")))
        with self.assertRaises(ValueError):
            compose_equivalences(_identity, (("z", "y"),))


if __name__ == "__main__":
    unittest.main()
