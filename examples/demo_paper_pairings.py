"""Compute simplex pairings and diagrams for fixed paper examples.

From the repository root, with canopytda installed, run::

    python -m examples.demo_paper_pairings

The printed simplex pairs are computed from the boundary matrix. The
expected L pairs are independent fixtures from Examples 2.17 and 2.19.
"""

from canopytda.pairing import PersistencePairing

from data.paper_complexes import A_SIMPLEX
from data.paper_complexes import B_SIMPLEX
from data.paper_complexes import ESSENTIAL_LOOP_FILTRATION
from data.paper_complexes import FILLED_TRIANGLE_FILTRATION
from data.paper_complexes import L_EARLY_SIMPLICES
from data.paper_complexes import L_F_FILTRATION
from data.paper_complexes import L_G_FILTRATION
from data.paper_complexes import L_G_ORDER_ONE
from data.paper_complexes import L_G_ORDER_TWO
from data.paper_complexes import L_G_PAIRS_ONE
from data.paper_complexes import L_G_PAIRS_TWO
from data.paper_complexes import TWO_DISJOINT_EDGES_FILTRATION
from data.paper_complexes import TWO_DISJOINT_EDGES_PAIRS
from data.paper_complexes import C_SIMPLEX
from data.paper_complexes import D_SIMPLEX


def _named_pairs(_pairing):
    """Select the named degree-one pairs of the L complex.

    :param _pairing: Computed L simplex pairing.
    :type _pairing: canopytda.pairing.PersistencePairing
    :return: Birth and death simplices for a and b.
    :rtype: frozenset of pairs
    """
    return frozenset(
        _pair for _pair in _pairing.finite_pairs
        if _pair[0] in (A_SIMPLEX, B_SIMPLEX)
    )


def main():
    """Run paper pairing checks and print the resulting diagrams."""
    print("PAPER EXAMPLE 2.19: TWO ORDERS ON L")
    for _label, _order, _expected in (
            ("a,b,c,d", L_G_ORDER_ONE, L_G_PAIRS_ONE),
            ("a,b,d,c", L_G_ORDER_TWO, L_G_PAIRS_TWO)):
        _pairing = PersistencePairing(L_G_FILTRATION, _order)
        _computed = _named_pairs(_pairing)
        if _computed != frozenset(_expected):
            raise AssertionError("L simplex pairings differ from the paper")
        print("  order {}: named pairs {}".format(
            _label, tuple(sorted(_computed))))
        print("    H1 augmented:", _pairing.augmented_diagram(1))
        print("    H1 diminished:", _pairing.diminished_diagram(1))

    print("\nPAPER EXAMPLE 2.17: TIED VALUES ON L")
    _order = L_EARLY_SIMPLICES + (
        B_SIMPLEX, A_SIMPLEX, D_SIMPLEX, C_SIMPLEX
    )
    _pairing = PersistencePairing(L_F_FILTRATION, _order)
    print("  named pairs:", tuple(sorted(_named_pairs(_pairing))))
    print("  H1 augmented (two equal points):",
          _pairing.augmented_diagram(1))

    print("\nDIAGONAL AND ESSENTIAL CLASSES")
    _filled = PersistencePairing(FILLED_TRIANGLE_FILTRATION)
    _unfilled = PersistencePairing(ESSENTIAL_LOOP_FILTRATION)
    print("  filled triangle H1 augmented:",
          _filled.augmented_diagram(1))
    print("  filled triangle H1 diminished:",
          _filled.diminished_diagram(1))
    print("  triangle boundary H1 augmented:",
          _unfilled.augmented_diagram(1))
    print("  triangle boundary essential pairs:",
          _unfilled.essential_pairs)

    print("\nPAPER EXAMPLE 3.16: SEPARATE COMPONENTS")
    _separate = PersistencePairing(TWO_DISJOINT_EDGES_FILTRATION)
    if set(_separate.finite_pairs) != set(TWO_DISJOINT_EDGES_PAIRS):
        raise AssertionError("Disconnected-edge pairings differ from the paper")
    print("  distinct simplex pairs:", _separate.finite_pairs)
    print("  H0 diagram (duplicates retained):",
          _separate.augmented_diagram(0))


if __name__ == "__main__":
    main()
