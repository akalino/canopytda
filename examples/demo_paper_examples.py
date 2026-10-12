"""Inspect fixed paper examples with the public canopytda classes.

From the repository root, with canopytda installed, run::

    python -m examples.demo_fixed_paper

Pairings shown below are expected data from paper_complexes.py. The current
six-module package validates orders and constructs boundary columns; it does
not compute persistence pairings.
"""

from canopytda.complex import boundary_columns
from canopytda.filtration import Filtration
from canopytda.filtration import custom_filtration
from canopytda.orders import MinimalConsistentOrder
from canopytda.orders import validate_refinement
from canopytda.stratification import Stratification

from data.paper_complexes import A_SIMPLEX
from data.paper_complexes import B_SIMPLEX
from data.paper_complexes import C_SIMPLEX
from data.paper_complexes import D_SIMPLEX
from data.paper_complexes import ESSENTIAL_LOOP_COMPLEX
from data.paper_complexes import L_COMPLEX
from data.paper_complexes import L_F_FILTRATION
from data.paper_complexes import L_G_FILTRATION
from data.paper_complexes import L_G_ORDER_ONE
from data.paper_complexes import L_G_ORDER_TWO
from data.paper_complexes import L_G_PAIRS_ONE
from data.paper_complexes import L_G_PAIRS_TWO
from data.paper_complexes import PAPER_COMPLEXES
from data.paper_complexes import SINGLE_EDGE_COMPLEX
from data.paper_complexes import SINGLE_EDGE_FILTRATION
from data.paper_complexes import SINGLE_EDGE_ORDER_ONE
from data.paper_complexes import SINGLE_EDGE_ORDER_TWO
from data.paper_complexes import SQUARE_LOOP_COMPLEX


def _value_pairs(_filtration, _pairs):
    """Give sorted (birth, death) values for expected simplex pairs.

    :param _filtration: Fixed filtration of all the paired simplices.
    :type _filtration: canopytda.filtration.Filtration
    :param _pairs: Expected simplex pairs supplied by the example data.
    :type _pairs: iterable of pairs of simplices
    :return: Sorted values, with multiplicity retained.
    :rtype: tuple of tuple of float
    """
    return tuple(sorted(
        (_filtration.value(_birth), _filtration.value(_death))
        for _birth, _death in _pairs
    ))


def main():
    """Run four short demonstrations and print their computed outputs."""
    print("COMPLEXES")
    for _name, _complex in PAPER_COMPLEXES.items():
        print("  {}: dimension {}, {} simplices, facets {}".format(
            _name, _complex.dimension, len(_complex), _complex.facets
        ))
    print("  triangle faces:", L_COMPLEX.faces(C_SIMPLEX))
    print("  cofaces of a in L:", L_COMPLEX.cofaces(A_SIMPLEX))
    print("  unfilled triangle facets:", ESSENTIAL_LOOP_COMPLEX.facets)
    print("  edge boundary columns:", boundary_columns(SINGLE_EDGE_COMPLEX))

    print("\nFILTRATIONS")
    print("  L, paper f (a,b,c,d):", tuple(
        L_F_FILTRATION.value(_simplex)
        for _simplex in (A_SIMPLEX, B_SIMPLEX, C_SIMPLEX, D_SIMPLEX)
    ))
    print("  L, paper g (a,b,c,d):", tuple(
        L_G_FILTRATION.value(_simplex)
        for _simplex in (A_SIMPLEX, B_SIMPLEX, C_SIMPLEX, D_SIMPLEX)
    ))
    _lower_star = Filtration.lower_star(
        SQUARE_LOOP_COMPLEX, {0: 0.0, 1: 1.0, 2: 2.0, 3: 1.0}
    )
    _custom = custom_filtration(
        SQUARE_LOOP_COMPLEX, lambda _simplex: float(len(_simplex) - 1)
    )
    print("  lower-star value of (1, 2):", _lower_star.value((1, 2)))
    print("  custom value of (1, 2):", _custom.value((1, 2)))

    print("\nCONSISTENT ORDERS")
    _order = MinimalConsistentOrder(L_G_FILTRATION)
    print("  L, paper g: {} relations, {} covers".format(
        len(_order.relations), len(_order.covers)
    ))
    for _number, _permutation, _pairs in (
            (1, L_G_ORDER_ONE, L_G_PAIRS_ONE),
            (2, L_G_ORDER_TWO, L_G_PAIRS_TWO)):
        validate_refinement(L_G_FILTRATION, _permutation)
        _columns = boundary_columns(L_COMPLEX, _order=_permutation)
        print("  L order {}: named suffix {}, final columns {}, expected pairs {}".format(
            _number, _permutation[-4:], _columns[-2:], _pairs
        ))
    print("  expected pair values, order 1:", _value_pairs(
        L_G_FILTRATION, L_G_PAIRS_ONE))
    print("  expected pair values, order 2:", _value_pairs(
        L_G_FILTRATION, L_G_PAIRS_TWO))
    print("  single edge, two valid orders:", tuple(
        validate_refinement(SINGLE_EDGE_FILTRATION, _permutation)
        for _permutation in (SINGLE_EDGE_ORDER_ONE, SINGLE_EDGE_ORDER_TWO)
    ))

    print("\nONE-POINT STRATIFICATION")
    _stratification = Stratification((SINGLE_EDGE_FILTRATION,))
    print("  strata:", _stratification.strata)
    print("  events:", _stratification.events)
    print("  filtration at the point:", dict(
        _stratification.filtration_at(0.0).values
    ))


if __name__ == "__main__":
    main()
