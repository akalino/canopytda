"""Reproduce the L-complex equivalences in paper Examples 2.19 and 2.22.

From the repository root, run::

    python -m examples.demo_equivalence
"""

from canopytda.equivalence import CombinatorialEquivalences
from canopytda.equivalence import FiltrationEquivalences
from canopytda.equivalence import common_refinement
from canopytda.equivalence import filtrations_compatible

from data.paper_complexes import A_SIMPLEX
from data.paper_complexes import B_SIMPLEX
from data.paper_complexes import C_SIMPLEX
from data.paper_complexes import D_SIMPLEX
from data.paper_complexes import L_EARLY_SIMPLICES
from data.paper_complexes import L_F_FILTRATION
from data.paper_complexes import L_G_FILTRATION
from data.paper_complexes import L_G_ORDER_ONE
from data.paper_complexes import L_G_ORDER_TWO
from data.paper_complexes import L_SIMPLEX_NAMES


def _named_maps(_mappings):
    """Show the source and target H1 pairs using the paper's labels."""
    return tuple(sorted({tuple(sorted(
        ((L_SIMPLEX_NAMES[_source[0]], L_SIMPLEX_NAMES[_source[1]]),
         (L_SIMPLEX_NAMES[_target[0]], L_SIMPLEX_NAMES[_target[1]]))
        for _source, _target in _mapping
        if _source[0] in (A_SIMPLEX, B_SIMPLEX)))
        for _mapping in _mappings}))


def main():
    """Print certified fixed-complex equivalences and their counts."""
    print("Paper f and g compatible:",
          filtrations_compatible(L_F_FILTRATION, L_G_FILTRATION))
    print("Common order:", common_refinement(
        L_F_FILTRATION, L_G_FILTRATION)[-4:])

    _g_equivalences = FiltrationEquivalences(
        L_G_FILTRATION, L_G_ORDER_ONE, L_G_ORDER_TWO)
    print("Example 2.19, g:", _named_maps(_g_equivalences.mappings))

    _f_order = L_EARLY_SIMPLICES + (
        B_SIMPLEX, A_SIMPLEX, D_SIMPLEX, C_SIMPLEX)
    _f_equivalences = FiltrationEquivalences(
        L_F_FILTRATION, _f_order, _f_order)
    print("Example 2.19, f self-maps:",
          _named_maps(_f_equivalences.mappings))

    _first_order = L_EARLY_SIMPLICES + (
        B_SIMPLEX, A_SIMPLEX, C_SIMPLEX, D_SIMPLEX)
    _combined = CombinatorialEquivalences(
        L_F_FILTRATION, L_G_FILTRATION, _first_order, L_G_ORDER_TWO)
    print("Example 2.22, f to g:", _named_maps(_combined.mappings))
    print("All searches complete:", all((
        _g_equivalences.complete, _f_equivalences.complete,
        _combined.complete)))


if __name__ == "__main__":
    main()
