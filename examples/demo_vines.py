"""Compare connected A-vines and D-vines on paper and diagonal examples.

Run from the repository root::

    python -m examples.demo_vines
"""

from canopytda.a_canopy import ACanopy
from canopytda.d_canopy import DCanopy
from canopytda.stratification import Stratification
from canopytda.topology import CanopyTopology
from canopytda.vines import VineDecomposition

from data.paper_complexes import D_DIAGONAL_FILTRATIONS
from data.paper_complexes import D_DIAGONAL_PARAMETERS
from data.paper_complexes import D_FINITE_PAIR
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS


def example_vines():
    """Build the paper's L example and a separate diagonal-entry example."""
    _family = Stratification(PAPER_3_7_FILTRATIONS,
                             PAPER_3_7_PARAMETERS)
    _a_canopy = ACanopy(_family, _homology_dimension=1)
    _a_vines = VineDecomposition(CanopyTopology(_a_canopy))
    _d_family = Stratification(D_DIAGONAL_FILTRATIONS,
                                D_DIAGONAL_PARAMETERS)
    _d_canopy = DCanopy(ACanopy(_d_family, _homology_dimension=0))
    _d_vines = VineDecomposition(_d_canopy)
    return _a_canopy, _a_vines, _d_canopy, _d_vines


def main():
    """Print the connected components, event multiplicity and D-limits."""
    _a_canopy, _a_vines, _d_canopy, _d_vines = example_vines()
    _paper_vine, = _a_vines.vines
    _at_event = _paper_vine.fiber(1 / 3)
    print("Example 3.12: A-vine decomposition of the L complex")
    print("  connected A-vines:", len(_a_vines.vines))
    print("  distinct points at t=1/3:",
          tuple(_point["pair"] for _point in _at_event))
    print("  diagram values at t=1/3:", _paper_vine.diagram(1 / 3, 1))
    print("  non-Hausdorff parameters:",
          tuple(_event["parameter"] for _event in
                _paper_vine["non_hausdorff_fibers"]))
    print("  regular:", _paper_vine.regular)

    _finite, = (_vine for _vine in _d_vines.vines if
                any(_point["pair"] == D_FINITE_PAIR for _point in
                    _vine.fiber(0.25)))
    print("Diagonal example: D-vine decomposition of two edges")
    print("  connected D-vines:", len(_d_vines.vines))
    print("  finite vine at t=1/2:",
          tuple(_point["eta_prime"] for _point in _finite.fiber(0.5)))
    print("  finite-vine diagram at t=1/2:", _finite.diagram(0.5, 0))
    print("  removed diagonal cells:",
          len(_d_canopy["removed_diagonal_cell_ids"]))
    assert len(_at_event) == 2
    assert len(_a_vines.vines) == 1
    assert len(_d_vines.vines) == 3
    assert _a_canopy.diagram(1 / 3, 1) == _paper_vine.diagram(1 / 3, 1)
    return _a_vines, _d_vines


if __name__ == "__main__":
    main()
