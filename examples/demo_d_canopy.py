"""Demonstrate the D-canopy closure and quotient diagram on two edges.

Run from the repository root::

    python -m examples.demo_d_canopy
"""

from canopytda.a_canopy import ACanopy
from canopytda.d_canopy import DELTA
from canopytda.d_canopy import DCanopy
from canopytda.stratification import Stratification
from canopytda.topology import CanopyTopology

from data.paper_complexes import D_ALWAYS_DIAGONAL_PAIR
from data.paper_complexes import D_DIAGONAL_FILTRATIONS
from data.paper_complexes import D_DIAGONAL_PARAMETERS
from data.paper_complexes import D_FINITE_PAIR


def example_d_canopy():
    """Build the fixed-complex diagonal-entry example."""
    _family = Stratification(D_DIAGONAL_FILTRATIONS,
                             D_DIAGONAL_PARAMETERS)
    _canopy = ACanopy(_family, _homology_dimension=0)
    return DCanopy(_canopy, CanopyTopology(_canopy))


def main():
    """Print the retained boundary and diminished diagrams."""
    _d_canopy = example_d_canopy()
    _boundaries = _d_canopy.boundary_cells()
    _ambient = _d_canopy["ambient_refinement"]["cells"]
    _removed = set(_d_canopy["removed_diagonal_cell_ids"])
    print("D-canopy: a finite H0 interval touches the diagonal")
    print("  A-canopy order events:",
          _d_canopy["source_a_canopy"]["stratification"]["events"])
    print("  retained diagonal limits:",
          tuple((_cell["pair"], _cell["start"]) for _cell in _boundaries))
    print("  always-diagonal cells removed:",
          sum(_cell["cell_id"] in _removed and
              _cell["pair"] == D_ALWAYS_DIAGONAL_PAIR
              for _cell in _ambient))
    for _parameter in (0.0, 0.25, 0.5, 0.75, 1.0):
        _finite = next(_point for _point in _d_canopy.fiber(_parameter)
                       if _point["pair"] == D_FINITE_PAIR)
        print("  t={:.2f}: finite pair maps to {}; diminished H0 = {}".format(
            _parameter, _finite["eta_prime"],
            _d_canopy.diagram(_parameter, 0)))
    assert len(_boundaries) == 1
    assert next(_point["eta_prime"] for _point in
                _d_canopy.fiber(0.5) if
                _point["pair"] == D_FINITE_PAIR) == DELTA
    assert _d_canopy["local_chart_certificate"]["all_charts_valid"]
    return _d_canopy


if __name__ == "__main__":
    main()
