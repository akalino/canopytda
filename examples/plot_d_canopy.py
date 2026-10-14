"""Plot retained, removed and diagonal-limit sheets from D-canopy cells.

Run from the repository root::

    python -m examples.plot_d_canopy outputs/d_canopy_diagonal_entry.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from canopytda.d_canopy import DELTA
from canopytda.pairing import STAR

from data.paper_complexes import D_ALWAYS_DIAGONAL_PAIR
from data.paper_complexes import D_FINITE_PAIR

from .demo_d_canopy import example_d_canopy


def _pair_label(_pair):
    """Format a chosen pair for the y axis."""
    _birth, _death = _pair
    _first = "".join(str(_vertex) for _vertex in _birth)
    _last = (r"$\star$" if _death == STAR else
             "".join(str(_vertex) for _vertex in _death))
    return "({}, {})".format(_first, _last)


def _cell_segment(_axis, _cell, _height, _color, _style="-", _marker="o"):
    """Draw one symbolic constructible cell at a fixed pair lane."""
    if _cell["kind"] == "interval":
        _axis.plot((_cell["start"], _cell["end"]),
                   (_height, _height), color=_color,
                   linestyle=_style, linewidth=3.4, solid_capstyle="butt")
    else:
        _axis.scatter((_cell["start"],), (_height,),
                      color=_color, edgecolor="#263746", linewidth=0.8,
                      s=78, marker=_marker, zorder=4)


def main(_output_path="outputs/d_canopy_diagonal_entry.png"):
    """Save the computed D-space and the finite persistence trajectory."""
    _d_canopy = example_d_canopy()
    _ambient = _d_canopy["ambient_refinement"]["cells"]
    _retained = _d_canopy.cells
    _removed_ids = set(_d_canopy["removed_diagonal_cell_ids"])
    _pairs = sorted({_cell["pair"] for _cell in _ambient}, key=repr)
    _lanes = {_pair: _index for _index, _pair in enumerate(_pairs)}

    _figure, (_space, _gap) = plt.subplots(1, 2, figsize=(12, 4.8),
                                           facecolor="white")
    _figure.subplots_adjust(left=0.09, right=0.96, wspace=0.28,
                            top=0.75, bottom=0.20)
    for _cell in _ambient:
        if _cell["cell_id"] in _removed_ids:
            _cell_segment(_space, _cell, _lanes[_cell["pair"]],
                          "#a5aeb7", _style="--")
    for _cell in _retained:
        _limit = _cell["status"] == "diagonal_boundary"
        _cell_segment(_space, _cell, _lanes[_cell["pair"]],
                      "#c45c31" if _limit else "#207d9a",
                      _marker="D" if _limit else "o")
    _space.set_yticks(tuple(_lanes.values()),
                      tuple(_pair_label(_pair) for _pair in _lanes))
    _space.set_ylim(-0.6, len(_lanes) - 0.4)
    _space.set_xlim(-0.035, 1.035)
    _space.set_xlabel("parameter t")
    _space.set_title("Retained sheets and removed diagonal sheet")
    _space.grid(axis="x", alpha=0.2)

    _finite = tuple(_cell for _cell in _ambient
                    if _cell["pair"] == D_FINITE_PAIR)
    for _cell in _finite:
        _left = _cell["start"]
        _right = _cell["end"]
        _left_value = next(_point["eta"] for _point in
                           _d_canopy.fiber(_left) if
                           _point["pair"] == D_FINITE_PAIR)
        if _cell["kind"] == "interval":
            _right_value = next(_point["eta"] for _point in
                                _d_canopy.fiber(_right) if
                                _point["pair"] == D_FINITE_PAIR)
            _gap.plot((_left, _right),
                      (_left_value[1] - _left_value[0],
                       _right_value[1] - _right_value[0]),
                      color="#207d9a", linewidth=3)
    _limit, = _d_canopy.boundary_cells()
    assert _limit["representative_eta_prime"] == DELTA
    _gap.scatter((_limit["start"],), (0,), marker="D", s=105,
                 edgecolor="#263746", color="#c45c31", zorder=5)
    _gap.annotate(r"$\Delta$: retained limit", (0.5, 0),
                  xytext=(15, 17), textcoords="offset points")
    _gap.axhline(0, color="#7c8790", linewidth=1)
    _gap.set_xlim(-0.035, 1.035)
    _gap.set_ylim(-0.11, 1.18)
    _gap.set_xlabel("parameter t")
    _gap.set_ylabel("finite persistence (death − birth)")
    _gap.set_title("D-diagram omits the middle limit")
    _gap.grid(alpha=0.2)

    _figure.suptitle("D-canopy: closure of the off-diagonal locus",
                     fontsize=15, color="#23354a", y=0.97)
    _figure.text(0.5, 0.82,
                 "Teal: off diagonal · orange diamond: retained diagonal limit · dashed gray: removed",
                 ha="center", fontsize=10.5, color="#4b6073")
    _destination = Path(_output_path).expanduser().resolve()
    _destination.parent.mkdir(parents=True, exist_ok=True)
    _figure.savefig(_destination, dpi=180, facecolor="white")
    plt.close(_figure)
    print("Saved", _destination)
    return _destination


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "outputs/d_canopy_diagonal_entry.png")
