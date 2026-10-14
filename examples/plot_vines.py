"""Visualize connected vines and a retained diagonal-limit point.

Run from the repository root::

    python -m examples.plot_vines outputs/vine_decompositions.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from canopytda.pairing import STAR

from data.paper_complexes import L_SIMPLEX_NAMES

from .demo_vines import example_vines


def _pair_label(_pair):
    """Format simplex pair labels with the paper's L notation."""
    _birth, _death = _pair
    _first = L_SIMPLEX_NAMES.get(_birth, "".join(map(str, _birth)))
    _last = (r"$\star$" if _death == STAR else
             L_SIMPLEX_NAMES.get(_death, "".join(map(str, _death))))
    return "({}, {})".format(_first, _last)


def _draw(_axis, _cell, _height, _color, _style="-", _marker="o"):
    """Render one interval segment or one isolated stratum point."""
    if _cell["kind"] == "interval":
        _axis.plot((_cell["start"], _cell["end"]),
                   (_height, _height), color=_color,
                   linestyle=_style, linewidth=3.5,
                   solid_capstyle="butt", zorder=2)
    else:
        _axis.scatter((_cell["start"],), (_height,), s=86,
                      marker=_marker, color=_color,
                      edgecolor="#263746", linewidth=0.75, zorder=5)


def _setup_axis(_axis, _pairs, _labels, _title):
    """Give each chosen pair a fixed lane along the parameter base."""
    _axis.set_xlim(-0.04, 1.04)
    _axis.set_ylim(-0.6, len(_pairs) - 0.4)
    _axis.set_yticks(range(len(_pairs)),
                     tuple(_pair_label(_pair) for _pair in _pairs))
    _axis.set_xticks(_labels)
    _axis.set_xlabel("parameter t")
    _axis.set_title(_title, fontsize=12.1, pad=10)
    _axis.grid(axis="x", alpha=0.19)


def main(_output_path="outputs/vine_decompositions.png"):
    """Save a two-panel figure drawn directly from the vine records."""
    _, _a_vines, _d_canopy, _d_vines = example_vines()
    _figure, (_left, _right) = plt.subplots(1, 2, figsize=(14.4, 5.2),
                                            facecolor="white")
    _figure.subplots_adjust(left=0.09, right=0.97, wspace=0.32,
                            top=0.76, bottom=0.19)

    _a_cells = _a_vines["source"]["cells"]
    _a_pairs = tuple(sorted({_cell["pair"] for _cell in _a_cells}, key=repr))
    _a_lanes = {_pair: _index for _index, _pair in enumerate(_a_pairs)}
    _a_by_id = {_cell["cell_id"]: _cell for _cell in _a_cells}
    _a_palette = ("#6660a9", "#299392", "#a05884", "#d08345")
    for _vine in _a_vines.vines:
        for _cell_id in _vine.cell_ids:
            _cell = _a_by_id[_cell_id]
            _draw(_left, _cell, _a_lanes[_cell["pair"]],
                  _a_palette[_vine["vine_id"] % len(_a_palette)])
    _left.axvline(1 / 3, linewidth=1.5, color="#c45c31", alpha=0.65)
    _setup_axis(_left, _a_pairs, (0, 1 / 3, 3 / 4, 1),
                "L example: four pieces, one connected A-vine")
    _left.set_xticklabels(("0", "1/3", "3/4", "1"))
    _left.text(1 / 3 + 0.018, len(_a_pairs) - 1.43,
               "two distinct event points", fontsize=9.2,
               color="#9e4528", va="center")

    _d_cells = _d_canopy["ambient_refinement"]["cells"]
    _d_pairs = tuple(sorted({_cell["pair"] for _cell in _d_cells}, key=repr))
    _d_lanes = {_pair: _index for _index, _pair in enumerate(_d_pairs)}
    _removed = set(_d_canopy["removed_diagonal_cell_ids"])
    for _cell in _d_cells:
        if _cell["cell_id"] in _removed:
            _draw(_right, _cell, _d_lanes[_cell["pair"]], "#acb6bf",
                  _style="--")
    _d_by_id = {_cell["cell_id"]: _cell for _cell in _d_canopy.cells}
    _d_palette = ("#207d9a", "#75a86c", "#a068b6")
    for _vine in _d_vines.vines:
        for _cell_id in _vine.cell_ids:
            _cell = _d_by_id[_cell_id]
            _boundary = _cell["status"] == "diagonal_boundary"
            _draw(_right, _cell, _d_lanes[_cell["pair"]],
                  "#c45c31" if _boundary else
                  _d_palette[_vine["vine_id"] % len(_d_palette)],
                  _marker="D" if _boundary else "o")
    _setup_axis(_right, _d_pairs, (0, 0.5, 1),
                "Diagonal example: three connected D-vines")
    _right.set_xticklabels(("0", "1/2", "1"))

    _paper_vine, = _a_vines.vines
    assert len(_paper_vine.fiber(1 / 3)) == 2
    assert not _paper_vine.regular
    assert sum(len(_vine.fiber(0.5)) for _vine in _d_vines.vines) == 3
    assert len(_d_canopy.boundary_cells()) == 1
    _figure.suptitle("Vines are connected canopy components",
                     fontsize=16, color="#23354a", y=0.97)
    _figure.text(0.5, 0.84,
                 "A: event points share a vine without merging.  D: the orange diagonal limit connects the finite sheet across t = 1/2.",
                 ha="center", fontsize=10.3, color="#4b6073")
    _figure.text(0.5, 0.055,
                 "Lines represent symbolic interval sheets; dots are point strata. Dashed gray cells are absent from the D-canopy.",
                 ha="center", fontsize=9.4, color="#4b6073")
    _destination = Path(_output_path).expanduser().resolve()
    _destination.parent.mkdir(parents=True, exist_ok=True)
    _figure.savefig(_destination, dpi=180, facecolor="white")
    plt.close(_figure)
    print("Saved", _destination)
    return _destination


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/vine_decompositions.png")
