"""Draw selected fixed examples, colored by filtration value.

From the repository root, with canopytda and matplotlib installed, run::

    python -m examples.plot_fixed_paper

To save to a different path, pass its filename as the first argument.
Coordinates below are solely for drawing; the abstract complexes and their
filtrations are defined in data/paper_complexes.py.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

from data.paper_complexes import A_SIMPLEX
from data.paper_complexes import B_SIMPLEX
from data.paper_complexes import C_SIMPLEX
from data.paper_complexes import D_SIMPLEX
from data.paper_complexes import FILLED_TRIANGLE_FILTRATION
from data.paper_complexes import L_G_FILTRATION
from data.paper_complexes import LOOP_KILLED_BY_TRIANGLE_FILTRATION
from data.paper_complexes import SINGLE_EDGE_FILTRATION
from data.paper_complexes import SQUARE_LOOP_FILTRATION
from data.paper_complexes import TWO_DISJOINT_EDGES_FILTRATION


TRIANGLE_COORDINATES = {
    0: (0.50, 0.88), 1: (0.12, 0.17), 2: (0.88, 0.17),
}
PANELS = (
    ("Single edge", SINGLE_EDGE_FILTRATION,
     {0: (0.12, 0.50), 1: (0.88, 0.50)}),
    ("Filled triangle", FILLED_TRIANGLE_FILTRATION,
     TRIANGLE_COORDINATES),
    ("Loop killed by triangle", LOOP_KILLED_BY_TRIANGLE_FILTRATION,
     TRIANGLE_COORDINATES),
    ("Paper L (filtration g)", L_G_FILTRATION,
     {0: (0.12, 0.88), 1: (0.50, 0.82),
      2: (0.50, 0.18), 3: (0.88, 0.55)}),
    ("Two disjoint edges", TWO_DISJOINT_EDGES_FILTRATION,
     {0: (0.08, 0.72), 1: (0.41, 0.72),
      2: (0.59, 0.28), 3: (0.92, 0.28)}),
    ("Square loop", SQUARE_LOOP_FILTRATION,
     {0: (0.18, 0.18), 1: (0.82, 0.18),
      2: (0.82, 0.82), 3: (0.18, 0.82)}),
)


def _midpoint(_simplex, _coordinates):
    """Return the drawing center of a simplex.

    :param _simplex: Vertices of a simplex.
    :type _simplex: tuple of int
    :param _coordinates: Drawing positions for vertices.
    :type _coordinates: mapping
    :return: Average planar position.
    :rtype: tuple of float
    """
    _count = len(_simplex)
    return (
        sum(_coordinates[_vertex][0] for _vertex in _simplex) / _count,
        sum(_coordinates[_vertex][1] for _vertex in _simplex) / _count,
    )


def _draw_example(_ax, _title, _filtration, _coordinates, _cmap, _scale):
    """Draw vertices, edges and filled triangles for a fixed example.

    :param _ax: Target Matplotlib axes.
    :type _ax: matplotlib.axes.Axes
    :param _title: Panel title.
    :type _title: str
    :param _filtration: One fixed filtration.
    :type _filtration: canopytda.filtration.Filtration
    :param _coordinates: Planar vertex positions used for display only.
    :type _coordinates: mapping
    :param _cmap: Colormap for filtration values.
    :type _cmap: matplotlib.colors.Colormap
    :param _scale: Shared value normalization.
    :type _scale: matplotlib.colors.Normalize
    """
    _complex = _filtration.complex
    for _simplex in _complex.simplices:
        if len(_simplex) == 3:
            _points = [_coordinates[_vertex] for _vertex in _simplex]
            _ax.add_patch(Polygon(
                _points, closed=True,
                facecolor=_cmap(_scale(_filtration.value(_simplex))),
                edgecolor="none", alpha=0.33,
            ))

    for _simplex in _complex.simplices:
        if len(_simplex) == 2:
            _first, _second = (_coordinates[_vertex]
                               for _vertex in _simplex)
            _ax.plot((_first[0], _second[0]), (_first[1], _second[1]),
                     color=_cmap(_scale(_filtration.value(_simplex))),
                     linewidth=3.5, solid_capstyle="round", zorder=2)

    for _vertex in _complex.vertices:
        _x, _y = _coordinates[_vertex]
        _ax.scatter(_x, _y, s=130,
                    color=_cmap(_scale(_filtration.value((_vertex,)))),
                    edgecolor="#183047",
                    linewidth=1.8, zorder=3)
        _ax.annotate(str(_vertex), (_x, _y), (8, 8),
                     textcoords="offset points", ha="left",
                     va="bottom", fontsize=10, color="#183047",
                     bbox={"facecolor": "white", "alpha": 0.8,
                           "edgecolor": "none", "pad": 0.6})

    if _filtration is L_G_FILTRATION:
        for _simplex, _letter in (
                (A_SIMPLEX, "a"), (B_SIMPLEX, "b"),
                (C_SIMPLEX, "c"), (D_SIMPLEX, "d")):
            _x, _y = _midpoint(_simplex, _coordinates)
            _offset = {
                "a": (-16, -2), "b": (-16, -2),
                "c": (-12, 10), "d": (20, 8),
            }[_letter]
            _ax.annotate(_letter, (_x, _y), xytext=_offset,
                         textcoords="offset points", fontsize=12,
                         fontweight="bold", ha="center", va="center",
                         bbox={"facecolor": "white", "alpha": 0.8,
                               "edgecolor": "none", "pad": 1})

    _ax.set_xlim(-0.04, 1.04)
    _ax.set_ylim(-0.04, 1.04)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title(_title, fontsize=12, color="#183047", pad=8)


def main(_output_path="fixed_paper_examples.png"):
    """Render a six-panel overview to a PNG file.

    :param _output_path: Destination image path.
    :type _output_path: str or pathlib.Path
    :return: Absolute path of the saved figure.
    :rtype: pathlib.Path
    """
    _cmap = plt.colormaps["viridis"]
    _scale = Normalize(vmin=0.0, vmax=3.0)
    _figure, _axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for _ax, (_title, _filtration, _coordinates) in zip(
            _axes.flat, PANELS):
        _draw_example(_ax, _title, _filtration, _coordinates, _cmap, _scale)

    _figure.suptitle("Fixed simplicial complexes and filtration values",
                     fontsize=16, color="#183047")
    _bar = _figure.colorbar(
        plt.cm.ScalarMappable(norm=_scale, cmap=_cmap),
        ax=_axes.ravel().tolist(), shrink=0.75, pad=0.02
    )
    _bar.set_label("Filtration value")
    _destination = Path(_output_path).expanduser().resolve()
    _destination.parent.mkdir(parents=True, exist_ok=True)
    _figure.savefig(_destination, dpi=180, facecolor="white")
    plt.close(_figure)
    return _destination


if __name__ == "__main__":
    output_path = sys.argv[1] if len(sys.argv) > 1 else "fixed_paper_examples.png"
    print("Saved", main(output_path))
