"""Plot computed neighborhoods and closure for the paper's L example.

Run from the repository root::

    python -m examples.plot_topology outputs/topology_example_3_12.png

The panels draw finite continuation witnesses between symbolic cells. They
are computed neighborhood checks for basis neighborhoods, not a geometric embedding of the
total canopy space.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

from canopytda.a_canopy import ACanopy
from canopytda.stratification import Stratification
from canopytda.topology import CanopyTopology

from data.paper_complexes import L_SIMPLEX_NAMES
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS


def _pair_label(_pair):
    """Return a compact label for a simplex pair in the paper."""
    return "({},{})".format(*(L_SIMPLEX_NAMES.get(_simplex, str(_simplex))
                              for _simplex in _pair))


def _local_data(_topology, _parameter):
    """Return cells and positions in a canonical local base window."""
    _local = _topology.local_components(_parameter)
    _active = set(_cell_id
                  for _component in _local["components"]
                  for _cell_id in _component["cell_ids"])
    _cells = tuple(_cell for _cell in _topology.cells
                   if _cell["cell_id"] in _active)
    _strata = tuple(sorted({_cell["stratum_id"] for _cell in _cells}))
    _positions = {}
    for _index, _stratum in enumerate(_strata):
        _sheets = sorted((_cell for _cell in _cells
                          if _cell["stratum_id"] == _stratum),
                         key=lambda _cell: _pair_label(_cell["pair"]))
        for _sheet_index, _cell in enumerate(_sheets):
            _height = 0.5 if len(_sheets) == 1 else 1.0 - _sheet_index
            _positions[_cell["cell_id"]] = (_index, _height)
    return _local, _cells, _strata, _positions


def _edge_witnesses(_topology, _positions):
    """Return distinct local continuation edges with both endpoints shown."""
    _edges = {}
    for _relation in _topology.relations:
        _source = _relation["source_cell_id"]
        _target = _relation["target_cell_id"]
        if _source not in _positions or _target not in _positions:
            continue
        if _source != _target:
            _edges.setdefault(frozenset((_source, _target)),
                              (_source, _target))
    return tuple(_edges.values())


def _panel(_axis, _topology, _parameter, _title, _closure=None):
    """Draw all local sheet IDs and certified continuation witnesses."""
    _local, _cells, _strata, _positions = _local_data(
        _topology, _parameter)
    _cell_components = {_cell_id: _component["component_id"]
                        for _component in _local["components"]
                        for _cell_id in _component["cell_ids"]}
    _palette = ("#744d93", "#098995")

    for _source, _target in _edge_witnesses(_topology, _positions):
        _start = _positions[_source]
        _end = _positions[_target]
        _color = (_palette[_cell_components[_source] % 2]
                  if _closure is None else "#b6bec7")
        _axis.plot((_start[0], _end[0]), (_start[1], _end[1]),
                   color=_color, linewidth=1.4, alpha=0.43, zorder=1)

    for _cell in _cells:
        _cell_id = _cell["cell_id"]
        _x, _y = _positions[_cell_id]
        if _closure is None:
            _color = _palette[_cell_components[_cell_id] % 2]
        elif _cell_id in _closure["input_cell_ids"]:
            _color = "#275b88"
        elif _cell_id in _closure["boundary_cell_ids"]:
            _color = "#d27937"
        else:
            _color = "#bdc6cf"
        _axis.scatter((_x,), (_y,), marker="o" if
                      _cell["stratum_id"] == _local["stratum_id"] else "s",
                      s=155, color=_color, edgecolor="#253746",
                      linewidth=0.8, zorder=3)
        _offset = -0.075 if _x == len(_strata) - 1 else 0.075
        _axis.text(_x + _offset, _y, _pair_label(_cell["pair"]),
                   ha="right" if _offset < 0 else "left", va="center",
                   fontsize=9.8, color="#23354a", zorder=4)

    _labels = ("left interval", "event point", "right interval")
    _axis.set_xlim(-0.28, 2.32)
    _axis.set_ylim(-0.22, 1.27)
    _axis.set_xticks(range(len(_strata)), _labels[:len(_strata)])
    _axis.tick_params(axis="x", length=0, labelsize=8.3, pad=9)
    _axis.set_yticks(())
    for _spine in _axis.spines.values():
        _spine.set_visible(False)
    _axis.set_title(_title, fontsize=11.5, color="#23354a", pad=10)
    return _local


def main(_output_path="outputs/topology_example_3_12.png"):
    """Save three local topology computed neighborhood checks and print their result.

    :param _output_path: PNG output filename.
    :type _output_path: str or pathlib.Path
    :return: Absolute image filename.
    :rtype: pathlib.Path
    """
    _family = Stratification(PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)
    _canopy = ACanopy(_family, _homology_dimension=1)
    _topology = CanopyTopology(_canopy)
    _first_cell = next(_cell["cell_id"] for _cell in _topology.cells
                       if _cell["stratum_id"] == 0 and
                       _pair_label(_cell["pair"]) == "(a,d)")
    _closure = _topology.closure((_first_cell,))
    _figure, _axes = plt.subplots(1, 3, figsize=(16.5, 5.1),
                                  facecolor="white")
    _figure.subplots_adjust(left=0.04, right=0.97, wspace=0.13,
                            top=0.75, bottom=0.21)
    _structural = _panel(_axes[0], _topology, 1 / 3,
                         "t = 1/3 · one local component")
    _regular = _panel(_axes[1], _topology, 3 / 4,
                      "t = 3/4 · two local components")
    _panel(_axes[2], _topology, 1 / 3,
           "Closure of the left (a,d) sheet", _closure=_closure)

    assert _structural["component_count"] == 1
    assert len(_structural["components"][0]["fiber_cell_ids"]) == 2
    assert _regular["component_count"] == 2
    assert _closure["boundary_cell_ids"] == ((1, 0), (1, 1))
    assert not any(_cell_id[0] == 2 for _cell_id in
                   _closure["closure_cell_ids"])
    assert _topology.structurality_certificate()["all_events_agree"]

    _figure.suptitle("Example 3.12: computed A-canopy topology computed neighborhood checks",
                     fontsize=16, color="#23354a", y=0.97)
    _figure.text(0.5, 0.86,
                 "Two distinct event points are nonseparable at 1/3; the points at 3/4 admit disjoint neighborhoods.",
                 fontsize=10.5, color="#4b6073", ha="center")
    _figure.text(0.5, 0.07,
                 "Squares: interval sheets. Circles: distinct event points. Blue: selected cell; orange: added by closure; gray: excluded.",
                 fontsize=10, color="#23354a", ha="center")
    _figure.text(0.5, 0.025,
                 "Lines are witnessed continuations in the finite presentation; they are not an embedding of the total space.",
                 fontsize=9.2, color="#566879", ha="center")
    _destination = Path(_output_path).expanduser().resolve()
    _destination.parent.mkdir(parents=True, exist_ok=True)
    _figure.savefig(_destination, dpi=180, facecolor="white")
    plt.close(_figure)
    print("Saved", _destination)
    return _destination


if __name__ == "__main__":
    output_path = (sys.argv[1] if len(sys.argv) > 1 else
                   "outputs/topology_example_3_12.png")
    main(output_path)
