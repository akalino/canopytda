"""Plot the computed stages of two fixed-complex paper examples.

From the repository root, run::

    python -m examples.plot_pipeline_states outputs

The first figure follows Example 3.12 through filtration, order strata,
simplex pairing, augmented diagrams and the finite A-canopy presentation.
The second compares the two consistent orders of Example 2.19. Diagrams
show exact point coordinates: overlapping points share a marker labeled
with their multiplicity. The incidence counts are combinatorial data, not
a drawing of the as-yet unimplemented canopy topology.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
from matplotlib.patches import Rectangle

from canopytda.a_canopy import ACanopy
from canopytda.pairing import PersistencePairing
from canopytda.stratification import Stratification

from data.paper_complexes import L_G_FILTRATION
from data.paper_complexes import L_G_ORDER_ONE
from data.paper_complexes import L_G_ORDER_TWO
from data.paper_complexes import L_G_PAIRS_ONE
from data.paper_complexes import L_G_PAIRS_TWO
from data.paper_complexes import L_NAMED_SIMPLICES
from data.paper_complexes import L_SIMPLEX_NAMES
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS


COLORS = ("#3566a2", "#cf5949", "#26907b", "#9064a1")
PAPER_STATES = ((0.0, "Interval 0"),
                (1.0 / 3.0, "Event 1/3"),
                (0.5, "Interval 1"),
                (3.0 / 4.0, "Event 3/4"),
                (1.0, "Interval 2"))


def _named_pair(_pair):
    """Return the paper letters of a birth-death simplex pair."""
    return "({},{})".format(*(L_SIMPLEX_NAMES[_simplex]
                              for _simplex in _pair))


def _named_value_order(_filtration):
    """Group the four named simplices by their filtration value."""
    _ordered = sorted(L_NAMED_SIMPLICES,
                      key=lambda _simplex: (_filtration.value(_simplex),
                                            L_SIMPLEX_NAMES[_simplex]))
    _groups = []
    for _simplex in _ordered:
        _value = _filtration.value(_simplex)
        if not _groups or not math.isclose(_groups[-1][0], _value,
                                           rel_tol=0.0, abs_tol=1e-10):
            _groups.append((_value, [L_SIMPLEX_NAMES[_simplex]]))
        else:
            _groups[-1][1].append(L_SIMPLEX_NAMES[_simplex])
    return " < ".join("=".join(_names) for _value, _names in _groups)


def _draw_diagram(_axis, _labeled_points):
    """Draw exact H1 coordinates, retaining coincident-point counts."""
    _axis.plot((-0.15, 2.25), (-0.15, 2.25), linestyle="--",
               color="#a2acb6", linewidth=1, zorder=0)
    _by_value = {}
    for _label, _eta in _labeled_points:
        _by_value.setdefault(tuple(_eta), []).append(_label)
    for _eta, _labels in _by_value.items():
        _birth, _death = _eta
        _axis.scatter((_birth,), (_death,), s=110 if len(_labels) == 1
                      else 190, color="#285d81" if len(_labels) == 1
                      else "#b64762", edgecolor="white", linewidth=1.2,
                      zorder=3)
        _caption = " · ".join(_labels)
        if len(_labels) > 1:
            _caption += "  ×{}".format(len(_labels))
        _axis.annotate(_caption, (_birth, _death),
                       xytext=(-5, 7) if _birth > 1.8 else (5, 7),
                       ha="right" if _birth > 1.8 else "left",
                       textcoords="offset points", fontsize=8.2,
                       color="#19304a", zorder=4)
    _axis.set_xlim(-0.17, 2.25)
    _axis.set_ylim(0.75, 3.24)
    _axis.set_xticks((0, 1, 2))
    _axis.set_yticks((1, 2, 3))
    _axis.grid(color="#e9ecf0", linewidth=0.7)
    _axis.set_axisbelow(True)
    _axis.tick_params(labelsize=8)


def _draw_state_card(_axis, _state, _filtration, _points,
                     _classification, _outgoing_count, _self_count):
    """Draw the computed stratum, order and chosen pairing at one state."""
    _parameter, _title = _state
    _event = _classification is not None
    _color = "#a63857" if _classification == "structural" else "#285d81"
    _axis.add_patch(Rectangle((0.015, 0.025), 0.97, 0.94,
                              transform=_axis.transAxes,
                              facecolor="#fff3f5" if
                              _classification == "structural" else
                              "#f2f6fa", edgecolor=_color if _event else
                              "#cbd8e4", linewidth=1.1))
    _pairs = "  ".join(_named_pair(_point["pair"]) for _point in _points)
    _lines = ["{}     t={:.3g}".format(_title, _parameter),
              "Order: " + _named_value_order(_filtration),
              "H1 pairs: " + _pairs]
    if _event:
        _lines.append("{}; {} self maps".format(_classification,
                                                _self_count))
    if _outgoing_count is not None:
        _lines.append("Maps to next stratum: {}".format(_outgoing_count))
    _axis.text(0.06, 0.9, "\n".join(_lines), va="top", ha="left",
               transform=_axis.transAxes, fontsize=9.2,
               linespacing=1.5, color="#19304a")
    _axis.set_axis_off()


def plot_example_3_12(_output_path):
    """Plot the computed filtration path and five A-canopy states.

    :param _output_path: Destination PNG filename.
    :type _output_path: str or pathlib.Path
    :return: Absolute filename of the rendered figure.
    :rtype: pathlib.Path
    """
    _family = Stratification(PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)
    _canopy = ACanopy(_family, _homology_dimension=1)
    _record = _canopy.to_record()
    _events = {_event["event_stratum_id"]: _event
               for _event in _record["structurality"]["events"]}
    _fibers = {_fiber["stratum_id"]: _fiber
               for _fiber in _record["representative_fibers"]}
    _outgoing = {_edge["source_stratum_id"]:
                 len(_canopy.continuations(_edge["source_stratum_id"],
                                           _edge["target_stratum_id"]))
                 for _edge in _record["incidences"]}
    assert _canopy.fiber_cardinality == 2
    assert _canopy.structural_parameters == (1.0 / 3.0,)
    assert tuple(_outgoing.values()) == (2, 2, 1, 1)

    _figure = plt.figure(figsize=(17, 10.5), facecolor="white")
    _layout = _figure.add_gridspec(3, 5, height_ratios=(1.45, 1.05, 1.7),
                                    hspace=0.48, wspace=0.34,
                                    top=0.88, bottom=0.105,
                                    left=0.07, right=0.97)
    _trend = _figure.add_subplot(_layout[0, :])
    _grid = tuple(_index / 100 for _index in range(101))
    for _simplex, _color in zip(L_NAMED_SIMPLICES, COLORS):
        _values = tuple(_family.filtration_at(_parameter).value(_simplex)
                        for _parameter in _grid)
        _trend.plot(_grid, _values, linewidth=2.6, color=_color,
                    label=L_SIMPLEX_NAMES[_simplex])
        _trend.scatter(tuple(_parameter for _parameter, _title in PAPER_STATES),
                       tuple(_family.filtration_at(_parameter).value(_simplex)
                             for _parameter, _title in PAPER_STATES),
                       s=25, color=_color, zorder=5)
    for _at, _label, _color in ((1 / 3, "1/3  structural", "#a63857"),
                                (3 / 4, "3/4  nonstructural", "#285d81")):
        _trend.axvline(_at, color=_color, linewidth=1.5, linestyle="--",
                       alpha=0.8)
        _trend.text(_at + 0.008, 3.28, _label, fontsize=9.5,
                    color=_color, va="center")
    _trend.set_xlim(-0.015, 1.015)
    _trend.set_ylim(-0.08, 3.54)
    _trend.set_xticks((0, 1 / 3, 0.5, 3 / 4, 1),
                      ("0", "1/3", "1/2", "3/4", "1"))
    _trend.set_ylabel("Filtration value")
    _trend.set_xlabel("Base parameter t")
    _trend.set_title("1  Filtration values on named simplices a, b, c, d",
                     loc="left", fontsize=11, fontweight="bold", pad=13)
    _trend.grid(color="#e7ebf0", linewidth=0.7)
    _trend.legend(title="Simplex", ncol=4, frameon=False,
                  loc="upper left", bbox_to_anchor=(0, 1.03),
                  fontsize=9, title_fontsize=9)

    for _index, _state in enumerate(PAPER_STATES):
        _parameter, _title = _state
        _filtration = _family.filtration_at(_parameter)
        _points = _canopy.fiber(_parameter)
        _stratum_id = _points[0]["stratum_id"]
        _event = _events.get(_stratum_id)
        _classification = (_event["classification"] if _event else None)
        _self_count = len(_fibers[_stratum_id]["self_continuations"])
        _draw_state_card(_figure.add_subplot(_layout[1, _index]), _state,
                         _filtration, _points, _classification,
                         _outgoing.get(_stratum_id), _self_count)
        _diagram = _canopy.diagram(_parameter, 1)
        assert _diagram == PersistencePairing(
            _filtration).augmented_diagram(1)
        _axis = _figure.add_subplot(_layout[2, _index])
        _draw_diagram(_axis, ((_named_pair(_point["pair"]), _point["eta"])
                              for _point in _points))
        _axis.set_title("{}  ·  t={:.3g}".format(_title, _parameter),
                        fontsize=10, fontweight="bold", pad=7)
        _axis.set_xlabel("birth", fontsize=9)
        if _index == 0:
            _axis.set_ylabel("death", fontsize=9)

    _figure.suptitle("From a fixed L complex to its H1 A-canopy: Example 3.12",
                     x=0.07, ha="left", fontsize=17, color="#19304a")
    _figure.text(0.07, 0.62, "2  Order strata, chosen simplex pairs and complete adjacent-map counts",
                 fontsize=11, weight="bold", color="#19304a")
    _figure.text(0.07, 0.4,
                 "3  Augmented diagrams: a single marker labeled ×2 represents two distinct pair-labeled points at exactly the same coordinate.",
                 fontsize=10, color="#19304a")
    _figure.text(0.07, 0.026,
                 "Early simplices have value −1. Counts summarize finite continuation data; these panels do not depict the canopy topology.",
                 fontsize=9.2, color="#526579")
    _destination = Path(_output_path).expanduser().resolve()
    _destination.parent.mkdir(parents=True, exist_ok=True)
    _figure.savefig(_destination, dpi=170, facecolor="white")
    plt.close(_figure)
    return _destination


def plot_example_2_19(_output_path):
    """Compare the paper's two consistent orders and computed H1 diagrams.

    :param _output_path: Destination PNG filename.
    :type _output_path: str or pathlib.Path
    :return: Absolute filename of the rendered figure.
    :rtype: pathlib.Path
    """
    _orders = ((L_G_ORDER_ONE, L_G_PAIRS_ONE),
               (L_G_ORDER_TWO, L_G_PAIRS_TWO))
    _figure, _axes = plt.subplots(1, 2, figsize=(10.8, 5.8),
                                  constrained_layout=False, facecolor="white")
    _figure.subplots_adjust(left=0.09, right=0.96, bottom=0.2,
                            top=0.78, wspace=0.33)
    _diagrams = []
    for _index, (_order, _expected) in enumerate(_orders):
        _pairing = PersistencePairing(L_G_FILTRATION, _order)
        _named = tuple(_pair for _pair in _pairing.finite_pairs
                       if _pair[0] in L_NAMED_SIMPLICES)
        assert set(_named) == set(_expected)
        _diagram = _pairing.augmented_diagram(1)
        _diagrams.append(_diagram)
        _labeled = ((_named_pair(_pair),
                     (L_G_FILTRATION.value(_pair[0]),
                      L_G_FILTRATION.value(_pair[1]))) for _pair in _named)
        _draw_diagram(_axes[_index], _labeled)
        _suffix = " < ".join(L_SIMPLEX_NAMES[_simplex]
                             for _simplex in _order[-4:])
        _axes[_index].set_title("Order {}: {}".format(_index + 1, _suffix),
                                fontsize=11, fontweight="bold", pad=39)
        _axes[_index].text(0.5, 1.045,
                           "Pairs: " + "  ".join(_named_pair(_pair)
                                                for _pair in _named),
                           transform=_axes[_index].transAxes,
                           ha="center", fontsize=10, color="#19304a")
        _axes[_index].set_xlabel("birth", fontsize=10)
        _axes[_index].set_ylabel("death", fontsize=10)
    assert _diagrams[0] == _diagrams[1]
    _figure.suptitle("Example 2.19: different simplex pairs, identical augmented H1 diagrams",
                     fontsize=14, color="#19304a", y=0.96)
    _figure.text(0.5, 0.07,
                 "One filtration g: values a=1, b=2, c=d=3. The two total orders refine the same value ties.",
                 ha="center", fontsize=10, color="#526579")
    _destination = Path(_output_path).expanduser().resolve()
    _destination.parent.mkdir(parents=True, exist_ok=True)
    _figure.savefig(_destination, dpi=180, facecolor="white")
    plt.close(_figure)
    return _destination


def main(_output_directory="outputs"):
    """Render the two paper comparisons and print their output paths."""
    _directory = Path(_output_directory).expanduser().resolve()
    _first = plot_example_3_12(_directory / "pipeline_example_3_12.png")
    _second = plot_example_2_19(_directory / "orders_example_2_19.png")
    print("Saved", _first)
    print("Saved", _second)
    return _first, _second


if __name__ == "__main__":
    output_directory = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    main(output_directory)
