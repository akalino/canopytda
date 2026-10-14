"""Summarize and plot the JSON from ``run_generated_point_cloud``.

Requires matplotlib. The ordinary Rips reference and the canopy computation
use the same fixed simplicial complex but different vertex birth values.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"ordinary": "#386cb0", "canopy": "#e07a2f",
          "essential": "#386cb0", "finite": "#e07a2f",
          "diagonal": "#779b71"}


def _read_diagram(_points, _label):
    """Validate serialized birth/death pairs and return numeric tuples.

    :param _points: Serialized diagram with ``"inf"`` for essential death.
    :type _points: list
    :param _label: Location to include in malformed-input errors.
    :type _label: str
    :return: Validated birth/death pairs.
    :rtype: tuple
    :raises ValueError: If any point is malformed or has an invalid interval.
    """
    if not isinstance(_points, list):
        raise ValueError("{} must be a list of [birth, death] pairs".format(_label))
    _result = []
    for _index, _point in enumerate(_points):
        if not isinstance(_point, list) or len(_point) != 2:
            raise ValueError("{}[{}] must contain two values".format(
                _label, _index))
        _birth, _death = _point
        if (isinstance(_birth, bool) or not isinstance(_birth, (int, float))
                or not math.isfinite(_birth)):
            raise ValueError("{}[{}] has an invalid birth".format(
                _label, _index))
        if _death == "inf":
            _death = math.inf
        elif (isinstance(_death, bool) or not isinstance(_death, (int, float))
              or not math.isfinite(_death)):
            raise ValueError("{}[{}] has an invalid death".format(
                _label, _index))
        if _death < _birth:
            raise ValueError("{}[{}] dies before birth".format(_label, _index))
        _result.append((float(_birth), float(_death)))
    return tuple(_result)


def _load_results(_input):
    """Read the pipeline record and validate the diagram fields used here.

    :param _input: Path to ``results.json``.
    :type _input: pathlib.Path
    :return: Record with validated diagram tuples and ordered samples.
    :rtype: dict
    :raises ValueError: If the JSON does not match the pipeline format.
    """
    try:
        _data = json.loads(_input.read_text(encoding="utf-8"))
        _reference = _data["ordinary_rips_reference"]
        _canopy = _data["full_canopy"]
        _fibers = _canopy["fibers"]
        if not isinstance(_fibers, list) or len(_fibers) < 2:
            raise ValueError("full_canopy.fibers needs at least two samples")
        _ordinary = {}
        for _sample in ("initial", "scaled"):
            _ordinary[_sample] = {}
            for _dimension in ("h0", "h1"):
                for _kind in ("augmented", "diminished"):
                    _key = "{}_{}".format(_dimension, _kind)
                    _ordinary[_sample][_key] = _read_diagram(
                        _reference[_sample][_key],
                        "ordinary_rips_reference.{}.{}".format(_sample, _key))
        _samples = []
        for _index, _fiber in enumerate(_fibers):
            _parameter = _fiber["parameter"]
            if (isinstance(_parameter, bool)
                    or not isinstance(_parameter, (int, float))
                    or not math.isfinite(_parameter)):
                raise ValueError("Invalid fiber parameter at index {}".format(
                    _index))
            _sample = {"parameter": float(_parameter)}
            for _kind in ("a_augmented", "d_diminished"):
                _sample[_kind] = {}
                for _dimension in ("h0", "h1"):
                    _sample[_kind][_dimension] = _read_diagram(
                        _fiber[_kind][_dimension],
                        "full_canopy.fibers[{}].{}.{}".format(
                            _index, _kind, _dimension))
            _samples.append(_sample)
        _samples.sort(key=lambda _sample: _sample["parameter"])
        if len({_sample["parameter"] for _sample in _samples}) != len(_samples):
            raise ValueError("Fiber parameters must be distinct")
        if (_samples[0]["parameter"] != 0.0
                or _samples[-1]["parameter"] != 1.0):
            raise ValueError("Ordinary initial/scaled comparison needs t=0 and t=1")
        return {"metadata": _data, "ordinary": _ordinary,
                "samples": _samples}
    except (KeyError, TypeError, AttributeError) as _error:
        raise ValueError("Unrecognized pipeline results JSON: {}".format(
            _error)) from _error


def _counts(_diagram):
    """Count off-diagonal finite, diagonal, and essential classes."""
    _essential = sum(math.isinf(_death) for _, _death in _diagram)
    _diagonal = sum(_death == _birth for _birth, _death in _diagram)
    return (len(_diagram) - _essential - _diagonal, _diagonal, _essential)


def _finite_lifetimes(_diagram):
    """Return sorted lifetimes of finite, off-diagonal classes."""
    return sorted(_death - _birth for _birth, _death in _diagram
                  if math.isfinite(_death) and _death > _birth)


def _plot_h0(_results, _destination):
    """Compare ranked H0 lifetimes and their sum along the sampled path."""
    _ordinary = _results["ordinary"]
    _samples = _results["samples"]
    _figure, (_rank_axis, _trend_axis) = plt.subplots(
        1, 2, figsize=(12, 4.7), constrained_layout=True)
    for _name, _sample, _style in (("initial", _samples[0], "-"),
                                   ("scaled", _samples[-1], "--")):
        _reference = _finite_lifetimes(
            _ordinary[_name]["h0_augmented"])
        _adjusted = _finite_lifetimes(_sample["a_augmented"]["h0"])
        _rank_axis.plot(range(1, len(_reference) + 1), _reference,
                        linestyle=_style, color=COLORS["ordinary"],
                        linewidth=2, label="Ordinary Rips, {}".format(_name))
        _rank_axis.plot(range(1, len(_adjusted) + 1), _adjusted,
                        linestyle=_style, marker="o", markersize=3,
                        markerfacecolor="none", linewidth=1.4,
                        color=COLORS["canopy"],
                        label="Canopy input, {}".format(_name))
    _rank_axis.set(title="Finite H₀ lifetimes (ranked)", xlabel="Rank",
                   ylabel="Death − birth")
    _rank_axis.legend(fontsize=8)
    _rank_axis.grid(alpha=0.2)

    _parameters = [_sample["parameter"] for _sample in _samples]
    _totals = [sum(_finite_lifetimes(_sample["a_augmented"]["h0"]))
               for _sample in _samples]
    _trend_axis.plot(_parameters, _totals, "-o", linewidth=2,
                     color=COLORS["canopy"], label="Canopy input")
    for _name, _sample in (("initial", _samples[0]),
                           ("scaled", _samples[-1])):
        _total = sum(_finite_lifetimes(_ordinary[_name]["h0_augmented"]))
        _trend_axis.scatter([_sample["parameter"]], [_total], marker="x",
                            color=COLORS["ordinary"], s=110,
                            linewidths=2.3, zorder=4,
                            label="Ordinary Rips, {}".format(_name))
    _trend_axis.set(title="Sum of finite H₀ lifetimes", xlabel="Parameter t",
                    ylabel="Total death − birth")
    _trend_axis.legend(fontsize=8)
    _trend_axis.grid(alpha=0.2)
    _figure.suptitle("H₀ on the fixed complex: 80 essential classes omitted from lifetime sums",
                     fontsize=11)
    _figure.savefig(_destination, dpi=180)
    plt.close(_figure)


def _plot_h1(_results, _destination):
    """Display H1 class counts, including diagonal points and infinities."""
    _ordinary = _results["ordinary"]
    _samples = _results["samples"]
    _labels = ["Rips\ninitial", "A\nt=0", "D\nt=0"]
    _diagrams = [_ordinary["initial"]["h1_augmented"],
                 _samples[0]["a_augmented"]["h1"],
                 _samples[0]["d_diminished"]["h1"]]
    for _sample in _samples[1:-1]:
        _parameter = "{:g}".format(_sample["parameter"])
        for _kind, _label in (("a_augmented", "A"),
                              ("d_diminished", "D")):
            _labels.append("{}\nt={}".format(_label, _parameter))
            _diagrams.append(_sample[_kind]["h1"])
    _labels.extend(("A\nt=1", "D\nt=1", "Rips\nscaled"))
    _diagrams.extend((_samples[-1]["a_augmented"]["h1"],
                      _samples[-1]["d_diminished"]["h1"],
                      _ordinary["scaled"]["h1_augmented"]))

    _figure, _axis = plt.subplots(figsize=(max(9, len(_labels) * 0.9), 5),
                                  constrained_layout=True)
    _bottom = [0] * len(_diagrams)
    for _index, (_category, _color) in enumerate(
            (("essential", COLORS["essential"]),
             ("finite", COLORS["finite"]),
             ("diagonal", COLORS["diagonal"]))):
        _values = [_counts(_diagram)[{"finite": 0, "diagonal": 1,
                                     "essential": 2}[_category]]
                   for _diagram in _diagrams]
        _axis.bar(range(len(_labels)), _values, bottom=_bottom,
                  color=_color, label={"essential": "Essential (∞ death)",
                                       "finite": "Off-diagonal finite",
                                       "diagonal": "Diagonal (zero lifetime)"}[_category])
        _bottom = [_base + _value for _base, _value in zip(_bottom, _values)]
    _axis.set(xticks=range(len(_labels)), xticklabels=_labels,
              ylabel="Number of H₁ classes", ylim=(0, max(_bottom) + 1),
              title="H₁ classes: A includes diagonal points; D removes them")
    _axis.legend(ncol=3, loc="upper right", fontsize=8)
    _axis.grid(axis="y", alpha=0.2)
    _axis.set_axisbelow(True)
    _figure.savefig(_destination, dpi=180)
    plt.close(_figure)


def _summary(_results):
    """Prepare a compact Markdown interpretation from the JSON itself."""
    _metadata = _results["metadata"]
    _canopy = _metadata["full_canopy"]
    _ordinary = _results["ordinary"]
    _samples = _results["samples"]
    _rows = []
    for _label, _h0, _h1 in (
            ("Ordinary Rips (initial)",
             _ordinary["initial"]["h0_augmented"],
             _ordinary["initial"]["h1_augmented"]),
            *(("A (t={:g})".format(_sample["parameter"]),
               _sample["a_augmented"]["h0"], _sample["a_augmented"]["h1"])
              for _sample in _samples),
            *(("D (t={:g})".format(_sample["parameter"]),
               _sample["d_diminished"]["h0"], _sample["d_diminished"]["h1"])
              for _sample in _samples),
            ("Ordinary Rips (scaled)",
             _ordinary["scaled"]["h0_augmented"],
             _ordinary["scaled"]["h1_augmented"])):
        _h0_counts = _counts(_h0)
        _h1_counts = _counts(_h1)
        _rows.append("| {} | {} | {} | {} | {} | {} | {:.9f} |".format(
            _label, _h0_counts[0], _h0_counts[2], _h1_counts[0],
            _h1_counts[1], _h1_counts[2], sum(_finite_lifetimes(_h0))))
    return "\n".join((
        "# Generated point cloud: comparison summary", "",
        "{} points in {} dimensions; {} simplices in the fixed complex.".format(
            _metadata["vertex_count"], _metadata["ambient_dimension"],
            _metadata["simplex_count"]), "",
        "The ordinary Rips reference has zero vertex births. The canopy input "
        "assigns distinct vertex births (step {:.3g}; largest {:.3g}); edge and "
        "triangle values use Rips lengths. The same fixed complex is used "
        "throughout. Comparisons between these filtrations are descriptive, "
        "not exact equality claims.".format(
            _canopy["vertex_birth_step"], _canopy["largest_vertex_birth"]),
        "", "| Computation | H₀ finite | H₀ essential | H₁ finite | "
        "H₁ diagonal | H₁ essential | Sum of finite H₀ lifetimes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *_rows, "",
        "A is augmented persistence; D is diminished persistence. Essential "
        "classes have infinite death and are excluded from finite lifetime "
        "plots and sums. Diagonal points have zero lifetime.", "",
        "The pipeline reports {} order changes, {} A-vines, {} D-vines, "
        "and direct-persistence agreement at its sampled parameters: {}. "
        "These data describe this single scaled-cloud example, not a "
        "benchmark across data sets.".format(
            _canopy["order_event_count"], _canopy["a_vine_count"],
            _canopy["d_vine_count"],
            _canopy["diagrams_match_direct_persistence"]), ""))


def main():
    """Read results JSON and save two figures plus a numerical summary."""
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("input", type=Path, help="Pipeline results.json")
    _parser.add_argument("--output-dir", type=Path,
                         help="Figure destination (default: JSON directory)")
    _args = _parser.parse_args()
    try:
        _results = _load_results(_args.input)
    except (OSError, json.JSONDecodeError, ValueError) as _error:
        _parser.error(str(_error))
    _destination = _args.output_dir or _args.input.parent
    _destination.mkdir(parents=True, exist_ok=True)
    _h0_path = _destination / "h0_persistence_comparison.png"
    _h1_path = _destination / "h1_class_comparison.png"
    _summary_path = _destination / "comparison_summary.md"
    _plot_h0(_results, _h0_path)
    _plot_h1(_results, _h1_path)
    _summary_path.write_text(_summary(_results), encoding="utf-8")
    for _path in (_h0_path, _h1_path, _summary_path):
        print(_path)


if __name__ == "__main__":
    main()
