"""Compare paper examples against independent fiber computations.

From the repository root, with the current canopytda package installed::

    python -m benchmarks.paper_comparisons --output-dir outputs/paper_benchmarks

GUDHI, when installed, supplies an additional independent off-diagonal
persistence reference over F_2. All comparisons retain multiplicities.
Construction times on these tiny examples are diagnostic only; they are not
evidence of asymptotic performance or a speed comparison between methods.
"""

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import gudhi as gd

from canopytda.a_canopy import ACanopy
from canopytda.d_canopy import DCanopy
from canopytda.pairing import PersistencePairing
from canopytda.stratification import Stratification
from canopytda.topology import CanopyTopology
from canopytda.vines import VineDecomposition

from data.paper_complexes import D_DIAGONAL_FILTRATIONS
from data.paper_complexes import D_DIAGONAL_PARAMETERS
from data.paper_complexes import D_FINITE_PAIR
from data.paper_complexes import FILLED_TRIANGLE_FILTRATION
from data.paper_complexes import L_G_FILTRATION
from data.paper_complexes import L_G_ORDER_ONE
from data.paper_complexes import L_G_ORDER_TWO
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS
from data.paper_complexes import TWO_DISJOINT_EDGES_FILTRATION
from data.paper_complexes import TWO_DISJOINT_EDGES_PAIRS


def _cases():
    """Return labeled families, dimensions, and event/interior probes."""
    return (
        ("paper_l_3_7", PAPER_3_7_FILTRATIONS,
         PAPER_3_7_PARAMETERS, 1,
         (0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5, 0.75, 0.875, 1.0)),
        ("paper_l_g", (L_G_FILTRATION,), (0.0,), 1, (0.0,)),
        ("paper_3_16_disjoint_edges", (TWO_DISJOINT_EDGES_FILTRATION,),
         (0.0,), 0, (0.0,)),
        ("diagonal_entry", D_DIAGONAL_FILTRATIONS,
         D_DIAGONAL_PARAMETERS, 0,
         (0.0, 0.25, 0.5, 0.75, 1.0)),
        ("zero_length_loop", (FILLED_TRIANGLE_FILTRATION,),
         (0.0,), 1, (0.0,)),
    )


def _encode_diagram(_diagram):
    """Represent essential deaths with a JSON-compatible infinity string."""
    return [[_birth, "inf" if math.isinf(_death) else _death]
            for _birth, _death in _diagram]


def _diagrams_agree(_first, _second, _atol=1e-9):
    """Compare sorted diagram multisets with numeric tolerance."""
    if len(_first) != len(_second):
        return False
    return all(
        math.isclose(_left[0], _right[0], abs_tol=_atol) and
        ((math.isinf(_left[1]) and math.isinf(_right[1])) or
         math.isclose(_left[1], _right[1], abs_tol=_atol))
        for _left, _right in zip(sorted(_first), sorted(_second))
    )


def _gudhi_diagram(_filtration, _dimension):
    """Compute off-diagonal intervals with GUDHI over F_2, if present."""
    if not hasattr(gd.SimplexTree, "compute_persistence"):
        return None
    _tree = _filtration.to_simplex_tree()
    _tree.compute_persistence(homology_coeff_field=2,
                              min_persistence=0,
                              persistence_dim_max=True)
    return tuple(sorted(tuple(map(float, _interval)) for _interval in
                        _tree.persistence_intervals_in_dimension(_dimension)))


def _case_result(_case):
    """Build a single family once, compare diagrams on every probe."""
    _name, _filtrations, _parameters, _dimension, _probes = _case
    _start = perf_counter()
    _family = Stratification(_filtrations, _parameters)
    _stratification_seconds = perf_counter() - _start
    _start = perf_counter()
    _canopy = ACanopy(_family, _homology_dimension=_dimension)
    _a_seconds = perf_counter() - _start
    _start = perf_counter()
    _topology = CanopyTopology(_canopy)
    _a_vines = VineDecomposition(_topology)
    _topology_seconds = perf_counter() - _start
    _start = perf_counter()
    _d_canopy = DCanopy(_canopy, _topology)
    _d_vines = VineDecomposition(_d_canopy)
    _d_seconds = perf_counter() - _start

    _rows = []
    for _parameter in _probes:
        _filtration = _family.filtration_at(_parameter)
        _direct = PersistencePairing(_filtration)
        _augmented = _direct.augmented_diagram(_dimension)
        _diminished = _direct.diminished_diagram(_dimension)
        _a_diagram = _canopy.diagram(_parameter, _dimension)
        _d_diagram = _d_canopy.diagram(_parameter, _dimension)
        _gudhi = _gudhi_diagram(_filtration, _dimension)
        _fiber = _canopy.fiber(_parameter)
        _pairs = tuple(_point["pair"] for _point in _fiber)
        _separable = (None if len(_pairs) != 2 else
                      _topology.points_separable(_parameter, _pairs[0],
                                                 _parameter, _pairs[1]))
        _counts = Counter(_augmented)
        _row = {
            "example": _name,
            "parameter": _parameter,
            "homology_dimension": _dimension,
            "simplices": len(_filtration.complex.simplices),
            "direct_augmented": _encode_diagram(_augmented),
            "a_canopy": _encode_diagram(_a_diagram),
            "direct_diminished": _encode_diagram(_diminished),
            "d_canopy": _encode_diagram(_d_diagram),
            "gudhi_diminished": (None if _gudhi is None else
                                 _encode_diagram(_gudhi)),
            "augmented_match": _diagrams_agree(_augmented, _a_diagram),
            "diminished_match": _diagrams_agree(_diminished, _d_diagram),
            "gudhi_match": (None if _gudhi is None else
                            _diagrams_agree(_gudhi, _diminished)),
            "a_fiber_size": len(_fiber),
            "max_coordinate_multiplicity": max(_counts.values(), default=0),
            "two_points_separable": _separable,
            "a_vine_count": len(_a_vines.vines),
            "d_vine_count": len(_d_vines.vines),
        }
        _rows.append(_row)

    _result = {
        "example": _name,
        "homology_dimension": _dimension,
        "input_parameters": list(_parameters),
        "a_vine_count": len(_a_vines.vines),
        "d_vine_count": len(_d_vines.vines),
        "a_vines_regular": _a_vines.regular,
        "d_vines_regular": _d_vines.regular,
        "structural_parameters": list(_canopy.structural_parameters),
        "diagonal_boundary_count": len(_d_canopy.boundary_cells()),
        "build_seconds": {
            "stratification": _stratification_seconds,
            "a_canopy": _a_seconds,
            "topology_and_a_vines": _topology_seconds,
            "d_canopy_and_d_vines": _d_seconds,
        },
    }
    return _result, _rows


def run_comparisons():
    """Return benchmark results and explicit paper-specific checks.

    :return: Complete JSON-compatible result, including checks and failures.
    :rtype: dict
    """
    _results = []
    _rows = []
    for _case in _cases():
        _result, _case_rows = _case_result(_case)
        _results.append(_result)
        _rows.extend(_case_rows)

    _by_key = {(_row["example"], _row["parameter"]): _row
               for _row in _rows}
    _event = _by_key["paper_l_3_7", 1.0 / 3.0]
    _later = _by_key["paper_l_3_7", 0.75]
    _disjoint = _by_key["paper_3_16_disjoint_edges", 0.0]
    _diagonal = _by_key["diagonal_entry", 0.5]
    _filled = _by_key["zero_length_loop", 0.0]
    _disjoint_family = Stratification((TWO_DISJOINT_EDGES_FILTRATION,))
    _disjoint_canopy = ACanopy(_disjoint_family, _homology_dimension=0)
    _disjoint_vines = VineDecomposition(CanopyTopology(_disjoint_canopy))
    _finite_vines = tuple(_vine for _vine in _disjoint_vines.vines
                          if any(_point["pair"] in TWO_DISJOINT_EDGES_PAIRS
                                 for _point in _vine.fiber(0.0)))
    _first_order = PersistencePairing(L_G_FILTRATION, L_G_ORDER_ONE)
    _second_order = PersistencePairing(L_G_FILTRATION, L_G_ORDER_TWO)
    _checks = {
        "paper_structural_event_nonseparable": (
            _event["a_fiber_size"] == 2 and
            _event["two_points_separable"] is False and
            _later["two_points_separable"] is True),
        "paper_3_16_equal_coordinates_distinct_vines": (
            _disjoint["max_coordinate_multiplicity"] >= 2 and
            len(_finite_vines) == 2 and
            len({_vine.cell_ids for _vine in _finite_vines}) == 2),
        "paper_l_g_order_changes_pairs_not_diagram": (
            _first_order.finite_pairs != _second_order.finite_pairs and
            _first_order.augmented_diagram(1) ==
            _second_order.augmented_diagram(1)),
        "diagonal_limit_retained_but_omitted_from_d_diagram": (
            _diagonal["direct_augmented"] != _diagonal["direct_diminished"] and
            _diagonal["d_canopy"] == _diagonal["direct_diminished"] and
            next(_result for _result in _results if
                 _result["example"] == "diagonal_entry")[
                     "diagonal_boundary_count"] >= 1),
        "zero_length_loop_removed_from_d_diagram": (
            _filled["direct_augmented"] != _filled["direct_diminished"] and
            _filled["d_canopy"] == _filled["direct_diminished"]),
    }
    _failures = ["{} at t={}".format(_row["example"], _row["parameter"])
                 for _row in _rows if not all(
                     _row[_key] in (True, None) for _key in
                     ("augmented_match", "diminished_match", "gudhi_match"))]
    _failures.extend(_key for _key, _value in _checks.items() if not _value)
    return {
        "scope": "Fixed-complex paper fixtures; selected event and interior fibers",
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "gudhi": getattr(gd, "__version__", None)},
        "references": {
            "direct": "PersistencePairing on independently evaluated filtrations over F_2",
            "gudhi": "GUDHI SimplexTree over F_2, off-diagonal only",
            "gudhi_available": hasattr(gd.SimplexTree, "compute_persistence"),
            "diagram_absolute_tolerance": 1e-9,
        },
        "interpretation": "Times are one-shot smoke measurements; no speed claims",
        "cases": _results,
        "fibers": _rows,
        "paper_checks": _checks,
        "failures": _failures,
    }


def write_results(_results, _output_dir):
    """Write structured results and a concise human-readable summary.

    :param _results: Result returned by :func:`run_comparisons`.
    :type _results: dict
    :param _output_dir: Destination for the three result files.
    :type _output_dir: str or pathlib.Path
    """
    _destination = Path(_output_dir)
    _destination.mkdir(parents=True, exist_ok=True)
    (_destination / "paper_comparisons.json").write_text(
        json.dumps(_results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    with (_destination / "paper_fibers.csv").open(
            "w", newline="", encoding="utf-8") as _stream:
        _writer = csv.DictWriter(_stream, fieldnames=_results["fibers"][0])
        _writer.writeheader()
        for _row in _results["fibers"]:
            _writer.writerow({
                _key: (json.dumps(_value, allow_nan=False)
                       if isinstance(_value, list) else _value)
                for _key, _value in _row.items()
            })

    _lines = ["# Paper complex comparisons", "",
              "One fixed complex per example. Diagrams are multisets;",
              "`inf` denotes an essential death. All computations use F_2.",
              "", "| Check | Result |", "| --- | --- |"]
    _lines.extend("| {} | {} |".format(_name.replace("_", " "),
                                       "pass" if _passed else "FAIL")
                  for _name, _passed in _results["paper_checks"].items())
    _lines.extend(["", "| Example | Fibers | A vines | D vines |",
                   "| --- | ---: | ---: | ---: |"])
    _lines.extend("| {} | {} | {} | {} |".format(
        _case["example"], sum(_row["example"] == _case["example"]
                              for _row in _results["fibers"]),
        _case["a_vine_count"], _case["d_vine_count"])
        for _case in _results["cases"])
    _lines.extend(["", "GUDHI off-diagonal comparison: {}.".format(
        "available" if _results["references"]["gudhi_available"] else
        "unavailable in this environment"),
        "", "Times in JSON are one-shot construction diagnostics on tiny inputs;",
        "they are not method speed or scaling results.", ""])
    if _results["failures"]:
        _lines.append("Failures: " + ", ".join(_results["failures"]))
    (_destination / "README.md").write_text("\n".join(_lines),
                                               encoding="utf-8")


def main():
    """Run comparisons, persist results, and fail for any discrepancy."""
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--output-dir", default="outputs/paper_benchmarks")
    _args = _parser.parse_args()
    _results = run_comparisons()
    write_results(_results, _args.output_dir)
    print("Checked {} fibers across {} examples: {}".format(
        len(_results["fibers"]), len(_results["cases"]),
        "PASS" if not _results["failures"] else "FAIL"))
    print("Results:", _args.output_dir)
    if _results["failures"]:
        raise SystemExit("Failed: " + ", ".join(_results["failures"]))


if __name__ == "__main__":
    main()
