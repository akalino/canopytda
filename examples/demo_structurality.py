"""Classify the two event points in paper Example 3.12.

From the repository root, run::

    python -m examples.demo_structurality
"""

from canopytda.stratification import Stratification
from canopytda.structurality import Structurality

from data.paper_complexes import L_SIMPLEX_NAMES
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS


def _named_pairs(_pairs):
    """Return readable labels for the paper's degree-one simplex pairs."""
    return tuple((L_SIMPLEX_NAMES[_birth], L_SIMPLEX_NAMES[_death])
                 for _birth, _death in _pairs)


def main():
    """Print incident equivalence counts and local event classifications."""
    _family = Stratification(PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)
    _analysis = Structurality(_family)
    _classification = _analysis.classify_events(1)
    print("Paper Example 3.12, fixed L complex")
    print("  strata:", len(_family.strata))
    print("  compatible adjacency edges:",
          len(_analysis.verified_adjacency()["edges"]))
    for _event in _classification["events"]:
        print("  parameter {:.6g}: {}".format(
            _event["parameter"], _event["classification"]))
        for _point in _event["multiplicity_points"]:
            print("    H{} ({:.6g}, {:.6g}): {}".format(
                _point["dimension"], _point["birth"], _point["death"],
                _named_pairs(_point["pairs"])))
            print("    inseparable pairs:",
                  _named_pairs(_point["non_hausdorff_pairs"]))
        print("    incident equivalences:", tuple(
            (_entry["neighbor_stratum_id"], _entry["equivalence_count"])
            for _entry in _event["adjacent_transitions"]))
    assert _classification["structural_parameters"] == (1.0 / 3.0,)
    print("  structural parameters:",
          _classification["structural_parameters"])


if __name__ == "__main__":
    main()
