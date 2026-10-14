"""Check neighborhoods and whole-cell closure in the paper's L example.

Run from the repository root::

    python -m examples.demo_topology
"""

from canopytda.a_canopy import ACanopy
from canopytda.stratification import Stratification
from canopytda.topology import CanopyTopology

from data.paper_complexes import L_SIMPLEX_NAMES
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS


def _named_pair(_pair):
    """Return the paper labels of one birth-death pair."""
    return tuple(L_SIMPLEX_NAMES.get(_simplex, str(_simplex))
                 for _simplex in _pair)


def main():
    """Print paper event certificates and the closure needed downstream."""
    _family = Stratification(PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)
    _canopy = ACanopy(_family, _homology_dimension=1)
    _topology = CanopyTopology(_canopy)
    print("Example 3.12: finite topology on the H1 A-canopy")
    print("  symbolic cells:", len(_topology.cells))
    print("  witnessed relations:", len(_topology.relations))

    for _parameter in (1.0 / 3.0, 3.0 / 4.0):
        _points = _canopy.fiber(_parameter)
        _local = _topology.local_components(_parameter)
        _certificate = _topology.separation_certificate(
            _parameter, _points[0]["pair"],
            _parameter, _points[1]["pair"])
        print("  t={:.6g}: {} local component(s); separable={}".format(
            _parameter, _local["component_count"],
            _certificate["separable"]))
        print("    pair labels:", tuple(_named_pair(_point["pair"])
                                        for _point in _points))
        print("    fiber IDs by component:", tuple(
            _component["fiber_cell_ids"]
            for _component in _local["components"]))

    _left_cell = next(_cell["cell_id"] for _cell in _topology.cells
                      if _cell["stratum_id"] == 0 and
                      _named_pair(_cell["pair"]) == ("a", "d"))
    _closure = _topology.closure((_left_cell,))
    print("  closure of the left (a,d) sheet:")
    print("    selected cells:", _closure["input_cell_ids"])
    print("    added event cells:", _closure["boundary_cell_ids"])
    _agreement = _topology.structurality_certificate()
    print("  independent event labels agree:",
          _agreement["all_events_agree"])

    _structural_pairs = tuple(_point["pair"]
                              for _point in _canopy.fiber(1 / 3))
    _ordinary_pairs = tuple(_point["pair"]
                            for _point in _canopy.fiber(3 / 4))
    assert not _topology.points_separable(
        1 / 3, _structural_pairs[0], 1 / 3, _structural_pairs[1])
    assert _topology.points_separable(
        3 / 4, _ordinary_pairs[0], 3 / 4, _ordinary_pairs[1])
    assert _closure["boundary_cell_ids"] == ((1, 0), (1, 1))
    assert _agreement["all_events_agree"]


if __name__ == "__main__":
    main()
