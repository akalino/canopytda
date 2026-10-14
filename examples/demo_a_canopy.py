"""Inspect the A-canopy for the paper's fixed-complex Example 3.12.

From the repository root, run::

    python -m examples.demo_a_canopy
"""

from canopytda.a_canopy import ACanopy
from canopytda.pairing import PersistencePairing
from canopytda.stratification import Stratification

from data.paper_complexes import L_SIMPLEX_NAMES
from data.paper_complexes import PAPER_3_7_FILTRATIONS
from data.paper_complexes import PAPER_3_7_PARAMETERS


def _named_pair(_pair):
    """Display a pair using the simplex names in the paper."""
    return tuple(L_SIMPLEX_NAMES.get(_simplex, str(_simplex))
                 for _simplex in _pair)


def main():
    """Show fibers, augmented diagrams, and all adjacent maps."""
    _family = Stratification(PAPER_3_7_FILTRATIONS, PAPER_3_7_PARAMETERS)
    _canopy = ACanopy(_family, _homology_dimension=1)
    print("Paper Example 3.12: the H1 A-canopy on the L complex")
    print("  strata:", len(_canopy["representative_fibers"]))
    print("  pair-labeled points per fiber:", _canopy.fiber_cardinality)
    print("  structural parameters:", _canopy.structural_parameters)

    for _parameter in (0.0, 1.0 / 3.0, 0.5, 3.0 / 4.0, 1.0):
        _points = _canopy.fiber(_parameter)
        _diagram = _canopy.diagram(_parameter, 1)
        _direct = PersistencePairing(
            _family.filtration_at(_parameter)).augmented_diagram(1)
        assert _diagram == _direct
        print("  t={:.6g}:".format(_parameter))
        for _point in _points:
            print("    {} -> {}".format(
                _named_pair(_point["pair"]), _point["eta"]))
        print("    augmented H1 diagram:", _diagram)

    print("  adjacent stratum maps:")
    for _edge in _canopy["incidences"]:
        _source = _edge["source_stratum_id"]
        _target = _edge["target_stratum_id"]
        _maps = _canopy.continuations(_source, _target)
        print("    {} -> {}: {}".format(_source, _target, len(_maps)))
        for _continuation in _maps:
            print("      {}".format(tuple(
                (_named_pair(_from), _named_pair(_to))
                for _from, _to in _continuation["pair_mapping"])))

    _event = next(_fiber for _fiber in _canopy["representative_fibers"]
                  if _fiber["representative_parameter"] == 1.0 / 3.0)
    assert len(_event["self_continuations"]) == 2
    print("  event t=1/3 self-continuations:",
          len(_event["self_continuations"]))
    print("  augmented diagrams agree with direct reduction at every sample")


if __name__ == "__main__":
    main()
