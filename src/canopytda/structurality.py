"""Local structurality certificates for paper-faithful parameter strata.

The input is a one-dimensional piecewise-linear filtration family on a fixed
labeled complex. Distinct pair labels over an event can share a diagram value
without sharing a local continuation. We use all verified equivalence maps
between adjacent strata, as well as self-equivalences at the event, to decide
which equal-valued pairs have inseparable local continuations.
"""

from copy import deepcopy
from itertools import combinations
import math

from .complex import SimplicialComplex
from .contract import MATHEMATICAL_SCOPE
from .equivalence import CombinatorialEquivalences
from .equivalence import FiltrationEquivalences
from .equivalence import compatibility_conflicts
from .filtration import Filtration
from .pairing import PersistencePairing
from .pairing import STAR
from .stratification import Stratification
from .utils import _nonnegative_integer
from .utils import _positive_integer


__all__ = ["Structurality", "verified_stratum_adjacency",
           "adjacent_stratum_equivalences", "classify_structural_events"]


def _as_stratification(_input):
    """Recover a validated public stratification from an existing record."""
    if isinstance(_input, Stratification):
        return _input
    if not hasattr(_input, "get"):
        raise TypeError("Expected a Stratification or its output record")
    if _input.get("mathematical_scope") != MATHEMATICAL_SCOPE:
        raise ValueError("Expected a fixed-complex stratification record")
    try:
        _simplices = tuple(_input["complex_simplices"])
        _parameters = tuple(_input["parameters"])
        _rows = tuple(_input["sample_filtration_values"])
        _complex = SimplicialComplex(_simplices)
        _filtrations = tuple(Filtration(
            _complex, dict(zip(_simplices, _row)))
            for _row in _rows)
        if len(_rows) != len(_parameters) or any(
                len(_row) != len(_simplices) for _row in _rows):
            raise ValueError("Stratification samples do not cover the complex")
        _stratification = Stratification(
            _filtrations, _parameters, _input["tie_tolerance"])
    except (KeyError, TypeError) as _error:
        raise ValueError("Incomplete stratification record") from _error
    _rebuilt = _stratification.to_record()
    for _key in ("strata", "events", "stratum_adjacency"):
        if _input.get(_key) != _rebuilt[_key]:
            raise ValueError("Stratification record does not match its samples")
    return _stratification


def _signature(_filtration, _pair):
    """Return degree and augmented-diagram coordinates for one pair."""
    _birth, _death = _pair
    return (len(_birth) - 1, _filtration.value(_birth),
            math.inf if _death is STAR else _filtration.value(_death))


def _same_point(_first, _second, _atol):
    """Compare two numerical diagram points within stratification tolerance."""
    if _first[0] != _second[0]:
        return False
    if abs(_first[1] - _second[1]) > _atol:
        return False
    if math.isinf(_first[2]) or math.isinf(_second[2]):
        return _first[2] == _second[2]
    return abs(_first[2] - _second[2]) <= _atol


def _multiplicities(_filtration, _pairs, _atol, _dimension):
    """Group repeated augmented points, preserving all simplex labels."""
    _groups = []
    for _pair in _pairs:
        _point = _signature(_filtration, _pair)
        if _dimension is not None and _point[0] != _dimension:
            continue
        _match = next((_group for _group in _groups
                       if _same_point(_group["signature"], _point, _atol)),
                      None)
        if _match is None:
            _match = {"signature": _point, "pairs": []}
            _groups.append(_match)
        _match["pairs"].append(_pair)
    return tuple({"dimension": _group["signature"][0],
                  "birth": _group["signature"][1],
                  "death": _group["signature"][2],
                  "pairs": tuple(_group["pairs"]),
                  "multiplicity": len(_group["pairs"])}
                 for _group in _groups if len(_group["pairs"]) > 1)


def _join_mapping(_relations, _mapping):
    """Add undirected relations induced by one event self-map."""
    for _source, _target in _mapping:
        _relations[_source].add(_target)
        _relations[_target].add(_source)


def _join_round_trips(_relations, _witnesses):
    """Relate event pairs continued to the same neighbor in two ways."""
    _maps = tuple(dict(_witness["mapping"]) for _witness in _witnesses)
    for _first, _second in combinations(_maps, 2):
        _inverse = {_target: _source for _source, _target in _second.items()}
        if set(_first.values()) != set(_inverse):
            raise RuntimeError("Adjacency maps have different target pairs")
        for _source, _intermediate in _first.items():
            _target = _inverse[_intermediate]
            _relations[_source].add(_target)
            _relations[_target].add(_source)


def _orbits(_relations):
    """Return the connected local continuation class of each event pair."""
    _found = {}
    for _pair in _relations:
        if _pair in _found:
            continue
        _pending = [_pair]
        _component = set()
        while _pending:
            _current = _pending.pop()
            if _current in _component:
                continue
            _component.add(_current)
            _pending.extend(_relations[_current] - _component)
        _orbit = frozenset(_component)
        for _member in _component:
            _found[_member] = _orbit
    return _found


class Structurality:
    """Inspect an interval stratification using exact pairing equivalences.

    Classification is a certificate for the supplied one-dimensional stratum
    graph. A result is returned only after every needed equivalence search
    finishes within the supplied limits.

    :param _stratification: Public stratification or its output record.
    :type _stratification: Stratification or mapping
    :param _max_refinements: Maximum support orders per filtration.
    :type _max_refinements: int
    :param _max_equivalences: Maximum maps per comparison.
    :type _max_equivalences: int
    :param _max_common_refinements: Maximum orders shared by adjacent strata.
    :type _max_common_refinements: int
    """

    def __init__(self, _stratification, _max_refinements=5000,
                 _max_equivalences=100, _max_common_refinements=5000):
        self.stratification = _as_stratification(_stratification)
        self.max_refinements = _positive_integer(
            _max_refinements, "Maximum refinements")
        self.max_equivalences = _positive_integer(
            _max_equivalences, "Maximum equivalences")
        self.max_common_refinements = _positive_integer(
            _max_common_refinements, "Maximum common refinements")
        _adjacency = self.stratification.adjacency()
        self.strata = {_stratum["id"]: _stratum
                       for _stratum in self.stratification.strata}
        self.filtrations = {
            _identifier: self.stratification.filtration_at(
                _stratum["representative_parameter"])
            for _identifier, _stratum in self.strata.items()
        }
        self.pairings = {}
        self._transitions = {}
        _edges = []
        for _raw_edge in _adjacency["edges"]:
            _edge = dict(_raw_edge)
            _left = self.filtrations[_edge["source_stratum_id"]]
            _right = self.filtrations[_edge["target_stratum_id"]]
            _conflicts = compatibility_conflicts(_left, _right)
            _edge["compatible"] = not _conflicts
            _edge["compatibility_conflicts"] = _conflicts
            _edges.append(_edge)
        self.graph = {"nodes": _adjacency["nodes"],
                      "edges": tuple(_edges),
                      "compatibility_verified": True,
                      "all_edges_compatible": all(
                          _edge["compatible"] for _edge in _edges)}

    def _pairing(self, _identifier):
        """Cache the canonical pairing at a stratum representative."""
        if _identifier not in self.pairings:
            self.pairings[_identifier] = PersistencePairing(
                self.filtrations[_identifier])
        return self.pairings[_identifier]

    def verified_adjacency(self):
        """Return the incident-stratum graph with compatibility checks."""
        return deepcopy(self.graph)

    def adjacent_equivalences(self, _source_stratum_id, _target_stratum_id):
        """Return all certified maps across one adjacent stratum edge.

        :raises RuntimeError: If any equivalence enumeration is incomplete.
        """
        _source = _nonnegative_integer(_source_stratum_id,
                                       "Source stratum identifier")
        _target = _nonnegative_integer(_target_stratum_id,
                                       "Target stratum identifier")
        if _source == _target:
            raise ValueError("Adjacent strata must be distinct")
        _edges = tuple(_edge for _edge in self.graph["edges"]
                       if {_edge["source_stratum_id"],
                           _edge["target_stratum_id"]} == {_source, _target})
        if len(_edges) != 1:
            raise ValueError("Strata must share exactly one adjacency edge")
        if not _edges[0]["compatible"]:
            raise ValueError("Adjacent filtrations have opposing strict orders")
        if (_source, _target) not in self._transitions:
            _result = CombinatorialEquivalences(
                self.filtrations[_source], self.filtrations[_target],
                self._pairing(_source).order,
                self._pairing(_target).order,
                _max_common_refinements=self.max_common_refinements,
                _max_refinements=self.max_refinements,
                _max_equivalences=self.max_equivalences)
            if not _result.complete:
                raise RuntimeError("Adjacency equivalence search is incomplete")
            if not _result.witnesses:
                raise RuntimeError("Compatible strata produced no equivalence")
            _record = _result.as_dict()
            _record["source_stratum_id"] = _source
            _record["target_stratum_id"] = _target
            _record["adjacency_edge"] = deepcopy(_edges[0])
            self._transitions[(_source, _target)] = _record
        return deepcopy(self._transitions[(_source, _target)])

    def classify_events(self, _homology_dimension=None):
        """Classify repeated augmented points at all isolated event strata.

        :param _homology_dimension: Optional degree to examine.
        :type _homology_dimension: int or None
        :return: Event labels, pair orbits, and structural parameters.
        :rtype: dict
        """
        if _homology_dimension is not None:
            _homology_dimension = _nonnegative_integer(
                _homology_dimension, "Homology dimension")
        if not self.graph["all_edges_compatible"]:
            raise ValueError("Adjacent strata must have compatible filtrations")
        _atol = self.stratification.to_record()["tie_tolerance"]
        _events = []
        for _event_index, _event in enumerate(self.stratification.events):
            _identifier = _event["event_stratum_id"]
            if _identifier is None:
                continue
            _pairing = self._pairing(_identifier)
            _filtration = self.filtrations[_identifier]
            _self = FiltrationEquivalences(
                _filtration, _pairing.order, _pairing.order,
                self.max_refinements, self.max_equivalences)
            if not _self.complete:
                raise RuntimeError("Event self-equivalence search is incomplete")
            _relations = {_pair: {_pair} for _pair in _pairing.pairs}
            for _witness in _self.witnesses:
                _join_mapping(_relations, _witness["mapping"])

            _transitions = []
            for _edge in self.graph["edges"]:
                if _identifier not in (_edge["source_stratum_id"],
                                        _edge["target_stratum_id"]):
                    continue
                _neighbor = (_edge["target_stratum_id"]
                             if _edge["source_stratum_id"] == _identifier else
                             _edge["source_stratum_id"])
                _transition = self.adjacent_equivalences(
                    _identifier, _neighbor)
                _join_round_trips(_relations, _transition["equivalences"])
                _transitions.append({
                    "edge_id": _edge["id"],
                    "neighbor_stratum_id": _neighbor,
                    "equivalence_count": len(_transition["equivalences"]),
                    "complete": True})

            _components = _orbits(_relations)
            _groups = []
            for _group in _multiplicities(
                    _filtration, _pairing.pairs, _atol, _homology_dimension):
                _members = set(_group["pairs"])
                _non_hausdorff = tuple(
                    _pair for _pair in _group["pairs"]
                    if len(_components[_pair] & _members) > 1)
                _groups.append({**_group,
                                "non_hausdorff_pairs": _non_hausdorff,
                                "classification": ("structural" if
                                                   _non_hausdorff else
                                                   "coincidental")})
            _structural = tuple(_group for _group in _groups
                                if _group["classification"] == "structural")
            _orbits_seen = set(_components[_pair] for _pair in _pairing.pairs
                               if len(_components[_pair]) > 1 and
                               (_homology_dimension is None or
                                len(_pair[0]) == _homology_dimension + 1))
            _events.append({
                "event_index": _event_index,
                "parameter": _event["parameter"],
                "event_type": _event["event_type"],
                "event_stratum_id": _identifier,
                "left_stratum_id": _event["left_stratum_id"],
                "right_stratum_id": _event["right_stratum_id"],
                "classification": ("structural" if _structural else
                                   "coincidental" if _groups else
                                   "nonstructural"),
                "structural": bool(_structural),
                "multiplicity_points": tuple(_groups),
                "structural_points": _structural,
                "self_equivalence_count": len(_self.witnesses),
                "transport_orbits": tuple(sorted(
                    _orbits_seen,
                    key=lambda _orbit: tuple(sorted(map(repr, _orbit))))),
                "adjacent_transitions": tuple(_transitions),
                "complete": True,
            })
        return {"events": tuple(_events),
                "structural_parameters": tuple(
                    _event["parameter"] for _event in _events
                    if _event["structural"]),
                "homology_dimension": _homology_dimension,
                "complete": True,
                "certificate_scope": "local_one_dimensional_stratum_graph"}


def verified_stratum_adjacency(_stratification):
    """Return adjacency with filtration-compatibility certificates."""
    return Structurality(_stratification).verified_adjacency()


def adjacent_stratum_equivalences(
        _stratification, _source_stratum_id, _target_stratum_id,
        _max_refinements=5000, _max_equivalences=100,
        _max_common_refinements=5000):
    """Return complete pairing maps across one incident-stratum edge."""
    return Structurality(
        _stratification, _max_refinements, _max_equivalences,
        _max_common_refinements).adjacent_equivalences(
            _source_stratum_id, _target_stratum_id)


def classify_structural_events(
        _stratification, _homology_dimension=None, _max_refinements=5000,
        _max_equivalences=100, _max_common_refinements=5000):
    """Return Definition 3.15 local event classifications as a record."""
    return Structurality(
        _stratification, _max_refinements, _max_equivalences,
        _max_common_refinements).classify_events(_homology_dimension)
