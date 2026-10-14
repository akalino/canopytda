"""Paper-faithful equivalences between persistence simplex pairings.

All filtrations share one fixed labeled simplicial complex. Exact support
checks enumerate total consistent orders and raise if the safety limit is
exceeded; a truncated list of equivalence maps has ``complete=False``.
"""

from itertools import combinations
import math

from .filtration import Filtration
from .orders import canonical_refinement
from .orders import total_order_refinements
from .orders import validate_refinement
from .pairing import PersistencePairing
from .pairing import STAR
from .utils import _nonnegative_integer
from .utils import _positive_integer


__all__ = ["FiltrationEquivalences", "CombinatorialEquivalences",
           "compatibility_conflicts", "filtrations_compatible",
           "common_refinement", "common_refinements",
           "filtration_equivalences", "combinatorial_equivalences",
           "all_combinatorial_equivalences", "compose_equivalences"]


def _as_filtration(_input):
    """Normalize one filtration using the public pairing adapter."""
    if isinstance(_input, Filtration):
        return _input
    return PersistencePairing(_input).filtration


def _shared_data(_left, _right):
    """Require exactly the same labeled simplices in both filtrations."""
    _left = _as_filtration(_left)
    _right = _as_filtration(_right)
    if _left.complex != _right.complex:
        raise ValueError("Filtrations must have the same labeled complex")
    return _left, _right, _left.complex.simplices


def compatibility_conflicts(_left, _right):
    """Return simplex pairs with opposing strict filtration orders.

    :param _left: First filtration.
    :type _left: Filtration or gudhi.SimplexTree
    :param _right: Second filtration on the same labeled complex.
    :type _right: Filtration or gudhi.SimplexTree
    :return: Reversed pairs, each occurring exactly once.
    :rtype: tuple
    """
    _left, _right, _simplices = _shared_data(_left, _right)
    return tuple(
        (_first, _second)
        for _first, _second in combinations(_simplices, 2)
        if ((_left.value(_first) < _left.value(_second) and
             _right.value(_first) > _right.value(_second)) or
            (_left.value(_first) > _left.value(_second) and
             _right.value(_first) < _right.value(_second)))
    )


def filtrations_compatible(_left, _right):
    """Check compatibility as in Definition 2.16 of *Canopies*."""
    return not compatibility_conflicts(_left, _right)


def common_refinement(_left, _right):
    """Return one deterministic total order valid for both filtrations.

    :raises ValueError: If the filtrations have conflicting strict orders.
    """
    _left, _right, _simplices = _shared_data(_left, _right)
    if compatibility_conflicts(_left, _right):
        raise ValueError("Filtrations have opposing strict simplex orders")
    _order = tuple(sorted(_simplices, key=lambda _simplex: (
        _left.value(_simplex), _right.value(_simplex),
        len(_simplex), _simplex)))
    validate_refinement(_left, _order)
    validate_refinement(_right, _order)
    return _order


def common_refinements(_left, _right, _limit=None):
    """Lazily enumerate orders consistent with both filtrations.

    :param _limit: Maximum orders to yield; ``None`` means no limit.
    :type _limit: int or None
    :return: Deterministically ordered common total refinements.
    :rtype: iterator of tuple
    """
    if _limit is not None:
        _limit = _nonnegative_integer(_limit, "Limit", _error_type=ValueError)
    _left, _right, _simplices = _shared_data(_left, _right)
    if compatibility_conflicts(_left, _right):
        raise ValueError("Filtrations have opposing strict simplex orders")

    _successors = {_simplex: set() for _simplex in _simplices}
    for _first, _second in combinations(_simplices, 2):
        if (_left.value(_first) < _left.value(_second) or
                _right.value(_first) < _right.value(_second)):
            _successors[_first].add(_second)
        elif (_left.value(_second) < _left.value(_first) or
              _right.value(_second) < _right.value(_first)):
            _successors[_second].add(_first)
        elif set(_first) < set(_second):
            _successors[_first].add(_second)
        elif set(_second) < set(_first):
            _successors[_second].add(_first)

    _indegrees = {_simplex: 0 for _simplex in _simplices}
    for _targets in _successors.values():
        for _target in _targets:
            _indegrees[_target] += 1
    _available = tuple(_simplex for _simplex in _simplices
                       if _indegrees[_simplex] == 0)
    _yielded = [0]

    def _extend(_prefix, _degrees, _ready):
        """Build one topological ordering at a time."""
        if _limit is not None and _yielded[0] >= _limit:
            return
        if len(_prefix) == len(_simplices):
            _yielded[0] += 1
            yield _prefix
            return
        for _candidate in _ready:
            _next_degrees = dict(_degrees)
            _next_ready = [
                _simplex for _simplex in _ready if _simplex != _candidate
            ]
            for _successor in _successors[_candidate]:
                _next_degrees[_successor] -= 1
                if _next_degrees[_successor] == 0:
                    _next_ready.append(_successor)
            _next_ready.sort(key=lambda _simplex: (len(_simplex), _simplex))
            yield from _extend(_prefix + (_candidate,), _next_degrees,
                               tuple(_next_ready))
            if _limit is not None and _yielded[0] >= _limit:
                return

    return _extend((), _indegrees, _available)


def _support_pairings(_filtration, _limit):
    """Index distinct valid pairings by their simplex pairs and one order."""
    _supports = {}
    _count = 0
    for _count, _order in enumerate(total_order_refinements(
            _filtration, _limit=_limit + 1), start=1):
        if _count > _limit:
            raise RuntimeError("Support-order search exceeds {}".format(_limit))
        _pairs = frozenset(PersistencePairing(_filtration, _order).pairs)
        _supports.setdefault(_pairs, _order)
    return _supports, _count


def _signature(_filtration, _pair):
    """Return the homology degree and both interval endpoints."""
    _birth, _death = _pair
    return (len(_birth) - 1, _filtration.value(_birth),
            math.inf if _death is STAR else _filtration.value(_death))


def _filtration_witnesses(_filtration, _source, _target, _max_refinements,
                          _max_equivalences, _support_cache=None):
    """Match pairs with a cross-pair support certificate where required."""
    _candidates = {_index: [] for _index in range(len(_source.pairs))}
    _crossed = {}
    for _index, _pair in enumerate(_source.pairs):
        for _other_index, _other_pair in enumerate(_target.pairs):
            if _signature(_filtration, _pair) != _signature(
                    _filtration, _other_pair):
                continue
            _birth, _death = _pair
            _other_birth, _other_death = _other_pair
            if _birth == _other_birth or _death == _other_death:
                _candidates[_index].append((_other_index, None))
            else:
                _crossed[(_index, _other_index)] = frozenset((
                    (_birth, _other_death), (_other_birth, _death)))

    _examined = 0
    if _crossed:
        if _support_cache is None:
            _supports, _examined = _support_pairings(
                _filtration, _max_refinements)
        else:
            if _support_cache["data"] is None:
                _support_cache["data"] = _support_pairings(
                    _filtration, _max_refinements)
            _supports, _examined = _support_cache["data"]
        for (_index, _other_index), _required in _crossed.items():
            for _pairs, _order in _supports.items():
                if _required.issubset(_pairs):
                    _candidates[_index].append((_other_index, _order))
                    break

    _search_order = sorted(_candidates,
                           key=lambda _index: (len(_candidates[_index]), _index))
    _assignments = {}
    _used = set()
    _found = []

    def _extend(_depth):
        """Enumerate bijections, stopping after the first excess map."""
        if len(_found) > _max_equivalences:
            return
        if _depth == len(_search_order):
            _found.append({
                "mapping": tuple((_source.pairs[_index],
                                  _target.pairs[_assignments[_index][0]])
                                 for _index in range(len(_source.pairs))),
                "support_orders": tuple(_assignments[_index][1]
                                        for _index in range(len(_source.pairs))),
            })
            return
        _index = _search_order[_depth]
        for _other_index, _support in _candidates[_index]:
            if _other_index in _used:
                continue
            _assignments[_index] = (_other_index, _support)
            _used.add(_other_index)
            _extend(_depth + 1)
            _used.remove(_other_index)
            del _assignments[_index]
            if len(_found) > _max_equivalences:
                return

    if len(_source) == len(_target):
        _extend(0)
    return tuple(_found[:_max_equivalences]), len(_found) <= _max_equivalences, _examined


class FiltrationEquivalences:
    """Enumerate Definition 2.18 maps for two orders of one filtration.

    :param _filtration: One fixed filtration.
    :type _filtration: Filtration or gudhi.SimplexTree
    :param _source_order: Consistent order for the source pairing.
    :type _source_order: iterable of simplices
    :param _target_order: Consistent order for the target pairing.
    :type _target_order: iterable of simplices
    :param _max_refinements: Bound on exact cross-pair support search.
    :type _max_refinements: int
    :param _max_equivalences: Bound on returned maps.
    :type _max_equivalences: int
    """

    def __init__(self, _filtration, _source_order, _target_order,
                 _max_refinements=5000, _max_equivalences=100):
        _filtration = _as_filtration(_filtration)
        _max_refinements = _positive_integer(_max_refinements,
                                              "Maximum refinements")
        _max_equivalences = _positive_integer(_max_equivalences,
                                              "Maximum equivalences")
        self.source = PersistencePairing(_filtration, _source_order)
        self.target = PersistencePairing(_filtration, _target_order)
        self.witnesses, self.complete, self.refinements_examined = \
            _filtration_witnesses(_filtration, self.source, self.target,
                                  _max_refinements, _max_equivalences)

    @property
    def mappings(self):
        """Return the pair bijections in source pairing order."""
        return tuple(_witness["mapping"] for _witness in self.witnesses)

    def as_dict(self):
        """Return the record consumed by existing structurality code."""
        return {"source_order": self.source.order,
                "target_order": self.target.order,
                "source_pairing": self.source.pairs,
                "target_pairing": self.target.pairs,
                "equivalences": self.witnesses,
                "complete": self.complete,
                "refinements_examined": self.refinements_examined}


def compose_equivalences(_first, _second):
    """Compose two bijections represented by source/target pair entries.

    :raises ValueError: If either map is not bijective or endpoints disagree.
    """
    try:
        _first = tuple((_source, _target) for _source, _target in _first)
        _second = tuple((_source, _target) for _source, _target in _second)
        _lookup = dict(_second)
        if (len(set(_source for _source, _ in _first)) != len(_first) or
                len(set(_target for _, _target in _first)) != len(_first) or
                len(_lookup) != len(_second) or
                len(set(_target for _, _target in _second)) != len(_second) or
                set(_target for _, _target in _first) != set(_lookup)):
            raise ValueError("Equivalence maps must be composable bijections")
        return tuple((_source, _lookup[_middle])
                     for _source, _middle in _first)
    except (TypeError, KeyError) as _error:
        raise ValueError("Equivalence maps must be composable bijections") from _error


class CombinatorialEquivalences:
    """Construct Definition 2.21 maps through common consistent orders.

    By default all common orders are examined, so ``complete`` applies to
    every possible shared pairing. A supplied ``_common_order`` restricts
    the search to that order and is useful for an individual witness.

    :param _left: Source filtration on the fixed complex.
    :type _left: Filtration or gudhi.SimplexTree
    :param _right: Target filtration on the same complex.
    :type _right: Filtration or gudhi.SimplexTree
    :param _left_order: Optional source order.
    :type _left_order: iterable or None
    :param _right_order: Optional target order.
    :type _right_order: iterable or None
    :param _common_order: Optional chosen shared order.
    :type _common_order: iterable or None
    :param _max_common_refinements: Exact common-order search cap.
    :type _max_common_refinements: int
    :param _max_refinements: Exact support-order search cap per filtration.
    :type _max_refinements: int
    :param _max_equivalences: Maximum maps returned.
    :type _max_equivalences: int
    """

    def __init__(self, _left, _right, _left_order=None, _right_order=None,
                 _common_order=None, _max_common_refinements=5000,
                 _max_refinements=5000, _max_equivalences=100):
        _left, _right, _simplices = _shared_data(_left, _right)
        if compatibility_conflicts(_left, _right):
            raise ValueError("Combinatorial equivalence requires compatible filtrations")
        _max_common_refinements = _positive_integer(
            _max_common_refinements, "Maximum common refinements")
        _max_refinements = _positive_integer(
            _max_refinements, "Maximum refinements")
        _max_equivalences = _positive_integer(
            _max_equivalences, "Maximum equivalences")
        self.left = PersistencePairing(_left, _left_order)
        self.right = PersistencePairing(_right, _right_order)
        if _common_order is None:
            _orders = tuple(common_refinements(
                _left, _right, _max_common_refinements + 1))
            if len(_orders) > _max_common_refinements:
                raise RuntimeError("Common-order search exceeds {}".format(
                    _max_common_refinements))
        else:
            _order = validate_refinement(_left, _common_order)
            validate_refinement(_right, _order)
            _orders = (_order,)
        self.common_refinement_count = len(_orders)
        _representatives = {}
        for _order in _orders:
            _pairing = PersistencePairing(_left, _order)
            _representatives.setdefault(frozenset(_pairing.pairs), _order)
        self.common_orders = tuple(_representatives.values())
        self.common_pairing_count = len(_representatives)
        self.search_scope = ("all_common_total_refinements"
                             if _common_order is None else
                             "one_common_total_refinement")
        self.common_order = self.common_orders[0]
        self.common_pairing = PersistencePairing(_left, self.common_order).pairs

        _left_cache = {"data": None}
        _right_cache = {"data": None}
        _seen = set()
        _found = []
        _complete = True
        _examined = 0
        for _order in self.common_orders:
            _middle = PersistencePairing(_left, _order)
            # Each common order produces the same pairing for both functions.
            if _middle.pairs != PersistencePairing(_right, _order).pairs:
                raise RuntimeError("The shared order produced different pairings")
            _before, _before_complete, _before_count = _filtration_witnesses(
                _left, self.left, _middle, _max_refinements,
                _max_equivalences, _left_cache)
            _after, _after_complete, _after_count = _filtration_witnesses(
                _right, _middle, self.right, _max_refinements,
                _max_equivalences, _right_cache)
            _examined = max(_examined, _before_count + _after_count)
            _complete = _complete and _before_complete and _after_complete
            for _first in _before:
                for _second in _after:
                    _mapping = compose_equivalences(_first["mapping"],
                                                    _second["mapping"])
                    if _mapping in _seen:
                        continue
                    _seen.add(_mapping)
                    if len(_found) == _max_equivalences:
                        _complete = False
                        continue
                    _found.append({"mapping": _mapping,
                                   "common_order": _order,
                                   "common_pairing": _middle.pairs,
                                   "left_filtration_witness": _first,
                                   "right_filtration_witness": _second})
        self.witnesses = tuple(_found)
        self.complete = _complete
        self.refinements_examined = _examined

    @property
    def mappings(self):
        """Return distinct Definition 2.21 pair bijections."""
        return tuple(_witness["mapping"] for _witness in self.witnesses)

    def as_dict(self):
        """Return the record consumed by existing canopy construction."""
        return {"left_order": self.left.order,
                "right_order": self.right.order,
                "common_order": self.common_order,
                "left_pairing": self.left.pairs,
                "right_pairing": self.right.pairs,
                "common_pairing": self.common_pairing,
                "equivalences": self.witnesses,
                "complete": self.complete,
                "common_refinement_count": self.common_refinement_count,
                "common_pairing_count": self.common_pairing_count,
                "search_scope": self.search_scope,
                "refinements_examined": self.refinements_examined}


def filtration_equivalences(_filtration, _source_order, _target_order,
                            _max_refinements=5000, _max_equivalences=100):
    """Return Definition 2.18 witnesses as a compatibility record."""
    return FiltrationEquivalences(
        _filtration, _source_order, _target_order, _max_refinements,
        _max_equivalences).as_dict()


def combinatorial_equivalences(_left, _right, _left_order=None,
                               _right_order=None, _common_order=None,
                               _max_refinements=5000, _max_equivalences=100):
    """Return maps through one deterministic or supplied shared order."""
    if _common_order is None:
        _common_order = common_refinement(_left, _right)
    return CombinatorialEquivalences(
        _left, _right, _left_order, _right_order, _common_order,
        _max_refinements=_max_refinements,
        _max_equivalences=_max_equivalences).as_dict()


def all_combinatorial_equivalences(
        _left, _right, _left_order=None, _right_order=None,
        _max_common_refinements=5000, _max_refinements=5000,
        _max_equivalences=100):
    """Return maps through every shared order, subject to explicit caps."""
    return CombinatorialEquivalences(
        _left, _right, _left_order, _right_order,
        _max_common_refinements=_max_common_refinements,
        _max_refinements=_max_refinements,
        _max_equivalences=_max_equivalences).as_dict()
