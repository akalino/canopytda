"""Consistent simplex orders on one fixed filtered complex.

For a Filtration on a fixed complex K, the minimal consistent strict order
contains sigma < tau exactly when sigma is a proper face of tau or has a
smaller filtration value.
Equal-valued simplices with no face relation remain incomparable.
"""
from itertools import combinations

import gudhi as gd  # pylint: disable=import-error

from .filtration import Filtration
from .utils import _is_proper_face
from .utils import _nonnegative_integer
from .utils import _normalize_simplex


__all__ = ["MinimalConsistentOrder", "canonical_refinement",
           "is_consistent_refinement", "minimal_consistent_covers",
           "minimal_consistent_order", "total_order_refinements",
           "validate_refinement"]


def _simplex_key(_simplex):
    """
    Fetch face-first key for one simplex.

    :param _simplex: Canonical simplex.
    :type _simplex: tuple of int
    :return: Dimension-lexiographical key.
    :rtype: tuple
    """
    return (len(_simplex), _simplex)


def _as_filtration(_filtration):
    """Return an immutable filtration, converting a GUDHI tree if needed.

    :param _filtration: Filtration, order object or filtered GUDHI tree.
    :type _filtration: Filtration, MinimalConsistentOrder or gudhi.SimplexTree
    :return: Validated filtration on one fixed complex.
    :rtype: Filtration
    :raises TypeError: If the input is not a supported filtered complex.
    :raises ValueError: If values in a GUDHI tree are invalid.
    """
    if isinstance(_filtration, MinimalConsistentOrder):
        return _filtration._filtration
    if isinstance(_filtration, Filtration):
        return _filtration
    if isinstance(_filtration, gd.SimplexTree):
        return Filtration.from_simplex_tree(_filtration)
    raise TypeError("Expected a Filtration or gudhi.SimplexTree")


def _filtration_data(_filtration):
    """Return canonical simplices and values for one validated filtration.

    :param _filtration: Filtration, order object or filtered GUDHI tree.
    :type _filtration: Filtration, MinimalConsistentOrder or gudhi.SimplexTree
    :return: Canonical simplex tuples and simplex-to-value mapping.
    :rtype: tuple
    """
    _filtration = _as_filtration(_filtration)
    return _filtration.complex.simplices, _filtration.values


def _minimal_relations(_simplices, _values):
    """Construct the unique minimal strict relation in deterministic order."""
    _relations = []
    for _left in _simplices:
        for _right in _simplices:
            if (_left != _right and
                    (_values[_left] < _values[_right]
                     or _is_proper_face(_left, _right))):
                _relations.append((_left, _right))
    _relations.sort(
        key=lambda _relation: (
            _values[_relation[0]], _simplex_key(_relation[0]),
            _values[_relation[1]], _simplex_key(_relation[1]),
        )
    )
    return tuple(_relations)


def _cover_relations(_simplices, _relations):
    """Keep only those relations with no intermediate simplex."""
    _successors = {_simplex: set() for _simplex in _simplices}
    _predecessors = {_simplex: set() for _simplex in _simplices}
    for _left, _right in _relations:
        _successors[_left].add(_right)
        _predecessors[_right].add(_left)
    return tuple(
        (_left, _right) for _left, _right in _relations
        if not _successors[_left].intersection(_predecessors[_right])
    )


class MinimalConsistentOrder:
    """
    The minimal strict order associated with one filtration on K.

    This stores the unique partial order.
    Finding the total refinement is a separate choice.
    This takes a GUDHI tree as an immutable Filtration so later changes to the
    tree cannot change the recorded order.

    :param _filtration: One filtration on a fixed complex, or GUDHI tree.
    :type _filtration: Filtration or gudhi.SimplexTree
    :raises TypeError: If the input is not a filtration or a GUDHI tree.
    :raises ValueError: If values in GUDHI tree are invalid for the filtration.
    """
    # pylint: disable=no-member

    __slots__ = ("_filtration", "_relations")

    def __init__(self, _filtration):
        if hasattr(self, "_filtration"):
            raise AttributeError("MinimalConsistentOrder is immutable")
        _filtration = _as_filtration(_filtration)
        _relations = _minimal_relations(
            _filtration.complex.simplices, _filtration.values
        )
        object.__setattr__(self, "_filtration", _filtration)
        object.__setattr__(self, "_relations", _relations)

    def __setattr__(self, _name, _value):
        raise AttributeError("MinimalConsistentOrder is immutable")

    def __delattr__(self, _name):
        raise AttributeError("MinimalConsistentOrder is immutable")

    @property
    def complex(self):
        """Return the fixed labeled simplicial complex K."""
        return self._filtration.complex

    @property
    def relations(self):
        """Return every required ordered pair, including transitive pairs."""
        return self._relations

    @property
    def covers(self):
        """Return the edges of the minimal strict order."""
        return _cover_relations(self.complex.simplices, self._relations)

    def canonical_refinement(self):
        """Return the deterministic value, dimension and label refinement."""
        return canonical_refinement(self._filtration)

    def validate_refinement(self, _order):
        """Validate and normalize a total refinement.

        :param _order: Proposed permutation of the complex's simplices.
        :type _order: iterable of simplex tuples
        :return: Normalized total order.
        :rtype: tuple of tuple
        :raises ValueError: If the order is incomplete or inconsistent.
        """
        return validate_refinement(self._filtration, _order)

    def is_consistent_refinement(self, _order):
        """Return whether a proposed total refinement is valid.

        :param _order: Proposed permutation of the complex's simplices.
        :type _order: iterable of simplex tuples
        :return: Whether it is a consistent total order.
        :rtype: bool
        """
        return is_consistent_refinement(self._filtration, _order)

    def total_refinements(self, _limit=None):
        """
        Lazy enumeration of valid total refinements.

        :param _limit: Maximum refinements to return; None for no limit.
        :type: _limit: int or None
        :return: Iterator of total consistent orders.
        :rtype: iterator of tuple
        """
        return total_order_refinements(self._filtration, _limit=_limit)


def minimal_consistent_order(_filtration):
    """
    Construct the paper's unique minimal consistent strict order.

    The result contains a pair (sigma, tau) exactly when sigma must
    precede tau because it is a proper face or has a strictly smaller
    filtration value.
    The returned relation is transitively closed.

    :param _filtration: Filtration, order object or filtered GUDHI tree.
    :type _filtration: Filtration, MinimalConsistentOrder or gudhi.SimplexTree
    :return: Ordered pairs in the minimal consistent strict relation.
    :rtype: tuple of tuple
    :raises ValueError: If a GUDHI tree has invalid filtration values.
    """
    if isinstance(_filtration, MinimalConsistentOrder):
        return _filtration.relations
    _simplices, _values = _filtration_data(_filtration)
    return _minimal_relations(_simplices, _values)


def minimal_consistent_covers(_filtration):
    """
    Return the edges of the minimal consistent order.

    A relation (sigma, tau) is retained only when there is no intermediate
    simplex rho with sigma < rho < tau.

    :param _filtration: Filtration, order object or filtered GUDHI tree.
    :type _filtration: Filtration, MinimalConsistentOrder or gudhi.SimplexTree
    :return: Cover relations of the minimal consistent order.
    :rtype: tuple of tuple
    """
    if isinstance(_filtration, MinimalConsistentOrder):
        return _filtration.covers
    _simplices, _values = _filtration_data(_filtration)
    return _cover_relations(_simplices,
                            _minimal_relations(_simplices, _values))


def canonical_refinement(_filtration):
    """Return one deterministic total consistent order.

    Simplices are ordered first by filtration value, then by dimension, then
    lexicographically.  Dimension breaks equal-value face/coface ties in the
    required direction.

    :param _filtration: Filtration, order object or filtered GUDHI tree.
    :type _filtration: Filtration, MinimalConsistentOrder or gudhi.SimplexTree
    :return: Deterministic total-order refinement.
    :rtype: tuple of tuple
    """
    _simplices, _values = _filtration_data(_filtration)
    return tuple(sorted(_simplices,
                        key=lambda _simplex: (_values[_simplex],
                                              _simplex_key(_simplex))))


def validate_refinement(_filtration, _order):
    """Validate and normalize a total consistent refinement.

    :param _filtration: Filtration, order object or filtered GUDHI tree.
    :type _filtration: Filtration, MinimalConsistentOrder or gudhi.SimplexTree
    :param _order: Proposed total order containing every simplex once.
    :type _order: iterable of iterable of int
    :return: Normalized total order.
    :rtype: tuple of tuple
    :raises ValueError: If the order is incomplete, duplicated or inconsistent.
    """
    _simplices, _values = _filtration_data(_filtration)
    _normalized_order = tuple(_normalize_simplex(_simplex)
                              for _simplex in _order)

    if len(_normalized_order) != len(set(_normalized_order)):
        raise ValueError("Refinement contains duplicate simplices")
    if set(_normalized_order) != set(_simplices):
        raise ValueError("Refinement must contain every simplex exactly once")

    for _index in range(1, len(_normalized_order)):
        _left = _normalized_order[_index - 1]
        _right = _normalized_order[_index]
        if _values[_left] > _values[_right]:
            raise ValueError("Refinement decreases in filtration value")

    _positions = dict(
        (_simplex, _index)
        for _index, _simplex in enumerate(_normalized_order)
    )
    for _simplex in _simplices:
        if len(_simplex) == 1:
            continue
        for _face in combinations(_simplex, len(_simplex) - 1):
            if _positions[_face] > _positions[_simplex]:
                _message = "Face {} must precede coface {}"
                raise ValueError(_message.format(_face, _simplex))
    return _normalized_order


def is_consistent_refinement(_filtration, _order):
    """Return whether an order is a total consistent refinement.

    :param _filtration: Filtration, order object or filtered GUDHI tree.
    :type _filtration: Filtration, MinimalConsistentOrder or gudhi.SimplexTree
    :param _order: Proposed total order.
    :type _order: iterable of iterable of int
    :return: Whether the proposed order is valid.
    :rtype: bool
    """
    try:
        validate_refinement(_filtration, _order)
    except (TypeError, ValueError):
        return False
    return True


def total_order_refinements(_filtration, _limit=None):
    """
    Lazy enumeration for total refinements of the minimal consistent order.

    :param _filtration: Filtration, ordered object, or filtered GUDHI tree.
    :type _filtration: Filtration, MinimalConsistentOrder or gudhi.SimplexTree
    :param _limit: Optional maximum number of refinements to yield.
    :type _limit: int or None
    :return: lazy iterator of total consistent order.
    :rtype: iterator of tuple
    :raises ValueError: If the limit is not a nonnegative integer.
    """
    if _limit is not None:
        _limit = _nonnegative_integer(_limit, "Limit",
                                      _error_type=ValueError)

    _simplices, _values = _filtration_data(_filtration)
    _groups_by_value = {}
    for _simplex in _simplices:
        _groups_by_value.setdefault(_values[_simplex], []).append(_simplex)
    _groups = tuple(
        tuple(sorted(_groups_by_value[_value], key=_simplex_key))
        for _value in sorted(_groups_by_value)
    )

    def _block_refinements(_group):
        """Yield linear extensions within one filtration-value block."""

        _group_set = set(_group)
        _successors = dict((_simplex, []) for _simplex in _group)
        _indegrees = dict((_simplex, 0) for _simplex in _group)

        for _coface in _group:
            if len(_coface) == 1:
                continue
            for _face in combinations(_coface, len(_coface) - 1):
                if _face in _group_set:
                    _successors[_face].append(_coface)
                    _indegrees[_coface] += 1

        _available = tuple(sorted(
            (_simplex for _simplex in _group if _indegrees[_simplex] == 0),
            key=_simplex_key))

        def _extend(_prefix, _current_indegrees, _current_available):
            """Recursively extend one tied filtration block."""

            if len(_prefix) == len(_group):
                yield _prefix
                return

            for _candidate in _current_available:
                _next_indegrees = dict(_current_indegrees)
                _next_available = [
                    _simplex
                    for _simplex in _current_available
                    if _simplex != _candidate
                ]

                for _successor in _successors[_candidate]:
                    _next_indegrees[_successor] -= 1
                    if _next_indegrees[_successor] == 0:
                        _next_available.append(_successor)

                _next_available.sort(key=_simplex_key)
                yield from _extend(_prefix + (_candidate,), _next_indegrees,
                                   tuple(_next_available))

        yield from _extend((), _indegrees, _available)

    _state = {"yielded": 0}

    def _combine_blocks(_group_index, _prefix):
        """Combine one refinement from each filtration-value block."""

        if _limit is not None and _state["yielded"] >= _limit:
            return
        if _group_index == len(_groups):
            _state["yielded"] += 1
            yield _prefix
            return

        for _block_order in _block_refinements(_groups[_group_index]):
            yield from _combine_blocks(_group_index + 1,
                                       _prefix + _block_order)
            if _limit is not None and _state["yielded"] >= _limit:
                return

    return _combine_blocks(0, ())
