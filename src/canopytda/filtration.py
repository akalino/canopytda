"""
Filtrations on a fixed complex and GUDHI construction adapters.

The paper permits any finite, face-monotone function on a fixed simplicial
complex. :class:`Filtration` represents one such function at one parameter
value. A piecewise-linear family of them is handled by ``stratification.py``.
"""
import math

from itertools import combinations
from numbers import Integral
from types import MappingProxyType

import gudhi as gd

from .complex import SimplicialComplex
from .complex import _normalize_simplex
from .complex import ordered_simplices
from .complex import simplex_tree_from_point_cloud


__all__ = ['Filtration']


def _finite_float(_value, _name):
    """
    Return a finite floating-point value.

    :param _value: Value to convert.
    :type _value: real
    :param _name: Name used in an error message.
    :type _name: str
    :return: Converted finite value.
    :rtype: float
    :raises TypeError: If the value cannot be converted to a float.
    :raises ValueError: If the value is not finite.
    """
    try:
        _number = float(_value)
    except (TypeError, ValueError) as _error:
        raise TypeError("{} must be a real number".format(_name)) from _error
    if not math.isfinite(_number):
        raise ValueError("{} must be finite".format(_name))
    return _number


def _normalized_simplex_values(_values):
    """Normalize a mapping from simplices to finite values.

    :param _values: Filtration values keyed by simplices.
    :type _values: mapping
    :return: Canonical simplex-to-value mapping.
    :rtype: dict
    :raises TypeError: If the input is not a mapping.
    :raises ValueError: If two keys denote the same simplex.
    """

    if not hasattr(_values, "items"):
        raise TypeError("Filtration values must be supplied as a mapping")

    _normalized = {}
    for _simplex, _value in _values.items():
        _canonical_simplex = _normalize_simplex(_simplex)
        if _canonical_simplex in _normalized:
            raise ValueError("Filtration mapping repeats a simplex")
        _name = "Filtration value for {}".format(_canonical_simplex)
        _normalized[_canonical_simplex] = _finite_float(_value, _name)
    return _normalized


def _validate_explicit_values(_simplices, _values):
    """Check coverage and face monotonicity for explicit values.

    :param _simplices: All simplices in a finite complex.
    :type _simplices: iterable of tuple
    :param _values: Canonical filtration mapping.
    :type _values: dict
    :raises ValueError: If coverage or face monotonicity fails.
    """
    _simplex_set = set(_simplices)
    _value_set = set(_values)
    _missing = sorted(_simplex_set - _value_set)
    _extra = sorted(_value_set - _simplex_set)

    if _missing or _extra:
        _parts = []
        if _missing:
            _parts.append("missing {}".format(_missing))
        if _extra:
            _parts.append("unknown {}".format(_extra))
        raise ValueError("Filtration mapping has " + " and ".join(_parts))

    for _simplex in _simplices:
        if len(_simplex) == 1:
            continue
        for _face in combinations(_simplex, len(_simplex) - 1):
            if _values[_face] > _values[_simplex]:
                _message = "Filtration decreases from face {} to simplex {}"
                raise ValueError(_message.format(_face, _simplex))


class Filtration:
    """A finite, face-monotone filtration function on one fixed complex.

    This is a single function ``f: K -> R`` with one value for every nonempty
     simplex; it has no parameter interval or point-cloud correspondence.
    Values and the underlying complex cannot be modified after construction.

    :param _complex: Fixed complex, or a GUDHI tree whose simplex domain is K.
    :type _complex: SimplicialComplex or gudhi.SimplexTree
    :param _values: One finite real value per simplex, including all faces.
    :type _values: mapping
    :raises ValueError: If values do not cover K or decrease along a face.
    """

    __slots__ = ("_complex", "_values")

    def __init__(self, _complex, _values):
        if hasattr(self, "_complex"):
            raise AttributeError("Filtration is immutable")
        if not isinstance(_complex, SimplicialComplex):
            _complex = SimplicialComplex.from_simplex_tree(_complex)
        _normalized = _normalized_simplex_values(_values)
        _validate_explicit_values(_complex.simplices, _normalized)
        object.__setattr__(self, "_complex", _complex)
        object.__setattr__(self, "_values", MappingProxyType(_normalized))

    def __setattr__(self, _name, _value):
        raise AttributeError("Filtration is immutable")

    def __delattr__(self, _name):
        raise AttributeError("Filtration is immutable")

    @property
    def complex(self):
        """Return the fixed labeled simplicial complex K."""
        return self._complex

    @property
    def values(self):
        """Return a read-only map from simplex tuples to finite floats."""
        return self._values

    def value(self, _simplex):
        """Return the value of a simplex, independent of vertex order.

        :param _simplex: A simplex in K.
        :type _simplex: iterable of int
        :return: Filtration value.
        :rtype: float
        :raises KeyError: If the simplex is not in K.
        """
        return self._values[_normalize_simplex(_simplex)]

    def to_simplex_tree(self):
        """Return an independent GUDHI simplex tree with these values.

        :return: Filtered copy of K for persistence and canopy algorithms.
        :rtype: gudhi.SimplexTree
        """
        _filtered_tree = self._complex.to_simplex_tree()
        for _simplex in self._complex.simplices:
            _filtered_tree.assign_filtration(_simplex, self._values[_simplex])
        return _filtered_tree

    @classmethod
    def from_simplex_tree(cls, _simplex_tree):
        """Capture the complex and filtration values from a GUDHI tree.

        :param _simplex_tree: Filtered simplex tree.
        :type _simplex_tree: gudhi.SimplexTree
        :return: Filtration on a copy of its fixed simplex domain.
        :rtype: Filtration
        """
        return cls(
            SimplicialComplex.from_simplex_tree(_simplex_tree),
            filtration_values(_simplex_tree)
        )

    @classmethod
    def lower_star(cls, _complex, _vertex_values):
        """Build the lower-star function induced by vertex values.

        :param _complex: Fixed complex or GUDHI tree with the same domain.
        :type _complex: SimplicialComplex or gudhi.SimplexTree
        :param _vertex_values: Finite values keyed by vertex identifier.
        :type _vertex_values: mapping
        :return: The induced face-monotone filtration.
        :rtype: Filtration
        """
        if not isinstance(_complex, SimplicialComplex):
            _complex = SimplicialComplex.from_simplex_tree(_complex)
        _values = _normalized_vertex_values(_complex.vertices, _vertex_values)
        return cls(
            _complex,
            dict(
                (_simplex, max(_values[_vertex] for _vertex in _simplex))
                for _simplex in _complex.simplices
            )
        )

def filtration_values(_simplex_tree):
    """
    Return simplex values from a filtration object or GUDHI tree.

    :param _simplex_tree: Filtered finite simplicial complex.
    :type _simplex_tree: Filtration or gudhi.SimplexTree
    :return: Filtration values keyed by sorted simplex tuples.
    :rtype: dict
    """

    if isinstance(_simplex_tree, Filtration):
        return dict(_simplex_tree.values)
    return dict(
        (tuple(sorted(_simplex)), _value)
        for _simplex, _value in _simplex_tree.get_simplices()
    )

def _normalized_vertex_values(_vertices, _vertex_values):
    """
    Validate values for exactly the vertices of a fixed complex.

    :param _vertices: Vertex identifiers in K.
    :type _vertices: iterable of int
    :param _vertex_values: Candidate vertex-to-value mapping.
    :type _vertex_values: mapping
    :return: Finite values keyed by integer identifiers.
    :rtype: dict
    """
    if not hasattr(_vertex_values, "items"):
        raise TypeError("Vertex values must be supplied as a mapping")
    _values = {}
    for _vertex, _value in _vertex_values.items():
        if isinstance(_vertex, bool) or not isinstance(_vertex, Integral):
            raise TypeError("Vertex-value keys must be integers")
        _vertex_id = int(_vertex)
        _name = "Value for vertex {}".format(_vertex_id)
        _values[_vertex_id] = _finite_float(_value, _name)
    _missing = sorted(set(_vertices) - set(_values))
    _extra = sorted(set(_values) - set(_vertices))
    if _missing or _extra:
        _parts = []
        if _missing:
            _parts.append("missing {}".format(_missing))
        if _extra:
            _parts.append("unknown {}".format(_extra))
        raise ValueError("Vertex-value mapping has " + " and ".join(_parts))
    return _values