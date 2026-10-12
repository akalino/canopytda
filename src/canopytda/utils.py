"""Shared input validation and simplex predicates.

This module uses only the standard library so that the input and
stratification modules can import it without circular dependencies.
"""

import math
from numbers import Integral


def _finite_float(_value, _name, _error_type=ValueError,
                  _allow_infinity=False):
    """Return a validated floating-point value.

    :param _value: Candidate value.
    :type _value: real
    :param _name: Input name used in error messages.
    :type _name: str
    :param _error_type: Error class for a failed conversion.
    :type _error_type: type
    :param _allow_infinity: Permit infinite values for unbounded thresholds.
    :type _allow_infinity: bool
    :return: Converted value.
    :rtype: float
    """
    try:
        _number = float(_value)
    except (TypeError, ValueError) as _error:
        raise _error_type("{} must be a real number".format(_name)) from _error
    if math.isnan(_number) or (
            not _allow_infinity and not math.isfinite(_number)):
        raise ValueError("{} must be finite".format(_name))
    return _number


def _nonnegative_float(_value, _name, _allow_infinity=False,
                       _error_type=ValueError):
    """Return a nonnegative float with the requested infinity policy.

    :param _value: Candidate value.
    :type _value: real
    :param _name: Input name used in error messages.
    :type _name: str
    :param _allow_infinity: Permit positive infinity.
    :type _allow_infinity: bool
    :param _error_type: Error class for a failed conversion.
    :type _error_type: type
    :return: Validated value.
    :rtype: float
    """
    _number = _finite_float(_value, _name, _error_type, _allow_infinity)
    if _number < 0.0:
        raise ValueError("{} must be nonnegative".format(_name))
    return _number


def _nonnegative_integer(_value, _name, _error_type=TypeError):
    """Return a nonnegative integer, excluding booleans.

    :param _value: Candidate dimension or count.
    :type _value: object
    :param _name: Input name used in error messages.
    :type _name: str
    :param _error_type: Error class for invalid integer inputs.
    :type _error_type: type
    :return: Validated integer.
    :rtype: int
    """
    if isinstance(_value, bool) or not isinstance(_value, Integral):
        raise _error_type("{} must be a nonnegative integer".format(_name))
    _integer = int(_value)
    if _integer < 0:
        raise ValueError("{} must be nonnegative".format(_name))
    return _integer


def _positive_integer(_value, _name):
    """Return a strictly positive integer.

    :param _value: Candidate count.
    :type _value: object
    :param _name: Input name used in error messages.
    :type _name: str
    :return: Validated integer.
    :rtype: int
    """
    _integer = _nonnegative_integer(_value, _name)
    if _integer == 0:
        raise ValueError("{} must be positive".format(_name))
    return _integer


def _normalize_simplex(_simplex):
    """Return a sorted nonempty simplex of distinct integer vertices.

    :param _simplex: Vertex identifiers.
    :type _simplex: iterable of int
    :return: Canonical simplex.
    :rtype: tuple of int
    """
    _vertices = tuple(_simplex)
    if not _vertices:
        raise ValueError("Empty simplices implied, should not be supplied")
    if any(isinstance(_vertex, bool) or
           not isinstance(_vertex, Integral) for _vertex in _vertices):
        raise TypeError("Vertex identifiers must be integers")
    _vertices = tuple(int(_vertex) for _vertex in _vertices)
    if len(set(_vertices)) != len(_vertices):
        raise ValueError("Vertex identifiers must be unique")
    return tuple(sorted(_vertices))


def _is_proper_face(_left, _right):
    """Return whether the first simplex is a proper face of the second.

    :param _left: Candidate face.
    :type _left: tuple of int
    :param _right: Candidate coface.
    :type _right: tuple of int
    :return: Whether proper containment holds.
    :rtype: bool
    """
    return len(_left) < len(_right) and all(
        _vertex in _right for _vertex in _left
    )


def _are_face_related(_left, _right):
    """Return whether either simplex is a proper face of the other.

    :param _left: First simplex.
    :type _left: tuple of int
    :param _right: Second simplex.
    :type _right: tuple of int
    :return: Whether proper containment holds in either direction.
    :rtype: bool
    """
    return _is_proper_face(_left, _right) or _is_proper_face(_right, _left)


def _validate_points(_points):
    """Reject an unsized or empty point collection.

    :param _points: Point cloud with one point per row.
    :type _points: array-like
    :raises TypeError: If the collection has no length.
    :raises ValueError: If it is empty.
    """
    try:
        _count = len(_points)
    except TypeError as _error:
        raise TypeError("Point cloud must be a sized collection") from _error
    if _count == 0:
        raise ValueError("Point cloud must contain at least one point")
