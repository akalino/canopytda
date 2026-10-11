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

import gudhi as gd #pylint: disable=import-error

from .complex import SimplicialComplex
from .complex import _normalize_simplex
from .complex import simplex_tree_from_point_cloud


__all__ = ["Filtration", "alpha_filtration", "available_filtrations",
           "build_filtration", "custom_filtration", "delaunay_cech_filtration",
           "explicit_filtration", "filtration_values",
           "lower_star_filtration", "rips_filtration"]


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
    # pylint: disable=no-member

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


def _validate_points(_points):
    """
    Reject an empty point cloud before calling GUDHI.

    :param _points: Point cloud with one point per row.
    :type _points: array-like
    :raises TypeError: If the point cloud has no length.
    :raises ValueError: If the point cloud is empty.
    """
    try:
        _number_of_points = len(_points)
    except TypeError as _error:
        raise TypeError("Point cloud must be a sized collection") from _error
    if _number_of_points == 0:
        raise ValueError("Point cloud must contain at least one point")


def _prune_dimension(_simplex_tree, _max_dimension):
    """
    Optionally remove simplices above a max dimension.

    :param _simplex_tree: Filtered simplicial complex to be modified.
    :type _simplex_tree: gudhi.SimplexTree
    :param _max_dimension: Largest dimension to keep,
                           or None to keep all dimensions.
    :type _max_dimension: int or None
    :return: The modified input simplex tree.
    :rtype: gudhi.SimplexTree
    :raises ValueError: If the dimension is invalid.
    """
    if _max_dimension is None:
        return _simplex_tree
    _dimension_is_integer = isinstance(_max_dimension, Integral)
    if isinstance(_max_dimension, bool) or not _dimension_is_integer:
        raise ValueError("Maximum dimension must be a nonnegative integer")
    if _max_dimension < 0:
        raise ValueError("Maximum dimension must be a nonnegative integer")
    _simplex_tree.prune_above_dimension(int(_max_dimension))
    return _simplex_tree


def _nonnegative_threshold(_value, _name):
    """
    Return a nonnegative threshold, allowing positive infinity.

    :param _value: Threshold to validate.
    :type _value: real
    :param _name: Name used in an error message.
    :type _name: str
    :return: Converted threshold.
    :rtype: float
    :raises TypeError: If the threshold cannot be converted to a float.
    :raises ValueError: If the threshold is negative or NaN.
    """
    try:
        _threshold = float(_value)
    except (TypeError, ValueError) as _error:
        raise TypeError("{} must be a real number".format(_name)) from _error
    if math.isnan(_threshold) or _threshold < 0.0:
        raise ValueError("{} must be nonnegative".format(_name))
    return _threshold


def alpha_filtration(_points, _max_alpha_square=math.inf,
                     _max_dimension=None, _precision="safe",
                     _output_squared_values=True):
    """
    Construct an Alpha filtration from a Euclidean point cloud.

    The construction threshold is always a squared radius.  Output values are
    squared radii by default and radii when _output_squared_values is
    false.

    :param _points: Euclidean point cloud with one point per row.
    :type _points: array-like of shape (n_points, ambient_dimension)
    :param _max_alpha_square: Largest squared alpha radius to include.
    :type _max_alpha_square: float
    :param _max_dimension: Optional largest dimension to retain.
    :type _max_dimension: int or None
    :param _precision: GUDHI precision mode: fast, safe or exact.
    :type _precision: str
    :param _output_squared_values: Whether output values remain squared.
    :type _output_squared_values: bool
    :return: Alpha filtration.
    :rtype: gudhi.SimplexTree
    """
    _validate_points(_points)
    _threshold = _nonnegative_threshold(_max_alpha_square,
                                        "Maximum alpha square")
    _alpha_complex = gd.AlphaComplex(points=_points, precision=_precision)
    _simplex_tree = _alpha_complex.create_simplex_tree(
        max_alpha_square=_threshold,
        output_squared_values=_output_squared_values)
    return _prune_dimension(_simplex_tree, _max_dimension)


def available_filtrations():
    """
    Return the canonical filtration choices accepted by the dispatcher.

    :return: Available filtration names.
    :rtype: tuple of str
    """
    return tuple(FILTRATION_BUILDERS)


def build_filtration(_choice, _data, **_parameters):
    """
    Build a filtration selected by name.

    For explicit and lower_star, _data is a fixed complex or simplex tree.
    For rips, alpha and delaunay_cech, it is a point cloud.
    Remaining keyword arguments are passed to the selected constructor.

    :param _choice: Filtration name or supported alias.
    :type _choice: str
    :param _data: Fixed complex, simplex tree or point cloud for the builder.
    :type _data: object
    :param _parameters: Arguments forwarded to the chosen builder.
    :type _parameters: dict
    :return: Selected filtration stored in a simplex tree.
    :rtype: gudhi.SimplexTree
    :raises ValueError: If the choice is not supported.
    """
    if not isinstance(_choice, str):
        raise TypeError("Filtration choice must be a string")
    _name = _choice.strip().lower().replace("-", "_")
    _name = FILTRATION_ALIASES.get(_name, _name)
    if _name not in FILTRATION_BUILDERS:
        _choices = ", ".join(available_filtrations())
        raise ValueError("Unknown filtration '{}'; "
                         "choose from {}".format(_choice, _choices))
    return FILTRATION_BUILDERS[_name](_data, **_parameters)


def custom_filtration(_complex, _function):
    """
    Build one filtration from a user-supplied simplex-value function.

    :param _complex: Fixed labeled simplicial complex.
    :type _complex: SimplicialComplex
    :param _function: Callable receiving a sorted simplex tuple.
    :type _function: callable
    :return: Validated filtration on the fixed complex.
    :rtype: Filtration
    :raises TypeError: If the complex or function has the wrong type.
    :raises ValueError: If values are nonfinite or violate face monotonicity.
    """
    if not isinstance(_complex, SimplicialComplex):
        raise TypeError("Expected a SimplicialComplex")
    if not callable(_function):
        raise TypeError("Expected a simplex-value function")

    _values = {
        _simplex: _function(_simplex)
        for _simplex in _complex.simplices
    }
    return Filtration(_complex, _values)


def delaunay_cech_filtration(_points, _max_radius_square=math.inf,
                             _max_dimension=None,
                             _precision="safe",
                             _output_squared_values=True):
    """
    Construct a Delaunay--Cech filtration from a point cloud.

    The Delaunay restriction has the persistence of the full Cech filtration
    while usually using fewer simplices.  The construction threshold is
    always a squared radius.

    :param _points: Euclidean point cloud with one point per row.
    :type _points: array-like of shape (n_points, ambient_dimension)
    :param _max_radius_square: Largest squared ball radius to include.
    :type _max_radius_square: float
    :param _max_dimension: Optional largest dimension to retain.
    :type _max_dimension: int or None
    :param _precision: GUDHI precision mode: fast, safe or exact.
    :type _precision: str
    :param _output_squared_values: Whether output values remain squared.
    :type _output_squared_values: bool
    :return: Delaunay--Cech filtration.
    :rtype: gudhi.SimplexTree
    """
    _validate_points(_points)
    _threshold = _nonnegative_threshold(_max_radius_square,
                                        "Maximum radius square")
    _cech_complex = gd.DelaunayCechComplex(points=_points,
                                           precision=_precision)
    _simplex_tree = _cech_complex.create_simplex_tree(
        max_alpha_square=_threshold,
        output_squared_values=_output_squared_values)
    return _prune_dimension(_simplex_tree, _max_dimension)


def explicit_filtration(_simplex_tree, _values):
    """
    Assign a paper-style filtration to a fixed simplicial complex.

    Every simplex must have one finite real value and every face value must be
    no larger than its coface value. A new filtered simplex tree is returned;
    the input complex or simplex tree is not modified.

    :param _simplex_tree: Fixed finite simplicial complex.
    :type _simplex_tree: SimplicialComplex or gudhi.SimplexTree
    :param _values: Filtration values keyed by simplex tuples.
    :type _values: mapping
    :return: Filtered copy of the input complex.
    :rtype: gudhi.SimplexTree
    :raises ValueError: If values are missing, extraneous or not monotone.
    """
    _filtration = Filtration(_simplex_tree, _values)
    if isinstance(_simplex_tree, SimplicialComplex):
        return _filtration.to_simplex_tree()
    _filtered_tree = _simplex_tree.copy()
    for _simplex in _filtration.complex.simplices:
        _filtered_tree.assign_filtration(_simplex, _filtration.value(_simplex))
    return _filtered_tree


def lower_star_filtration(_simplex_tree,
                          _vertex_values):
    """
    Construct the lower-star filtration induced by vertex values.

    The value of a simplex is the maximum value of one of its vertices.
    A new filtered simplex tree is returned; the input is not modified.

    :param _simplex_tree: Fixed finite simplicial complex.
    :type _simplex_tree: SimplicialComplex or gudhi.SimplexTree
    :param _vertex_values: Finite values keyed by integer vertex identifiers.
    :type _vertex_values: mapping
    :return: Lower-star filtered copy of the input complex.
    :rtype: gudhi.SimplexTree
    :raises ValueError: If a vertex value is missing or extraneous.
    """
    _filtration = Filtration.lower_star(_simplex_tree, _vertex_values)
    if isinstance(_simplex_tree, SimplicialComplex):
        return _filtration.to_simplex_tree()
    _filtered_tree = _simplex_tree.copy()
    for _simplex in _filtration.complex.simplices:
        _filtered_tree.assign_filtration(_simplex, _filtration.value(_simplex))
    return _filtered_tree


def rips_filtration(_points, _max_edge_length, _max_dimension=2):
    """
    Construct a Vietoris--Rips filtration from a point cloud.

    :param _points: Point cloud with one point per row.
    :type _points: array-like of shape (n_points, ambient_dimension)
    :param _max_edge_length: Largest edge length to include.
    :type _max_edge_length: float
    :param _max_dimension: Largest simplex dimension to construct.
    :type _max_dimension: int
    :return: Vietoris--Rips filtration.
    :rtype: gudhi.SimplexTree
    """
    return simplex_tree_from_point_cloud(_points, _max_edge_length,
                                         _max_dimension)


FILTRATION_BUILDERS = {"explicit": explicit_filtration,
                       "lower_star": lower_star_filtration,
                       "rips": rips_filtration, "alpha": alpha_filtration,
                       "delaunay_cech": delaunay_cech_filtration}

FILTRATION_ALIASES = {"vr": "rips", "vietoris_rips": "rips",
                      "cech": "delaunay_cech"}
