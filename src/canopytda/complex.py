"""
Base classes for representing fixed simplicial complexes.
"""
import math

from itertools import combinations
from numbers import Integral

import gudhi as gd


__all__ = ["SimplicialComplex", "boundary", "boundary_columns",
           "ordered_simplices", "simplex_tree_from_point_cloud",
           "simplex_tree_from_simplices"]


def _normalize_simplex(_simplex):
    """
    Valid simplices as tuples of integers.

    :param _simplex: Vertex identifiers for one nonempty simplex.
    :type _simplex: iterable of int
    :return: Canonical simplex.
    :rtype: tuple
    :raises ValueError: If the simplex is empty or repeats a vertex.
    :raises TypeError: If a vertex identifier is not an integer.
    """
    _vertices = tuple(_simplex)
    if not _vertices:
        raise ValueError("Empty simplices implied, should not be supplied")
    if any(not isinstance(_vertex, Integral) for _vertex in _vertices):
        raise TypeError("Vertex identifiers must be integers")

    _vertices = tuple(int(_vertices) for _vertices in _vertices)
    if len(set(_vertices)) != len(_vertices):
        raise ValueError("Vertex identifiers must be unique")
    return tuple(sorted(_vertices))


class SimplicialComplex:
    """
    A nonempty finite abstract simplicial complex with stable vertex IDs.

    Input simplices may be facets and all nonempty faces are included.
    The resulting object has no filtration values or Euclidean embedding.
    A filtered GUDHI simplex tree can be built from it independently at each
    parameter knot.

    :param _simplices: Explicit simplices or facets with integer vertices.
    :type _simplices: iterable of iterable of int
    :raises ValueError: If no simplex is supplied or a simplex is invalid.
    """
    __slots__ = ("_simplices", "_simplex_set")

    def __init__(self, _simplices):
        if hasattr(self, "_simplices"):
            raise AttributeError("SimplicialComplex is immutable")
        _closed_simplices = set()
        for _simplex in _simplices:
            _vertices = _normalize_simplex(_simplex)
            for _size in range(1, len(_vertices) + 1):
                _closed_simplices.update(combinations(_vertices, _size))

        if not _closed_simplices:
            raise ValueError("At least one simplex is required")

        object.__setattr__(self, "_simplex_set", frozenset(_closed_simplices))
        object.__setattr__(
            self, "_simplices",
            tuple(sorted(_closed_simplices,
                         key=lambda _simplex: (len(_simplex), _simplex))),
        )

    def __setattr__(self, _name, _value):
        raise AttributeError("SimplicialComplex is immutable")

    def __delattr__(self, _name):
        raise AttributeError("SimplicialComplex is immutable")

    def __iter__(self):
        """
        Iterate over simplices in dimension-lexicographical order.
        """
        return iter(self._simplices)

    def __len__(self):
        """
        Returns number of non-empty simplices.
        """
        return len(self._simplices)

    def __contains__(self, _simplex):
        """
        Test membership by vertex set, independent of order.
        """
        return _normalize_simplex(_simplex) in self._simplex_set

    def __eq__(self, _other):
        """
        Compare fixed labeled simplex domains.
        """
        if not isinstance(_other, SimplicialComplex):
            return NotImplemented
        return self._simplices == _other._simplices

    def __hash__(self):
        """
        Hash the labeled simplex domains.
        """
        return hash(self._simplices)

    def __repr__(self):
        """
        String representation of simplices.
        """
        return "SimplicialComplex(r!)".format(self.facets)

    @property
    def simplices(self):
        """Return all nonempty simplices in canonical face-first order."""
        return self._simplices

    @property
    def vertices(self):
        """Return the sorted vertex identifiers."""
        return tuple(
            _simplex[0] for _simplex in self._simplices if len(_simplex) == 1)

    @property
    def dimension(self):
        """Return the maximum simplex dimension."""
        return len(self._simplices[-1]) - 1

    @property
    def facets(self):
        """Return maximal simplices in canonical order."""
        return tuple(
            _simplex for _simplex in self._simplices
            if not any(
                len(_coface) > len(_simplex) and set(_simplex).issubset(
                    _coface)
                for _coface in self._simplices
            )
        )

    def faces(self, _simplex):
        """
        Return every non-empty proper face of a simplex in the complex.

        :param _simplex: Simplex to fetch faces from.
        :type _simplex: iterable of int
        :return: Canonically ordered proper faces.
        :rtype: tuple of tuples
        :raises ValueError: If the simplex is outside the complex.
        """
        _vertices = _normalize_simplex(_simplex)
        if _vertices not in self._simplices:
            raise ValueError("Simplex not in the complex.")
        return tuple(
            _face for _face in self._simplices
            if len(_face) < len(_vertices) and set(_face).issubset(_vertices)
        )

    def cofaces(self, _simplex):
        """
        Return every proper coface of a simplex in the complex.

        :param _simplex: Simplex to fetch cofaces from.
        :type _simplex: iterable of int
        :return: Canonically ordered proper cofaces.
        :rtype: tuple of tuples
        :raises ValueError: If the simplex is outside the complex.
        """
        _vertices = _normalize_simplex(_simplex)
        if _vertices not in self._simplex_set:
            raise ValueError("Simplex not in the complex.")
        return tuple(
            _coface for _coface in self._simplices
            if len(_coface) > len(_vertices) and
            set(_coface).issubset(_coface)
        )

    @classmethod
    def from_simplex_tree(cls, _simplex_tree):
        """
        Take the simplex from a GUDHI tree, discarding values.

        :param _simplex_tree: A GUDHI simplex tree, filtered or unfiltered.
        :type _simplex_tree: gudhi.SimplexTree
        :return: Fixed underlying complex.
        :rtype: SimplicialComplex
        """
        return cls(_simplex for _simplex, _value
                   in _simplex_tree.get_simplices())

    def to_simplex_tree(self):
        """
        Return a new GUDHI simplex tree.

        :param _simplex_tree: New tree whose simplex domain is the complex.
        :return: gudhi.SimplexTree
        """
        _simplex_tree = gd.SimplexTree()
        for _simplex in self._simplices:
            _simplex_tree.insert(_simplex, filtration=0.0)
        return _simplex_tree


def simplex_tree_from_simplices(_simplices):
    """
    Build a GUDHI simplex tree from explicit simplices or facets.

    GUDHI automatically inserts all missing faces. Consequently, the input
    may contain every simplex, as in the paper fixtures, or only maximal
    simplices.

    :param _simplices: Explicit simplices or facets.
    :type _simplices: iterable of iterable of int
    :return: Finite simplicial complex with zero construction filtration.
    :rtype: gudhi.SimplexTree
    :raises ValueError: If no nonempty simplex is supplied.
    """
    return SimplicialComplex(_simplices).to_simplex_tree()


def simplex_tree_from_point_cloud(_points, _max_edge_length,
                                  _max_dimension=2):
    """Build a Vietoris--Rips complex from a finite Euclidean point cloud.

    Point rows become GUDHI vertices 0 through n - 1. A finite edge
    threshold and maximum dimension are required to bound the size of the
    complex.

    :param _points: Point cloud with one point per row.
    :type _points: array-like of shape (n_points, ambient_dimension)
    :param _max_edge_length: Largest Rips edge length to include.
    :type _max_edge_length: float
    :param _max_dimension: Largest simplex dimension to construct.
    :type _max_dimension: int
    :return: Vietoris--Rips filtration stored in a simplex tree.
    :rtype: gudhi.SimplexTree
    :raises ValueError: If a construction bound is invalid.
    """

    _dimension_is_integer = isinstance(_max_dimension, Integral)
    if isinstance(_max_dimension, bool) or not _dimension_is_integer:
        raise ValueError("Maximum dimension must be a nonnegative integer")
    if _max_dimension < 0:
        raise ValueError("Maximum dimension must be a nonnegative integer")

    _edge_length = float(_max_edge_length)
    if not math.isfinite(_edge_length) or _edge_length < 0.0:
        raise ValueError("Maximum edge length must be finite and nonnegative")
    if len(_points) == 0:
        raise ValueError("Point cloud must contain at least one point")

    _rips_complex = gd.RipsComplex(points=_points,
                                   max_edge_length=_edge_length)
    return _rips_complex.create_simplex_tree(max_dimension=int(_max_dimension))


def ordered_simplices(_simplex_tree):
    """
    Return simplices in deterministic dimension-lexicographical order.

    :param _simplex_tree: Finite simplicial complex.
    :type _simplex_tree: SimplicialComplex or gudhi.SimplexTree
    :return: Canonically ordered simplices.
    :rtype: tuple of tuple
    """

    if isinstance(_simplex_tree, SimplicialComplex):
        return _simplex_tree.simplices

    _simplices = list(
        tuple(sorted(_simplex))
        for _simplex, _filtration_value in _simplex_tree.get_simplices()
    )
    return tuple(sorted(_simplices,
                        key=lambda _simplex: (len(_simplex), _simplex)))


def boundary(_simplex):
    """
    Return the codimension-one boundary of a simplex over F_2.

    :param _simplex: One nonempty simplex.
    :type _simplex: iterable of int
    :return: Codimension-one faces. A vertex has empty boundary.
    :rtype: tuple of tuple
    """

    _vertices = _normalize_simplex(_simplex)
    if len(_vertices) == 1:
        return ()
    return tuple(combinations(_vertices, len(_vertices) - 1))


def boundary_columns(_simplex_tree, _order=None):
    """
    Encode boundary columns as integer bitsets over F_2.

    :param _simplex_tree: Finite simplicial complex.
    :type _simplex_tree: SimplicialComplex or gudhi.SimplexTree
    :param _order: Optional face-first total order of all simplices.
    :type _order: iterable of iterable of int or None
    :return: Boundary columns aligned with the chosen simplex order.
    :rtype: tuple of int
    :raises ValueError: If the order is not a face-first permutation.
    """

    _complex_simplices = ordered_simplices(_simplex_tree)
    if _order is None:
        _ordered_simplices = _complex_simplices
    else:
        _ordered_simplices = tuple(_normalize_simplex(_simplex)
                                   for _simplex in _order)

    if len(_ordered_simplices) != len(set(_ordered_simplices)):
        raise ValueError("Simplex order contains duplicates")
    if set(_ordered_simplices) != set(_complex_simplices):
        raise ValueError("Order must contain every simplex exactly once")

    _indices = dict(
        (_simplex, _index)
        for _index, _simplex in enumerate(_ordered_simplices)
    )
    _columns = []

    for _column_index, _simplex in enumerate(_ordered_simplices):
        _column = 0
        for _face in boundary(_simplex):
            _row_index = _indices[_face]
            if _row_index >= _column_index:
                raise ValueError("Every face must precede its cofaces")
            _column |= 1 << _row_index
        _columns.append(_column)
    return tuple(_columns)
