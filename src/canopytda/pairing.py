"""Persistence simplex pairings and diagrams for one fixed filtration.

The chosen total consistent order determines simplex-level partners.
Augmented diagrams retain zero-length intervals and their multiplicity;
diminished diagrams omit them. The paper's star symbol is represented by
``None`` for a class that persists in the full complex.
"""

import math
from types import MappingProxyType

import gudhi as gd  # pylint: disable=import-error

from .filtration import Filtration
from .orders import canonical_refinement
from .orders import validate_refinement
from .reduction import BoundaryReduction
from .utils import _nonnegative_integer
from .utils import _normalize_simplex


STAR = None

__all__ = ["STAR", "PersistencePairing", "compute_pairing",
           "augmented_diagram", "diminished_diagram"]


class PersistencePairing:
    """Compute the chosen simplex pairing for one filtered complex over F_2.

    The default is the canonical value, dimension, vertex-label order.
    A supplied order must refine the filtration and put every face before
    its cofaces. Every simplex occurs in exactly one pair; ``STAR`` denotes
    the endpoint of an essential class.

    :param _filtration: Fixed filtration or filtered GUDHI simplex tree.
    :type _filtration: Filtration or gudhi.SimplexTree
    :param _order: Optional consistent total order of all simplices.
    :type _order: iterable of simplices or None
    :raises TypeError: If the filtration has an unsupported type.
    :raises ValueError: If the order is incomplete or inconsistent.
    """
    # pylint: disable=no-member

    __slots__ = ("_filtration", "_order", "_reduction", "_pairs",
                 "_finite_pairs", "_essential_pairs", "_partners")

    def __init__(self, _filtration, _order=None):
        if hasattr(self, "_filtration"):
            raise AttributeError("PersistencePairing is immutable")
        if isinstance(_filtration, gd.SimplexTree):
            _filtration = Filtration.from_simplex_tree(_filtration)
        if not isinstance(_filtration, Filtration):
            raise TypeError("Expected a Filtration or gudhi.SimplexTree")

        _ordered = (
            canonical_refinement(_filtration) if _order is None else
            validate_refinement(_filtration, _order)
        )
        _reduction = BoundaryReduction(_filtration.complex, _ordered)
        _finite_by_birth = dict(_reduction.pivot_pairs)
        _essential = frozenset(_reduction.essential_simplices)
        _pairs = tuple(
            (_simplex, _finite_by_birth[_simplex]
             if _simplex in _finite_by_birth else STAR)
            for _simplex in _ordered
            if _simplex in _finite_by_birth or _simplex in _essential
        )
        _finite_pairs = tuple(
            _pair for _pair in _pairs if _pair[1] is not STAR
        )
        _essential_pairs = tuple(
            _pair for _pair in _pairs if _pair[1] is STAR
        )
        _partners = {}
        for _birth, _death in _pairs:
            _partners[_birth] = _death
            if _death is not STAR:
                _partners[_death] = _birth
        if len(_partners) != len(_ordered):
            raise RuntimeError("A simplex is missing from the pairing")

        object.__setattr__(self, "_filtration", _filtration)
        object.__setattr__(self, "_order", _ordered)
        object.__setattr__(self, "_reduction", _reduction)
        object.__setattr__(self, "_pairs", _pairs)
        object.__setattr__(self, "_finite_pairs", _finite_pairs)
        object.__setattr__(self, "_essential_pairs", _essential_pairs)
        object.__setattr__(self, "_partners", MappingProxyType(_partners))

    def __setattr__(self, _name, _value):
        raise AttributeError("PersistencePairing is immutable")

    def __delattr__(self, _name):
        raise AttributeError("PersistencePairing is immutable")

    def __iter__(self):
        """Iterate over all (birth, death) pairs in birth-simplex order."""
        return iter(self._pairs)

    def __len__(self):
        """Return the number of pairs, counting essential births."""
        return len(self._pairs)

    @property
    def filtration(self):
        """Return the validated, immutable filtration."""
        return self._filtration

    @property
    def order(self):
        """Return the chosen total consistent simplex order."""
        return self._order

    @property
    def reduction(self):
        """Return the underlying boundary-matrix reduction."""
        return self._reduction

    @property
    def pairs(self):
        """Return all pairs, with ``STAR`` for essential classes."""
        return self._pairs

    @property
    def finite_pairs(self):
        """Return pairs with a simplex as their death endpoint."""
        return self._finite_pairs

    @property
    def essential_pairs(self):
        """Return pairs whose death endpoint is ``STAR``."""
        return self._essential_pairs

    @property
    def betti_numbers(self):
        """Return Betti numbers of the full complex, starting with H_0."""
        return tuple(
            sum(len(_birth) == _dimension + 1
                for _birth, _death in self._essential_pairs)
            for _dimension in range(self._filtration.complex.dimension + 1)
        )

    def partner(self, _simplex):
        """Return the unique paired simplex, or ``STAR`` if essential.

        :param _simplex: Birth or death simplex in the complex.
        :type _simplex: iterable of int
        :return: Its partner, or ``STAR`` for an essential birth.
        :rtype: tuple of int or None
        :raises KeyError: If the simplex is outside the complex.
        """
        return self._partners[_normalize_simplex(_simplex)]

    def augmented_diagram(self, _dimension=None):
        """Return a sorted multiset of (birth, death) values.

        Diagonal pairs are retained. Essential deaths have value ``inf``.
        Duplicate points remain repeated tuples. Dimension is determined
        by the birth simplex.

        :param _dimension: Optional homology dimension to select.
        :type _dimension: int or None
        :return: Augmented diagram, sorted by birth and death values.
        :rtype: tuple of tuple of float
        """
        if _dimension is not None:
            _dimension = _nonnegative_integer(_dimension, "Dimension")
        return tuple(sorted(
            (self._filtration.value(_birth),
             math.inf if _death is STAR else self._filtration.value(_death))
            for _birth, _death in self._pairs
            if _dimension is None or len(_birth) == _dimension + 1
        ))

    def diminished_diagram(self, _dimension=None):
        """Return off-diagonal values, retaining essential classes.

        :param _dimension: Optional homology dimension to select.
        :type _dimension: int or None
        :return: Augmented diagram with exactly equal endpoints removed.
        :rtype: tuple of tuple of float
        """
        return tuple(
            _point for _point in self.augmented_diagram(_dimension)
            if _point[0] != _point[1]
        )


def compute_pairing(_filtration, _order=None):
    """Return the simplex pairing for one filtration and total order.

    :param _filtration: Fixed filtration or filtered GUDHI simplex tree.
    :type _filtration: Filtration or gudhi.SimplexTree
    :param _order: Optional consistent total simplex order.
    :type _order: iterable of simplices or None
    :return: Computed simplex pairing and diagrams.
    :rtype: PersistencePairing
    """
    return PersistencePairing(_filtration, _order)


def augmented_diagram(_filtration, _order=None, _dimension=None):
    """Compute an augmented diagram including diagonal points.

    :param _filtration: Fixed filtration or filtered GUDHI simplex tree.
    :type _filtration: Filtration or gudhi.SimplexTree
    :param _order: Optional consistent total simplex order.
    :type _order: iterable of simplices or None
    :param _dimension: Optional homology degree.
    :type _dimension: int or None
    :return: Sorted (birth, death) values, with multiplicity.
    :rtype: tuple of tuple of float
    """
    return PersistencePairing(_filtration, _order).augmented_diagram(_dimension)


def diminished_diagram(_filtration, _order=None, _dimension=None):
    """Compute a diagram with all diagonal points omitted.

    :param _filtration: Fixed filtration or filtered GUDHI simplex tree.
    :type _filtration: Filtration or gudhi.SimplexTree
    :param _order: Optional consistent total simplex order.
    :type _order: iterable of simplices or None
    :param _dimension: Optional homology degree.
    :type _dimension: int or None
    :return: Sorted off-diagonal (birth, death) values.
    :rtype: tuple of tuple of float
    """
    return PersistencePairing(_filtration, _order).diminished_diagram(_dimension)
