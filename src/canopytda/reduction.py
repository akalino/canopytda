"""Boundary-matrix reduction over the field with two elements.

Columns follow a face-first total simplex order. A nonzero reduced column
with lowest nonzero row i pairs the simplex in row i with that column.
The original and reduced columns are integer bitsets indexed by that order.
"""

from types import MappingProxyType

from .complex import SimplicialComplex
from .complex import boundary_columns
from .utils import _normalize_simplex


__all__ = ["BoundaryReduction", "reduce_boundary_matrix"]


class BoundaryReduction:
    """Reduce the boundary matrix of one finite complex over F_2.

    This object does not choose a filtration order. Supply a face-first
    total order, or use the complex's dimension-lexicographic order.
    For persistence, :class:`canopytda.pairing.PersistencePairing` first
    validates that the order also respects filtration values.

    :param _complex: Fixed labeled simplicial complex.
    :type _complex: canopytda.complex.SimplicialComplex
    :param _order: Face-first permutation of all simplices, or None.
    :type _order: iterable of simplices or None
    :raises TypeError: If the complex has the wrong type.
    :raises ValueError: If the order omits, repeats, or misorders simplices.
    """
    # pylint: disable=no-member

    __slots__ = ("_complex", "_order", "_indices", "_boundary_columns",
                 "_reduced_columns", "_change_columns", "_pivot_pairs",
                 "_essential_simplices")

    def __init__(self, _complex, _order=None):
        if hasattr(self, "_complex"):
            raise AttributeError("BoundaryReduction is immutable")
        if not isinstance(_complex, SimplicialComplex):
            raise TypeError("Expected a SimplicialComplex")
        _ordered = (
            _complex.simplices if _order is None else
            tuple(_normalize_simplex(_simplex) for _simplex in _order)
        )
        _boundary = boundary_columns(_complex, _order=_ordered)
        _reduced = []
        _changes = []
        _pivot_columns = {}

        for _index, _initial in enumerate(_boundary):
            _column = _initial
            _change = 1 << _index
            while _column:
                _low = _column.bit_length() - 1
                _earlier = _pivot_columns.get(_low)
                if _earlier is None:
                    break
                _column ^= _reduced[_earlier]
                _change ^= _changes[_earlier]
            if _column:
                _pivot_columns[_column.bit_length() - 1] = _index
            _reduced.append(_column)
            _changes.append(_change)

        _finite = tuple(
            (_ordered[_column.bit_length() - 1], _ordered[_index])
            for _index, _column in enumerate(_reduced) if _column
        )
        _paired_rows = frozenset(_pivot_columns)
        _essential = tuple(
            _ordered[_index]
            for _index, _column in enumerate(_reduced)
            if not _column and _index not in _paired_rows
        )
        if 2 * len(_finite) + len(_essential) != len(_ordered):
            raise RuntimeError("Boundary reduction did not pair each simplex")

        object.__setattr__(self, "_complex", _complex)
        object.__setattr__(self, "_order", _ordered)
        object.__setattr__(self, "_indices", MappingProxyType({
            _simplex: _index for _index, _simplex in enumerate(_ordered)
        }))
        object.__setattr__(self, "_boundary_columns", _boundary)
        object.__setattr__(self, "_reduced_columns", tuple(_reduced))
        object.__setattr__(self, "_change_columns", tuple(_changes))
        object.__setattr__(self, "_pivot_pairs", _finite)
        object.__setattr__(self, "_essential_simplices", _essential)

    def __setattr__(self, _name, _value):
        raise AttributeError("BoundaryReduction is immutable")

    def __delattr__(self, _name):
        raise AttributeError("BoundaryReduction is immutable")

    @property
    def complex(self):
        """Return the underlying fixed simplicial complex."""
        return self._complex

    @property
    def order(self):
        """Return the complete total order of simplex tuples."""
        return self._order

    @property
    def boundary_columns(self):
        """Return original boundary columns as integer bitsets."""
        return self._boundary_columns

    @property
    def reduced_columns(self):
        """Return reduced boundary columns as integer bitsets."""
        return self._reduced_columns

    @property
    def change_columns(self):
        """Return columns of V such that D times V equals reduced D."""
        return self._change_columns

    @property
    def pivot_pairs(self):
        """Return finite (birth simplex, death simplex) pivot pairs."""
        return self._pivot_pairs

    @property
    def essential_simplices(self):
        """Return zero-column births never used as a pivot row."""
        return self._essential_simplices

    def cycle_representative(self, _simplex):
        """Return a cycle chain for a zero reduced column.

        :param _simplex: Simplex that gives birth to a homology class.
        :type _simplex: iterable of int
        :return: Simplex tuples whose sum has zero boundary over F_2.
        :rtype: tuple of tuple
        :raises KeyError: If the simplex is outside this complex.
        :raises ValueError: If the simplex is a death simplex.
        """
        _index = self._indices[_normalize_simplex(_simplex)]
        if self._reduced_columns[_index]:
            raise ValueError("Simplex is not a birth simplex")
        _chain = self._change_columns[_index]
        return tuple(
            _simplex for _position, _simplex in enumerate(self._order)
            if _chain & (1 << _position)
        )


def reduce_boundary_matrix(_complex, _order=None):
    """Return the F_2 boundary reduction of a fixed complex.

    :param _complex: Fixed labeled simplicial complex.
    :type _complex: canopytda.complex.SimplicialComplex
    :param _order: Optional face-first total order.
    :type _order: iterable of simplices or None
    :return: Reduced columns, pivot pairs, and essential births.
    :rtype: BoundaryReduction
    """
    return BoundaryReduction(_complex, _order)
