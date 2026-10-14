"""Point-cloud input for the fixed-complex, parameterized pipeline.

One :class:`PointCloud` contains one finite array of Euclidean points.
The .npz, .pt and .parquet readers are alternative constructors for that
same object. ``rips_family`` assigns a single labeled simplicial complex to
several clouds and interpolates filtration *values* between samples.
"""

import math
from numbers import Integral
from pathlib import Path

import numpy as np

from .complex import SimplicialComplex
from .filtration import Filtration
from .stratification import Stratification
from .utils import _nonnegative_float
from .utils import _nonnegative_integer
from .utils import _positive_integer


__all__ = ["PointCloud"]


def _path_with_suffix(_path, _suffix):
    """Validate the expected single-file format."""
    _file = Path(_path)
    if _file.suffix.lower() != _suffix:
        raise ValueError("Expected a {} file".format(_suffix))
    return _file


def _ids_from_source(_stored, _supplied):
    """Reconcile identifiers stored in a file with a supplied override."""
    if _stored is not None and _supplied is not None:
        _stored_ids = tuple(_stored)
        _supplied_ids = tuple(_supplied)
        if _stored_ids != _supplied_ids:
            raise ValueError("Supplied vertex IDs differ from stored vertex IDs")
        return _stored_ids
    return _stored if _supplied is None else _supplied


def _checkpoint_value(_checkpoint, _key):
    """Find a literal key or traverse a dotted path through maps and lists."""
    if _key is None:
        return _checkpoint
    if not isinstance(_key, str) or not _key:
        raise ValueError("Checkpoint key must be a nonempty string")
    if hasattr(_checkpoint, "keys") and _key in _checkpoint:
        return _checkpoint[_key]
    _value = _checkpoint
    for _piece in _key.split("."):
        if hasattr(_value, "keys") and _piece in _value:
            _value = _value[_piece]
        elif isinstance(_value, (tuple, list)) and _piece.isdecimal():
            try:
                _value = _value[int(_piece)]
            except IndexError as _error:
                raise KeyError("Checkpoint field '{}' is absent".format(
                    _key)) from _error
        else:
            raise KeyError("Checkpoint field '{}' is absent".format(_key))
    return _value


class PointCloud:
    """One immutable finite point cloud with stable integer vertex labels.

    Rows are points and columns are Euclidean coordinates. The input is
    copied; the ``points`` property returns another copy. If vertex IDs are
    omitted, row positions become IDs ``0, 1, ...``. In a family, those
    inferred IDs require an explicit decision to trust row correspondence.

    :param _points: Two-dimensional numeric array, one point per row.
    :type _points: array-like
    :param _vertex_ids: Optional unique integer IDs in row order.
    :type _vertex_ids: iterable of int or None
    """

    __slots__ = ("_points", "_vertex_ids", "_ids_explicit")

    def __init__(self, _points, _vertex_ids=None):
        try:
            _array = np.asarray(_points, dtype=float)
        except (TypeError, ValueError, OverflowError) as _error:
            raise ValueError("Point coordinates must be numeric") from _error
        if _array.ndim != 2 or min(_array.shape) == 0:
            raise ValueError("Points must be a nonempty rectangular 2D array")
        if not np.isfinite(_array).all():
            raise ValueError("All point coordinates must be finite")
        if _vertex_ids is None:
            _ids = tuple(range(_array.shape[0]))
        else:
            try:
                _raw_ids = tuple(_vertex_ids)
            except TypeError as _error:
                raise ValueError("Vertex IDs must be iterable") from _error
            if len(_raw_ids) != _array.shape[0]:
                raise ValueError("One vertex ID is required per point")
            if any(isinstance(_id, (bool, np.bool_)) or
                   not isinstance(_id, Integral) for _id in _raw_ids):
                raise TypeError("Vertex IDs must be integers, not booleans")
            _ids = tuple(int(_id) for _id in _raw_ids)
            if len(set(_ids)) != len(_ids):
                raise ValueError("Vertex IDs must be unique")
        _copy = np.array(_array, dtype=float, copy=True)
        _copy.flags.writeable = False
        object.__setattr__(self, "_points", _copy)
        object.__setattr__(self, "_vertex_ids", _ids)
        object.__setattr__(self, "_ids_explicit", _vertex_ids is not None)

    def __setattr__(self, _name, _value):
        raise AttributeError("PointCloud is immutable")

    def __delattr__(self, _name):
        raise AttributeError("PointCloud is immutable")

    @property
    def points(self):
        """Return a writable copy of the point matrix in row order."""
        return self._points.copy()

    @property
    def vertex_ids(self):
        """Return vertex identifiers aligned with rows of ``points``."""
        return self._vertex_ids

    @property
    def ids_explicit(self):
        """Report whether identifiers were provided or stored in the file."""
        return self._ids_explicit

    @property
    def ambient_dimension(self):
        """Return the number of coordinate columns."""
        return self._points.shape[1]

    def __len__(self):
        """Return the number of vertices."""
        return len(self._vertex_ids)

    @classmethod
    def from_file(cls, _path, **_options):
        """Load .npz, .pt or .parquet based on the path suffix.

        Format-specific keyword options are forwarded to the corresponding
        ``from_npz``, ``from_pt`` or ``from_parquet`` constructor.

        :param _path: File in one of the three supported formats.
        :type _path: str or pathlib.Path
        :param _options: Reader-specific keyword arguments.
        :type _options: dict
        :return: A validated, immutable point cloud.
        :rtype: PointCloud
        """
        _suffix = Path(_path).suffix.lower()
        _readers = {".npz": cls.from_npz, ".pt": cls.from_pt,
                    ".parquet": cls.from_parquet}
        if _suffix not in _readers:
            raise ValueError("Supported point-cloud files: .npz, .pt, .parquet")
        return _readers[_suffix](_path, **_options)

    @classmethod
    def from_npz(cls, _path, _points_key="points", _vertex_ids_key="vertex_ids",
                 _vertex_ids=None):
        """Read a numeric point matrix and optional vertex IDs from .npz.

        NumPy pickled arrays are never loaded.

        :param _path: Path to the .npz archive.
        :type _path: str or pathlib.Path
        :param _points_key: Key containing the two-dimensional point matrix.
        :type _points_key: str
        :param _vertex_ids_key: Optional key containing integer row labels.
        :type _vertex_ids_key: str or None
        :param _vertex_ids: Optional supplied row labels.
        :type _vertex_ids: iterable of int or None
        :return: A validated point cloud.
        :rtype: PointCloud
        """
        _file = _path_with_suffix(_path, ".npz")
        with np.load(_file, allow_pickle=False) as _archive:
            if _points_key not in _archive.files:
                raise KeyError("Point array '{}' is absent".format(_points_key))
            _points = _archive[_points_key]
            _stored = (_archive[_vertex_ids_key]
                       if _vertex_ids_key in _archive.files else None)
            _ids = _ids_from_source(_stored, _vertex_ids)
            return cls(_points, _ids)

    @classmethod
    def from_pt(cls, _path, _points_key="points", _vertex_ids_key="vertex_ids",
                _vertex_ids=None):
        """Read a numeric 2D tensor from a PyTorch .pt checkpoint.

        ``_points_key`` may be an exact key or a dotted path through nested
        mappings and lists, e.g. ``hidden_layers.0.weight``. Pass None when
        the checkpoint itself is a tensor. Loading is restricted to weights
        and CPU data; arbitrary pickled Python objects are unsupported.

        :param _path: Path to the .pt checkpoint.
        :type _path: str or pathlib.Path
        :param _points_key: Literal key, dotted path, or None for a bare tensor.
        :type _points_key: str or None
        :param _vertex_ids_key: Optional top-level integer-label field.
        :type _vertex_ids_key: str or None
        :param _vertex_ids: Optional supplied row labels.
        :type _vertex_ids: iterable of int or None
        :return: A validated point cloud.
        :rtype: PointCloud
        """
        _file = _path_with_suffix(_path, ".pt")
        try:
            import torch  # pylint: disable=import-outside-toplevel,import-error
        except ImportError as _error:
            raise ImportError("Loading .pt requires PyTorch") from _error
        _checkpoint = torch.load(_file, map_location="cpu", weights_only=True)
        _tensor = _checkpoint_value(_checkpoint, _points_key)
        if not torch.is_tensor(_tensor):
            raise TypeError("Checkpoint field '{}' is not a tensor".format(
                _points_key))
        _stored = None
        if _vertex_ids_key is not None and hasattr(_checkpoint, "keys") and \
                _vertex_ids_key in _checkpoint:
            _stored = _checkpoint[_vertex_ids_key]
            if torch.is_tensor(_stored):
                _stored = _stored.detach().cpu().numpy()
        _ids = _ids_from_source(_stored, _vertex_ids)
        return cls(_tensor.detach().cpu().numpy(), _ids)

    @classmethod
    def from_parquet(cls, _path, _coordinate_columns=None,
                     _vertex_ids_key="vertex_id", _vertex_ids=None):
        """Read a wide Parquet table: one point per row, one column per axis.

        By default all columns except ``vertex_id`` are coordinates. Supply
        ``_coordinate_columns`` when the table also contains metadata fields.
        ``pandas`` and a Parquet engine (usually ``pyarrow``) are optional
        dependencies required only by this constructor.

        :param _path: Path to the .parquet table.
        :type _path: str or pathlib.Path
        :param _coordinate_columns: Columns defining one Euclidean point.
        :type _coordinate_columns: iterable of str or None
        :param _vertex_ids_key: Optional integer-label column.
        :type _vertex_ids_key: str or None
        :param _vertex_ids: Optional supplied row labels.
        :type _vertex_ids: iterable of int or None
        :return: A validated point cloud.
        :rtype: PointCloud
        """
        _file = _path_with_suffix(_path, ".parquet")
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel,import-error
        except ImportError as _error:
            raise ImportError("Loading .parquet requires pandas and a Parquet engine") from _error
        _frame = pd.read_parquet(_file)
        _columns = (tuple(_frame.columns)
                    if _coordinate_columns is None else
                    tuple(_coordinate_columns))
        if _coordinate_columns is None and _vertex_ids_key is not None:
            _columns = tuple(_column for _column in _columns
                             if _column != _vertex_ids_key)
        if not _columns or len(set(_columns)) != len(_columns):
            raise ValueError("Specify one or more distinct coordinate columns")
        if _vertex_ids_key in _columns:
            raise ValueError("Vertex ID column cannot be a coordinate")
        _missing = set(_columns) - set(_frame.columns)
        if _missing:
            raise KeyError("Coordinate columns are absent: {}".format(
                sorted(_missing)))
        _stored = (_frame[_vertex_ids_key].to_numpy()
                   if _vertex_ids_key is not None and
                   _vertex_ids_key in _frame else None)
        _ids = _ids_from_source(_stored, _vertex_ids)
        return cls(_frame.loc[:, list(_columns)].to_numpy(), _ids)

    def rips_complex(self, _max_edge_length, _max_dimension=2,
                     _max_simplices=20000):
        """Return a bounded Rips clique complex using the stored vertex IDs.

        :param _max_edge_length: Inclusive edge length bound.
        :type _max_edge_length: nonnegative float
        :param _max_dimension: Highest simplex dimension to construct.
        :type _max_dimension: nonnegative int
        :param _max_simplices: Stop before adding more than this many simplices.
        :type _max_simplices: positive int
        :return: Unfiltered fixed labeled complex.
        :rtype: SimplicialComplex
        """
        _threshold = _nonnegative_float(_max_edge_length, "Maximum edge length")
        _dimension = _nonnegative_integer(_max_dimension, "Maximum dimension")
        _limit = _positive_integer(_max_simplices, "Maximum simplices")
        _ids = sorted(self._vertex_ids)
        if len(_ids) > _limit:
            raise ValueError("Rips complex exceeds the simplex limit")
        _indices = {_id: _index for _index, _id in
                    enumerate(self._vertex_ids)}
        _neighbors = {_id: set() for _id in _ids}
        for _offset, _first in enumerate(_ids):
            for _second in _ids[_offset + 1:]:
                if math.dist(self._points[_indices[_first]],
                             self._points[_indices[_second]]) <= _threshold:
                    _neighbors[_first].add(_second)

        _simplices = []

        def _extend(_prefix, _candidates):
            """Enumerate increasing cliques, stopping at the given cap."""
            for _offset, _vertex in enumerate(_candidates):
                _simplex = _prefix + (_vertex,)
                _simplices.append(_simplex)
                if len(_simplices) > _limit:
                    raise ValueError("Rips complex exceeds the simplex limit")
                if len(_simplex) < _dimension + 1:
                    _next = tuple(_candidate for _candidate in
                                  _candidates[_offset + 1:]
                                  if _candidate in _neighbors[_vertex])
                    _extend(_simplex, _next)

        if _dimension:
            _extend((), tuple(_ids))
        else:
            _simplices = [(_id,) for _id in _ids]
        return SimplicialComplex(_simplices)

    def rips_filtration(self, _complex=None, _max_edge_length=None,
                        _max_dimension=2, _max_simplices=20000):
        """Evaluate Rips edge lengths on a chosen fixed simplex domain.

        If ``_complex`` is given, evaluate every one of its simplices even if
        some edge lengths exceed the bound used to construct that complex.
        Otherwise ``_max_edge_length`` is required to build the domain.

        :param _complex: Shared fixed simplex domain, if already selected.
        :type _complex: SimplicialComplex or None
        :param _max_edge_length: Rips bound when constructing a new domain.
        :type _max_edge_length: nonnegative float or None
        :param _max_dimension: Maximum dimension for a new domain.
        :type _max_dimension: nonnegative int
        :param _max_simplices: Maximum number of simplices in a new domain.
        :type _max_simplices: positive int
        :return: Rips values on the fixed labeled complex.
        :rtype: Filtration
        """
        if _complex is None:
            if _max_edge_length is None:
                raise ValueError("Supply a complex or a maximum edge length")
            _complex = self.rips_complex(_max_edge_length, _max_dimension,
                                         _max_simplices)
        if not isinstance(_complex, SimplicialComplex):
            raise TypeError("Expected a SimplicialComplex")
        if set(_complex.vertices) != set(self._vertex_ids):
            raise ValueError("Complex vertices must equal point-cloud vertex IDs")
        _lookup = dict(zip(self._vertex_ids, self._points))
        _edge_lengths = {
            (_first, _second): math.dist(_lookup[_first], _lookup[_second])
            for _first, _second in
            ((_simplex[0], _simplex[1]) for _simplex in _complex.simplices
             if len(_simplex) == 2)
        }
        _values = {}
        for _simplex in _complex.simplices:
            if len(_simplex) == 1:
                _values[_simplex] = 0.0
            elif len(_simplex) == 2:
                _values[_simplex] = _edge_lengths[_simplex]
            else:
                _values[_simplex] = max(
                    _edge_lengths[(_simplex[_first], _simplex[_second])]
                    for _first in range(len(_simplex))
                    for _second in range(_first + 1, len(_simplex)))
            if not math.isfinite(_values[_simplex]):
                raise ValueError("Point distances must be finite")
        return Filtration(_complex, _values)

    @staticmethod
    def rips_family(_clouds, _parameters=None, _complex=None,
                    _max_edge_length=None, _max_dimension=2,
                    _max_simplices=20000, _interpolation="filtration_values",
                    _assume_row_order=False):
        """Return the paper's fixed-complex filtration family from clouds.

        Each cloud must use the same vertex IDs and ambient dimension.
        Without explicit IDs, set ``_assume_row_order=True`` to assert that
        rows refer to the same vertices in every cloud. With no ``_complex``,
        build the union of bounded Rips complexes across samples. The union
        is used at *all* parameters; absent edges at one sample still receive
        their actual Rips values there. Filtration values, not coordinates,
        are linearly interpolated between samples.

        :param _clouds: Two or more labeled point clouds.
        :type _clouds: iterable of PointCloud
        :param _parameters: Increasing parameter values, one per cloud.
        :type _parameters: iterable of real or None
        :param _complex: Optional shared labeled simplex domain.
        :type _complex: SimplicialComplex or None
        :param _max_edge_length: Edge threshold for the union domain.
        :type _max_edge_length: float or None
        :param _max_dimension: Maximum dimension for each bounded domain.
        :type _max_dimension: nonnegative int
        :param _max_simplices: Maximum simplices in each and union domains.
        :type _max_simplices: positive int
        :param _interpolation: Must be ``filtration_values``.
        :type _interpolation: str
        :param _assume_row_order: Explicitly accept inferred IDs as matches.
        :type _assume_row_order: bool
        :return: Piecewise-linear family on one fixed complex.
        :rtype: Stratification
        """
        _samples = tuple(_clouds)
        if len(_samples) < 2 or any(not isinstance(_cloud, PointCloud)
                                    for _cloud in _samples):
            raise ValueError("Supply at least two PointCloud objects")
        if _interpolation != "filtration_values":
            raise ValueError("Only filtration_values interpolation is supported")
        if not isinstance(_assume_row_order, bool):
            raise TypeError("Assume row order must be a boolean")
        if (not _assume_row_order and
                any(not _cloud.ids_explicit for _cloud in _samples)):
            raise ValueError("Supply vertex IDs or explicitly trust row order")
        _ids = set(_samples[0].vertex_ids)
        _dimension = _samples[0].ambient_dimension
        if any(set(_cloud.vertex_ids) != _ids or
               _cloud.ambient_dimension != _dimension
               for _cloud in _samples[1:]):
            raise ValueError("Clouds need identical IDs and ambient dimension")
        if _complex is None:
            if _max_edge_length is None:
                raise ValueError("Supply a fixed complex or an edge threshold")
            _domains = tuple(_cloud.rips_complex(
                _max_edge_length, _max_dimension, _max_simplices)
                for _cloud in _samples)
            _complex = SimplicialComplex(
                _simplex for _domain in _domains
                for _simplex in _domain.simplices)
            if len(_complex) > _max_simplices:
                raise ValueError("Union Rips complex exceeds the simplex limit")
        _filtrations = tuple(_cloud.rips_filtration(_complex)
                             for _cloud in _samples)
        return Stratification(_filtrations, _parameters)
