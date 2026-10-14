r"""Connected A- and D-vines for a fixed-complex interval family.

Definition 5.1 calls a connected A-canopy an A-vine, and a connected
D-canopy a D-vine.  The finite presentations in ``topology.py`` and
``d_canopy.py`` store all continuation relations, so connected components of
their symbolic cell graphs represent these vines.  Distinct points at a
non-Hausdorff event remain distinct; they can belong to one connected vine.

The D-construction uses *all* retained cells, including diagonal limits and
structural points.  The latter are removed only from local sheet domains,
not from the D-canopy total space or its vine decomposition.
"""

from collections.abc import Mapping
from copy import deepcopy

from .a_canopy import ACanopy
from .a_canopy import a_canopy_fiber
from .contract import MATHEMATICAL_SCOPE
from .contract import OUTPUT_RECORD_VERSION
from .d_canopy import DELTA
from .d_canopy import DCanopy
from .d_canopy import d_canopy_fiber
from .d_canopy import validate_d_canopy
from .topology import CanopyTopology
from .topology import build_canopy_topology
from .topology import validate_canopy_topology
from .utils import _finite_float
from .utils import _nonnegative_integer


__all__ = ["Vine", "VineDecomposition", "build_vine_decomposition"]
__all__ += ["validate_vine_decomposition", "vine_fiber", "vine_diagram"]


def _cell_sort_key(_cell_id):
    """Return the stratum and chosen-sheet order for a symbolic cell."""
    return int(_cell_id[0]), int(_cell_id[1])


def _source_record(_source):
    """Detach and validate the A-topology or complete D-canopy input."""
    if isinstance(_source, ACanopy):
        _source = build_canopy_topology(_source)
    elif isinstance(_source, (CanopyTopology, DCanopy)):
        _source = _source.to_record()
    elif not isinstance(_source, Mapping):
        raise TypeError("Vines require an A-canopy topology or a D-canopy")
    else:
        _source = deepcopy(_source)

    _kind = _source.get("object_type")
    if _kind == "finite A-canopy topology":
        validate_canopy_topology(_source)
        return "A", _source
    if _kind == "D-canopy":
        validate_d_canopy(_source)
        return "D", _source
    raise ValueError("Vines require a complete A-canopy topology or D-canopy")


def _components(_adjacency):
    """Find connected components while preserving every distinct cell ID."""
    _unvisited = set(_adjacency)
    _result = []
    while _unvisited:
        _start = min(_unvisited, key=_cell_sort_key)
        _visited = set()
        _stack = [_start]
        while _stack:
            _cell_id = _stack.pop()
            if _cell_id in _visited:
                continue
            _visited.add(_cell_id)
            _stack.extend(_neighbor for _neighbor in _adjacency[_cell_id]
                          if _neighbor not in _visited)
        _unvisited.difference_update(_visited)
        _result.append(tuple(sorted(_visited, key=_cell_sort_key)))
    return tuple(_result)


def _local_strata(_strata, _index):
    """Choose a small base window: one interval or a point and its sides."""
    _stratum = _strata[_index]
    _ids = {_stratum["id"]}
    if _stratum["kind"] == "point":
        for _neighbor_index in (_index - 1, _index + 1):
            if (0 <= _neighbor_index < len(_strata) and
                    _strata[_neighbor_index]["kind"] == "interval"):
                _ids.add(_strata[_neighbor_index]["id"])
    return _ids


def _local_non_hausdorff(_source, _cell_ids):
    """Find fiber points sharing a local trajectory component in one vine."""
    _strata = tuple(_source["base"]["strata"])
    _by_id = {_cell["cell_id"]: _cell for _cell in _source["cells"]}
    _chosen = set(_cell_ids)
    _events = []
    for _index, _stratum in enumerate(_strata):
        _active_strata = _local_strata(_strata, _index)
        _active = {_cell_id for _cell_id in _chosen
                   if _by_id[_cell_id]["stratum_id"] in _active_strata}
        if not _active:
            continue
        for _component in _components({
                _cell_id: tuple(_neighbor for _neighbor in
                                _source["adjacency"][_cell_id]
                                if _neighbor in _active)
                for _cell_id in _active}):
            _fiber = tuple(_cell_id for _cell_id in _component
                           if _by_id[_cell_id]["stratum_id"] ==
                           _stratum["id"])
            if len(_fiber) < 2:
                continue
            _events.append({
                "stratum_id": _stratum["id"],
                "kind": _stratum["kind"],
                "parameter": float(_stratum["representative_parameter"]),
                "fiber_cell_ids": _fiber,
                "fiber_pairs": tuple(_by_id[_id]["pair"] for _id in _fiber),
            })
    return tuple(_events)


def _vine_records(_kind, _source):
    """Build one exact finite record for each connected total-space part."""
    _cells = {_cell["cell_id"]: _cell for _cell in _source["cells"]}
    _strata = tuple(_source["base"]["strata"])
    _vines = []
    for _vine_id, _cell_ids in enumerate(_components(_source["adjacency"])):
        _member_ids = set(_cell_ids)
        _pieces = []
        for _stratum in _strata:
            _fiber_ids = tuple(_id for _id in _cell_ids if
                               _cells[_id]["stratum_id"] == _stratum["id"])
            if not _fiber_ids:
                continue
            _pieces.append({
                "stratum_id": _stratum["id"],
                "kind": _stratum["kind"],
                "start": float(_stratum["start"]),
                "end": float(_stratum["end"]),
                "left_closed": bool(_stratum["left_closed"]),
                "right_closed": bool(_stratum["right_closed"]),
                "cell_ids": _fiber_ids,
            })
        _relations = tuple(_relation["relation_id"] for _relation in
                           _source["relations"] if
                           _relation["source_cell_id"] in _member_ids and
                           _relation["target_cell_id"] in _member_ids)
        _non_hausdorff = _local_non_hausdorff(_source, _cell_ids)
        _vines.append({
            "vine_id": _vine_id,
            "kind": _kind + "-vine",
            "cell_ids": _cell_ids,
            "relation_ids": _relations,
            "base_pieces": tuple(_pieces),
            "homology_dimensions": tuple(sorted({
                _cells[_id]["homology_dimension"] for _id in _cell_ids})),
            "non_hausdorff_fibers": _non_hausdorff,
            "connected": True,
            "regular": not bool(_non_hausdorff),
        })
    return tuple(_vines)


def build_vine_decomposition(_source):
    """Decompose an A- or D-canopy into connected vines.

    :param _source: A-canopy topology, D-canopy, or A-canopy to build a
        topology from; records of the first two are also accepted.
    :type _source: CanopyTopology, DCanopy, ACanopy or mapping
    :return: Finite decomposition with cells and relation witnesses.
    :rtype: dict
    """
    _kind, _source = _source_record(_source)
    _result = {
        "object_type": "finite vine decomposition",
        "paper_definition": "Definition 5.1 and Proposition 5.2",
        "output_record_version": OUTPUT_RECORD_VERSION,
        "mathematical_scope": MATHEMATICAL_SCOPE,
        "kind": _kind,
        "source": _source,
        "vines": _vine_records(_kind, _source),
        "complete": True,
    }
    _result["vine_count"] = len(_result["vines"])
    _result["regular"] = all(_vine["regular"] for _vine in
                             _result["vines"])
    validate_vine_decomposition(_result)
    return _result


def validate_vine_decomposition(_decomposition):
    """Check the source and the connected-component partition.

    :param _decomposition: Candidate finite vine decomposition.
    :type _decomposition: VineDecomposition or mapping
    :return: The same validated input.
    :rtype: VineDecomposition or mapping
    :raises ValueError: If stored vines disagree with complete continuations.
    """
    if not isinstance(_decomposition, Mapping):
        raise TypeError("Vine decomposition must be a mapping")
    if _decomposition.get("object_type") != "finite vine decomposition":
        raise ValueError("Object is not a finite vine decomposition")
    if not _decomposition.get("complete", False):
        raise ValueError("Vine decomposition requires complete continuations")
    _kind, _source = _source_record(_decomposition["source"])
    if _decomposition["kind"] != _kind:
        raise ValueError("Vine type does not match its A- or D-canopy")
    if tuple(_decomposition["vines"]) != _vine_records(_kind, _source):
        raise ValueError("Vines disagree with connected components of source")
    if _decomposition.get("vine_count") != len(_decomposition["vines"]):
        raise ValueError("Vine count disagrees with connected components")
    if _decomposition.get("regular") != all(
            _vine["regular"] for _vine in _decomposition["vines"]):
        raise ValueError("Regularity disagrees with local vine topology")
    return _decomposition


def vine_fiber(_decomposition, _vine_id, _parameter):
    """Return the chosen-pair points of one vine above a base parameter.

    The points retain distinct identities at non-Hausdorff events. An empty
    tuple means that this vine has no point over the requested parameter.

    :param _decomposition: Complete vine decomposition or its output record.
    :type _decomposition: VineDecomposition or mapping
    :param _vine_id: Component identifier.
    :type _vine_id: int
    :param _parameter: Point of the source canopy's interval base.
    :type _parameter: real
    :return: Pair-labeled points with projection and eta values.
    :rtype: tuple of dict
    """
    validate_vine_decomposition(_decomposition)
    _vine_id = _nonnegative_integer(_vine_id, "Vine identifier")
    if _vine_id >= _decomposition["vine_count"]:
        raise ValueError("Unknown vine identifier")
    _parameter = _finite_float(_parameter, "Base parameter")
    _source = _decomposition["source"]
    if _decomposition["kind"] == "A":
        _points = a_canopy_fiber(_source["canopy"], _parameter)
        _key = "sheet_id"
    else:
        _points = d_canopy_fiber(_source, _parameter)
        _key = "cell_id"
    _ids = set(_decomposition["vines"][_vine_id]["cell_ids"])
    return tuple(dict(_point, vine_id=_vine_id) for _point in _points
                 if _point[_key] in _ids)


def vine_diagram(_decomposition, _vine_id, _parameter,
                 _homology_dimension=None):
    """Return the vine's induced diagram, retaining multiplicity.

    A-vines include diagonal points; D-vines omit ``DELTA``. If a
    homological dimension is selected, return its sorted tuple, otherwise
    return a dimension-indexed dictionary.

    :param _decomposition: Complete vine decomposition.
    :type _decomposition: VineDecomposition or mapping
    :param _vine_id: Connected-component identifier.
    :type _vine_id: int
    :param _parameter: Point of the parameter base.
    :type _parameter: real
    :param _homology_dimension: Optional homology dimension.
    :type _homology_dimension: int or None
    :return: Diagram values of just this vine.
    :rtype: tuple or dict
    """
    if _homology_dimension is not None:
        _homology_dimension = _nonnegative_integer(
            _homology_dimension, "Homology dimension")
    _kind = _decomposition["kind"]
    _grouped = {}
    for _point in vine_fiber(_decomposition, _vine_id, _parameter):
        _value = (_point["eta"] if _kind == "A" else _point["eta_prime"])
        if _value == DELTA:
            continue
        _grouped.setdefault(_point["homology_dimension"], []).append(_value)
    _diagram = {_dimension: tuple(sorted(_values)) for _dimension, _values in
                sorted(_grouped.items())}
    if _homology_dimension is None:
        return _diagram
    return _diagram.get(_homology_dimension, ())


class VineDecomposition(Mapping):
    """Immutable public collection of connected A- or D-vines.

    :param _source: A-canopy topology, D-canopy, or an A-canopy whose
        topology will be built; validated records are accepted as well.
    :type _source: CanopyTopology, DCanopy, ACanopy or mapping
    """

    __slots__ = ("_record",)

    def __init__(self, _source):
        if hasattr(self, "_record"):
            raise AttributeError("VineDecomposition is immutable")
        object.__setattr__(self, "_record", build_vine_decomposition(_source))

    def __setattr__(self, _name, _value):
        raise AttributeError("VineDecomposition is immutable")

    def __delattr__(self, _name):
        raise AttributeError("VineDecomposition is immutable")

    def __getitem__(self, _key):
        """Return one detached finite-record field."""
        return deepcopy(self._record[_key])

    def __iter__(self):
        """Iterate over finite-record field names."""
        return iter(self._record)

    def __len__(self):
        """Return the number of finite-record fields."""
        return len(self._record)

    @property
    def kind(self):
        """Return ``A`` or ``D`` for this decomposition."""
        return self._record["kind"]

    @property
    def vines(self):
        """Return the connected components as individual Vine objects."""
        return tuple(Vine(self, _vine_id) for _vine_id in
                     range(self._record["vine_count"]))

    @property
    def regular(self):
        """Report whether all vine components are Hausdorff."""
        return self._record["regular"]

    def vine(self, _vine_id):
        """Return one connected A- or D-vine by its integer ID."""
        return Vine(self, _vine_id)

    def to_record(self):
        """Return an independent finite-record copy."""
        return deepcopy(self._record)


class Vine(Mapping):
    """One connected part of an A- or D-canopy total space.

    A Vine is obtained through :meth:`VineDecomposition.vine` or its
    ``vines`` property. Its ``fiber`` and ``diagram`` methods return only
    this connected part's pair-labeled points.

    :param _decomposition: Complete collection of connected vines.
    :type _decomposition: VineDecomposition
    :param _vine_id: Component identifier.
    :type _vine_id: int
    """

    __slots__ = ("_decomposition", "_vine_id")

    def __init__(self, _decomposition, _vine_id):
        if hasattr(self, "_vine_id"):
            raise AttributeError("Vine is immutable")
        if not isinstance(_decomposition, VineDecomposition):
            raise TypeError("A Vine requires its VineDecomposition")
        _vine_id = _nonnegative_integer(_vine_id, "Vine identifier")
        if _vine_id >= _decomposition["vine_count"]:
            raise ValueError("Unknown vine identifier")
        object.__setattr__(self, "_decomposition", _decomposition)
        object.__setattr__(self, "_vine_id", _vine_id)

    def __setattr__(self, _name, _value):
        raise AttributeError("Vine is immutable")

    def __delattr__(self, _name):
        raise AttributeError("Vine is immutable")

    def __getitem__(self, _key):
        """Return a detached record field of this vine."""
        return deepcopy(self._decomposition._record["vines"][self._vine_id][
            _key])

    def __iter__(self):
        """Iterate over this vine's finite-record field names."""
        return iter(self._decomposition._record["vines"][self._vine_id])

    def __len__(self):
        """Return the number of stored vine fields."""
        return len(self._decomposition._record["vines"][self._vine_id])

    @property
    def cell_ids(self):
        """Return all symbolic cells in this connected component."""
        return self._decomposition._record["vines"][self._vine_id]["cell_ids"]

    @property
    def regular(self):
        """Report whether the vine has no non-Hausdorff fibers."""
        return self._decomposition._record["vines"][self._vine_id]["regular"]

    def to_record(self):
        """Return an independent record of this connected vine."""
        return deepcopy(self._decomposition._record["vines"][self._vine_id])

    def fiber(self, _parameter):
        """Return just this vine's points above one parameter."""
        return vine_fiber(self._decomposition, self._vine_id, _parameter)

    def diagram(self, _parameter, _homology_dimension=None):
        """Return this vine's augmented or diminished diagram."""
        return vine_diagram(self._decomposition, self._vine_id,
                            _parameter, _homology_dimension)
