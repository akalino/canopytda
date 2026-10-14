r"""Finite diminished canopies obtained from an A-canopy.

The paper constructs the D-canopy by taking the closure, in the A-canopy
topology, of the locus whose persistence coordinates are strictly
off-diagonal.  This module implements that construction for the package's
fixed-complex, one-parameter, piecewise-linear model.

An A-canopy order stratum need not split when a paired face and coface have
equal filtration value: their face relation already fixes their relative
order.  Equality of those two values is nevertheless decisive for a
D-canopy.  The implementation therefore refines the A-canopy presentation at
the sampled parameters and at every relevant coordinate equality before computing
the closure.  This is a D-specific refinement of the finite presentation,
not an additional assumption on the paper's construction.

The symbol :data:`DELTA` represents the single quotient point obtained by
collapsing the diagonal in the target space.  It is retained in D-canopy
fibers when it is a boundary point of the off-diagonal locus, but it is
removed by :func:`d_canopy_diagram` as required by Definition 4.2.
"""

from bisect import bisect_right
from collections.abc import Mapping
from copy import deepcopy
from itertools import combinations
import math

from .a_canopy import ACanopy
from .a_canopy import validate_a_canopy
from .contract import MATHEMATICAL_SCOPE
from .contract import OUTPUT_RECORD_VERSION
from .contract import TIE_BREAK_RULE
from .pairing import STAR
from .topology import build_canopy_topology
from .topology import CanopyTopology
from .topology import validate_canopy_topology
from .utils import _finite_float
from .utils import _nonnegative_float
from .utils import _nonnegative_integer
from .utils import _positive_integer


DELTA = "delta"

__all__ = ["DELTA", "DCanopy", "build_d_canopy", "d_canopy_boundary_cells"]
__all__ += ["d_canopy_diagram", "d_canopy_fiber"]
__all__ += ["d_canopy_sheet_domains", "validate_d_canopy"]


def _cell_sort_key(_cell_id):
    """Return a deterministic key for one refined cell identifier."""

    return int(_cell_id[0]), int(_cell_id[1])


def _absolute_tolerance(_canopy, _atol):
    """Return the tolerance used for diagonal and equality decisions."""

    if _atol is None:
        _atol = _canopy["stratification"].get("tie_tolerance", 1e-12)
    return _nonnegative_float(_atol, "Absolute tolerance")


def _parameter_tolerance(_domain):
    """Return a scale-aware tolerance for base parameters."""

    _start, _end = tuple(float(_value) for _value in _domain)
    _scale = max(1.0, abs(_start), abs(_end), abs(_end - _start))
    return 1e-12 * _scale


def _filtration_model(_canopy):
    """Return the piecewise-linear filtration data indexed by simplex."""

    _stratification = _canopy["stratification"]
    try:
        _parameters = tuple(
            float(_value)
            for _value in _stratification["parameters"]
        )
        _simplices = tuple(_stratification["complex_simplices"])
        _rows = tuple(
            tuple(float(_value) for _value in _row)
            for _row in _stratification["sample_filtration_values"]
        )
    except (KeyError, TypeError, ValueError) as _error:
        raise ValueError(
            "A-canopy stratification lacks a piecewise-linear value model"
        ) from _error
    if len(_parameters) != len(_rows):
        raise ValueError("Sample parameters and value rows have different lengths")
    if not _parameters or any(len(_row) != len(_simplices) for _row in _rows):
        raise ValueError("Filtration value rows do not cover the simplices of K")
    _model = {}
    _model["parameters"] = _parameters
    _model["simplices"] = _simplices
    _model["rows"] = _rows
    _model["simplex_indices"] = dict(
        (_simplex, _index)
        for _index, _simplex in enumerate(_simplices)
    )
    return _model


def _value_at(_model, _simplex, _parameter):
    """Evaluate one simplex filtration value at a base parameter."""

    try:
        _simplex_index = _model["simplex_indices"][_simplex]
    except KeyError as _error:
        raise ValueError("Chosen pair uses a simplex outside K") from _error
    _parameters = _model["parameters"]
    _rows = _model["rows"]
    if len(_parameters) == 1 or _parameter <= _parameters[0]:
        return _rows[0][_simplex_index]
    if _parameter >= _parameters[-1]:
        return _rows[-1][_simplex_index]
    _segment = bisect_right(_parameters, _parameter) - 1
    _left = _parameters[_segment]
    _right = _parameters[_segment + 1]
    _fraction = (_parameter - _left) / (_right - _left)
    return (
        (1.0 - _fraction) * _rows[_segment][_simplex_index]
        + _fraction * _rows[_segment + 1][_simplex_index]
    )


def _eta_at(_model, _pair, _parameter, _atol):
    """Return augmented coordinates and whether they are off-diagonal."""

    _birth, _death = _pair
    _birth_value = float(_value_at(_model, _birth, _parameter))
    if _death == STAR:
        return (_birth_value, math.inf), True
    _death_value = float(_value_at(_model, _death, _parameter))
    _gap = _death_value - _birth_value
    if _gap < -_atol:
        raise RuntimeError("A chosen pair has death strictly before birth")
    return (_birth_value, _death_value), _gap > _atol


def _used_coordinate_simplices(_topology):
    """Return simplices occurring as finite persistence coordinates."""

    _simplices = set()
    for _cell in _topology["cells"]:
        _birth, _death = _cell["pair"]
        _simplices.add(_birth)
        if _death != STAR:
            _simplices.add(_death)
    return tuple(sorted(_simplices, key=repr))


def _unique_parameters(_values, _tolerance):
    """Sort parameters and merge numerical duplicates."""

    _unique = []
    for _value in sorted(float(_item) for _item in _values):
        if not _unique or abs(_value - _unique[-1]) > _tolerance:
            _unique.append(_value)
    return tuple(_unique)


def _refinement_breakpoints(_canopy, _topology, _model, _atol, _max_refinement_events):
    """Return a complete PL refinement for D-coordinates and multiplicity."""

    _parameters = _model["parameters"]
    _domain = tuple(float(_value) for _value in _canopy["base"]["domain"])
    _parameter_atol = _parameter_tolerance(_domain)
    _breakpoints = list(_parameters)
    for _stratum in _canopy["base"]["strata"]:
        _breakpoints.append(float(_stratum["start"]))
        _breakpoints.append(float(_stratum["end"]))

    _simplices = _used_coordinate_simplices(_topology)
    _rows = _model["rows"]
    _indices = _model["simplex_indices"]
    _new_roots = []
    for _segment in range(len(_parameters) - 1):
        _left_parameter = _parameters[_segment]
        _right_parameter = _parameters[_segment + 1]
        for _left_simplex, _right_simplex in combinations(_simplices, 2):
            _left_index = _indices[_left_simplex]
            _right_index = _indices[_right_simplex]
            _left_difference = (
                _rows[_segment][_left_index]
                - _rows[_segment][_right_index]
            )
            _right_difference = (
                _rows[_segment + 1][_left_index]
                - _rows[_segment + 1][_right_index]
            )
            if abs(_left_difference) <= _atol:
                continue
            if abs(_right_difference) <= _atol:
                continue
            if _left_difference * _right_difference >= 0.0:
                continue
            _fraction = _left_difference / (
                _left_difference - _right_difference
            )
            _root = _left_parameter + _fraction * (
                _right_parameter - _left_parameter
            )
            if _root <= _left_parameter + _parameter_atol:
                continue
            if _root >= _right_parameter - _parameter_atol:
                continue
            _new_roots.append(_root)
            if len(_new_roots) > _max_refinement_events:
                raise RuntimeError(
                    "D-canopy coordinate refinement exceeded its event limit"
                )
    _breakpoints.extend(_new_roots)
    _breakpoints.extend(_domain)
    return _unique_parameters(_breakpoints, _parameter_atol)


def _stratum_contains(_stratum, _parameter, _tolerance):
    """Return whether a stored constructible stratum contains a parameter."""

    _start = float(_stratum["start"])
    _end = float(_stratum["end"])
    if _parameter < _start - _tolerance or _parameter > _end + _tolerance:
        return False
    if abs(_parameter - _start) <= _tolerance and not _stratum["left_closed"]:
        return False
    if abs(_parameter - _end) <= _tolerance and not _stratum["right_closed"]:
        return False
    return True


def _parent_a_stratum(_canopy, _parameter):
    """Return the unique A-canopy stratum containing a parameter."""

    _tolerance = _parameter_tolerance(_canopy["base"]["domain"])
    _matches = tuple(
        _stratum
        for _stratum in _canopy["base"]["strata"]
        if _stratum_contains(_stratum, _parameter, _tolerance)
    )
    if len(_matches) != 1:
        raise RuntimeError("D-refinement point has no unique parent A-stratum")
    return _matches[0]


def _refined_base_strata(_canopy, _breakpoints):
    """Build point and open-interval cells on the refined base."""

    _strata = []
    for _index, _parameter in enumerate(_breakpoints):
        _parent = _parent_a_stratum(_canopy, _parameter)
        _point = {}
        _point["id"] = len(_strata)
        _point["kind"] = "point"
        _point["start"] = _parameter
        _point["end"] = _parameter
        _point["left_closed"] = True
        _point["right_closed"] = True
        _point["representative_parameter"] = _parameter
        _point["parent_a_stratum_id"] = _parent["id"]
        _strata.append(_point)
        if _index == len(_breakpoints) - 1:
            continue
        _next_parameter = _breakpoints[_index + 1]
        _representative = 0.5 * (_parameter + _next_parameter)
        _parent = _parent_a_stratum(_canopy, _representative)
        _interval = {}
        _interval["id"] = len(_strata)
        _interval["kind"] = "interval"
        _interval["start"] = _parameter
        _interval["end"] = _next_parameter
        _interval["left_closed"] = False
        _interval["right_closed"] = False
        _interval["representative_parameter"] = _representative
        _interval["parent_a_stratum_id"] = _parent["id"]
        _strata.append(_interval)
    return tuple(_strata)


def _ambient_cells(_canopy, _topology, _strata, _model, _atol):
    """Refine every symbolic A-cell over the D-specific base strata."""

    _fibers = dict(
        (_fiber["stratum_id"], _fiber)
        for _fiber in _canopy["representative_fibers"]
    )
    _a_cells = dict(
        (_cell["cell_id"], _cell)
        for _cell in _topology["cells"]
    )
    _cells = []
    for _stratum in _strata:
        _a_stratum_id = _stratum["parent_a_stratum_id"]
        _fiber = _fibers[_a_stratum_id]
        _parameter = float(_stratum["representative_parameter"])
        for _point in _fiber["points"]:
            _a_cell_id = _point["sheet_id"]
            _eta, _off_diagonal = _eta_at(
                _model,
                _point["pair"],
                _parameter,
                _atol
            )
            _cell = {}
            _cell["cell_id"] = (
                _stratum["id"],
                _point["pair_index"]
            )
            _cell["a_cell_id"] = _a_cell_id
            _cell["a_stratum_id"] = _a_stratum_id
            _cell["stratum_id"] = _stratum["id"]
            _cell["pair_index"] = _point["pair_index"]
            _cell["pair"] = _point["pair"]
            _cell["homology_dimension"] = _point["homology_dimension"]
            _cell["kind"] = _stratum["kind"]
            _cell["start"] = float(_stratum["start"])
            _cell["end"] = float(_stratum["end"])
            _cell["left_closed"] = bool(_stratum["left_closed"])
            _cell["right_closed"] = bool(_stratum["right_closed"])
            _cell["representative_parameter"] = _parameter
            _cell["representative_eta"] = _eta
            _cell["representative_eta_prime"] = (
                _eta if _off_diagonal else DELTA
            )
            _cell["ambient_status"] = (
                "off_diagonal" if _off_diagonal else "diagonal"
            )
            _cell["source_a_cell"] = dict(_a_cells[_a_cell_id])
            _cells.append(_cell)
    return tuple(sorted(_cells, key=lambda _cell: _cell_sort_key(_cell["cell_id"])))


def _relation_record(_relation_id, _kind, _source, _target, _a_relation_id=None):
    """Return one relation in the refined symbolic A-topology."""

    _record = {}
    _record["relation_id"] = _relation_id
    _record["kind"] = _kind
    _record["source_cell_id"] = _source
    _record["target_cell_id"] = _target
    _record["a_relation_id"] = _a_relation_id
    return _record


def _refined_relations(_topology, _strata, _cells):
    """Lift A-continuations and within-cell continuity to the refinement."""

    _by_stratum = {_stratum["id"]: {} for _stratum in _strata}
    for _cell in _cells:
        _by_stratum.setdefault(_cell["stratum_id"], {})[
            _cell["a_cell_id"]
        ] = _cell["cell_id"]

    _relations = []
    _seen = set()

    def _add(_kind, _source, _target, _a_relation_id=None):
        if _source == _target:
            return
        _edge = frozenset((_source, _target))
        if _edge in _seen:
            return
        _seen.add(_edge)
        _relations.append(
            _relation_record(
                len(_relations),
                _kind,
                _source,
                _target,
                _a_relation_id
            )
        )

    for _stratum in _strata:
        _mapping = _by_stratum[_stratum["id"]]
        for _relation in _topology["relations"]:
            _source = _relation["source_cell_id"]
            _target = _relation["target_cell_id"]
            if _source not in _mapping or _target not in _mapping:
                continue
            _add(
                "lifted_a_relation",
                _mapping[_source],
                _mapping[_target],
                _relation["relation_id"]
            )

    _ordered = tuple(sorted(_strata, key=lambda _stratum: _stratum["id"]))
    for _left, _right in zip(_ordered, _ordered[1:]):
        _left_mapping = _by_stratum[_left["id"]]
        _right_mapping = _by_stratum[_right["id"]]
        for _a_cell_id in set(_left_mapping).intersection(_right_mapping):
            _add(
                "within_a_cell_continuity",
                _left_mapping[_a_cell_id],
                _right_mapping[_a_cell_id]
            )
        for _relation in _topology["relations"]:
            _a_source = _relation["source_cell_id"]
            _a_target = _relation["target_cell_id"]
            if _a_source in _left_mapping and _a_target in _right_mapping:
                _add(
                    "lifted_a_incidence",
                    _left_mapping[_a_source],
                    _right_mapping[_a_target],
                    _relation["relation_id"]
                )
            if _a_target in _left_mapping and _a_source in _right_mapping:
                _add(
                    "lifted_a_incidence",
                    _left_mapping[_a_target],
                    _right_mapping[_a_source],
                    _relation["relation_id"]
                )
    return tuple(_relations)


def _adjacency(_cells, _relations):
    """Return symmetric adjacency generated by refined relations."""

    _adjacency = dict(
        (_cell["cell_id"], {_cell["cell_id"]})
        for _cell in _cells
    )
    for _relation in _relations:
        _source = _relation["source_cell_id"]
        _target = _relation["target_cell_id"]
        _adjacency[_source].add(_target)
        _adjacency[_target].add(_source)
    return dict(
        (
            _cell_id,
            tuple(sorted(_neighbors, key=_cell_sort_key))
        )
        for _cell_id, _neighbors in _adjacency.items()
    )


def _local_stratum_ids(_strata, _stratum_id):
    """Return refined strata in one canonical local base neighborhood."""

    _index = next(
        _index
        for _index, _stratum in enumerate(_strata)
        if _stratum["id"] == _stratum_id
    )
    _seed = _strata[_index]
    if _seed["kind"] == "interval":
        return (_stratum_id,)
    _identifiers = [_stratum_id]
    if _index > 0 and _strata[_index - 1]["kind"] == "interval":
        _identifiers.append(_strata[_index - 1]["id"])
    if _index + 1 < len(_strata) and _strata[_index + 1]["kind"] == "interval":
        _identifiers.append(_strata[_index + 1]["id"])
    return tuple(sorted(_identifiers))


def _graph_components(_adjacency, _allowed):
    """Return connected components of an induced finite graph."""

    _allowed = set(_allowed)
    _components = []
    _unvisited = set(_allowed)
    while _unvisited:
        _seed = min(_unvisited, key=_cell_sort_key)
        _visited = set()
        _stack = [_seed]
        while _stack:
            _current = _stack.pop()
            if _current in _visited:
                continue
            _visited.add(_current)
            for _neighbor in _adjacency[_current]:
                if _neighbor in _allowed and _neighbor not in _visited:
                    _stack.append(_neighbor)
        _unvisited.difference_update(_visited)
        _components.append(
            tuple(sorted(_visited, key=_cell_sort_key))
        )
    return tuple(_components)


def _local_components(_strata, _cells, _adjacency, _stratum_id, _allowed=None):
    """Return local trajectory components meeting one refined fiber."""

    _local_strata = set(_local_stratum_ids(_strata, _stratum_id))
    _active = {
        _cell["cell_id"]
        for _cell in _cells
        if _cell["stratum_id"] in _local_strata
    }
    if _allowed is not None:
        _active.intersection_update(_allowed)
    _fiber_ids = {
        _cell["cell_id"]
        for _cell in _cells
        if _cell["stratum_id"] == _stratum_id
        and (_allowed is None or _cell["cell_id"] in _allowed)
    }
    _components = []
    for _component in _graph_components(_adjacency, _active):
        _fiber_component = tuple(
            _cell_id
            for _cell_id in _component
            if _cell_id in _fiber_ids
        )
        if not _fiber_component:
            continue
        _record = {}
        _record["cell_ids"] = _component
        _record["fiber_cell_ids"] = _fiber_component
        _components.append(_record)
    return tuple(_components)


def _closure_of_off_diagonal(_strata, _cells, _adjacency):
    """Compute ``cl(E_tilde)`` in the finite refined A-topology."""

    _off_diagonal = {
        _cell["cell_id"]
        for _cell in _cells
        if _cell["ambient_status"] == "off_diagonal"
    }
    _component_by_cell = {}
    for _stratum in _strata:
        for _component in _local_components(
            _strata,
            _cells,
            _adjacency,
            _stratum["id"]
        ):
            for _cell_id in _component["fiber_cell_ids"]:
                _component_by_cell[_cell_id] = _component["cell_ids"]
    _closure = set()
    for _cell in _cells:
        _cell_id = _cell["cell_id"]
        if _off_diagonal.intersection(_component_by_cell[_cell_id]):
            _closure.add(_cell_id)
    _boundary = _closure.difference(_off_diagonal)
    _removed = set(_component_by_cell).difference(_closure)
    return (
        tuple(sorted(_off_diagonal, key=_cell_sort_key)),
        tuple(sorted(_closure, key=_cell_sort_key)),
        tuple(sorted(_boundary, key=_cell_sort_key)),
        tuple(sorted(_removed, key=_cell_sort_key)),
        _component_by_cell
    )


def _coordinates_equal(_left, _right, _atol):
    """Return tolerance-aware equality for two off-diagonal coordinates."""

    for _left_value, _right_value in zip(_left, _right):
        if math.isinf(_left_value) or math.isinf(_right_value):
            if _left_value != _right_value:
                return False
            continue
        if abs(_left_value - _right_value) > _atol:
            return False
    return True


def _multiplicity_groups(_cells, _atol):
    """Group equal off-diagonal coordinates within one refined fiber."""

    _groups = []
    for _cell in _cells:
        if _cell["status"] != "off_diagonal":
            continue
        _match = None
        for _group in _groups:
            if _group["homology_dimension"] != _cell["homology_dimension"]:
                continue
            if _coordinates_equal(
                _group["eta_prime"],
                _cell["representative_eta_prime"],
                _atol
            ):
                _match = _group
                break
        if _match is None:
            _match = {}
            _match["homology_dimension"] = _cell["homology_dimension"]
            _match["eta_prime"] = _cell["representative_eta_prime"]
            _match["cell_ids"] = []
            _groups.append(_match)
        _match["cell_ids"].append(_cell["cell_id"])
    _records = []
    for _group in _groups:
        if len(_group["cell_ids"]) < 2:
            continue
        _record = dict(_group)
        _record["cell_ids"] = tuple(
            sorted(_record["cell_ids"], key=_cell_sort_key)
        )
        _records.append(_record)
    return tuple(_records)


def _d_structurality(_strata, _retained_cells, _component_by_cell, _atol):
    """Classify D-diagram multiplicities using ambient A-topology."""

    _events = []
    _structural_locus = []
    for _stratum in _strata:
        _fiber = tuple(
            _cell
            for _cell in _retained_cells
            if _cell["stratum_id"] == _stratum["id"]
        )
        _groups = []
        for _group in _multiplicity_groups(_fiber, _atol):
            _component_buckets = {}
            for _cell_id in _group["cell_ids"]:
                _key = frozenset(_component_by_cell[_cell_id])
                _component_buckets.setdefault(_key, []).append(_cell_id)
            _non_hausdorff = set()
            for _bucket in _component_buckets.values():
                if len(_bucket) > 1:
                    _non_hausdorff.update(_bucket)
            _record = dict(_group)
            _record["non_hausdorff_cell_ids"] = tuple(
                sorted(_non_hausdorff, key=_cell_sort_key)
            )
            _record["classification"] = (
                "structural" if _non_hausdorff else "coincidental"
            )
            _groups.append(_record)
        _structural_groups = tuple(
            _group
            for _group in _groups
            if _group["classification"] == "structural"
        )
        _event = {}
        _event["stratum_id"] = _stratum["id"]
        _event["kind"] = _stratum["kind"]
        _event["start"] = float(_stratum["start"])
        _event["end"] = float(_stratum["end"])
        _event["representative_parameter"] = float(
            _stratum["representative_parameter"]
        )
        _event["multiplicity_points"] = tuple(_groups)
        _event["structural_points"] = _structural_groups
        _event["structural"] = bool(_structural_groups)
        _events.append(_event)
        if _event["structural"]:
            _structural_locus.append(
                dict(
                    (_key, _event[_key])
                    for _key in (
                        "stratum_id",
                        "kind",
                        "start",
                        "end",
                        "representative_parameter"
                    )
                )
            )
    _result = {}
    _result["events"] = tuple(_events)
    _result["structural_locus"] = tuple(_structural_locus)
    _result["structural_stratum_ids"] = tuple(
        _record["stratum_id"]
        for _record in _structural_locus
    )
    _result["structural_parameters"] = tuple(
        _record["start"]
        for _record in _structural_locus
        if _record["kind"] == "point"
    )
    _result["complete"] = True
    _result["certificate_scope"] = "refined_piecewise_linear_base"
    return _result


def _induced_relations(_relations, _retained_ids):
    """Restrict ambient relations to the closed D-total space."""

    _retained_ids = set(_retained_ids)
    _restricted = []
    for _relation in _relations:
        if _relation["source_cell_id"] not in _retained_ids:
            continue
        if _relation["target_cell_id"] not in _retained_ids:
            continue
        _record = dict(_relation)
        _record["relation_id"] = len(_restricted)
        _record["ambient_relation_id"] = _relation["relation_id"]
        _restricted.append(_record)
    return tuple(_restricted)


def _chart_certificate(_strata, _cells, _adjacency, _structural_ids):
    """Check Definition 4.1 local sheets on the complement of ``S``."""

    _structural_ids = set(_structural_ids)
    _allowed = {
        _cell["cell_id"]
        for _cell in _cells
        if _cell["stratum_id"] not in _structural_ids
    }
    _records = []
    for _stratum in _strata:
        if _stratum["id"] in _structural_ids:
            continue
        _components = _local_components(
            _strata,
            _cells,
            _adjacency,
            _stratum["id"],
            _allowed=_allowed
        )
        _projection_injective = True
        for _component in _components:
            _counts = {}
            for _cell_id in _component["cell_ids"]:
                _counts[_cell_id[0]] = _counts.get(_cell_id[0], 0) + 1
            if any(_count > 1 for _count in _counts.values()):
                _projection_injective = False
        _diagonal_condition = all(
            _cell["representative_eta_prime"] == DELTA
            for _cell in _cells
            if _cell["stratum_id"] == _stratum["id"]
            and _cell["status"] == "diagonal_boundary"
        )
        _record = {}
        _record["stratum_id"] = _stratum["id"]
        _record["kind"] = _stratum["kind"]
        _record["representative_parameter"] = float(
            _stratum["representative_parameter"]
        )
        _record["local_sheet_count"] = len(_components)
        _record["projection_injective_on_sheets"] = _projection_injective
        _record["new_boundaries_map_to_delta"] = _diagonal_condition
        _record["valid"] = _projection_injective and _diagonal_condition
        _records.append(_record)
    _result = {}
    _result["charts"] = tuple(_records)
    _result["all_charts_valid"] = all(_record["valid"] for _record in _records)
    _result["excluded_structural_stratum_ids"] = tuple(
        sorted(_structural_ids)
    )
    _result["complete"] = True
    _result["paper_condition"] = "Definition 4.1"
    return _result


def _sheet_domains(_cells, _adjacency, _structural_ids):
    """Return connected retained domains after deleting the structural locus."""

    _structural_ids = set(_structural_ids)
    _allowed = {
        _cell["cell_id"]
        for _cell in _cells
        if _cell["stratum_id"] not in _structural_ids
    }
    _cell_by_id = dict(
        (_cell["cell_id"], _cell)
        for _cell in _cells
    )
    _domains = []
    for _domain_id, _component in enumerate(
        _graph_components(_adjacency, _allowed)
    ):
        _pieces = []
        for _cell_id in _component:
            _cell = _cell_by_id[_cell_id]
            _piece = {}
            _piece["cell_id"] = _cell_id
            _piece["stratum_id"] = _cell["stratum_id"]
            _piece["kind"] = _cell["kind"]
            _piece["start"] = _cell["start"]
            _piece["end"] = _cell["end"]
            _piece["left_closed"] = _cell["left_closed"]
            _piece["right_closed"] = _cell["right_closed"]
            _piece["status"] = _cell["status"]
            _pieces.append(_piece)
        _record = {}
        _record["domain_id"] = _domain_id
        _record["cell_ids"] = _component
        _record["base_pieces"] = tuple(_pieces)
        _record["homology_dimensions"] = tuple(
            sorted(
                {
                    _cell_by_id[_cell_id]["homology_dimension"]
                    for _cell_id in _component
                }
            )
        )
        _domains.append(_record)
    return tuple(_domains)


def _diagram_from_points(_points):
    """Return the dimension-indexed D-diagram of fiber points."""

    _by_dimension = {}
    for _point in _points:
        if _point["eta_prime"] == DELTA:
            continue
        _dimension = _point["homology_dimension"]
        _by_dimension.setdefault(_dimension, []).append(_point["eta_prime"])
    return dict(
        (_dimension, tuple(sorted(_by_dimension[_dimension])))
        for _dimension in sorted(_by_dimension)
    )


def _representative_fibers(_strata, _cells):
    """Return variable-cardinality representative D-fibers."""

    _fibers = []
    for _stratum in _strata:
        _points = []
        for _cell in _cells:
            if _cell["stratum_id"] != _stratum["id"]:
                continue
            _point = dict(_cell)
            _point["parameter"] = float(_stratum["representative_parameter"])
            _point["projection"] = _point["parameter"]
            _point["eta_prime"] = _cell["representative_eta_prime"]
            _points.append(_point)
        _fiber = {}
        _fiber["stratum_id"] = _stratum["id"]
        _fiber["kind"] = _stratum["kind"]
        _fiber["representative_parameter"] = float(
            _stratum["representative_parameter"]
        )
        _fiber["points"] = tuple(_points)
        _fiber["fiber_cardinality"] = len(_points)
        _fiber["diminished_diagram"] = _diagram_from_points(_points)
        _fibers.append(_fiber)
    return tuple(_fibers)


def build_d_canopy(_canopy, _topology=None, _atol=None, _max_refinement_events=100000):
    """Construct the paper's D-canopy from a complete A-canopy.

    The implementation forms the off-diagonal locus ``E_tilde`` and then
    takes its closure in a D-refined symbolic presentation of the A-canopy
    topology.  Diagonal cells retained only by closure map to
    :data:`DELTA`; diagonal cells with no off-diagonal approach are removed.

    The returned object is an approximate D-canopy exactly when its
    off-diagonal diagram has structural points.  Their constructible locus
    is stored rather than assuming in advance that it consists only of
    isolated parameters.

    :param _canopy: Complete fixed-complex one-parameter A-canopy.
    :type _canopy: mapping
    :param _topology: Optional matching finite A-canopy topology.
    :type _topology: mapping or None
    :param _atol: Tolerance for floating-point diagonal equality.
    :type _atol: float or None
    :param _max_refinement_events: Safety cap on added coordinate roots.
    :type _max_refinement_events: int
    :return: Finite constructible D-canopy presentation.
    :rtype: dict
    """

    validate_a_canopy(_canopy)
    _canopy = (_canopy.to_record() if isinstance(_canopy, ACanopy)
               else deepcopy(_canopy))
    if _canopy.get("representation") != "finite_constructible_one_parameter":
        raise ValueError("D-canopy construction requires a one-parameter A-canopy")
    if not _canopy.get("complete", False):
        raise ValueError("D-canopy construction requires complete A-continuations")
    if _topology is None:
        _topology = build_canopy_topology(_canopy)
    else:
        validate_canopy_topology(_topology)
        _topology = (_topology.to_record() if isinstance(_topology, CanopyTopology)
                     else deepcopy(_topology))
        if _topology["canopy"] != _canopy:
            raise ValueError("Supplied topology belongs to a different A-canopy")

    _atol = _absolute_tolerance(_canopy, _atol)
    _max_refinement_events = _positive_integer(
        _max_refinement_events,
        "Maximum refinement events"
    )
    _model = _filtration_model(_canopy)
    _breakpoints = _refinement_breakpoints(
        _canopy,
        _topology,
        _model,
        _atol,
        _max_refinement_events
    )
    _strata = _refined_base_strata(_canopy, _breakpoints)
    _ambient_cells_value = _ambient_cells(
        _canopy,
        _topology,
        _strata,
        _model,
        _atol
    )
    _ambient_relations = _refined_relations(
        _topology,
        _strata,
        _ambient_cells_value
    )
    _ambient_adjacency = _adjacency(
        _ambient_cells_value,
        _ambient_relations
    )
    (
        _off_diagonal_ids,
        _retained_ids,
        _boundary_ids,
        _removed_ids,
        _component_by_cell
    ) = _closure_of_off_diagonal(
        _strata,
        _ambient_cells_value,
        _ambient_adjacency
    )

    _off_diagonal_set = set(_off_diagonal_ids)
    _boundary_set = set(_boundary_ids)
    _retained_set = set(_retained_ids)
    _retained_cells = []
    for _ambient_cell in _ambient_cells_value:
        _cell_id = _ambient_cell["cell_id"]
        if _cell_id not in _retained_set:
            continue
        _cell = dict(_ambient_cell)
        if _cell_id in _off_diagonal_set:
            _cell["status"] = "off_diagonal"
        elif _cell_id in _boundary_set:
            _cell["status"] = "diagonal_boundary"
        else:
            raise RuntimeError("D-closure produced an unclassified retained cell")
        _retained_cells.append(_cell)
    _retained_cells = tuple(_retained_cells)
    _relations = _induced_relations(
        _ambient_relations,
        _retained_ids
    )
    _adjacency_value = _adjacency(_retained_cells, _relations)
    _structurality = _d_structurality(
        _strata,
        _retained_cells,
        _component_by_cell,
        _atol
    )
    _chart_certificate_value = _chart_certificate(
        _strata,
        _retained_cells,
        _adjacency_value,
        _structurality["structural_stratum_ids"]
    )
    if not _chart_certificate_value["all_charts_valid"]:
        raise RuntimeError(
            "Closed off-diagonal locus failed a Definition 4.1 local chart"
        )
    _domains = _sheet_domains(
        _retained_cells,
        _adjacency_value,
        _structurality["structural_stratum_ids"]
    )
    _fibers = _representative_fibers(_strata, _retained_cells)
    _cardinalities = tuple(
        _fiber["fiber_cardinality"]
        for _fiber in _fibers
    )
    _base = dict(_canopy["base"])
    _base["a_strata"] = tuple(
        dict(_stratum)
        for _stratum in _canopy["base"]["strata"]
    )
    _base["strata"] = _strata
    _base["refinement_breakpoints"] = _breakpoints
    _base["refinement"] = "A-strata plus PL coordinate equalities"

    _ambient = {}
    _ambient["object_type"] = "D-refined A-canopy topology"
    _ambient["cells"] = _ambient_cells_value
    _ambient["relations"] = _ambient_relations
    _ambient["adjacency"] = _ambient_adjacency
    _ambient["cell_count"] = len(_ambient_cells_value)
    _ambient["relation_count"] = len(_ambient_relations)
    _ambient["complete"] = True

    _result = {}
    _result["object_type"] = "D-canopy"
    _result["output_record_version"] = OUTPUT_RECORD_VERSION
    _result["mathematical_scope"] = MATHEMATICAL_SCOPE
    _result["tie_break_rule"] = TIE_BREAK_RULE
    _result["paper_definition"] = "Definitions 4.1-4.3 and Theorem 4.4"
    _result["representation"] = "finite_piecewise_linear_one_parameter_d_canopy"
    _result["homology_dimension"] = _canopy["homology_dimension"]
    _result["diagonal_symbol"] = DELTA
    _result["absolute_tolerance"] = _atol
    _result["base"] = _base
    _result["source_a_canopy"] = _canopy
    _result["source_a_topology"] = _topology
    _result["ambient_refinement"] = _ambient
    _result["cells"] = _retained_cells
    _result["relations"] = _relations
    _result["adjacency"] = _adjacency_value
    _result["projection"] = tuple(
        (_cell["cell_id"], _cell["stratum_id"])
        for _cell in _retained_cells
    )
    _result["representative_fibers"] = _fibers
    _result["fiber_cardinality_range"] = (
        min(_cardinalities, default=0),
        max(_cardinalities, default=0)
    )
    _result["off_diagonal_cell_ids"] = _off_diagonal_ids
    _result["retained_cell_ids"] = _retained_ids
    _result["diagonal_boundary_cell_ids"] = _boundary_ids
    _result["removed_diagonal_cell_ids"] = _removed_ids
    _result["closure_scope"] = "complete_refined_constructible_PL_model"
    _result["structurality"] = _structurality
    _result["structural_locus"] = _structurality["structural_locus"]
    _result["excluded_structural_parameters"] = _structurality[
        "structural_parameters"
    ]
    _result["excluded_structural_stratum_ids"] = _structurality[
        "structural_stratum_ids"
    ]
    _result["approximate"] = bool(_structurality["structural_locus"])
    _result["local_chart_certificate"] = _chart_certificate_value
    _result["sheet_domains"] = _domains
    _result["complete"] = True
    _result["continuum_enumerated"] = False
    validate_d_canopy(_result)
    return _result


def _d_canopy_data(_d_canopy):
    """Return validated core records from a D-canopy candidate."""

    if not hasattr(_d_canopy, "__getitem__"):
        raise TypeError("D-canopy must be a mapping")
    try:
        _object_type = _d_canopy["object_type"]
        _cells = tuple(_d_canopy["cells"])
        _relations = tuple(_d_canopy["relations"])
        _adjacency_value = dict(_d_canopy["adjacency"])
        _strata = tuple(_d_canopy["base"]["strata"])
    except KeyError as _error:
        raise ValueError(
            "D-canopy is missing {}".format(_error.args[0])
        ) from _error
    except TypeError as _error:
        raise ValueError("D-canopy records must be iterable") from _error
    if _object_type != "D-canopy":
        raise ValueError("Object is not a D-canopy presentation")
    return _cells, _relations, _adjacency_value, _strata


def validate_d_canopy(_d_canopy):
    """Validate finite closure, projection and local-chart invariants.

    :param _d_canopy: Candidate returned by :func:`build_d_canopy`.
    :type _d_canopy: mapping
    :return: The validated input mapping.
    :rtype: mapping
    :raises ValueError: If a stored D-canopy invariant fails.
    """

    _cells, _relations, _adjacency_value, _strata = _d_canopy_data(
        _d_canopy
    )
    if _d_canopy.get("representation") != "finite_piecewise_linear_one_parameter_d_canopy":
        raise ValueError("Unsupported D-canopy representation")
    if not _d_canopy.get("complete", False):
        raise ValueError("D-canopy must certify a complete finite construction")
    if _d_canopy.get("diagonal_symbol") != DELTA:
        raise ValueError("D-canopy uses an unknown diagonal quotient symbol")
    validate_a_canopy(_d_canopy["source_a_canopy"])
    validate_canopy_topology(_d_canopy["source_a_topology"])
    if (_d_canopy["source_a_topology"]["canopy"] !=
            _d_canopy["source_a_canopy"]):
        raise ValueError("D-canopy topology belongs to another A-canopy")

    _stratum_ids = tuple(_stratum["id"] for _stratum in _strata)
    if len(set(_stratum_ids)) != len(_stratum_ids):
        raise ValueError("D-canopy repeats a refined base stratum")
    _cell_ids = tuple(_cell["cell_id"] for _cell in _cells)
    if len(set(_cell_ids)) != len(_cell_ids):
        raise ValueError("D-canopy repeats a retained cell identifier")
    if set(_cell_ids) != set(_adjacency_value):
        raise ValueError("D-canopy adjacency does not cover retained cells")
    if tuple(_d_canopy["retained_cell_ids"]) != tuple(
        sorted(_cell_ids, key=_cell_sort_key)
    ):
        raise ValueError("D-canopy retained identifiers disagree with its cells")

    _off_diagonal = set(_d_canopy["off_diagonal_cell_ids"])
    _boundary = set(_d_canopy["diagonal_boundary_cell_ids"])
    _removed = set(_d_canopy["removed_diagonal_cell_ids"])
    if _off_diagonal.intersection(_boundary):
        raise ValueError("Off-diagonal and boundary cells must be disjoint")
    if _off_diagonal.union(_boundary) != set(_cell_ids):
        raise ValueError("Retained D-cells do not equal the closed locus")
    _ambient_ids = {
        _cell["cell_id"]
        for _cell in _d_canopy["ambient_refinement"]["cells"]
    }
    if set(_cell_ids).union(_removed) != _ambient_ids:
        raise ValueError("Retained and removed cells do not partition the refinement")
    _ambient = _d_canopy["ambient_refinement"]
    _closure = _closure_of_off_diagonal(
        _strata, _ambient["cells"], _ambient["adjacency"])
    for _actual, _stored in zip(
            _closure[:4],
            (_d_canopy["off_diagonal_cell_ids"],
             _d_canopy["retained_cell_ids"],
             _d_canopy["diagonal_boundary_cell_ids"],
             _d_canopy["removed_diagonal_cell_ids"])):
        if _actual != tuple(_stored):
            raise ValueError("D-canopy cells disagree with off-diagonal closure")
    for _cell in _cells:
        _cell_id = _cell["cell_id"]
        if _cell_id in _off_diagonal:
            if _cell["status"] != "off_diagonal":
                raise ValueError("Off-diagonal cell has an incorrect status")
            if _cell["representative_eta_prime"] == DELTA:
                raise ValueError("Off-diagonal cell cannot map to DELTA")
        else:
            if _cell["status"] != "diagonal_boundary":
                raise ValueError("Closure boundary has an incorrect status")
            if _cell["representative_eta_prime"] != DELTA:
                raise ValueError("Every retained diagonal boundary must map to DELTA")

    _expected = dict((_cell_id, {_cell_id}) for _cell_id in _cell_ids)
    for _relation in _relations:
        _source = _relation.get("source_cell_id")
        _target = _relation.get("target_cell_id")
        if _source not in _expected or _target not in _expected:
            raise ValueError("D-canopy relation uses a nonretained cell")
        _expected[_source].add(_target)
        _expected[_target].add(_source)
    for _cell_id, _neighbors in _adjacency_value.items():
        if tuple(sorted(_neighbors, key=_cell_sort_key)) != tuple(
            sorted(_expected[_cell_id], key=_cell_sort_key)
        ):
            raise ValueError("D-canopy adjacency disagrees with its relations")

    _projection = tuple(
        (_cell["cell_id"], _cell["stratum_id"])
        for _cell in _cells
    )
    if tuple(_d_canopy["projection"]) != _projection:
        raise ValueError("D-canopy projection disagrees with retained cells")
    if not _d_canopy["local_chart_certificate"].get(
        "all_charts_valid",
        False
    ):
        raise ValueError("D-canopy lacks valid local charts away from S")
    if bool(_d_canopy["structural_locus"]) != bool(
        _d_canopy.get("approximate")
    ):
        raise ValueError("Approximate flag disagrees with structural locus")
    return _d_canopy


def _stratum_at_parameter(_d_canopy, _parameter):
    """Return the unique refined base stratum containing a parameter."""

    _parameter = _finite_float(_parameter, "Base parameter")
    _tolerance = _parameter_tolerance(_d_canopy["base"]["domain"])
    _matches = tuple(
        _stratum
        for _stratum in _d_canopy["base"]["strata"]
        if _stratum_contains(_stratum, _parameter, _tolerance)
    )
    if len(_matches) != 1:
        raise ValueError("Parameter does not lie in exactly one D-stratum")
    return _parameter, _matches[0]


def d_canopy_fiber(_d_canopy, _parameter):
    """Evaluate one finite D-canopy fiber, including retained ``DELTA``.

    :param _d_canopy: Finite D-canopy from :func:`build_d_canopy`.
    :type _d_canopy: mapping
    :param _parameter: Point of the modeled interval base.
    :type _parameter: real
    :return: Retained D-points with projection and quotient eta values.
    :rtype: tuple of dict
    """

    validate_d_canopy(_d_canopy)
    _parameter, _stratum = _stratum_at_parameter(_d_canopy, _parameter)
    _model = _filtration_model(_d_canopy["source_a_canopy"])
    _atol = float(_d_canopy["absolute_tolerance"])
    _points = []
    for _cell in _d_canopy["cells"]:
        if _cell["stratum_id"] != _stratum["id"]:
            continue
        _eta, _off_diagonal = _eta_at(
            _model,
            _cell["pair"],
            _parameter,
            _atol
        )
        _point = dict(_cell)
        _point["parameter"] = _parameter
        _point["projection"] = _parameter
        _point["canopy_point"] = (
            _parameter,
            _cell["pair"][0],
            _cell["pair"][1]
        )
        _point["eta"] = _eta
        _point["eta_prime"] = _eta if _off_diagonal else DELTA
        _point["on_diagonal"] = not _off_diagonal
        _points.append(_point)
    return tuple(_points)


def d_canopy_diagram(_d_canopy, _parameter):
    """Return the induced diminished diagram at one base parameter.

    Retained D-canopy boundary points mapping to :data:`DELTA` are omitted,
    while multiplicity of every strictly off-diagonal point is preserved.

    :param _d_canopy: Finite D-canopy from :func:`build_d_canopy`.
    :type _d_canopy: mapping
    :param _parameter: Point of the modeled interval base.
    :type _parameter: real
    :return: Diminished diagram keyed by homological dimension.
    :rtype: dict
    """

    return _diagram_from_points(d_canopy_fiber(_d_canopy, _parameter))


def d_canopy_boundary_cells(_d_canopy):
    """Return retained cells that map to ``DELTA`` by closure.

    :param _d_canopy: Finite D-canopy presentation.
    :type _d_canopy: mapping
    :return: Diagonal boundary cell records.
    :rtype: tuple of dict
    """

    validate_d_canopy(_d_canopy)
    return tuple(
        dict(_cell)
        for _cell in _d_canopy["cells"]
        if _cell["status"] == "diagonal_boundary"
    )


def d_canopy_sheet_domains(_d_canopy):
    """Return connected local-sheet domains on ``B`` minus structural loci.

    :param _d_canopy: Finite D-canopy presentation.
    :type _d_canopy: mapping
    :return: Constructible connected-domain records.
    :rtype: tuple of dict
    """

    validate_d_canopy(_d_canopy)
    return tuple(dict(_domain) for _domain in _d_canopy["sheet_domains"])


class DCanopy(Mapping):
    """Closure of the off-diagonal locus in a finite A-canopy topology.

    The finite presentation refines the A-canopy at filtration-value
    equalities, keeps limits that map to :data:`DELTA`, and excludes the
    resulting structural locus when reporting local sheet domains.

    :param _canopy: Complete fixed-complex A-canopy or its output record.
    :type _canopy: ACanopy or mapping
    :param _topology: Optional topology built from the same A-canopy.
    :type _topology: CanopyTopology, mapping or None
    :param _atol: Nonnegative tolerance for diagonal equality.
    :type _atol: float or None
    :param _max_refinement_events: Maximum additional coordinate roots.
    :type _max_refinement_events: int
    """

    __slots__ = ("_record",)

    def __init__(self, _canopy, _topology=None, _atol=None,
                 _max_refinement_events=100000):
        if hasattr(self, "_record"):
            raise AttributeError("DCanopy is immutable")
        object.__setattr__(self, "_record", build_d_canopy(
            _canopy, _topology, _atol, _max_refinement_events))

    def __setattr__(self, _name, _value):
        raise AttributeError("DCanopy is immutable")

    def __delattr__(self, _name):
        raise AttributeError("DCanopy is immutable")

    def __getitem__(self, _key):
        """Return one detached field of the finite record."""
        return deepcopy(self._record[_key])

    def __iter__(self):
        """Iterate over finite-record field names."""
        return iter(self._record)

    def __len__(self):
        """Return the number of finite-record fields."""
        return len(self._record)

    @property
    def cells(self):
        """Return retained symbolic cells, including diagonal boundaries."""
        return deepcopy(self._record["cells"])

    @property
    def relations(self):
        """Return continuation relations between retained cells."""
        return deepcopy(self._record["relations"])

    @property
    def structural_locus(self):
        """Return refined base strata excluded from local sheet domains."""
        return deepcopy(self._record["structural_locus"])

    @property
    def approximate(self):
        """Report whether structural points occur in the diminished diagram."""
        return self._record["approximate"]

    def to_record(self):
        """Return an independent copy of the finite D-canopy presentation."""
        return deepcopy(self._record)

    def fiber(self, _parameter):
        """Evaluate retained points above one parameter, including DELTA."""
        return d_canopy_fiber(self, _parameter)

    def diagram(self, _parameter, _homology_dimension=None):
        """Return the diminished diagram with multiplicities and no DELTA.

        With a degree selected, return its sorted point tuple. Otherwise,
        return a dictionary keyed by homology degree.
        """
        _diagram = d_canopy_diagram(self, _parameter)
        if _homology_dimension is None:
            return _diagram
        _dimension = _nonnegative_integer(
            _homology_dimension, "Homology dimension")
        return _diagram.get(_dimension, ())

    def boundary_cells(self):
        """Return retained diagonal limits of off-diagonal points."""
        return d_canopy_boundary_cells(self)

    def sheet_domains(self):
        """Return connected sheet domains away from structural strata."""
        return d_canopy_sheet_domains(self)
