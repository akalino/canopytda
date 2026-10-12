"""Order strata for piecewise-linear filtrations on one fixed complex K.

Only one labeled simplicial complex is considered. Filtration values are
specified at sampled parameters and interpolated linearly between them.
Pairing and equivalence are computed in later modules.
"""
from bisect import bisect_right
from copy import deepcopy

import gudhi as gd #pylint: disable=import-error

from .complex import SimplicialComplex
from .contract import MATHEMATICAL_SCOPE
from .contract import OUTPUT_RECORD_VERSION
from .contract import TIE_BREAK_RULE
from .filtration import Filtration
from .filtration import filtration_values
from .orders import canonical_refinement
from .utils import _are_face_related
from .utils import _finite_float
from .utils import _nonnegative_float
from .utils import _nonnegative_integer
from .utils import _positive_integer

__all__ = ["Stratification", "stratify_filtration_family"]
__all__ += ["materialize_filtration", "materialize_stratum"]
__all__ += ["stratum_adjacency"]


def _normalize_parameters(_count, _parameters):
    """Return strictly increasing finite parameter values."""

    if _parameters is None:
        return tuple(float(_index) for _index in range(_count))
    try:
        _values = tuple(
            _finite_float(_value, "Parameter")
            for _value in _parameters
        )
    except TypeError as _error:
        raise ValueError("Parameters must be iterable") from _error
    if len(_values) != _count:
        raise ValueError("Parameters and filtrations must have equal length")
    for _index in range(1, len(_values)):
        if _values[_index - 1] >= _values[_index]:
            raise ValueError("Parameters must be strictly increasing")
    return _values


def _fixed_family_data(_filtrations):
    """Validate one fixed-complex family and return aligned value rows.

    :param _filtrations: Filtrations on one finite simplicial complex.
    :type _filtrations: iterable of Filtration or gudhi.SimplexTree
    :return: Filtrations, canonical simplices and aligned value rows.
    :rtype: tuple
    """

    try:
        _inputs = tuple(_filtrations)
    except TypeError as _error:
        raise ValueError("Filtrations must be iterable") from _error
    if not _inputs:
        raise ValueError("At least one filtration is required")

    _normalized = []
    for _filtration in _inputs:
        if isinstance(_filtration, Filtration):
            _normalized.append(_filtration)
        elif isinstance(_filtration, gd.SimplexTree):
            _normalized.append(Filtration.from_simplex_tree(_filtration))
        else:
            raise TypeError("Expected a Filtration or gudhi.SimplexTree")

    _complex = _normalized[0].complex
    for _filtration in _normalized[1:]:
        if _filtration.complex != _complex:
            raise ValueError("Filtrations must use the same finite simplicial "
                             "complex K and vertex identifiers")

    _complex_simplices = _complex.simplices
    _value_rows = tuple(
        tuple(_filtration.value(_simplex) for _simplex in _complex_simplices)
        for _filtration in _normalized
    )
    return tuple(_normalized), _complex_simplices, _value_rows


def _sign(_value, _atol):
    """Return a tolerance-aware sign in ``{-1, 0, 1}``."""

    if abs(_value) <= _atol:
        return 0
    if _value < 0.0:
        return -1
    return 1


def _order_blocks(_simplices, _values, _atol):
    """Return compact value blocks encoding the minimal order.

    Face relations are fixed by the underlying complex. Consequently, the ordered
    blocks of equal filtration values determine the remaining relations in
    the minimal consistent order without materializing its quadratic
    transitive closure.
    """

    _entries = sorted(
        zip(_values, _simplices),
        key=lambda _entry: (_entry[0], len(_entry[1]), _entry[1])
    )
    _blocks = []
    _block = []
    _block_minimum = None

    for _value, _simplex in _entries:
        if not _block:
            _block = [_simplex]
            _block_minimum = _value
            continue
        if _value - _block_minimum <= _atol:
            _block.append(_simplex)
            continue
        _blocks.append(tuple(_block))
        _block = [_simplex]
        _block_minimum = _value

    if _block:
        _blocks.append(tuple(_block))
    return tuple(_blocks)


def _parameter_tolerance(_parameters):
    """Return a scale-aware tolerance for coalescing event parameters."""

    _scale = max(1.0, abs(_parameters[0]), abs(_parameters[-1]))
    _scale = max(_scale, _parameters[-1] - _parameters[0])
    return 1e-12 * _scale


def _critical_records(_simplices, _value_rows,
                      _parameters, _atol, _max_comparisons,
                      _max_events):
    """Return all isolated equality records for a linear family."""

    _segment_count = len(_parameters) - 1
    _simplex_count = len(_simplices)
    _candidate_count = _segment_count * _simplex_count * (_simplex_count - 1) // 2
    if _candidate_count > _max_comparisons:
        _message = ("Exact stratification requires {} pairwise segment"
                    " comparisons, above the limit of {}")
        raise RuntimeError(_message.format(_candidate_count, _max_comparisons))

    _records = []
    _comparisons = 0
    for _segment in range(_segment_count):
        _left_parameter = _parameters[_segment]
        _right_parameter = _parameters[_segment + 1]
        _left_values = _value_rows[_segment]
        _right_values = _value_rows[_segment + 1]

        for _first in range(_simplex_count):
            for _second in range(_first + 1, _simplex_count):
                _comparisons += 1
                if _are_face_related(_simplices[_first], _simplices[_second]):
                    continue

                _left_difference = _left_values[_first] - _left_values[_second]
                _right_difference = _right_values[_first] - _right_values[_second]
                _left_sign = _sign(_left_difference, _atol)
                _right_sign = _sign(_right_difference, _atol)

                if _left_sign == 0 and _right_sign == 0:
                    continue
                if _left_sign == 0:
                    _records.append((_left_parameter, _first, _second))
                elif _right_sign == 0:
                    _records.append((_right_parameter, _first, _second))
                elif _left_sign != _right_sign:
                    _fraction = _left_difference / (_left_difference - _right_difference)
                    _parameter = _left_parameter + _fraction * (_right_parameter - _left_parameter)
                    _records.append((_parameter, _first, _second))

                if len(_records) > _max_events:
                    _message = "Exact stratification found more than {} simplex-pair events"
                    raise RuntimeError(_message.format(_max_events))
    return _records, _comparisons


def _coalesced_events(_records, _parameters):
    """Group numerically coincident equality records."""

    if not _records:
        return ()
    _tolerance = _parameter_tolerance(_parameters)
    _ordered = sorted(_records, key=lambda _record: (_record[0], _record[1], _record[2]))
    _groups = []

    for _parameter, _first, _second in _ordered:
        if not _groups or abs(_parameter - _groups[-1]["parameter"]) > _tolerance:
            _group = {}
            _group["parameter"] = _parameter
            _group["pair_indices"] = {(_first, _second)}
            _groups.append(_group)
            continue
        _groups[-1]["pair_indices"].add((_first, _second))

    for _group in _groups:
        _group["pair_indices"] = tuple(sorted(_group["pair_indices"]))
    return tuple(_groups)


def _values_at_parameter(_parameters, _value_rows, _parameter):
    """Evaluate the piecewise-linear filtration value vector."""

    if len(_parameters) == 1 or _parameter <= _parameters[0]:
        return _value_rows[0]
    if _parameter >= _parameters[-1]:
        return _value_rows[-1]

    _segment = bisect_right(_parameters, _parameter) - 1
    _left_parameter = _parameters[_segment]
    _right_parameter = _parameters[_segment + 1]
    _fraction = (_parameter - _left_parameter) / (_right_parameter - _left_parameter)
    _left_values = _value_rows[_segment]
    _right_values = _value_rows[_segment + 1]
    return tuple(
        (1.0 - _fraction) * _left + _fraction * _right
        for _left, _right in zip(_left_values, _right_values)
    )


def _difference_at_parameter(_parameters, _value_rows, _parameter, _first, _second):
    """Evaluate one pairwise filtration-value difference."""

    if len(_parameters) == 1 or _parameter <= _parameters[0]:
        return _value_rows[0][_first] - _value_rows[0][_second]
    if _parameter >= _parameters[-1]:
        return _value_rows[-1][_first] - _value_rows[-1][_second]

    _segment = bisect_right(_parameters, _parameter) - 1
    _left_parameter = _parameters[_segment]
    _right_parameter = _parameters[_segment + 1]
    _fraction = (_parameter - _left_parameter) / (_right_parameter - _left_parameter)
    _left_difference = _value_rows[_segment][_first] - _value_rows[_segment][_second]
    _right_difference = _value_rows[_segment + 1][_first] - _value_rows[_segment + 1][_second]
    return (1.0 - _fraction) * _left_difference + _fraction * _right_difference


def _unique_parameters(_parameters, _tolerance):
    """Return sorted parameter values with numerical duplicates removed."""

    _unique = []
    for _parameter in sorted(_parameters):
        if not _unique or abs(_parameter - _unique[-1]) > _tolerance:
            _unique.append(_parameter)
    return tuple(_unique)


def _atomic_cells(_simplices, _parameters, _value_rows, _events, _atol):
    """Build point and open-interval cells at every critical parameter."""

    _event_parameters = tuple(_event["parameter"] for _event in _events)
    _tolerance = _parameter_tolerance(_parameters)
    _boundaries = _unique_parameters(
        (_parameters[0],) + _event_parameters + (_parameters[-1],),
        _tolerance
    )
    _atoms = []

    for _index, _parameter in enumerate(_boundaries):
        _values = _values_at_parameter(_parameters, _value_rows, _parameter)
        _atom = {}
        _atom["start"] = _parameter
        _atom["end"] = _parameter
        _atom["left_closed"] = True
        _atom["right_closed"] = True
        _atom["order_blocks"] = _order_blocks(_simplices, _values, _atol)
        _atoms.append(_atom)

        if _index == len(_boundaries) - 1:
            continue
        _next_parameter = _boundaries[_index + 1]
        _midpoint = 0.5 * (_parameter + _next_parameter)
        _values = _values_at_parameter(_parameters, _value_rows, _midpoint)
        _atom = {"start": _parameter, "end": _next_parameter,
                 "left_closed": False, "right_closed": False,
                 "order_blocks": _order_blocks(_simplices, _values, _atol)}
        _atoms.append(_atom)
    return tuple(_atoms)


def _merged_strata(_atoms, _parameters):
    """Merge adjacent atomic cells with the same minimal-order signature."""

    _regions = []
    for _atom in _atoms:
        if _regions and _regions[-1]["order_blocks"] == _atom["order_blocks"]:
            _regions[-1]["end"] = _atom["end"]
            _regions[-1]["right_closed"] = _atom["right_closed"]
            continue
        _regions.append(dict(_atom))

    _signature_ids = {}
    _strata = []
    _tolerance = _parameter_tolerance(_parameters)
    for _identifier, _region in enumerate(_regions):
        _signature = _region["order_blocks"]
        if _signature not in _signature_ids:
            _signature_ids[_signature] = len(_signature_ids)

        _start = _region["start"]
        _end = _region["end"]
        _is_point = abs(_end - _start) <= _tolerance
        _stratum = {"id": _identifier,
                    "kind": "point" if _is_point else "interval",
                    "start": _start, "end": _end,
                    "left_closed": _region["left_closed"],
                    "right_closed": _region["right_closed"],
                    "representative_parameter":
                        _start if _is_point else 0.5 * (_start + _end),
                    "signature_id": _signature_ids[_signature],
                    "order_blocks": _signature}
        _sample_indices = []
        for _index, _parameter in enumerate(_parameters):
            if (_parameter < _start - _tolerance or
                    _parameter > _end + _tolerance):
                continue
            if (abs(_parameter - _start) <= _tolerance and
                    not _region["left_closed"]):
                continue
            if (abs(_parameter - _end) <= _tolerance and
                    not _region["right_closed"]):
                continue
            _sample_indices.append(_index)
        _stratum["sample_indices"] = tuple(_sample_indices)
        _strata.append(_stratum)
    return tuple(_strata), len(_signature_ids)


def _stratum_at(_strata, _parameter, _tolerance):
    """Return the stratum identifier containing one parameter."""

    for _stratum in _strata:
        if _parameter < _stratum["start"] - _tolerance:
            continue
        if _parameter > _stratum["end"] + _tolerance:
            continue
        if (abs(_parameter - _stratum["start"]) <= _tolerance and
                not _stratum["left_closed"]):
            continue
        if (abs(_parameter - _stratum["end"]) <= _tolerance and
                not _stratum["right_closed"]):
            continue
        return _stratum["id"]
    return None


def _event_output(_events, _simplices, _parameters, _value_rows, _strata, _atol):
    """Attach event types and adjacent strata to equality parameters."""

    _output = []
    _critical = tuple(_event["parameter"] for _event in _events)
    _tolerance = _parameter_tolerance(_parameters)

    for _index, _event in enumerate(_events):
        _parameter = _event["parameter"]
        _left_boundary = _parameters[0] if _index == 0 else _critical[_index - 1]
        _right_boundary = _parameters[-1] if (
            _index == len(_events) - 1) else _critical[_index + 1]
        _left_probe = None
        _right_probe = None
        if _parameter > _parameters[0] + _tolerance:
            _left_probe = 0.5 * (_left_boundary + _parameter)
        if _parameter < _parameters[-1] - _tolerance:
            _right_probe = 0.5 * (_parameter + _right_boundary)

        _types = set()
        _pairs = []
        for _first, _second in _event["pair_indices"]:
            _pairs.append((_simplices[_first], _simplices[_second]))
            _left_sign = None
            _right_sign = None
            if _left_probe is not None:
                _difference = _difference_at_parameter(
                    _parameters,
                    _value_rows,
                    _left_probe,
                    _first,
                    _second
                )
                _left_sign = _sign(_difference, _atol)
            if _right_probe is not None:
                _difference = _difference_at_parameter(
                    _parameters,
                    _value_rows,
                    _right_probe,
                    _first,
                    _second
                )
                _right_sign = _sign(_difference, _atol)

            if (_left_sign is not None and _right_sign is not None and
                    _left_sign * _right_sign < 0):
                _types.add("order_crossing")
            elif _left_sign == 0 or _right_sign == 0:
                _types.add("persistent_tie")
            else:
                _types.add("order_tie")

        _record = {"parameter": _parameter,
                   "event_type": next(iter(_types)) if len(
                       _types) == 1 else "mixed",
                   "event_types": tuple(sorted(_types)),
                   "simplex_pairs": tuple(_pairs),
                   "left_stratum_id": None if _left_probe is None else _stratum_at(
                       _strata, _left_probe, _tolerance),
                   "event_stratum_id": _stratum_at(_strata, _parameter,
                                                   _tolerance),
                   "right_stratum_id": None if _right_probe is None else _stratum_at(
                       _strata, _right_probe, _tolerance)}
        _output.append(_record)
    return tuple(_output)


def _adjacency_records(_strata, _events):
    """Return lightweight incidence records between neighboring strata."""

    _records = []
    _seen = set()

    for _event_index, _event in enumerate(_events):
        _event_stratum = _event["event_stratum_id"]
        _incidences = (
            (_event["left_stratum_id"], _event_stratum, "left"),
            (_event_stratum, _event["right_stratum_id"], "right")
        )
        for _source, _target, _side in _incidences:
            if _source is None or _target is None or _source == _target:
                continue
            _key = (_source, _target, _event_index)
            if _key in _seen:
                continue
            _seen.add(_key)
            _record = {"id": len(_records), "source_stratum_id": _source,
                       "target_stratum_id": _target,
                       "relation": "event_incidence",
                       "event_index": _event_index,
                       "event_parameter": _event["parameter"],
                       "event_type": _event["event_type"],
                       "event_types": _event["event_types"],
                       "event_side": _side}
            _records.append(_record)

    _incident_pairs = {
        frozenset(
            (
                _record["source_stratum_id"],
                _record["target_stratum_id"]
            )
        )
        for _record in _records
    }
    _ordered = tuple(
        sorted(
            _strata,
            key=lambda _stratum: (
                _stratum["representative_parameter"],
                0 if _stratum["kind"] == "point" else 1,
                _stratum["id"]
            )
        )
    )
    if _ordered:
        _scale = max(
            1.0,
            abs(float(_ordered[0]["start"])),
            abs(float(_ordered[-1]["end"]))
        )
    else:
        _scale = 1.0
    _tolerance = 1e-12 * _scale
    for _left, _right in zip(_ordered, _ordered[1:]):
        if abs(float(_left["end"]) - float(_right["start"])) > _tolerance:
            continue
        _pair = frozenset((_left["id"], _right["id"]))
        if _pair in _incident_pairs:
            continue
        _record = {"id": len(_records), "source_stratum_id": _left["id"],
                   "target_stratum_id": _right["id"],
                   "relation": "stratum_incidence", "event_index": None,
                   "event_parameter": float(_left["end"]), "event_type": None,
                   "event_types": (()), "event_side": None}
        _records.append(_record)
        _incident_pairs.add(_pair)
    return tuple(_records)


def _stratification_record(_samples, _simplices,
                           _value_rows, _parameters,
                           _atol, _max_pairwise_comparisons,
                           _max_events,
                           _max_critical_parameters):
    """Compute a versioned record from validated samples on one complex.

    The current continuous implementation assumes a one-dimensional base and
    piecewise-linear interpolation of every simplex filtration value between
    supplied sample parameters.
    It returns maximal connected cells on which the compact
    minimal-consistent-order signature is constant.
    Point cells record isolated ties, while interval cells record stable strict
    orders or persistent ties.

    :param _samples: Validated filtrations on the same labeled complex.
    :type _samples: tuple of Filtration
    :param _simplices: Simplices of the common complex.
    :type _simplices: tuple
    :param _value_rows: Sample values aligned with the simplices.
    :type _value_rows: tuple
    :param _parameters: Strictly increasing sample parameters.
    :type _parameters: tuple of float
    :param _atol: Absolute tolerance for filtration-value ties.
    :type _atol: float
    :param _max_pairwise_comparisons: Safety cap for exact event detection.
    :type _max_pairwise_comparisons: int
    :param _max_events: Safety cap for simplex-pair equality records.
    :type _max_events: int
    :param _max_critical_parameters: Safety cap for distinct event times.
    :type _max_critical_parameters: int
    :return: Common complex, sampled values, order strata and equality events.
    :rtype: dict
    :raises RuntimeError: If an exact-enumeration safety cap is exceeded.
    """

    _atol = _nonnegative_float(_atol, "Absolute tolerance")
    _comparison_limit = _positive_integer(_max_pairwise_comparisons,
                                          "Maximum pairwise comparisons")
    _event_limit = _positive_integer(_max_events, "Maximum events")
    _critical_limit = _positive_integer(_max_critical_parameters,
                                        "Maximum critical parameters")
    _parameters = _normalize_parameters(len(_samples), _parameters)

    _comparisons = 0
    if len(_samples) > 1:
        _records, _comparisons = _critical_records(
            _simplices,
            _value_rows,
            _parameters,
            _atol,
            _comparison_limit,
            _event_limit
        )
        _event_groups = _coalesced_events(_records, _parameters)
        if len(_event_groups) > _critical_limit:
            _message = ("Exact stratification has {} critical parameters, "
                        "above the limit of {}")
            raise RuntimeError(_message.format(len(_event_groups), _critical_limit))
        _atoms = _atomic_cells(
            _simplices,
            _parameters,
            _value_rows,
            _event_groups,
            _atol
        )
        _strata, _signature_count = _merged_strata(_atoms, _parameters)
        _events = _event_output(
            _event_groups,
            _simplices,
            _parameters,
            _value_rows,
            _strata,
            _atol
        )
        _base_kind = "interval"
    else:
        _atoms = _atomic_cells(_simplices, _parameters, _value_rows, (), _atol)
        _strata, _signature_count = _merged_strata(_atoms, _parameters)
        _events = ()
        _base_kind = "point"

    _result = {"output_record_version": OUTPUT_RECORD_VERSION,
               "mathematical_scope": MATHEMATICAL_SCOPE,
               "tie_break_rule": TIE_BREAK_RULE,
               "entry_point": "filtration_family", "exact": False,
               "event_search_complete": True, "arithmetic": "floating_point",
               "tie_tolerance": _atol,
               "exactness_scope": "complete_piecewise_linear_event_search_within_tolerance",
               "base_kind": _base_kind, "parameters": _parameters,
               "complex_simplices": _simplices,
               "sample_filtration_values": _value_rows, "strata": _strata,
               "events": _events,
               "stratum_adjacency": _adjacency_records(_strata, _events),
               "signature_count": _signature_count,
               "pairwise_comparisons": _comparisons,
               "simplex_pair_events": sum(
                   len(_event["simplex_pairs"]) for _event in _events)}
    # Complete event search within tolerance is not exact arithmetic.
    return _result


class Stratification:
    """Order strata for a one-parameter family on one fixed complex K.

    Simplex values vary piecewise linearly between sampled parameters.
    Events are detected with floating-point arithmetic and the supplied
    absolute tolerance. The input filtrations and result are immutable.

    :param _filtrations: One or more filtrations on the same labeled complex.
    :type _filtrations: iterable of Filtration or gudhi.SimplexTree
    :param _parameters: Strictly increasing finite sample parameters.
    :type _parameters: iterable of real or None
    :param _atol: Absolute tolerance for equality comparisons.
    :type _atol: float
    :param _max_pairwise_comparisons: Limit on simplex-pair comparisons.
    :type _max_pairwise_comparisons: int
    :param _max_events: Limit on individual simplex-pair event records.
    :type _max_events: int
    :param _max_critical_parameters: Limit on distinct event parameters.
    :type _max_critical_parameters: int
    :raises ValueError: If sample data, filtrations or limits are invalid.
    :raises RuntimeError: If an event enumeration limit is exceeded.
    """
    # pylint: disable=no-member

    __slots__ = ("_complex", "_filtrations", "_record")

    def __init__(self, _filtrations, _parameters=None, _atol=1e-12,
                 _max_pairwise_comparisons=5000000,
                 _max_events=100000, _max_critical_parameters=5000):
        if hasattr(self, "_record"):
            raise AttributeError("Stratification is immutable")
        _normalized, _simplices, _value_rows = _fixed_family_data(_filtrations)
        _record = _stratification_record(
            _normalized, _simplices, _value_rows, _parameters, _atol,
            _max_pairwise_comparisons, _max_events,
            _max_critical_parameters
        )
        object.__setattr__(self, "_complex", _normalized[0].complex)
        object.__setattr__(self, "_filtrations", _normalized)
        object.__setattr__(self, "_record", _record)

    def __setattr__(self, _name, _value):
        raise AttributeError("Stratification is immutable")

    def __delattr__(self, _name):
        raise AttributeError("Stratification is immutable")

    @property
    def complex(self):
        """Return the shared fixed labeled simplicial complex K."""
        return self._complex

    @property
    def filtrations(self):
        """Return immutable filtrations at the sampled parameters."""
        return self._filtrations

    @property
    def parameters(self):
        """Return sampled parameters in increasing order."""
        return self._record["parameters"]

    @property
    def strata(self):
        """Return detached records for all parameter strata."""
        return deepcopy(self._record["strata"])

    @property
    def events(self):
        """Return detached records for all reported equality events."""
        return deepcopy(self._record["events"])

    def to_record(self):
        """Return an independent copy of the versioned result record.

        :return: Stratification data for downstream canopy modules.
        :rtype: dict
        """
        return deepcopy(self._record)

    def filtration_at(self, _parameter):
        """Return a validated filtration at a modeled parameter.

        :param _parameter: Parameter in the modeled interval.
        :type _parameter: real
        :return: Filtration on the same fixed labeled complex.
        :rtype: Filtration
        :raises ValueError: If the parameter is outside the modeled base.
        """
        return _filtration_at(self, _parameter)

    def materialize_stratum(self, _stratum_id):
        """Reconstruct a representative filtration and its order data.

        :param _stratum_id: Identifier of an existing stratum.
        :type _stratum_id: int
        :return: Representative data with a public Filtration object.
        :rtype: dict
        """
        _result = materialize_stratum(self, _stratum_id)
        _result["filtration"] = Filtration(
            self._complex, _result["filtration_values"]
        )
        return _result

    def adjacency(self):
        """Return the graph of incident strata.

        :return: Adjacency graph with incidence records.
        :rtype: dict
        """
        return stratum_adjacency(self)


def stratify_filtration_family(_filtrations, _parameters=None, _atol=1e-12,
                               _max_pairwise_comparisons=5000000,
                               _max_events=100000, _max_critical_parameters=5000):
    """Return a record for filtrations on the same fixed complex K.

    This function is a record-producing adapter to :class:`Stratification`.

    :param _filtrations: Sampled filtrations on one labeled complex.
    :type _filtrations: iterable of Filtration or gudhi.SimplexTree
    :param _parameters: Strictly increasing sample parameters.
    :type _parameters: iterable of real or None
    :param _atol: Absolute tolerance for filtration-value ties.
    :type _atol: float
    :param _max_pairwise_comparisons: Event-search comparison limit.
    :type _max_pairwise_comparisons: int
    :param _max_events: Simplex-pair event limit.
    :type _max_events: int
    :param _max_critical_parameters: Distinct event parameter limit.
    :type _max_critical_parameters: int
    :return: Independent fixed-complex stratification record.
    :rtype: dict
    """

    return Stratification(
        _filtrations, _parameters, _atol, _max_pairwise_comparisons,
        _max_events, _max_critical_parameters
    ).to_record()


def _fixed_stratification_data(_stratification):
    """Return validated data needed to reconstruct representative fibers."""

    if isinstance(_stratification, Stratification):
        _stratification = _stratification._record
    if not hasattr(_stratification, "__getitem__"):
        raise TypeError("Stratification must be a mapping")
    try:
        _parameters = tuple(_stratification["parameters"])
        _simplices = tuple(_stratification["complex_simplices"])
        _value_rows = tuple(
            tuple(_row)
            for _row in _stratification["sample_filtration_values"]
        )
        _strata = tuple(_stratification["strata"])
    except KeyError as _error:
        raise ValueError(
            "Stratification is missing {}".format(_error.args[0])
        ) from _error
    except TypeError as _error:
        raise ValueError("Stratification data must be iterable") from _error

    if _stratification.get("mathematical_scope") != MATHEMATICAL_SCOPE:
        raise ValueError("Materialization requires a fixed-complex "
                         "stratification")
    if not _parameters or not _simplices:
        raise ValueError("Stratification has no parameter or "
                         "common-complex data")
    if len(_parameters) != len(_value_rows):
        raise ValueError("Sample parameters and filtration "
                         "rows must have equal length")
    if any(len(_row) != len(_simplices) for _row in _value_rows):
        raise ValueError("Every filtration row must cover the fixed complex")
    return _parameters, _simplices, _value_rows, _strata


def _filtration_at(_stratification, _parameter):
    """Evaluate the family on its fixed complex at one parameter."""

    _parameters, _simplices, _value_rows, _strata = _fixed_stratification_data(
        _stratification
    )
    _parameter = _finite_float(_parameter, "Evaluation parameter")
    _tolerance = _parameter_tolerance(_parameters)
    if (_parameter < _parameters[0] - _tolerance or
            _parameter > _parameters[-1] + _tolerance):
        raise ValueError("Evaluation parameter is outside the modeled interval")
    _parameter = min(max(_parameter, _parameters[0]), _parameters[-1])
    _values = _values_at_parameter(_parameters, _value_rows, _parameter)
    _complex = (
        _stratification.complex
        if isinstance(_stratification, Stratification)
        else SimplicialComplex(_simplices)
    )
    return Filtration(_complex, dict(zip(_simplices, _values)))


def materialize_filtration(_stratification, _parameter):
    """Build an independent GUDHI tree from a sampled parameter family.

    :param _stratification: Fixed-complex stratification result.
    :type _stratification: Stratification or mapping
    :param _parameter: Parameter at which to reconstruct the filtration.
    :type _parameter: real
    :return: Filtered copy of the fixed underlying complex.
    :rtype: gudhi.SimplexTree
    :raises ValueError: If the parameter is outside the modeled base.
    """

    return _filtration_at(_stratification, _parameter).to_simplex_tree()


def materialize_stratum(_stratification, _stratum_id):
    """Materialize the canonical representative of one stratum.

    The returned record contains a GUDHI filtration, its compact equality
    blocks and the deterministic total refinement. Pairing and persistence
    diagrams belong to the downstream pairing stage.

    :param _stratification: Fixed-complex stratification result.
    :type _stratification: Stratification or mapping
    :param _stratum_id: Identifier stored in ``stratification["strata"]``.
    :type _stratum_id: int
    :return: Representative filtration and order data.
    :rtype: dict
    """

    _stratum_id = _nonnegative_integer(_stratum_id, "Stratum identifier")
    _parameters, _simplices, _value_rows, _strata = _fixed_stratification_data(
        _stratification
    )
    _matches = tuple(
        _stratum
        for _stratum in _strata
        if _stratum["id"] == _stratum_id
    )
    if len(_matches) != 1:
        raise ValueError("Unknown stratum identifier {}".format(_stratum_id))

    _stratum = _matches[0]
    _parameter = _stratum["representative_parameter"]
    _filtration = materialize_filtration(_stratification, _parameter)
    _order = canonical_refinement(_filtration)
    _result = dict(_stratum)
    _result["stratum_id"] = _stratum["id"]
    _result["filtration"] = _filtration
    _result["filtration_values"] = filtration_values(_filtration)
    _result["equality_blocks"] = tuple(
        _block
        for _block in _stratum["order_blocks"]
        if len(_block) > 1
    )
    _result["canonical_order"] = _order
    return _result


def stratum_adjacency(_stratification):
    """Return the explicit graph of neighboring strata.

    Endpoint events determine which interval and point strata are incident.

    :param _stratification: Stratification on one fixed complex K.
    :type _stratification: Stratification or mapping
    :return: Nodes and their incidence records.
    :rtype: dict
    """

    if isinstance(_stratification, Stratification):
        _stratification = _stratification._record
    _parameters, _simplices, _value_rows, _strata = _fixed_stratification_data(
        _stratification
    )
    _raw_edges = _stratification["stratum_adjacency"]

    _node_ids = set(_stratum["id"] for _stratum in _strata)
    _nodes = []
    for _stratum in _strata:
        _node = {"id": _stratum["id"], "kind": _stratum["kind"],
                 "start": _stratum["start"], "end": _stratum["end"],
                 "representative_parameter": _stratum[
                     "representative_parameter"],
                 "signature_id": _stratum["signature_id"]}
        _nodes.append(_node)

    _edges = []
    for _raw_edge in _raw_edges:
        _edge = dict(_raw_edge)
        _source = _edge["source_stratum_id"]
        _target = _edge["target_stratum_id"]
        if _source not in _node_ids or _target not in _node_ids:
            raise ValueError("Adjacency edge refers to an unknown stratum")

        _edges.append(_edge)

    _result = {"nodes": tuple(_nodes), "edges": tuple(_edges)}
    return _result
