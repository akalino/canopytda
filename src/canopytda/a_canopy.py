r"""Finite constructible presentations of augmented canopies.

The paper defines an A-canopy as a topological bundle-like object
``(E, B, pi, N, eta)``.  For the one-dimensional piecewise-linear model used
by this package, the infinite total space can be stored by finite data:

* one chosen-pair fiber over every base stratum;
* self-continuation generators within each representative fiber;
* every combinatorial-equivalence map across each stratum incidence; and
* the value of ``eta`` on a fiber at any requested base parameter.

This module builds that finite presentation for the paper's fixed-complex
setting.  It deliberately does not turn the incidence graph into a geometric
realization, because an ordinary graph realization would erase the
non-Hausdorff behavior that canopies are designed to retain.  Basis
neighborhoods and separation queries belong in ``topology.py``.
"""

from collections.abc import Mapping
from copy import deepcopy
import math

from .equivalence import all_combinatorial_equivalences
from .contract import MATHEMATICAL_SCOPE
from .contract import OUTPUT_RECORD_VERSION
from .contract import TIE_BREAK_RULE
from .orders import canonical_refinement
from .pairing import STAR
from .pairing import PersistencePairing
from .structurality import Structurality
from .utils import _nonnegative_integer


__all__ = ["ACanopy", "a_canopy_diagram", "a_canopy_fiber"]
__all__ += ["adjacent_continuations", "build_a_canopy"]
__all__ += ["validate_a_canopy"]


def _normalize_dimension(_homology_dimension):
    """Return a nonnegative homology dimension or ``None``."""

    if _homology_dimension is None:
        return None
    return _nonnegative_integer(_homology_dimension, "Homology dimension")


def _selected_pairing(_pairing, _homology_dimension):
    """Return the chosen pairs represented by one canopy."""

    if _homology_dimension is None:
        return tuple(_pairing)
    return tuple(
        _pair
        for _pair in _pairing
        if len(_pair[0]) - 1 == _homology_dimension
    )


def _eta_value(_values, _pair):
    """Evaluate the paper's map ``eta`` on one simplex pair."""

    _birth, _death = _pair
    _death_value = math.inf
    if _death != STAR:
        _death_value = float(_values[_death])
    return float(_values[_birth]), _death_value


def _diagram_from_points(_points):
    """Return a dimension-indexed augmented multiset from fiber points."""

    _by_dimension = {}
    for _point in _points:
        _dimension = _point["homology_dimension"]
        _by_dimension.setdefault(_dimension, []).append(_point["eta"])
    return dict(
        (_dimension, tuple(sorted(_by_dimension[_dimension])))
        for _dimension in sorted(_by_dimension)
    )


def _stratification_input(_analysis):
    """Validate scope and return the stratum graph from Structurality."""

    _stratification = _analysis.stratification.to_record()
    if _stratification["mathematical_scope"] != MATHEMATICAL_SCOPE:
        raise ValueError("A-canopy construction requires a fixed finite simplicial complex K")
    if not _stratification.get("event_search_complete", False):
        raise ValueError("A-canopy construction requires a complete stratum search")

    _base_kind = _stratification.get("base_kind")
    if _base_kind not in {"interval", "point"}:
        raise ValueError("A-canopy construction requires a modeled continuous base")

    try:
        _strata = tuple(_stratification["strata"])
        _parameters = tuple(float(_value) for _value in _stratification["parameters"])
    except KeyError as _error:
        raise ValueError(
            "Stratification is missing {}".format(_error.args[0])
        ) from _error
    except TypeError as _error:
        raise ValueError("Stratification base data must be iterable") from _error
    if not _strata or not _parameters:
        raise ValueError("Stratification must contain a nonempty base")

    _identifiers = tuple(_stratum.get("id") for _stratum in _strata)
    if len(set(_identifiers)) != len(_identifiers):
        raise ValueError("Stratum identifiers must be unique")
    _graph = _analysis.verified_adjacency()
    if not _graph["all_edges_compatible"]:
        raise ValueError("Every stratum incidence must join compatible filtrations")
    return _strata, _parameters, _graph


def _fiber_points(_materialized, _pairing):
    """Build finite sheet labels over one representative stratum point."""

    _stratum_id = _materialized["stratum_id"]
    _parameter = float(_materialized["representative_parameter"])
    _values = _materialized["filtration_values"]
    _points = []
    for _pair_index, _pair in enumerate(_pairing):
        _point = {}
        _point["sheet_id"] = (_stratum_id, _pair_index)
        _point["stratum_id"] = _stratum_id
        _point["pair_index"] = _pair_index
        _point["pair"] = _pair
        _point["homology_dimension"] = len(_pair[0]) - 1
        _point["representative_parameter"] = _parameter
        _point["eta"] = _eta_value(_values, _pair)
        _points.append(_point)
    return tuple(_points)


def _projected_continuations(_witnesses, _source_points, _target_points):
    """Project complete pairing maps to stored canopy sheet identifiers."""

    _source_by_pair = dict(
        (_point["pair"], _point["sheet_id"])
        for _point in _source_points
    )
    _target_by_pair = dict(
        (_point["pair"], _point["sheet_id"])
        for _point in _target_points
    )
    _source_pairs = set(_source_by_pair)
    _target_pairs = set(_target_by_pair)
    _records = []
    _seen = set()

    for _witness in _witnesses:
        _pair_mapping = tuple(
            (_source_pair, _target_pair)
            for _source_pair, _target_pair in _witness["mapping"]
            if _source_pair in _source_pairs
        )
        if set(_source for _source, _target in _pair_mapping) != _source_pairs:
            raise RuntimeError("A continuation does not cover its source fiber")
        if set(_target for _source, _target in _pair_mapping) != _target_pairs:
            raise RuntimeError("A continuation does not cover its target fiber")

        _mapping = tuple(
            (_source_by_pair[_source], _target_by_pair[_target])
            for _source, _target in _pair_mapping
        )
        if _mapping in _seen:
            continue
        _seen.add(_mapping)
        _record = {}
        _record["id"] = len(_records)
        _record["mapping"] = _mapping
        _record["pair_mapping"] = _pair_mapping
        _record["witness"] = _witness
        _records.append(_record)
    if not _records:
        raise RuntimeError("Compatible strata produced no stored continuation")
    return tuple(_records)


def _representative_fibers(_stratification, _strata, _homology_dimension, _max_common_refinements, _max_refinements, _max_equivalences):
    """Materialize every stratum fiber and its self-continuations."""

    _fibers = []
    _fiber_cardinality = None

    for _stratum in _strata:
        _stratum_id = _stratum["id"]
        _filtration = _stratification.filtration_at(
            _stratum["representative_parameter"])
        _order = canonical_refinement(_filtration)
        _data = dict(_stratum)
        _data["stratum_id"] = _stratum_id
        _data["filtration"] = _filtration
        _data["filtration_values"] = dict(_filtration.values)
        _data["canonical_order"] = _order
        _data["chosen_pairing"] = PersistencePairing(
            _filtration, _order).pairs
        _pairing = _selected_pairing(
            _data["chosen_pairing"],
            _homology_dimension
        )
        if _fiber_cardinality is None:
            _fiber_cardinality = len(_pairing)
        elif len(_pairing) != _fiber_cardinality:
            raise RuntimeError("Chosen-pair fiber cardinality is not constant")

        _points = _fiber_points(_data, _pairing)
        _self_result = all_combinatorial_equivalences(
            _data["filtration"],
            _data["filtration"],
            _left_order=_data["canonical_order"],
            _right_order=_data["canonical_order"],
            _max_common_refinements=_max_common_refinements,
            _max_refinements=_max_refinements,
            _max_equivalences=_max_equivalences
        )
        if not _self_result["complete"]:
            raise RuntimeError(
                "A-canopy construction requires complete self-continuation enumeration"
            )
        _self_continuations = _projected_continuations(
            _self_result["equivalences"],
            _points,
            _points
        )

        _fiber = {}
        _fiber["stratum_id"] = _stratum_id
        _fiber["kind"] = _stratum["kind"]
        _fiber["representative_parameter"] = _data["representative_parameter"]
        _fiber["pairing"] = _pairing
        _fiber["points"] = _points
        _fiber["augmented_diagram"] = _diagram_from_points(_points)
        _fiber["self_continuations"] = _self_continuations
        _fiber["self_continuation_complete"] = True
        _fiber["self_continuation_search"] = {}
        _fiber["self_continuation_search"]["common_refinement_count"] = _self_result["common_refinement_count"]
        _fiber["self_continuation_search"]["common_pairing_count"] = _self_result["common_pairing_count"]
        _fiber["self_continuation_search"]["refinements_examined"] = _self_result["refinements_examined"]
        _fiber["self_continuation_search"]["search_scope"] = _self_result["search_scope"]
        _fibers.append(_fiber)

    if _fiber_cardinality is None:
        _fiber_cardinality = 0
    return tuple(_fibers), _fiber_cardinality


def _incidence_continuations(_analysis, _fibers):
    """Build all pairing continuations across every adjacent stratum pair."""

    _fiber_by_id = dict(
        (_fiber["stratum_id"], _fiber)
        for _fiber in _fibers
    )
    _incidences = []
    for _edge in _analysis.graph["edges"]:
        _source_id = _edge["source_stratum_id"]
        _target_id = _edge["target_stratum_id"]
        _result = _analysis.adjacent_equivalences(
            _source_id, _target_id)
        if not _result["complete"]:
            raise RuntimeError(
                "A-canopy construction requires every adjacent continuation"
            )
        _continuations = _projected_continuations(
            _result["equivalences"],
            _fiber_by_id[_source_id]["points"],
            _fiber_by_id[_target_id]["points"]
        )

        _incidence = dict(_edge)
        _incidence["incidence_id"] = _edge["id"]
        _incidence["continuations"] = _continuations
        _incidence["continuation_count"] = len(_continuations)
        _incidence["complete"] = True
        _incidence["search_scope"] = _result["search_scope"]
        _incidence["common_refinement_count"] = _result["common_refinement_count"]
        _incidence["common_pairing_count"] = _result["common_pairing_count"]
        _incidence["refinements_examined"] = _result["refinements_examined"]
        _incidences.append(_incidence)
    return tuple(_incidences)


def _base_record(_stratification, _strata, _parameters, _graph):
    """Return the finite stratified representation of the base space."""

    _base = {}
    _base["kind"] = "stratified_interval" if len(_parameters) > 1 else "singleton"
    _base["domain"] = (_parameters[0], _parameters[-1])
    _base["parameters"] = _parameters
    _base["strata"] = tuple(dict(_stratum) for _stratum in _strata)
    _base["adjacency"] = tuple(dict(_edge) for _edge in _graph["edges"])
    return _base


def build_a_canopy(_stratification, _homology_dimension=None, _max_common_refinements=5000, _max_refinements=5000, _max_equivalences=100):
    """Construct a finite A-canopy presentation from a stratified family.

    The construction implements the finite one-parameter form of Definitions
    3.1--3.6 and Theorem 3.8.  It stores a chosen-pair sheet over every base
    stratum and every certified local continuation, including multiple maps
    at a structural event.  Enumeration bounds are certificates: if any
    search is incomplete, construction raises rather than selecting a branch.

    ``None`` for ``_homology_dimension`` builds the graded disjoint union of
    all homological dimensions.  Supplying an integer builds the corresponding
    p-dimensional A-canopy from Definition 2.12.

    :param _stratification: Complete fixed-complex linear stratification.
    :type _stratification: mapping
    :param _homology_dimension: Optional homological dimension to retain.
    :type _homology_dimension: int or None
    :param _max_common_refinements: Maximum common orders per local search.
    :type _max_common_refinements: int
    :param _max_refinements: Maximum support-order search per filtration.
    :type _max_refinements: int
    :param _max_equivalences: Maximum distinct maps per local search.
    :type _max_equivalences: int
    :return: Finite constructible A-canopy presentation.
    :rtype: dict
    """

    _homology_dimension = _normalize_dimension(_homology_dimension)
    _analysis = Structurality(
        _stratification, _max_refinements, _max_equivalences,
        _max_common_refinements)
    _stratification = _analysis.stratification
    _strata, _parameters, _graph = _stratification_input(_analysis)
    _fibers, _fiber_cardinality = _representative_fibers(
        _stratification,
        _strata,
        _homology_dimension,
        _max_common_refinements,
        _max_refinements,
        _max_equivalences
    )
    _incidences = _incidence_continuations(_analysis, _fibers)
    _structurality = _analysis.classify_events(_homology_dimension)
    _projection = []
    for _fiber in _fibers:
        for _point in _fiber["points"]:
            _projection.append(
                (_point["sheet_id"], _fiber["stratum_id"])
            )

    _result = {}
    _result["output_record_version"] = OUTPUT_RECORD_VERSION
    _result["mathematical_scope"] = MATHEMATICAL_SCOPE
    _result["tie_break_rule"] = TIE_BREAK_RULE
    _result["object_type"] = "A-canopy"
    _result["paper_definition"] = "Definitions 3.1-3.6 and Theorem 3.8"
    _result["representation"] = "finite_constructible_one_parameter"
    _result["homology_dimension"] = _homology_dimension
    _result["base"] = _base_record(
        _stratification,
        _strata,
        _parameters,
        _graph
    )
    _result["complex_simplices"] = _stratification.complex.simplices
    _result["fiber_cardinality"] = _fiber_cardinality
    _result["representative_fibers"] = _fibers
    _result["projection"] = tuple(_projection)
    _result["incidences"] = _incidences
    _result["structurality"] = _structurality
    _result["structural_parameters"] = _structurality["structural_parameters"]
    _result["structurality_scope"] = _structurality["certificate_scope"]
    _result["complete"] = True
    _result["continuation_search_scope"] = "all_common_total_refinements"
    _result["topology_materialized"] = False
    _result["stratification"] = _stratification.to_record()
    validate_a_canopy(_result)
    return _result


def _canopy_data(_canopy):
    """Return core records from a candidate A-canopy mapping."""

    if not hasattr(_canopy, "__getitem__"):
        raise TypeError("A-canopy must be a mapping")
    try:
        _object_type = _canopy["object_type"]
        _fiber_cardinality = _canopy["fiber_cardinality"]
        _fibers = tuple(_canopy["representative_fibers"])
        _incidences = tuple(_canopy["incidences"])
        _projection = tuple(_canopy["projection"])
    except KeyError as _error:
        raise ValueError(
            "A-canopy is missing {}".format(_error.args[0])
        ) from _error
    except TypeError as _error:
        raise ValueError("A-canopy records must be iterable") from _error
    if _object_type != "A-canopy":
        raise ValueError("Object is not an A-canopy presentation")
    _fiber_cardinality = _nonnegative_integer(
        _fiber_cardinality, "Fiber cardinality", _error_type=ValueError)
    return _fiber_cardinality, _fibers, _incidences, _projection


def _validate_continuation(_continuation, _source_ids, _target_ids):
    """Validate one stored bijection between finite sheet fibers."""

    try:
        _mapping = tuple(_continuation["mapping"])
    except (KeyError, TypeError) as _error:
        raise ValueError("Continuation is missing a mapping") from _error
    _sources = tuple(_source for _source, _target in _mapping)
    _targets = tuple(_target for _source, _target in _mapping)
    if len(set(_sources)) != len(_sources):
        raise ValueError("Continuation repeats a source sheet")
    if len(set(_targets)) != len(_targets):
        raise ValueError("Continuation repeats a target sheet")
    if set(_sources) != set(_source_ids):
        raise ValueError("Continuation does not cover its source fiber")
    if set(_targets) != set(_target_ids):
        raise ValueError("Continuation does not cover its target fiber")


def validate_a_canopy(_canopy):
    """Validate the finite invariants of an A-canopy presentation.

    This checks constant fiber cardinality, projection data, augmented-diagram
    multiplicity and the bijectivity of every stored continuation.  It does
    not manufacture the topology or replace the later basis-neighborhood
    checks in ``topology.py``.

    :param _canopy: Candidate finite A-canopy presentation.
    :type _canopy: mapping
    :return: The validated input mapping.
    :rtype: mapping
    :raises ValueError: If a finite canopy invariant fails.
    """

    _fiber_cardinality, _fibers, _incidences, _projection = _canopy_data(
        _canopy
    )
    _fiber_by_id = {}
    _expected_projection = []
    _all_sheet_ids = set()
    for _fiber in _fibers:
        try:
            _stratum_id = _fiber["stratum_id"]
            _points = tuple(_fiber["points"])
            _pairing = tuple(_fiber["pairing"])
            _self_continuations = tuple(_fiber["self_continuations"])
        except (KeyError, TypeError) as _error:
            raise ValueError("Fiber record is incomplete") from _error
        if _stratum_id in _fiber_by_id:
            raise ValueError("A-canopy repeats a stratum fiber")
        if len(_points) != _fiber_cardinality:
            raise ValueError("A-canopy fiber cardinality is not constant")
        if tuple(_point["pair"] for _point in _points) != _pairing:
            raise ValueError("Fiber points and chosen pairing disagree")
        if _diagram_from_points(_points) != _fiber["augmented_diagram"]:
            raise ValueError("Fiber eta image does not equal its augmented diagram")

        _sheet_ids = tuple(_point["sheet_id"] for _point in _points)
        if len(set(_sheet_ids)) != len(_sheet_ids):
            raise ValueError("A fiber repeats a sheet identifier")
        if _all_sheet_ids.intersection(_sheet_ids):
            raise ValueError("Sheet identifiers must be globally unique")
        _all_sheet_ids.update(_sheet_ids)
        for _sheet_id in _sheet_ids:
            _expected_projection.append((_sheet_id, _stratum_id))
        for _continuation in _self_continuations:
            _validate_continuation(
                _continuation,
                _sheet_ids,
                _sheet_ids
            )
        _fiber_by_id[_stratum_id] = _fiber

    if tuple(_expected_projection) != _projection:
        raise ValueError("Stored projection does not match the sheet fibers")

    _seen_edges = set()
    for _incidence in _incidences:
        try:
            _source_id = _incidence["source_stratum_id"]
            _target_id = _incidence["target_stratum_id"]
            _continuations = tuple(_incidence["continuations"])
        except (KeyError, TypeError) as _error:
            raise ValueError("Incidence record is incomplete") from _error
        if _source_id not in _fiber_by_id or _target_id not in _fiber_by_id:
            raise ValueError("Incidence refers to an unknown stratum")
        _edge = frozenset((_source_id, _target_id))
        if _edge in _seen_edges:
            raise ValueError("A-canopy repeats a stratum incidence")
        _seen_edges.add(_edge)
        if not _continuations:
            raise ValueError("Every incidence must retain a continuation")
        _source_ids = tuple(
            _point["sheet_id"]
            for _point in _fiber_by_id[_source_id]["points"]
        )
        _target_ids = tuple(
            _point["sheet_id"]
            for _point in _fiber_by_id[_target_id]["points"]
        )
        for _continuation in _continuations:
            _validate_continuation(
                _continuation,
                _source_ids,
                _target_ids
            )
    return _canopy


def _parameter_tolerance(_parameters):
    """Return the stratification's scale-aware base tolerance."""

    _scale = max(1.0, abs(_parameters[0]), abs(_parameters[-1]))
    _scale = max(_scale, _parameters[-1] - _parameters[0])
    return 1e-12 * _scale


def _stratum_at_parameter(_canopy, _parameter):
    """Return the unique stored stratum containing a base parameter."""

    try:
        _parameter = float(_parameter)
    except (TypeError, ValueError) as _error:
        raise ValueError("Base parameter must be a real number") from _error
    if not math.isfinite(_parameter):
        raise ValueError("Base parameter must be finite")

    _base = _canopy["base"]
    _parameters = tuple(_base["parameters"])
    _tolerance = _parameter_tolerance(_parameters)
    _candidates = []
    for _stratum in _base["strata"]:
        if _parameter < _stratum["start"] - _tolerance:
            continue
        if _parameter > _stratum["end"] + _tolerance:
            continue
        if abs(_parameter - _stratum["start"]) <= _tolerance and not _stratum["left_closed"]:
            continue
        if abs(_parameter - _stratum["end"]) <= _tolerance and not _stratum["right_closed"]:
            continue
        _candidates.append(_stratum)
    if len(_candidates) != 1:
        raise ValueError("Base parameter does not lie in exactly one canopy stratum")
    return _parameter, _candidates[0]


def a_canopy_fiber(_canopy, _parameter):
    """Evaluate the chosen-pair fiber and ``eta`` at one base parameter.

    :param _canopy: Finite A-canopy from :func:`build_a_canopy`.
    :type _canopy: mapping
    :param _parameter: Point of the modeled interval base.
    :type _parameter: real
    :return: Canopy points with projection, pair labels and eta values.
    :rtype: tuple of dict
    """

    validate_a_canopy(_canopy)
    _parameter, _stratum = _stratum_at_parameter(_canopy, _parameter)
    _stratum_id = _stratum["id"]
    _fiber = next(
        _item
        for _item in _canopy["representative_fibers"]
        if _item["stratum_id"] == _stratum_id
    )
    if isinstance(_canopy, ACanopy):
        _filtration = _canopy.stratification.filtration_at(_parameter)
    else:
        _filtration = Structurality(
            _canopy["stratification"]).stratification.filtration_at(
                _parameter)
    _values = _filtration.values
    _points = []
    for _sheet in _fiber["points"]:
        _point = dict(_sheet)
        _point["parameter"] = _parameter
        _point["projection"] = _parameter
        _point["canopy_point"] = (
            _parameter,
            _sheet["pair"][0],
            _sheet["pair"][1]
        )
        _point["eta"] = _eta_value(_values, _sheet["pair"])
        _points.append(_point)
    return tuple(_points)


def a_canopy_diagram(_canopy, _parameter):
    """Return the augmented diagram induced by one canopy fiber.

    Tuple entries retain multiplicity, including points on the diagonal and
    essential points with infinite death coordinate.

    :param _canopy: Finite A-canopy from :func:`build_a_canopy`.
    :type _canopy: mapping
    :param _parameter: Point of the modeled interval base.
    :type _parameter: real
    :return: Augmented diagram keyed by homological dimension.
    :rtype: dict
    """

    return _diagram_from_points(a_canopy_fiber(_canopy, _parameter))


def adjacent_continuations(_canopy, _source_stratum_id, _target_stratum_id):
    """Return all stored continuations across one stratum incidence.

    The query may reverse the stored orientation.  In that case every sheet
    bijection and simplex-pair map is inverted without choosing one branch.

    :param _canopy: Finite A-canopy from :func:`build_a_canopy`.
    :type _canopy: mapping
    :param _source_stratum_id: Source stratum identifier.
    :type _source_stratum_id: int
    :param _target_stratum_id: Adjacent target stratum identifier.
    :type _target_stratum_id: int
    :return: Complete tuple of directed continuation records.
    :rtype: tuple of dict
    """

    validate_a_canopy(_canopy)
    _source_stratum_id = _nonnegative_integer(
        _source_stratum_id, "Source stratum identifier")
    _target_stratum_id = _nonnegative_integer(
        _target_stratum_id, "Target stratum identifier")

    _matches = []
    for _incidence in _canopy["incidences"]:
        _stored_source = _incidence["source_stratum_id"]
        _stored_target = _incidence["target_stratum_id"]
        if {_stored_source, _stored_target} == {_source_stratum_id, _target_stratum_id}:
            _matches.append(_incidence)
    if len(_matches) != 1:
        raise ValueError("Strata must share exactly one canopy incidence")

    _incidence = _matches[0]
    if _incidence["source_stratum_id"] == _source_stratum_id:
        return tuple(dict(_record) for _record in _incidence["continuations"])

    _reversed = []
    for _record in _incidence["continuations"]:
        _inverse = dict(_record)
        _inverse["mapping"] = tuple(
            (_target, _source)
            for _source, _target in _record["mapping"]
        )
        _inverse["pair_mapping"] = tuple(
            (_target, _source)
            for _source, _target in _record["pair_mapping"]
        )
        _inverse["reversed"] = True
        _reversed.append(_inverse)
    return tuple(_reversed)


class ACanopy(Mapping):
    """Finite public A-canopy with pair-labeled fibers and continuations.

    The class stores the finite data for the paper's one-dimensional
    construction. ``fiber`` and ``diagram`` evaluate the projection and eta
    at any parameter in the modeled base. The separate topology module will
    supply trajectory neighborhoods and separation queries.

    :param _stratification: Fixed-complex piecewise-linear filtration family.
    :type _stratification: Stratification or its output record
    :param _homology_dimension: Optional homology degree, or all degrees.
    :type _homology_dimension: int or None
    :param _max_common_refinements: Exact common-order enumeration cap.
    :type _max_common_refinements: int
    :param _max_refinements: Exact cross-pair witness enumeration cap.
    :type _max_refinements: int
    :param _max_equivalences: Maximum allowed continuation maps.
    :type _max_equivalences: int
    """

    __slots__ = ("_stratification", "_record")

    def __init__(self, _stratification, _homology_dimension=None,
                 _max_common_refinements=5000, _max_refinements=5000,
                 _max_equivalences=100):
        if hasattr(self, "_record"):
            raise AttributeError("ACanopy is immutable")
        _analysis = Structurality(
            _stratification, _max_refinements, _max_equivalences,
            _max_common_refinements)
        _record = build_a_canopy(
            _analysis.stratification, _homology_dimension,
            _max_common_refinements, _max_refinements, _max_equivalences)
        object.__setattr__(self, "_stratification", _analysis.stratification)
        object.__setattr__(self, "_record", _record)

    def __setattr__(self, _name, _value):
        raise AttributeError("ACanopy is immutable")

    def __delattr__(self, _name):
        raise AttributeError("ACanopy is immutable")

    def __getitem__(self, _key):
        """Read a detached value from the finite canopy record."""
        return deepcopy(self._record[_key])

    def __iter__(self):
        """Iterate over finite-record field names."""
        return iter(self._record)

    def __len__(self):
        """Return the number of finite-record fields."""
        return len(self._record)

    @property
    def stratification(self):
        """Return the immutable parameter stratification."""
        return self._stratification

    @property
    def fiber_cardinality(self):
        """Return the number of simplex-pair points over every parameter."""
        return self._record["fiber_cardinality"]

    @property
    def structural_parameters(self):
        """Return parameters carrying certified structural points."""
        return self._record["structural_parameters"]

    def to_record(self):
        """Return an independent copy of the finite canopy presentation."""
        return deepcopy(self._record)

    def fiber(self, _parameter):
        """Evaluate pair-labeled canopy points over one base parameter."""
        return a_canopy_fiber(self, _parameter)

    def diagram(self, _parameter, _homology_dimension=None):
        """Return the augmented diagram, retaining diagonal multiplicity.

        With a degree selected, return its sorted point tuple. Otherwise,
        return a dictionary indexed by homology degree.
        """
        _diagram = a_canopy_diagram(self, _parameter)
        if _homology_dimension is None:
            return _diagram
        _dimension = _nonnegative_integer(
            _homology_dimension, "Homology dimension")
        return _diagram.get(_dimension, ())

    def continuations(self, _source_stratum_id, _target_stratum_id):
        """Return every directed map across the requested stratum edge."""
        return adjacent_continuations(
            self, _source_stratum_id, _target_stratum_id)
