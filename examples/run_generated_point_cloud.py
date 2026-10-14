"""Run full-cloud Rips persistence and an exact 100-point canopy smoke test.

The shell wrapper ``run_point_cloud_pipeline.sh`` creates the input. A second
cloud scales every coordinate, retaining the same vertex IDs. Exact canopies
use a declared, tiny rank-based offset for each vertex birth; standard Rips
assigns all vertices zero and exceeds the current 5000 common-order search
cap on this input. Edge and higher-simplex values stay equal to Rips values.
Ordinary Rips diagrams are also reported separately on the same fixed K.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from canopytda.a_canopy import ACanopy
from canopytda.complex import SimplicialComplex
from canopytda.d_canopy import DCanopy
from canopytda.filtration import Filtration
from canopytda.pairing import PersistencePairing
from canopytda.point_cloud import PointCloud
from canopytda.stratification import Stratification
from canopytda.topology import CanopyTopology
from canopytda.vines import VineDecomposition


SCALE_FACTOR = 1.01
EDGE_QUANTILE = 0.005
MAX_SIMPLICES = 1500


def _edge_distances(_points):
    """Return the unique Euclidean edge lengths of a finite point matrix.

    :param _points: Point coordinates, one point per row.
    :type _points: numpy.ndarray
    :return: Lengths of all unordered point pairs.
    :rtype: numpy.ndarray
    """
    _rows, _columns = np.triu_indices(len(_points), 1)
    return np.linalg.norm(_points[_rows] - _points[_columns], axis=1)


def _serialize_diagram(_diagram):
    """Represent essential deaths without nonstandard JSON infinity values."""
    return [[_birth, "inf" if math.isinf(_death) else _death]
            for _birth, _death in _diagram]


def _pairing_record(_filtration):
    """Return complete H0 and H1 multisets of a filtered fixed complex."""
    _pairing = PersistencePairing(_filtration)
    return {
        "h0_augmented": _serialize_diagram(_pairing.augmented_diagram(0)),
        "h0_diminished": _serialize_diagram(_pairing.diminished_diagram(0)),
        "h1_augmented": _serialize_diagram(_pairing.augmented_diagram(1)),
        "h1_diminished": _serialize_diagram(_pairing.diminished_diagram(1)),
    }


def _separated_vertex_filtration(_cloud, _complex, _step, _ranks):
    """Separate tied vertex births without changing Rips simplex lengths.

    :param _cloud: One labeled point cloud.
    :type _cloud: PointCloud
    :param _complex: Fixed complex shared by both samples.
    :type _complex: SimplicialComplex
    :param _step: Small positive offset between consecutive vertex births.
    :type _step: float
    :param _ranks: Rank of each vertex ID, independent of row order.
    :type _ranks: dict
    :return: A valid face-monotone filtration on the fixed complex.
    :rtype: Filtration
    """
    _values = dict(_cloud.rips_filtration(_complex).values)
    for _vertex in _complex.vertices:
        _values[(_vertex,)] = _step * _ranks[_vertex]
    return Filtration(_complex, _values)


def analyze_generated_cloud(_input):
    """Compute both ordinary Rips persistence and the full canopy pipeline.

    :param _input: Generated .npz containing ``points`` and ``vertex_ids``.
    :type _input: str or pathlib.Path
    :return: Machine-readable result with the filtration choice stated.
    :rtype: dict
    :raises ValueError: If the input has fewer than two points or is invalid.
    """
    _initial = PointCloud.from_npz(_input)
    if len(_initial) < 2:
        raise ValueError("At least two points are required")
    _scaled = PointCloud(_initial.points * SCALE_FACTOR,
                         _initial.vertex_ids)
    _distances = _edge_distances(_initial.points)
    _positive_distances = _distances[_distances > 0.0]
    if len(_positive_distances) != len(_distances):
        raise ValueError("This example expects distinct input points")
    _threshold = float(np.quantile(_distances, EDGE_QUANTILE))
    _domains = tuple(_cloud.rips_complex(
        _threshold, _max_dimension=2, _max_simplices=MAX_SIMPLICES)
        for _cloud in (_initial, _scaled))
    _fixed_complex = SimplicialComplex(
        _simplex for _domain in _domains
        for _simplex in _domain.simplices)
    if len(_fixed_complex) > MAX_SIMPLICES:
        raise ValueError("Full-cloud union exceeds the simplex limit")

    _vertex_step = float(_positive_distances.min()) / (
        1000000.0 * len(_initial))
    _ranks = {_vertex: _rank for _rank, _vertex in
              enumerate(sorted(_initial.vertex_ids))}
    _filtrations = tuple(_separated_vertex_filtration(
        _cloud, _fixed_complex, _vertex_step, _ranks)
        for _cloud in (_initial, _scaled))
    _family = Stratification(_filtrations, (0.0, 1.0))
    _a_canopy = ACanopy(_family)
    _topology = CanopyTopology(_a_canopy)
    _d_canopy = DCanopy(_a_canopy, _topology)
    _a_vines = VineDecomposition(_topology)
    _d_vines = VineDecomposition(_d_canopy)

    _fibers = []
    for _parameter in (0.0, 0.5, 1.0):
        _direct = PersistencePairing(_family.filtration_at(_parameter))
        _augmented = {}
        _diminished = {}
        for _dimension in (0, 1):
            _a_values = _a_canopy.diagram(_parameter, _dimension)
            _d_values = _d_canopy.diagram(_parameter, _dimension)
            if (_a_values != _direct.augmented_diagram(_dimension) or
                    _d_values != _direct.diminished_diagram(_dimension)):
                raise RuntimeError(
                    "Canopy diagram differs from direct persistence")
            _augmented["h{}".format(_dimension)] = _serialize_diagram(_a_values)
            _diminished["h{}".format(_dimension)] = _serialize_diagram(_d_values)
        _fibers.append({"parameter": _parameter,
                        "a_augmented": _augmented,
                        "d_diminished": _diminished})

    return {
        "input": str(Path(_input).resolve()),
        "vertex_count": len(_initial),
        "ambient_dimension": _initial.ambient_dimension,
        "simplex_count": len(_fixed_complex),
        "edge_quantile": EDGE_QUANTILE,
        "edge_threshold": _threshold,
        "construction": {
            "second_sample": "All original coordinates multiplied by 1.01",
            "scale_factor": SCALE_FACTOR,
            "vertex_correspondence": "Same vertex ID in both clouds",
            "fixed_complex": "Union of sparse Rips complexes at both samples",
            "family_interpolation": "Linear interpolation of filtration values",
        },
        "ordinary_rips_reference": {
            "scope": "All vertices, zero vertex births, at the two samples",
            "initial": _pairing_record(
                _initial.rips_filtration(_fixed_complex)),
            "scaled": _pairing_record(
                _scaled.rips_filtration(_fixed_complex)),
        },
        "full_canopy": {
            "scope": "All vertices; complete A/D and vine pipeline",
            "filtration": "Rips edge/triangle values, distinct vertex births",
            "vertex_birth_step": _vertex_step,
            "largest_vertex_birth": (len(_initial) - 1) * _vertex_step,
            "order_event_count": len(_family.events),
            "structural_parameters": list(_a_canopy.structural_parameters),
            "a_vine_count": len(_a_vines.vines),
            "d_vine_count": len(_d_vines.vines),
            "fibers": _fibers,
            "diagrams_match_direct_persistence": True,
        },
    }


def main():
    """Load the generated file, print scopes, and write a strict JSON record."""
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--input", type=Path, required=True)
    _parser.add_argument("--output", type=Path, required=True)
    _args = _parser.parse_args()
    _result = analyze_generated_cloud(_args.input)
    _args.output.parent.mkdir(parents=True, exist_ok=True)
    _args.output.write_text(
        json.dumps(_result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    _canopy = _result["full_canopy"]
    print("Full cloud: {} points in R^{}, {} fixed simplices".format(
        _result["vertex_count"], _result["ambient_dimension"],
        _result["simplex_count"]))
    print("Ordinary Rips reference: direct persistence at both samples")
    print("Full A/D canopies: vertex birth step {:.3g}; "
          "{} A-vines, {} D-vines; diagrams agree at 0, 0.5, 1".format(
              _canopy["vertex_birth_step"], _canopy["a_vine_count"],
              _canopy["d_vine_count"]))
    print("Results:", _args.output)


if __name__ == "__main__":
    main()
