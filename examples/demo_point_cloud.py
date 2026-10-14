"""Load point clouds and run a small fixed-complex canopy example.

With no file arguments, use two in-memory labeled clouds::

    python -m examples.demo_point_cloud

With user data, provide at least two files, which may use different formats::

    python -m examples.demo_point_cloud --file first.npz --file second.parquet \
        --coordinates x y --max-edge-length 0.8

Use ``--points-key hidden_layers.0.weight`` for a nested tensor in .pt.
Explicit vertex IDs should be present in the files. If rows are known to
represent identical vertices, use ``--assume-row-order`` instead.
"""

import argparse
from pathlib import Path

from canopytda.a_canopy import ACanopy
from canopytda.d_canopy import DCanopy
from canopytda.point_cloud import PointCloud
from canopytda.topology import CanopyTopology
from canopytda.vines import VineDecomposition


def _load_cloud(_path, _points_key, _coordinates):
    """Pass only the applicable file-reader options to the public class."""
    _suffix = Path(_path).suffix.lower()
    _options = {}
    if _suffix in (".npz", ".pt"):
        _options["_points_key"] = _points_key
    if _suffix == ".parquet" and _coordinates:
        _options["_coordinate_columns"] = _coordinates
    return PointCloud.from_file(_path, **_options)


def main():
    """Check augmented and diminished diagrams at knots and midpoints."""
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--file", action="append", default=[])
    _parser.add_argument("--points-key", default="points")
    _parser.add_argument("--coordinates", nargs="+", default=None)
    _parser.add_argument("--max-edge-length", type=float, default=1.0)
    _parser.add_argument("--max-dimension", type=int, default=2)
    _parser.add_argument("--homology-dimension", type=int, default=0)
    _parser.add_argument("--assume-row-order", action="store_true")
    _args = _parser.parse_args()
    if _args.file and len(_args.file) < 2:
        _parser.error("Provide two or more --file arguments")
    _clouds = (tuple(_load_cloud(_path, _args.points_key, _args.coordinates)
                     for _path in _args.file) if _args.file else
               (PointCloud([[0, 0], [1, 0]], [0, 1]),
                PointCloud([[0, 0], [0.5, 0]], [0, 1])))
    _family = PointCloud.rips_family(
        _clouds, _max_edge_length=_args.max_edge_length,
        _max_dimension=_args.max_dimension,
        _assume_row_order=_args.assume_row_order)
    print("Fixed complex: {} vertices, {} simplices".format(
        len(_family.complex.vertices), len(_family.complex.simplices)))
    if len(_family.complex) > 12:
        raise ValueError("This exact canopy demo needs at most 12 simplices; "
                         "reduce the edge bound or use fewer points")
    _a_canopy = ACanopy(_family,
                        _homology_dimension=_args.homology_dimension)
    _topology = CanopyTopology(_a_canopy)
    _d_canopy = DCanopy(_a_canopy, _topology)
    _a_vines = VineDecomposition(_topology)
    _d_vines = VineDecomposition(_d_canopy)
    _parameters = _family.parameters
    _probes = sorted(set(_parameters) | {
        0.5 * (_left + _right) for _left, _right in
        zip(_parameters, _parameters[1:])
    })
    for _parameter in _probes:
        print("parameter={}: augmented={}, diminished={}".format(
            _parameter,
            _a_canopy.diagram(_parameter, _args.homology_dimension),
            _d_canopy.diagram(_parameter, _args.homology_dimension)))
    print("A vines: {}; D vines: {}".format(
        len(_a_vines.vines), len(_d_vines.vines)))


if __name__ == "__main__":
    main()
