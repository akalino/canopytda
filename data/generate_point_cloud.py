"""Generate a reproducible Gaussian point cloud in NumPy .npz format.

From the repository root::

    python -m data.generate_point_cloud --n 100 --d 3

The archive contains ``points`` with shape ``(n, d)`` and ``vertex_ids``
equal to ``0, ..., n-1``, matching :class:`canopytda.point_cloud.PointCloud`.
These labels identify rows in one generated file; independently generated
clouds do not acquire a meaningful point correspondence just by sharing IDs.
"""

import argparse
from numbers import Integral
from pathlib import Path

import numpy as np


def _positive_integer(_text):
    """Parse a strictly positive command-line integer.

    :param _text: User-supplied number of points or coordinate dimensions.
    :type _text: str
    :return: Positive integer value.
    :rtype: int
    :raises argparse.ArgumentTypeError: If the value is not positive.
    """
    try:
        _number = int(_text)
    except ValueError as _error:
        raise argparse.ArgumentTypeError("Expected a positive integer") from _error
    if _number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return _number


def _nonnegative_integer(_text):
    """Parse a nonnegative random seed.

    :param _text: User-supplied random seed.
    :type _text: str
    :return: Nonnegative integer seed.
    :rtype: int
    :raises argparse.ArgumentTypeError: If the seed is invalid.
    """
    try:
        _number = int(_text)
    except ValueError as _error:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer") from _error
    if _number < 0:
        raise argparse.ArgumentTypeError("Expected a nonnegative integer")
    return _number


def generate_point_cloud(_n, _d, _seed=0):
    """Return a reproducible ``n`` by ``d`` Gaussian point matrix.

    Each coordinate is sampled independently from the standard normal
    distribution. Row indices provide stable labels for this one cloud.

    :param _n: Number of points.
    :type _n: positive int
    :param _d: Number of Euclidean coordinates per point.
    :type _d: positive int
    :param _seed: Random seed.
    :type _seed: nonnegative int
    :return: Points and corresponding integer vertex IDs.
    :rtype: tuple of numpy.ndarray
    :raises ValueError: If a count or seed is invalid.
    """
    for _name, _value in (("n", _n), ("d", _d)):
        if isinstance(_value, bool) or not isinstance(_value, Integral) or _value <= 0:
            raise ValueError("{} must be a positive integer".format(_name))
    if isinstance(_seed, bool) or not isinstance(_seed, Integral) or _seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    _random = np.random.default_rng(_seed)
    _points = _random.standard_normal((_n, _d))
    _vertex_ids = np.arange(_n, dtype=np.int64)
    return _points, _vertex_ids


def main():
    """Parse command-line options and save the point cloud archive."""
    _parser = argparse.ArgumentParser(description=__doc__)
    _parser.add_argument("--n", type=_positive_integer, required=True,
                         help="number of points")
    _parser.add_argument("--d", type=_positive_integer, required=True,
                         help="number of coordinates per point")
    _parser.add_argument("--seed", type=_nonnegative_integer, default=0,
                         help="random seed (default: 0)")
    _parser.add_argument("--output", type=Path,
                         help="output .npz path (default: data/cloud_nN_dD_seedS.npz)")
    _parser.add_argument("--overwrite", action="store_true",
                         help="replace an existing output file")
    _args = _parser.parse_args()
    _output = (_args.output if _args.output is not None else
               Path(__file__).resolve().parent /
               "cloud_n{}_d{}_seed{}.npz".format(_args.n, _args.d, _args.seed))
    if _output.suffix.lower() != ".npz":
        _parser.error("--output must end in .npz")
    if _output.exists() and not _args.overwrite:
        _parser.error("Output exists; choose another path or use --overwrite")
    _points, _vertex_ids = generate_point_cloud(_args.n, _args.d, _args.seed)
    _output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(_output, points=_points, vertex_ids=_vertex_ids)
    print("Saved {} points in R^{} to {}".format(
        _args.n, _args.d, _output))


if __name__ == "__main__":
    main()
