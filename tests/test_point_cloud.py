"""Test point-cloud loading, fixed IDs and filtration-family integration."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from canopytda.a_canopy import ACanopy
from canopytda.complex import SimplicialComplex
from canopytda.pairing import PersistencePairing
from canopytda.point_cloud import PointCloud


class PointCloudTests(unittest.TestCase):
    """Cover the three readers and the fixed-complex Rips contract."""

    def test_npz_loads_explicit_ids_without_pickle_and_copies_points(self):
        """The file and caller cannot mutate a validated point cloud."""
        with tempfile.TemporaryDirectory() as _directory:
            _path = Path(_directory) / "cloud.npz"
            np.savez(_path, points=[[0.0, 0.0], [2.0, 0.0]],
                     vertex_ids=[17, 12])
            _cloud = PointCloud.from_file(_path)
            self.assertEqual(_cloud.vertex_ids, (17, 12))
            self.assertTrue(_cloud.ids_explicit)
            _copy = _cloud.points
            _copy[0, 0] = 99.0
            self.assertEqual(_cloud.points[0, 0], 0.0)
            self.assertEqual(_cloud.rips_filtration(
                SimplicialComplex(((12, 17),))).value((12, 17)), 2.0)
            with self.assertRaises(ValueError):
                PointCloud.from_npz(_path, _vertex_ids=[17, 11])

    def test_pt_supports_nested_tensor_and_safe_cpu_loading(self):
        """A dotted layer path reaches a real 2D tensor-like field."""
        class _Tensor:
            def __init__(self, _array):
                self.array = np.asarray(_array)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.array

        _checkpoint = {"hidden_layers": [{"weight": _Tensor(
            [[0.0, 1.0], [2.0, 3.0]])}],
            "vertex_ids": _Tensor([30, 40])}
        _fake_torch = types.ModuleType("torch")
        _fake_torch.is_tensor = lambda _value: isinstance(_value, _Tensor)
        _calls = []

        def _load(_path, **_options):
            _calls.append(_options)
            return _checkpoint

        _fake_torch.load = _load
        with patch.dict(sys.modules, {"torch": _fake_torch}):
            _cloud = PointCloud.from_file(
                "cloud.pt", _points_key="hidden_layers.0.weight")
            self.assertEqual(_cloud.vertex_ids, (30, 40))
            self.assertEqual(_cloud.points.tolist(), [[0.0, 1.0], [2.0, 3.0]])
            self.assertEqual(_calls, [{"map_location": "cpu",
                                       "weights_only": True}])
            _checkpoint["hidden_layers.0.weight"] = _Tensor([[8.0, 9.0]])
            self.assertEqual(PointCloud.from_pt(
                "cloud.pt", _points_key="hidden_layers.0.weight",
                _vertex_ids_key=None).points.tolist(), [[8.0, 9.0]])
            with self.assertRaises(KeyError):
                PointCloud.from_pt("cloud.pt", _points_key="missing.0")

    def test_parquet_coordinate_selection_and_ids(self):
        """Wide tables select coordinates explicitly when metadata is present."""
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas unavailable")
        _frame = pd.DataFrame({"vertex_id": [5, 8], "x": [0.0, 1.0],
                               "y": [1.0, 2.0], "label": ["a", "b"]})
        with patch.object(pd, "read_parquet", return_value=_frame) as _reader:
            _cloud = PointCloud.from_file(
                "points.parquet", _coordinate_columns=("x", "y"))
            self.assertEqual(_cloud.vertex_ids, (5, 8))
            self.assertEqual(_cloud.points.tolist(), [[0.0, 1.0], [1.0, 2.0]])
            _reader.assert_called_once()
            with self.assertRaises(ValueError):
                PointCloud.from_parquet("points.parquet")
            with self.assertRaises(ValueError):
                PointCloud.from_parquet("points.parquet",
                                        _coordinate_columns=("vertex_id", "x"))

    def test_invalid_shape_values_and_identifiers(self):
        """Reject ambiguous or unsafe input before constructing simplices."""
        for _points in ([], [1, 2], [[0.0], [float("nan")]],
                        [[0.0], [1.0, 2.0]]):
            with self.subTest(points=_points), self.assertRaises(ValueError):
                PointCloud(_points)
        for _ids in ([1, 1], [0, True], [1.0, 2.0]):
            with self.subTest(ids=_ids), self.assertRaises((ValueError, TypeError)):
                PointCloud([[0.0], [1.0]], _ids)
        with self.assertRaises(ValueError):
            PointCloud.from_file("cloud.csv")

    def test_rips_complex_and_filtration_preserve_ids(self):
        """Higher simplices take the greatest edge length, including ties."""
        _cloud = PointCloud([[0, 0], [0, 1], [1, 0]], [11, 7, 4])
        _complex = _cloud.rips_complex(2.0, _max_dimension=2)
        self.assertIn((4, 7, 11), _complex.simplices)
        _filtration = _cloud.rips_filtration(_complex)
        self.assertAlmostEqual(_filtration.value((4, 7, 11)), 2 ** 0.5)
        self.assertEqual(_filtration.value((11,)), 0.0)
        with self.assertRaises(ValueError):
            _cloud.rips_complex(2.0, _max_simplices=6)
        with self.assertRaises(ValueError):
            _cloud.rips_filtration(SimplicialComplex(((4, 7),)))

    def test_union_domain_and_pipeline_with_reordered_rows(self):
        """One fixed K includes an edge visible at only one sampled cloud."""
        with tempfile.TemporaryDirectory() as _directory:
            _first_path = Path(_directory) / "first.npz"
            _last_path = Path(_directory) / "last.npz"
            np.savez(_first_path, points=[[0, 0], [1, 0]],
                     vertex_ids=[10, 20])
            np.savez(_last_path, points=[[0.5, 0], [0, 0]],
                     vertex_ids=[20, 10])
            _first = PointCloud.from_file(_first_path)
            _last = PointCloud.from_file(_last_path)
        _family = PointCloud.rips_family(
            (_first, _last), (0.0, 1.0), _max_edge_length=0.75)
        self.assertEqual(_family.complex.simplices,
                         ((10,), (20,), (10, 20)))
        self.assertEqual(_family.filtration_at(0).value((10, 20)), 1.0)
        self.assertEqual(_family.filtration_at(1).value((10, 20)), 0.5)
        self.assertEqual(_family.filtration_at(0.5).value((10, 20)), 0.75)
        _canopy = ACanopy(_family, _homology_dimension=0)
        self.assertEqual(_canopy.diagram(0.5, 0),
                         PersistencePairing(_family.filtration_at(0.5))
                         .augmented_diagram(0))

    def test_family_requires_explicit_correspondence_and_valid_policy(self):
        """Row-order assumptions and coordinate interpolation are not silent."""
        _first = PointCloud([[0, 0], [1, 0]])
        _last = PointCloud([[0, 0], [0.5, 0]])
        with self.assertRaisesRegex(ValueError, "vertex IDs"):
            PointCloud.rips_family((_first, _last),
                                   _max_edge_length=1)
        _family = PointCloud.rips_family(
            (_first, _last), _max_edge_length=1, _assume_row_order=True)
        self.assertEqual(_family.filtration_at(0.5).value((0, 1)), 0.75)
        with self.assertRaisesRegex(ValueError, "interpolation"):
            PointCloud.rips_family((_first, _last), _max_edge_length=1,
                                   _interpolation="point_coordinates",
                                   _assume_row_order=True)
        with self.assertRaisesRegex(ValueError, "identical IDs"):
            PointCloud.rips_family((PointCloud([[0], [1]], [1, 2]),
                                    PointCloud([[0], [1]], [1, 3])),
                                   _max_edge_length=1)


if __name__ == "__main__":
    unittest.main()
