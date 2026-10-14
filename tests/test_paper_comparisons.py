"""Check the named comparison claims and exported benchmark records."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.paper_comparisons import run_comparisons
from benchmarks.paper_comparisons import write_results


class PaperComparisonTests(unittest.TestCase):
    """Verify the reference fibers and publication-ready output contract."""

    @classmethod
    def setUpClass(cls):
        cls.results = run_comparisons()

    def test_paper_fibers_match_with_multiplicity(self):
        """Augmented and diminished diagrams match at all selected fibers."""
        self.assertEqual(self.results["failures"], [])
        self.assertEqual(len(self.results["fibers"]), 15)
        for _fiber in self.results["fibers"]:
            with self.subTest(example=_fiber["example"],
                              parameter=_fiber["parameter"]):
                self.assertTrue(_fiber["augmented_match"])
                self.assertTrue(_fiber["diminished_match"])
                if _fiber["gudhi_match"] is not None:
                    self.assertTrue(_fiber["gudhi_match"])

    def test_paper_comparison_checks_distinguish_vines_and_diagonal(self):
        """Coincident values, event separation, and diagonal have witnesses."""
        self.assertTrue(all(self.results["paper_checks"].values()))
        _fibers = {(_fiber["example"], _fiber["parameter"]): _fiber
                   for _fiber in self.results["fibers"]}
        self.assertFalse(_fibers["paper_l_3_7", 1 / 3][
            "two_points_separable"])
        self.assertTrue(_fibers["paper_l_3_7", 0.75][
            "two_points_separable"])
        self.assertEqual(_fibers["paper_3_16_disjoint_edges", 0.0][
            "max_coordinate_multiplicity"], 2)

    def test_files_are_parseable_and_keep_essential_points(self):
        """Machine outputs preserve infinities without nonstandard JSON."""
        with tempfile.TemporaryDirectory() as _directory:
            write_results(self.results, _directory)
            _path = Path(_directory)
            _written = json.loads((_path / "paper_comparisons.json").read_text())
            self.assertEqual(_written["paper_checks"],
                             self.results["paper_checks"])
            with (_path / "paper_fibers.csv").open(newline="") as _stream:
                _rows = list(csv.DictReader(_stream))
            self.assertEqual(len(_rows), 15)
            self.assertTrue(any("inf" in _row["direct_augmented"]
                                for _row in _rows))
            self.assertIn("Paper complex comparisons",
                          (_path / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
