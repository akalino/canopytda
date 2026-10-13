#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p outputs
PYTHON="${PYTHON:-python}"

"$PYTHON" -m pip install -e . gudhi matplotlib 2>&1 | tee outputs/install.log
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v 2>&1 | tee outputs/tests.log
"$PYTHON" -m examples.demo_paper_examples 2>&1 | tee outputs/demo_fixed_paper.txt
"$PYTHON" -m examples.demo_paper_pairings 2>&1 | tee outputs/demo_paper_pairings.txt
"$PYTHON" -m examples.plot_fixed_paper outputs/fixed_paper_examples.png 2>&1 | tee outputs/plot_fixed_paper.log