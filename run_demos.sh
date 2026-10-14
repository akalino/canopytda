#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p outputs
PYTHON="${PYTHON:-python}"

"$PYTHON" -m pip install -e . gudhi matplotlib 2>&1 | tee outputs/install.log
"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v 2>&1 | tee outputs/tests.log
"$PYTHON" -m examples.demo_paper_examples 2>&1 | tee outputs/demo_fixed_paper.txt
"$PYTHON" -m examples.demo_paper_pairings 2>&1 | tee outputs/demo_paper_pairings.txt
"$PYTHON" -m examples.demo_equivalence 2>&1 | tee outputs/demo_equivalence.txt
"$PYTHON" -m examples.demo_structurality 2>&1 | tee outputs/demo_structurality.txt
"$PYTHON" -m examples.demo_a_canopy 2>&1 | tee outputs/demo_a_canopy.txt
"$PYTHON" -m examples.demo_topology 2>&1 | tee outputs/demo_topology.txt
"$PYTHON" -m examples.demo_d_canopy 2>&1 | tee outputs/demo_d_canopy.txt
"$PYTHON" -m examples.demo_vines 2>&1 | tee outputs/demo_vines.txt
"$PYTHON" -m examples.plot_fixed_paper outputs/fixed_paper_examples.png 2>&1 | tee outputs/plot_fixed_paper.log
"$PYTHON" -m examples.plot_pipeline_states outputs 2>&1 | tee outputs/plot_pipeline_states.log
"$PYTHON" -m examples.plot_topology outputs/topology_example_3_12.png 2>&1 | tee outputs/plot_topology.log
"$PYTHON" -m examples.plot_d_canopy outputs/d_canopy_diagonal_entry.png 2>&1 | tee outputs/plot_d_canopy.log
"$PYTHON" -m examples.plot_vines outputs/vine_decompositions.png 2>&1 | tee outputs/plot_vines.log