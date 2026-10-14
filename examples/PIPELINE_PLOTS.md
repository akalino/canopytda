# Plots of the paper examples

Run from the repository root after installing the package and Matplotlib:

```bash
python -m examples.plot_pipeline_states outputs
```

The script saves two PNGs to `outputs/`. `run_paper_demos.sh` runs it after the
tests and the demos.

## Example 3.12: five states of the fixed L complex

`pipeline_example_3_12.png` reads the public `Stratification`,
`PersistencePairing`, and `ACanopy` objects. Its first row shows the four named
simplex values as the parameter moves from 0 to 1. The middle row samples the
five strata, showing their value orders, chosen pairs, and numbers of computed
maps into the next stratum. The bottom row gives the corresponding augmented
degree-one diagrams. Each diagram is checked against direct persistence
reduction before plotting.

At `t = 1/3`, the two distinct simplex-pair points have exactly the same
birth and death values. The plot uses one marker labeled `×2` at that exact
coordinate, while the state card lists both pair labels. Two self maps and
two maps across each adjacent incidence accompany this structural event.
The crossing at `t = 3/4` has one map across each adjacent incidence and is
classified as nonstructural.

The earlier `plot_fixed_paper.py` shows the underlying fixed complex alongside
the other fixtures. Together the plots cover the complex, filtration values,
order strata, reduced pairs, diagrams and finite A-canopy data.

## Example 2.19: two consistent orders

`orders_example_2_19.png` computes persistence twice from the same filtration
`g` of the L complex. The two orders produce different simplex pairs but the
same augmented degree-one diagram. This illustrates why a diagram alone does
not display the simplex labels retained by the A-canopy.

The A-canopy incidence counts in these plots describe its finite presentation;
the plots do not depict its neighborhood topology.
