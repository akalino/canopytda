# canopytda
Python computation tools for computing Canopies.

## Data types

We call $K$ the *fixed underlying simplicial complex*, 
which is stored as an immutable ```SimplicialComplex``` without 
filtration or embedding data.
The ```Filtration``` object specifies which filtration is being applied to the 
simplicial complex $K$, with choices ```alpha, custom, delaunay_cech, explicit, lower_star, rips```.


## Module dependencies

![Canopy module dependency diagram](docs/images/canopy-dependencies.svg)


## Experimental branch

### Benchmark comparisons for evolving point clouds

- Full persistence recomputation at observed checkpoints
- Fixed "carrier" complex and vineyard updates
- Matching framewise PDs with Bottleneck or Wasserstein
- Zigzag persistence at selected radii
- Crocker plots or stacks
- Mover schedules / MMHM
- CKA and simple baselines

Starting with controlled trajectories and the toy examples to measure:
- event timing
- false positives under re-sampling/seeds
- accuracy vs. ground truth
- runtime
- memory consumption
- complex size

Deferred: point additions/deletions, resampling, probabilistic extensions