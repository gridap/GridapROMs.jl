# GridapROMs.jl

GridapROMs provides tools for the reduced order modelling (ROM) of parameterised partial
differential equations (PDEs). It is built on top of [Gridap](https://github.com/gridap/Gridap.jl):
parameterised problems are written as in Gridap, with the parameter `μ` as an extra argument,
and reduced with a few additional lines of code.

The library supports

- steady and transient, linear and nonlinear, single- and multi-field problems
- reduced bases built with POD or TT-SVD, in any norm, with supremizer enrichment
- hyper-reduction with DEIM, S-OPT and radial basis function interpolation
- global and local (clustered) reduced operators
- distributed-memory parallelism of the whole offline and online phases

## Installation

```julia
pkg> add GridapROMs
```

## Tutorials

| | |
|:--|:--|
| [Tutorial 1: Parameterised PDEs](tutorials/t001_parameterised_pdes.md) | parameter spaces, parameterised data and FE spaces, steady and transient FE problems |
| [Tutorial 2: Reduced order models](tutorials/t002_reduced_order_models.md) | offline/online phases, POD, TT-SVD, hyper-reduction, reduced operators, multi-field, nonlinear and transient problems |
| [Tutorial 3: Local reduced operators](tutorials/t003_local_roms.md) | clustering of the parameter space and local reduced operators |
| [Tutorial 4: Distributed-memory ROMs](tutorials/t004_distributed.md) | running the ROM pipeline in parallel with MPI |

## Reference

The library is organised in modules. The most common functions and types are exported by
`GridapROMs`; the rest are accessible from the respective module.

| Module | Content |
|:-------|:--------|
| [`Utils`](utils.md) | integration domains, linear/nonlinear bundles, performance trackers, errors |
| [`DofMaps`](dof_maps.md) | DOF orderings of FE spaces and sparse matrices |
| [`TProduct`](tproduct.md) | tensor-product geometries and FE spaces |
| [`ParamDataStructures`](param_data_structures.md) | parameter spaces, realisations, parameterised arrays and snapshots |
| [`ParamAlgebra`](param_algebra.md) | parameterised algebraic operators and solvers |
| [`ParamGeometry`](param_geometry.md) | parameterised geometries |
| [`ParamFESpaces`](param_fe_spaces.md) | parameterised FE spaces and functions |
| [`ParamSteady`](param_steady.md) | steady parameterised FE operators |
| [`ParamODEs`](param_odes.md) | transient parameterised FE operators and time-marching schemes |
| [`Extensions`](extensions.md) | extension of solutions from active to background domains |
| [`RBSteady`](rbsteady.md) | reduction, hyper-reduction and reduced operators for steady problems |
| [`RBTransient`](rbtransient.md) | their space-time extension for transient problems |
| [`Distributed`](distributed.md) | distributed-memory parallelism |

## How to cite

```
@article{MUELLER2026109985,
  title = {GridapROMs.jl: Efficient reduced order modelling in the Julia programming language},
  journal = {Computer Physics Communications},
  volume = {320},
  pages = {109985},
  year = {2026},
  doi = {https://doi.org/10.1016/j.cpc.2025.109985},
  author = {Nicholas Mueller and Santiago Badia}
}
```
