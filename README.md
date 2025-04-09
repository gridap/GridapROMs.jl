# GridapROMs

This package provides a set of tools for the solution of parameterized partial differential equations (PDEs) with reduced order models (ROMs). The presence of parameters severely impacts the feasibility of running high-fidelity (HF) codes such as the finite element (FE) method, because typically the solution is required for many different values of the parameters. ROMs create surrogate models that approximate the solution manifold on a lower-dimensional manifold. These surrogates provide accurate solutions in a much shorter time and with much fewer computational resources. The library is developed in close collaboration with [Gridap.jl](https://github.com/gridap/Gridap.jl).

| **Documentation** |
|:------------ |
| [![docdev](https://img.shields.io/badge/docs-dev-blue.svg)](https://gridap.github.io/GridapROMs.jl/dev/) | 
| **Citation** |
| [![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jcp.2022.111162-blue)](https://arxiv.org/abs/2503.15994) |
|**Build Status** |
| [![CI](https://github.com/gridap/GridapROMs.jl/workflows/CI/badge.svg)](https://github.com:gridap/GridapROMs.jl/actions?query=workflow%3ACI) [![codecov](https://codecov.io/gh/github.com:gridap/GridapROMs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/github.com:gridap/GridapROMs.jl) |

## Installation

```julia
# Type ] to enter package mode
pkg> add GridapROMs
```

## Examples

Before running the following examples, it is necessary to import from file some geometries which can be found [here](https://gridap.github.io/GridapROMs.jl/docs/assets). The file name is `models.zip`. The geometries must be unzipped and moved to a directory where the numerical experiments are ran. This directory should be placed inside the `data` directory of the `Julia` project which is being used to run these experiments. To find this directory, first add the package `DrWatson` with 

```julia
# Type ] to enter package mode
pkg> add DrWatson
```

and call

```julia
julia> test_dir = datadir()
```

Now we can unzip the compressed folder in `dir` with 

```julia
julia> model_dir = joinpath(@__DIR__,"docs/src/assets")
# Type ; to enter shell mode
shell> unzip $model_dir/models.zip -d $test_dir
```

In the following numerical examples, we provide a plot of the convergence errors for a series of tolerances (which determine the accuracy of the method), and a solution plot obtained with a fixed tolerance of `1e-5`.

### Test 1 

Solve a steady elasticity problem with a [proper orthogonal decomposition algorithm](https://link.springer.com/book/10.1007/978-3-319-15431-2) (POD). The presence of parameters affecting the problem's LHS/RHS are dealt with by employing a [discrete empirical interpolation method in matrix form](https://www.sciencedirect.com/science/article/pii/S0021999115006543) (MDEIM). 

```julia
julia> include("examples/SteadyElasticityPOD.jl")
```
Solution             |  Convergence
:-------------------------:|:-------------------------:
<img src="docs/src/assets/results/elasticity_pod/plot/rbsol.png" alt="drawing" style="width:400px; height:250px;"/>  |  <img src="docs/src/assets/results/elasticity_pod/results/convergence.png" alt="drawing" style="width:400px; height:250px;"/>

### Test 2

Solve the same problem, but with a tensor-train (TT) decomposition approach. In particular, we employ the [TT-SVD](https://epubs.siam.org/doi/10.1137/090752286) method to compute the reduced approximation subspace, and [TT-MDEIM](https://arxiv.org/abs/2412.14460) for the system approximation. 

```julia
julia> include("examples/SteadyElasticityTTSVD.jl")
```
Solution             |  Convergence
:-------------------------:|:-------------------------:
<img src="docs/src/assets/results/elasticity_ttsvd/plot/rbsol.png" alt="drawing" style="width:400px; height:250px;"/>  |  <img src="docs/src/assets/results/elasticity_ttsvd/results/convergence.png" alt="drawing" style="width:400px; height:250px;"/>

### Test 3

Solve a steady Stokes equation with a POD+MDEIM method.

```julia
julia> include("examples/SteadyStokesPOD.jl")
```

Solution - velocity          |  Solution - pressure        |  Convergence
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/src/assets/results/stokes_pod/plot/rbvel.png" alt="drawing" style="width:275px; height:250px;"/>  |  <img src="docs/src/assets/results/stokes_pod/plot/rbpress.png" alt="drawing" style="width:275px; height:250px;"/>  |  <img src="docs/src/assets/results/stokes_pod/results/convergence.png" alt="drawing" style="width:275px; height:250px;"/> 

### Test 4 

Moving to transient applications, we first solve a heat equation with a [space-time RB-MDEIM method](https://www.sciencedirect.com/science/article/pii/S0377042724000165).

```julia
julia> include("examples/HeatEquationSTRB.jl")
```

Solution             |  Convergence
:-------------------------:|:-------------------------:
<img src="docs/src/assets/results/heateq_pod/plot/rbsol.gif" alt="drawing" style="width:400px; height:250px;"/>  |  <img src="docs/src/assets/results/heateq_pod/results/convergence.png" alt="drawing" style="width:400px; height:250px;"/> 

### Test 5

Lastly, we solve a transient Navier-Stokes equation with the same space-time RB method as in `Test 4`.

```julia
julia> include("examples/NStokesTransientSTRB.jl")
```

Solution - velocity          |  Solution - pressure        |  Convergence
:-------------------------:|:-------------------------:|:-------------------------:
<img src="docs/src/assets/results/nstokes_strb/plot/rbvel.gif" alt="drawing" style="width:275px; height:250px;"/>  |  <img src="docs/src/assets/results/nstokes_strb/plot/rbpress.gif" alt="drawing" style="width:275px; height:250px;"/>  |  <img src="docs/src/assets/results/nstokes_strb/results/convergence.png" alt="drawing" style="width:275px; height:250px;"/> 

## How to cite ROManifolds

In order to give credit to the `ROManifolds` contributors, we simply ask you to cite the references below in any publication in which you have made use of the `ROManifolds` project. 

```
@misc{mueller2025frameworkefficientreducedorder,
    title={A framework for efficient reduced order modelling in the Julia programming language}, 
    author={Nicholas Mueller and Santiago Badia},
    year={2025},
    eprint={2503.15994},
    archivePrefix={arXiv},
    primaryClass={math.NA},
    url={https://arxiv.org/abs/2503.15994}, 
}
```
