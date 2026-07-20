"""
    module Nonlinear

Extension of [`RBSteady`](@ref) with nonlinear (neural-network based) models
for hyper-reduction and operator regression.

Two complementary strategies are provided:

1. **NN coefficient interpolation** (`NNHyperReduction`): keeps the standard
   linear POD basis and replaces only the EIM coefficient prediction step with
   a neural network.  Drop-in replacement for `RBFHyperReduction`.

2. **NN operator regression** (`NNRegression`): bypasses FE assembly
   entirely during the online phase.  The NN is trained to predict the
   Galerkin-projected residual from parameter values directly.

Main types:

- [`NNHyperReduction`](@ref) / [`NNRegression`](@ref) — reduction strategy
  types, passed to [`RBSolver`](@ref) as `residual_reduction` /
  `jacobian_reduction`
- [`NNHRProjection`](@ref) — the resulting projection for strategy 1
- [`NNProjection`](@ref) — the resulting projection for strategy 2
- [`NNInterpolation`](@ref) — online interpolant produced by strategy 1
- [`AbstractNNModel`](@ref) / [`NNModel`](@ref) — backend-agnostic wrappers
  for any callable (Flux chain, Lux closure, custom Julia function)
- [`MLP`](@ref) — built-in MLP trained with ForwardDiff; no external
  NN library required for quick experiments

Quick-start:

    using GridapROMs.Nonlinear

    # Strategy 1: NN replaces EIM coefficient prediction
    red = PODReduction(1e-4;nparams=100)
    nn_res = NNHyperReduction(red) # uses built-in MLP factory by default
    nn_jac = NNHyperReduction(red)
    solver = RBSolver(fesolver,red,nn_res,nn_jac)

    # Strategy 2: NN predicts projected residual without assembly
    nn_reg = NNRegression(red)
    solver = RBSolver(fesolver,red,nn_reg,nn_jac)

    # Custom Flux factory (example)
    using Flux
    my_factory = (r,coeff) -> begin
      d = length(first(r)); n = prod(innersize(coeff))
      chain = Chain(Dense(d,64,tanh),Dense(64,n))
      # ... training loop with Flux optimiser ...
      NNModel(x -> chain(x))
    end
    nn_res = NNHyperReduction(red;nn_factory=my_factory)
"""
module Nonlinear

using ForwardDiff
using LinearAlgebra
using Optimisers

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.Helpers

using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady

import GridapROMs.RBSteady: get_at_domain

export NNStrategy
export NNHyperReduction
export get_strategy
export loss_mse,loss_mae
include("ReductionMethods.jl")

export AbstractNNModel
export NNModel
export MLP
export train!
export train_model
include("Models.jl")

export NNHRProjection
export NNContribution
export NNProjection
export NNOpContribution
export NNInterpolation
include("Interpolations.jl")

export NNRegression
include("HyperReductions.jl")

end # module
