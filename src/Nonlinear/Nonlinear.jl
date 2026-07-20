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
- [`NNProjection`](@ref) / [`NNContribution`](@ref) — the resulting projection/contribution for strategy 2
- [`NNInterpolation`](@ref) — online interpolant produced by strategy 1
- [`NNStrategy`](@ref) — bundles MLP architecture, optimiser, loss, and epoch count
- [`NeuralNetwork`](@ref) / [`GenericNeuralNetwork`](@ref) / [`MultiLayerPerceptron`](@ref)
  — NN model types; `GenericNeuralNetwork` wraps any callable (Flux chain, Lux closure)

Quick-start:

    using GridapROMs.Nonlinear

    # Strategy 1: NN replaces EIM coefficient prediction
    red = PODReduction(1e-4;nparams=100)
    nn_res = NNHyperReduction(red)   # default MLP, tanh, Adam(1e-3), 1000 epochs
    nn_jac = NNHyperReduction(red;layers=(128,128),epochs=2000)
    solver = RBSolver(fesolver,red,nn_res,nn_jac)

    # Strategy 2: NN predicts projected operator without assembly
    nn_reg = NNRegression(;nparams=100,layers=(64,64))
    solver = RBSolver(fesolver,red,nn_reg,nn_jac)

    # Custom external model (e.g. Flux)
    using Flux
    chain = Chain(Dense(d,64,tanh),Dense(64,n))
    nn_res = NNHyperReduction(red;strategy=NNStrategy(type=GenericType()))
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

export NNType, MLPType, GenericType
export NNStrategy
export NNRegression
export NNHyperReduction
export get_strategy
export loss_mse, loss_mae
include("ReductionMethods.jl")

export NeuralNetwork
export GenericNeuralNetwork
export MultiLayerPerceptron
export TrainedNeuralNetwork
export train!
include("Models.jl")

export NNInterpolation
include("Interpolations.jl")

export NNProjection
export NNContribution
export NNHRProjection
include("HyperReductions.jl")

include("ReducedOperators.jl")

end # module
