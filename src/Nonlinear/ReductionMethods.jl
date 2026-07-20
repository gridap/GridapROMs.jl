abstract type NNType end
struct GenericType <: NNType end
struct MLPType <: NNType end

"""
    struct NNStrategy{A,L,O,F}
      type::A
      layers::NTuple{L,Int}
      optimiser::O
      loss::F
      epochs::Int
    end

Bundles all training hyperparameters for an MultiLayerPerceptron:
- `layers`: widths of the hidden layers as an `NTuple` (e.g. `(64, 64)`)
- `optimiser`: an `Optimisers.jl` rule (default: `Adam(1e-3)`)
- `loss`: `loss(ŷ, y) -> scalar`; built-ins are [`loss_mse`](@ref) and [`loss_mae`](@ref)
- `epochs`: number of full-batch gradient steps

Pass to [`NNHyperReduction`](@ref) or [`NNRegression`](@ref) via the `strategy` keyword.
"""
struct NNStrategy{A,L,O,F}
  type::A
  layers::NTuple{L,Int}
  optimiser::O
  loss::F
  epochs::Int
end

"""
    NNStrategy(; type=MLPType(), layers=(64,64), lr=1e-3, optimiser=Adam(lr), loss=loss_mse, epochs=1000)
"""
function NNStrategy(;
  type::NNType=MLPType(),
  layers::Union{AbstractVector,Tuple}=(64,64),
  lr::Real=1e-3,
  optimiser=Optimisers.Adam(lr),
  loss=loss_mse,
  epochs::Int=1000
  )

  NNStrategy(type,Tuple(layers),optimiser,loss,epochs)
end

"""
    struct NNRegression <: TrivialHyperReduction
      nparams::Int
      strategy::NNStrategy
    end

A hyper-reduction strategy for **operator regression**: the NN directly maps
parameter values to the Galerkin-projected residual vector, bypassing FE
assembly entirely during the online phase. Only suitable for residual
(vector-valued) operators.

The offline phase projects the residual snapshots onto the test space and
trains the NN to reproduce the projected vectors. The online phase calls
the NN forward pass, producing the projected residual without any assembly.

`strategy` controls the MultiLayerPerceptron architecture and training.
`nparams` controls how many parameter samples to use for NN training.
"""
struct NNRegression <: TrivialHyperReduction
  nparams::Int
  strategy::NNStrategy
end

function NNRegression(
  ;
  nparams::Int=20,
  type=MLPType(),
  layers=(64,64),
  lr=1e-3,
  optimiser=Optimisers.Adam(lr),
  loss=loss_mse,
  epochs=1000,
  strategy=NNStrategy(;type,layers,lr,optimiser,loss,epochs)
  )

  NNRegression(nparams,strategy)
end

ParamDataStructures.num_params(r::NNRegression) = r.nparams
get_strategy(r::NNRegression) = r.strategy

"""
    struct NNHyperReduction{A} <: HyperReduction{A}
      reduction::Reduction{A,EuclideanNorm}
      strategy::NNStrategy
    end

A hyper-reduction strategy that uses a neural network to predict EIM
coefficients from parameter values. The offline phase:

1. applies empirical interpolation on the snapshot basis to extract coefficients
2. trains a [`MultiLayerPerceptron`](@ref) via `strategy` on the `(μ, coefficient)` pairs

The online phase calls the NN forward pass instead of assembling the FE
operator on the reduced integration domain.
"""
struct NNHyperReduction{A} <: HyperReduction{A}
  reduction::Reduction{A,EuclideanNorm}
  strategy::NNStrategy
end

"""
    NNHyperReduction(args...; strategy=NNStrategy(), kwargs...) -> NNHyperReduction

Constructs a `NNHyperReduction` from a `Reduction` built with the same
positional/keyword arguments accepted by [`Reduction`](@ref). An optional
`strategy` keyword overrides the default [`NNStrategy`](@ref).
"""
function NNHyperReduction(
  args...;
  type=MLPType(),
  layers=(64,64),
  lr=1e-3,
  optimiser=Optimisers.Adam(lr),
  loss=loss_mse,
  epochs=1000,
  strategy=NNStrategy(;type,layers,lr,optimiser,loss,epochs),
  kwargs...
  )

  reduction = Reduction(args...;kwargs...)
  NNHyperReduction(reduction,strategy)
end

RBSteady.get_reduction(r::NNHyperReduction) = r.reduction
get_strategy(r::NNHyperReduction) = r.strategy

# utils 

function loss_mse(ŷ,y)
  s = 0.0
  @inbounds for i in eachindex(ŷ,y)
    d = ŷ[i] - y[i]
    s += d * d
  end
  s / length(y)
end

function loss_mae(ŷ,y)
  s = 0.0
  @inbounds for i in eachindex(ŷ,y)
    s += abs(ŷ[i] - y[i])
  end
  s / length(y)
end