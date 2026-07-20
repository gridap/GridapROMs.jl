loss_mse(ŷ,y) = sum(abs2,ŷ .- y) / length(y)
loss_mae(ŷ,y) = sum(abs,ŷ .- y) / length(y)

"""
    struct NNStrategy{L,O,Lf}
      layers::NTuple{L,Int}
      optimiser::O
      loss::Lf
      epochs::Int
    end

Bundles all training hyperparameters for an MLP:
- `layers`: widths of the hidden layers as an `NTuple` (e.g. `(64, 64)`)
- `optimiser`: an `Optimisers.jl` rule (default: `Adam(1e-3)`)
- `loss`: `loss(ŷ, y) -> scalar`; built-ins are [`loss_mse`](@ref) and [`loss_mae`](@ref)
- `epochs`: number of full-batch gradient steps

Pass to [`NNHyperReduction`](@ref) or [`NNOpRegression`](@ref) via the `strategy` keyword.
"""
struct NNStrategy{L,O,Lf}
  layers::NTuple{L,Int}
  optimiser::O
  loss::Lf
  epochs::Int
end

"""
    NNStrategy(; layers=(64,64), lr=1e-3, optimiser=Adam(lr), loss=loss_mse, epochs=1000)
"""
function NNStrategy(;
  layers::Union{AbstractVector{Int},Tuple}=(64,64),
  lr::Real=1e-3,
  optimiser=Optimisers.Adam(lr),
  loss=loss_mse,
  epochs::Int=1000
  )

  NNStrategy(Tuple(layers),optimiser,loss,epochs)
end

"""
    struct NNHyperReduction{A} <: HyperReduction{A}
      reduction::Reduction{A,EuclideanNorm}
      strategy::NNStrategy
    end

A hyper-reduction strategy that uses a neural network to predict EIM
coefficients from parameter values. The offline phase:

1. applies empirical interpolation on the snapshot basis to extract coefficients
2. trains a [`MLP`](@ref) via `strategy` on the `(μ, coefficient)` pairs

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
  layers=(64,64),
  lr=1e-3,
  optimiser=Optimisers.Adam(lr),
  loss=loss_mse,
  epochs=1000,
  strategy=NNStrategy(;layers,lr,optimiser,loss,epochs),
  kwargs...
  )

  reduction = Reduction(args...;kwargs...)
  NNHyperReduction(reduction,strategy)
end

RBSteady.get_reduction(r::NNHyperReduction) = r.reduction
get_strategy(r::NNHyperReduction) = r.strategy
