"""
    abstract type AbstractNNModel <: Map end

Abstract supertype for neural network models. Any concrete subtype must
implement `(m::T)(x::AbstractMatrix) -> AbstractMatrix` where `x` is a
`(param_dim × batch_size)` matrix of parameters and the output is a
`(output_dim × batch_size)` matrix of predictions.

Wrap external Flux/Lux models with [`NNModel`](@ref):

    model = NNModel(flux_chain)     # Flux
    model = NNModel(p -> lux_apply(chain, p, ps, st))  # Lux closure
"""
abstract type AbstractNNModel <: Map end

"""
    struct NNModel{A} <: AbstractNNModel
      model::A
    end

Wraps any callable `A` (Flux chain, Lux closure, plain Julia function) as
an [`AbstractNNModel`](@ref). The wrapped callable must accept a
`(d × k)` parameter matrix and return an `(n × k)` prediction matrix.
"""
struct NNModel{A} <: AbstractNNModel
  model::A
end

function Arrays.return_cache(m::NNModel,x::AbstractMatrix)
  return_cache(m.model,x)
end

function Arrays.evaluate!(cache,m::NNModel,x::AbstractMatrix)
  evaluate!(cache,m.model,x)
end

"""
    struct MLP{L,A<:AbstractVector} <: AbstractNNModel
      layers::NTuple{L,Int}
      σ::Function
      θ::A
    end

A minimal dense feed-forward network whose parameters are stored as a flat
vector `θ` for ForwardDiff compatibility. Architecture is determined by
`layers = (d, h₁, h₂, …, n)` where `d` is input dim and `n` is output dim.
Hidden layers apply `σ`; the output layer is linear.

Use [`NNStrategy`](@ref) to construct and train one as part of
[`NNHyperReduction`](@ref). For large networks or long training runs,
prefer injecting a Flux/Lux model via [`NNModel`](@ref).
"""
struct MLP{L,A<:AbstractVector} <: AbstractNNModel
  layers::NTuple{L,Int}
  σ::Function
  θ::A
end

function MLP(layers::NTuple{L,Int}; activation::Function=tanh, T::Type=Float64) where {L}
  nθ = sum(layers[i+1]*(layers[i]+1) for i in 1:L-1)
  θ = randn(T,nθ) .* T(0.01)
  MLP(layers,activation,θ)
end

function MLP(layers::AbstractVector{<:Int},args...;kwargs...)
  MLP(Tuple(layers),args...;kwargs...)
end

function _mlp_forward(θ::AbstractVector,layers::Tuple,σ,x::AbstractMatrix)
  offset = 0
  h = x
  nlayers = length(layers) - 1
  for l in 1:nlayers
    n_out = layers[l+1]
    n_in = layers[l]
    W = reshape(θ[offset+1:offset+n_out*n_in],n_out,n_in)
    offset += n_out * n_in
    b = θ[offset+1:offset+n_out]
    offset += n_out
    z = W * h .+ b
    h = l < nlayers ? σ.(z) : z
  end
  h
end

(m::MLP)(x::AbstractMatrix) = _mlp_forward(m.θ,m.layers,m.σ,x)

"""
    train!(mlp::MLP, x::AbstractMatrix, y::AbstractMatrix, strategy::NNStrategy) -> MLP

Trains `mlp` with the optimiser, loss, and epoch count from `strategy` using
ForwardDiff for gradients. `x` is `(param_dim × n_samples)`, `y` is
`(output_dim × n_samples)`. Updates `mlp.θ` in-place and returns `mlp`.
"""
function train!(
  mlp::MLP,
  x::AbstractMatrix,
  y::AbstractMatrix,
  strategy::NNStrategy
  )

  loss_fn = strategy.loss
  σ = mlp.σ
  opt_state = Optimisers.setup(strategy.optimiser,mlp.θ)
  grad = similar(mlp.θ)
  for _ in 1:strategy.epochs
    ForwardDiff.gradient!(grad,p -> loss_fn(_mlp_forward(p,mlp.layers,σ,x),y),mlp.θ)
    opt_state, _ = Optimisers.update!(opt_state,mlp.θ,grad)
  end
  mlp
end

"""
    train_model(strategy::NNStrategy, r::AbstractRealisation, coeff::ConsecutiveParamArray)
      -> NNModel

Builds a [`MLP`](@ref) whose input dimension is inferred from `r` and output
dimension from `coeff`, trains it with `strategy`, and wraps the result in an
[`NNModel`](@ref).
"""
function train_model(strategy::NNStrategy,r::AbstractRealisation,coeff::ConsecutiveParamArray)
  d = length(first(r))
  n = prod(innersize(coeff))
  layers = (d,strategy.layers...,n)
  x = matrix_of_params(r)
  y = reshape(get_all_data(coeff),n,param_length(coeff))
  mlp = MLP(layers;T=eltype(x))
  train!(mlp,x,y,strategy)
  NNModel(mlp)
end

# utils

dimension(μ::Realisation) = length(first(μ))
dimension(μ::TransientRealisation) = dimension(get_params(μ))

function matrix_of_params(r::AbstractRealisation)
  params = zeros(dimension(r),num_params(r))
  matrix_of_params!(params,r)
end

function matrix_of_params!(params,r::AbstractRealisation)
  @check size(params,2) == num_params(r)
  μ = get_params(r)
  @inbounds @views for i in axes(params,2)
    params[:,i] = μ.params[i]
  end
  params
end
