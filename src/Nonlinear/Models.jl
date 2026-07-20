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
    struct NNModel{M} <: AbstractNNModel
      model::M
    end

Wraps any callable `M` (Flux chain, Lux closure, plain Julia function) as
an [`AbstractNNModel`](@ref). The wrapped callable must accept a
`(d × k)` parameter matrix and return an `(n × k)` prediction matrix.
"""
struct NNModel{M} <: AbstractNNModel
  model::M
end

function Gridap.return_cache(m::NNModel,x::AbstractMatrix)
  return_cache(m.model,x)
end

function Gridap.evaluate!(cache,m::NNModel,x::AbstractMatrix)
  evaluate!(cache,m.model,x)
end

"""
    struct MLP{P,S} <: AbstractNNModel

A minimal dense feed-forward network whose parameters are stored as a flat
vector `params` for ForwardDiff compatibility. Architecture is determined by
`sizes = (d, h₁, h₂, …, n)` where `d` is input dim and `n` is output dim.
Hidden layers use `tanh`; the output layer is linear.

Use [`build_mlp_factory`](@ref) to construct and train one as part of
[`NNHyperReduction`](@ref).  For large networks or long training runs,
prefer injecting a Flux/Lux model via [`NNModel`](@ref).
"""
struct MLP{P<:AbstractVector,S} <: AbstractNNModel
  params::P
  sizes::S    # NTuple{L,Int} with L = nlayers + 1
end

function MLP(sizes::AbstractVector{Int};T::Type=Float64)
  n_params = sum(sizes[i+1] * (sizes[i] + 1) for i in 1:length(sizes)-1)
  params = randn(T,n_params) .* T(0.01)
  MLP(params,Tuple(sizes))
end

function _mlp_forward(params::AbstractVector,sizes::Tuple,x::AbstractMatrix)
  offset = 0
  h = x
  nlayers = length(sizes) - 1
  for l in 1:nlayers
    n_out = sizes[l+1]
    n_in = sizes[l]
    W = reshape(params[offset+1:offset+n_out*n_in],n_out,n_in)
    offset += n_out * n_in
    b = params[offset+1:offset+n_out]
    offset += n_out
    z = W * h .+ b
    h = l < nlayers ? tanh.(z) : z
  end
  h
end

(m::MLP)(x::AbstractMatrix) = _mlp_forward(m.params,m.sizes,x)

"""
    train!(mlp::MLP, x::AbstractMatrix, y::AbstractMatrix;
           epochs=1000, lr=1e-3) -> MLP

Trains `mlp` with full-batch gradient descent using ForwardDiff for gradients.
`x` is `(param_dim × n_samples)`, `y` is `(output_dim × n_samples)`.
Returns `mlp` in-place.
"""
function train!(
  mlp::MLP,
  x::AbstractMatrix,
  y::AbstractMatrix;
  epochs::Int=1000,
  lr::Real=1e-3
  )

  loss(p) = sum(abs2,_mlp_forward(p,mlp.sizes,x) .- y) / length(y)
  grad = similar(mlp.params)
  for _ in 1:epochs
    ForwardDiff.gradient!(grad,loss,mlp.params)
    mlp.params .-= lr .* grad
  end
  mlp
end

"""
    build_mlp_factory(; hidden_sizes=[64,64], epochs=2000, lr=1e-3) -> Function

Returns a factory `(r::Realisation, coeff::ConsecutiveParamArray) -> NNModel`
that constructs and trains a [`MLP`](@ref) on the provided training
data. Pass the result as `nn_factory` to [`NNHyperReduction`](@ref) or
[`NNOpRegression`](@ref).

For production use, replace this with a factory that wraps a Flux or Lux
model for faster training and larger capacity.
"""
function build_mlp_factory(;
  hidden_sizes::AbstractVector{Int}=[64,64],
  epochs::Int=2000,
  lr::Real=1e-3
  )

  function factory(r::Realisation,coeff::ConsecutiveParamArray)
    d = length(first(r))
    n = prod(innersize(coeff))
    sizes = [d;hidden_sizes;n]
    x = matrix_of_params(r)
    y = reshape(get_all_data(coeff),n,param_length(coeff))
    mlp = MLP(sizes;T=eltype(x))
    train!(mlp,x,y;epochs,lr)
    NNModel(mlp)
  end

  factory
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