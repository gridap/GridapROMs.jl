"""
    abstract type NeuralNetwork <: Map end

Abstract supertype for neural network models. Any concrete subtype must
implement `(m::T)(x::AbstractMatrix) -> AbstractMatrix` where `x` is a
`(param_dim × batch_size)` matrix of parameters and the output is a
`(output_dim × batch_size)` matrix of predictions.

Wrap external Flux/Lux models with [`GenericNeuralNetwork`](@ref):

    model = GenericNeuralNetwork(flux_chain)     # Flux
    model = GenericNeuralNetwork(p -> lux_apply(chain, p, ps, st))  # Lux closure
"""
abstract type NeuralNetwork <: Map end

function NeuralNetwork(strategy::NNStrategy,args...)
  @abstractmethod
end

function NeuralNetwork(strategy::NNStrategy{MLPType},r::AbstractRealisation,coeff::AbstractParamArray)
  d = dimension(r)
  n = innerlength(coeff)
  layers = (d,strategy.layers...,n)
  MultiLayerPerceptron(layers;T=eltype(y))
end

function TrainedNeuralNetwork(strategy,args...)
  a = NeuralNetwork(strategy,args...)
  train!(a,strategy,args...)
  return a
end

"""
    struct GenericNeuralNetwork{A} <: NeuralNetwork
      model::A
    end

Wraps any callable `A` (Flux chain, Lux closure, plain Julia function) as
an [`NeuralNetwork`](@ref). The wrapped callable must accept a
`(d × k)` parameter matrix and return an `(n × k)` prediction matrix.
"""
struct GenericNeuralNetwork{A} <: NeuralNetwork
  model::A
end

function Arrays.return_cache(m::GenericNeuralNetwork,x::AbstractMatrix)
  return_cache(m.model,x)
end

function Arrays.evaluate!(cache,m::GenericNeuralNetwork,x::AbstractMatrix)
  evaluate!(cache,m.model,x)
end

"""
    struct MultiLayerPerceptron{L,A<:AbstractVector} <: NeuralNetwork
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
prefer injecting a Flux/Lux model via [`GenericNeuralNetwork`](@ref).
"""
struct MultiLayerPerceptron{L,A<:AbstractVector} <: NeuralNetwork
  layers::NTuple{L,Int}
  σ::Function
  θ::A
end

function MultiLayerPerceptron(layers::NTuple{L,Int}; activation::Function=tanh, T::Type=Float64) where L
  nθ = sum(layers[i+1]*(layers[i]+1) for i in 1:L-1)
  θ = randn(T,nθ) .* T(0.01)
  MultiLayerPerceptron(layers,activation,θ)
end

function MultiLayerPerceptron(layers::AbstractVector{Int},args...;kwargs...)
  MultiLayerPerceptron(Tuple(layers),args...;kwargs...)
end

function Arrays.return_cache(m::MultiLayerPerceptron,x::AbstractMatrix)
  T = eltype(m.θ)
  nin,nout, = m.layers
  h = CachedArray(similar(x,T))
  W = CachedArray(zeros(T,nout,nin))
  b = CachedArray(zeros(T,nout))
  z = CachedArray(zeros(T,nout,size(x,2)))
  return (h,W,b,z)
end

function Arrays.evaluate!(cache,m::MultiLayerPerceptron{L},x::AbstractMatrix) where L
  h,W,b,z = cache 
  _init!(h,x)

  offset = 0
  for l in 1:L-1
    nout = m.layers[l+1]
    nin = m.layers[l]
    setsize!(W,(nout,nin))
    setsize!(b,(nout,))
    setsize!(z,(nout,size(x,2)))

    _fill_weights!(W,b,m.θ,offset)
    l < L-1 ? _apply_layer!(z,W,h,b,m.σ) : _apply_layer!(z,W,h,b)

    offset += nout * (nin + 1)
  end

  return h.array 
end

function Arrays.return_cache(m::MultiLayerPerceptron,x::AbstractMatrix,θ::AbstractVector)
  T = eltype(θ)
  nin,nout, = m.layers
  h = CachedArray(similar(x,T))
  W = CachedArray(zeros(T,nout,nin))
  b = CachedArray(zeros(T,nout))
  z = CachedArray(zeros(T,nout,size(x,2)))
  return (h,W,b,z)
end

function Arrays.evaluate!(cache,m::MultiLayerPerceptron{L},x::AbstractMatrix,θ::AbstractVector) where L
  h,W,b,z = cache 
  _init!(h,x)

  offset = 0
  for l in 1:L-1
    nout = m.layers[l+1]
    nin = m.layers[l]
    setsize!(W,(nout,nin))
    setsize!(b,(nout,))
    setsize!(z,(nout,size(x,2)))

    _fill_weights!(W,b,θ,offset)
    l < L-1 ? _apply_layer!(z,W,h,b,m.σ) : _apply_layer!(z,W,h,b)

    offset += nout * (nin + 1)
  end

  return h.array 
end

"""
    train!(mlp::MultiLayerPerceptron, x::AbstractMatrix, y::AbstractMatrix, strategy::NNStrategy) -> MultiLayerPerceptron

Trains `mlp` with the optimiser, loss, and epoch count from `strategy` using
ForwardDiff for gradients. `x` is `(param_dim × n_samples)`, `y` is
`(output_dim × n_samples)`. Updates `mlp.θ` in-place and returns `mlp`.
"""
function train!(
  mlp::MultiLayerPerceptron,
  strategy::NNStrategy,
  x::AbstractMatrix,
  y::AbstractMatrix
  )

  opt_state = Optimisers.setup(strategy.optimiser,mlp.θ)
  grad = similar(mlp.θ)
  p = ForwardDiff.GradientConfig(nothing,mlp.θ).duals
  cache = return_cache(mlp,x,p) 
  ynn(p) = evaluate!(cache,mlp,x,p)
  
  for _ in 1:strategy.epochs
    ForwardDiff.gradient!(grad,p -> strategy.loss(ynn(p),y),mlp.θ)
    opt_state,_ = Optimisers.update!(opt_state,mlp.θ,grad)
  end

  return mlp
end

function train!(
  mlp::MultiLayerPerceptron,
  strategy::NNStrategy,
  r::AbstractRealisation,
  coeff::ConsecutiveParamArray
  )

  x = matrix_of_params(r)
  y = get_all_data(coeff)
  train!(mlp,strategy,x,y)
  return mlp
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

function _init!(h,x)
  setsize!(h,size(x))
  copyto!(h.array,x)
end

function _fill_weights!(W,b,θ,offset)
  Wa = W.array 
  ba = b.array
  nout,nin = size(W)
  @inbounds for i in 1:nout 
    for j in 1:nin
      Wa[i,j] = θ[offset + (j-1)*nout + i]
    end
    ba[i] = θ[offset + nout*nin + i]
  end
end

function _apply_layer!(z,W,h,b,σ=identity)
  mul!(z.array,W.array,h.array)
  setsize!(h,size(z))
  @inbounds for i in axes(z,1)
    for j in axes(z,2)
      h.array[i,j] = σ(z.array[i,j] + b.array[i])
    end
  end
  h
end