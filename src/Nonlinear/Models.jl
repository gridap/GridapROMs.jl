"""
    abstract type NeuralNetwork <: Map end

Abstract supertype for neural network models. Any concrete subtype must
implement `(a::T)(x::AbstractMatrix) -> AbstractMatrix` where `x` is a
`(param_dim × batch_size)` matrix of parameters and the output is a
`(output_dim × batch_size)` matrix of predictions.

Wrap external Flux/Lux models with [`GenericNeuralNetwork`](@ref):

    model = GenericNeuralNetwork(flux_chain)     # Flux
    model = GenericNeuralNetwork(p -> lux_apply(chain, p, ps, st))  # Lux closure
"""
abstract type NeuralNetwork <: Map end

function NeuralNetwork(s::NNStrategy,args...)
  @abstractmethod
end

function NeuralNetwork(s::NNStrategy{MLPType},r::AbstractRealisation,coeff)
  d = dimension(r)
  y = _get_data(coeff)
  n = size(y,1)
  layers = (d,s.layers...,n)
  MultiLayerPerceptron(layers;T=eltype(y))
end

"""
    TrainedNeuralNetwork(s::NNStrategy, r::AbstractRealisation, coeff) -> NeuralNetwork

Constructs a [`NeuralNetwork`](@ref) from `s` and immediately trains it on
the `(r, coeff)` data pair. For `MLPType` strategies a [`MultiLayerPerceptron`](@ref)
is built whose output dimension is inferred from `coeff`.
"""
function TrainedNeuralNetwork(s,args...)
  a = NeuralNetwork(s,args...)
  train!(a,s,args...)
  return a
end

get_weights(a::NeuralNetwork) = @abstractmethod

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

function Arrays.return_cache(a::GenericNeuralNetwork,x::AbstractMatrix)
  return_cache(a.model,x)
end

function Arrays.evaluate!(cache,a::GenericNeuralNetwork,x::AbstractMatrix)
  evaluate!(cache,a.model,x)
end

"""
    struct MultiLayerPerceptron{L,A<:AbstractVector} <: NeuralNetwork
      layers::NTuple{L,Int}
      activation::Function
      θ::A
    end

A minimal dense feed-forward network whose parameters are stored as a flat
vector `θ` for ForwardDiff compatibility. Architecture is determined by
`layers = (d, h₁, h₂, …, n)` where `d` is input dim and `n` is output dim.
Hidden layers apply `activation` (default: `tanh`); the output layer is linear.

Use [`NNStrategy`](@ref) with `MLPType()` (the default) to build and train one
automatically via [`TrainedNeuralNetwork`](@ref). For large networks or long
training runs prefer injecting a Flux/Lux model via [`GenericNeuralNetwork`](@ref).
"""
struct MultiLayerPerceptron{L,A<:AbstractVector} <: NeuralNetwork
  layers::NTuple{L,Int}
  activation::Function
  θ::A
end

function MultiLayerPerceptron(layers::NTuple{L,Int};activation::Function=tanh,T::Type=Float64) where L
  nθ = sum(layers[i+1]*(layers[i]+1) for i in 1:L-1)
  θ = randn(T,nθ) .* T(0.01)
  MultiLayerPerceptron(layers,activation,θ)
end

function MultiLayerPerceptron(layers::AbstractVector{Int},args...;kwargs...)
  MultiLayerPerceptron(Tuple(layers),args...;kwargs...)
end

get_weights(a::MultiLayerPerceptron) = a.θ

function Arrays.return_cache(a::MultiLayerPerceptron,x::AbstractMatrix)
  return_cache(a,x,a.θ)
end

function Arrays.evaluate!(cache,a::MultiLayerPerceptron,x::AbstractMatrix) 
  evaluate!(cache,a,x,a.θ)
end

function Arrays.return_cache(a::MultiLayerPerceptron,x::AbstractMatrix,θ::AbstractVector)
  T = eltype(θ)
  nin,nout, = a.layers
  h = CachedArray(similar(x,T))
  W = CachedArray(zeros(T,nout,nin))
  b = CachedArray(zeros(T,nout))
  z = CachedArray(zeros(T,nout,size(x,2)))
  return (h,W,b,z)
end

function Arrays.evaluate!(cache,a::MultiLayerPerceptron{L},x::AbstractMatrix,θ::AbstractVector) where L
  h,W,b,z = cache 
  _init!(h,x)

  offset = 0
  for l in 1:L-1
    nout = a.layers[l+1]
    nin = a.layers[l]
    setsize!(W,(nout,nin))
    setsize!(b,(nout,))
    setsize!(z,(nout,size(x,2)))

    _fill_weights!(W,b,θ,offset)
    l < L-1 ? _apply_layer!(z,W,h,b,a.activation) : _apply_layer!(z,W,h,b)

    offset += nout * (nin + 1)
  end

  return h.array 
end

"""
    train!(a::MultiLayerPerceptron, s::NNStrategy, x::AbstractMatrix, y::AbstractMatrix) -> MultiLayerPerceptron

Trains `a` using the hyperparameters in `s` with ForwardDiff for gradients.
`x` is `(param_dim × n_samples)`, `y` is `(output_dim × n_samples)`. Updates
`a.θ` in-place and returns `a`.

- Mini-batching: when `s.batch_size > 0` samples are shuffled and looped
  each epoch; the last incomplete batch is dropped.
- LR scheduling: when `s.lr_schedule` is not `nothing`, it is called as
  `lr_schedule(epoch, total_epochs)` and applied via `Optimisers.adjust!`.
- Early stopping: when `s.patience > 0`, a `s.val_fraction`
  fraction of samples is held out; training stops when the validation loss fails
  to improve for `patience` consecutive epochs.
"""
function train!(
  a::MultiLayerPerceptron,
  s::NNStrategy,
  x::AbstractMatrix,
  y::AbstractMatrix
  )

  k = size(x,2)
  early_stop = s.patience > 0

  if early_stop
    n_val = max(1,round(Int,k*s.val_fraction))
    n_tr = k - n_val
    @check n_tr > 0 "val_fraction=$(s.val_fraction) leaves no training samples"
    perm0 = randperm(k)
    x_tr = view(x,:,perm0[1:n_tr])
    y_tr = view(y,:,perm0[1:n_tr])
    x_val = view(x,:,perm0[n_tr+1:k])
    y_val = view(y,:,perm0[n_tr+1:k])
  else
    x_tr,y_tr = x,y
    n_tr = k
  end

  bs = s.batch_size == 0 ? n_tr : min(s.batch_size,n_tr)
  full_batch = bs == n_tr

  opt_state = Optimisers.setup(s.optimiser,a.θ)
  grad = similar(a.θ)
  cfg = ForwardDiff.GradientConfig(nothing,a.θ)
  cache = return_cache(a,full_batch ? x_tr : x_tr[:,1:bs],cfg.duals)
  cache_val = early_stop ? return_cache(a,x_val) : nothing

  ynn(p) = evaluate!(cache,a,x_tr,p)
  ynn_b(x,p) = evaluate!(cache,a,x,p)

  best_val = typemax(Float64)
  patience_count = 0

  for epoch in 1:s.epochs
    _adjust_lr!(opt_state,s,epoch)
    if full_batch
      ForwardDiff.gradient!(grad,p -> s.loss(ynn(p),y_tr),a.θ,cfg)
      opt_state,_ = Optimisers.update!(opt_state,a.θ,grad)
    else
      perm = randperm(n_tr)
      for start in 1:bs:(n_tr-bs+1)
        idx = view(perm,start:start+bs-1)
        xb = view(x_tr,:,idx)
        yb = view(y_tr,:,idx)
        ForwardDiff.gradient!(grad,p -> s.loss(ynn_b(xb,p),yb),a.θ,cfg)
        opt_state,_ = Optimisers.update!(opt_state,a.θ,grad)
      end
    end
    if early_stop
      val_loss = s.loss(evaluate!(cache_val,a,x_val),y_val)
      if val_loss < best_val
        best_val = val_loss
        patience_count = 0
      else
        patience_count += 1
        patience_count >= s.patience && break
      end
    end
  end

  return a
end

function train!(
  a::MultiLayerPerceptron,
  s::NNStrategy,
  r::AbstractRealisation,
  coeff
  )

  x = matrix_of_params(r)
  y = _get_data(coeff)
  train!(a,s,x,y)
  return a
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

_get_data(a) = get_all_data(a)
_get_data(a::AbstractParamMatrix) = reshape(get_all_data(a),innerlength(a),:)
_get_data(a::AbstractMatrix) = a
_get_data(a::AbstractArray{T,3}) where T = reshape(a,:,size(a,3))

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

function _apply_layer!(z,W,h,b,activation=identity)
  mul!(z.array,W.array,h.array)
  setsize!(h,size(z))
  @inbounds for i in axes(z,1)
    for j in axes(z,2)
      h.array[i,j] = activation(z.array[i,j] + b.array[i])
    end
  end
  h
end

function _adjust_lr!(state,s::NNStrategy,epoch::Int)
  isnothing(s.lr_schedule) && return
  Optimisers.adjust!(state,s.lr_schedule(epoch,s.epochs))
end