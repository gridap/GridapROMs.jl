abstract type NNType end
struct GenericNNType <: NNType end
struct MLPType <: NNType end

"""
    struct NNStrategy{A,L,O,F,S}
      type::A
      layers::NTuple{L,Int}
      optimiser::O
      loss::F
      epochs::Int
      batch_size::Int
      lr_schedule::S
      patience::Int
      val_fraction::Float64
    end

Bundles all training hyperparameters for a [`MultiLayerPerceptron`](@ref):
- `layers`: widths of the hidden layers as an `NTuple` (e.g. `(64, 64)`)
- `optimiser`: an `Optimisers.jl` rule (default: `Adam(1e-3)`); weight decay is
  folded in at construction time via `OptimiserChain(..., WeightDecay(λ))`
- `loss`: `loss(ŷ, y) -> scalar`; built-ins are [`loss_mse`](@ref) and [`loss_mae`](@ref)
- `epochs`: maximum number of gradient steps per run
- `batch_size`: mini-batch column count; `0` (default) uses all training samples (full-batch)
- `lr_schedule`: `nothing` (default, constant lr) or a callable
  `(epoch::Int, total_epochs::Int) -> Float64` invoked each epoch via `Optimisers.adjust!`
- `patience`: number of consecutive epochs without validation-loss improvement before
  stopping early; `0` (default) disables early stopping and runs all `epochs`
- `val_fraction`: fraction of samples held out for validation when `patience > 0`
  (default `0.1`; ignored when `patience == 0`)

Pass to [`NNHyperReduction`](@ref) or [`NNOperatorReduction`](@ref) via the `strategy` keyword.
"""
struct NNStrategy{A,L,O,F,S}
  type::A
  layers::NTuple{L,Int}
  optimiser::O
  loss::F
  epochs::Int
  batch_size::Int
  lr_schedule::S
  patience::Int
  val_fraction::Float64
end

"""
    NNStrategy(; type=MLPType(), layers=(64,64), lr=1e-3, optimiser=Adam(lr),
               loss=loss_mse, epochs=1000, weight_decay=0.0,
               batch_size=0, lr_schedule=nothing, patience=0, val_fraction=0.1)
"""
function NNStrategy(;
  type::NNType=MLPType(),
  layers::Union{AbstractVector,Tuple}=(64,64),
  lr::Real=1e-3,
  optimiser=Optimisers.Adam(lr),
  loss=loss_mse,
  epochs::Int=1000,
  weight_decay::Real=0.0,
  batch_size::Int=0,
  lr_schedule=nothing,
  patience::Int=0,
  val_fraction::Real=0.1
  )

  decay = Optimisers.WeightDecay(weight_decay)
  opt = weight_decay > 0 ? Optimisers.OptimiserChain(optimiser,decay) : optimiser
  NNStrategy(type,Tuple(layers),opt,loss,epochs,batch_size,lr_schedule,patience,Float64(val_fraction))
end

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

function loss_mse(ŷ,y)
  @check length(ŷ) == length(y)
  s = 0.0
  @inbounds for i in eachindex(y)
    d = ŷ[i] - y[i]
    s += d * d
  end
  s / length(y)
end

function loss_mae(ŷ,y)
  @check length(ŷ) == length(y)
  s = 0.0
  @inbounds for i in eachindex(y)
    s += abs(ŷ[i] - y[i])
  end
  s / length(y)
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

# ===== Nonlinear reduced spaces: auto-encoders and auto-decoders ============

struct AutoEncoderType <: NNType end
struct VariationalAutoEncoderType <: NNType end
struct AutoDecoderType <: NNType end

# Differentiable MLP forward pass using weight layout of MultiLayerPerceptron.
# Allocating but ForwardDiff-transparent; used during training only.
function _mlp_apply(
  layers::NTuple{L,Int}, activation::Function, θ::AbstractVector, x::AbstractMatrix
) where L
  T = eltype(θ)
  h = T === eltype(x) ? x : T.(x)
  offset = 0
  for l in 1:L-1
    nout = layers[l+1]; nin = layers[l]
    W = reshape(view(θ, offset+1:offset+nout*nin), nout, nin)
    b = view(θ, offset+nout*nin+1:offset+nout*(nin+1))
    z = W * h .+ b
    h = l < L-1 ? activation.(z) : z
    offset += nout*(nin+1)
  end
  h
end

"""
    AutoEncoder{E,D} <: NeuralNetwork

Encoder–decoder pair for unsupervised dimensionality reduction. Both
sub-networks are [`MultiLayerPerceptron`](@ref)s with parameters stored as a
flat vector for ForwardDiff-based joint training.

`evaluate!(cache, ae, z)` applies the **decoder** (latent → high-dim). Use
[`encode`](@ref) for the encoder direction (high-dim → latent).

**Convolutional variants**: wrap a Lux/Flux CNN chain in
[`GenericNeuralNetwork`](@ref) and supply a custom `train!` using
`Lux.Training.single_train_step!` with `AutoZygote`.

Build and train in one call via [`TrainedNeuralNetwork`](@ref):

    s = NNStrategy(type=AutoEncoderType(), layers=(128,64,16))
    ae = TrainedNeuralNetwork(s, r, snapshots)

where `s.layers = (h₁, h₂, …, latent_dim)` defines the encoder hidden widths;
the decoder mirrors them symmetrically.
"""
struct AutoEncoder{E<:MultiLayerPerceptron,D<:MultiLayerPerceptron} <: NeuralNetwork
  encoder::E
  decoder::D
end

function AutoEncoder(enc_layers, dec_layers; activation::Function=tanh, T::Type=Float64)
  AutoEncoder(
    MultiLayerPerceptron(Tuple(enc_layers); activation, T),
    MultiLayerPerceptron(Tuple(dec_layers); activation, T),
  )
end

function NeuralNetwork(s::NNStrategy{AutoEncoderType}, ::AbstractRealisation, coeff)
  y = _get_data(coeff); n_h = size(y,1); T = eltype(y)
  enc_layers = (n_h, s.layers...)
  dec_layers = (last(s.layers), reverse(s.layers[1:end-1])..., n_h)
  AutoEncoder(enc_layers, dec_layers; T)
end

function Arrays.return_cache(a::AutoEncoder, z::AbstractMatrix)
  return_cache(a.decoder, z)
end

function Arrays.evaluate!(cache, a::AutoEncoder, z::AbstractMatrix)
  evaluate!(cache, a.decoder, z, a.decoder.θ)
end

function encode(a::AutoEncoder, X::AbstractMatrix)
  cache = return_cache(a.encoder, X)
  evaluate!(cache, a.encoder, X, a.encoder.θ)
end

function decode(a::AutoEncoder, Z::AbstractMatrix)
  cache = return_cache(a.decoder, Z)
  evaluate!(cache, a.decoder, Z, a.decoder.θ)
end

"""
    train!(ae::AutoEncoder, s::NNStrategy, X::AbstractMatrix) -> AutoEncoder

Train `ae` on the snapshot matrix `X` (`n_h × k`) by minimising the
reconstruction loss `s.loss(decoder(encoder(X)), X)`. Encoder and decoder
parameters are optimised jointly via ForwardDiff. Supports mini-batching,
LR scheduling, and early stopping from `s`.
"""
function train!(a::AutoEncoder, s::NNStrategy, X::AbstractMatrix)
  nθe = length(a.encoder.θ); nθd = length(a.decoder.θ)
  θ = vcat(a.encoder.θ, a.decoder.θ)
  k = size(X,2)
  early_stop = s.patience > 0

  if early_stop
    n_val = max(1, round(Int, k*s.val_fraction)); n_tr = k - n_val
    @check n_tr > 0 "val_fraction=$(s.val_fraction) leaves no training samples"
    perm0 = randperm(k)
    X_tr = view(X,:,perm0[1:n_tr]); X_val = view(X,:,perm0[n_tr+1:k])
  else
    X_tr = X; n_tr = k; X_val = nothing
  end
  bs = s.batch_size == 0 ? n_tr : min(s.batch_size, n_tr); full = bs == n_tr

  function recon_loss(p, Xb)
    pe = view(p,1:nθe); pd = view(p,nθe+1:nθe+nθd)
    Z = _mlp_apply(a.encoder.layers, a.encoder.activation, pe, Xb)
    X̂ = _mlp_apply(a.decoder.layers, a.decoder.activation, pd, Z)
    s.loss(X̂, Xb)
  end

  opt_state = Optimisers.setup(s.optimiser, θ)
  grad = similar(θ); best_val = typemax(Float64); patience_count = 0

  for epoch in 1:s.epochs
    _adjust_lr!(opt_state, s, epoch)
    if full
      ForwardDiff.gradient!(grad, p -> recon_loss(p, X_tr), θ)
      opt_state,_ = Optimisers.update!(opt_state, θ, grad)
    else
      perm = randperm(n_tr)
      for start in 1:bs:(n_tr-bs+1)
        Xb = view(X_tr, :, view(perm, start:start+bs-1))
        ForwardDiff.gradient!(grad, p -> recon_loss(p, Xb), θ)
        opt_state,_ = Optimisers.update!(opt_state, θ, grad)
      end
    end
    if early_stop
      val = recon_loss(θ, X_val)
      if val < best_val; best_val = val; patience_count = 0
      else; patience_count += 1; patience_count >= s.patience && break
      end
    end
  end

  copyto!(a.encoder.θ, view(θ, 1:nθe))
  copyto!(a.decoder.θ, view(θ, nθe+1:nθe+nθd))
  a
end

function train!(a::AutoEncoder, s::NNStrategy, ::AbstractRealisation, coeff)
  train!(a, s, _get_data(coeff))
end

"""
    VariationalAutoEncoder{E,D} <: NeuralNetwork

VAE with reparameterisation trick. The encoder outputs `[μ; log σ²]` (2 ×
latent_dim values per sample); during training a latent sample
`z = μ + ε·exp(log σ²/2)` with ε ~ N(0,I) is passed to the decoder.

Training loss: `recon_loss + β · KL`, where
`KL = -½ mean(1 + log σ² - μ² - σ²)`.

`NNStrategy` interpretation: `layers = (h₁, …, h_{L-1}, latent_dim)`;
the encoder hidden widths are `(h₁, …, h_{L-1})` and the decoder mirrors them.
`β` controls the KL weight (keyword to `NeuralNetwork`/`TrainedNeuralNetwork`).

**Convolutional variants**: use `GenericNeuralNetwork` wrapping a Lux/Flux CNN.
"""
struct VariationalAutoEncoder{E<:MultiLayerPerceptron,D<:MultiLayerPerceptron} <: NeuralNetwork
  encoder::E
  decoder::D
  β::Float64
end

function VariationalAutoEncoder(enc_layers, dec_layers; β=1.0, activation::Function=tanh, T::Type=Float64)
  VariationalAutoEncoder(
    MultiLayerPerceptron(Tuple(enc_layers); activation, T),
    MultiLayerPerceptron(Tuple(dec_layers); activation, T),
    Float64(β),
  )
end

function NeuralNetwork(s::NNStrategy{VariationalAutoEncoderType}, ::AbstractRealisation, coeff; β=1.0)
  y = _get_data(coeff); n_h = size(y,1); T = eltype(y)
  latent_dim = last(s.layers)
  hidden = s.layers[1:end-1]
  enc_layers = (n_h, hidden..., 2*latent_dim)
  dec_layers = (latent_dim, reverse(hidden)..., n_h)
  VariationalAutoEncoder(enc_layers, dec_layers; β, T)
end

function Arrays.return_cache(a::VariationalAutoEncoder, z::AbstractMatrix)
  return_cache(a.decoder, z)
end

function Arrays.evaluate!(cache, a::VariationalAutoEncoder, z::AbstractMatrix)
  evaluate!(cache, a.decoder, z, a.decoder.θ)
end

function encode(a::VariationalAutoEncoder, X::AbstractMatrix)
  latent_dim = a.decoder.layers[1]
  cache = return_cache(a.encoder, X)
  enc_out = evaluate!(cache, a.encoder, X, a.encoder.θ)
  μ = enc_out[1:latent_dim, :]
  log_var = enc_out[latent_dim+1:end, :]
  ε = randn(eltype(a.encoder.θ), latent_dim, size(X,2))
  z = μ .+ ε .* exp.(log_var ./ 2)
  (μ, log_var, z)
end

function train!(a::VariationalAutoEncoder, s::NNStrategy, X::AbstractMatrix)
  nθe = length(a.encoder.θ); nθd = length(a.decoder.θ)
  θ = vcat(a.encoder.θ, a.decoder.θ)
  k = size(X,2); latent_dim = a.decoder.layers[1]
  opt_state = Optimisers.setup(s.optimiser, θ)
  grad = similar(θ); best_val = typemax(Float64); patience_count = 0
  early_stop = s.patience > 0

  function vae_loss(p, ε)
    pe = view(p,1:nθe); pd = view(p,nθe+1:nθe+nθd)
    enc_out = _mlp_apply(a.encoder.layers, a.encoder.activation, pe, X)
    μ = enc_out[1:latent_dim, :]; log_var = enc_out[latent_dim+1:end, :]
    z = μ .+ ε .* exp.(log_var ./ 2)
    X̂ = _mlp_apply(a.decoder.layers, a.decoder.activation, pd, z)
    recon = s.loss(X̂, X)
    kl = -sum(1 .+ log_var .- μ .^ 2 .- exp.(log_var)) / (2*k)
    recon + a.β * kl
  end

  for epoch in 1:s.epochs
    _adjust_lr!(opt_state, s, epoch)
    ε = randn(eltype(θ), latent_dim, k)
    ForwardDiff.gradient!(grad, p -> vae_loss(p, ε), θ)
    opt_state,_ = Optimisers.update!(opt_state, θ, grad)
    if early_stop
      val = vae_loss(θ, randn(eltype(θ), latent_dim, k))
      if val < best_val; best_val = val; patience_count = 0
      else; patience_count += 1; patience_count >= s.patience && break
      end
    end
  end

  copyto!(a.encoder.θ, view(θ, 1:nθe))
  copyto!(a.decoder.θ, view(θ, nθe+1:nθe+nθd))
  a
end

function train!(a::VariationalAutoEncoder, s::NNStrategy, ::AbstractRealisation, coeff)
  train!(a, s, _get_data(coeff))
end

"""
    AutoDecoder{D,A} <: NeuralNetwork

Decoder-only model (Park et al., 2019). Per-sample latent codes are stored in
`latent_codes` (`latent_dim × n_train`) and optimised **jointly** with the
decoder parameters during training. Inference for unseen samples requires
fitting a latent code via [`infer_latent`](@ref).

`evaluate!` applies the decoder (`latent_dim × k → n_h × k`).

`NNStrategy` interpretation: `layers = (h₁, …, latent_dim)` defines the
decoder from last to first (i.e. decoder layers are
`(latent_dim, reverse(h₁,…,h_{L-1})…, n_h)`).
"""
struct AutoDecoder{D<:MultiLayerPerceptron,A<:AbstractMatrix} <: NeuralNetwork
  decoder::D
  latent_codes::A
end

function AutoDecoder(dec_layers, n_train::Int; activation::Function=tanh, T::Type=Float64)
  d = MultiLayerPerceptron(Tuple(dec_layers); activation, T)
  latent_dim = first(dec_layers)
  Z = randn(T, latent_dim, n_train) .* T(0.01)
  AutoDecoder(d, Z)
end

function NeuralNetwork(s::NNStrategy{AutoDecoderType}, ::AbstractRealisation, coeff)
  y = _get_data(coeff); n_h = size(y,1); k = size(y,2); T = eltype(y)
  latent_dim = last(s.layers)
  dec_layers = (latent_dim, reverse(s.layers[1:end-1])..., n_h)
  AutoDecoder(dec_layers, k; T)
end

function Arrays.return_cache(a::AutoDecoder, z::AbstractMatrix)
  return_cache(a.decoder, z)
end

function Arrays.evaluate!(cache, a::AutoDecoder, z::AbstractMatrix)
  evaluate!(cache, a.decoder, z, a.decoder.θ)
end

"""
    train!(ad::AutoDecoder, s::NNStrategy, X::AbstractMatrix) -> AutoDecoder

Jointly optimise decoder parameters and all latent codes to minimise
`s.loss(decoder(Z), X)`. After training, `ad.latent_codes[:,i]` is the
learned latent representation of snapshot `X[:,i]`.
"""
function train!(a::AutoDecoder, s::NNStrategy, X::AbstractMatrix)
  nθd = length(a.decoder.θ); latent_dim, k = size(a.latent_codes)
  @check size(X,2) == k "AutoDecoder.latent_codes columns must match number of snapshots"
  θ = vcat(a.decoder.θ, vec(a.latent_codes))
  opt_state = Optimisers.setup(s.optimiser, θ)
  grad = similar(θ)

  function recon_loss(p)
    pd = view(p, 1:nθd)
    Z = reshape(view(p, nθd+1:nθd+latent_dim*k), latent_dim, k)
    X̂ = _mlp_apply(a.decoder.layers, a.decoder.activation, pd, Z)
    s.loss(X̂, X)
  end

  for epoch in 1:s.epochs
    _adjust_lr!(opt_state, s, epoch)
    ForwardDiff.gradient!(grad, recon_loss, θ)
    opt_state,_ = Optimisers.update!(opt_state, θ, grad)
  end

  copyto!(a.decoder.θ, view(θ, 1:nθd))
  copyto!(vec(a.latent_codes), view(θ, nθd+1:nθd+latent_dim*k))
  a
end

function train!(a::AutoDecoder, s::NNStrategy, ::AbstractRealisation, coeff)
  train!(a, s, _get_data(coeff))
end

"""
    infer_latent(a::AutoDecoder, x_target::AbstractVector, s::NNStrategy) -> AbstractVector

Fit a latent code `z` for an unseen snapshot `x_target` by minimising
`s.loss(decoder(z), x_target)` with the decoder weights fixed. Uses the
optimiser and epoch count from `s`.
"""
function infer_latent(a::AutoDecoder, x_target::AbstractVector, s::NNStrategy)
  T = eltype(a.decoder.θ); latent_dim = size(a.latent_codes, 1)
  z = randn(T, latent_dim) .* T(0.01)
  X_t = reshape(x_target, :, 1)
  opt_state = Optimisers.setup(s.optimiser, z)
  grad = similar(z)
  for epoch in 1:s.epochs
    _adjust_lr!(opt_state, s, epoch)
    ForwardDiff.gradient!(grad, z_ -> begin
      X̂ = _mlp_apply(a.decoder.layers, a.decoder.activation, a.decoder.θ, reshape(z_, :, 1))
      s.loss(X̂, X_t)
    end, z)
    opt_state,_ = Optimisers.update!(opt_state, z, grad)
  end
  z
end

"""
    struct DeepONet{F}
      branch_layers::Tuple{Vararg{Int}}
      trunk_layers::Tuple{Vararg{Int}}
      activation::F
    end

Architectural configuration for a DeepONet network.
"""
struct DeepONet{F}
  branch_layers::Tuple{Vararg{Int}}
  trunk_layers::Tuple{Vararg{Int}}
  activation::F
end

function DeepONet(;branch_layers,trunk_layers,activation=tanh)
  DeepONet(Tuple(branch_layers),Tuple(trunk_layers),activation)
end