# Training loop for neural operator ROMs.
#
# Follows the standard Lux.jl pattern: explicit parameters, Zygote AD,
# Optimisers.jl for parameter updates. The training data comes from
# SnapshotData produced by collect_snapshots.

"""
    NormalizationStats

Per-feature mean/std for input and output normalization.
Normalization is critical for training stability — FEM DOF values can span
orders of magnitude, and parameter ranges are problem-dependent.
"""
struct NormalizationStats
  mean::Vector{Float32}
  std::Vector{Float32}
end

function normalize(s::NormalizationStats,x::AbstractMatrix)
  return (x .- s.mean) ./ s.std
end

function denormalize(s::NormalizationStats,x::AbstractMatrix)
  return x .* s.std .+ s.mean
end

function denormalize(s::NormalizationStats,x::AbstractVector)
  return x .* s.std .+ s.mean
end

function compute_normalization(x::AbstractMatrix)
  m = vec(mean(x,dims=2))
  s = vec(std(x,dims=2))
  s[s .< 1f-8] .= 1f0
  return NormalizationStats(Float32.(m),Float32.(s))
end

"""
    TrainingConfig(; kwargs...)

Hyperparameters for neural operator training.
"""
struct TrainingConfig
  epochs::Int
  lr::Float64
  batch_size::Int
  validation_split::Float64
  patience::Int
  verbose::Bool
end

function TrainingConfig(;
  epochs=500,lr=1e-3,batch_size=16,
  validation_split=0.1,patience=50,verbose=true
)
  TrainingConfig(epochs,lr,batch_size,validation_split,patience,verbose)
end

"""
    TrainingResult

Output of `train_operator`: the trained model, its parameters/state,
normalization statistics, the precomputed trunk matrix, and loss history.
"""
struct TrainingResult
  model::DeepONetLayer
  params::Any
  state::Any
  trunk_matrix::Matrix{Float32}
  input_norm::NormalizationStats
  output_norm::NormalizationStats
  train_losses::Vector{Float64}
  val_losses::Vector{Float64}
  best_epoch::Int
end

"""
    train_operator(data::SnapshotData, model::DeepONetLayer;
                   config=TrainingConfig(), rng=Random.default_rng()) -> TrainingResult

Train a DeepONet on FEM snapshot data.

The training loop:
1. Normalizes parameters and DOF vectors
2. Precomputes trunk matrix T from mesh coordinates (recomputed each epoch
   since trunk weights change)
3. Minimizes MSE between predicted and true DOF vectors
4. Uses early stopping on a held-out validation set
"""
function train_operator(
  data::SnapshotData,
  model::DeepONetLayer;
  config::TrainingConfig=TrainingConfig(),
  rng::AbstractRNG=Random.default_rng()
)
  M = size(data.parameters,2)

  # Normalization
  input_norm = compute_normalization(data.parameters)
  output_norm = compute_normalization(data.solutions)

  params_n = Float32.(normalize(input_norm,data.parameters))
  sols_n = Float32.(normalize(output_norm,data.solutions))
  coords_f32 = Float32.(data.coordinates)

  # Train/val split
  n_val = max(1,round(Int,M * config.validation_split))
  n_train = M - n_val
  perm = randperm(rng,M)
  train_idx = perm[1:n_train]
  val_idx = perm[n_train+1:end]

  # Initialize model
  ps,st = Lux.setup(rng,model)
  opt_state = Optimisers.setup(Adam(Float32(config.lr)),ps)

  # Tracking
  train_losses = Float64[]
  val_losses = Float64[]
  best_val_loss = Inf
  best_ps = deepcopy(ps)
  best_epoch = 0
  patience_counter = 0

  for epoch in 1:config.epochs
    # Precompute trunk matrix with current trunk weights
    trunk_matrix = precompute_trunk_matrix(model,coords_f32,ps,st)

    # Mini-batch SGD
    shuffle!(rng,train_idx)
    epoch_loss = 0.0
    n_batches = 0

    for batch_start in 1:config.batch_size:n_train
      batch_end = min(batch_start + config.batch_size - 1,n_train)
      idx = train_idx[batch_start:batch_end]

      mu_batch = params_n[:,idx]
      u_batch = sols_n[:,idx]

      (loss,_),grads = Zygote.withgradient(ps) do p
        u_hat,_ = Lux.apply(model,(mu_batch,trunk_matrix),p,st)
        mse = mean((u_hat .- u_batch).^2)
        return mse,nothing
      end

      opt_state,ps = Optimisers.update(opt_state,ps,grads[1])
      epoch_loss += loss
      n_batches += 1
    end

    avg_train_loss = epoch_loss / n_batches
    push!(train_losses,avg_train_loss)

    # Validation
    trunk_matrix = precompute_trunk_matrix(model,coords_f32,ps,st)
    mu_val = params_n[:,val_idx]
    u_val = sols_n[:,val_idx]
    u_hat_val,_ = Lux.apply(model,(mu_val,trunk_matrix),ps,st)
    val_loss = mean((u_hat_val .- u_val).^2)
    push!(val_losses,val_loss)

    if config.verbose && (epoch % 50 == 0 || epoch == 1)
      println("  Epoch $epoch/$( config.epochs): " *
              "train_loss=$(round(avg_train_loss,digits=6)), " *
              "val_loss=$(round(Float64(val_loss),digits=6))")
    end

    # Early stopping
    if val_loss < best_val_loss
      best_val_loss = val_loss
      best_ps = deepcopy(ps)
      best_epoch = epoch
      patience_counter = 0
    else
      patience_counter += 1
      if patience_counter >= config.patience
        config.verbose && println("  Early stopping at epoch $epoch (best: $best_epoch)")
        break
      end
    end
  end

  # Final trunk matrix with best parameters
  trunk_matrix = precompute_trunk_matrix(model,coords_f32,best_ps,st)

  return TrainingResult(
    model,best_ps,st,trunk_matrix,
    input_norm,output_norm,
    train_losses,val_losses,best_epoch
  )
end
