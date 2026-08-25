# Helpers and Devices

const CDEV = Lux.cpu_device()
const XDEV = Lux.reactant_device(;force=true)

function format_eta(eta_seconds::Real)
  eta_sec = round(Int,eta_seconds)
  h = div(eta_sec,3600)
  m = div(rem(eta_sec,3600),60)
  s = rem(eta_sec,60)
  return h > 0 ? "$(lpad(h,2,'0')):$(lpad(m,2,'0')):$(lpad(s,2,'0'))" :
         "$(lpad(m,2,'0')):$(lpad(s,2,'0'))"
end

function resolve_batch_size(batch_config::Int,total_samples::Int)
  return batch_config <= 0 ? total_samples : min(batch_config,total_samples)
end

function compute_zscore_stats(data::AbstractMatrix)
  μ = mean(data,dims=2)
  σ = std(data,dims=2)
  # Avoid dividing by zero if a feature is constant
  σ[σ .== 0] .= 1.0f0
  return (μ=Float32.(μ),σ=Float32.(σ))
end

resolve_model(model::DeepONet,n_branch_in::Int,n_trunk_in::Int) = model

function resolve_model(config::AutoDeepONet,n_branch_in::Int,n_trunk_in::Int)
  # Generate homogeneous hidden layers according to configuration
  hidden = ntuple(_ -> config.width,config.depth)

  # Enforce matching dimensions for the final dot product layer
  branch_layers = (n_branch_in,hidden...,config.width)
  trunk_layers  = (n_trunk_in,hidden...,config.width)

  DeepONet(branch_layers,trunk_layers,config.activation)
end

resolve_model(model::NOMAD,n_sensors_in::Int,n_coords_in::Int) = model

function resolve_model(config::AutoNOMAD,n_sensors_in::Int,n_coords_in::Int)
  # Shared hidden layer structure for symmetric sub-networks
  hidden = ntuple(_ -> config.width,config.depth)

  # Approximator maps sensors to the latent space
  approximator_layers = (n_sensors_in,hidden...,config.width)
  # Decoder maps (latent_space + coords) to scalar prediction
  decoder_layers = (config.width + n_coords_in,hidden...,1)

  NOMAD(approximator_layers,decoder_layers,config.activation)
end

function get_coords_with_order(V::SingleFieldFESpace)
  trian = get_triangulation(V)
  D_phys = length(get_node_coordinates(trian)[1])
  N_dofs = num_free_dofs(V)

  x_raw = zeros(Float32,D_phys,N_dofs)

  # Extract physical coordinates and map to algebraic DoF numbering
  for d in 1:D_phys
    coord_d(x) = x[d]
    coord_fn = interpolate_everywhere(coord_d,V)
    free_coords = get_free_dof_values(coord_fn)

    for i in 1:N_dofs
      x_raw[d,i] = Float32(free_coords[i])
    end
  end

  return x_raw
end

# Build model

function build_lux_chain(layers::Tuple,activation)
  lux_layers = []
  for i in 1:(length(layers)-1)
    if i < length(layers) - 1
      push!(lux_layers,Lux.Dense(layers[i] => layers[i+1],activation))
    else
      push!(lux_layers,Lux.Dense(layers[i] => layers[i+1]))
    end
  end
  Lux.Chain(lux_layers...)
end

function LuxDeepONet(branch_net,trunk_net)
  Lux.Chain(
    Lux.Parallel(
      *; 
      branch = Lux.Chain(branch_net,Lux.WrappedFunction(adjoint)),
      trunk = trunk_net
    ),
    Lux.WrappedFunction(adjoint)
  )
end

function build_model(model::DeepONet)
  branch_net = build_lux_chain(model.branch_layers,model.activation)
  trunk_net  = build_lux_chain(model.trunk_layers,model.activation)
  LuxDeepONet(branch_net,trunk_net)
end

function LuxNOMAD(approximator_net,decoder_net)
  Lux.Chain(
    Lux.Parallel(
      vcat; 
      approximator = approximator_net,
      y_pass_through = Lux.NoOpLayer()
    ),
    decoder_net
  )
end

function build_model(model::NOMAD)
  approximator_net = build_lux_chain(model.approximator_layers,model.activation)
  decoder_net = build_lux_chain(model.decoder_layers,model.activation)
  LuxNOMAD(approximator_net,decoder_net)
end

# Training loop

function train_deeponet!(train_state,dataloader,x_data_dev,lr_scheduler;logger::TrainingLog)
  init!(logger)

  Reactant.with_config(;dot_general_precision=PrecisionConfig.HIGH) do
    for epoch = 1:logger.max_epochs
      local current_loss = 0.0f0

      for (f_batch,u_batch) in dataloader
        batch_dev = ((f_batch |> XDEV,x_data_dev),u_batch |> XDEV)

        _,loss_val,_,train_state = Lux.Training.single_train_step!(
          Lux.AutoEnzyme(),
          Lux.MSELoss(),
          batch_dev,
          train_state;
          return_gradients=Val(false)
        )
        current_loss += Float32(loss_val)
      end
      current_loss /= length(dataloader)

      step_scheduler!(lr_scheduler,train_state.optimizer_state,epoch,logger.max_epochs,current_loss;verbose=logger.verbose)
      update!(logger,epoch,current_loss)
    end
  end

  finalize!(logger)
  return train_state.parameters,train_state.states
end

function train_nomad!(train_state,dataloader,lr_scheduler;logger::TrainingLog)
  init!(logger)

  Reactant.with_config(;dot_general_precision=PrecisionConfig.HIGH) do
    for epoch in 1:logger.max_epochs
      local current_loss = 0.0f0

      for ((u_batch,y_batch),v_batch) in dataloader
        batch_dev = (
          (u_batch |> XDEV,y_batch |> XDEV),
          v_batch |> XDEV
        )

        _,loss_val,_,train_state = Lux.Training.single_train_step!(
            Lux.AutoEnzyme(),
            Lux.MSELoss(),
            batch_dev,
            train_state;
            return_gradients=Val(false)
        )
        current_loss += Float32(loss_val)
      end
      current_loss /= length(dataloader)

      step_scheduler!(lr_scheduler,train_state.optimizer_state,epoch,logger.max_epochs,current_loss;verbose=logger.verbose)
      update!(logger,epoch,current_loss)
    end
  end

  finalize!(logger)
  return train_state.parameters,train_state.states
end

# Public API

function train_neural_operator(
  red::NeuralOpReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
)
  data = _extract_operator_data(red,feop,s)
  return _core_train_operator(red,data...)
end

function train_neural_operator(
  red::NeuralOpReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool = false
)
  data = _extract_operator_data(red,feop,s)
  return _core_train_operator(red,data...,pretrained_op;update_stats=update_stats)
end

# Data Extraction (Steady State)

function _extract_operator_data(red::DeepONetReduction,feop::ParamOperator,s::AbstractSnapshots)
  strategy = red.strategy

  target_data_full = Float32.(get_all_data(s))
  N_dofs = size(target_data_full,1)

  idx_x = 1:strategy.step_x:N_dofs
  target_data = target_data_full[idx_x,:]

  realisation = get_realisation(s)
  raw_params = Float32.(matrix_of_params(realisation))
  n_samples = size(raw_params,2)

  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)

  V = get_test(feop)
  if !(V isa OrderedFESpace || (V isa TrialFESpace && V.space isa OrderedFESpace))
    throw(ArgumentError("The FE space MUST be an OrderedFESpace. Standard FESpaces do not guarantee DoF ordering, which would silently corrupt the neural operator training mapping."))
  end

  x_train_full = get_coords_with_order(V)
  x_train = x_train_full[:,idx_x]

  return params_matrix,x_train,target_data
end

function _extract_operator_data(red::NOMADReduction,feop::ParamOperator,s::AbstractSnapshots)
  strategy = red.strategy

  target_data_full = Float32.(get_all_data(s))
  N_dofs = size(target_data_full,1)

  idx_x = 1:strategy.step_x:N_dofs
  N_x_red = length(idx_x)

  realisation = get_realisation(s)
  raw_params = Float32.(matrix_of_params(realisation))
  n_samples = size(raw_params,2)

  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)
  n_sensors = size(params_matrix,1)

  V = get_test(feop)
  if !(V isa OrderedFESpace || (V isa TrialFESpace && V.space isa OrderedFESpace))
    throw(ArgumentError("The FE space MUST be an OrderedFESpace. Standard FESpaces do not guarantee DoF ordering, which would silently corrupt the neural operator training mapping."))
  end

  x_train_full = get_coords_with_order(V)
  x_red = x_train_full[:,idx_x]
  D_phys = size(x_red,1)

  N_tot = N_x_red * n_samples

  u_in  = zeros(Float32,n_sensors,N_tot)
  y_in  = zeros(Float32,D_phys,N_tot)
  v_out = zeros(Float32,1,N_tot)

  col_idx = 1
  for i in 1:n_samples
    sensor_vals = params_matrix[:,i]
    for (x_idx_reduced,x_idx_full) in enumerate(idx_x)
      u_in[:,col_idx] .= sensor_vals
      y_in[:,col_idx] .= x_red[:,x_idx_reduced]
      v_out[1,col_idx] = target_data_full[x_idx_full,i]
      col_idx += 1
    end
  end

  return u_in,y_in,v_out
end

# Core Training Logic

function _core_train_operator(
  red::DeepONetReduction,
  branch_in::AbstractMatrix,
  trunk_in::AbstractMatrix,
  target_data::AbstractMatrix
)
  strategy = red.strategy
  n_branch_in = size(branch_in,1)
  n_trunk_in  = size(trunk_in,1)

  model_def = resolve_model(strategy.model,n_branch_in,n_trunk_in)
  deepONet = build_model(model_def)

  rng = Random.default_rng()
  Random.seed!(rng,42)
  ps,st = Lux.setup(rng,deepONet) |> XDEV

  max_u = maximum(abs.(target_data))
  branch_stats = compute_zscore_stats(branch_in)
  trunk_stats = compute_zscore_stats(trunk_in)

  return _exec_deeponet(strategy,deepONet,ps,st,branch_in,trunk_in,target_data,max_u,branch_stats,trunk_stats)
end

function _core_train_operator(
  red::DeepONetReduction,
  branch_in::AbstractMatrix,
  trunk_in::AbstractMatrix,
  target_data::AbstractMatrix,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool=false
)
  strategy = red.strategy

  expected_branch_in = length(pretrained_op.norm_stats.branch.μ)
  expected_trunk_in  = length(pretrained_op.norm_stats.trunk.μ)
  n_branch_in = size(branch_in,1)
  n_trunk_in  = size(trunk_in,1)
  @assert n_branch_in == expected_branch_in "Branch dimension mismatch: expected $expected_branch_in, got $n_branch_in."
  @assert n_trunk_in == expected_trunk_in "Trunk dimension mismatch: expected $expected_trunk_in, got $n_trunk_in."

  deepONet = pretrained_op.model
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV

  if update_stats
    strategy.verbose && @info "Recomputing the normalization statistics."
    max_u = maximum(abs.(target_data))
    branch_stats = compute_zscore_stats(branch_in)
    trunk_stats = compute_zscore_stats(trunk_in)
  else
    strategy.verbose && @info "Inheriting the normalization statistics from the pre-trained model."
    max_u = pretrained_op.max_u
    branch_stats = pretrained_op.norm_stats.branch
    trunk_stats = pretrained_op.norm_stats.trunk
  end

  return _exec_deeponet(strategy,deepONet,ps,st,branch_in,trunk_in,target_data,max_u,branch_stats,trunk_stats)
end

function _exec_deeponet(
  strategy,
  deepONet,
  ps,
  st,
  branch_in::AbstractMatrix,
  trunk_in::AbstractMatrix,
  target_data::AbstractMatrix,
  max_u::Real,
  branch_stats::NamedTuple,
  trunk_stats::NamedTuple
)
  n_samples = size(branch_in,2)

  target_data ./= max_u
  branch_in = (branch_in .- branch_stats.μ) ./ branch_stats.σ
  trunk_in = (trunk_in .- trunk_stats.μ) ./ trunk_stats.σ

  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader = MLUtils.DataLoader((branch_in,target_data);batchsize=bs,shuffle=true,partial=false)

  x_data_dev = trunk_in |> XDEV

  initial_lr = get_initial_lr(strategy.lr_scheduler)
  opt = Optimisers.Adam(initial_lr)
  train_state = Lux.Training.TrainState(deepONet,ps,st,opt)

  logger = TrainingLog("DeepONet",strategy.epochs;verbose=strategy.verbose,print_every=strategy.print_every)

  ps_trained,st_trained = train_deeponet!(train_state,dataloader,x_data_dev,strategy.lr_scheduler;logger=logger)

  st_test = Lux.testmode(st_trained) |> CDEV
  norm_stats = (branch = branch_stats,trunk = trunk_stats)

  return deepONet,ps_trained|>CDEV,st_test,norm_stats,Float32(max_u)
end

function _core_train_operator(
  red::NOMADReduction,
  u_in::AbstractMatrix,
  y_in::AbstractMatrix,
  v_out::AbstractMatrix
)
  strategy = red.strategy
  n_sensors = size(u_in,1)
  D_phys = size(y_in,1)

  model_def = resolve_model(strategy.model,n_sensors,D_phys)
  nomad_net = build_model(model_def)

  rng = Random.default_rng()
  Random.seed!(rng,42)
  ps,st = Lux.setup(rng,nomad_net) |> XDEV

  max_u = maximum(abs.(v_out))
  u_in_stats = compute_zscore_stats(u_in)
  y_in_stats = compute_zscore_stats(y_in)

  return _exec_nomad(strategy,nomad_net,ps,st,u_in,y_in,v_out,max_u,u_in_stats,y_in_stats)
end

function _core_train_operator(
  red::NOMADReduction,
  u_in::AbstractMatrix,
  y_in::AbstractMatrix,
  v_out::AbstractMatrix,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool=false
)
  strategy = red.strategy

  n_sensors = size(u_in,1)
  D_phys = size(y_in,1)
  expected_u_in = length(pretrained_op.norm_stats.u_in.μ)
  expected_y_in = length(pretrained_op.norm_stats.y_in.μ)
  @assert n_sensors == expected_u_in "Sensors input dimension mismatch: expected $expected_u_in,got $n_sensors."
  @assert D_phys == expected_y_in "Coords input dimension mismatch: expected $expected_y_in,got $D_phys."

  nomad_net = pretrained_op.model
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV

  if update_stats
    strategy.verbose && @info "Recomputing the normalization statistics."
    max_u = maximum(abs.(v_out))
    u_in_stats = compute_zscore_stats(u_in)
    y_in_stats = compute_zscore_stats(y_in)
  else
    strategy.verbose && @info "Inheriting the normalization statistics from the pre-trained model."
    max_u = pretrained_op.max_u
    u_in_stats = pretrained_op.norm_stats.u_in
    y_in_stats = pretrained_op.norm_stats.y_in
  end

  return _exec_nomad(strategy,nomad_net,ps,st,u_in,y_in,v_out,max_u,u_in_stats,y_in_stats)
end

function _exec_nomad(
  strategy,
  nomad_net,
  ps,
  st,
  u_in::AbstractMatrix,
  y_in::AbstractMatrix,
  v_out::AbstractMatrix,
  max_u::Real,
  u_in_stats::NamedTuple,
  y_in_stats::NamedTuple
)
  N_tot = size(v_out,2)

  v_out ./= max_u
  u_in = (u_in .- u_in_stats.μ) ./ u_in_stats.σ
  y_in = (y_in .- y_in_stats.μ) ./ y_in_stats.σ

  bs = resolve_batch_size(strategy.batch_size,N_tot)
  dataloader = MLUtils.DataLoader(((u_in,y_in),v_out);batchsize=bs,shuffle=true,partial=false)

  initial_lr = get_initial_lr(strategy.lr_scheduler)
  opt = Optimisers.Adam(initial_lr)
  train_state = Lux.Training.TrainState(nomad_net,ps,st,opt)

  logger = TrainingLog("NOMAD",strategy.epochs;verbose=strategy.verbose,print_every=strategy.print_every)

  ps_trained,st_trained = train_nomad!(train_state,dataloader,strategy.lr_scheduler;logger=logger)

  st_test = Lux.testmode(st_trained) |> CDEV
  norm_stats = (u_in = u_in_stats,y_in = y_in_stats)

  return nomad_net,ps_trained|>CDEV,st_test,norm_stats,Float32(max_u)
end