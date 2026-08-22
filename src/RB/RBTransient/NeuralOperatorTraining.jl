function RBSteady.train_neural_operator(
  red::DeepONetReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots
)

  strategy = red.strategy

  # Data extraction
  target_data = Float32.(get_all_data(s))     # shape (N_dofs,N_samples,N_time)

  realisation = get_realisation(s)

  # Extract the spatial parameters
  param_realisation = get_params(realisation)
  raw_params = Float32.(matrix_of_params(param_realisation))
  n_samples = size(raw_params,2)
  
  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)

  # Time grid
  t_grid = Float32.(get_times(realisation))

  N_dofs = size(target_data,1)
  N_time = size(target_data,3)

  # Subsampling indices
  idx_x = 1:strategy.step_x:N_dofs
  idx_t = 1:strategy.step_t:N_time
  N_x_red = length(idx_x)
  N_t_red = length(idx_t)
  N_points = N_x_red * N_t_red

  # Coordinates extraction (Trunk input)
  V = get_test(feop)

  # Safety check for proper spatial mapping
  if !(V isa OrderedFESpace || (V isa TrialFESpace && V.space isa OrderedFESpace))
    throw(ArgumentError("The FE space MUST be an OrderedFESpace. Standard FESpaces do not guarantee DoF ordering, which would silently corrupt the neural operator training mapping."))
  end

  coords_raw = get_coords_with_order(V) # Shape: (D_phys,N_dofs)
  D_phys = size(coords_raw,1)

  # Spatio-Temporal coordinate matrix (D_phys + 1 for time,N_points)
  x_train = zeros(Float32,D_phys + 1,N_points)
  col = 1
  for t_idx in idx_t
    t_val = t_grid[t_idx]
    for x_idx in idx_x
      # Copy all the physical dimensions of the spatial point
      x_train[1:D_phys,col] .= coords_raw[:,x_idx]
      # Adding time as last coordinate
      x_train[D_phys+1,col] = t_val
      col += 1
    end
  end

  # Flatten the target data to (N_points,n_samples) to match x_train columns
  u_train = zeros(Float32,N_points,n_samples)
  for sample_idx = 1:n_samples
    col = 1
    for t_idx in idx_t
      for x_idx in idx_x
        u_train[col,sample_idx] = target_data[x_idx,sample_idx,t_idx]
        col += 1
      end
    end
  end

  # Data dimension
  n_branch_in = size(params_matrix,1)
  n_trunk_in  = size(x_train,1)
  
  # Normalization
  max_u = maximum(abs.(u_train))
  u_train ./= max_u
  
  branch_stats = compute_zscore_stats(params_matrix)
  params_matrix = (params_matrix .- branch_stats.μ) ./ branch_stats.σ
  
  trunk_stats = compute_zscore_stats(x_train)
  x_train = (x_train .- trunk_stats.μ) ./ trunk_stats.σ

  # DeepONet architecture
  # Input of the Trunk Net is D_phys + 1
  model_def = resolve_model(strategy.model,n_branch_in,n_trunk_in)
  deepONet = build_model(model_def)

  # Dataloader and Lux setup
  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader =
    MLUtils.DataLoader((params_matrix,u_train); batchsize=bs,shuffle=true,partial=false)

  x_data_dev = x_train |> XDEV

  rng = Random.default_rng()
  Random.seed!(rng,42)
  ps,st = Lux.setup(rng,deepONet) |> XDEV
  
  initial_lr = get_initial_lr(strategy.lr_scheduler)

  opt = Optimisers.Adam(initial_lr)
  train_state = Lux.Training.TrainState(deepONet,ps,st,opt)
  
  # Verbosity level setup
  logger = TrainingLog("DeepONet",strategy.epochs;verbose=strategy.verbose,print_every=strategy.print_every)

  # Training execution defined in RBSteady
  ps_trained,st_trained =
    train_deeponet!(train_state,dataloader,x_data_dev,strategy.lr_scheduler;logger=logger)

  st_test = Lux.testmode(st_trained) |> CDEV
  
  norm_stats = (branch = branch_stats,trunk = trunk_stats)
  
  return deepONet,ps_trained |> CDEV,st_test,norm_stats,Float32(max_u)
end

function RBSteady.train_neural_operator(
  red::DeepONetReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool = false
)

  strategy = red.strategy

  # Data extraction
  target_data = Float32.(get_all_data(s))  # shape (N_dofs,N_samples,N_time)
  realisation = get_realisation(s)

  param_realisation = get_params(realisation)
  raw_params = Float32.(matrix_of_params(param_realisation))
  n_samples = size(raw_params,2)
  
  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)

  t_grid = Float32.(get_times(realisation))

  N_dofs = size(target_data,1)
  N_time = size(target_data,3)

  idx_x = 1:strategy.step_x:N_dofs
  idx_t = 1:strategy.step_t:N_time
  N_x_red = length(idx_x)
  N_t_red = length(idx_t)
  N_points = N_x_red * N_t_red

  V = get_test(feop)

  if !(V isa OrderedFESpace || (V isa TrialFESpace && V.space isa OrderedFESpace))
    throw(ArgumentError("The FE space MUST be an OrderedFESpace. Standard FESpaces do not guarantee DoF ordering, which would silently corrupt the neural operator training mapping."))
  end

  coords_raw = get_coords_with_order(V) 
  D_phys = size(coords_raw,1)

  x_train = zeros(Float32,D_phys + 1,N_points)
  col = 1
  for t_idx in idx_t
    t_val = t_grid[t_idx]
    for x_idx in idx_x
      x_train[1:D_phys,col] .= coords_raw[:,x_idx]
      x_train[D_phys+1,col] = t_val
      col += 1
    end
  end

  u_train = zeros(Float32,N_points,n_samples)
  for sample_idx = 1:n_samples
    col = 1
    for t_idx in idx_t
      for x_idx in idx_x
        u_train[col,sample_idx] = target_data[x_idx,sample_idx,t_idx]
        col += 1
      end
    end
  end

  n_branch_in = size(params_matrix,1)
  n_trunk_in  = size(x_train,1)
  
  # Dimensions check for fine-tuning
  expected_branch_in = length(pretrained_op.norm_stats.branch.μ)
  expected_trunk_in  = length(pretrained_op.norm_stats.trunk.μ)
  @assert n_branch_in == expected_branch_in "Branch dimension mismatch: expected $expected_branch_in, got $n_branch_in. Check branch_sampler."
  @assert n_trunk_in == expected_trunk_in "Trunk dimension mismatch: expected $expected_trunk_in, got $n_trunk_in."

  # Normalization setup
  if update_stats
    strategy.verbose && @info "Updating the normalization statistics."
    max_u = maximum(abs.(u_train))
    branch_stats = compute_zscore_stats(params_matrix)
    trunk_stats = compute_zscore_stats(x_train)
  else
    strategy.verbose && @info "Keeping the normalization statistics from the pre-trained model."
    max_u = pretrained_op.max_u
    branch_stats = pretrained_op.norm_stats.branch
    trunk_stats = pretrained_op.norm_stats.trunk
  end

  u_train ./= max_u
  params_matrix = (params_matrix .- branch_stats.μ) ./ branch_stats.σ
  x_train = (x_train .- trunk_stats.μ) ./ trunk_stats.σ

  # Pretrained Network
  deepONet = pretrained_op.model
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV

  # Dataloader and Optimizer setup
  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader = MLUtils.DataLoader((params_matrix,u_train); batchsize=bs,shuffle=true,partial=false)

  x_data_dev = x_train |> XDEV

  initial_lr = get_initial_lr(strategy.lr_scheduler)
  opt = Optimisers.Adam(initial_lr)
  train_state = Lux.Training.TrainState(deepONet,ps,st,opt)
  
  # Verbosity level setup
  logger = TrainingLog("DeepONet",strategy.epochs;verbose=strategy.verbose,print_every=strategy.print_every)

  # Training
  ps_trained,st_trained = train_deeponet!(
    train_state,dataloader,x_data_dev,strategy.lr_scheduler;logger=logger
  )

  st_test = Lux.testmode(st_trained) |> CDEV
  norm_stats = (branch = branch_stats,trunk = trunk_stats)
  
  return deepONet,ps_trained |> CDEV,st_test,norm_stats,Float32(max_u)
end

function RBSteady.train_neural_operator(
  red::NOMADReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots
)

  strategy = red.strategy

  # Data extraction
  target_data = Float32.(get_all_data(s))  # shape (N_dofs,N_samples,N_time)
  realisation = get_realisation(s)

  # Extract the spatial parameters (sensors or Branch input)
  param_realisation = get_params(realisation)
  raw_params = Float32.(matrix_of_params(param_realisation))
  n_samples = size(raw_params,2)
  
  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)
  n_sensors = size(params_matrix,1)

  # Time grid
  t_grid = Float32.(get_times(realisation))

  N_dofs = size(target_data,1)
  N_time = size(target_data,3)

  # Subsampling indices
  idx_x = 1:strategy.step_x:N_dofs
  idx_t = 1:strategy.step_t:N_time
  N_x_red = length(idx_x)
  N_t_red = length(idx_t)
  
  # Computing total number of points
  N_points = N_x_red * N_t_red
  N_tot = N_points * n_samples

  # Coordinates extraction (Trunk input)
  V = get_test(feop)
  if !(V isa OrderedFESpace || (V isa TrialFESpace && V.space isa OrderedFESpace))
    throw(ArgumentError("The FE space MUST be an OrderedFESpace. Standard FESpaces do not guarantee DoF ordering, which would silently corrupt the neural operator training mapping."))
  end

  coords_raw = get_coords_with_order(V) # Shape: (D_phys,N_dofs)
  x_red = coords_raw[:,idx_x]
  D_phys = size(x_red,1)

  # Flattening for NOMAD
  u_in  = zeros(Float32,n_sensors,N_tot)
  y_in  = zeros(Float32,D_phys + 1,N_tot)  # D_phys + 1 for time
  v_out = zeros(Float32,1,N_tot)

  col = 1
  for sample_idx in 1:n_samples
    sensor_vals = params_matrix[:,sample_idx]
    
    for t_idx in idx_t
      t_val = t_grid[t_idx]
      
      for (x_idx_reduced,x_idx_full) in enumerate(idx_x)
        # Replicating the sensor for that sample
        u_in[:,col] .= sensor_vals
        
        # Space-time coordinates
        y_in[1:D_phys,col] .= x_red[:,x_idx_reduced]
        y_in[D_phys+1,col]   = t_val
        
        # Ground truth extraction from the 3D snapshot
        v_out[1,col] = target_data[x_idx_full,sample_idx,t_idx]
        
        col += 1
      end
    end
  end

  # Normalization (z-score and Max)
  max_u = maximum(abs.(v_out))
  v_out ./= max_u
  
  u_in_stats = compute_zscore_stats(u_in)
  u_in = (u_in .- u_in_stats.μ) ./ u_in_stats.σ
  
  y_in_stats = compute_zscore_stats(y_in)
  y_in = (y_in .- y_in_stats.μ) ./ y_in_stats.σ

  # Building the NOMAD model
  # The network input is: sensors + (physical coordinates + 1 for time)
  model_def = resolve_model(strategy.model,n_sensors,D_phys + 1)
  nomad_net = build_model(model_def)

  # Dataloader and Lux setup
  bs = resolve_batch_size(strategy.batch_size,N_tot)
  dataloader = MLUtils.DataLoader(
    ((u_in,y_in),v_out);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  rng = Random.default_rng()
  Random.seed!(rng,42)
  ps,st = Lux.setup(rng,nomad_net) |> XDEV
  
  initial_lr = get_initial_lr(strategy.lr_scheduler)
  opt = Optimisers.Adam(initial_lr)
  train_state = Lux.Training.TrainState(nomad_net,ps,st,opt)
  
  # Verbosity level setup
  logger = TrainingLog("NOMAD",strategy.epochs;verbose=strategy.verbose,print_every=strategy.print_every)

  # Running the pipeline
  ps_trained,st_trained = train_nomad!(
    train_state,dataloader,strategy.lr_scheduler;logger=logger
  )

  st_test = Lux.testmode(st_trained) |> CDEV
  
  # norm_stats
  norm_stats = (u_in = u_in_stats,y_in = y_in_stats)
  
  return nomad_net,ps_trained |> CDEV,st_test,norm_stats,Float32(max_u)
end


function RBSteady.train_neural_operator(
  red::NOMADReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool = false
)

  strategy = red.strategy

  # Data Extraction
  target_data = Float32.(get_all_data(s))  
  realisation = get_realisation(s)

  param_realisation = get_params(realisation)
  raw_params = Float32.(matrix_of_params(param_realisation))
  n_samples = size(raw_params,2)
  
  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)
  n_sensors = size(params_matrix,1)

  t_grid = Float32.(get_times(realisation))

  N_dofs = size(target_data,1)
  N_time = size(target_data,3)

  idx_x = 1:strategy.step_x:N_dofs
  idx_t = 1:strategy.step_t:N_time
  N_x_red = length(idx_x)
  N_t_red = length(idx_t)
  
  N_points = N_x_red * N_t_red
  N_tot = N_points * n_samples

  V = get_test(feop)
  if !(V isa OrderedFESpace || (V isa TrialFESpace && V.space isa OrderedFESpace))
    throw(ArgumentError("The FE space MUST be an OrderedFESpace. Standard FESpaces do not guarantee DoF ordering, which would silently corrupt the neural operator training mapping."))
  end

  coords_raw = get_coords_with_order(V) 
  x_red = coords_raw[:,idx_x]
  D_phys = size(x_red,1)

  u_in  = zeros(Float32,n_sensors,N_tot)
  y_in  = zeros(Float32,D_phys + 1,N_tot)  
  v_out = zeros(Float32,1,N_tot)

  col = 1
  for sample_idx in 1:n_samples
    sensor_vals = params_matrix[:,sample_idx]
    for t_idx in idx_t
      t_val = t_grid[t_idx]
      for (x_idx_reduced,x_idx_full) in enumerate(idx_x)
        u_in[:,col] .= sensor_vals
        y_in[1:D_phys,col] .= x_red[:,x_idx_reduced]
        y_in[D_phys+1,col]   = t_val
        v_out[1,col] = target_data[x_idx_full,sample_idx,t_idx]
        col += 1
      end
    end
  end

  # Dimensionality Check
  expected_u_in = length(pretrained_op.norm_stats.u_in.μ)
  expected_y_in = length(pretrained_op.norm_stats.y_in.μ)
  @assert n_sensors == expected_u_in "Sensors input dimension mismatch: expected $expected_u_in,got $n_sensors."
  @assert D_phys + 1 == expected_y_in "Coords (Space+Time) input dimension mismatch: expected $expected_y_in,got $(D_phys + 1)."

  # Normalization setup
  if update_stats
    strategy.verbose && @info "Updating the normalization statistics."
    max_u = maximum(abs.(v_out))
    u_in_stats = compute_zscore_stats(u_in)
    y_in_stats = compute_zscore_stats(y_in)
  else
    strategy.verbose && @info "Keeping the normalization statistics from the pre-trained model."
    max_u = pretrained_op.max_u
    u_in_stats = pretrained_op.norm_stats.u_in
    y_in_stats = pretrained_op.norm_stats.y_in
  end

  v_out ./= max_u
  u_in = (u_in .- u_in_stats.μ) ./ u_in_stats.σ
  y_in = (y_in .- y_in_stats.μ) ./ y_in_stats.σ

  # Pretrained Network
  nomad_net = pretrained_op.model
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV

  # Dataloader and Optimizer setup
  bs = resolve_batch_size(strategy.batch_size,N_tot)
  dataloader = MLUtils.DataLoader(
    ((u_in,y_in),v_out);
    batchsize=bs,
    shuffle=true,
    partial=false
  )
  
  initial_lr = get_initial_lr(strategy.lr_scheduler)
  opt = Optimisers.Adam(initial_lr)
  train_state = Lux.Training.TrainState(nomad_net,ps,st,opt)
  
  # Verbosity level setup
  logger = TrainingLog("NOMAD",strategy.epochs;verbose=strategy.verbose,print_every=strategy.print_every)

  # Training
  ps_trained,st_trained = train_nomad!(
    train_state,dataloader,strategy.lr_scheduler;logger=logger
  )

  st_test = Lux.testmode(st_trained) |> CDEV
  norm_stats = (u_in = u_in_stats,y_in = y_in_stats)
  
  return nomad_net,ps_trained |> CDEV,st_test,norm_stats,Float32(max_u)
end