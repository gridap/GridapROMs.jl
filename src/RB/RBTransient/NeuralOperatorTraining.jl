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
  n_samples = size(raw_params, 2)
  
  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:, i])) for i in 1:n_samples]
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
    @warn "The FE space is not an OrderedFESpace. The order of the extracted coordinates might not match the DoF order in target_data. Training results might be incorrect."
  end

  coords_raw = get_coords_with_order(V) # Shape: (D_phys, N_dofs)
  D_phys = size(coords_raw,1)

  # Spatio-Temporal coordinate matrix (D_phys + 1 for time, N_points)
  x_train = zeros(Float32,D_phys + 1,N_points)
  col = 1
  for t_idx in idx_t
    t_val = t_grid[t_idx]
    for x_idx in idx_x
      # Copy all the physical dimensions of the spatial point
      x_train[1:D_phys, col] .= coords_raw[:, x_idx]
      # Adding time as last coordinate
      x_train[D_phys+1, col] = t_val
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

  # Normalization
  max_u = maximum(abs.(u_train))
  u_train ./= max_u
  
  branch_stats = compute_zscore_stats(params_matrix)
  params_matrix = (params_matrix .- branch_stats.μ) ./ branch_stats.σ
  
  trunk_stats = compute_zscore_stats(x_train)
  x_train = (x_train .- trunk_stats.μ) ./ trunk_stats.σ

  # DeepONet architecture
  # Input of the Trunk Net is D_phys + 1
  deepONet = build_model(strategy.model)

  # Dataloader and Lux setup
  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader =
    MLUtils.DataLoader((params_matrix,u_train); batchsize=bs,shuffle=true,partial=false)

  x_data_dev = x_train |> XDEV

  rng = Random.default_rng()
  Random.seed!(rng,42)
  ps,st = Lux.setup(rng,deepONet) |> XDEV

  opt = Optimisers.Adam(0.001f0)
  train_state = Lux.Training.TrainState(deepONet,ps,st,opt)

  # Training execution defined in RBSteady
  ps_trained,st_trained =
    train_deeponet!(train_state,dataloader,x_data_dev; epochs=strategy.epochs)

  st_test = Lux.testmode(st_trained) |> CDEV
  
  return deepONet, ps_trained |> CDEV, st_test, branch_stats, trunk_stats, Float32(max_u)
end