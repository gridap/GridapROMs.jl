function RBSteady._extract_operator_data(
  red::DeepONetReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots
)

  strategy = red.strategy

  target_data = Float32.(get_all_data(s)) 
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

  coords_raw = RBSteady.get_coords_with_order(V)
  D_phys = size(coords_raw,1)

  # Construct Spatio-Temporal coordinate matrix (D_phys + 1)
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

  # Flatten the 3D target data to match x_train column ordering
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

  return params_matrix,x_train,u_train
end

function RBSteady._extract_operator_data(
  red::NOMADReduction,
  feop::ODEParamOperator,
  s::AbstractSnapshots
)

  strategy = red.strategy

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

  coords_raw = RBSteady.get_coords_with_order(V)
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
        y_in[D_phys+1,col] = t_val
        v_out[1,col] = target_data[x_idx_full,sample_idx,t_idx]
        col += 1
      end
    end
  end

  return u_in,y_in,v_out
end