function Algebra.solve(
  solver::NeuralOpSolver,
  op::NeuralRBOperator,
  r::TransientRealisation,
  )

  deepONet = op.model
  ps = op.model_weights
  st = op.model_states
  max_u = op.max_u
  strategy = solver.state_reduction.strategy

  # Branch Input (Parameter Extraction)
  param_realisation = get_params(r)
  raw_params = Float32.(matrix_of_params(param_realisation))
  n_samples = size(raw_params,2)

  # Apply the branch_sampler
  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)

  f_in = (params_matrix .- op.branch_stats.μ) ./ op.branch_stats.σ

  # Trunk Input (Spatiotemporal Coordinates Extraction)
  t_grid = Float32.(get_times(r))
  N_time = length(t_grid)

  V = get_test(op.op)
  x_raw = RBSteady.get_coords_with_order(V) # Shape: (D_phys,N_dofs)
  D_phys = size(x_raw,1)
  N_dofs = size(x_raw,2)

  N_points = N_dofs * N_time
  x_test = zeros(Float32,D_phys + 1,N_points)

  # Building the grid (x,t) equal to the one used during the training
  col = 1
  for t_val in t_grid
    for x_idx in 1:N_dofs
      x_test[1:D_phys,col] .= x_raw[:,x_idx]
      x_test[D_phys+1,col] = t_val
      col += 1
    end
  end

  x_in = (x_test .- op.trunk_stats.μ) ./ op.trunk_stats.σ

  # Inference Execution
  t = @timed begin
    pred_cpu,_ = deepONet((f_in,x_in),ps,st)
  end

  # Denormalize
  pred_cpu .*= max_u

  # Data Reshaping
  # Inverting the flattening to get (N_dofs,N_samples,N_time)
  pred_3d = zeros(Float64,N_dofs,n_samples,N_time)
  for sample_idx in 1:n_samples
    col = 1
    for t_idx in 1:N_time
      for x_idx in 1:N_dofs
        pred_3d[x_idx,sample_idx,t_idx] = pred_cpu[col,sample_idx]
        col += 1
      end
    end
  end

  # Wrap in GridapROMs types
  fe_data = ConsecutiveParamArray(pred_3d)
  #dummy_red_data = ConsecutiveParamArray(zeros(Float64,1,n_samples,N_time))

  #x̂ = RBParamVector(dummy_red_data,fe_data)
  x̂ = (fe_data = fe_data,)
  stats = CostTracker(t,nruns=n_samples,name="NeuralOperator Transient Inference")

  return x̂,stats
end