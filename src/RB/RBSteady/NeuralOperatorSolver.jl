function Algebra.solve(
  solver::NeuralOpSolver{<:Any,<:DeepONetReduction},
  op::NeuralRBOperator,
  r::Realisation
  )
  
  deepONet = op.model
  ps = op.model_weights
  st = op.model_states
  max_u = op.max_u
  strategy = solver.state_reduction.strategy
  
  branch_stats = op.norm_stats.branch
  trunk_stats = op.norm_stats.trunk
  
  # Branch Input (Parameters extraction)
  raw_params = Float32.(matrix_of_params(r))
  n_samples = size(raw_params,2)

  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)
  
  f_in = (params_matrix .- branch_stats.μ) ./ branch_stats.σ

  # Trunk Input (Coordinates extraction)
  V = get_test(op.op)
  #=
  coords_raw = RBSteady.get_coords_with_order(V)
  D_phys = ndims(coords_raw)
  interior_indices = ntuple(d -> 2:(size(coords_raw,d) - 1),D_phys)
  coords_interior = coords_raw[interior_indices...]
  coords_vec = vec(coords_interior)

  N_dofs = length(coords_vec)

  x_test = zeros(Float32,D_phys,N_dofs)
  for i = 1:N_dofs
    for d = 1:D_phys
      x_test[d,i] = Float32(coords_vec[i][d])
    end
  end
  x_in = (x_test .- trunk_stats.μ) ./ trunk_stats.σ
  =#
  x_test = RBSteady.get_coords_with_order(V)
  x_in = (x_test .- trunk_stats.μ) ./ trunk_stats.σ

  # Inference Execution
  t = @timed begin
    pred_cpu,_ = deepONet((f_in,x_in),ps,st)
  end

  # Denormalize output
  pred_cpu .*= max_u

  # Data Packaging for GridapROMs
  fe_data = ConsecutiveParamArray(Float64.(pred_cpu))

  # Dummy low-dimensional projection
  dummy_red_data = ConsecutiveParamArray(zeros(Float64,1,n_samples))

  x̂ = RBParamVector(dummy_red_data,fe_data)
  stats = CostTracker(t,nruns=n_samples,name="NeuralOperator Inference")

  return x̂,stats
end


function Algebra.solve(
  solver::NeuralOpSolver{<:Any,<:NOMADReduction},
  op::NeuralRBOperator,
  r::Realisation
  )

  nomad_net = op.model
  ps = op.model_weights
  st = op.model_states
  max_u = op.max_u
  strategy = solver.state_reduction.strategy
  
  u_in_stats = op.norm_stats.u_in
  y_in_stats = op.norm_stats.y_in
  
  # Parameters and Sensors extraction
  raw_params = Float32.(matrix_of_params(r))
  n_samples = size(raw_params,2)

  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)
  n_sensors = size(params_matrix,1)

  # Coordinates extraction
  V = get_test(op.op)
  x_test = RBSteady.get_coords_with_order(V)
  D_phys = size(x_test,1)
  N_dofs = size(x_test,2)

  # Flattening of input tensors
  N_tot = N_dofs * n_samples
  u_in = zeros(Float32,n_sensors,N_tot)
  y_in = zeros(Float32,D_phys,N_tot)

  col = 1
  for sample_idx in 1:n_samples
    sensor_vals = params_matrix[:,sample_idx]
    for x_idx in 1:N_dofs
      u_in[:,col] .= sensor_vals
      y_in[:,col] .= x_test[:,x_idx]
      col += 1
    end
  end

  # Normalization
  u_in = (u_in .- u_in_stats.μ) ./ u_in_stats.σ
  y_in = (y_in .- y_in_stats.μ) ./ y_in_stats.σ
  
  # Concatenation
  uy_in = vcat(u_in, y_in)

  # Inference
  t = @timed begin
    pred_cpu,_ = nomad_net(uy_in,ps,st)
  end

  # Denormalization
  pred_cpu .*= max_u

  # Reshaping of the output for GridapROMs (N_dofs,n_samples)
  pred_2d = zeros(Float64,N_dofs,n_samples)
  col = 1
  for sample_idx in 1:n_samples
    for x_idx in 1:N_dofs
      pred_2d[x_idx,sample_idx] = pred_cpu[1,col]
      col += 1
    end
  end

  # Packaging in GridapROMs types
  fe_data = ConsecutiveParamArray(pred_2d)
  dummy_red_data = ConsecutiveParamArray(zeros(Float64,1,n_samples))

  x̂ = RBParamVector(dummy_red_data,fe_data)
  stats = CostTracker(t,nruns=n_samples,name="NOMAD Inference")

  return x̂,stats
end