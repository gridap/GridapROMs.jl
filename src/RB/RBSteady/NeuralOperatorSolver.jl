function Algebra.solve(solver::NeuralOpSolver,op::NeuralRBOperator,r::Realisation)
  deepONet = op.model
  ps = op.model_weights
  st = op.model_states
  max_u = op.max_u
  strategy = solver.state_reduction.strategy
  
  # Branch Input (Parameters extraction)
  raw_params = Float32.(matrix_of_params(r))
  n_samples = size(raw_params, 2)

  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:, i])) for i in 1:n_samples]
  params_matrix = reduce(hcat, f_in_list)
  
  f_in = (params_matrix .- op.branch_stats.μ) ./ op.branch_stats.σ

  # Trunk Input (Coordinates extraction)
  V = get_test(op.op)
  #=
  coords_raw = RBSteady.get_coords_with_order(V)
  D_phys = ndims(coords_raw)
  interior_indices = ntuple(d -> 2:(size(coords_raw, d) - 1), D_phys)
  coords_interior = coords_raw[interior_indices...]
  coords_vec = vec(coords_interior)

  N_dofs = length(coords_vec)

  x_test = zeros(Float32,D_phys,N_dofs)
  for i = 1:N_dofs
    for d = 1:D_phys
      x_test[d, i] = Float32(coords_vec[i][d])
    end
  end
  x_in = (x_test .- op.trunk_stats.μ) ./ op.trunk_stats.σ
  =#
  x_test = RBSteady.get_coords_with_order(V)
  x_in = (x_test .- op.trunk_stats.μ) ./ op.trunk_stats.σ

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
