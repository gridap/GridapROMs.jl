function Algebra.solve(solver::NeuralOpSolver,op::NeuralRBOperator,r::Realisation)
  strategy = op.strategy
  ps = op.model_weights
  st = op.model_states
  max_u = op.max_u

  # Branch Input (Parameters extraction)
  params_matrix = Float32.(matrix_of_params(r))
  f_in = params_matrix

  param_dim = size(params_matrix,1)
  n_samples = size(params_matrix,2)

  # Trunk Input (Coordinates extraction)
  V = get_test(op.op)
  coords_raw = RBSteady.get_coords_with_order(V)
  coords_vec = vec(coords_raw)

  D_phys = length(coords_vec[1])
  N_dofs = length(coords_vec)

  x_test = zeros(Float32,D_phys,N_dofs)
  for i = 1:N_dofs
    for d = 1:D_phys
      x_test[d, i] = Float32(coords_vec[i][d])
    end
  end
  x_in = x_test

  # Reconstruct Model Architecture
  deepONet = NeuralOperators.DeepONet(
    Lux.Chain(
      Lux.Dense(param_dim => strategy.hidden,Lux.tanh),
      Lux.Dense(strategy.hidden => strategy.hidden,Lux.tanh),
      Lux.Dense(strategy.hidden => strategy.hidden,Lux.tanh),
      Lux.Dense(strategy.hidden => strategy.p_latent)
    ),
    Lux.Chain(
      Lux.Dense(D_phys => strategy.hidden,Lux.tanh),
      Lux.Dense(strategy.hidden => strategy.hidden,Lux.tanh),
      Lux.Dense(strategy.hidden => strategy.hidden,Lux.tanh),
      Lux.Dense(strategy.hidden => strategy.hidden,Lux.tanh),
      Lux.Dense(strategy.hidden => strategy.p_latent)
    )
  )

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
