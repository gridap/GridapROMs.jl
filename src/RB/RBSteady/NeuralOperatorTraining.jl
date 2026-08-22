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
  # Generating hidden layers (ex. 3 layer of 64)
  hidden = ntuple(_ -> config.width,config.depth)
  
  # Last layer of Branch and Trunk nets must have same dimension (width) for dot product
  branch_layers = (n_branch_in,hidden...,config.width)
  trunk_layers  = (n_trunk_in,hidden...,config.width)
  
  DeepONet(branch_layers,trunk_layers,config.activation)
end

resolve_model(model::NOMAD,n_sensors_in::Int,n_coords_in::Int) = model

# Resolve AutoNOMAD dimensions with symmetric sub-networks
function resolve_model(config::AutoNOMAD,n_sensors_in::Int,n_coords_in::Int)
  # Shared hidden layer structure
  hidden = ntuple(_ -> config.width,config.depth)
  
  # Approximator: from n_sensors_in to a latent space of size 'width'
  approximator_layers = (n_sensors_in,hidden...,config.width)
  
  # Decoder: from (latent space + n_coords_in) to 1 (scalar output)
  decoder_layers = (config.width + n_coords_in,hidden...,1)
  
  NOMAD(approximator_layers,decoder_layers,config.activation)
end

# Coordinate Extraction
#=
function get_coords_with_order(space::SingleFieldFESpace)
  orders = get_polynomial_orders(space)
  trian = get_triangulation(space)
  model = get_background_model(trian)
  get_coords_with_order(model,orders)
end

function get_coords_with_order(
  model::CartesianDiscreteModel{D},
  orders::NTuple{D,Int}
) where {D}
  desc = get_cartesian_descriptor(model)
  periodic = desc.isperiodic
  ncells = desc.partition
  sizes = desc.sizes
  x0 = desc.origin
  cells = CartesianIndices(ncells)
  nodes = CartesianIndices(orders .* ncells .+ 1 .- periodic)
  coords = Array{NTuple{D,Float64}}(undef,size(nodes))
  for cell in cells
    first_new_node = orders .* (Tuple(cell) .- 1) .+ 1
    nodes_range = map(enumerate(first_new_node)) do (i,ni)
      ni:(ni+orders[i])
    end
    for inode in Iterators.product(nodes_range...)
      _is_periodic_node(inode,nodes) && continue
      coords[inode...] = ntuple(d -> x0[d] + (inode[d]-1)*sizes[d],Val{D}())
    end
  end
  return coords
end

function _is_periodic_node(inode,nodes)
    try
        nodes[inode]
        return false
    catch
        return true
    end
end
=#

function get_coords_with_order(V::SingleFieldFESpace)
  # Retrieve the underlying triangulation and its physical dimensionality (1D,2D,3D)
  trian = get_triangulation(V)
  D_phys = length(get_node_coordinates(trian)[1])
  
  # Get the exact number of free DoFs (automatically excluding Dirichlet boundaries)
  N_dofs = num_free_dofs(V)

  # Initialize the tensor that will feed the Trunk Net: shape (D_phys,N_dofs)
  x_raw = zeros(Float32,D_phys,N_dofs)
  
  # Extract coordinates dimension by dimension. This avoids TypeErrors 
  # when interpolating a physical vector (Point) into a purely scalar FESpace.
  for d in 1:D_phys
    # Define a scalar spatial function for the d-th physical dimension
    coord_d(x) = x[d]
    
    # Interpolate the coordinate field over the entire FESpace.
    # This maps the physical space to the algebraic DoF numbering.
    coord_fn = interpolate_everywhere(coord_d,V)
    
    # Extract only the values corresponding to the free DoFs
    free_coords = get_free_dof_values(coord_fn)
    
    # Populate the corresponding row in our Trunk Net input tensor
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
      # last layer (no activation)
      push!(lux_layers,Lux.Dense(layers[i] => layers[i+1]))
    end
  end
  Lux.Chain(lux_layers...)
end

# Create a DeepONet layers
function LuxDeepONet(branch_net,trunk_net)
  Lux.Chain(
    # Process inputs (u,y) independently,then matrix-multiply them
    Lux.Parallel(
      *; 
      # Branch: process 'u' -> shape (Features,Batch)
      # then transpose (adjoint) -> shape (Batch,Features)
      branch = Lux.Chain(branch_net,Lux.WrappedFunction(adjoint)),
      
      # Trunk: process 'y' -> shape (Features,Points)
      trunk = trunk_net
    ),
    # The '*' gives (Batch,Points). 
    # Final transpose (adjoint) -> target shape: (Points,Batch)
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
    # Apply approximator to 'u',pass 'y' untouched,and concatenate them (vcat)
    Lux.Parallel(
      vcat; 
      approximator = approximator_net,
      y_pass_through = Lux.NoOpLayer()
    ),
    # Pass the concatenated vector [approximator(u); y] to the decoder
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

  Reactant.with_config(; dot_general_precision=PrecisionConfig.HIGH) do
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

      update!(logger, epoch, current_loss)
    end
  end

  finalize!(logger)
  return train_state.parameters,train_state.states
end

function train_nomad!(train_state,dataloader,lr_scheduler;logger::TrainingLog)
  init!(logger)
  
  Reactant.with_config(; dot_general_precision=PrecisionConfig.HIGH) do
    for epoch in 1:logger.max_epochs
      local current_loss = 0.0f0
      
      for ((u_batch,y_batch),v_batch) in dataloader
        # Single concatenated tensor (Sensors + Coordinates)
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
      
      update!(logger, epoch, current_loss)
    end
  end
  
  finalize!(logger)
  return train_state.parameters,train_state.states
end

# Generic Dispatch (Steady)

function train_neural_operator(
  red::DeepONetReduction,
  feop::ParamOperator,
  s::AbstractSnapshots
)
  strategy = red.strategy

  # Data extraction
  # RBSteady => get_all_data(s) is 2D: (N_dofs,N_samples)
  target_data_full = Float32.(get_all_data(s))
  N_dofs = size(target_data_full,1)
  
  idx_x = 1:strategy.step_x:N_dofs
  target_data = target_data_full[idx_x,:]

  realisation = get_realisation(s)
  raw_params = Float32.(matrix_of_params(realisation))
  n_samples = size(raw_params,2)
  
  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)

  # normalization
  max_u = maximum(abs.(target_data))
  target_data ./= max_u

  # DoF coordinates extraction (Trunk input)
  V = get_test(feop)

  # Safety check for proper spatial mapping
  if !(V isa OrderedFESpace || (V isa TrialFESpace && V.space isa OrderedFESpace))
    throw(ArgumentError("The FE space MUST be an OrderedFESpace. Standard FESpaces do not guarantee DoF ordering, which would silently corrupt the neural operator training mapping."))
  end
  #=coords_raw = get_coords_with_order(V)
  D_phys = ndims(coords_raw)
  
  interior_indices = ntuple(d -> 2:(size(coords_raw,d) - 1),D_phys)
  coords_interior = coords_raw[interior_indices...]

  # Flattening coordinates into a 1D vector
  coords_vec = vec(coords_interior)
  

  # Converting the vector of point in a Float32 Matrix of shape (D_phys,N_dofs)
  x_train = zeros(Float32,D_phys,N_dofs)
  for i = 1:N_dofs
    for d = 1:D_phys
      x_train[d,i] = Float32(coords_vec[i][d])
    end
  end
  =#
  
  x_train_full = get_coords_with_order(V) # shape: (D_phys,full_N_dofs)
  x_train = x_train_full[:,idx_x]
  
  # Data dimension
  n_branch_in = size(params_matrix,1)
  n_trunk_in  = size(x_train,1)
  
  # Input normalization
  branch_stats = compute_zscore_stats(params_matrix)
  params_matrix = (params_matrix .- branch_stats.μ) ./ branch_stats.σ
  
  trunk_stats = compute_zscore_stats(x_train)
  x_train = (x_train .- trunk_stats.μ) ./ trunk_stats.σ
  

  # Building the DeepONet
  model_def = resolve_model(strategy.model,n_branch_in,n_trunk_in)
  deepONet = build_model(model_def)

  # Dataloader and setup
  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader = MLUtils.DataLoader(
    (params_matrix,target_data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  x_data_dev = x_train |> XDEV

  rng = Random.default_rng()
  Random.seed!(rng,42)
  ps,st = Lux.setup(rng,deepONet) |> XDEV
  
  initial_lr = get_initial_lr(strategy.lr_scheduler)

  opt = Optimisers.Adam(initial_lr)
  train_state = Lux.Training.TrainState(deepONet,ps,st,opt)
  
  # Verbosity level setup
  logger = TrainingLog("DeepONet",strategy.epochs;verbose=strategy.verbose,print_every=strategy.print_every)

  # Executing the pipeline
  ps_trained,st_trained =
    train_deeponet!(train_state,dataloader,x_data_dev,strategy.lr_scheduler;logger=logger)
    
  st_test = Lux.testmode(st_trained) |> CDEV
  
  norm_stats = (branch = branch_stats,trunk = trunk_stats)

  return deepONet,ps_trained |> CDEV,st_test,norm_stats,Float32(max_u)
end

function train_neural_operator(
  red::DeepONetReduction,
  feop::ParamOperator,
  s::AbstractSnapshots,
  pretrained_op::NeuralRBOperator;
  update_stats::Bool = false
)
  strategy = red.strategy

  # Data extraction
  # RBSteady => get_all_data(s) is 2D: (N_dofs,N_samples)
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
  
  n_branch_in = size(params_matrix,1)
  n_trunk_in  = size(x_train,1)
  
  # Dimensions check for fine-tuning
  expected_branch_in = length(pretrained_op.norm_stats.branch.μ)
  expected_trunk_in  = length(pretrained_op.norm_stats.trunk.μ)
  @assert n_branch_in == expected_branch_in "Branch dimension mismatch: expected $expected_branch_in,got $n_branch_in. Check branch_sampler."
  @assert n_trunk_in == expected_trunk_in "Trunk dimension mismatch: expected $expected_trunk_in,got $n_trunk_in."

  # Normalization setup
  if update_stats
    strategy.verbose && @info "Recomputing the normalization statistics."
    max_u = maximum(abs.(target_data))
    branch_stats = compute_zscore_stats(params_matrix)
    trunk_stats = compute_zscore_stats(x_train)
  else
    strategy.verbose && @info "Inheriting the normalization statistics from the pre-trained model."
    max_u = pretrained_op.max_u
    branch_stats = pretrained_op.norm_stats.branch
    trunk_stats = pretrained_op.norm_stats.trunk
  end

  # Normalization
  target_data ./= max_u
  params_matrix = (params_matrix .- branch_stats.μ) ./ branch_stats.σ
  x_train = (x_train .- trunk_stats.μ) ./ trunk_stats.σ
  
  # Pretrained-model
  deepONet = pretrained_op.model
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV

  # Dataloader and optimization setup
  bs = resolve_batch_size(strategy.batch_size,n_samples)
  dataloader = MLUtils.DataLoader(
    (params_matrix,target_data);
    batchsize=bs,
    shuffle=true,
    partial=false
  )

  x_data_dev = x_train |> XDEV
  
  initial_lr = get_initial_lr(strategy.lr_scheduler)
  opt = Optimisers.Adam(initial_lr) # New LR
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

function train_neural_operator(
    red::NOMADReduction,
    feop::ParamOperator,
    s::AbstractSnapshots
)
  strategy = red.strategy
  
  # Data extraction
  # RBSteady => get_all_data(s) is 2D: (N_dofs,N_samples)
  target_data_full = Float32.(get_all_data(s))
  N_dofs = size(target_data_full,1)
  
  idx_x = 1:strategy.step_x:N_dofs
  N_x_red = length(idx_x)
  
  realisation = get_realisation(s)
  raw_params = Float32.(matrix_of_params(realisation))
  n_samples = size(raw_params,2)
  
  # Sensors extraction (like branch input in DeepONet)
  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:,i])) for i in 1:n_samples]
  params_matrix = reduce(hcat,f_in_list)
  n_sensors = size(params_matrix,1)
  
  # DoF coordinates extraction (like trunk input in DeepONet)
  V = get_test(feop)
  if !(V isa OrderedFESpace || (V isa TrialFESpace && V.space isa OrderedFESpace))
    throw(ArgumentError("The FE space MUST be an OrderedFESpace. Standard FESpaces do not guarantee DoF ordering, which would silently corrupt the neural operator training mapping."))
  end
  
  x_train_full = get_coords_with_order(V) # shape: (D_phys,full_N_dofs)
  x_red = x_train_full[:,idx_x]
  D_phys = size(x_red,1)
  
  # Flattening for NOMAD
  N_tot = N_x_red * n_samples
  
  u_in  = zeros(Float32,n_sensors,N_tot)
  y_in  = zeros(Float32,D_phys,N_tot)
  v_out = zeros(Float32,1,N_tot)
  
  col_idx = 1
  for i in 1:n_samples
    sensor_vals = params_matrix[:,i]
    for (x_idx_reduced,x_idx_full) in enumerate(idx_x)
      u_in[:,col_idx]  .= sensor_vals
      y_in[:,col_idx]  .= x_red[:,x_idx_reduced]
      v_out[1,col_idx]  = target_data_full[x_idx_full,i]
      col_idx += 1
    end
  end
  
  # normalization
  max_u = maximum(abs.(v_out))
  v_out ./= max_u
  
  u_in_stats = compute_zscore_stats(u_in)
  u_in = (u_in .- u_in_stats.μ) ./ u_in_stats.σ
  
  y_in_stats = compute_zscore_stats(y_in)
  y_in = (y_in .- y_in_stats.μ) ./ y_in_stats.σ
  
  # Building the NOMAD model
  model_def = resolve_model(strategy.model,n_sensors,D_phys)
  nomad_net = build_model(model_def)
  
  # DataLoader and Lux setup
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

function train_neural_operator(
    red::NOMADReduction,
    feop::ParamOperator,
    s::AbstractSnapshots,
    pretrained_op::NeuralRBOperator;
    update_stats::Bool = false
)
  strategy = red.strategy
  
  # Data extraction
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
  
  # Flattening for NOMAD
  N_tot = N_x_red * n_samples
  
  u_in  = zeros(Float32,n_sensors,N_tot)
  y_in  = zeros(Float32,D_phys,N_tot)
  v_out = zeros(Float32,1,N_tot)
  
  col_idx = 1
  for i in 1:n_samples
    sensor_vals = params_matrix[:,i]
    for (x_idx_reduced,x_idx_full) in enumerate(idx_x)
      u_in[:,col_idx]  .= sensor_vals
      y_in[:,col_idx]  .= x_red[:,x_idx_reduced]
      v_out[1,col_idx]  = target_data_full[x_idx_full,i]
      col_idx += 1
    end
  end
  
  # Dimensions check for fine-tuning
  expected_u_in = length(pretrained_op.norm_stats.u_in.μ)
  expected_y_in = length(pretrained_op.norm_stats.y_in.μ)
  @assert n_sensors == expected_u_in "Sensors input dimension mismatch: expected $expected_u_in, got $n_sensors."
  @assert D_phys == expected_y_in "Coords input dimension mismatch: expected $expected_y_in, got $D_phys."

  # Normalization
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

  v_out ./= max_u
  u_in = (u_in .- u_in_stats.μ) ./ u_in_stats.σ
  y_in = (y_in .- y_in_stats.μ) ./ y_in_stats.σ
  
  # Pretrained model
  nomad_net = pretrained_op.model
  ps = pretrained_op.model_weights |> XDEV
  st = pretrained_op.model_states |> XDEV
  
  # Dataloader and optimization setup
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