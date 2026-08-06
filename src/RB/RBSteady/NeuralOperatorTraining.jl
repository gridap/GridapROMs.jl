# Helpers and Devices

const CDEV = cpu_device()
const XDEV = reactant_device(;force=true)

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
  μ = mean(data, dims=2)
  σ = std(data, dims=2)
  # Avoid dividing by zero if a feature is constant
  σ[σ .== 0] .= 1.0f0
  return (μ=Float32.(μ), σ=Float32.(σ))
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
=#

function get_coords_with_order(V::SingleFieldFESpace)
  # Retrieve the underlying triangulation and its physical dimensionality (1D, 2D, 3D)
  trian = get_triangulation(V)
  D_phys = length(get_node_coordinates(trian)[1])
  
  # Get the exact number of free DoFs (automatically excluding Dirichlet boundaries)
  N_dofs = num_free_dofs(V)

  # Initialize the tensor that will feed the Trunk Net: shape (D_phys, N_dofs)
  x_raw = zeros(Float32, D_phys, N_dofs)
  
  # Extract coordinates dimension by dimension. This avoids TypeErrors 
  # when interpolating a physical vector (Point) into a purely scalar FESpace.
  for d in 1:D_phys
    # Define a scalar spatial function for the d-th physical dimension
    coord_d(x) = x[d]
    
    # Interpolate the coordinate field over the entire FESpace.
    # This maps the physical space to the algebraic DoF numbering.
    coord_fn = interpolate_everywhere(coord_d, V)
    
    # Extract only the values corresponding to the free DoFs
    free_coords = get_free_dof_values(coord_fn)
    
    # Populate the corresponding row in our Trunk Net input tensor
    for i in 1:N_dofs
      x_raw[d, i] = Float32(free_coords[i])
    end
  end
  
  return x_raw
end

function _is_periodic_node(inode,nodes)
  try
    nodes[inode]
    return false
  catch
    return true
  end
end

# Build model

function build_lux_chain(layers::Tuple,activation)
  lux_layers = []
  for i in 1:(length(layers)-1)
    if i < length(layers) - 1
      push!(lux_layers, Lux.Dense(layers[i] => layers[i+1], activation))
    else
      # last layer (no activation)
      push!(lux_layers, Lux.Dense(layers[i] => layers[i+1]))
    end
  end
  Lux.Chain(lux_layers...)
end

function build_model(model::DeepONet)
  branch_net = build_lux_chain(model.branch_layers, model.activation)
  trunk_net  = build_lux_chain(model.trunk_layers, model.activation)
  NeuralOperators.DeepONet(branch_net, trunk_net)
end

# Training loop

function train_deeponet!(train_state,dataloader,x_data_dev,lr_scheduler; max_epochs=5000)
  @info "Starting Training on Reactant Device (First epoch compiles XLA...)"
  t_start = time()
  t_start_fast = time()

  Reactant.with_config(; dot_general_precision=PrecisionConfig.HIGH) do
    for epoch = 1:max_epochs
      local current_loss = 0.0f0

      for (f_batch,u_batch) in dataloader
        batch_dev = ((f_batch |> XDEV,x_data_dev),u_batch |> XDEV)

        _,loss_val,_,train_state = Lux.Training.single_train_step!(
          AutoEnzyme(),
          MSELoss(),
          batch_dev,
          train_state;
          return_gradients=Val(false)
        )
        current_loss += Float32(loss_val)
      end
      current_loss /= length(dataloader)

      step_scheduler!(lr_scheduler,train_state.optimizer_state,epoch,max_epochs,current_loss)

      if epoch == 1
        comp_mins = round((time() - t_start) / 60,digits=2)
        t_start_fast = time()
        @info "Compilation finished in $comp_mins min. Fast training started."
      elseif epoch % 500 == 0
        elapsed_fast = time() - t_start_fast
        time_per_epoch = elapsed_fast / (epoch - 1)
        eta_seconds = time_per_epoch * (max_epochs - epoch)
        println(
          "Epoch: $epoch \t Loss: $(Float32(current_loss)) \t ETA: $(format_eta(eta_seconds))"
        )
      end
    end
  end

  @info "Training Completed in $(round((time() - t_start) / 60,digits=2)) minutes"
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
  N_dofs = size(target_data_full, 1)
  
  idx_x = 1:strategy.step_x:N_dofs
  target_data = target_data_full[idx_x,:]

  realisation = get_realisation(s)
  raw_params = Float32.(matrix_of_params(realisation))
  n_samples = size(raw_params, 2)
  
  f_in_list = [Float32.(strategy.branch_sampler(raw_params[:, i])) for i in 1:n_samples]
  params_matrix = reduce(hcat, f_in_list)

  # normalization
  max_u = maximum(abs.(target_data))
  target_data ./= max_u

  # DoF coordinates extraction (Trunk input)
  V = get_test(feop)

  # Safety check for proper spatial mapping
  if !(V isa OrderedFESpace || (V isa TrialFESpace && V.space isa OrderedFESpace))
    @warn "The FE space is not an OrderedFESpace. The order of the extracted coordinates might not match the DoF order in target_data. Training results might be incorrect."
  end
  #=coords_raw = get_coords_with_order(V)
  D_phys = ndims(coords_raw)
  
  interior_indices = ntuple(d -> 2:(size(coords_raw, d) - 1), D_phys)
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
  x_train_full = get_coords_with_order(V) # shape: (D_phys, full_N_dofs)
  x_train = x_train_full[:,idx_x]
  
  # Input normalization
  branch_stats = compute_zscore_stats(params_matrix)
  params_matrix = (params_matrix .- branch_stats.μ) ./ branch_stats.σ
  
  trunk_stats = compute_zscore_stats(x_train)
  x_train = (x_train .- trunk_stats.μ) ./ trunk_stats.σ

  # Building the DeepONet
  deepONet = build_model(strategy.model)

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

  # Executing the pipeline
  ps_trained,st_trained =
    train_deeponet!(train_state,dataloader,x_data_dev,strategy.lr_scheduler; max_epochs=strategy.epochs)
    
  st_test = Lux.testmode(st_trained) |> CDEV

  return deepONet, ps_trained |> CDEV, st_test, branch_stats, trunk_stats, Float32(max_u)
end
