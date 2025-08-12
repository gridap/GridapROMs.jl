for T in (:(GridapDistributed.PSparseMatrixBuilderCOO),:(GridapDistributed.PVectorBuilder))
  @eval begin
    function ParamDataStructures.parameterize(a::$T,plength::Int)
      ParamBuilder(a,plength)
    end
  end
end

function Algebra.nz_counter(
  builder::ParamBuilder{<:GridapDistributed.PSparseMatrixBuilderCOO{A}},
  axs::Tuple{<:PRange,<:PRange}
  ) where A

  par_strategy = builder.builder.par_strategy
  plength = builder.plength

  test_dofs_gids_prange,trial_dofs_gids_prange = axs
  counters = map(partition(test_dofs_gids_prange),partition(trial_dofs_gids_prange)) do r,c
    axs = (Base.OneTo(local_length(r)),Base.OneTo(local_length(c)))
    counter = Algebra.CounterCOO{A}(axs)
    ParamCounter(counter,plength)
  end
  DistributedParamCounterCOO(
    par_strategy,
    counters,
    test_dofs_gids_prange,
    trial_dofs_gids_prange
  )
end

struct DistributedParamCounterCOO{A,B,C,D} <: GridapType
  par_strategy::A
  counters::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedParamCounterCOO(
    par_strategy,
    counters::AbstractArray{<:ParamCounter},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(counters)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,counters,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function GridapDistributed.DistributedCounterCOO(
  par_strategy,
  counters::AbstractArray{<:ParamCounter},
  test_dofs_gids_prange,
  trial_dofs_gids_prange
  )

  DistributedParamCounterCOO(
    par_strategy,
    counters,
    test_dofs_gids_prange,
    trial_dofs_gids_prange
    )
end

function GridapDistributed.local_views(a::DistributedParamCounterCOO)
  a.counters
end

function GridapDistributed.local_views(
  a::DistributedParamCounterCOO,test_dofs_gids_prange,trial_dofs_gids_prange)
  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  a.counters
end

function Algebra.nz_allocation(a::DistributedParamCounterCOO)
  allocs = map(nz_allocation,a.counters)
  DistributedParamAllocationCOO(
    a.par_strategy,
    allocs,
    a.test_dofs_gids_prange,
    a.trial_dofs_gids_prange
  )
end

struct DistributedParamAllocationCOO{A,B,C,D} <: GridapType
  par_strategy::A
  allocs::B
  test_dofs_gids_prange::C
  trial_dofs_gids_prange::D
  function DistributedParamAllocationCOO(
    par_strategy,
    allocs::AbstractArray{<:ParamAlgebra.ParamAllocationCOO},
    test_dofs_gids_prange::PRange,
    trial_dofs_gids_prange::PRange)
    A = typeof(par_strategy)
    B = typeof(allocs)
    C = typeof(test_dofs_gids_prange)
    D = typeof(trial_dofs_gids_prange)
    new{A,B,C,D}(par_strategy,allocs,test_dofs_gids_prange,trial_dofs_gids_prange)
  end
end

function GridapDistributed.DistributedAllocationCOO(
  par_strategy,
  allocs::AbstractArray{<:ParamAlgebra.ParamAllocationCOO},
  test_dofs_gids_prange,
  trial_dofs_gids_prange
  )

  DistributedParamAllocationCOO(
    par_strategy,
    allocs,
    test_dofs_gids_prange,
    trial_dofs_gids_prange
    )
end

function GridapDistributed.change_axes(
  a::DistributedParamAllocationCOO{A,B,<:PRange,<:PRange},
  axes::Tuple{<:PRange,<:PRange}
  ) where {A,B}

  local_axes = map(partition(axes[1]),partition(axes[2])) do rows,cols
    (Base.OneTo(local_length(rows)),Base.OneTo(local_length(cols)))
  end
  allocs = map(change_axes,a.allocs,local_axes)
  DistributedParamAllocationCOO(a.par_strategy,allocs,axes[1],axes[2])
end

function GridapDistributed.change_axes(
  a::MatrixBlock{<:DistributedParamAllocationCOO},
  axes::Tuple{<:Vector,<:Vector}
  )

  block_ids = CartesianIndices(a.array)
  rows,cols = axes
  array = map(block_ids) do I
    change_axes(a[I],(rows[I[1]],cols[I[2]]))
  end
  return ArrayBlock(array,a.touched)
end

function GridapDistributed.local_views(a::DistributedParamAllocationCOO)
  a.allocs
end

function GridapDistributed.local_views(
  a::DistributedParamAllocationCOO,
  test_dofs_gids_prange,
  trial_dofs_gids_prange
  )

  @check test_dofs_gids_prange === a.test_dofs_gids_prange
  @check trial_dofs_gids_prange === a.trial_dofs_gids_prange
  a.allocs
end

function GridapDistributed.local_views(a::MatrixBlock{<:DistributedParamAllocationCOO})
  array = map(local_views,a.array) |> to_parray_of_arrays
  return map(ai -> ArrayBlock(ai,a.touched),array)
end

function GridapDistributed.get_allocations(a::DistributedParamAllocationCOO)
  I,J,M = map(local_views(a)) do alloc
    alloc.I,alloc.J,alloc.M
  end |> tuple_of_arrays
  return I,J,M
end

function GridapDistributed.get_allocations(a::ArrayBlock{<:DistributedParamAllocationCOO})
  tuple_of_array_of_parrays = map(get_allocations,a.array) |> tuple_of_arrays
  return tuple_of_array_of_parrays
end

GridapDistributed.get_test_gids(a::DistributedParamAllocationCOO)  = a.test_dofs_gids_prange
GridapDistributed.get_trial_gids(a::DistributedParamAllocationCOO) = a.trial_dofs_gids_prange
GridapDistributed.get_test_gids(a::ArrayBlock{<:DistributedParamAllocationCOO})  = map(get_test_gids,diag(a.array))
GridapDistributed.get_trial_gids(a::ArrayBlock{<:DistributedParamAllocationCOO}) = map(get_trial_gids,diag(a.array))

function Algebra.create_from_nz(a::DistributedParamAllocationCOO{<:FullyAssembledRows})
  f(x) = nothing
  A, = GridapDistributed._fa_create_from_nz_with_callback(f,a)
  return A
end

function Algebra.create_from_nz(a::ArrayBlock{<:DistributedParamAllocationCOO{<:FullyAssembledRows}})
  f(x) = nothing
  A, = GridapDistributed._fa_create_from_nz_with_callback(f,a)
  return A
end

function Algebra.create_from_nz(a::DistributedParamAllocationCOO{<:SubAssembledRows})
  f(x) = nothing
  A, = GridapDistributed._sa_create_from_nz_with_callback(f,f,a,nothing)
  return A
end

function Algebra.create_from_nz(a::ArrayBlock{<:DistributedParamAllocationCOO{<:SubAssembledRows}})
  f(x) = nothing
  A, = GridapDistributed._sa_create_from_nz_with_callback(f,f,a,nothing)
  return A
end

function Algebra.nz_counter(
  builder::ParamBuilder{<:GridapDistributed.PVectorBuilder},
  axs::Tuple{<:PRange}
  )

  T = builder.builder.local_vector_type
  par_strategy = builder.builder.par_strategy
  plength = builder.plength
  rows, = axs
  counters = map(partition(rows)) do rows
    axs = (Base.OneTo(local_length(rows)),)
    counter = nz_counter(ArrayBuilder(T),axs)
    ParamCounter(counter,plength)
  end
  GridapDistributed.PVectorCounter(par_strategy,counters,rows)
end

function GridapDistributed._setup_touched_and_allocations_arrays(
  values::AbstractVector{<:AbstractParamVector})

  touched = map(values) do values
    fill!(Vector{Bool}(undef,innerlength(values)),false)
  end
  allocations = map(values,touched) do values,touched
    GridapDistributed.ArrayAllocationTrackTouchedAndValues(touched,values)
  end
  touched,allocations
end

function GridapDistributed._rhs_callback(
  partition::GridapDistributed.PVectorAllocationTrackOnlyValues{A,<:AbstractVector{<:AbstractParamVector}},
  rows) where A

  _param_rhs_callback(partition,rows)
end

function GridapDistributed._rhs_callback(
  partition::GridapDistributed.PVectorAllocationTrackTouchedAndValues{A,<:AbstractVector{<:AbstractParamVector}},
  rows) where A

  _param_rhs_callback(partition,rows)
end

function _param_rhs_callback(row_partitioned_vector_partition,rows)
  # The ghost values in row_partitioned_vector_partition are
  # aligned with the FESpace but not with the ghost values in the rows of A
  b_fespace = PVector(row_partitioned_vector_partition.values,
                      partition(row_partitioned_vector_partition.test_dofs_gids_prange))

  # This one is aligned with the rows of A
  b = similar(b_fespace,eltype(b_fespace),(rows,))

  # First transfer owned values
  b .= b_fespace

  # Now transfer ghost
  function transfer_ghost(b,b_fespace,ids,ids_fespace)
    num_ghosts_vec = ghost_length(ids)
    gho_to_loc_vec = ghost_to_local(ids)
    loc_to_glo_vec = local_to_global(ids)
    gid_to_lid_fe  = global_to_local(ids_fespace)
    for ghost_lid_vec in 1:num_ghosts_vec
      lid_vec = gho_to_loc_vec[ghost_lid_vec]
      gid = loc_to_glo_vec[lid_vec]
      lid_fespace = gid_to_lid_fe[gid]
      for i = param_eachindex(b)
        b.data[lid_vec,i] = b_fespace[i][lid_fespace]
      end
    end
  end
  map(
    transfer_ghost,
    partition(b),
    partition(b_fespace),
    b.index_partition,
    b_fespace.index_partition
  )

  return b
end

function PartitionedArrays.assemble_coo!(I,J,M::AbstractVector{<:AbstractMatrix},row_partition)
  """
    Returns three JaggedArrays with the coo triplets
    to be sent to the corresponding owner parts in parts_snd
  """
  function setup_snd(part,parts_snd,row_lids,coo_values)
    global_to_local_row = global_to_local(row_lids)
    local_row_to_owner = local_to_owner(row_lids)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_gi,k_gj,k_m = coo_values
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    gi_snd_data = zeros(eltype(k_gi),ptrs[end]-1)
    gj_snd_data = zeros(eltype(k_gj),ptrs[end]-1)
    v_snd_data = zeros(eltype(k_m),ptrs[end]-1)
    pv_snd_data = global_parameterize(v_snd_data,size(k_m,2))
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        gj = k_gj[k]
        p = ptrs[owner_to_i[owner]]
        gi_snd_data[p] = gi
        gj_snd_data[p] = gj
        ptrs[owner_to_i[owner]] += 1
        for i = param_eachindex(pv_snd_data)
          m = k_m[k,i]
          pv_snd_data.data[p,i] = m
          k_m[k,i] = zero(m)
        end
      end
    end
    rewind_ptrs!(ptrs)
    gi_snd = JaggedArray(gi_snd_data,ptrs)
    gj_snd = JaggedArray(gj_snd_data,ptrs)
    v_snd = JaggedArray(pv_snd_data,ptrs)
    gi_snd,gj_snd,v_snd
  end
  """
    Pushes to coo_values the triplets gi_rcv,gj_rcv,v_rcv
    received from remote processes
  """
  function setup_rcv!(coo_values,gi_rcv,gj_rcv,v_rcv)
    k_gi,k_gj,k_m = coo_values
    current_n = length(k_gi)
    new_n = current_n + length(gi_rcv.data)
    resize!(k_gi,new_n)
    resize!(k_gj,new_n)
    k_m′ = resize_rows(k_m,new_n)
    for p in 1:length(gi_rcv.data)
      k_gi[current_n+p] = gi_rcv.data[p]
      k_gj[current_n+p] = gj_rcv.data[p]
      for i in param_eachindex(v_rcv)
        k_m′[current_n+p,i] = v_rcv.data.data[p,i]
      end
    end
    k_m′
  end
  part = linear_indices(row_partition)
  parts_snd,parts_rcv = assembly_neighbors(row_partition)
  coo_values = map(tuple,I,J,M)
  gi_snd,gj_snd,v_snd = map(setup_snd,part,parts_snd,row_partition,coo_values) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t1 = exchange(gi_snd,graph)
  t2 = exchange(gj_snd,graph)
  t3 = exchange(v_snd,graph)
  @async begin
    gi_rcv = fetch(t1)
    gj_rcv = fetch(t2)
    v_rcv = fetch(t3)
    M′ = map(setup_rcv!,coo_values,gi_rcv,gj_rcv,v_rcv)
    I,J,M′
  end
end

function GridapDistributed.assemble_coo_with_column_owner!(
  I,J,M::AbstractArray{<:AbstractMatrix},row_partition,Jown)

  """
    Returns three (Param)JaggedArrays with the coo triplets
    to be sent to the corresponding owner parts in parts_snd
  """
  function setup_snd(part,parts_snd,row_lids,coo_entries_with_column_owner)
    global_to_local_row = global_to_local(row_lids)
    local_row_to_owner = local_to_owner(row_lids)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_gi,k_gj,k_jo,k_m = coo_entries_with_column_owner
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    PartitionedArrays.length_to_ptrs!(ptrs)
    gi_snd_data = zeros(eltype(k_gi),ptrs[end]-1)
    gj_snd_data = zeros(eltype(k_gj),ptrs[end]-1)
    jo_snd_data = zeros(eltype(k_jo),ptrs[end]-1)
    v_snd_data = zeros(eltype(k_m),ptrs[end]-1)
    pv_snd_data = global_parameterize(v_snd_data,size(k_m,2))
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        gj = k_gj[k]
        p = ptrs[owner_to_i[owner]]
        gi_snd_data[p] = gi
        gj_snd_data[p] = gj
        jo_snd_data[p] = k_jo[k]
        for i = param_eachindex(pv_snd_data)
          v = k_m[k,i]
          pv_snd_data.data[p,i] = v
          k_m[k,i] = zero(v)
        end
        ptrs[owner_to_i[owner]] += 1
      end
    end
    PartitionedArrays.rewind_ptrs!(ptrs)
    gi_snd = JaggedArray(gi_snd_data,ptrs)
    gj_snd = JaggedArray(gj_snd_data,ptrs)
    jo_snd = JaggedArray(jo_snd_data,ptrs)
    v_snd = JaggedArray(pv_snd_data,ptrs)
    gi_snd,gj_snd,jo_snd,v_snd
  end
  """
    Pushes to coo_entries_with_column_owner the tuples
    gi_rcv,gj_rcv,jo_rcv,v_rcv received from remote processes
  """
  function setup_rcv!(coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
    k_gi,k_gj,k_jo,k_m = coo_entries_with_column_owner
    current_n = length(k_gi)
    new_n = current_n + length(gi_rcv.data)
    resize!(k_gi,new_n)
    resize!(k_gj,new_n)
    resize!(k_jo,new_n)
    k_m′ = resize_rows(k_m,new_n)
    for p in 1:length(gi_rcv.data)
      k_gi[current_n+p] = gi_rcv.data[p]
      k_gj[current_n+p] = gj_rcv.data[p]
      k_jo[current_n+p] = jo_rcv.data[p]
      for i = param_eachindex(v_rcv)
        k_m′[current_n+p,i] = v_rcv.data[p,i]
      end
    end
    k_m′
  end
  part = linear_indices(row_partition)
  parts_snd,parts_rcv = assembly_neighbors(row_partition)
  coo_entries_with_column_owner = map(tuple,I,J,Jown,M)
  gi_snd,gj_snd,jo_snd,v_snd = map(setup_snd,part,parts_snd,row_partition,coo_entries_with_column_owner) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t1 = exchange(gi_snd,graph)
  t2 = exchange(gj_snd,graph)
  t3 = exchange(jo_snd,graph)
  t4 = exchange(v_snd,graph)
  @async begin
    gi_rcv = fetch(t1)
    gj_rcv = fetch(t2)
    jo_rcv = fetch(t3)
    v_rcv = fetch(t4)
    M′ = map(setup_rcv!,coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
    I,J,Jown,M′
  end
end
