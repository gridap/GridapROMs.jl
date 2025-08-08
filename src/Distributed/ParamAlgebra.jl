function GridapDistributed._setup_touched_and_allocations_arrays(
  values::AbstractVector{<:AbstractParamVector})

  touched = map(values) do values
    fill!(Vector{Bool}(undef,innerlength(values)),false)
  end
  allocations = map(values,touched) do values,touched
    ArrayAllocationTrackTouchedAndValues(touched,values)
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
      lid_vec     = gho_to_loc_vec[ghost_lid_vec]
      gid         = loc_to_glo_vec[lid_vec]
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
    b_fespace.index_partition)

  return b
end

function PartitionedArrays.assemble_coo!(
  I,J,V::AbstractArray{<:AbstractParamVector},row_partition)
  """
    Returns three JaggedArrays with the coo triplets
    to be sent to the corresponding owner parts in parts_snd
  """
  function setup_snd(part,parts_snd,row_lids,coo_values)
    global_to_local_row = global_to_local(row_lids)
    local_row_to_owner = local_to_owner(row_lids)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_gi,k_gj,k_v = coo_values
    for k in 1:length(k_gi)
      gi = k_gi[k]
      li = global_to_local_row[gi]
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
      k_gi[current_n+p] = gi_rcv.data[p]
      k_gj[current_n+p] = gj_rcv.data[p]
      for i = param_eachindex(k_v)
        k_v.data[current_n+p,i] = v_rcv.data[p,i]
      end
    end
  end
  part = linear_indices(row_partition)
  parts_snd,parts_rcv = assembly_neighbors(row_partition)
  coo_values = map(tuple,I,J,V)
  gi_snd,gj_snd,v_snd = map(setup_snd,part,parts_snd,row_partition,coo_values) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  t1 = exchange(gi_snd,graph)
  t2 = exchange(gj_snd,graph)
  t3 = exchange(v_snd,graph)
  @async begin
    gi_rcv = fetch(t1)
    gj_rcv = fetch(t2)
    v_rcv = fetch(t3)
    map(setup_rcv!,coo_values,gi_rcv,gj_rcv,v_rcv)
    I,J,V
  end
end

function GridapDistributed.assemble_coo_with_column_owner!(
  I,J,V::AbstractArray{<:AbstractParamVector},row_partition,Jown)

  """
    Returns three (Param)JaggedArrays with the coo triplets
    to be sent to the corresponding owner parts in parts_snd
  """
  function setup_snd(part,parts_snd,row_lids,coo_entries_with_column_owner)
    global_to_local_row = global_to_local(row_lids)
    local_row_to_owner = local_to_owner(row_lids)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    k_gi,k_gj,k_jo,k_v = coo_entries_with_column_owner
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
    v_snd_data = zeros(eltype2(k_v),ptrs[end]-1)
    pv_snd_data = global_parameterize(v_snd_data,param_length(k_v))
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
          v = k_v.data[k,i]
          pv_snd_data.data[p,i] = v
          k_v.data[k,i] = zero(v)
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
    k_gi,k_gj,k_jo,k_v = coo_entries_with_column_owner
    current_n = length(k_gi)
    new_n = current_n + length(gi_rcv.data)
    resize!(k_gi,new_n)
    resize!(k_gj,new_n)
    resize!(k_jo,new_n)
    resize!(k_v,new_n)
    for p in 1:length(gi_rcv.data)
      k_gi[current_n+p] = gi_rcv.data[p]
      k_gj[current_n+p] = gj_rcv.data[p]
      k_jo[current_n+p] = jo_rcv.data[p]
      for i = param_eachindex(k_v)
        k_v.data[current_n+p,i] = v_rcv.data[p,i]
      end
    end
  end
  part = linear_indices(row_partition)
  parts_snd,parts_rcv = assembly_neighbors(row_partition)
  coo_entries_with_column_owner = map(tuple,I,J,Jown,V)
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
      map(setup_rcv!,coo_entries_with_column_owner,gi_rcv,gj_rcv,jo_rcv,v_rcv)
      I,J,Jown,V
  end
end
