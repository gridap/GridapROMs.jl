const DistributedSnapshots{T,N,I<:PartitionedDofMap,R} = Snapshots{T,N,I,R}

for T in (:PVector,:PSparseMatrix)
  @eval begin
    function ParamDataStructures.Snapshots(s::$T,i::PartitionedDofMap,r::AbstractRealisation)
      GenericSnapshots(get_all_data(s),s,i,r)
    end
  end
end

function ParamDataStructures.select_all_data(data::GenericPArray,args...)
  values = map(local_values(data)) do d
    select_all_data(d,args...)
  end
  GenericPArray(values,row_partition(data))
end

for f in (:partition,:local_values,:own_values,:ghost_values)
  @eval begin
    PartitionedArrays.$f(s::DistributedSnapshots) = $f(get_all_data(s))
  end
end

GridapDistributed.local_views(s::DistributedSnapshots) = partition(s)

function GridapDistributed.change_ghost(s::DistributedSnapshots,ids::PRange;kwargs...)
  data′ = change_ghost(get_param_data(s),ids;kwargs...)
  i = get_dof_map(s)
  r = get_realisation(s)
  Snapshots(data′,i,r)
end

# sparse interface

const DistributedSparseSnapshots{T,N,I<:PSparseMatrixDofMap,R} = DistributedSnapshots{T,N,I,R}

function DofMaps.recast(a::GenericPArray,i::PSparseMatrixDofMap)
  values = map(recast,local_values(a),local_views(i))
  PSparseMatrix(values,row_partition(i),col_partition(i))
end

# transient interface

const DistributedTransientSnapshots{T,N,I<:PartitionedDofMap,R<:TransientRealisation} = DistributedSnapshots{T,N,I,R}

function ParamDataStructures.Snapshots(
  s::PVector,
  s0::Tuple{Vararg{PVector}},
  i::PartitionedDofMap,
  r::TransientRealisation
  )

  snaps = Snapshots(s,i,r)
  TransientSnapshotsWithIC(s0,snaps)
end

function GridapDistributed.change_ghost(s::TransientSnapshotsWithIC,ids::PRange;kwargs...)
  data0′ = map(d0 -> change_ghost(d0,ids;kwargs...),get_initial_param_data(s))
  snaps′ = change_ghost(s.snaps,ids;kwargs...)
  TransientSnapshotsWithIC(data0′,snaps′)
end

const DistributedTransientSparseSnapshots{T,N,I<:PSparseMatrixDofMap,R<:TransientRealisation} = DistributedTransientSnapshots{T,N,I,R}

# multi-field interface

"""
    DistributedBlockSnapshots{S<:DistributedSnapshots,N} = BlockSnapshots{S,N}
"""
const DistributedBlockSnapshots{S<:DistributedSnapshots,N} = BlockSnapshots{S,N}

function ParamDataStructures.Snapshots(
  data::BlockPArray,
  i::AbstractArray{<:PartitionedDofMap},
  r::AbstractRealisation
  )

  block_values = blocks(data)
  array = map(enumerate(block_values)) do (j,dataj)
    Snapshots(dataj,i[j],r)
  end
  BlockSnapshots(array)
end

function ParamDataStructures.Snapshots(
  data::PVector,
  i::AbstractArray{<:PartitionedDofMap},
  r::AbstractRealisation
  )

  Snapshots(blockify(data,i),i,r)
end

function ParamDataStructures.Snapshots(
  data::BlockPArray,
  data0::Tuple{Vararg{BlockPArray}},
  i::AbstractArray{<:PartitionedDofMap},
  r::TransientRealisation
  )

  block_values = blocks(data)
  array = map(enumerate(block_values)) do (j,dataj)
    data0j = map(d0 -> blocks(d0)[j],data0)
    Snapshots(dataj,data0j,i[j],r)
  end
  BlockSnapshots(array)
end

function ParamDataStructures.Snapshots(
  data::PVector,
  data0::Tuple{Vararg{PVector}},
  i::AbstractArray{<:PartitionedDofMap},
  r::TransientRealisation
  )

  Snapshots(blockify(data,i),map(d0->blockify(d0,i),data0),i,r)
end

function ParamDataStructures.select_param_data(a::PVector,args...;kwargs...)
  vector_partition = map(local_values(a)) do d
    select_param_data(d,args...;kwargs...)
  end
  PVector(vector_partition,a.index_partition)
end

function ParamDataStructures.select_param_data(a::PSparseMatrix,args...;kwargs...)
  matrix_partition = map(local_values(a)) do d
    select_param_data(d,args...;kwargs...)
  end
  PSparseMatrix(matrix_partition,a.row_partition,a.col_partition)
end

function ParamDataStructures.select_param_data(a::BlockPArray,args...;kwargs...)
  map(blocks(a)) do p
    select_param_data(p,args...;kwargs...)
  end |> mortar
end

function PartitionedArrays.local_values(a::DistributedBlockSnapshots)
  map(local_values,blocks(a)) |> to_parray_of_arrays
end

function PartitionedArrays.own_values(a::DistributedBlockSnapshots)
  map(own_values,blocks(a)) |> to_parray_of_arrays
end

function PartitionedArrays.ghost_values(a::DistributedBlockSnapshots)
  map(ghost_values,blocks(a)) |> to_parray_of_arrays
end

function GridapDistributed.to_parray_of_arrays(a::AbstractArray{<:MPIArray{<:Snapshots}})
  indices = linear_indices(first(a))
  map(indices) do i
    array = map(aj -> getany(aj),a)
    BlockSnapshots(array)
  end
end

function GridapDistributed.to_parray_of_arrays(a::AbstractArray{<:DebugArray{<:Snapshots}})
  indices = linear_indices(first(a))
  map(indices) do i
    array = map(aj -> aj.items[i],a)
    BlockSnapshots(array)
  end
end

# index handling

flat_row_partition(a::DistributedSnapshots) = flat_row_partition(get_all_data(a))
row_partition(a::DistributedSnapshots) = row_partition(get_all_data(a))
col_partition(a::DistributedSnapshots) = col_partition(get_all_data(a))