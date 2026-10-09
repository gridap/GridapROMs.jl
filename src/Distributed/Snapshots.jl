const DistributedSnapshots{T,N,I<:AbstractArray{<:AbstractDofMap},R,A<:GenericPArray,B} = GenericSnapshots{T,N,I,R,A,B}

for T in (:PVector,:PSparseMatrix)
  @eval begin
    function ParamDataStructures.Snapshots(s::$T,i::AbstractArray{<:AbstractDofMap},r::Realisation)
      GenericSnapshots(get_all_data(s),s,i,r)
    end

    function ParamDataStructures.Snapshots(s::$T,i::AbstractArray{<:AbstractDofMap},r::TransientRealisation)
      data = get_all_data(s)
      dims(d) = (innerlength(d),num_params(r),num_times(r))
      idata = map(d -> reshape(d,dims(d)),local_values(data))
      pidata = GenericPArray(idata,row_partition(data))
      GenericSnapshots(pidata,s,i,r)
    end
  end
end

function ParamDataStructures.select_all_data(s::DistributedSnapshots,args...;kwargs...)
  data = get_all_data(s)
  values = map(local_values(data)) do d
    select_all_data(d,args...;kwargs...)
  end
  GenericPArray(values,row_partition(data))
end

for f in (:partition,:local_values,:own_values,:ghost_values)
  @eval begin
    PartitionedArrays.$f(s::DistributedSnapshots) = $f(s.data)
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

const DistributedSparseSnapshots{T,N,I<:AbstractArray{<:AbstractSparseDofMap},R,A<:GenericPArray,B} = DistributedSnapshots{T,N,I,R,A,B}

function DofMaps.recast(a::GenericPArray,i::AbstractArray{<:AbstractSparseDofMap})
  PSparseMatrix(partition(a),row_partition(i),col_partition(i))
end

# transient interface

const DistributedTransientSnapshots{T,N,I<:AbstractArray{<:AbstractDofMap},R<:TransientRealisation,A<:GenericPArray,B} = DistributedSnapshots{T,N,I,R,A,B}

const DistributedTransientSnapshotsWithIC{T,N,I,R,A,B<:DistributedTransientSnapshots} = TransientSnapshotsWithIC{T,N,I,R,A,B}

function ParamDataStructures.Snapshots(
  s::PVector,
  s0::Tuple{Vararg{PVector}},
  i::AbstractArray{<:AbstractDofMap},
  r::TransientRealisation
  )

  snaps = Snapshots(s,i,r)
  TransientSnapshotsWithIC(s0,snaps)
end

for f in (:partition,:local_values,:own_values,:ghost_values)
  @eval begin
    PartitionedArrays.$f(s::DistributedTransientSnapshotsWithIC) = $f(s.snaps)
  end
end

GridapDistributed.local_views(s::DistributedTransientSnapshotsWithIC) = partition(s)

function GridapDistributed.change_ghost(s::DistributedTransientSnapshotsWithIC,ids::PRange;kwargs...)
  data′ = change_ghost(get_param_data(s),ids;kwargs...)
  data0′ = map(d0 -> change_ghost(d0,ids;kwargs...),get_initial_param_data(s))
  i = get_dof_map(s)
  r = get_realisation(s)
  Snapshots(data′,data0′,i,r)
end

# multi-field interface

"""
    DistributedBlockSnapshots{S<:DistributedSnapshots,N} = BlockSnapshots{S,N}
"""
const DistributedBlockSnapshots{S<:Union{
  DistributedSnapshots,
  DistributedTransientSnapshots,
  DistributedTransientSnapshotsWithIC},N
  } = BlockSnapshots{S,N}

function ParamDataStructures.Snapshots(
  data::BlockPArray,
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
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
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
  r::AbstractRealisation
  )

  Snapshots(blockify(data,i),i,r)
end

function ParamDataStructures.Snapshots(
  data::BlockPArray,
  data0::Tuple{Vararg{BlockPArray}},
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
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
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
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

flat_row_partition(a::DistributedSnapshots) = flat_row_partition(a.data)
row_partition(a::DistributedSnapshots) = row_partition(a.data)
col_partition(a::DistributedSnapshots) = col_partition(a.data)

# linear algebra

_getvals(a) = a
_getvals(a::DistributedSnapshots) = a.data

for S in (:AbstractMatrix,:PSparseMatrix,:GenericPMatrix,:DistributedSnapshots), T in (:AbstractMatrix,:PSparseMatrix,:GenericPMatrix,:DistributedSnapshots)
  !(S == :DistributedSnapshots || T == :DistributedSnapshots) && continue
  @eval begin
    Base.:*(a::$S,b::$T) = _getvals(a) * _getvals(b)
    Base.:*(a::Adjoint{<:Any,<:$S},b::$T) = _getvals(a.parent)' * _getvals(b)
    Base.:*(a::$S,b::Adjoint{<:Any,<:$T}) = _getvals(a) * _getvals(b.parent)'
    Base.:*(a::Adjoint{<:Any,<:$S},b::Adjoint{<:Any,<:$T}) = _getvals(a.parent)' * _getvals(b.parent)'
  end
end

for op in (:+,:-)
  @eval function Base.$op(a::DistributedSnapshots,b::DistributedSnapshots)
    $op(a.data,b.data)
  end
end