for T in (:PVector,:PSparseMatrix)
  @eval begin
    function ParamDataStructures.Snapshots(s::$T,i::AbstractArray{<:AbstractDofMap},r::AbstractRealisation)
      data = map(local_values(s),i) do s,i
        Snapshots(s,i,r)
      end
      snaps = GenericPArray(data,flat_row_partition(s))
      DistributedSnapshots(snaps)
    end
  end
end

function ParamDataStructures.Snapshots(
  s::PVector,
  s0::Tuple{Vararg{PVector}},
  i::AbstractArray{<:AbstractDofMap},
  r::TransientRealisation
  )

  data = map(local_values(s),i,local_values.(s0)...) do s,i,s0...
    Snapshots(s,s0,i,r)
  end
  snaps = GenericPArray(data,flat_row_partition(s))
  DistributedSnapshots(snaps)
end

struct DistributedSnapshots{T,N,I,R,A} <: Snapshots{T,N,I,R}
  snaps::A
  function DistributedSnapshots(snaps::GenericPArray{<:Snapshots{T,N,I,R}}) where {T,N,I,R}
    A = typeof(snaps)
    new{T,N,I,R,A}(snaps)
  end
end

const DistributedTransientSnapshots{T,N,I,R<:TransientRealisation,A} = DistributedSnapshots{T,N,I,R,A}

Base.size(s::DistributedSnapshots) = size(s.snaps)
Base.axes(s::DistributedSnapshots) = axes(s.snaps)
Base.getindex(s::DistributedSnapshots,ids...) = getindex(s.snaps,ids...)
Base.setindex!(s::DistributedSnapshots,v,ids...) = setindex!(s.snaps,v,ids...)

function Base.show(io::IO,k::MIME"text/plain",s::DistributedSnapshots)
  n,usizes... = size(s)
  vals = local_values(s)
  nparts = length(vals)
  map_main(vals) do s
    println(io,"Snapshots of partitioned size ($n,) - into $nparts parts - and unpartitioned sizes $(usizes)")
  end
end

ParamDataStructures.get_realisation(s::DistributedSnapshots) = get_realisation(getany(local_values(s)))

function ParamDataStructures.get_all_data(s::DistributedSnapshots)
  data = map(local_values(s)) do s
    get_all_data(s)
  end
  GenericPArray(data,flat_row_partition(s))
end

function ParamDataStructures.get_param_data(s::DistributedSnapshots)
  data = map(local_values(s)) do s
    get_param_data(s)
  end
  PVector(data,row_partition(s))
end

function ParamDataStructures.get_initial_param_data(s::DistributedSnapshots)
  data = map(local_values(s)) do s
    get_initial_param_data(s)
  end |> tuple_of_arrays
  map(d->PVector(d,row_partition(s)),data)
end

function DofMaps.get_dof_map(s::DistributedSnapshots)
  map(local_values(s)) do s
    get_dof_map(s)
  end
end

function DofMaps.flatten(s::DistributedSnapshots)
  data = map(local_values(s)) do s
    flatten(s)
  end
  GenericPArray(data,flat_row_partition(s))
end

function ParamDataStructures.select_snapshots(s::DistributedSnapshots,pindex)
  data = map(local_values(s)) do s
    select_snapshots(s,pindex)
  end
  snaps = GenericPArray(data,flat_row_partition(s))
  DistributedSnapshots(snaps)
end

function ParamDataStructures.select_times(s::DistributedTransientSnapshots,tindex)
  data = map(local_values(s)) do s
    select_times(s,tindex)
  end
  snaps = GenericPArray(data,flat_row_partition(s))
  DistributedSnapshots(snaps)
end

PartitionedArrays.partition(s::DistributedSnapshots) = partition(s.snaps)
PartitionedArrays.local_values(s::DistributedSnapshots) = partition(s)
PartitionedArrays.own_values(s::DistributedSnapshots) = own_values(s.snaps)
PartitionedArrays.ghost_values(s::DistributedSnapshots) = ghost_values(s.snaps)
GridapDistributed.local_views(s::DistributedSnapshots) = partition(s)

function GridapDistributed.change_ghost(s::DistributedSnapshots,ids::PRange;kwargs...)
  data′ = change_ghost(get_param_data(s),ids;kwargs...)
  i = get_dof_map(s)
  r = get_realisation(s)
  Snapshots(data′,i,r)
end

function GridapDistributed.change_ghost(s::DistributedTransientSnapshots,ids::PRange;kwargs...)
  data′ = change_ghost(get_param_data(s),ids;kwargs...)
  data0′ = map(d0 -> change_ghost(d0,ids;kwargs...),get_initial_param_data(s))
  i = get_dof_map(s)
  r = get_realisation(s)
  Snapshots(data′,data0′,i,r)
end

# sparse interface

const DistributedSparseSnapshots{T,N,I<:AbstractSparseDofMap,R,A} = DistributedSnapshots{T,N,I,R,A}
const DistributedTransientSparseSnapshots{T,N,I<:AbstractSparseDofMap,R<:TransientRealisation,A} = DistributedTransientSnapshots{T,N,I,R,A}

function DofMaps.recast(a::GenericPArray,i::AbstractArray{<:AbstractSparseDofMap})
  data = map(local_values(a),i) do a,i
    recast(a,i)
  end
  PSparseMatrix(data,row_partition(a),col_partition(a))
end

function ParamDataStructures.get_param_data(s::DistributedSparseSnapshots)
  data = map(local_values(s)) do s
    get_param_data(s)
  end
  PSparseMatrix(data,row_partition(s),col_partition(s))
end

# multi-field interface

"""
    const DistributedBlockSnapshots{S<:DistributedSnapshots,N} = BlockSnapshots{S,N}
"""
const DistributedBlockSnapshots{S<:DistributedSnapshots,N} = BlockSnapshots{S,N}

const DistributedTransientBlockSnapshots{N} = DistributedBlockSnapshots{<:DistributedTransientSnapshots,N}

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

  offsets = _get_local_ranges(i)
  array = map(enumerate(i)) do (j,ij)
    dataj = get_param_entry(data,offsets[j])
    Snapshots(dataj,ij,r)
  end
  BlockSnapshots(array)
end

function ParamDataStructures.Snapshots(
  data::BlockPArray,
  data0::Tuple{Vararg{BlockPArray}},
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
  r::TransientRealisation
  )

  block_values = blocks(data)
  s = size(block_values)
  @check s == size(i)
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

  offsets = _get_local_ranges(i)
  array = map(enumerate(i)) do (j,ij)
    dataj = get_param_entry(data,offsets[j])
    data0j = map(d0 -> get_param_entry(d0,offsets[j]),data0)
    Snapshots(dataj,data0j,ij,r)
  end
  BlockSnapshots(array)
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

function Base.show(io::IO,k::MIME"text/plain",s::DistributedBlockSnapshots)
  vals = local_values(first(blocks(s)))
  nparts = length(vals)
  map_main(vals) do _
    println(io,"Block snapshots of size $(size(s)), partitioned into $nparts parts")
  end
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

flat_row_partition(a::DistributedSnapshots) = flat_row_partition(a.snaps)
row_partition(a::DistributedSnapshots) = row_partition(a.snaps)
col_partition(a::DistributedSnapshots) = col_partition(a.snaps)

# linear algebra 

_getvals(a) = a
_getvals(a::DistributedSnapshots) = a.snaps 

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
    $op(a.snaps,b.snaps)
  end
end

# utils

function _get_local_ranges(i::AbstractArray{<:AbstractArray})
  llength(a) = length(a)
  llength(a::AbstractLocalIndices) = local_length(a)
  lengths = map(ij -> map(llength,ij),i)
  nfields = length(lengths)
  offsets = Vector{Any}(undef,nfields)
  offsets[1] = map(l -> zero(l),lengths[1])
  for j in 2:nfields
    offsets[j] = map(+,offsets[j-1],lengths[j-1])
  end
  map(1:nfields) do j
    map((o,l) -> o+1:o+l,offsets[j],lengths[j])
  end
end