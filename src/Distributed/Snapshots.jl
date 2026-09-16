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

# sparse interface

const DistributedSparseSnapshots{T,N,I<:AbstractSparseDofMap,R,A} = DistributedSnapshots{T,N,I,R,A}
const DistributedTransientSparseSnapshots{T,N,I<:AbstractSparseDofMap,R<:TransientRealisation,A} = TransientSparseSnapshots{T,N,I,R,A}

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
    const DistributedBlockSnapshots{S<:DistributedSnapshots,N,B} = BlockSnapshots{S,N,B}
"""
const DistributedBlockSnapshots{S<:DistributedSnapshots,N,B} = BlockSnapshots{S,N,B}

const DistributedTransientBlockSnapshots{N} = DistributedBlockSnapshots{<:Any,N,<:StoredParamData}

function ParamDataStructures.Snapshots(
  data::BlockPArray{V,T,N},
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
  r::AbstractRealisation
  ) where {V,T,N}

  block_values = blocks(data)
  array = map(enumerate(block_values)) do (j,dataj)
    Snapshots(dataj,i[j],r)
  end
  BlockSnapshots(array,data)
end

function ParamDataStructures.Snapshots(
  data::Union{PVector,PSparseMatrix},
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
  r::AbstractRealisation
  )

  s = size(i)
  ids = ParamDataStructures.offset_indices(i)
  array = map(eachindex(i)) do j
    dataj = get_param_entry(data,ids[j]...)
    Snapshots(dataj,i[j],r)
  end
  BlockSnapshots(reshape(array,s),data)
end

function ParamDataStructures.Snapshots(
  data::BlockPArray{V,T,N},
  data0::BlockPArray,
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
  r::TransientRealisation
  ) where {V,T,N}

  block_values = blocks(data)
  s = size(block_values)
  @check s == size(i)
  array = map(enumerate(block_values)) do (j,dataj)
    data0j = map(d0 -> blocks(d0)[j],data0)
    Snapshots(dataj,data0j,i[j],r)
  end
  stored_data = StoredParamData(data,data0)
  BlockSnapshots(array,stored_data)
end

function ParamDataStructures.Snapshots(
  data::PVector,
  data0::Tuple{Vararg{PVector}},
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
  r::TransientRealisation
  )

  s = size(i)
  ids = ParamDataStructures.offset_indices(i)
  array = map(eachindex(i)) do j
    dataj = get_param_entry(data,ids[j]...)
    data0j = map(d0 -> blocks(d0)[j],data0)
    Snapshots(dataj,data0j,i[j],r)
  end
  stored_data = StoredParamData(data,data0)
  BlockSnapshots(reshape(array,s),stored_data)
end

function ParamDataStructures.select_snapshots(s::DistributedBlockSnapshots,pindex)
  array = map(sj -> select_snapshots(sj,pindex),blocks(s))
  pdata = mortar(map(get_param_data,array))
  BlockSnapshots(array,pdata)
end

function ParamDataStructures.select_times(s::DistributedBlockSnapshots,tindex)
  array = map(sj -> select_times(sj,tindex),blocks(s))
  pdata = mortar(map(get_param_data,array))
  BlockSnapshots(array,pdata)
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
    array,data = map(a) do aj
      s = getany(aj)
      d = get_param_data(s)
      s,d
    end |> tuple_of_arrays
    BlockSnapshots(array,data)
  end
end

function GridapDistributed.to_parray_of_arrays(a::AbstractArray{<:DebugArray{<:Snapshots}})
  indices = linear_indices(first(a))
  map(indices) do i
    array,data = map(a) do aj
      s = aj.items[i]
      d = get_param_data(s)
      s,d
    end |> tuple_of_arrays
    BlockSnapshots(array,data)
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