function ParamDataStructures.Snapshots(s::PVector,i::AbstractArray,r::AbstractRealisation)
  data = map(local_views(s),local_views(i)) do s,i
    Snapshots(s,i,r)
  end
  snaps = GenericPArray(data,flat_row_partition(s))
  DistributedSnapshots(snaps)
end

function ParamDataStructures.Snapshots(s::PSparseMatrix,i::AbstractArray,r::AbstractRealisation)
  data = map(own_values(s),local_views(i)) do s,i
    Snapshots(s,i,r)
  end
  snaps = GenericPArray(data,flat_row_partition(s))
  DistributedSnapshots(snaps)
end

function ParamDataStructures.Snapshots(
  s::PVector,
  s0::Tuple{Vararg{PVector}},
  i::AbstractArray,
  r::TransientRealisation
  )

  data = map(local_views(s),local_views(i),local_views.(s0)...) do s,i,s0...
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
  vals = local_views(s)
  nparts = length(vals)
  map_main(vals) do s
    println(io,"Snapshots of partitioned size ($n,) - into $nparts parts - and unpartitioned sizes $(usizes)")
  end
end

ParamDataStructures.get_realisation(s::DistributedSnapshots) = get_realisation(getany(local_views(s)))

function ParamDataStructures.get_all_data(s::DistributedSnapshots)
  data = map(local_views(s)) do s
    get_all_data(s)
  end
  GenericPArray(data,row_partition(s))
end

function ParamDataStructures.get_param_data(s::DistributedSnapshots)
  data = map(local_views(s)) do s
    get_param_data(s)
  end
  PVector(data,row_partition(s))
end

function DofMaps.get_dof_map(s::DistributedSnapshots)
  map(local_views(s)) do s
    get_dof_map(s)
  end
end

function ParamDataStructures.select_snapshots(s::DistributedSnapshots,pindex)
  data = map(local_views(s)) do s
    select_snapshots(s,pindex)
  end
  snaps = GenericPArray(data,row_partition(s))
  DistributedSnapshots(snaps)
end

function ParamDataStructures.select_times(s::DistributedTransientSnapshots,tindex)
  data = map(local_views(s)) do s
    select_times(s,tindex)
  end
  snaps = GenericPArray(data,row_partition(s))
  DistributedSnapshots(snaps)
end

GridapDistributed.local_views(s::DistributedSnapshots) = local_views(s.snaps)
PartitionedArrays.partition(s::DistributedSnapshots) = partition(s.snaps)
PartitionedArrays.local_values(s::DistributedSnapshots) = local_values(s.snaps)
PartitionedArrays.own_values(s::DistributedSnapshots) = own_values(s.snaps)
PartitionedArrays.ghost_values(s::DistributedSnapshots) = ghost_values(s.snaps)

# sparse interface

const DistributedSparseSnapshots{T,N,I<:AbstractSparseDofMap,R,A} = DistributedSnapshots{T,N,I,R,A}

function DofMaps.recast(a::GenericPArray,i::AbstractArray{<:AbstractSparseDofMap})
  data = map(local_views(a),local_views(i)) do a,i
    recast(a,i)
  end
  PSparseMatrix(data,row_partition(a),col_partition(a))
end

function ParamDataStructures.get_param_data(s::DistributedSparseSnapshots)
  data = map(local_views(s)) do s
    get_param_data(s)
  end
  PSparseMatrix(data,row_partition(s),col_partition(s))
end

# multi-field interface

struct DistributedBlockSnapshots{N,B} <: AbstractSnapshots{DistributedSnapshots,N}
  array::AbstractArray{<:Any,N}
  touched::Array{Bool,N}
  param_data::B

  function DistributedBlockSnapshots(
    array::AbstractArray{<:Any,N},
    touched::Array{Bool,N},
    param_data::B
    ) where {N,B}

    @check size(array) == size(touched)
    new{N,B}(array,touched,param_data)
  end
end

const DistributedTransientBlockSnapshots{N} = DistributedBlockSnapshots{N,<:StoredParamData}

function ParamDataStructures.Snapshots(
  data::BlockPArray{V,T,N},
  i::ArrayBlock,
  r::AbstractRealisation
  ) where {V,T,N}

  block_values = blocks(data)
  s = size(block_values)
  array = Array{Any,N}(undef,s)
  for (j,dataj) in enumerate(block_values)
    if i.touched[j]
      array[j] = Snapshots(dataj,i[j],r)
    end
  end
  DistributedBlockSnapshots(array,i.touched,data)
end

function ParamDataStructures.Snapshots(
  data::Union{PVector,PSparseMatrix},
  i::ArrayBlock{<:Any,N},
  r::AbstractRealisation
  ) where N

  s = size(i)
  ids = ParamDataStructures.offset_indices(i)
  array = Array{Any,N}(undef,s)
  for j in eachindex(i)
    if i.touched[j]
      dataj = get_param_entry(data,ids[j]...)
      array[j] = Snapshots(dataj,i[j],r)
    end
  end

  DistributedBlockSnapshots(array,i.touched,data)
end

function ParamDataStructures.Snapshots(
  data::BlockPArray{V,T,N},
  data0::BlockPArray,
  i::ArrayBlock{<:Any,N},
  r::TransientRealisation
  ) where {V,T,N}

  block_values = blocks(data)
  s = size(block_values)
  @check s == size(i)

  array = Array{Any,N}(undef,s)
  for j in eachindex(block_values)
    if i.touched[j]
      dataj = block_values[j]
      data0j = map(d0 -> blocks(d0)[j],data0)
      array[j] = Snapshots(dataj,data0j,i[j],r)
    end
  end

  stored_data = StoredParamData(data,data0)
  DistributedBlockSnapshots(array,i.touched,stored_data)
end

function ParamDataStructures.Snapshots(
  data::PVector,
  data0::Tuple{Vararg{PVector}},
  i::ArrayBlock{<:Any,N},
  r::TransientRealisation
  ) where N

  s = size(i)
  ids = ParamDataStructures.offset_indices(i)
  array = Array{Any,N}(undef,s)
  for j in eachindex(i)
    if i.touched[j]
      dataj = get_param_entry(data,ids[j]...)
      data0j = map(d0 -> get_param_entry(d0,ids[j]...),data0)
      array[j] = Snapshots(dataj,data0j,i[j],r)
    end
  end

  stored_data = StoredParamData(data,data0)
  DistributedBlockSnapshots(array,i.touched,stored_data)
end

blocks(s::DistributedBlockSnapshots) = s.array

Base.size(s::DistributedBlockSnapshots) = size(s.array)

function Base.show(io::IO,k::MIME"text/plain",s::DistributedBlockSnapshots)
  vals = local_views(first(blocks(s)))
  nparts = length(vals)
  map_main(vals) do _
    println(io,"Block snapshots of size $(size(s)), partitioned into $nparts parts")
  end
end

function GridapDistributed.local_views(s::DistributedBlockSnapshots)
  a = map(local_values,blocks(s))
  to_parray_of_blocksnaps(a,s.touched)
end

function PartitionedArrays.partition(s::DistributedBlockSnapshots)
  a = map(partition,blocks(s))
  to_parray_of_blocksnaps(a,s.touched)
end

function to_parray_of_blocksnaps(a::AbstractArray{<:MPIArray{<:Snapshots}},touched)
  indices = linear_indices(first(a))
  map(indices) do i
    array = map(a) do aj
      getany(aj)
    end
    BlockSnapshots(array,touched,nothing)
  end
end

function to_parray_of_blocksnaps(a::AbstractArray{<:DebugArray{<:Snapshots}},touched)
  indices = linear_indices(first(a))
  map(indices) do i
    array = map(a) do aj
      aj.items[i]
    end
    BlockSnapshots(array,touched,nothing)
  end
end

# index handling for sparse matrix snapshots

struct NZIndexPartition{I<:AbstractLocalIndices,R<:AbstractLocalIndices,C<:AbstractLocalIndices} <: AbstractLocalIndices
  nz::I
  row::R
  col::C
end

PartitionedArrays.part_id(a::NZIndexPartition) = part_id(a.nz)
PartitionedArrays.local_to_global(a::NZIndexPartition) = local_to_global(a.nz)
PartitionedArrays.local_to_owner(a::NZIndexPartition) = local_to_owner(a.nz)
PartitionedArrays.own_to_global(a::NZIndexPartition) = own_to_global(a.nz)
PartitionedArrays.ghost_to_global(a::NZIndexPartition) = ghost_to_global(a.nz)
PartitionedArrays.ghost_to_owner(a::NZIndexPartition) = ghost_to_owner(a.nz)
PartitionedArrays.own_to_local(a::NZIndexPartition) = own_to_local(a.nz)
PartitionedArrays.ghost_to_local(a::NZIndexPartition) = ghost_to_local(a.nz)
PartitionedArrays.global_to_own(a::NZIndexPartition) = global_to_own(a.nz)
PartitionedArrays.global_to_local(a::NZIndexPartition) = global_to_local(a.nz)
PartitionedArrays.global_to_ghost(a::NZIndexPartition) = global_to_ghost(a.nz)
PartitionedArrays.own_length(a::NZIndexPartition) = own_length(a.nz)
PartitionedArrays.assembly_cache(a::NZIndexPartition) = PartitionedArrays.assembly_cache(a.nz)

function flat_row_partition(a::PSparseMatrix)
  nnz_own = map(own_values(a)) do aown
    inv_rows, = aown.inv_indices
    rv = rowvals(aown.parent)
    count(i -> inv_rows[i] > 0,rv)
  end
  n_nz_global = reduce(+,nnz_own,init=0)
  nz_part = variable_partition(nnz_own,n_nz_global)
  map(nz_part,local_views(a.row_partition),local_views(a.col_partition)) do nzidx,lrow,lcol
    NZIndexPartition(nzidx,lrow,lcol)
  end
end

flat_row_partition(a) = local_views(row_partition(a))

row_partition(a) = a
row_partition(a::PVector) = a.index_partition
row_partition(a::PSparseMatrix) = a.row_partition
row_partition(a::GenericPArray) = row_partition(a.index_partition)
row_partition(a::DistributedSnapshots) = row_partition(a.snaps)
row_partition(a::NZIndexPartition) = a.row
row_partition(a::AbstractArray{<:NZIndexPartition}) = map(row_partition,local_views(a))

col_partition(a) = a
col_partition(a::PVector) = @notimplemented
col_partition(a::PSparseMatrix) = a.col_partition
col_partition(a::GenericPArray) = col_partition(a.index_partition)
col_partition(a::DistributedSnapshots) = col_partition(a.snaps)
col_partition(a::NZIndexPartition) = a.col
col_partition(a::AbstractArray{<:NZIndexPartition}) = map(col_partition,local_views(a))

# linear algebra 

_gettr(a) = a
_gettr(a::DistributedSnapshots) = a.snaps 

for S in (:AbstractMatrix,:PSparseMatrix,:GenericPMatrix,:DistributedSnapshots), T in (:AbstractMatrix,:PSparseMatrix,:GenericPMatrix,:DistributedSnapshots)
  !(S == :DistributedSnapshots || T == :DistributedSnapshots) && continue
  @eval begin
    Base.:*(a::$S,b::$T) = _gettr(a) * _gettr(b)
    Base.:*(a::Adjoint{<:Any,<:$S},b::$T) = _gettr(a.parent)' * _gettr(b)
    Base.:*(a::$S,b::Adjoint{<:Any,<:$T}) = _gettr(a) * _gettr(b.parent)'
    Base.:*(a::Adjoint{<:Any,<:$S},b::Adjoint{<:Any,<:$T}) = _gettr(a.parent)' * _gettr(b.parent)'
  end
end