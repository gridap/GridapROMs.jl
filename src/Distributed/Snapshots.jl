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
    const DistributedBlockSnapshots{S<:DistributedSnapshots,N,B} = BlockSnapshots{S,N,B}
"""
const DistributedBlockSnapshots{S<:DistributedSnapshots,N,B} = BlockSnapshots{S,N,B}

const DistributedTransientBlockSnapshots{N} = DistributedBlockSnapshots{<:DistributedTransientSnapshots,N,<:StoredParamData}

function ParamDataStructures.Snapshots(
  data::BlockPArray,
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
  r::AbstractRealisation
  )

  block_values = blocks(data)
  array = map(enumerate(block_values)) do (j,dataj)
    Snapshots(dataj,i[j],r)
  end
  BlockSnapshots(array,data)
end

function ParamDataStructures.Snapshots(
  data::PVector,
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
  r::AbstractRealisation
  )

  s = size(i)
  offsets = _get_local_ranges(i)
  array = map(eachindex(i)) do j
    dataj = get_param_entry(data,offsets[j])
    Snapshots(dataj,i[j],r)
  end
  BlockSnapshots(reshape(array,s),data)
end

# i here is the (ntest,ntrial) field-pair dof-map grid from get_sparse_dof_map
# (each i[k,l] a TrivialSparseMatrixDofMap/SparseMatrixDofMap built from the
# already row/col-restricted (k,l) block -- see restr_to_fields in
# ParamFESpaces.jl), NOT a 1D per-field array like the PVector case above:
# linearizing it the same way (treating field-pair (k,l) as the "field" at
# linear index k+(l-1)*ntest, and using each pair's own NNZ count for the
# offset) mixes the test and trial dimensions into one sequence, which does
# not correspond to "test field k's own rows" or "trial field l's own
# columns" at all. Instead, the row range for test field k (same for every
# l) and the column range for trial field l (same for every k) are computed
# independently, each DOF-based (via num_rows/num_cols on the dof-map's own
# sparsity, no FE space needed), exactly matching the PVector/residual
# convention -- then get_param_entry(data,row_range,col_range) (which scans
# for the matching NZ positions itself, the efficient CSC/CSR-native way to
# do this restriction, not a contiguous-NNZ-range shortcut) extracts each
# field-pair's own block.
function ParamDataStructures.Snapshots(
  data::PSparseMatrix,
  i::AbstractMatrix{<:AbstractArray{<:AbstractDofMap}},
  r::AbstractRealisation
  )

  ntest,ntrial = size(i)
  row_sizes = map(k -> map(dm -> DofMaps.num_rows(get_sparsity(dm)),i[k,1]),1:ntest)
  col_sizes = map(l -> map(dm -> DofMaps.num_cols(get_sparsity(dm)),i[1,l]),1:ntrial)
  row_ranges = _ranges_from_sizes(row_sizes)
  col_ranges = _ranges_from_sizes(col_sizes)
  array = map(Iterators.product(1:ntest,1:ntrial)) do (k,l)
    dataj = get_param_entry(data,row_ranges[k],col_ranges[l])
    Snapshots(dataj,i[k,l],r)
  end
  BlockSnapshots(array,data)
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
  offsets = _get_local_ranges(i)
  array = map(eachindex(i)) do j
    dataj = get_param_entry(data,offsets[j])
    data0j = map(d0 -> blocks(d0)[j],data0)
    Snapshots(dataj,data0j,i[j],r)
  end
  stored_data = StoredParamData(data,data0)
  BlockSnapshots(reshape(array,s),stored_data)
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