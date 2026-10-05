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

# ParamDataStructures.offset_indices(i) (used in the serial field-splitting
# counterpart in FEM/ParamDataStructures/Snapshots.jl) assumes field j's dofs
# occupy one contiguous GLOBAL range -- true in serial (and happens to still
# hold on exactly 1 rank), but false in general for a DistributedMultiField
# FESpace with the default (consecutive) numbering: GridapDistributed's own
# generate_multi_field_gids numbers dofs RANK-MAJOR then field-minor (each
# rank gets one contiguous block of global ids containing ALL its own
# fields' own dofs concatenated), not field-major -- so field j's dofs are
# scattered across one disjoint sub-block per rank, interleaved with every
# other field's sub-blocks from that same rank. Verified directly: on a
# 2-rank, 2-field toy space, rank 1 owns globals 1-10 (field 1) + 11-20
# (field 2), rank 2 owns 21-35 (field 1) + 36-50 (field 2) -- field 1 is
# {1..10}∪{21..35}, not one range.
#
# What IS field-major-contiguous is each RANK's OWN local (own+ghost) data:
# locally, a rank's combined array holds field 1's local entries then field
# 2's, etc., the same way a serial multi-field space would. So the per-rank
# LOCAL offset for field j is just the cumulative LOCAL size of the
# preceding fields *on that same rank*, which needs nothing beyond each
# field's own per-rank dof-map length (already in `i`, no FE space
# involved). _local_field_ranges computes these.
# shared core: `sizes[j]` is a PArray (over ranks) of field j's LOCAL
# (own+ghost) dof count; returns, for each field, a PArray of per-rank LOCAL
# index ranges within that rank's combined local array. Used both here, from
# dof-map lengths (no FE space needed), and in EnergyNorms.jl's EnergyNorm
# multi-field case, from each field's own get_free_dof_ids -- both measure
# the same per-rank local size, just from different, equally valid sources,
# so the resulting ranges (and hence the restricted partitions built from
# them) agree between the two.
function _local_ranges_from_sizes(sizes::AbstractVector)
  nfields = length(sizes)
  offsets = Vector{Any}(undef,nfields)
  offsets[1] = map(l -> zero(l),sizes[1])
  for j in 2:nfields
    offsets[j] = map(+,offsets[j-1],sizes[j-1])
  end
  map(1:nfields) do j
    map((o,l) -> o+1:o+l,offsets[j],sizes[j])
  end
end

# # i is field-outer (i[j] is a PArray over ranks), so offset_indices(i) can't
# # be called on it directly: it would compute length(i[j]) = the number of
# # ranks, not field j's dof count. Transposing to rank-outer first -- for
# # each rank, a plain Vector of that rank's own per-field (scalar) dof maps
# # -- makes length(i[j]) mean what offset_indices expects again (field j's
# # LOCAL dof count on that one rank), reusing the exact same serial logic
# # locally per rank instead of a parallel reimplementation.
# function _local_field_ranges(i::AbstractArray{<:AbstractArray{<:AbstractDofMap}})
#   per_rank = map((ijs...) -> collect(ijs),i...)
#   ranges_per_rank = map(ParamDataStructures.offset_indices,per_rank)
#   nfields = length(i)
#   map(1:nfields) do j
#     map(rr -> rr[j][1],ranges_per_rank)
#   end
# end

function ParamDataStructures.offset_indices(i::AbstractArray{<:AbstractArray{<:AbstractDofMap},N}) where N
  array = Array{Any,N}(undef,size(i))
  offset = 0
  for j in eachindex(i)
    n = sreduce(map(length,i[j]))
    array[j] = (offset+1:offset+n,)
    offset += n
  end
  return array
end

# Restricts `old_row_partition` to the per-rank LOCAL index ranges in
# `local_ranges` (as returned by _local_field_ranges), by keeping only the
# OWN entries therein and assigning them a fresh, ghost-free global
# numbering via a distributed exclusive prefix sum of each rank's own-count
# -- the same technique GridapDistributed's generate_multi_field_gids uses
# internally to number fields within the combined space, just applied here
# to carve a field back out. Ghost-free is fine since method_of_snapshots/
# tpod (the only consumer) only ever reads own_values.
function _restrict_rows_to_local_range(old_row_partition,local_ranges)
  owners = map(part_id,old_row_partition)
  own_positions = map(old_row_partition,local_ranges) do idx,lr
    l2o = local_to_owner(idx)
    owner_p = part_id(idx)
    filter(l -> l2o[l]==owner_p,collect(lr))
  end
  n_owns = map(length,own_positions)
  firstgid = scan(+,n_owns;init=1,type=:exclusive)
  ngids = PartitionedArrays.reduction(+,n_owns;destination=:all,init=0)
  info = map(own_positions,firstgid,ngids,owners) do ownpos,fg,ng,owner_p
    n = length(ownpos)
    new_l2g = collect(fg:fg+n-1)
    new_l2o = fill(Int32(owner_p),n)
    new_indices = LocalIndices(ng,owner_p,new_l2g,new_l2o)
    (ownpos,new_indices)
  end
  tuple_of_arrays(info)
end

function _restrict_to_field(a::PVector,local_range)
  local_positions,new_row_partition = _restrict_rows_to_local_range(partition(axes(a,1)),local_range)
  new_values = map(partition(a),local_positions) do values,lids
    get_param_entry(values,lids)
  end
  PVector(new_values,new_row_partition)
end

function _restrict_to_field(a::PSparseMatrix,local_range)
  local_positions,new_row_partition = _restrict_rows_to_local_range(partition(axes(a,1)),local_range)
  new_values = map(partition(a),local_positions) do values,lids
    get_param_entry(values,lids)
  end
  PSparseMatrix(new_values,new_row_partition,partition(axes(a,2)))
end

function ParamDataStructures.Snapshots(
  data::Union{PVector,PSparseMatrix},
  i::AbstractArray{<:AbstractArray{<:AbstractDofMap}},
  r::AbstractRealisation
  )

  s = size(i)
  ids = ParamDataStructures.offset_indices(i)
  array = map(eachindex(i)) do j
    dataj = _restrict_to_field(data,ids[j])
    Snapshots(dataj,i[j],r)
  end
  BlockSnapshots(reshape(array,s),data)
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
  ids = ParamDataStructures.offset_indices(i)
  array = map(eachindex(i)) do j
    dataj = _restrict_to_field(data,ids[j])
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