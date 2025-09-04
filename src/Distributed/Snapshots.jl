for T in (:PVector,:PSparseMatrix)
  @eval begin
    function ParamDataStructures.Snapshots(s::$T,i::AbstractArray,r::AbstractRealization)
      data = map(local_views(s),local_views(i)) do s,i
        Snapshots(s,i,r)
      end
      snaps = GenericPArray(data,s.index_partition)
      DistributedSnapshots(snaps)
    end
  end
end

function ParamDataStructures.Snapshots(s::PVector,s0::PVector,i::AbstractArray,r::TransientRealization)
  data = map(local_views(s),local_views(s0),local_views(i)) do s,s0,i
    Snapshots(s,s0,i,r)
  end
  snaps = GenericPArray(data,s.index_partition)
  DistributedSnapshots(snaps)
end

struct DistributedSnapshots{T,N,A,B} <: AbstractSnapshots{T,N}
  snaps::A
  function DistributedSnapshots(snaps::GenericPArray{B}) where {T,N,B<:AbstractSnapshots{T,N}}
    A = typeof(snaps)
    new{T,N,A,B}(snaps)
  end
end

const DistributedSteadySnapshots{T,N,A} = DistributedSnapshots{T,N,A,<:SteadySnapshots}

const DistributedTransientSnapshots{T,N,A} = DistributedSnapshots{T,N,A,<:TransientSnapshots}

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

ParamDataStructures.get_realization(s::DistributedSnapshots) = get_realization(getany(local_views(s)))

function DofMaps.get_dof_map(s::DistributedSnapshots)
  map(local_views(s)) do s
    get_dof_map(s)
  end
end

function ParamDataStructures.select_snapshots(s::DistributedSnapshots,pindex)
  snaps = map(local_views(s)) do s
    select_snapshots(s,pindex)
  end
  DistributedSnapshots(snaps)
end

GridapDistributed.local_views(s::DistributedSnapshots) = local_views(s.snaps)
PartitionedArrays.partition(s::DistributedSnapshots) = partition(s.snaps)
PartitionedArrays.local_values(s::DistributedSnapshots) = local_values(s.snaps)
PartitionedArrays.own_values(s::DistributedSnapshots) = own_values(s.snaps)
PartitionedArrays.ghost_values(s::DistributedSnapshots) = own_values(s.snaps)

# multi-field interface

const DistributedBlockSnapshots{N} = BlockSnapshots{<:DistributedSnapshots,N}

function ParamDataStructures.Snapshots(b::BlockPArray,i::AbstractArray,r::AbstractRealization)
  N = ndims(b)
  block_values = blocks(b)
  s = size(block_values)
  array = Array{DistributedSnapshots,N}(undef,s)
  touched = Array{Bool,N}(undef,s)
  for (j,dataj) in enumerate(block_values)
    if !iszero(dataj)
      array[j] = Snapshots(dataj,i[j],r)
      touched[j] = true
    else
      touched[j] = false
    end
  end
  BlockSnapshots(array,touched)
end

function Base.show(io::IO,k::MIME"text/plain",s::DistributedBlockSnapshots)
  vals = local_views(s)
  nparts = length(vals)
  n = length(s.snaps)
  map_main(vals) do s
    println(io,"Block snapshots of size $(size(s)), partitioned into $nparts parts")
  end
end

function PartitionedArrays.partition(a::DistributedBlockSnapshots)
  vals = map(partition,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function PartitionedArrays.local_values(a::DistributedBlockSnapshots)
  vals = map(local_values,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function PartitionedArrays.own_values(a::DistributedBlockSnapshots)
  vals = map(own_values,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end

function PartitionedArrays.ghost_values(a::DistributedBlockSnapshots)
  vals = map(ghost_values,blocks(a)) |> to_parray_of_arrays
  return map(mortar,vals)
end
