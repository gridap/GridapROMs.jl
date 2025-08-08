struct ParamVectorAssemblyCache{T,A}
  neighbors_snd::Vector{Int32}
  neighbors_rcv::Vector{Int32}
  local_indices_snd::JaggedArray{Int32,Int32}
  local_indices_rcv::JaggedArray{Int32,Int32}
  buffer_snd::ParamJaggedArray{T,Int32,A}
  buffer_rcv::ParamJaggedArray{T,Int32,A}
end

function PartitionedArrays.VectorAssemblyCache(
  neighbors_snd::AbstractVector,
  neighbors_rcv::AbstractVector,
  local_indices_snd::JaggedArray,
  local_indices_rcv::JaggedArray,
  buffer_snd::ParamJaggedArray,
  buffer_rcv::ParamJaggedArray
  )

  ParamVectorAssemblyCache(
    neighbors_snd,
    neighbors_rcv,
    local_indices_snd,
    local_indices_rcv,
    buffer_snd,
    buffer_rcv
    )
end

function Base.reverse(a::ParamVectorAssemblyCache)
  ParamVectorAssemblyCache(
    a.neighbors_rcv,
    a.neighbors_snd,
    a.local_indices_rcv,
    a.local_indices_snd,
    a.buffer_rcv,
    a.buffer_snd)
end

function PartitionedArrays.copy_cache(a::ParamVectorAssemblyCache)
  buffer_snd = JaggedArray(copy(a.buffer_snd.data),a.buffer_snd.ptrs)
  buffer_rcv = JaggedArray(copy(a.buffer_rcv.data),a.buffer_rcv.ptrs)
  ParamVectorAssemblyCache(
    a.neighbors_snd,
    a.neighbors_rcv,
    a.local_indices_snd,
    a.local_indices_rcv,
    buffer_snd,
    buffer_rcv)
end

struct ParamSparseMatrixAssemblyCache
  cache::ParamVectorAssemblyCache
end

function PartitionedArrays.SparseMatrixAssemblyCache(cache::ParamVectorAssemblyCache)
  ParamSparseMatrixAssemblyCache(cache)
end

Base.reverse(a::ParamSparseMatrixAssemblyCache) = ParamSparseMatrixAssemblyCache(reverse(a.cache))

function PartitionedArrays.copy_cache(a::ParamSparseMatrixAssemblyCache)
  ParamSparseMatrixAssemblyCache(copy_cache(a.cache))
end

struct ParamJaggedArrayAssemblyCache{T}
  cache::ParamVectorAssemblyCache{T}
end

function PartitionedArrays.JaggedArrayAssemblyCache(cache::ParamVectorAssemblyCache)
  ParamJaggedArrayAssemblyCache(cache)
end

Base.reverse(a::ParamJaggedArrayAssemblyCache) = ParamJaggedArrayAssemblyCache(reverse(a.cache))

function PartitionedArrays.copy_cache(a::ParamJaggedArrayAssemblyCache)
  ParamJaggedArrayAssemblyCache(copy_cache(a.cache))
end
