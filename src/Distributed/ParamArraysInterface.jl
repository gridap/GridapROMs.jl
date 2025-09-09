# param utils

for T in (:PVector,:PSparseMatrix,:BlockPArray)
  @eval begin
    function ParamDataStructures.param_length(a::$T)
      PartitionedArrays.getany(map(param_length,local_views(a)))
    end
  end
end

function ParamDataStructures.param_getindex(a::PVector,i::Integer)
  vector_partition,cache = map(a.vector_partition,a.cache) do values,cache
    param_getindex(values,i),param_getindex(cache,i)
  end |> tuple_of_arrays
  PVector(vector_partition,a.index_partition,cache)
end

function ParamDataStructures.param_getindex(a::PSparseMatrix,i::Integer)
  matrix_partition,cache = map(a.matrix_partition,a.cache) do values,cache
    param_getindex(values,i),param_getindex(cache,i)
  end |> tuple_of_arrays
  PSparseMatrix(matrix_partition,a.row_partition,a.col_partition,cache)
end

function ParamDataStructures.param_getindex(a::BlockPArray,i::Integer)
  b = map(blocks(a)) do a
    param_getindex(a,i)
  end
  BlockPArray(b,a.axes)
end

function ParamDataStructures.parameterize(a::PVector,plength::Integer)
  vector_partition,cache = map(a.vector_partition,a.cache) do values,cache
    parameterize(values,plength),parameterize(cache,plength)
  end |> tuple_of_arrays
  PVector(vector_partition,a.index_partition,cache)
end

function ParamDataStructures.parameterize(a::PSparseMatrix,plength::Integer)
  matrix_partition,cache = map(a.matrix_partition,a.cache) do values,cache
    parameterize(values,plength),parameterize(cache,plength)
  end |> tuple_of_arrays
  PSparseMatrix(matrix_partition,a.row_partition,a.col_partition,cache)
end

function ParamDataStructures.parameterize(a::BlockPArray,plength::Integer)
  b = map(blocks(a)) do a
    parameterize(a,plength)
  end
  BlockPArray(b,a.axes)
end

function PartitionedArrays.default_local_values(I,V::ConsecutiveParamVector{T},indices) where T
  data = zeros(T,local_length(indices),param_length(V))
  for k in 1:length(I)
    for l in param_eachindex(V)
      data[I[k],l] += V.data[k,l]
    end
  end
  ConsecutiveParamArray(data)
end

# FE space utils

function PartitionedArrays.own_values(a::ConsecutiveParamVector,indices)
  ov = view(a.data,own_to_local(indices),:)
  ConsecutiveParamArray(ov)
end

function PartitionedArrays.ghost_values(a::ConsecutiveParamVector,indices)
  gv = view(a.data,ghost_to_local(indices),:)
  ConsecutiveParamArray(gv)
end

function PartitionedArrays.own_values(a::AbstractMatrix{<:Number},indices)
  view(a,own_to_local(indices),:)
end

function PartitionedArrays.ghost_values(a::AbstractMatrix{<:Number},indices)
  view(a,ghost_to_local(indices),:)
end

# caches

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

function ParamDataStructures.param_getindex(a::ParamVectorAssemblyCache,i::Integer)
  PartitionedArrays.VectorAssemblyCache(
    a.neighbors_snd,
    a.neighbors_rcv,
    a.local_indices_snd,
    a.local_indices_rcv,
    param_getindex(a.buffer_snd,i),
    param_getindex(a.buffer_rcv,i)
  )
end

function ParamDataStructures.parameterize(
  a::PartitionedArrays.VectorAssemblyCache,plength::Integer
  )
  ParamVectorAssemblyCache(
    a.neighbors_snd,
    a.neighbors_rcv,
    a.local_indices_snd,
    a.local_indices_rcv,
    parameterize(a.buffer_snd,plength),
    parameterize(a.buffer_rcv,plength)
  )
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

function ParamDataStructures.param_getindex(a::ParamSparseMatrixAssemblyCache,i::Integer)
  PartitionedArrays.SparseMatrixAssemblyCache(param_getindex(a.cache,i))
end

function ParamDataStructures.parameterize(
  a::PartitionedArrays.SparseMatrixAssemblyCache,plength::Integer
  )

  function _parameterize(a::JaggedArray,plength)
    pdata = parameterize(a.data,plength)
    ParamJaggedArray(get_all_data(pdata),a.ptrs)
  end

  cache = ParamVectorAssemblyCache(
    a.neighbors_snd,
    a.neighbors_rcv,
    a.local_indices_snd,
    a.local_indices_rcv,
    _parameterize(a.buffer_snd,plength),
    _parameterize(a.buffer_rcv,plength)
  )
  ParamSparseMatrixAssemblyCache(cache)
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

function ParamDataStructures.param_getindex(a::ParamJaggedArrayAssemblyCache,i::Integer)
  PartitionedArrays.JaggedArrayAssemblyCache(param_getindex(a.cache,i))
end

function ParamDataStructures.parameterize(
  a::PartitionedArrays.JaggedArrayAssemblyCache,plength::Integer
  )
  ParamJaggedArrayAssemblyCache(parameterize(a.cache,plength))
end

# remove the whole function when fixing the issue inside
function PartitionedArrays.p_sparse_matrix_cache_impl(
  ::Type{<:ParamSparseMatrix},matrix_partition,row_partition,col_partition
  )

  function setup_snd(part,parts_snd,row_indices,col_indices,values)
    local_row_to_owner = local_to_owner(row_indices)
    local_to_global_row = local_to_global(row_indices)
    local_to_global_col = local_to_global(col_indices)
    owner_to_i = Dict(( owner=>i for (i,owner) in enumerate(parts_snd) ))
    ptrs = zeros(Int32,length(parts_snd)+1)
    for (li,lj,v) in nziterator(values)
      owner = local_row_to_owner[li]
      if owner != part
        ptrs[owner_to_i[owner]+1] +=1
      end
    end
    length_to_ptrs!(ptrs)
    k_snd_data = zeros(Int32,ptrs[end]-1)
    gi_snd_data = zeros(Int,ptrs[end]-1)
    gj_snd_data = zeros(Int,ptrs[end]-1)
    for (k,(li,lj,v)) in enumerate(nziterator(values))
      owner = local_row_to_owner[li]
      if owner != part
        p = ptrs[owner_to_i[owner]]
        k_snd_data[p] = k
        gi_snd_data[p] = local_to_global_row[li]
        gj_snd_data[p] = local_to_global_col[lj]
        ptrs[owner_to_i[owner]] += 1
      end
    end
    rewind_ptrs!(ptrs)
    k_snd = JaggedArray(k_snd_data,ptrs)
    gi_snd = JaggedArray(gi_snd_data,ptrs)
    gj_snd = JaggedArray(gj_snd_data,ptrs)
    k_snd,gi_snd,gj_snd
  end
  function setup_rcv(part,row_indices,col_indices,gi_rcv,gj_rcv,values)
    global_to_local_row = global_to_local(row_indices)
    global_to_local_col = global_to_local(col_indices)
    ptrs = gi_rcv.ptrs
    k_rcv_data = zeros(Int32,ptrs[end]-1)
    for p in 1:length(gi_rcv.data)
      gi = gi_rcv.data[p]
      gj = gj_rcv.data[p]
      li = global_to_local_row[gi]
      lj = global_to_local_col[gj]
      k = nzindex(values,li,lj)
      @boundscheck @assert k > 0 "The sparsity pattern of the ghost layer is inconsistent"
      k_rcv_data[p] = k
    end
    k_rcv = JaggedArray(k_rcv_data,ptrs)
    k_rcv
  end
  part = linear_indices(row_partition)
  parts_snd,parts_rcv = PartitionedArrays.assembly_neighbors(row_partition)
  k_snd,gi_snd,gj_snd = map(setup_snd,part,parts_snd,row_partition,col_partition,matrix_partition) |> tuple_of_arrays
  graph = ExchangeGraph(parts_snd,parts_rcv)
  gi_rcv = PartitionedArrays.exchange_fetch(gi_snd,graph)
  gj_rcv = PartitionedArrays.exchange_fetch(gj_snd,graph)
  k_rcv = map(setup_rcv,part,row_partition,col_partition,gi_rcv,gj_rcv,matrix_partition)
  buffers = map(PartitionedArrays.assembly_buffers,matrix_partition,k_snd,k_rcv) |> tuple_of_arrays
  cache = map(PartitionedArrays.VectorAssemblyCache,parts_snd,parts_rcv,k_snd,k_rcv,buffers...)
  # TODO why is this wrong???
  # map(PartitionedArrays.SparseMatrixAssemblyCache,cache)
  map(cache) do c
    PartitionedArrays.SparseMatrixAssemblyCache(c)
  end
end
