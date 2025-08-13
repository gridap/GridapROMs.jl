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

@inline function Base.iterate(a::PartitionedArrays.NZIteratorCSC{<:ConsecutiveParamSparseMatrixCSC})
  if nnz(a.matrix) == 0
    return nothing
  end
  col = 0
  knext = nothing
  while knext === nothing
    col += 1
    ks = nzrange(a.matrix,col)
    knext = iterate(ks)
  end
  k,kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = view(get_all_data(a.matrix),k,:)
  (i,j,v),(col,kstate)
end

@inline function Base.iterate(a::PartitionedArrays.NZIteratorCSC{<:ConsecutiveParamSparseMatrixCSC},state)
  col,kstate = state
  ks = nzrange(a.matrix,col)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if col == size(a.matrix,2)
        return nothing
      end
      col += 1
      ks = nzrange(a.matrix,col)
      knext = iterate(ks)
    end
  end
  k,kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = view(get_all_data(a.matrix),k,:)
  (i,j,v),(col,kstate)
end

@inline function Base.iterate(a::PartitionedArrays.NZIteratorCSR{<:ConsecutiveParamSparseMatrixCSR})
  if nnz(a.matrix) == 0
    return nothing
  end
  row = 0
  ptrs = a.matrix.rowptr
  knext = nothing
  while knext === nothing
    row += 1
    ks = nzrange(a.matrix,row)
    knext = iterate(ks)
  end
  k,kstate = knext
  i = row
  j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
  v = view(get_all_data(a.matrix),k,:)
  (i,j,v),(row,kstate)
end

@inline function Base.iterate(a::PartitionedArrays.NZIteratorCSR{<:ConsecutiveParamSparseMatrixCSR},state)
  row,kstate = state
  ks = nzrange(a.matrix,row)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if row == size(a.matrix,1)
        return nothing
      end
      row += 1
      ks = nzrange(a.matrix,row)
      knext = iterate(ks)
    end
  end
  k,kstate = knext
  i = row
  j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
  v = view(get_all_data(a.matrix),k,:)
  (i,j,v),(row,kstate)
end

function PartitionedArrays.nziterator(a::ConsecutiveParamSparseMatrixCSR)
  PartitionedArrays.NZIteratorCSR(a)
end

function PartitionedArrays.nziterator(a::ConsecutiveParamSparseMatrixCSC)
  PartitionedArrays.NZIteratorCSC(a)
end

function PartitionedArrays.nzindex(a::ParamSparseMatrix,args...)
  PartitionedArrays.nzindex(testitem(a),args...)
end

function PartitionedArrays.compresscoo(
  ::Type{<:ConsecutiveParamSparseMatrixCSC},
  I::AbstractVector,
  J::AbstractVector,
  M::AbstractMatrix,
  m::Integer,
  n::Integer
  )

  combine = +
  sparse(I,J,M,m,n,combine)
end

struct ParamSubSparseMatrix{T,A,B,C} <: AbstractParamArray{T,2,PartitionedArrays.SubSparseMatrix{T,A,B,C}}
  data::A
  indices::B
  inv_indices::C
  function ParamSubSparseMatrix(
    data::ParamSparseMatrix{Tv,Ti},
    indices::Tuple,
    inv_indices::Tuple
    ) where {Tv,Ti}

    A = typeof(data)
    B = typeof(indices)
    C = typeof(inv_indices)
    new{Tv,A,B,C}(data,indices,inv_indices)
  end
end

function PartitionedArrays.SubSparseMatrix(data::ParamSparseMatrix,indices::Tuple,inv_indices::Tuple)
  ParamSubSparseMatrix(data,indices,inv_indices)
end

ParamDataStructures.param_length(A::ParamSubSparseMatrix) = param_length(A.data)
innersize(a::ParamSubSparseMatrix) = map(length,a.indices)

Base.@propagate_inbounds function Base.getindex(A::ParamSubSparseMatrix{T},i::Integer,j::Integer) where T
  @boundscheck checkbounds(A,i...)
  if i == j
    PartitionedArrays.SubSparseMatrix(param_getindex(A.data,i),A.indices,A.inv_indices)
  else
    fill(zero(T),innersize(A))
  end
end

const ParamLocalView{T<:AbstractArray,N,A<:AbstractParamArray,B} = GridapDistributed.LocalView{T,N,A,B}

ParamDataStructures.param_length(A::ParamLocalView) = param_length(A.plids_to_value)
ParamDataStructures.get_all_data(A::ParamLocalView) = get_all_data(A.plids_to_value)

@inline function Algebra.add_entry!(combine::Function,A::ParamLocalView,v::Number,i)
  data = get_all_data(A)
  @inbounds for k = param_eachindex(A)
    aik = data[i,k]
    data[i,k] = combine(aik,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ParamLocalView,v::AbstractVector,i)
  data = get_all_data(A)
  @inbounds for k = param_eachindex(A)
    aik = data[i,k]
    vk = v[k]
    data[i,k] = combine(aik,vk)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ParamLocalView,v::Number,i,j)
  l = nz_index(A,i,j)
  nz = get_all_data(nonzeros(A))
  @inbounds for k = param_eachindex(A)
    aijk = nz[l,k]
    nz[l,k] = combine(aijk,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ParamLocalView,v::AbstractVector,i,j)
  l = nz_index(A,i,j)
  nz = get_all_data(nonzeros(A))
  @inbounds for k = param_eachindex(A)
    aijk = nz[l,k]
    vk = v[k]
    nz[l,k] = combine(aijk,vk)
  end
  A
end
