# these structs are only needed when dealing with a parametric version of a FE
# space of type FESpaceWithConstantFixed

"""
    struct ParamVectorWithEntryRemoved{T,A} <: ParamVector{T}
      a::A
      index::Int
    end
"""
struct ParamVectorWithEntryRemoved{T,A} <: ParamVector{T}
  a::A
  index::Int
  function ParamVectorWithEntryRemoved(a::A,index::Integer) where {T,A<:ConsecutiveParamVector{T}}
    @assert 1 <= index <= innerlength(a)
    new{T,A}(a,index)
  end
end

function Arrays.VectorWithEntryRemoved(a::AbstractParamVector,index::Int)
  ParamVectorWithEntryRemoved(a,index)
end

get_all_data(v::ParamVectorWithEntryRemoved) = MatrixWithRowRemoved(get_all_data(v.a),v.index)

param_length(v::ParamVectorWithEntryRemoved) = param_length(v.a)

Base.size(v::ParamVectorWithEntryRemoved) = size(v.a)

function Base.getindex(v::ParamVectorWithEntryRemoved,i::Integer)
  VectorWithEntryRemoved(v.a[i],v.index)
end

function Base.sum(v::ParamVectorWithEntryRemoved)
  data = get_all_data(v.a)
  sum(data,dims=1) - data[v.index,:]
end

"""
    struct ParamVectorWithEntryInserted{T,A} <: ParamVector{T}
      a::A
      index::Int
      value::Vector{T}
    end
"""
struct ParamVectorWithEntryInserted{T,A} <: ParamVector{T}
  a::A
  index::Int
  value::Vector{T}
  function ParamVectorWithEntryInserted(a::A,index::Integer,value::Vector{T}) where {T,A<:ConsecutiveParamVector{T}}
    @assert 1 <= index <= innerlength(a)+1
    new{T,A}(a,index,value)
  end
end

function Arrays.VectorWithEntryInserted(a::AbstractParamVector,index::Int,value::AbstractVector)
  ParamVectorWithEntryInserted(a,index,value)
end

get_all_data(v::ParamVectorWithEntryInserted) = MatrixWithRowInserted(get_all_data(v.a),v.index,v.value)

param_length(v::ParamVectorWithEntryInserted) = param_length(v.a)

Base.size(v::ParamVectorWithEntryInserted) = size(v.a)

function Base.getindex(v::ParamVectorWithEntryInserted,i::Integer)
  VectorWithEntryInserted(v.a[i],v.index,v.value[i])
end

function Base.sum(v::ParamVectorWithEntryInserted)
  data = get_all_data(v.a)
  sum(data,dims=1) + v.value
end

struct MatrixWithRowRemoved{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
  matrix::A
  index::Int
end

Base.size(a::MatrixWithRowRemoved) = (size(a.matrix,1)-1,size(a.matrix,2))

function Base.getindex(a::MatrixWithRowRemoved,i::Integer,j::Integer)
  i < a.index ? a.matrix[i,j] : a.matrix[i+1,j]
end

function Base.setindex!(a::MatrixWithRowRemoved,v,i::Integer,j::Integer)
  i < a.index ? (a.matrix[i,j]=v) : (a.matrix[i+1,j]=v)
end

struct MatrixWithRowInserted{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
  matrix::A
  index::Int
  value::Vector{T}
end

Base.size(a::MatrixWithRowInserted) = (size(a.matrix,1)+1,size(a.matrix,2))

function Base.getindex(a::MatrixWithRowInserted,i::Integer,j::Integer)
  i < a.index ? a.matrix[i,j] : (i==a.index ? a.value[j] : a.matrix[i-1,j])
end

function Base.setindex!(a::MatrixWithRowInserted,v,i::Integer,j::Integer)
  i < a.index ? (a.matrix[i,j]=v) : (i==a.index ? a.value[j] = v : a.matrix[i-1,j] = v)
end
