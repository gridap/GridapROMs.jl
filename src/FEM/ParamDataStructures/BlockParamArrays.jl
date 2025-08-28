ParamArray(A::AbstractArray{<:AbstractParamArray}) = mortar(A)
ParamArray(a::BlockArray,l::Integer) = mortar(map(b -> ParamArray(b,l),blocks(a)))

GenericParamArray(A::AbstractArray{<:AbstractParamArray}) = mortar(A)

for f in (:ParamArray,:GenericParamArray)
  @eval begin
    function $f(A::AbstractArray{<:BlockArray})
      map(1:blocklength(first(A))) do n
        $f(map(a -> blocks(a)[n],A))
      end |> mortar
    end
  end
end

"""
    struct BlockParamArray{T,N,A<:AbstractArray{<:AbstractParamArray{T,N},N},B<:NTuple{N,AbstractUnitRange{Int}}} <: ParamArray{T,N}
      data::A
      axes::B
    end

Is to a [`ParamArray`](@ref) as a `BlockArray` is to a regular `AbstractArray`
"""
struct BlockParamArray{T,N,A<:AbstractArray{<:AbstractParamArray{T,N},N},B<:NTuple{N,AbstractUnitRange{Int}}} <: ParamArray{T,N}
  data::A
  axes::B
end

"""
    const BlockParamVector{T,A,B} = BlockParamArray{T,1,A,B}
"""
const BlockParamVector{T,A,B} = BlockParamArray{T,1,A,B}

"""
    const BlockParamMatrix{T,A,B} = BlockParamArray{T,2,A,B}
"""
const BlockParamMatrix{T,A,B} = BlockParamArray{T,2,A,B}

"""
    const BlockConsecutiveParamArray{T,N,A<:Vector{<:ConsecutiveParamArray{T,N}},B} = BlockParamArray{T,N,A,B}
"""
const BlockConsecutiveParamArray{T,N,A<:Vector{<:ConsecutiveParamArray{T,N}},B} = BlockParamArray{T,N,A,B}

"""
    const BlockConsecutiveParamVector{T,A<:Vector{<:ConsecutiveParamVector{T}},B} = BlockParamVector{T,A,B}
"""
const BlockConsecutiveParamVector{T,A<:Vector{<:ConsecutiveParamVector{T}},B} = BlockParamVector{T,A,B}

"""
    const BlockConsecutiveParamMatrix{T,A<:Matrix{<:ConsecutiveParamMatrix{T}},B} = BlockParamMatrix{T,A,B}
"""
const BlockConsecutiveParamMatrix{T,A<:Matrix{<:ConsecutiveParamMatrix{T}},B} = BlockParamMatrix{T,A,B}

function BlockArrays._BlockArray(data::AbstractArray{<:AbstractParamArray,N},axes::NTuple{N,AbstractUnitRange{Int}}) where N
  @assert all(param_length(d)==param_length(first(data)) for d in data)
  BlockParamArray(data,axes)
end

param_length(A::BlockParamArray) = param_length(first(blocks(A)))

@inline Base.size(A::BlockParamArray) = map(length,axes(A))
Base.axes(A::BlockParamArray) = A.axes

@inline function innersize(A::BlockParamArray)
  map(innersize,A.data)
end

@inline function inneraxes(A::BlockParamArray{T,N}) where {T,N}
  is = innersize(A)
  _inneraxes(is)
end

_inneraxes(i) = @abstractmethod

@inline function _inneraxes(i::AbstractVector{Tuple{Int}})
  (blockedrange(getindex.(i,1)),)
end

@inline function _inneraxes(i::AbstractMatrix{Tuple{Int,Int}})
  diagi = diag(i)
  ntuple(j->blockedrange(getindex.(diagi,j)),Val{length(diagi)}())
end

BlockArrays.blocks(A::BlockParamArray) = A.data

function param_getindex(A::BlockParamArray,i::Integer)
  mortar(map(a->param_getindex(a,i),blocks(A)))
end

function param_setindex!(A::BlockParamArray,v::BlockArray,i::Integer)
  map((a,b)->param_setindex!(a,b,i),blocks(A),blocks(v))
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  v = A[findblockindex.(axes(A),i)...]
  return v
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamArray{T,N},i::BlockIndex{N}) where {T,N}
  BlockArrays._blockindex_getindex(A,i)
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamVector{T},i::BlockIndex{1}) where {T}
  BlockArrays._blockindex_getindex(A,i)
end

@inline function BlockArrays._blockindex_getindex(A::BlockParamArray{T,N},i::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(A,Block(i.I))
  @inbounds bl = A.data[i.I...]
  @boundscheck checkbounds(bl,i.α...)
  @inbounds v = bl[i.α...]
  return v
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamArray{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  @inbounds A[findblockindex.(axes(A),i)...] = v
  return A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamArray{T,N},v,i::Vararg{BlockIndex{1},N}) where {T,N}
  _blockindex_setindex!(A,v,BlockIndex(i))
  A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamArray{T,N},v,i::BlockIndex{N}) where {T,N}
  _blockindex_setindex!(A,v,i)
  A
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamVector{T},v,i::BlockIndex{1}) where {T}
  _blockindex_setindex!(A,v,i)
  A
end

@inline function _blockindex_setindex!(A::BlockParamArray{T,N},v,i::BlockIndex{N}) where {T,N}
  @boundscheck blockcheckbounds(A,Block(i.I))
  @inbounds bl = A.data[i.I...]
  @boundscheck checkbounds(bl,i.α...)
  @inbounds bl[i.α...] = v
  return A
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamVector{T},i::Block{1}) where T
  @boundscheck blockcheckbounds(A,i)
  getindex(A.data,i.n...)
end

Base.@propagate_inbounds function Base.getindex(A::BlockParamArray{T,N},i::Block{N}) where {T,N}
  @boundscheck blockcheckbounds(A,i)
  getindex(A.data,i.n...)
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamVector{T},v,i::Block{1}) where T
  @boundscheck blockcheckbounds(A,i)
  setindex!(A.data,v,i.n...)
end

Base.@propagate_inbounds function Base.setindex!(A::BlockParamArray{T,N},v,i::Block{N}) where {T,N}
  @boundscheck blockcheckbounds(A,i)
  setindex!(A.data,v,i.n...)
end

function Base.similar(A::BlockParamArray)
  BlockParamArray(map(similar,blocks(A)),A.axes)
end

function Base.similar(A::BlockParamArray{T,N},::Type{S},axes::Vararg{BlockArrays.AbstractBlockedUnitRange}) where {T,T′,N,S<:AbstractArray{T′,N}}
  A′ = map(eachindex(blocks(A))) do i
    ai = blocks(A)[i]
    axi = map(ax -> blocks(ax)[i],axes)
    si = length.(axi)
    similar(ai,S,si)
  end
  BlockParamArray(A′,A.axes)
end

function Base.copy(A::BlockParamArray)
  A′ = map(copy,blocks(A))
  BlockParamArray(A′,A.axes)
end

function Base.copyto!(A::BlockParamArray,B::BlockParamArray)
  @check size(A) == size(B) && innersize(A) == innersize(B)
  map(copyto!,A.data,B.data)
  A
end

function param_cat(A::Vector{<:BlockParamArray})
  a = first(A)
  @check all(blocklength(ai)==blocklength(a) for ai in A)
  B = map(1:blocklength(a)) do i
    param_cat(map(a -> blocks(a)[i],A))
  end
  mortar(B)
end

for op in (:+,:-)
  @eval begin
    function ($op)(A::BlockParamArray,B::BlockParamArray)
      @check size(A) == size(B)
      AB = map($op,blocks(A),blocks(B))
      BlockParamArray(AB,A.axes)
    end
  end
end

for op in (:*,:/)
  @eval begin
    function ($op)(A::BlockParamArray,b::Number)
      Ab = map(a -> $op(a,b),blocks(A))
      BlockParamArray(AB,A.axes)
    end
  end
end

for f in (:(Base.fill!),:(LinearAlgebra.fillstored!))
  @eval begin
    function $f(A::BlockParamArray,b::Number)
      map(a -> $f(a,b),blocks(A))
      return A
    end
  end
end

function LinearAlgebra.rmul!(A::BlockParamArray,b::Number)
  map(a -> rmul!(a,b),blocks(A))
  return A
end

function LinearAlgebra.axpy!(α::Number,A::BlockParamArray,B::BlockParamArray)
  for (a,b) in zip(blocks(A),blocks(B))
    (iszero(α) || iszero(a)) && continue
    axpy!(α,a,b)
  end
  return B
end

function LinearAlgebra.norm(A::BlockParamArray)
  n = zeros(param_length(A))
  for b in blocks(A)
    n .+= norm(b).^2
  end
  return sqrt.(n)
end

function get_param_entry!(v::BlockVector,A::BlockParamArray,i...)
  map((v,a) -> get_param_entry!(v,a,i...),blocks(v),blocks(A))
end

function get_param_entry(A::BlockParamArray,i...)
  entries = map(a -> get_param_entry(a,i...),blocks(A))
  mortar(entries)
end
