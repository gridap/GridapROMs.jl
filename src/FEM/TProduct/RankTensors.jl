@doc raw"""
    abstract type AbstractRankTensor{D,K} end

Type representing a tensor `a` of dimension `D` and rank `K`, i.e. assuming the form

  ``a = \sum\limits_{k=1}^K a_1^k \otimes \cdots \otimes a_D^k``

Subtypes:

- [`Rank1Tensor`](@ref)
- [`GenericRankTensor`](@ref)
"""
abstract type AbstractRankTensor{D,K} end

dimension(a::AbstractRankTensor{D,K}) where {D,K} = D
LinearAlgebra.rank(a::AbstractRankTensor{D,K}) where {D,K} = K
get_decomposition(a::AbstractRankTensor) = ntuple(k -> get_decomposition(a,k),Val{rank(a)}())

@doc raw"""
    get_decomposition(a::AbstractRankTensor,k::Integer) -> Vector{<:AbstractArray}

For a tensor `a` of dimension `D` and rank `K` assuming the form

  ``a = \sum\limits_{k=1}^K a_1^k \otimes \cdots \otimes a_D^k``

returns the decomposition relative to the `k`th rank:

``[a_1^k, \hdots , a_D^k]``
"""
get_decomposition(a::AbstractRankTensor,k::Integer) = @abstractmethod

@doc raw"""
    struct Rank1Tensor{D,A<:AbstractArray} <: AbstractRankTensor{D,1}
      factors::Vector{A}
    end

Structure representing rank-1 tensors, i.e. assuming the form

  ``a = a_1 \otimes \cdots \otimes a_D``
"""
struct Rank1Tensor{D,A<:AbstractArray} <: AbstractRankTensor{D,1}
  factors::Vector{A}
  function Rank1Tensor(factors::Vector{A}) where A
    D = length(factors)
    new{D,A}(factors)
  end
end

"""
    get_factors(a::Rank1Tensor) -> Vector

Returns the vector of D factor arrays of the rank-1 tensor `a`.
"""
get_factors(a::Rank1Tensor) = a.factors

"""
    get_factor(a::AbstractRankTensor,d::Integer,k::Integer) -> AbstractArray

Returns the `d`-th factor array of the `k`-th rank-1 decomposition of `a`.
For a [`Rank1Tensor`](@ref) `k` must equal 1.
"""
function get_factor(a::Rank1Tensor,d::Integer,k::Integer)
  @check k==1
  get_factors(a)[d]
end
get_decomposition(a::Rank1Tensor,k::Integer) = k == 1 ? a : error("Exceeded rank 1 with rank $k")
Base.size(a::Rank1Tensor) = (dimension(a),)
Base.getindex(a::Rank1Tensor,d::Integer) = get_factors(a)[d]
Base.setindex!(a::Rank1Tensor,v,d::Integer) = (get_factors(a)[d]=v)

function LinearAlgebra.cholesky(a::Rank1Tensor)
  cholesky.(get_factors(a))
end

@doc raw"""
    struct GenericRankTensor{D,K,A<:AbstractArray} <: AbstractRankTensor{D,K}
      decompositions::Vector{Rank1Tensor{D,A}}
    end

Structure representing a generic rank-K tensor, i.e. assuming the form

  ``a = \sum\limits_{k=1}^K a_1^k \otimes \cdots \otimes a_D^k``
"""
struct GenericRankTensor{D,K,A<:AbstractArray} <: AbstractRankTensor{D,K}
  decompositions::Vector{Rank1Tensor{D,A}}
  function GenericRankTensor(decompositions::Vector{Rank1Tensor{D,A}}) where {D,A}
    K = length(decompositions)
    new{D,K,A}(decompositions)
  end
end

get_decomposition(a::GenericRankTensor,k::Integer) = a.decompositions[k]
get_factor(a::GenericRankTensor,d::Integer,k::Integer) = get_factors(get_decomposition(a,k))[d]
Base.size(a::GenericRankTensor) = (rank(a),)
Base.getindex(a::GenericRankTensor,k::Integer) = get_decomposition(a,k)

function get_crossnorm(a::GenericRankTensor{D,K}) where {D,K}
  sd = 0
  dim = 1
  for d in 1:D
    factor = get_factor(a,d,1)
    if size(factor,1) > sd
      sd = size(factor,1)
      dim = d
    end
  end
  get_decomposition(a,dim)
end

# this is not true, but it is sufficient to correctly run the supremizing procedure
function LinearAlgebra.cholesky(a::GenericRankTensor{D,K}) where {D,K}
  cholesky(get_crossnorm(a))
end

"""
    struct BlockRankTensor{A<:AbstractRankTensor,N} <: AbstractArray{A,N}
      array::Array{A,N}
    end

Multi-field version of a [`AbstractRankTensor`](@ref)
"""
struct BlockRankTensor{A<:AbstractRankTensor,N} <: AbstractArray{A,N}
  array::Array{A,N}
end

Base.size(a::BlockRankTensor) = size(a.array)

function Base.getindex(
  a::BlockRankTensor{A,N},
  i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  getindex(a.array,i...)
end

function Base.setindex!(
  a::BlockRankTensor{A,N},
  v,i::Vararg{Integer,N}
  ) where {A,N}

  @boundscheck checkbounds(a.array,i...)
  setindex!(a.array,v,i...)
end

function Base.getindex(a::BlockRankTensor{A,N},i::Block{N}) where {A,N}
  getindex(a.array,i.n...)
end

# wrapper

"""
    const MatrixOrTensor = Union{AbstractMatrix,AbstractRankTensor}
"""
const MatrixOrTensor = Union{AbstractMatrix,AbstractRankTensor,BlockRankTensor}

# linear algebra

Base.:*(a::AbstractRankTensor,b::AbstractArray) = tpmul(a,b)
Base.:*(a::AbstractArray,b::AbstractRankTensor) = tpmul(a,b)

function tpmul(a::Rank1Tensor{2},b::AbstractMatrix)
  return a[1]*b*a[2]'
end

function tpmul(a::AbstractMatrix,b::Rank1Tensor{2})
  return b[1]*a*b[2]'
end

function tpmul(a::Rank1Tensor{3},b::AbstractArray{T,3} where T)
  hcat([vec(a[1]*bi*a[2]') for bi in eachslice(b,dims=3)]...)*a[3]'
end

function tpmul(a::AbstractRankTensor{D,K},b::AbstractArray) where {D,K}
  sum(map(k -> tpmul(get_decomposition(a,k),b),1:K))
end

function tpmul(a::AbstractArray,b::AbstractRankTensor{D,K}) where {D,K}
  sum(map(k -> tpmul(a,get_decomposition(b,k)),1:K))
end

function Utils.induced_norm(a::AbstractArray{T,D},X::AbstractRankTensor{D}) where {T,D}
  sqrtabs(dot(vec(a),vec(X*a)))
end

function Utils.induced_norm(a::AbstractArray{T,D′},X::AbstractRankTensor{D}) where {T,D,D′}
  D ≥ D′ && @notimplemented
  sqrtabs(sum(induced_norm(ai,X)^2 for ai in eachslice(a,dims=D′)))
end

# to global array - should try avoiding using these functions

function LinearAlgebra.kron(a::AbstractRankTensor{D,1}) where D
  kron(reverse(get_factors(a))...)
end

function LinearAlgebra.kron(a::AbstractRankTensor{D,K}) where {D,K}
  sum([kron(get_decomposition(a,k)) for k in 1:K])
end
