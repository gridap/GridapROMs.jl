"""
    abstract type AbstractRankTensor{D,K} end

Type representing a tensor `a` of dimension `D` and rank `K`, i.e. assuming the form

  `a = ∑_{k=1}^{K} a_1^k ⊗ ⋯ ⊗ a_D^k`

Subtypes:

- [`Rank1Tensor`](@ref)
- [`GenericRankTensor`](@ref)
"""
abstract type AbstractRankTensor{D,K} end

dimension(a::AbstractRankTensor{D,K}) where {D,K} = D
LinearAlgebra.rank(a::AbstractRankTensor{D,K}) where {D,K} = K
get_decomposition(a::AbstractRankTensor) = ntuple(k -> get_decomposition(a,k),Val{rank(a)}())

"""
    get_decomposition(a::AbstractRankTensor,k::Integer) -> Vector{<:AbstractArray}

For a tensor `a` of dimension `D` and rank `K` assuming the form

  `a = ∑_{k=1}^{K} a_1^k ⊗ ⋯ ⊗ a_D^k`

returns the decomposition relative to the `k`th rank `[a_1^k, ⋯, a_D^k]`
"""
get_decomposition(a::AbstractRankTensor,k::Integer) = @abstractmethod

"""
    struct Rank1Tensor{D,A<:AbstractArray} <: AbstractRankTensor{D,1}
      factors::Vector{A}
    end

Structure representing rank-1 tensors, i.e. assuming the form

  `a = a_1 ⊗ ⋯ ⊗ a_D`
"""
struct Rank1Tensor{D,A<:AbstractArray} <: AbstractRankTensor{D,1}
  factors::Vector{A}
  function Rank1Tensor(factors::Vector{A}) where A
    D = length(factors)
    new{D,A}(factors)
  end
end

get_factors(a::Rank1Tensor) = a.factors
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

"""
    struct GenericRankTensor{D,K,A<:AbstractArray} <: AbstractRankTensor{D,K}
      decompositions::Vector{Rank1Tensor{D,A}}
    end

Structure representing a generic rank-K tensor, i.e. assuming the form

  `a = ∑_{k=1}^{K} a_1^k ⊗ ⋯ ⊗ a_D^k`
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
  factors = get_factors(get_decomposition(a,1))
  for k in 2:K
    factors_k = get_decomposition(a,k)
    for d in 1:D
      factors[d] += factors_k[d]
    end
  end
  Rank1Tensor(factors)
end

# this is not true, but it is sufficient to correctly run the supremizing procedure
function LinearAlgebra.cholesky(a::GenericRankTensor{D,K}) where {D,K}
  cholesky(get_crossnorm(a))
end

"""
    tproduct_array(arrays_1d::Vector{<:AbstractArray}) -> Rank1Tensor
    tproduct_array(op,arrays_1d::Vector{<:AbstractArray},gradients_1d::Vector{<:AbstractArray},args...) -> GenericRankTensor

Returns a [`AbstractRankTensor`](@ref) storing the arrays `arrays_1d` (usually matrices)
arising from an integration routine on D 1-d triangulations whose tensor product
gives a D-dimensional triangulation. In the absence of the field `gradients_1d`,
the output is a [`Rank1Tensor`](@ref); when provided, the output is a [`GenericRankTensor`](@ref)

    tproduct_array(arrays_1d::Vector{<:BlockArray}) -> BlockRankTensor
    tproduct_array(op,arrays_1d::Vector{<:BlockArray},gradients_1d::Vector{<:BlockArray},args...) -> BlockRankTensor

Generalization of the previous functions to multi-field scenarios
"""
function tproduct_array(arrays_1d::Vector{<:AbstractArray})
  Rank1Tensor(arrays_1d)
end

function tproduct_array(
  ::Nothing,
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  summation=nothing)

  tproduct_array(arrays_1d)
end

function tproduct_array(
  ::typeof(gradient),
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  summation=nothing)

  decompositions = _find_decompositions(summation,arrays_1d,gradients_1d)
  GenericRankTensor(decompositions)
end

function tproduct_array(
  ::PartialDerivative{N},
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  args...
  ) where N

  arrays_1d[N] = gradients_1d[N]
  Rank1Tensor(arrays_1d)
end

function tproduct_array(
  ::Utils.Divergence,
  arrays_1d::Vector{<:AbstractArray},
  gradients_1d::Vector{<:AbstractArray},
  args...
  )

  decompositions = _find_decompositions(nothing,arrays_1d,gradients_1d)
  GenericRankTensor(decompositions)
end

function _find_decompositions(::Nothing,arrays_1d,gradients_1d)
  @check length(arrays_1d) == length(gradients_1d)
  inds = LinearIndices(arrays_1d)
  d = map(inds) do i
    di = copy(arrays_1d)
    di[i] = gradients_1d[i]
    return Rank1Tensor(di)
  end
  return d
end

function _find_decompositions(::typeof(+),arrays_1d,gradients_1d)
  @check length(arrays_1d) == length(gradients_1d)
  inds = LinearIndices(arrays_1d)
  d = map(inds) do i
    di = copy(arrays_1d)
    di[i] += gradients_1d[i]
    return Rank1Tensor(di)
  end
  return d
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

function tproduct_array(arrays_1d::Vector{<:BlockArray})
  s_blocks = blocksize(first(arrays_1d))
  arrays = map(CartesianIndices(s_blocks)) do i
    iblock = Block(Tuple(i))
    arrays_1d_i = getindex.(arrays_1d,iblock)
    tproduct_array(arrays_1d_i)
  end
  BlockRankTensor(arrays)
end

function tproduct_array(op::ArrayBlock,arrays_1d::Vector{<:BlockArray},gradients_1d::Vector{<:BlockArray},s::ArrayBlock)
  s_blocks = blocksize(first(arrays_1d))
  arrays = map(CartesianIndices(s_blocks)) do i
    iblock = Block(Tuple(i))
    arrays_1d_i = getindex.(arrays_1d,iblock)
    gradients_1d_i = getindex.(gradients_1d,iblock)
    op_i = op[Tuple(i)...]
    s_i = s[Tuple(i)...]
    tproduct_array(op_i,arrays_1d_i,gradients_1d_i,s_i)
  end
  BlockRankTensor(arrays)
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
  sqrt(dot(vec(a),vec(X*a)))
end

function Utils.induced_norm(a::AbstractArray{T,D′},X::AbstractRankTensor{D}) where {T,D,D′}
  D ≥ D′ && @notimplemented
  sqrt(sum(induced_norm(ai,X)^2 for ai in eachslice(a,dims=D′)))
end

# to global array - should try avoiding using these functions

function LinearAlgebra.kron(a::AbstractRankTensor{D,1}) where D
  kron(reverse(get_factors(a))...)
end

function LinearAlgebra.kron(a::AbstractRankTensor{D,K}) where {D,K}
  sum([kron(get_decomposition(a,k)) for k in 1:K])
end
