"""
    abstract type AbstractParamArray{T,N,A<:AbstractArray{T,N}} <: AbstractParamData{A,N} end

Type representing parametric abstract arrays of type A.
Subtypes:
- [`ParamArray`](@ref)
- [`ParamSparseMatrix`](@ref)
"""
abstract type AbstractParamArray{T,N,A<:AbstractArray{T,N}} <: AbstractParamData{A,N} end

"""
    const AbstractParamVector{T} = AbstractParamArray{T,1,<:AbstractVector{T}}
"""
const AbstractParamVector{T} = AbstractParamArray{T,1,<:AbstractVector{T}}

"""
    const AbstractParamMatrix{T} = AbstractParamArray{T,2,<:AbstractMatrix{T}}
"""
const AbstractParamMatrix{T} = AbstractParamArray{T,2,<:AbstractMatrix{T}}

"""
    const AbstractParamArray3D{T} = AbstractParamArray{T,3,<:AbstractArray{T,3}}
"""
const AbstractParamArray3D{T} = AbstractParamArray{T,3,<:AbstractArray{T,3}}

"""
    abstract type ParamArray{T,N} <: AbstractParamArray{T,N,Array{T,N}} end

Type representing parametric arrays of type A.
Subtypes:
- [`TrivialParamArray`](@ref)
- [`ConsecutiveParamArray`](@ref)
- [`GenericParamVector`](@ref)
- [`GenericParamMatrix`](@ref)
- [`BlockParamArray`](@ref)

Also acts as a constructor according to the following rules:
- ParamArray(A::AbstractArray{<:Number}) -> ParamNumber
- ParamArray(A::AbstractArray{<:Number},plength::Int) -> TrivialParamArray
- ParamArray(A::AbstractVector{<:AbstractArray}) -> ParamArray
- ParamArray(A::AbstractVector{<:AbstractSparseMatrix}) -> ParamSparseMatrix
- ParamArray(A::AbstractArray{<:ParamArray}) -> BlockParamArray
"""
abstract type ParamArray{T,N} <: AbstractParamArray{T,N,Array{T,N}} end

"""
    const ParamVector{T} = ParamArray{T,1}
"""
const ParamVector{T} = ParamArray{T,1}

"""
    const ParamMatrix{T} = ParamArray{T,2}
"""
const ParamMatrix{T} = ParamArray{T,2}

"""
    abstract type ParamSparseMatrix{Tv,Ti,A<:AbstractSparseMatrix{Tv,Ti}
      } <: AbstractParamArray{Tv,2,A} end

Type representing parametric abstract sparse matrices of type A.
Subtypes:
- [`ParamSparseMatrixCSC`](@ref)
- [`ParamSparseMatrixCSR`](@ref)
"""
abstract type ParamSparseMatrix{Tv,Ti,A<:AbstractSparseMatrix{Tv,Ti}} <: AbstractParamArray{Tv,2,A} end

ParamArray(args...) = @abstractmethod
ParamArray(A::AbstractArray{<:Number}) = ParamNumber(A)
ParamArray(A::AbstractParamArray) = A

GenericParamArray(args...) = @abstractmethod
GenericParamArray(A::AbstractArray{<:Number}) = ParamNumber(A)
GenericParamArray(A::AbstractParamArray) = A

"""
    global_parameterize(a,plength::Integer) -> AbstractParamArray

Returns a [`AbstractParamArray`](@ref) with parametric length `plength` from `a`.
This parameterization involves quantities defined at the global (or assembled) level.
For local parameterizations, see the function [`local_parameterize`](@ref)
"""
global_parameterize(args...) = ParamArray(args...)

param_getindex(A::AbstractParamArray{T,N},i::Integer) where {T,N} = getindex(A,tfill(i,Val{N}())...)
param_setindex!(A::AbstractParamArray{T,N},v,i::Integer) where {T,N} = setindex!(A,v,tfill(i,Val{N}())...)

"""
    innerlength(A::AbstractParamArray) -> Int

Returns the length of `A` for a single parameter. Thus, the total entries of `A`
is equals to `param_length(A)*innerlength(A)`
"""
innerlength(A::AbstractParamArray) = prod(innersize(A))

"""
    inneraxes(A::AbstractParamArray) -> Tuple{Vararg{Base.OneTo}}

Returns the axes of `A` for a single parameter
"""
inneraxes(A::AbstractParamArray) = Base.OneTo.(innersize(A))

"""
    get_param_entry(A::AbstractParamArray{T},i...) where T -> Vector{eltype(T)}

Returns a vector of the entries of `A` at index `i`, for every parameter. The
length of the output is equals to `param_length(A)`
"""
get_param_entry(A::AbstractParamArray,i...) = @abstractmethod

# small hack, zero(::Type{<:AbstractArray}) is not implemented in Base
function Base.zero(::Type{<:AbstractArray{T,N}}) where {T<:Number,N}
  zeros(T,tfill(1,Val{N}()))
end
# small hack, one(::Type{<:AbstractArray}) is not implemented in Base
function Base.one(::Type{<:AbstractArray{T,N}}) where {T<:Number,N}
  ones(T,tfill(1,Val{N}()))
end

# small hack, we shouldn't be able to fill an abstract array with a non-scalar
for f in (:(Base.fill!),:(LinearAlgebra.fillstored!))
  @eval begin
    function $f(A::AbstractParamArray{T,N},z::AbstractArray{<:Number,N}) where {T,N}
      @check all(z.==first(z))
      $f(A,first(z))
      return A
    end
  end
end

function (*)(A::AbstractParamMatrix,x::AbstractParamVector)
  TS = LinearAlgebra.promote_op(LinearAlgebra.matprod,eltype(A),eltype(x))
  mul!(similar(x,TS,inneraxes(A)[1]),A,x)
end

function (*)(A::AbstractParamMatrix,B::AbstractParamMatrix)
  TS = LinearAlgebra.promote_op(LinearAlgebra.matprod,eltype(A),eltype(B))
  mul!(similar(B,TS,(innersize(A)[1],innersize(B)[2])),A,B)
end

function LinearAlgebra.mul!(
  C::AbstractParamArray,
  A::AbstractVecOrMat,
  B::AbstractParamArray,
  α::Number,β::Number)

  @check param_length(C) == param_length(B)
  @inbounds for i in param_eachindex(C)
    ci = param_getindex(C,i)
    bi = param_getindex(B,i)
    mul!(ci,A,bi,α,β)
  end
  return C
end

function LinearAlgebra.mul!(
  C::AbstractParamArray,
  A::AbstractParamArray,
  B::AbstractParamArray,
  α::Number,β::Number)

  @check param_length(C) == param_length(A) == param_length(B)
  @inbounds for i in param_eachindex(C)
    ci = param_getindex(C,i)
    ai = param_getindex(A,i)
    bi = param_getindex(B,i)
    mul!(ci,ai,bi,α,β)
  end
  return C
end

function (\)(A::AbstractParamMatrix,B::AbstractParamVector)
  TS = LinearAlgebra.promote_op(LinearAlgebra.matprod,eltype(A),eltype(B))
  C = similar(B,TS,axes(A,1))
  @inbounds for i in param_eachindex(C)
    param_getindex(C,i) .= param_getindex(A,i)\param_getindex(B,i)
  end
  return C
end

function LinearAlgebra.dot(A::AbstractParamArray,B::AbstractParamArray)
  @check size(A) == size(B)
  return map(dot,get_param_data(A),get_param_data(B))
end

function LinearAlgebra.norm(A::AbstractParamArray)
  return map(norm,get_param_data(A))
end

for factorization in (:LU,:Cholesky)
  @eval begin
    function LinearAlgebra.ldiv!(a::$factorization,B::AbstractParamArray)
      @inbounds for i in param_eachindex(B)
        bi = param_getindex(B,i)
        ldiv!(a,bi)
      end
      return B
    end
  end
end

function LinearAlgebra.ldiv!(A::AbstractParamArray,b::Factorization,C::AbstractParamArray)
  @check param_length(A) == param_length(C)
  @inbounds for i in param_eachindex(A)
    ai = param_getindex(A,i)
    ci = param_getindex(C,i)
    ldiv!(ai,b,ci)
  end
  return A
end

function LinearAlgebra.ldiv!(A::AbstractParamArray,B::ParamBlock,C::AbstractParamArray)
  @check param_length(A) == param_length(B) == param_length(C)
  @inbounds for i in param_eachindex(A)
    ai = param_getindex(A,i)
    bi = param_getindex(B,i)
    ci = param_getindex(C,i)
    ldiv!(ai,bi,ci)
  end
  return A
end

function LinearAlgebra.lu(A::AbstractParamArray;kwargs...)
  @notimplemented
end

function LinearAlgebra.lu!(A::AbstractParamArray,B::AbstractParamArray;kwargs...)
  @check param_length(A) == param_length(B)
  @inbounds for i in param_eachindex(A)
    ai = param_getindex(A,i)
    bi = param_getindex(B,i)
    lu!(ai,bi)
  end
  return A
end

# Gridap interface

Arrays.testitem(A::AbstractParamArray) = param_getindex(A,1)

function Arrays.testvalue(A::AbstractParamArray{T,N}) where {T,N}
  tv = testvalue(Array{T,N})
  plength = param_length(A)
  parameterize(tv,plength)
end

function Arrays.testvalue(::Type{A}) where {T,N,A<:AbstractParamArray{T,N}}
  tv = testvalue(Array{T,N})
  plength = one(Int)
  parameterize(tv,plength)
end

function Arrays.CachedArray(A::AbstractParamArray)
  @notimplemented
end

function Arrays.setsize!(A::AbstractParamArray{T,N},s::NTuple{N,Integer}) where {T,N}
  @notimplemented
end

function Geometry.pos_neg_data(
  ipos_to_val::AbstractArray{<:AbstractParamArray{<:Number}},
  i_to_iposneg::PosNegPartition)
  nineg = length(i_to_iposneg.ineg_to_i)
  val = testitem(ipos_to_val)
  zs = 0 .* size(val)
  void = similar(val,eltype(val),zs)
  ineg_to_val = Fill(void,nineg)
  ipos_to_val,ineg_to_val
end

function Geometry._setsize_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._setsize_compress!(ai,bi)
  end
end

function Geometry._uncached_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._uncached_compress!(ai,bi)
  end
  a
end

function Geometry._uncached_compress!(a::AbstractParamArray,b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    Geometry._uncached_compress!(ai,b)
  end
  a
end

function Geometry._copyto_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._copyto_compress!(ai,bi)
  end
  a
end

function Geometry._addto_compress!(a::AbstractParamArray,b::AbstractParamArray)
  @check param_length(a) == param_length(b)
  for i in param_eachindex(a)
    ai = param_getindex(a,i)
    bi = param_getindex(b,i)
    Geometry._addto_compress!(ai,bi)
  end
  a
end
