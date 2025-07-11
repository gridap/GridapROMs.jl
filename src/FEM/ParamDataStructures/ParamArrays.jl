# generic constructors

ParamArray(a::AbstractArray{<:Number},l::Integer) = ConsecutiveParamArray(a,l)
ParamArray(A::AbstractArray{<:AbstractArray}) = ConsecutiveParamArray(A)

function ParamArray(A::AbstractArray{<:AbstractArray},l::Integer)
  plength(A) = length(A)
  plength(A::AbstractParamArray) = param_length(A)
  @assert plength(A) == l
  ParamArray(A)
end

GenericParamArray(A::AbstractArray{<:AbstractVector}) = GenericParamVector(A)
GenericParamArray(A::AbstractArray{<:AbstractMatrix}) = GenericParamMatrix(A)

"""
    get_all_data(A::ParamArray) -> AbstractArray{<:Any}

Returns all the entries stored in `A`, assuming `A` stores its entries consecutively
"""
get_all_data(A::ParamArray) = @abstractmethod

function Base.setindex!(
  A::ParamArray{T,N},
  v::ParamArray{T′,N},
  i::Vararg{Integer,N}
  ) where {T,T′,N}

  setindex!(get_all_data(A),get_all_data(v),i...,:)
end

for op in (:+,:-,:*,:/)
  @eval begin
    function ($op)(A::ParamArray,b::AbstractArray{<:Number})
      @check param_length(A) == length(b)
      B = copy(A)
      @inbounds for i in eachindex(B)
        bi = B[i]
        B[i] = $op(bi,fill(b[i],size(bi)))
      end
      return B
    end
  end
end

function (+)(b::AbstractArray{<:Number},A::ParamArray)
  (+)(A,b)
end

function (-)(b::AbstractArray{<:Number},A::ParamArray)
  (-)((-)(A,b))
end

function (*)(b::Number,A::ParamArray)
  (*)(A,b)
end

function Base.fill!(A::ParamArray,b::Number)
  fill!(get_all_data(A),b)
  return A
end

function LinearAlgebra.rmul!(A::ParamArray,b::Number)
  rmul!(get_all_data(A),b)
  return A
end

function LinearAlgebra.axpy!(α::Number,A::ParamArray,B::ParamArray)
  axpy!(α,get_all_data(A),get_all_data(B))
  return B
end

function Arrays.setsize!(A::ParamArray{T,N},s::NTuple{N,Integer}) where {T,N}
  setsize!(get_all_data(A),(s...,param_length(A)))
  A
end

function Base.getproperty(A::ParamArray,sym::Symbol)
  if sym == :array
    getfield(get_all_data(A),sym)
  else
    getfield(A,sym)
  end
end

"""
    struct TrivialParamArray{T<:Number,N,A<:AbstractArray{T,N}} <: ParamArray{T,N}
      data::A
      plength::Int
    end

Wrapper for a non-parametric array `data` that we wish assumed a parametric `length`
"""
struct TrivialParamArray{T<:Number,N,A<:AbstractArray{T,N}} <: ParamArray{T,N}
  data::A
  plength::Int
  function TrivialParamArray(data::AbstractArray{T,N},plength::Int=1) where {T<:Number,N}
    A = typeof(data)
    new{T,N,A}(data,plength)
  end
end

param_length(A::TrivialParamArray) = A.plength
get_all_data(A::TrivialParamArray) = A.data

function TrivialParamArray(A::AbstractParamArray,args...)
  A
end

Base.size(A::TrivialParamArray{T,N}) where {T,N} = tfill(A.plength,Val{N}())

innersize(A::TrivialParamArray) = size(A.data)

Base.@propagate_inbounds function Base.getindex(A::TrivialParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    A.data
  else
    fill(zero(T),innersize(A))
  end
end

Base.@propagate_inbounds function Base.setindex!(A::TrivialParamArray,v,i::Integer...)
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  all(i.==iblock) && copyto!(A.data,v)
end

function Base.copy(A::TrivialParamArray)
  data′ = copy(get_all_data(A))
  TrivialParamArray(data′,param_length(A))
end

function Base.similar(A::TrivialParamArray{T,N},::Type{<:AbstractArray{T′}}) where {T,T′,N}
  data′ = similar(get_all_data(A),T′)
  TrivialParamArray(data′,param_length(A))
end

function Base.similar(A::TrivialParamArray{T,N},::Type{<:AbstractArray{T′}},dims::Dims{N}) where {T,T′,N}
  data′ = similar(get_all_data(A),T′,dims...)
  TrivialParamArray(data′,param_length(A))
end

function Base.copyto!(A::TrivialParamArray,B::TrivialParamArray)
  @check size(A) == size(B)
  copyto!(get_all_data(A),get_all_data(B))
  A
end

function Base.vec(A::TrivialParamArray)
  data′ = vec(get_all_data(A))
  TrivialParamArray(data′,param_length(A))
end

function Arrays.CachedArray(A::TrivialParamArray)
  data′ = CachedArray(get_all_data(A))
  TrivialParamArray(data′,param_length(A))
end

function get_param_entry!(v::AbstractVector{T},A::TrivialParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  entry = getindex(get_all_data(A),i...)
  fill!(v,entry)
end

function get_param_entry(A::TrivialParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  entry = getindex(get_all_data(A),i...)
  fill(entry,param_length(A))
end

"""
    struct ConsecutiveParamArray{T,N,M,A<:AbstractArray{T,M}} <: ParamArray{T,N}
      data::A
    end

Parametric array with entries stored consecutively in memory. It is
characterized by an inner size equal to `size(data)[1:N]`, and parametric length
equal to `size(data,N+1)`, where `data` is an `AbstractArray` of dimension `M = N+1`
"""
struct ConsecutiveParamArray{T,N,M,A<:AbstractArray{T,M}} <: ParamArray{T,N}
  data::A
  function ConsecutiveParamArray(data::AbstractArray{T,M}) where {T<:Number,M}
    N = M - 1
    A = typeof(data)
    new{T,N,M,A}(data)
  end
end

"""
    const ConsecutiveParamVector{T,A<:AbstractArray{T,2}} = ConsecutiveParamArray{T,1,2,A}
"""
const ConsecutiveParamVector{T,A<:AbstractArray{T,2}} = ConsecutiveParamArray{T,1,2,A}

"""
    const ConsecutiveParamMatrix{T,A<:AbstractArray{T,3}} = ConsecutiveParamArray{T,2,3,A}
"""
const ConsecutiveParamMatrix{T,A<:AbstractArray{T,3}} = ConsecutiveParamArray{T,2,3,A}

param_length(A::ConsecutiveParamArray{T,N,M}) where {T,N,M} = size(A.data,M)
get_all_data(A::ConsecutiveParamArray) = A.data

function ConsecutiveParamArray(a::AbstractVector{<:AbstractArray{T,N}}) where {T,N}
  if all(size(ai)==size(first(a)) for ai in a)
    ConsecutiveParamArray(stack(a))
  else
    if N==1
      GenericParamVector(a)
    elseif N==2
      GenericParamMatrix(a)
    else
      @notimplemented
    end
  end
end

function ConsecutiveParamArray(a::AbstractArray{<:Number,N},l::Integer) where N
  outer = (tfill(1,Val{N}())...,l)
  data = repeat(a;outer)
  ConsecutiveParamArray(data)
end

Base.size(A::ConsecutiveParamArray{T,N}) where {T,N} = tfill(param_length(A),Val{N}())

innersize(A::ConsecutiveParamArray{T,N}) where {T,N} = front_tuple(size(get_all_data(A)),Val{N}())

Base.@propagate_inbounds function Base.getindex(A::ConsecutiveParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    A.data[_ncolons(Val{N}())...,iblock]
  else
    fill(zero(T),innersize(A))
  end
end

Base.@propagate_inbounds function Base.setindex!(A::ConsecutiveParamArray{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    A.data[_ncolons(Val{N}())...,iblock] = v
  end
end

function Base.copy(A::ConsecutiveParamArray)
  data′ = copy(get_all_data(A))
  ConsecutiveParamArray(data′)
end

function Base.similar(A::ConsecutiveParamArray{T,N},::Type{<:AbstractArray{T′,N}}) where {T,T′,N}
  data′ = similar(get_all_data(A),T′)
  ConsecutiveParamArray(data′)
end

function Base.similar(A::ConsecutiveParamArray{T,N},::Type{<:AbstractArray{T′}},dims::Dims{N}) where {T,T′,N}
  pdims = (dims...,param_length(A))
  data′ = similar(get_all_data(A),T′,pdims...)
  ConsecutiveParamArray(data′)
end

function Base.copyto!(A::ConsecutiveParamArray,B::ConsecutiveParamArray)
  copyto!(get_all_data(A),get_all_data(B))
  A
end

function Base.vec(A::ConsecutiveParamArray)
  data′ = reshape(get_all_data(A),:,param_length(A))
  ConsecutiveParamArray(data′)
end

function Base.vcat(A::ConsecutiveParamArray,B::ConsecutiveParamArray)
  data′ = vcat(get_all_data(A),get_all_data(B))
  ConsecutiveParamArray(data′)
end

function Base.hcat(A::ConsecutiveParamArray,B::ConsecutiveParamArray)
  data′ = hcat(get_all_data(A),get_all_data(B))
  ConsecutiveParamArray(data′)
end

function param_cat(A::Vector{<:ConsecutiveParamArray{T,N}}) where {T,N}
  data′ = cat(map(get_all_data,A)...;dims=N+1)
  ConsecutiveParamArray(data′)
end

for op in (:+,:-)
  @eval begin
    function ($op)(A::ConsecutiveParamArray,B::ConsecutiveParamArray)
      AB = ($op)(get_all_data(A),get_all_data(B))
      ConsecutiveParamArray(AB)
    end
  end
end

for op in (:*,:/)
  @eval begin
    function ($op)(A::ConsecutiveParamArray,b::Number)
      Ab = ($op)(get_all_data(A),b)
      ConsecutiveParamArray(Ab)
    end
  end
end

function Arrays.CachedArray(A::ConsecutiveParamArray)
  data′ = CachedArray(get_all_data(A))
  ConsecutiveParamArray(data′)
end

function param_getindex(A::ConsecutiveParamArray{T,N},i::Integer) where {T,N}
  view(A.data,_ncolons(Val{N}())...,i)
end

function param_setindex!(A::ConsecutiveParamArray{T,N},v,i::Integer) where {T,N}
  @views A.data[_ncolons(Val{N}())...,i] = v
  v
end

function get_param_entry!(v::AbstractVector{T},A::ConsecutiveParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  data = get_all_data(A)
  for j in eachindex(v)
    @inbounds v[j] = data[i...,j]
  end
  v
end

function get_param_entry(A::ConsecutiveParamArray{T,N},i::Vararg{Integer,N}) where {T,N}
  v = zeros(T,param_length(A))
  get_param_entry!(v,A,i...)
end

function get_param_entry(A::ConsecutiveParamArray,i...)
  data′ = view(get_all_data(A),i...,:)
  ConsecutiveParamArray(data′)
end

"""
    struct GenericParamVector{Tv,Ti} <: ParamArray{Tv,1}
      data::Vector{Tv}
      ptrs::Vector{Ti}
    end

Parametric vector with entries stored non-consecutively in memory
"""
struct GenericParamVector{Tv,Ti} <: ParamArray{Tv,1}
  data::Vector{Tv}
  ptrs::Vector{Ti}
end

param_length(A::GenericParamVector) = length(A.ptrs)-1
get_all_data(A::GenericParamVector) = A.data
get_ptrs(A::GenericParamVector) = A.ptrs

function GenericParamVector(a::AbstractVector{<:AbstractVector{Tv}}) where Tv
  ptrs = _vec_of_pointers(a)
  n = length(a)
  u = one(eltype(ptrs))
  ndata = ptrs[end]-u
  data = Vector{Tv}(undef,ndata)
  p = 1
  @inbounds for i in 1:n
    ai = a[i]
    for j in 1:length(ai)
      aij = ai[j]
      data[p] = aij
      p += 1
    end
  end
  GenericParamVector(data,ptrs)
end

Base.size(A::GenericParamVector) = (length(A.ptrs)-1,)

function innersize(A::GenericParamVector)
  innerlength = ptrs[2]-ptrs[1]
  @check all(ptrs[i+1]-ptrs[i] == innerlength for i in 1:length(ptrs)-1)
  (innerlength,)
end

Base.@propagate_inbounds function Base.getindex(A::GenericParamVector,i::Integer)
  @boundscheck checkbounds(A,i)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  A.data[pini:pend]
end

Base.@propagate_inbounds function Base.setindex!(A::GenericParamVector,v,i::Integer)
  @boundscheck checkbounds(A,i)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  A.data[pini:pend] = v
end

function Base.copy(A::GenericParamVector)
  data′ = copy(get_all_data(A))
  ptrs = get_ptrs(A)
  GenericParamVector(data′,ptrs)
end

function Base.similar(A::GenericParamVector{T},::Type{<:AbstractVector{T′}}) where {T,T′}
  data′ = similar(get_all_data(A),T′)
  ptrs = get_ptrs(A)
  GenericParamVector(data′,ptrs)
end

function Base.copyto!(A::GenericParamVector,B::GenericParamVector)
  @check get_ptrs(A) == get_ptrs(B)
  copyto!(get_all_data(A),get_all_data(B))
  A
end

function Base.vec(A::GenericParamVector)
  A
end

function param_cat(A::Vector{<:GenericParamVector})
  A1, = A
  data = copy(A1.data)
  ptrs = copy(A1.ptrs)
  for (i,Ai) in enumerate(A)
    if i != 1
      append!(data,Ai.data)
      append_ptrs!(ptrs,Ai.ptrs)
    end
  end
  GenericParamVector(data,ptrs)
end

for op in (:+,:-)
  @eval begin
    function ($op)(A::GenericParamVector,B::GenericParamVector)
      @check get_ptrs(A) == get_ptrs(B)
      AB = ($op)(get_all_data(A),get_all_data(B))
      ptrs = get_ptrs(A)
      GenericParamVector(AB,ptrs)
    end
  end
end

for op in (:*,:/)
  @eval begin
    function ($op)(A::GenericParamVector,b::Number)
      Ab = ($op)(get_all_data(A),b)
      ptrs = get_ptrs(A)
      GenericParamVector(AB,ptrs)
    end
  end
end

for op in (:+,:-,:*,:/)
  @eval begin
    function ($op)(A::GenericParamVector,b::AbstractArray{<:Number})
      @check param_length(A) == length(b)
      B = copy(A)
      @inbounds for i in eachindex(B)
        bi = B[i]
        B[i] = $op(bi,b[i])
      end
      return B
    end
  end
end

function Arrays.CachedArray(A::GenericParamVector)
  data′ = CachedArray(get_all_data(A))
  ptrs = get_ptrs(A)
  GenericParamVector(data′,ptrs)
end

function get_param_entry!(v::AbstractVector{T},A::GenericParamVector{T},i::Integer) where T
  for k in eachindex(v)
    @inbounds v[k] = A[k][i]
  end
  v
end

function get_param_entry(A::GenericParamVector{T},i::Integer) where T
  v = Vector{T}(undef,param_length(A))
  get_param_entry!(v,A,i)
  v
end

function ConsecutiveParamArray(A::GenericParamVector)
  msg = "Cannot transform this GenericParamVector to a ConsecutiveParamVector"
  δ = A.ptrs[2]-A.ptrs[1]
  for i in 2:length(A.ptrs)-1
    @check A.ptrs[i+1]-A.ptrs[i] == δ msg
  end
  plength = param_length(A)
  data = reshape(A.data,:,plength)
  ConsecutiveParamArray(data)
end

"""
    struct GenericParamMatrix{Tv,Ti} <: ParamArray{Tv,2}
      data::Vector{Tv}
      ptrs::Vector{Ti}
      nrows::Vector{Ti}
    end

Parametric matrix with entries stored non-consecutively in memory
"""
struct GenericParamMatrix{Tv,Ti} <: ParamArray{Tv,2}
  data::Vector{Tv}
  ptrs::Vector{Ti}
  nrows::Vector{Ti}
end

param_length(A::GenericParamMatrix) = length(A.ptrs)-1
get_all_data(A::GenericParamMatrix) = A.data
get_ptrs(A::GenericParamMatrix) = A.data
get_nrows(A::GenericParamMatrix) = A.nrows

function GenericParamMatrix(a::AbstractVector{<:AbstractMatrix{Tv}}) where Tv
  ptrs = _vec_of_pointers(a)
  n = length(a)
  Ti = eltype(ptrs)
  u = one(Ti)
  ndata = ptrs[end]-u
  data = Vector{Tv}(undef,ndata)
  nrows = Vector{Ti}(undef,n)
  p = 1
  @inbounds for i in 1:n
    ai = a[i]
    nrows[i] = size(ai,1)
    for j in 1:length(ai)
      aij = ai[j]
      data[p] = aij
      p += 1
    end
  end
  GenericParamMatrix(data,ptrs,nrows)
end

Base.size(A::GenericParamMatrix) = (length(A.ptrs)-1,length(A.ptrs)-1)

innersize(A::GenericParamMatrix) = @notimplemented

Base.@propagate_inbounds function Base.getindex(A::GenericParamMatrix{Tv},i::Integer,j::Integer) where Tv
  @boundscheck checkbounds(A,i,j)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  nrow = A.nrows[i]
  ncol = Int((pend-pini+1)/nrow)
  if i == j
    reshape(A.data[pini:pend],nrow,ncol)
  else
    fill(zero(Tv),nrow,ncol)
  end
end

Base.@propagate_inbounds function Base.setindex!(A::GenericParamMatrix,v,i::Integer,j::Integer)
  @boundscheck checkbounds(A,i,j)
  u = one(eltype(A.ptrs))
  pini = A.ptrs[i]
  pend = A.ptrs[i+1]-u
  if i == j
    A.data[pini:pend] = vec(v)
  end
end

function Base.copy(A::GenericParamMatrix)
  data′ = copy(get_all_data(A))
  ptrs = get_ptrs(A)
  nrows = get_nrows(A)
  GenericParamMatrix(data′,ptrs,nrows)
end

function Base.similar(A::GenericParamMatrix{T,N},::Type{<:AbstractMatrix{T′}}) where {T,T′,N}
  data′ = similar(get_all_data(A),T′)
  ptrs = get_ptrs(A)
  nrows = get_nrows(A)
  GenericParamMatrix(data′,ptrs,nrows)
end

function Base.copyto!(A::GenericParamMatrix,B::GenericParamMatrix)
  @check get_ptrs(A) == get_ptrs(B)
  @check get_nrows(A) == get_nrows(B)
  copyto!(get_all_data(A),get_all_data(B))
  A
end

function Base.vec(A::GenericParamMatrix)
  data = get_all_data(A)
  ptrs = get_ptrs(A)
  GenericParamVector(data,ptrs)
end

function param_cat(A::Vector{<:GenericParamMatrix})
  A1, = A
  data = copy(A1.data)
  ptrs = copy(A1.ptrs)
  rows = copy(A1.rows)
  for (i,Ai) in enumerate(A)
    if i != 1
      append!(data,Ai.data)
      append_ptrs!(ptrs,Ai.ptrs)
      append!(rows,Ai.rows)
    end
  end
  GenericParamMatrix(data,ptrs,rows)
end

for op in (:+,:-)
  @eval begin
    function ($op)(A::GenericParamMatrix,B::GenericParamMatrix)
      @check get_ptrs(A) == get_ptrs(B)
      @check get_nrows(A) == get_nrows(B)
      AB = ($op)(get_all_data(A),get_all_data(B))
      GenericParamMatrix(AB,get_ptrs(A),get_nrows(A))
    end
  end
end

for op in (:*,:/)
  @eval begin
    function ($op)(A::GenericParamMatrix,b::Number)
      Ab = ($op)(get_all_data(A),b)
      GenericParamMatrix(AB,get_ptrs(A),get_nrows(A))
    end
  end
end

function Arrays.CachedArray(A::GenericParamMatrix)
  data′ = CachedArray(get_all_data(A))
  ptrs = get_ptrs(A)
  nrows = get_nrows(A)
  GenericParamMatrix(data′,ptrs,nrows)
end

function get_param_entry!(v::AbstractVector{T},A::GenericParamMatrix{T},i::Integer,j::Integer) where T
  for k in eachindex(v)
    @inbounds v[k] = A[k][i,j]
  end
  v
end

function get_param_entry(A::GenericParamMatrix{T},i::Integer,j::Integer) where T
  v = Vector{T}(undef,param_length(A))
  get_param_entry!(v,A,i,j)
  v
end

function ConsecutiveParamArray(A::GenericParamMatrix)
  msg = "Cannot transform this GenericParamMatrix to a ConsecutiveParamMatrix"
  δ = A.ptrs[2]-A.ptrs[1]
  for i in 2:length(A.ptrs)-1
    @check A.ptrs[i+1]-A.ptrs[i] == δ msg
  end
  plength = param_length(A)
  data = reshape(A.data,first(A.nrows),:,plength)
  ConsecutiveParamArray(data)
end

# utils

function _vec_of_pointers(a::AbstractVector{<:AbstractArray})
  n = length(a)
  ptrs = Vector{Int}(undef,n+1)
  @inbounds for i in 1:n
    ai = a[i]
    ptrs[i+1] = length(ai)
  end
  length_to_ptrs!(ptrs)
  ptrs
end
