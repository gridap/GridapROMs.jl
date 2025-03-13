# generic constructors

function ParamArray(a::AbstractArray{<:Number,N},l::Integer) where N
  ConsecutiveParamArray(a,l)
end

function ParamArray(A::AbstractArray{<:AbstractArray},l::Integer)
  plength(A) = length(A)
  plength(A::AbstractParamArray) = param_length(A)
  @assert plength(A) == l
  ParamArray(A)
end

function ParamArray(A::AbstractArray{<:AbstractArray{T,N}}) where {T,N}
  ConsecutiveParamArray(A)
end

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

"""
    struct ArrayOfArrays{T,N,A<:AbstractArray{T,N}} <: ParamArray{T,N}
      data::Vector{A}
    end

Parametric array with entries stored non-consecutively in memory. It is
characterized by an inner size equal to `size(data[1])`, and parametric length
equal to `length(data)`, where `data` is a `Vector{<:AbstractArray}`
"""
struct ArrayOfArrays{T,N,A<:AbstractArray{T,N}} <: ParamArray{T,N}
  data::Vector{A}
end

param_length(A::ArrayOfArrays) = length(A.data)

function ArrayOfArrays(a::AbstractArray{<:Number},l::Integer)
  data = Vector{typeof(a)}(undef,l)
  @inbounds for i in 1:l
    data[i] = copy(a)
  end
  ArrayOfArrays(data)
end

Base.size(A::ArrayOfArrays{T,N}) where {T,N} = tfill(param_length(A),Val{N}())

innersize(A::ArrayOfArrays{T,N}) where {T,N} = size(first(A.data))

Base.@propagate_inbounds function Base.getindex(A::ArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    A.data[iblock]
  else
    fill(zero(T),innersize(A))
  end
end

Base.@propagate_inbounds function Base.setindex!(A::ArrayOfArrays{T,N},v,i::Vararg{Integer,N}) where {T,N}
  @boundscheck checkbounds(A,i...)
  iblock = first(i)
  if all(i.==iblock)
    A.data[iblock] = v
  end
end

function Base.copy(A::ArrayOfArrays)
  data′ = map(copy,A.data)
  ArrayOfArrays(data′)
end

function Base.similar(A::ArrayOfArrays{T,N},::Type{<:AbstractArray{T′,N}}) where {T,T′,N}
  data′ = map(a -> similar(a,T′),A.data)
  ArrayOfArrays(data′)
end

function Base.similar(A::ArrayOfArrays{T,N},::Type{<:AbstractArray{T′}},dims::Dims{N}) where {T,T′,N}
  data′ = map(a -> similar(a,T′,dims...),A.data)
  ArrayOfArrays(data′)
end

function Base.copyto!(A::ArrayOfArrays,B::ArrayOfArrays)
  map(copyto!,A.data,B.data)
  A
end

function Base.vec(A::ArrayOfArrays)
  data′ = map(vec,A.data)
  ArrayOfArrays(data′)
end

function Base.setindex!(
  A::ArrayOfArrays{T,N},
  v::ArrayOfArrays{T′,N},
  i::Vararg{Integer,N}
  ) where {T,T′,N}

  for (Ak,vk) in zip(A.data,v.data)
    setindex!(Ak,vk,i...)
  end
end

for op in (:+,:-)
  @eval begin
    function ($op)(A::ArrayOfArrays,B::ArrayOfArrays)
      AB = similar(A)
      @inbounds for i in param_eachindex(A)
        AB[i] = ($op)(A[i],B[i])
      end
      AB
    end
  end
end

for op in (:*,:/)
  @eval begin
    function ($op)(A::ArrayOfArrays,b::Number)
      Ab = similar(A)
      @inbounds for i in param_eachindex(A)
        Ab[i] = ($op)(A[i],b)
      end
      Ab
    end
  end
end

function Base.fill!(A::ArrayOfArrays,b::Number)
  map(a -> fill!(a,b),A.data)
  return A
end

function LinearAlgebra.rmul!(A::ArrayOfArrays,b::Number)
  map(a -> rmul!(a,b),A.data)
  return A
end

function LinearAlgebra.axpy!(α::Number,A::ArrayOfArrays,B::ArrayOfArrays)
  map(a -> axpy!(α,a,b),A.data,B.data)
  return B
end

function Arrays.CachedArray(A::ArrayOfArrays)
  data′ = map(CachedArray,A.data)
  ArrayOfArrays(data′)
end

function Arrays.setsize!(A::ArrayOfArrays{T,N},s::NTuple{N,Integer}) where {T,N}
  map(a -> setsize!(a,s),A.data)
  A
end

function Base.getproperty(A::ArrayOfArrays,sym::Symbol)
  if sym == :array
    data′ = map(a -> getfield(a,sym),A.data)
    ArrayOfArrays(data′)
  else
    getfield(A,sym)
  end
end

function get_param_entry!(v::AbstractVector{T},A::ArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  for k in eachindex(v)
    @inbounds v[k] = A.data[k][i...]
  end
  v
end

function get_param_entry(A::ArrayOfArrays{T,N},i::Vararg{Integer,N}) where {T,N}
  v = Vector{T}(undef,param_length(A))
  get_param_entry!(v,A,i...)
  v
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
