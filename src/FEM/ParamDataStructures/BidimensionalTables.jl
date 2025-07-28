"""
     struct BidimensionalTable{T,Vd<:AbstractVector{T},Vp<:AbstractVector} <: AbstractVector{Matrix{T}}
      data::Vd
      ptrs::Vp
     end

Type representing a list of lists (i.e., a BidimensionalTable) in
compressed format.
"""
struct BidimensionalTable{T,Vd<:AbstractMatrix{T},Vp<:AbstractVector} <: AbstractVector{Matrix{T}}
  data::Vd
  ptrs::Vp
  function BidimensionalTable(data::AbstractMatrix,ptrs::AbstractVector)
    new{eltype(data),typeof(data),typeof(ptrs)}(data,ptrs)
  end
end

"""
    BidimensionalTable(a::AbstractArray{<:AbstractArray})

Build a BidimensionalTable from a vector of vectors. If the inputs are
multidimensional arrays instead of vectors, they are flattened.
"""
function BidimensionalTable(a::AbstractArray{<:AbstractArray})
  data,ptrs = generate_1d_data_and_ptrs(a)
  BidimensionalTable(data,ptrs)
end

function BidimensionalTable(a::BidimensionalTable)
  a
end

function Base.convert(::Type{BidimensionalTable{T,Vd,Vp}},table::BidimensionalTable{Ta,Vda,Vpa}) where {T,Vd,Vp,Ta,Vda,Vpa}
  data = convert(Vd,table.data)
  ptrs = convert(Vp,table.ptrs)
  BidimensionalTable(data,ptrs)
end

function Base.convert(::Type{BidimensionalTable{T,Vd,Vp}},table::BidimensionalTable{T,Vd,Vp}) where {T,Vd,Vp}
  table
end

Base.IndexStyle(::Type{<:BidimensionalTable}) = IndexLinear()

Base.size(a::BidimensionalTable) = (length(a.ptrs)-1,)

function Arrays.array_cache(a::BidimensionalTable)
  T = eltype(a.data)
  if length(a.ptrs) == 0
    return CachedArray(testvalue(Matrix{T}))
  end

  pini = a.ptrs[1]
  pend = a.ptrs[2]
  nr = pend-pini
  nc = size(a.data,2)
  r = Matrix{T}(undef,nr,nc)
  CachedArray(r)
end

function Arrays.getindex!(c,a::BidimensionalTable,i::Integer)
  pini = a.ptrs[i]
  nr = a.ptrs[i+1]-pini
  nc = typeof(nr)(size(a.data,2))
  setsize!(c,(nr,nc))
  pini -= 1
  r = c.array
  # @inbounds begin
    for j in 1:nc, i in 1:nr
      r[i,j] = a.data[pini+i,j]
    end
  # end
  r
end

function Base.getindex(a::BidimensionalTable,i::Integer)
  cache = array_cache(a)
  getindex!(cache,a,i)
end

function Base.getindex(a::BidimensionalTable,i::UnitRange)
  start = a.ptrs[i.start]
  r = a.ptrs[i.start]:(a.ptrs[i.stop+1]-1)
  data = a.data[r,:]
  r = i.start:(i.stop+1)
  ptrs = a.ptrs[r]
  o = ptrs[1]-1
  ptrs .-= o
  BidimensionalTable(data,ptrs)
end

function Base.getindex(a::BidimensionalTable,ids::AbstractVector{<:Integer})
  ptrs = similar(a.ptrs,eltype(a.ptrs),length(ids)+1)
  for (i,id) in enumerate(ids)
    ptrs[i+1] = a.ptrs[id+1]-a.ptrs[id]
  end
  length_to_ptrs!(ptrs)
  nr = ptrs[end]-1
  nc = size(a.data,2)
  data = similar(a.data,eltype(a.data),nr,size(a.data,2))
  for (i,id) in enumerate(ids)
    n = a.ptrs[id+1]-a.ptrs[id]
    p1 = ptrs[i]-1
    p2 = a.ptrs[id]-1
    for j in 1:nc, i in 1:nr
      data[p1+i,j] = a.data[p2+i,j]
    end
  end
  BidimensionalTable(data,ptrs)
end

function Base.copy(a::BidimensionalTable)
  BidimensionalTable(copy(a.data),copy(a.ptrs))
end

param_length(a::BidimensionalTable) = size(a.data,2)
param_getindex(a::BidimensionalTable,i::Int) = Table(view(a.data,:,i),a.ptrs)
Arrays.testitem(a::BidimensionalTable) = param_getindex(a,1)

# utils

function generate_1d_data_and_ptrs(vm::AbstractArray{<:AbstractMatrix{T}}) where T
  ptrs = Vector{Int32}(undef,length(vm)+1)
  _generate_1d_data_and_ptrs_fill_ptrs!(ptrs,vm)
  length_to_ptrs!(ptrs)
  ndata = ptrs[end]-1
  data = Vector{T}(undef,ndata)
  _generate_1d_data_and_ptrs_fill_data!(data,vm)
  (data,ptrs)
end

function _generate_1d_data_and_ptrs_fill_ptrs!(ptrs,vm::AbstractArray{<:AbstractMatrix})
  c = array_cache(vm)
  k = 1
  for i in eachindex(vm)
    m = getindex!(c,vm,i)
    ptrs[k+1] = size(m,1)
    k += 1
  end
end

function _generate_1d_data_and_ptrs_fill_data!(data,vm)
  c = array_cache(vm)
  k = 1
  for i in eachindex(vm)
    m = getindex!(c,vm,i)
    for mi in m
      data[k] = mi
      k += 1
    end
  end
end
