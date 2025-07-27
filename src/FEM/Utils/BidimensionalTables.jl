"""
     struct BidimensionalTable{T,Vd<:AbstractVector{T},Vp<:AbstractVector,Ti<:Integer} <: AbstractVector{Matrix{T}}
      data::Vd
      ptrs::Vp
      cols::Vector{Ti}
     end

Type representing a list of lists (i.e., a BidimensionalTable) in
compressed format.
"""
struct BidimensionalTable{T,Vd<:AbstractVector{T},Vp<:AbstractVector,Ti<:Integer} <: AbstractVector{Matrix{T}}
  data::Vd
  ptrs::Vp
  cols::Vector{Ti}
  function BidimensionalTable(data::AbstractVector,ptrs::AbstractVector,cols::AbstractVector)
    new{eltype(data),typeof(data),typeof(ptrs),eltype(cols)}(data,ptrs,cols)
  end
end

"""
    BidimensionalTable(a::AbstractArray{<:AbstractArray})

Build a BidimensionalTable from a vector of vectors. If the inputs are
multidimensional arrays instead of vectors, they are flattened.
"""
function BidimensionalTable(a::AbstractArray{<:AbstractArray})
  data,ptrs,cols = generate_1d_data_and_ptrs(a)
  BidimensionalTable(data,ptrs,cols)
end

function BidimensionalTable(a::AbstractMatrix{<:Number},ptrs::AbstractVector)
  data,ptrs,cols = generate_1d_data_and_ptrs(a,ptrs)
  BidimensionalTable(data,ptrs,cols)
end

function BidimensionalTable(a::BidimensionalTable)
  a
end

function Base.convert(::Type{BidimensionalTable{T,Vd,Vp,Ti}},table::BidimensionalTable{Ta,Vda,Vpa,Tj}) where {T,Vd,Vp,Ta,Vda,Vpa,Ti,Tj}
  data = convert(Vd,table.data)
  ptrs = convert(Vp,table.ptrs)
  cols = convert(Ti,table.cols)
  BidimensionalTable(data,ptrs,cols)
end

function Base.convert(::Type{BidimensionalTable{T,Vd,Vp,Ti}},table::BidimensionalTable{T,Vd,Vp,Ti}) where {T,Vd,Vp,Ti}
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
  nc = a.cols[1]
  nr = Int(pend - pini / nc)
  r = Matrix{T}(undef,nr,nc)
  CachedArray(r)
end

function Arrays.getindex!(c,a::BidimensionalTable,i::Integer)
  pini = a.ptrs[i]
  nc = a.cols[i]
  nr = Int(a.ptrs[i+1] - pini / nc)
  setsize!(c,(nr,nc))
  pini -= 1
  r = c.array
  @inbounds begin
    for i in 1:nr*nc
      r[i] = a.data[pini+i]
    end
  end
  r
end

function Base.getindex(a::BidimensionalTable,i::Integer)
  cache = array_cache(a)
  getindex!(cache,a,i)
end

function Base.getindex(a::BidimensionalTable,i::UnitRange)
  start = a.ptrs[i.start]
  r = a.ptrs[i.start]:(a.ptrs[i.stop+1]-1)
  data = a.data[r]
  r = i.start:(i.stop+1)
  ptrs = a.ptrs[r]
  o = ptrs[1]-1
  ptrs .-= o
  BidimensionalTable(data,ptrs,a.cols)
end

function Base.getindex(a::BidimensionalTable,ids::AbstractVector{<:Integer})
  ptrs = similar(a.ptrs,eltype(a.ptrs),length(ids)+1)
  for (i,id) in enumerate(ids)
    ptrs[i+1] = a.ptrs[id+1]-a.ptrs[id]
  end
  length_to_ptrs!(ptrs)
  ndata = ptrs[end]-1
  data = similar(a.data,eltype(a.data),ndata)
  for (i,id) in enumerate(ids)
    n = a.ptrs[id+1]-a.ptrs[id]
    p1 = ptrs[i]-1
    p2 = a.ptrs[id]-1
    for i in 1:n
      data[p1+i] = a.data[p2+i]
    end
  end
  BidimensionalTable(data,ptrs,a.cols)
end

# utils

function generate_1d_data_and_ptrs(vm::AbstractArray{<:AbstractMatrix{T}}) where T
  ptrs = Vector{Int32}(undef,length(vm)+1)
  cols = Vector{Int32}(undef,length(vm))
  _generate_1d_data_and_ptrs_fill_ptrs!(ptrs,cols,vm)
  length_to_ptrs!(ptrs)
  ndata = ptrs[end]-1
  data = Vector{T}(undef,ndata)
  _generate_1d_data_and_ptrs_fill_data!(data,vm)
  (data,ptrs,cols)
end

function generate_1d_data_and_ptrs(m::AbstractMatrix{T},ptrs::AbstractVector) where T
  data = vec(m)
  cols = Vector{Int32}(undef,length(ptrs)-1)
  fill!(m,size(m,2))
  length_to_ptrs!(ptrs)
  (data,ptrs,cols)
end

function _generate_1d_data_and_ptrs_fill_ptrs!(ptrs,cols,vm::AbstractArray{<:AbstractMatrix})
  c = array_cache(vm)
  kp = 1
  kc = 1
  for i in eachindex(vm)
    m = getindex!(c,vm,i)
    ptrs[kp+1] = length(m)
    cols[kc] = size(m,2)
    kp += 1
    kc += 1
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
