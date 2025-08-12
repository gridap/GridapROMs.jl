function Algebra.allocate_vector(::Type{V},n::Integer) where V<:AbstractParamVector
  @warn "Allocating a vector of unit parametric length, will likely result in an error"
  vector = allocate_vector(eltype(V),n)
  global_parameterize(vector,1)
end

for f in (:(Algebra.allocate_in_range),:(Algebra.allocate_in_domain))
  @eval begin
    function $f(::Type{PV},matrix::AbstractParamMatrix) where {T,PV<:AbstractParamVector{T}}
      V = Vector{T}
      item = testitem(matrix)
      plength = param_length(matrix)
      v = $f(V,item)
      global_parameterize(v,plength)
    end

    function $f(::Type{PV},matrix::BlockParamMatrix) where {T,PV<:BlockParamVector{T}}
      V = BlockVector{T,Vector{Vector{T}}}
      item = testitem(matrix)
      plength = param_length(matrix)
      v = $f(V,item)
      global_parameterize(v,plength)
    end

    function $f(matrix::AbstractParamMatrix{T}) where T
      V = ConsecutiveParamVector{T}
      $f(V,matrix)
    end

    function $f(matrix::BlockParamMatrix{T}) where T
      V = BlockConsecutiveParamVector{T}
      $f(V,matrix)
    end
  end
end

function Arrays.return_cache(k::AddEntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),param_length(vs))
end

function Arrays.evaluate!(cache,k::AddEntriesMap,A,vs::ParamBlock,is)
  add_entries!(cache,k.combine,A,vs,is)
end

function Arrays.evaluate!(cache,k::AddEntriesMap,A,vs::ParamBlock,is,js)
  add_entries!(cache,k.combine,A,vs,is,js)
end

@inline function Algebra.add_entries!(cache,combine::Function,A,vs::ParamBlock,is)
  Algebra._add_entries!(cache,combine,A,vs,is)
end

@inline function Algebra.add_entries!(cache,combine::Function,A,vs::ParamBlock,is,js)
  Algebra._add_entries!(cache,combine,A,vs,is,js)
end

@inline function Algebra.add_entries!(cache,combine::Function,A,vs::ParamBlock,is::OIdsToIds)
  add_ordered_entries!(cache,combine,A,vs,is)
end

@inline function Algebra.add_entries!(cache,combine::Function,A,vs::ParamBlock,is::OIdsToIds,js::OIdsToIds)
  add_ordered_entries!(cache,combine,A,vs,is,js)
end

@inline function Algebra._add_entries!(
  vij,combine::Function,A,vs::ParamBlock,is,js)

  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          get_param_entry!(vij,vs,li,lj)
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function add_ordered_entries!(
  vij,combine::Function,A,vs::ParamBlock,is::OIdsToIds,js::OIdsToIds)

  for (lj,j) in enumerate(js)
    if j>0
      ljp = js.terms[lj]
      for (li,i) in enumerate(is)
        if i>0
          lip = is.terms[li]
          get_param_entry!(vij,vs,lip,ljp)
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function Algebra._add_entries!(
  vi,combine::Function,A,vs::ParamBlock,is)

  for (li,i) in enumerate(is)
    if i>0
      get_param_entry!(vi,vs,li)
      add_entry!(combine,A,vi,i)
    end
  end
  A
end

@inline function add_ordered_entries!(
  vi,combine::Function,A,vs::ParamBlock,is::OIdsToIds)

  for (li,i) in enumerate(is)
    if i>0
      lip = is.terms[li]
      get_param_entry!(vi,vs,lip)
      add_entry!(combine,A,vi,i)
    end
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamVector,v::Number,i)
  @inbounds for k = param_eachindex(A)
    aik = A[k][i]
    A[k][i] = combine(aik,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamVector,v::AbstractVector,i)
  @inbounds for k = param_eachindex(A)
    aik = A[k][i]
    vk = v[k]
    A[k][i] = combine(aik,vk)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamVector,v::Number,i)
  data = get_all_data(A)
  @inbounds for k = param_eachindex(A)
    aik = data[i,k]
    data[i,k] = combine(aik,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamVector,v::AbstractVector,i)
  data = get_all_data(A)
  @inbounds for k = param_eachindex(A)
    aik = data[i,k]
    vk = v[k]
    data[i,k] = combine(aik,vk)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamMatrix,v::Number,i,j)
  @inbounds for k = param_eachindex(A)
    aijk = A[k][i,j]
    A[k][i,j] = combine(aijk,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::AbstractParamMatrix,v::AbstractVector,i,j)
  @inbounds for k = param_eachindex(A)
    aijk = A[k][i,j]
    vk = v[k]
    A[k][i,j] = combine(aijk,vk)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamMatrix,v::Number,i,j)
  data = get_all_data(A)
  @inbounds for k = param_eachindex(A)
    aijk = data[i,j,k]
    data[i,j,k] = combine(aijk,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamMatrix,v::AbstractVector,i,j)
  data = get_all_data(A)
  @inbounds for k = param_eachindex(A)
    aijk = data[i,j,k]
    vk = v[k]
    data[i,j,k] = combine(aijk,vk)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamSparseMatrix,v::Number,i,j)
  l = nz_index(A,i,j)
  nz = get_all_data(nonzeros(A))
  @inbounds for k = param_eachindex(A)
    aijk = nz[l,k]
    nz[l,k] = combine(aijk,v)
  end
  A
end

@inline function Algebra.add_entry!(combine::Function,A::ConsecutiveParamSparseMatrix,v::AbstractVector,i,j)
  l = nz_index(A,i,j)
  nz = get_all_data(nonzeros(A))
  @inbounds for k = param_eachindex(A)
    aijk = nz[l,k]
    vk = v[k]
    nz[l,k] = combine(aijk,vk)
  end
  A
end

struct ParamBuilder{A} <: GridapType
  builder::A
  plength::Int
end

for T in (:(Algebra.SparseMatrixBuilder),:(Algebra.ArrayBuilder))
  @eval begin
    function ParamDataStructures.parameterize(a::$T,plength::Int)
      ParamBuilder(a,plength)
    end
  end
end

ParamDataStructures.param_length(b::ParamBuilder) = b.plength

Algebra.get_array_type(b::ParamBuilder) = Algebra.get_array_type(b.builder)

function Algebra.nz_counter(b::ParamBuilder,axes)
  counter = nz_counter(b.builder,axes)
  ParamCounter(counter,b.plength)
end

struct ParamCounter{A}
  counter::A
  plength::Int
end

ParamDataStructures.param_length(a::ParamCounter) = a.plength

Algebra.LoopStyle(::Type{<:ParamCounter{C}}) where C = LoopStyle(C)

@inline function Algebra.add_entry!(::typeof(+),a::ParamCounter,v,i,j)
  add_entry!(+,a.counter,v,i,j)
end

@inline function Algebra.add_entry!(::typeof(+),a::ParamCounter,v,i)
  add_entry!(+,a.counter,v,i)
end

function Algebra.nz_allocation(a::ParamCounter{<:Algebra.ArrayCounter})
  v = nz_allocation(a.counter)
  global_parameterize(v,a.plength)
end

# csc

function Algebra.nz_allocation(a::ParamCounter{<:Algebra.CounterCSC{Tv,Ti}}) where {Tv,Ti}
  counter = a.counter
  colptr = Vector{Ti}(undef,counter.ncols+1)
  @inbounds for i in 1:counter.ncols
    colptr[i+1] = counter.colnnzmax[i]
  end
  length_to_ptrs!(colptr)
  plength = a.plength
  ndata = colptr[end] - one(Ti)
  pndata = ndata*plength
  rowval = Vector{Ti}(undef,ndata)
  nzval = zeros(Tv,pndata)
  colnnz = counter.colnnzmax
  fill!(colnnz,zero(Ti))
  ParamInserterCSC(counter.nrows,counter.ncols,colptr,colnnz,rowval,nzval,plength)
end

struct ParamInserterCSC{Tv,Ti}
  nrows::Int
  ncols::Int
  colptr::Vector{Ti}
  colnnz::Vector{Ti}
  rowval::Vector{Ti}
  nzval::Vector{Tv}
  plength::Int
end

ParamDataStructures.param_length(inserter::ParamInserterCSC) = inserter.plength

Algebra.LoopStyle(::Type{<:ParamInserterCSC}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::ParamInserterCSC,v::Nothing,i,j)
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check  pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
  end
  nothing
end

@noinline function Algebra.add_entry!(::typeof(+),a::ParamInserterCSC,v::Number,i,j)
  add_entry!(+,a,fill(v,param_length(a)),i,j)
end

@noinline function Algebra.add_entry!(::typeof(+),a::ParamInserterCSC,v::AbstractArray,i,j)
  pini = Int(a.colptr[j])
  pend = pini + Int(a.colnnz[j]) - 1
  ndata = length(a.rowval)
  pndata = length(a.nzval)
  p = searchsortedfirst(a.rowval,i,pini,pend,Base.Order.Forward)
  if (p>pend)
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    for (l,vl) in enumerate(p:ndata:pndata)
      @inbounds a.nzval[vl] = v[l]
    end
  elseif a.rowval[p] != i
    # shift one forward from p to pend
    @check pend+1 < Int(a.colptr[j+1])
    for k in pend:-1:p
      o = k + 1
      a.rowval[o] = a.rowval[k]
      for vl in k:ndata:pndata
        @inbounds a.nzval[vl+1] = a.nzval[vl]
      end
    end
    # add new entry
    a.colnnz[j] += 1
    a.rowval[p] = i
    for (l,vl) in enumerate(p:ndata:pndata)
      @inbounds a.nzval[vl] = v[l]
    end
  else
    # update existing entry
    for (l,vl) in enumerate(p:ndata:pndata)
      @inbounds a.nzval[vl] += v[l]
    end
  end
  nothing
end

function Algebra.create_from_nz(a::ParamInserterCSC)
  k = 1
  ndata = a.colptr[end]-1
  pndata = length(a.nzval)
  plength = param_length(a)
  for j in 1:a.ncols
    pini = Int(a.colptr[j])
    pend = pini + Int(a.colnnz[j]) - 1
    for p in pini:pend
      @inbounds for (il,l) in enumerate(p:ndata:pndata)
        α = k + (il-1)*ndata
        a.nzval[α] = a.nzval[l]
      end
      a.rowval[k] = a.rowval[p]
      k += 1
    end
  end
  @inbounds for j in 1:a.ncols
    a.colptr[j+1] = a.colnnz[j]
  end
  length_to_ptrs!(a.colptr)
  nnz = a.colptr[end]-1
  resize!(a.rowval,nnz)
  δ = Int(length(a.nzval)/plength) - nnz
  if δ > 0
    for l in 1:plength
      Base._deleteat!(a.nzval,l*nnz+1,δ)
    end
  end
  data = reshape(a.nzval,nnz,plength)
  ConsecutiveParamSparseMatrixCSC(a.nrows,a.ncols,a.colptr,a.rowval,data)
end

function resize_rows(a::AbstractMatrix,m::Int)
  v = vec(a)
  n = size(a,2)
  _resize_rows!(v,m,n)
  reshape(v,m,n)
end

function _resize_rows!(v::AbstractVector,m::Int,n::Int)
  δ = Int(length(v)/n) - m
  if δ > 0
    for l in 1:n
      Base._deleteat!(v,l*m+1,δ)
    end
  end
  v
end

# coo

function Algebra.nz_allocation(a::ParamCounter{<:Algebra.CounterCOO{T}}) where T
  counter = Algebra.CounterCOO{T}(a.counter.axes)
  I,J,V = allocate_coo_vectors(T,a.counter.nnz)
  M = repeat(V;outer=(1,a.plength))
  ParamAllocationCOO(counter,I,J,M,a.plength)
end

struct ParamAllocationCOO{T,A,B,C}
  counter::Algebra.CounterCOO{T,A}
  I::B
  J::B
  M::C
  plength::Int
end

ParamDataStructures.param_length(inserter::ParamAllocationCOO) = inserter.plength

Algebra.LoopStyle(::Type{<:ParamAllocationCOO}) = Loop()

@inline function Algebra.add_entry!(::typeof(+),a::ParamAllocationCOO{T},::Nothing,i,j) where T
  if is_entry_stored(T,i,j)
    a.counter.nnz = a.counter.nnz + 1
    k = a.counter.nnz
    a.I[k] = i
    a.J[k] = j
  end
  nothing
end

@inline function Algebra.add_entry!(::typeof(+),a::ParamAllocationCOO{T},v::AbstractArray,i,j) where T
  if is_entry_stored(T,i,j)
    a.counter.nnz = a.counter.nnz + 1
    k = a.counter.nnz
    a.I[k] = i
    a.J[k] = j
    for l in 1:a.plength
      @inbounds a.M[k,l] = v[l]
    end
  end
  nothing
end

function Algebra.create_from_nz(a::ParamAllocationCOO{T}) where T
  m,n = map(length,a.counter.axes)
  finalize_coo!(T,a.I,a.J,a.M,m,n)
  sparse_from_coo(T,a.I,a.J,a.M,m,n)
end

# csr: implentation needed

# utils

Base.@propagate_inbounds function Algebra.nz_index(A::ParamSparseMatrixCSC,i0::Integer,i1::Integer)
  if !(1 <= i0 <= innersize(A,1) && 1 <= i1 <= innersize(A,2)); throw(BoundsError()); end
  ptrs = SparseArrays.getcolptr(A)
  r1 = Int(ptrs[i1])
  r2 = Int(ptrs[i1+1]-1)
  (r1 > r2) && return -1
  r1 = searchsortedfirst(rowvals(A),i0,r1,r2,Base.Order.Forward)
  ((r1 > r2) || (rowvals(A)[r1] != i0)) ? -1 : r1
end
