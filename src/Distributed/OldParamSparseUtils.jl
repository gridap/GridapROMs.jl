@inline function Base.iterate(a::NZIteratorCSC{<:ParamSparseMatrix})
  if nnz(a.matrix) == 0
      return nothing
  end
  col = 0
  knext = nothing
  while knext === nothing
    col += 1
    ks = nzrange(a.matrix,col)
    knext = iterate(ks)
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = nonzeros(a.matrix)[k,:]
  (i,j,v),(col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSC{<:ParamSparseMatrix},state)
  col, kstate = state
  ks = nzrange(a.matrix,col)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if col == size(a.matrix,2)
          return nothing
      end
      col += 1
      ks = nzrange(a.matrix,col)
      knext = iterate(ks)
    end
  end
  k, kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = nonzeros(a.matrix)[k,:]
  (i,j,v),(col,kstate)
end

@inline function Base.iterate(a::NZIteratorCSR{<:ParamSparseMatrix})
  if nnz(a.matrix) == 0
    return nothing
  end
  row = 0
  ptrs = a.matrix.rowptr
  knext = nothing
  while knext === nothing
    row += 1
    ks = nzrange(a.matrix,row)
    knext = iterate(ks)
  end
  k, kstate = knext
  i = row
  j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
  v = nonzeros(a.matrix)[k,:]
  (i,j,v),(row,kstate)
end

@inline function Base.iterate(a::NZIteratorCSR{<:ParamSparseMatrix},state)
  row, kstate = state
  ks = nzrange(a.matrix,row)
  knext = iterate(ks,kstate)
  if knext === nothing
    while knext === nothing
      if row == size(a.matrix,1)
        return nothing
      end
      row += 1
      ks = nzrange(a.matrix,row)
      knext = iterate(ks)
    end
  end
  k, kstate = knext
  i = row
  j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
  v = nonzeros(a.matrix)[k,:]
  (i,j,v),(row,kstate)
end

function PartitionedArrays.nziterator(a::ParamSparseMatrixCSR)
  PartitionedArrays.NZIteratorCSR(a)
end

function PartitionedArrays.nziterator(a::ParamSparseMatrixCSC)
  PartitionedArrays.NZIteratorCSC(a)
end

function PartitionedArrays.nzindex(a::ParamSparseMatrix,args...)
  PartitionedArrays.nzindex(testitem(a),args...)
end

function PartitionedArrays.compresscoo(
  ::Type{MatrixOfSparseMatricesCSC{Tv,Ti,L}},
  I::AbstractVector,
  J::AbstractVector,
  V::AbstractParamVector,
  m::Integer,
  n::Integer) where {Tv,Ti,L}

  Vitem = testitem(V)
  Aitem = compresscoo(SparseMatrixCSC{Tv,Ti},I,J,V,M,n)
  rowval = Aitem.rowval
  colval = Aitem.colval
  MatrixOfSparseMatricesCSC(rowval,colval,V,m,n)
end

function PartitionedArrays.compresscoo(
  ::Type{MatrixOfSparseMatricesCSR{Bi,Tv,Ti,L}},
  I::AbstractVector,
  J::AbstractVector,
  V::AbstractParamVector,
  m::Integer,
  n::Integer) where {Bi,Tv,Ti,L}

  Vitem = testitem(V)
  Aitem = compresscoo(SparseMatrixCSR{Bi,Tv,Ti},I,J,V,M,n)
  rowptr = Aitem.rowptr
  colval = Aitem.colval
  MatrixOfSparseMatricesCSR(rowptr,colval,V,m,n)
end

#TODO figure out if I need this
# Base.length(a::LocalView{T,N,<:ParamArray}) where {T,N} = length(a.plids_to_value)
# Base.size(a::LocalView{T,N,<:ParamArray}) where {T,N} = (length(a),)

# function Base.getindex(a::LocalView{T,N,<:ParamArray},i::Integer...) where {T,N}
#   LocalView(a.plids_to_value[i...],a.d_to_lid_to_plid)
# end

struct ParamSubSparseMatrix{T,A,B,C} <: AbstractParamArray{T,2,L,SubSparseMatrix{T,A,B,C}}
  data::A
  indices::B
  inv_indices::C
  function ParamSubSparseMatrix(
    data::ParamSparseMatrix{Tv,Ti,L},
    indices::Tuple,
    inv_indices::Tuple) where {Tv,Ti,L}

    A = typeof(data)
    B = typeof(indices)
    C = typeof(inv_indices)
    new{Tv,A,B,C}(data,indices,inv_indices)
  end
end

function PartitionedArrays.SubSparseMatrix(
  data::ParamSparseMatrix,
  indices::Tuple,
  inv_indices::Tuple)

  ParamSubSparseMatrix(data,indices,inv_indices)
end

Base.size(a::ParamSubSparseMatrix) = tfill(param_length(a),2)
Base.IndexStyle(::Type{<:ParamSubSparseMatrix}) = IndexCartesian()

@inline function ArraysOfArrays.innersize(a::ParamSubSparseMatrix)
  map(length,a.indices)
end

function Base.getindex(a::ParamSubSparseMatrix{T},i::Integer,j::Integer) where T
  @boundscheck checkbounds(a,i...)
  if i == j
    SubSparseMatrix(param_getindex(a.data,i),a.indices,a.inv_indices)
  else
    fill(zero(T),innersize(a))
  end
end

function Base.:\(
  a::PSparseMatrix{<:ParamMatrix{Ta,A,L}},
  b::PVector{<:ParamVector{Tb,B,L}}
  ) where {Ta,Tb,A,B,L}

  T = typeof(one(Ta)\one(Tb)+one(Ta)\one(Tb))
  PT = typeof(ParamVector{Vector{T}}(undef,L))
  c = PVector{PT}(undef,partition(axes(a,2)))
  fill!(c,zero(T))
  a_in_main = to_trivial_partition(a)
  b_in_main = to_trivial_partition(b,partition(axes(a_in_main,1)))
  c_in_main = to_trivial_partition(c,partition(axes(a_in_main,2)))
  map_main(partition(c_in_main),partition(a_in_main),partition(b_in_main)) do myc, mya, myb
    myc .= mya\myb
    nothing
  end
  PartitionedArrays.from_trivial_partition!(c,c_in_main)
  c
end

function Base.:*(a::PSparseMatrix,b::PVector{ParamVector{Tb,A,L}}) where {Tb,A,L}
  Ta = eltype(a)
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  c = PVector{ParamVector{T,L,A}}(undef,partition(axes(a,1)))
  mul!(c,a,b)
  c
end

function LinearAlgebra.mul!(
  C::ParamVector,
  A::PartitionedArrays.SubSparseMatrix{T,<:AbstractSparseMatrixCSC} where T,
  B::ParamVector,
  α::Number,
  β::Number)

  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  size(B, 2) == size(C, 2) || throw(DimensionMismatch())
  if β != 1
    β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
  end
  rows, cols = A.indices
  invrows, invcols = A.inv_indices
  Ap = A.parent
  nzv = nonzeros(Ap)
  rv = rowvals(Ap)
  for k = eachindex(B)
    for (j,J) in enumerate(cols)
      αxj = B[k][j] * α
      for p in nzrange(Ap,J)
        I = rv[p]
        i = invrows[I]
        if i>0
          C[k][i] += nzv[p]*αxj
        end
      end
    end
  end
  C
end

struct AdjointPVector{T,P} <: AbstractMatrix{T}
  parent::P
  function AdjointPVector(parent::P) where P<:PVector
    T = eltype(parent)
    new{T,P}(parent)
  end
end

Base.adjoint(parent::PVector) = AdjointPVector(parent)

function PartitionedArrays.partition(a::AdjointPVector)
  map(partition(a.parent)) do a
    a'
  end
end

Base.axes(a::AdjointPVector) = (Base.OneTo(1),axes(a.parent,1))

function PartitionedArrays.local_values(a::AdjointPVector)
  partition(a)
end

function PartitionedArrays.own_values(a::AdjointPVector)
  map(own_values,partition(a),partition(axes(a,1)))
end

Base.size(a::AdjointPVector) = (length(axes(a,1)),)
Base.IndexStyle(::Type{<:AdjointPVector}) = IndexLinear()
function Base.getindex(a::AdjointPVector,gid::Int)
  PartitionedArrays.scalar_indexing_action(a)
end

function Base.setindex(a::AdjointPVector,v,gid::Int)
  PartitionedArrays.scalar_indexing_action(a)
end

function Base.:*(a::AdjointPVector,b::PVector)
  Ta = eltype(a)
  Tb = eltype(b)
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  row_partition_in_main = trivial_partition(partition(axes(b,1)))
  a_in_main = PartitionedArrays.to_trivial_partition(a.parent,row_partition_in_main)
  b_in_main = PartitionedArrays.to_trivial_partition(b,row_partition_in_main)
  map_main(partition(a_in_main),partition(b_in_main)) do mya,myb
    c = zeros(T,length(mya))
    for i = eachindex(mya)
      c[i] = dot(mya[i],myb)
    end
    c
  end
end

function Base.:*(a::AdjointPVector,b::PVector{<:ParamArray})
  Ta = eltype(a)
  Tb = eltype(b)
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  row_partition_in_main = trivial_partition(partition(axes(b,1)))
  a_in_main = PartitionedArrays.to_trivial_partition(a.parent,row_partition_in_main)
  b_in_main = PartitionedArrays.to_trivial_partition(b,row_partition_in_main)
  map_main(partition(a_in_main),partition(b_in_main)) do mya,myb
    c = zeros(T,length(mya),length(myb))
    for i = eachindex(mya)
      for j = eachindex(myb)
        c[i,j] = dot(mya[i],myb[j])
      end
    end
    c
  end
end

struct AdjointPSparseMatrix{T,P} <: AbstractMatrix{T}
  parent::P
  function AdjointPSparseMatrix(parent::P) where P<:PSparseMatrix
    T = eltype(parent)
    new{T,P}(parent)
  end
end

Base.adjoint(parent::PSparseMatrix) = AdjointPSparseMatrix(parent)

function PartitionedArrays.partition(a::AdjointPSparseMatrix)
  map(partition(a.parent)) do a
    a'
  end
end

Base.axes(a::AdjointPSparseMatrix) = (axes(a.parent,2),axes(a.parent,1))

function PartitionedArrays.local_values(a::AdjointPSparseMatrix)
  partition(a)
end

function PartitionedArrays.own_values(a::AdjointPSparseMatrix)
  map(own_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

function PartitionedArrays.ghost_values(a::AdjointPSparseMatrix)
  map(ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

function PartitionedArrays.own_ghost_values(a::AdjointPSparseMatrix)
  map(own_ghost_values,partition(a),partition(axes(a,1)),partition(axes(a,2)))
end

Base.size(a::AdjointPSparseMatrix) = map(length,axes(a))
Base.IndexStyle(::Type{<:AdjointPSparseMatrix}) = IndexCartesian()
function Base.getindex(a::AdjointPSparseMatrix,gi::Int,gj::Int)
  PartitionedArrays.scalar_indexing_action(a)
end

function Base.setindex(a::AdjointPSparseMatrix,v,gi::Int,gj::Int)
  PartitionedArrays.scalar_indexing_action(a)
end

Base.:*(x::AdjointPVector,A::PSparseMatrix) = (A'*x.parent)'

function Base.:*(a::AdjointPSparseMatrix{Ta},b::PVector{ParamVector{Tb,A,L}}) where {Ta,Tb,A,L}
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  c = PVector{ParamVector{T,L,A}}(undef,partition(axes(a,1)))
  mul!(c,a,b)
  c
end

function LinearAlgebra.mul!(c::PVector,a::AdjointPSparseMatrix,b::PVector,α::Number,β::Number)
  @boundscheck @assert PartitionedArrays.matching_own_indices(axes(c,1),axes(a,1))
  @boundscheck @assert PartitionedArrays.matching_own_indices(axes(a,2),axes(b,1))
  @boundscheck @assert PartitionedArrays.matching_ghost_indices(axes(a,2),axes(b,1))
  t = consistent!(b)
  map(own_values(c),own_values(a),own_values(b)) do co,aoo,bo
    if β != 1
      β != 0 ? rmul!(co, β) : fill!(co,zero(eltype(co)))
    end
    mul!(co,aoo,bo,α,1)
  end
  wait(t)
  map(own_values(c),own_ghost_values(a),ghost_values(b)) do co,aoh,bh
    mul!(co,aoh,bh,α,1)
  end
  c
end

struct AdjointSubSparseMatrix{T,A,B,C} <: AbstractMatrix{T}
  parent::A
  indices::B
  inv_indices::C
  function AdjointSubSparseMatrix(
    parent::Adjoint{T,<:AbstractSparseMatrix{T}},
    indices::Tuple,
    inv_indices::Tuple) where T

    A = typeof(parent)
    B = typeof(indices)
    C = typeof(inv_indices)
    new{T,A,B,C}(parent,indices,inv_indices)
  end
end

function PartitionedArrays.SubSparseMatrix(
  parent::Adjoint{<:Any,<:AbstractSparseMatrix},
  indices::Tuple,
  inv_indices::Tuple)

  AdjointSubSparseMatrix(parent,indices,inv_indices)
end

Base.size(a::AdjointSubSparseMatrix) = map(length,a.indices)
Base.IndexStyle(::AdjointSubSparseMatrix) = IndexLinear()

function Base.getindex(a::AdjointSubSparseMatrix,i::Integer,j::Integer)
  I = a.indices[1][i]
  J = a.indices[2][j]
  a.parent[I,J]
end

function LinearAlgebra.mul!(
  C::ParamVector,
  A::AdjointSubSparseMatrix,
  B::ParamVector,
  α::Number,
  β::Number)

  size(A, 2) == size(B, 1) || throw(DimensionMismatch())
  size(A, 1) == size(C, 1) || throw(DimensionMismatch())
  size(B, 2) == size(C, 2) || throw(DimensionMismatch())
  if β != 1
    β != 0 ? rmul!(C, β) : fill!(C, zero(eltype(C)))
  end
  rows, cols = A.indices
  invrows, invcols = A.inv_indices
  Ap = A.parent.parent
  nzv = nonzeros(Ap)
  rv = rowvals(Ap)
  for k = eachindex(B)
    for (j,J) in enumerate(cols)
      αxj = B[k][j] * α
      for p in nzrange(Ap,J)
        I = rv[p]
        i = invrows[I]
        if i>0
          C[k][i] += nzv[p]'*αxj
        end
      end
    end
  end
  C
end
