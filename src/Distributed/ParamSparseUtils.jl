@inline function Base.iterate(a::PartitionedArrays.NZIteratorCSC{<:ConsecutiveParamSparseMatrixCSC})
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
  k,kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = view(get_all_data(a.matrix),k,:)
  (i,j,v),(col,kstate)
end

@inline function Base.iterate(a::PartitionedArrays.NZIteratorCSC{<:ConsecutiveParamSparseMatrixCSC},state)
  col,kstate = state
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
  k,kstate = knext
  i = Int(rowvals(a.matrix)[k])
  j = col
  v = view(get_all_data(a.matrix),k,:)
  (i,j,v),(col,kstate)
end

@inline function Base.iterate(a::PartitionedArrays.NZIteratorCSR{<:ConsecutiveParamSparseMatrixCSR})
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
  k,kstate = knext
  i = row
  j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
  v = view(get_all_data(a.matrix),k,:)
  (i,j,v),(row,kstate)
end

@inline function Base.iterate(a::PartitionedArrays.NZIteratorCSR{<:ConsecutiveParamSparseMatrixCSR},state)
  row,kstate = state
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
  k,kstate = knext
  i = row
  j = Int(colvals(a.matrix)[k]+getoffset(a.matrix))
  v = view(get_all_data(a.matrix),k,:)
  (i,j,v),(row,kstate)
end

function PartitionedArrays.nziterator(a::ConsecutiveParamSparseMatrixCSR)
  PartitionedArrays.NZIteratorCSR(a)
end

function PartitionedArrays.nziterator(a::ConsecutiveParamSparseMatrixCSC)
  PartitionedArrays.NZIteratorCSC(a)
end

function PartitionedArrays.nzindex(a::ParamSparseMatrix,args...)
  PartitionedArrays.nzindex(testitem(a),args...)
end

function PartitionedArrays.compresscoo(
  ::Type{<:ConsecutiveParamSparseMatrixCSC},
  I::AbstractVector,
  J::AbstractVector,
  M::AbstractMatrix,
  m::Integer,
  n::Integer
  )

  combine = +
  sparse(I,J,M,m,n,combine)
end

struct ParamSubSparseMatrix{T,A,B,C} <: AbstractParamArray{T,2,PartitionedArrays.SubSparseMatrix{T,A,B,C}}
  parent::A
  indices::B
  inv_indices::C
  function ParamSubSparseMatrix(
    parent::ParamSparseMatrix{Tv,Ti},
    indices::Tuple,
    inv_indices::Tuple
    ) where {Tv,Ti}

    A = typeof(parent)
    B = typeof(indices)
    C = typeof(inv_indices)
    new{Tv,A,B,C}(parent,indices,inv_indices)
  end
end

function PartitionedArrays.SubSparseMatrix(parent::ParamSparseMatrix,indices::Tuple,inv_indices::Tuple)
  ParamSubSparseMatrix(parent,indices,inv_indices)
end

ParamDataStructures.param_length(A::ParamSubSparseMatrix) = param_length(A.parent)
innersize(a::ParamSubSparseMatrix) = map(length,a.indices)

Base.@propagate_inbounds function Base.getindex(A::ParamSubSparseMatrix{T},i::Integer,j::Integer) where T
  @boundscheck checkbounds(A,i...)
  if i == j
    PartitionedArrays.SubSparseMatrix(param_getindex(A.parent,i),A.indices,A.inv_indices)
  else
    fill(zero(T),innersize(A))
  end
end

function LinearAlgebra.mul!(
  C::ConsecutiveParamVector,
  A::ParamSubSparseMatrix{T,<:ParamDataStructures.ParamSparseMatrixCSC} where T,
  B::ConsecutiveParamVector,
  α::Number,
  β::Number
  )

  size(A,2) == size(B,1) || throw(DimensionMismatch())
  size(A,1) == size(C,1) || throw(DimensionMismatch())
  size(B,2) == size(C,2) || throw(DimensionMismatch())
  if β != 1
    β != 0 ? rmul!(C,β) : fill!(C,zero(eltype(C)))
  end
  rows,cols = A.indices
  invrows,invcols = A.inv_indices
  Ap = A.parent
  Cdata = get_all_data(C)
  nzv = get_all_data(Ap)
  Bdata = get_all_data(B)
  rv = rowvals(Ap)
  for (j,J) in enumerate(cols)
    for p in nzrange(Ap,J)
      I = rv[p]
      i = invrows[I]
      if i>0
        for l in param_eachindex(A)
          C[i,l] += nzv[p,l]*Bdata[j,l]*α
        end
      end
    end
  end
  C
end

function LinearAlgebra.fillstored!(
  A::ParamSubSparseMatrix{T,<:ParamDataStructures.ParamSparseMatrixCSC},v::Number
  ) where T

  rows,cols = A.indices
  invrows,invcols = A.inv_indices
  Ap = A.parent
  nzv = get_all_data(Ap)
  rv = rowvals(Ap)
  for (j,J) in enumerate(cols)
    for p in nzrange(Ap,J)
      I = rv[p]
      i = invrows[I]
      if i > 0
        for j in param_eachindex(A)
          nzv[p,j] = v
        end
      end
    end
  end
  A
end
