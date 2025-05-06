"""
"""
struct ExtensionOperator{T<:NonlinearOperator} <: NonlinearOperator
  op::T
  assem::ExtensionAssembler
end

const ExtensionLinearOperator = ExtensionOperator{LinearOperator}

function ExtensionLinearOperator(args...)
  op = LinearOperator(args...)
  trial = get_trial(op)
  test = get_test(op)
  assem = ExtensionAssembler(trial,test)
  ExtensionOperator(op,assem)
end

function ExtensionOperator(args...)
  op = AlgebraicOpFromFEOp(FEOperator(args...))
  trial = get_trial(op)
  test = get_test(op)
  assem = ExtensionAssembler(trial,test)
  ExtensionOperator(op,assem)
end

FESpaces.get_test(extop::ExtensionOperator) = get_test(extop.op)
FESpaces.get_trial(extop::ExtensionOperator) = get_trial(extop.op)
Algebra.get_matrix(extop::ExtensionOperator) = get_matrix(extop.op)
Algebra.get_vector(extop::ExtensionOperator) = get_vector(extop.op)
ODEs.get_assembler(extop::ExtensionOperator) = extop.assem

# linear interface

get_extended_matrix(extop::ExtensionLinearOperator) = extend_matrix(get_assembler(extop),get_matrix(extop))
get_extended_vector(extop::ExtensionLinearOperator) = extend_vector(get_assembler(extop),get_vector(extop))

function Algebra.residual!(b::AbstractVector,extop::ExtensionLinearOperator,x::AbstractVector)
  matrix = get_extended_matrix(extop)
  vector = get_extended_vector(extop)
  mul!(b,matrix,x)
  b .-= vector
  b
end

function Algebra.jacobian!(A::AbstractMatrix,extop::ExtensionLinearOperator,x::AbstractVector)
  matrix = get_extended_matrix(extop)
  copy_entries!(A,matrix)
  A
end

function Algebra.jacobian(extop::ExtensionLinearOperator,x::AbstractVector)
  get_extended_matrix(extop)
end

function Algebra.allocate_residual(extop::ExtensionLinearOperator,x::AbstractVector)
  x = get_extended_vector(extop)
  fill!(x,zero(eltype(x)))
  x
end

function Algebra.allocate_jacobian(extop::ExtensionLinearOperator,x::AbstractVector)
  get_extended_matrix(extop)
end

# nonlinear interface


# utils

function extend_matrix(a::ExtensionAssembler,A::AbstractSparseMatrix)
  i,j,bg_v = findnz(A)
  row_to_bg_rows = get_rows_to_bg_rows(a)
  col_to_bg_cols = get_cols_to_bg_cols(a)
  bg_m = maximum(row_to_bg_rows)
  bg_n = maximum(col_to_bg_cols)
  bg_i = similar(i)
  bg_j = similar(j)
  for k in eachindex(bg_i)
    bg_i[k] = row_to_bg_rows[i[k]]
    bg_j[k] = col_to_bg_cols[j[k]]
  end
  sparse(bg_i,bg_j,bg_v,bg_m,bg_n)
end

function extend_vector(a::ExtensionAssembler,v::AbstractVector)
  row_to_bg_rows = get_rows_to_bg_rows(a)
  bg_m = maximum(row_to_bg_rows)
  bg_v = zeros(eltype(v),bg_m)
  for (row,vrow) in enumerate(v)
    bg_row = row_to_bg_rows[row]
    bg_v[bg_row] = vrow
  end
  bg_v
end
