"""
"""
struct ExtensionOperator{O<:OperatorType} <: NonlinearDomainOperator{O}
  op::NonlinearDomainOperator{O}
  assem::ExtensionAssembler
end

for (extf,f) in zip((:ExtensionLinearOperator,:ExtensionOperator),(:LinearFEOperator,:FEOperator))
  @eval begin
    function $extf(args...;kwargs...)
      feop = $f(args...;kwargs...)
      op = get_algebraic_operator(feop)
      assem = ExtensionAssembler(get_trial(op),get_test(op))
      ExtensionOperator(op,assem)
    end
  end
end

Utils.get_fe_operator(extop::ExtensionOperator) = get_fe_operator(extop.op)
ODEs.get_assembler(extop::ExtensionOperator) = extop.assem

function Algebra.residual(extop::ExtensionOperator,u::AbstractVector)
  b = residual(extop.op,u)
  extend_vector(extop.assem,b)
end

function Algebra.jacobian(extop::ExtensionOperator,u::AbstractVector)
  A = jacobian(extop.op,u)
  extend_matrix(extop.assem,A)
end

function Algebra.allocate_residual(op::ExtensionOperator,u)
  allocate_residual(extop.op,u)
end

function Algebra.residual!(b::AbstractVector,op::ExtensionOperator,u)
  residual!(b,extop.op,u)
  extend_vector(extop.assem,b)
end

function Algebra.allocate_jacobian(op::ExtensionOperator,u)
  allocate_jacobian(extop.op,u)
end

function Algebra.jacobian!(A::AbstractMatrix,op::ExtensionOperator,u)
  jacobian!(A,extop.op,u)
  extend_matrix(extop.assem,A)
end

# utils

function extend_matrix(assem::ExtensionAssembler,A::AbstractSparseMatrix)
  i,j,bg_v = findnz(A)
  row_to_bg_rows = get_rows_to_bg_rows(assem)
  col_to_bg_cols = get_cols_to_bg_cols(assem)
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

function extend_vector(assem::ExtensionAssembler,v::AbstractVector)
  row_to_bg_rows = get_rows_to_bg_rows(assem)
  bg_m = maximum(row_to_bg_rows)
  bg_v = zeros(eltype(v),bg_m)
  for (row,vrow) in enumerate(v)
    bg_row = row_to_bg_rows[row]
    bg_v[bg_row] = vrow
  end
  bg_v
end
