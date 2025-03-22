"""
    struct GenericParamOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: ParamOperator{O,T}
      op::ParamFEOperator{O,T}
    end

Wrapper that transforms a `ParamFEOperator` into an `ParamOperator`
"""
struct GenericParamOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: ParamOperator{O,T}
  op::ParamFEOperator{O,T}
end

get_fe_operator(op::GenericParamOperator) = op.op

struct GenericLinearNonlinearParamOperator{O,T} <: ParamOperator{O,T}
  op::LinearNonlinearParamFEOperator{O,T}
end

get_fe_operator(op::GenericLinearNonlinearParamOperator) = op.op

function ParamAlgebra.get_linear_operator(op::GenericLinearNonlinearParamOperator)
  op_lin = get_linear_operator(op.op)
  get_algebraic_operator(op_lin)
end

function ParamAlgebra.get_nonlinear_operator(op::GenericLinearNonlinearParamOperator)
  op_nlin = get_nonlinear_operator(op.op)
  get_algebraic_operator(op_nlin)
end

# utils

"""
    function collect_cell_matrix_for_trian(
      trial::FESpace,
      test::FESpace,
      a::DomainContribution,
      strian::Triangulation
      ) -> Tuple{Vector{<:Any},Vector{<:Any},Vector{<:Any}}

Computes the cell-wise data needed to assemble a global sparse matrix for a given
input triangulation `strian`
"""
function collect_cell_matrix_for_trian(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(trial,trian)
  Any[cell_mat_rc],Any[rows],Any[cols]
end

"""
    function collect_cell_vector_for_trian(
      test::FESpace,
      a::DomainContribution,
      strian::Triangulation
      ) -> Tuple{Vector{<:Any},Vector{<:Any}}

Computes the cell-wise data needed to assemble a global vector for a given
input triangulation `strian`
"""
function collect_cell_vector_for_trian(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)
  Any[cell_vec_r],Any[rows]
end
