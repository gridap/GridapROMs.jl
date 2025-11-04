"""
    struct LinearNonlinearParamFEOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: ParamFEOperator{O,T}
      op_linear::ParamFEOperator
      op_nonlinear::ParamFEOperator
    end

Interface to accommodate the separation of terms depending on their linearity in
a nonlinear problem. This allows to build and store once and for all linear
residuals/Jacobians, and in the Newton-like iterations only evaluate and assemble
only the nonlinear components
"""
struct LinearNonlinearParamFEOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: ParamFEOperator{O,T}
  op_linear::ParamFEOperator
  op_nonlinear::ParamFEOperator

  function LinearNonlinearParamFEOperator{O}(
    op_linear::ParamFEOperator{<:UnEvalOperatorType,T},
    op_nonlinear::ParamFEOperator{<:UnEvalOperatorType,T}
    ) where {O,T}

    new{O,T}(op_linear,op_nonlinear)
  end
end

function LinearNonlinearParamFEOperator(
  op_lin::ParamFEOperator,
  op_nlin::ParamFEOperator)

  LinearNonlinearParamFEOperator{LinearNonlinearParamEq}(op_lin,op_nlin)
end

ParamAlgebra.get_linear_operator(op::LinearNonlinearParamFEOperator) = op.op_linear
ParamAlgebra.get_nonlinear_operator(op::LinearNonlinearParamFEOperator) = op.op_nonlinear

function FESpaces.get_algebraic_operator(op::LinearNonlinearParamFEOperator)
  GenericLinearNonlinearParamOperator(op)
end

function FESpaces.get_test(op::LinearNonlinearParamFEOperator)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  get_test(op.op_nonlinear)
end

function FESpaces.get_trial(op::LinearNonlinearParamFEOperator)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  get_trial(op.op_nonlinear)
end

function ParamDataStructures.realization(op::LinearNonlinearParamFEOperator;kwargs...)
  realization(op.op_nonlinear;kwargs...)
end

function FESpaces.assemble_matrix(op::LinearNonlinearParamFEOperator,form::Function)
  @check get_test(op.op_linear) === get_test(op.op_nonlinear)
  @check get_trial(op.op_linear) === get_trial(op.op_nonlinear)
  assemble_matrix(op.op_nonlinear,form)
end

function join_operators(
  op_lin::ParamFEOperator,
  op_nlin::ParamFEOperator)

  op_lin = set_domains(op_lin)
  op_nlin = set_domains(op_nlin)

  @check get_trial(op_lin) == get_trial(op_nlin)
  @check get_test(op_lin) == get_test(op_nlin)
  @check op_lin.pspace === op_nlin.pspace

  trial = get_trial(op_lin)
  test = get_test(op_lin)

  res(μ,u,v) = get_res(op_lin)(μ,u,v) + get_res(op_nlin)(μ,u,v)
  jac(μ,u,du,v) = get_jac(op_lin)(μ,u,du,v) + get_jac(op_nlin)(μ,u,du,v)
  ParamFEOperator(res,jac,op_lin.pspace,trial,test)
end

"""
    join_operators(op::LinearNonlinearParamFEOperator) -> ParamFEOperator

Joins the linear/nonlinear parts of the operator and returns the resulting operator
"""
function join_operators(op::LinearNonlinearParamFEOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  join_operators(op_lin,op_nlin)
end

function set_domains(op::LinearNonlinearParamFEOperator)
  op_lin = set_domains(get_linear_operator(op))
  op_nlin = set_domains(get_nonlinear_operator(op))
  LinearNonlinearParamFEOperator(op_lin,op_nlin)
end
