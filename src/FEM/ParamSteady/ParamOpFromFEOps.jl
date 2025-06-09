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
