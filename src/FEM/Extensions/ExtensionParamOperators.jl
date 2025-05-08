struct ExtensionParamOperator{O,T} <: ParamOperator{O,T}
  op::ParamOperator{O,T}
end

# constructors

function ExtensionLinearParamOperator(args...;kwargs...)
  op = LinearParamOperator(args...;kwargs...)
  ExtensionParamOperator(op)
end

function ExtensionParamOperator(args...;kwargs...)
  op = ParamOperator(args...;kwargs...)
  ExtensionParamOperator(op)
end

function ExtensionLinearNonlinearParamOperator(op_lin::ParamOperator,op_nlin::ParamOperator)
  op = LinearNonlinearParamOperator(op_lin,op_nlin)
  ExtensionParamOperator(op)
end

Utils.get_fe_operator(extop::ExtensionParamOperator) = get_fe_operator(extop.op)

function ODEs.get_assembler(extop::ExtensionParamOperator)
  trial = get_trial(extop)
  test = get_test(extop)
  ExtensionAssembler(trial,test)
end

function Utils.set_domains(extop::ExtensionParamOperator,args...)
  ExtensionParamOperator(set_domains(extop.op,args...))
end

function Utils.change_domains(extop::ExtensionParamOperator,args...)
  ExtensionParamOperator(change_domains(extop.op,args...))
end

# transient

const ODEExtensionParamOperator{O<:ODEParamOperatorType,T<:TriangulationStyle} = ExtensionParamOperator{O,T}

function TransientExtensionLinearParamOperator(args...;kwargs...)
  op = TransientLinearParamOperator(args...;kwargs...)
  ExtensionParamOperator(op)
end

function TransientExtensionParamOperator(args...;kwargs...)
  op = TransientParamOperator(args...;kwargs...)
  ExtensionParamOperator(op)
end

function TransientExtensionLinearNonlinearParamOperator(op_lin::ParamOperator,op_nlin::ParamOperator)
  op = TransientLinearNonlinearParamOperator(op_lin,op_nlin)
  ExtensionParamOperator(op)
end
