"""
"""
struct ExtensionOperator{O<:OperatorType,T<:TriangulationStyle} <: DomainOperator{O,T}
  op::DomainOperator{O,T}
end

for (extf,f) in zip((:ExtensionLinearOperator,:ExtensionOperator),(:LinearFEOperator,:FEOperator))
  @eval begin
    function $extf(args...;kwargs...)
      feop = $f(args...;kwargs...)
      op = get_algebraic_operator(feop)
      ExtensionOperator(op)
    end
  end
end

Utils.get_fe_operator(extop::ExtensionOperator) = get_fe_operator(extop.op)

function ODEs.get_assembler(extop::ExtensionOperator)
  trial = get_trial(extop)
  test = get_test(extop)
  ExtensionAssembler(trial,test)
end

function Utils.set_domains(extop::ExtensionOperator,args...)
  ExtensionOperator(set_domains(extop.op,args...))
end

function Utils.change_domains(extop::ExtensionOperator,args...)
  ExtensionOperator(change_domains(extop.op,args...))
end
