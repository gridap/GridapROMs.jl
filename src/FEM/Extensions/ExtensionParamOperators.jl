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

ParamSteady.get_fe_operator(extop::ExtensionParamOperator) = get_fe_operator(extop.op)

function get_extended_assembler(extop::ExtensionParamOperator)
  trial = get_trial(extop)
  test = get_test(extop)
  ExtensionAssemblerInsertIn(trial,test)
end

function get_param_extended_assembler(extop::ExtensionParamOperator,r::Realization)
  parameterize(get_extended_assembler(extop),r)
end

const JointExtensionParamOperator{O<:UnEvalOperatorType} = ExtensionParamOperator{O,JointDomains}

function allocate_extended_residual(
  extop::JointExtensionParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(extop)
  v = get_fe_basis(test)
  assem = get_param_extended_assembler(extop,μ)

  res = get_res(extop)
  dc = res(μ,uh,v)
  vecdata = collect_cell_vector(test,dc)
  b = allocate_vector(assem,vecdata)

  b
end

function Algebra.residual!(
  b::AbstractVector,
  extop::JointExtensionParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(extop)
  v = get_fe_basis(test)
  assem = get_param_extended_assembler(extop,μ)

  res = get_res(extop)
  dc = res(μ,uh,v)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  extop::JointExtensionParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(extop)
  du = get_trial_fe_basis(trial)
  test = get_test(extop)
  v = get_fe_basis(test)
  assem = get_param_extended_assembler(extop,μ)

  jac = get_jac(extop)
  matdata = collect_cell_matrix(trial,test,jac(μ,uh,du,v))
  A = allocate_matrix(assem,matdata)

  A
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  extop::JointExtensionParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(extop)
  du = get_trial_fe_basis(trial)
  test = get_test(extop)
  v = get_fe_basis(test)
  assem = get_param_extended_assembler(extop,μ)

  jac = get_jac(extop)
  dc = jac(μ,uh,du,v)
  matdata = collect_cell_matrix(trial,test,dc)
  assemble_matrix_add!(A,assem,matdata)

  A
end

const SplitExtensionParamOperator{O<:UnEvalOperatorType} = ExtensionParamOperator{O,SplitDomains}

function Algebra.allocate_residual(
  extop::SplitExtensionParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(extop)
  v = get_fe_basis(test)
  assem = get_param_extended_assembler(extop,μ)

  trian_res = get_domains_res(extop)
  res = get_res(extop)
  dc = res(μ,uh,v)
  contribution(trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end
end

function Algebra.residual!(
  b::Contribution,
  extop::SplitExtensionParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(extop)
  v = get_fe_basis(test)
  assem = get_param_extended_assembler(extop,μ)

  trian_res = get_domains_res(extop)
  res = get_res(extop)
  dc = res(μ,uh,v)

  map(b.values,trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end

  b
end

function Algebra.allocate_jacobian(
  extop::SplitExtensionParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(extop)
  du = get_trial_fe_basis(trial)
  test = get_test(extop)
  v = get_fe_basis(test)
  assem = get_param_extended_assembler(extop,μ)

  trian_jac = get_domains_jac(extop)
  jac = get_jac(extop)
  dc = jac(μ,uh,du,v)
  contribution(trian_jac) do trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    allocate_matrix(assem,matdata)
  end
end

function ODEs.jacobian_add!(
  A::Contribution,
  extop::SplitExtensionParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(extop)
  du = get_trial_fe_basis(trial)
  test = get_test(extop)
  v = get_fe_basis(test)
  assem = get_param_extended_assembler(extop,μ)

  trian_jac = get_domains_jac(extop)
  jac = get_jac(extop)
  dc = jac(μ,uh,du,v)
  map(A.values,trian_jac) do values,trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    assemble_matrix_add!(values,assem,matdata)
  end

  A
end

# # transient

struct ODEExtensionParamOperator{O,T} <: ODEParamOperator{O,T}
  op::ODEParamOperator{O,T}
end

ParamSteady.get_fe_operator(extop::ODEExtensionParamOperator) = get_fe_operator(extop.op)
