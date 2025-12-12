"""
    abstract type UnEvalOperatorType <: GridapType end

Type representing operators that are not evaluated yet. This may include operators
representing transient problems (although the implementation in [`Gridap`](@ref)
differs), parametric problems, and a combination thereof. Could become a supertype
of `ODEOperatorType` in [`Gridap`](@ref)
"""
abstract type UnEvalOperatorType <: GridapType end

"""
    struct LinearParamEq <: UnEvalOperatorType end
"""
struct LinearParamEq <: UnEvalOperatorType end

"""
    struct NonlinearParamEq <: UnEvalOperatorType end
"""
struct NonlinearParamEq <: UnEvalOperatorType end

"""
    struct LinearNonlinearParamEq <: UnEvalOperatorType end
"""
struct LinearNonlinearParamEq <: UnEvalOperatorType end

UnEvalOperatorType(::Type{LinearEq}) = LinearParamEq
UnEvalOperatorType(::Type{NonlinearEq}) = NonlinearParamEq
UnEvalOperatorType(::Type{LinearNonlinearEq}) = LinearNonlinearParamEq

"""
    abstract type ParamOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: NonlinearParamOperator end

Type representing algebraic operators when solving parametric differential problems
"""
abstract type ParamOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: NonlinearParamOperator end

"""
    const JointParamOperator{O<:UnEvalOperatorType} = ParamOperator{O,JointDomains}
"""
const JointParamOperator{O<:UnEvalOperatorType} = ParamOperator{O,JointDomains}

"""
    const SplitParamOperator{O<:UnEvalOperatorType} = ParamOperator{O,SplitDomains}
"""
const SplitParamOperator{O<:UnEvalOperatorType} = ParamOperator{O,SplitDomains}

"""
    get_fe_operator(op::ParamOperator) -> ParamFEOperator

Fetches the underlying FE operator of an algebraic operator `op`
"""
get_fe_operator(op::ParamOperator) = @abstractmethod
FESpaces.get_test(op::ParamOperator) = get_test(get_fe_operator(op))
FESpaces.get_trial(op::ParamOperator) = get_trial(get_fe_operator(op))
ODEs.get_res(op::ParamOperator) = ODEs.get_res(get_fe_operator(op))
get_jac(op::ParamOperator) = get_jac(get_fe_operator(op))

DofMaps.get_dof_map(op::ParamOperator) = get_dof_map(get_fe_operator(op))
DofMaps.get_sparse_dof_map(op::ParamOperator) = get_sparse_dof_map(get_fe_operator(op))
get_dof_map_at_domains(op::ParamOperator) = get_dof_map_at_domains(get_fe_operator(op))
get_sparse_dof_map_at_domains(op::ParamOperator) = get_sparse_dof_map_at_domains(get_fe_operator(op))
CellData.get_domains(op::ParamOperator) = get_domains(get_fe_operator(op))

set_domains(op::ParamOperator,args...) = get_algebraic_operator(set_domains(get_fe_operator(op),args...))
change_domains(op::ParamOperator,args...) = get_algebraic_operator(change_domains(get_fe_operator(op),args...))
Utils.get_domains_res(op::ParamOperator) = get_domains_res(get_fe_operator(op))
Utils.get_domains_jac(op::ParamOperator) = get_domains_jac(get_fe_operator(op))

get_param_space(op::ParamOperator) = get_param_space(get_fe_operator(op))
ParamDataStructures.realization(op::ParamOperator;kwargs...) = realization(get_fe_operator(op);kwargs...)

ODEs.get_assembler(op::ParamOperator) = get_assembler(get_fe_operator(op))
get_param_assembler(op::ParamOperator,r::AbstractRealization) = parameterize(get_assembler(op),r)
FESpaces.assemble_matrix(op::ParamOperator,form::Function) = assemble_matrix(get_fe_operator(op),form)

# basic interface

function ParamAlgebra.allocate_paramcache(op::ParamOperator,μ::AbstractRealization)
  feop = get_fe_operator(op)
  ptrial = get_trial(feop)
  trial = evaluate(ptrial,μ)
  ParamCache(trial,ptrial)
end

function Algebra.zero_initial_guess(op::ParamOperator)
  trial = get_trial(op)
  zero_free_values(trial)
end

function Algebra.zero_initial_guess(op::ParamOperator,μ::AbstractRealization)
  x = zero_initial_guess(op)
  parameterize(x,μ)
end

function Algebra.allocate_residual(
  op::JointParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,μ)

  res = get_res(op)
  dc = res(μ,uh,v)
  vecdata = collect_cell_vector(test,dc)
  b = allocate_vector(assem,vecdata)

  b
end

function Algebra.residual!(
  b::AbstractVector,
  op::JointParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,μ)

  res = get_res(op)
  dc = res(μ,uh,v)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  op::JointParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,μ)

  jac = get_jac(op)
  matdata = collect_cell_matrix(trial,test,jac(μ,uh,du,v))
  A = allocate_matrix(assem,matdata)

  A
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  op::JointParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,μ)

  jac = get_jac(op)
  dc = jac(μ,uh,du,v)
  matdata = collect_cell_matrix(trial,test,dc)
  assemble_matrix_add!(A,assem,matdata)

  A
end

function Algebra.allocate_residual(
  op::SplitParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,μ)

  trian_res = get_domains_res(op)
  res = get_res(op)
  dc = res(μ,uh,v)
  contribution(trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end
end

function Algebra.residual!(
  b::Contribution,
  op::SplitParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache;
  add::Bool=false)

  !add && fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,μ)

  trian_res = get_domains_res(op)
  res = get_res(op)
  dc = res(μ,uh,v)

  map(b.values,trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end

  b
end

function Algebra.allocate_jacobian(
  op::SplitParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,μ)

  trian_jac = get_domains_jac(op)
  jac = get_jac(op)
  dc = jac(μ,uh,du,v)
  contribution(trian_jac) do trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    allocate_matrix(assem,matdata)
  end
end

function ODEs.jacobian_add!(
  A::Contribution,
  op::SplitParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,μ)

  trian_jac = get_domains_jac(op)
  jac = get_jac(op)
  dc = jac(μ,uh,du,v)
  map(A.values,trian_jac) do values,trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    assemble_matrix_add!(values,assem,matdata)
  end

  A
end

# nonlinear interface

const LinearNonlinearParamOperator{T<:TriangulationStyle} = ParamOperator{LinearNonlinearParamEq,T}

get_fe_operator(op::LinearNonlinearParamOperator) = get_fe_operator(get_nonlinear_operator(op))
join_operators(op::LinearNonlinearParamOperator) = get_algebraic_operator(join_operators(get_fe_operator(op)))

function ParamAlgebra.allocate_paramcache(op::LinearNonlinearParamOperator,μ::AbstractRealization)
  op_nlin = get_nonlinear_operator(op)
  allocate_paramcache(op_nlin,μ)
end

function ParamAlgebra.allocate_systemcache(op::LinearNonlinearParamOperator,u::AbstractVector)
  op_nlin = get_nonlinear_operator(op)
  allocate_systemcache(op_nlin,u)
end

function ParamAlgebra.update_paramcache!(
  paramcache::AbstractParamCache,
  op::LinearNonlinearParamOperator,
  μ::AbstractRealization)

  op_nlin = get_nonlinear_operator(op)
  update_paramcache!(paramcache,op_nlin,μ)
end

function ParamDataStructures.parameterize(op::LinearNonlinearParamOperator,μ::AbstractRealization)
  op_lin = parameterize(get_linear_operator(op),μ)
  op_nlin = parameterize(get_nonlinear_operator(op),μ)
  syscache_lin = allocate_systemcache(op_lin)
  LinNonlinParamOperator(op_lin,op_nlin,syscache_lin)
end

function Algebra.allocate_residual(
  op::LinearNonlinearParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function Algebra.residual!(
  b,
  op::LinearNonlinearParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function ODEs.jacobian_add!(
  A,
  op::LinearNonlinearParamOperator,
  μ::Realization,
  u::AbstractVector,
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

# constructors

function LinearParamOperator(args...;kwargs...)
  feop = LinearParamFEOperator(args...;kwargs...)
  get_algebraic_operator(feop)
end

function ParamOperator(args...;kwargs...)
  feop = ParamFEOperator(args...;kwargs...)
  get_algebraic_operator(feop)
end

function LinearNonlinearParamOperator(op_lin::ParamOperator,op_nlin::ParamOperator)
  feop_lin = get_fe_operator(op_lin)
  feop_nlin = get_fe_operator(op_nlin)
  feop = LinearNonlinearParamFEOperator(feop_lin,feop_nlin)
  get_algebraic_operator(feop)
end