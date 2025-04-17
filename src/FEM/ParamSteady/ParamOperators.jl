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

"""
    abstract type TriangulationStyle <: GridapType end

Subtypes:

- [`JointDomains`](@ref)
- [`SplitDomains`](@ref)
"""
abstract type TriangulationStyle <: GridapType end

"""
    struct JointDomains <: TriangulationStyle end

Trait for a FE operator indicating that residuals/Jacobiansin this operator
should be computed summing the contributions relative to each triangulation as
occurs in [`Gridap`](@ref)
"""
struct JointDomains <: TriangulationStyle end

"""
    struct SplitDomains <: TriangulationStyle end

Trait for a FE operator indicating that residuals/Jacobiansin this operator
should be computed keeping the contributions relative to each triangulation separate
"""
struct SplitDomains <: TriangulationStyle end

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
DofMaps.get_internal_dof_map(op::ParamOperator) = get_internal_dof_map(get_fe_operator(op))
DofMaps.get_sparse_dof_map(op::ParamOperator) = get_sparse_dof_map(get_fe_operator(op))
get_dof_map_at_domains(op::ParamOperator) = get_dof_map_at_domains(get_fe_operator(op))
get_sparse_dof_map_at_domains(op::ParamOperator) = get_sparse_dof_map_at_domains(get_fe_operator(op))
CellData.get_domains(op::ParamOperator) = get_domains(get_fe_operator(op))

set_domains(op::ParamOperator,args...) = get_algebraic_operator(set_domains(get_fe_operator(op),args...))
change_domains(op::ParamOperator,args...) = get_algebraic_operator(change_domains(get_fe_operator(op),args...))
get_domains_res(op::ParamOperator) = get_domains_res(get_fe_operator(op))
get_domains_jac(op::ParamOperator) = get_domains_jac(get_fe_operator(op))

get_param_space(op::ParamOperator) = get_param_space(get_fe_operator(op))
ParamDataStructures.realization(op::ParamOperator;kwargs...) = realization(get_fe_operator(op);kwargs...)

get_param_assembler(op::ParamOperator,r::AbstractRealization) = get_param_assembler(get_fe_operator(op),r)
FESpaces.assemble_matrix(op::ParamOperator,form::Function) = assemble_matrix(get_fe_operator(op),form)

function ParamAlgebra.allocate_paramcache(op::ParamOperator,μ::AbstractRealization)
  feop = get_fe_operator(op)
  ptrial = get_trial(feop)
  trial = evaluate(ptrial,μ)
  ParamCache(trial,ptrial)
end

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
