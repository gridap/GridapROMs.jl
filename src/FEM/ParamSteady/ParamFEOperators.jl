"""
    abstract type ParamFEOperator{O<:UnEvalOperatorType} <: FEOperator end

Parametric extension of a `FEOperator` in [`Gridap`](@ref). Compared to
a standard FEOperator, there are the following novelties:

- a ParamSpace is provided, so that parametric realisations can be extracted
  directly from the ParamFEOperator
- a function representing a norm matrix is provided, so that errors in the
  desired norm can be automatically computed

Subtypes:

- [`ParamFEOpFromWeakForm`](@ref)
- [`TransientParamFEOperator`](@ref)
"""
abstract type ParamFEOperator{O<:UnEvalOperatorType} <: FEOperator end

function FESpaces.get_test(feop::ParamFEOperator)
  @abstractmethod
end

function FESpaces.get_trial(feop::ParamFEOperator)
  @abstractmethod
end

"""
    get_param_space(feop::ParamFEOperator) -> ParamSpace
    get_param_space(feop::TransientParamFEOperator) -> TransientParamSpace

Returns the space of parameters contained in `feop`
"""
function get_param_space(feop::ParamFEOperator)
  @abstractmethod
end

function FESpaces.get_algebraic_operator(op::ParamFEOperator)
  GenericParamOperator(op)
end

ParamDataStructures.realisation(op::ParamFEOperator;kwargs...) = realisation(get_param_space(op);kwargs...)

get_param_assembler(op::ParamFEOperator,r::AbstractRealisation) = parameterise(get_assembler(op),r)

CellData.get_domains(op::ParamFEOperator) = @abstractmethod
Utils.get_domains_res(op::ParamFEOperator) = get_domains_res(get_domains(op))
Utils.get_domains_jac(op::ParamFEOperator) = get_domains_jac(get_domains(op))

DofMaps.get_dof_map(op::ParamFEOperator,args...) = get_dof_map(get_trial(op),args...)
DofMaps.get_sparse_dof_map(op::ParamFEOperator,args...) = get_sparse_dof_map(get_trial(op),get_test(op),args...)

"""
    struct ParamFEOpFromWeakForm{O<:UnEvalOperatorType} <: ParamFEOperator{O}
      res::Union{Function,Bundle}
      jac::Union{Function,Bundle}
      pspace::ParamSpace
      assem::Assembler
      trial::FESpace
      test::FESpace
      domains::Union{FEDomains,Bundle}
    end

Instance of [`ParamFEOperator`](@ref), used for nonlinear (`O=NonlinearParamEq`),
linear (`O=LinearParamEq`), and linear+nonlinear split (`O=LinearNonlinearParamEq`,
`res`/`jac`/`domains` each a [`Bundle`](@ref)) steady problems alike
"""
struct ParamFEOpFromWeakForm{O<:UnEvalOperatorType} <: ParamFEOperator{O}
  res::Union{Function,Bundle}
  jac::Union{Function,Bundle}
  pspace::ParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  domains::Union{FEDomains,Bundle}
end

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  ParamFEOpFromWeakForm{NonlinearParamEq}(
    res,jac,pspace,assem,trial,test,domains)
end

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test,domains::FEDomains)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{NonlinearParamEq}(
    res,jac,pspace,assem,trial,test,domains)
end

"""
    LinearParamFEOperator(rhs::Function,jac::Function,pspace,trial,test,args...
      ) -> ParamFEOperator{LinearParamEq}

Returns a linear parametric FE operator. Unlike [`ParamFEOperator`](@ref), which
takes a residual `res(μ,u,v)`, this takes the right-hand side `rhs(μ,v)` directly:
a genuinely linear problem has no residual to evaluate at a current state `u`,
only a (μ-dependent, but not u-dependent) jacobian and right-hand side. Assembling
`allocate_residual`/`residual!` on the resulting operator therefore just assembles
`rhs(μ,v)`, and solving accounts for the sign accordingly (see [`ParamSolvers`](@ref))
"""
function LinearParamFEOperator(rhs::Function,jac::Function,pspace,trial,test)
  jac′(μ,u,du,v) = jac(μ,du,v)
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  ParamFEOpFromWeakForm{LinearParamEq}(
    rhs,jac′,pspace,assem,trial,test,domains)
end

function LinearParamFEOperator(
  rhs::Function,
  jac::Function,
  pspace,
  trial,
  test,
  domains::FEDomains
  )

  jac′(μ,u,du,v) = jac(μ,du,v)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{LinearParamEq}(
    rhs,jac′,pspace,assem,trial,test,domains)
end

function LinearParamFEOperator(rhs::Function,jac::Function,pspace,trial,test,trians...)
  domains = FEDomains(trians...)
  LinearParamFEOperator(rhs,jac,pspace,trial,test,domains)
end

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test,trians...)
  domains = FEDomains(trians...)
  ParamFEOperator(res,jac,pspace,trial,test,domains)
end

function ParamFEOperator(res::Function,pspace,trial,test,args...)
  jac(μ,u,du,dv) = jacobian(x->res(μ,x,dv),u)
  ParamFEOperator(res,jac,pspace,trial,test,args...)
end

"""
    ParamFEOperator(res::Bundle,jac::Bundle,pspace,trial,test,domains::Bundle=Bundle(FEDomains(),FEDomains())
      ) -> ParamFEOperator{LinearNonlinearParamEq}

Returns a parametric FE operator that bundles a linear and a nonlinear part
"""
function ParamFEOperator(res::Bundle,jac::Bundle,pspace,trial,test,domains::Bundle)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{LinearNonlinearParamEq}(
    res,jac,pspace,assem,trial,test,domains)
end

function ParamFEOperator(res::Bundle,jac::Bundle,pspace,trial,test)
  domains = Bundle(FEDomains(),FEDomains())
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{LinearNonlinearParamEq}(
    res,jac,pspace,assem,trial,test,domains)
end

FESpaces.get_test(op::ParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::ParamFEOpFromWeakForm) = op.trial
get_param_space(op::ParamFEOpFromWeakForm) = op.pspace
ODEs.get_res(op::ParamFEOpFromWeakForm) = op.res
get_jac(op::ParamFEOpFromWeakForm) = op.jac
ODEs.get_assembler(op::ParamFEOpFromWeakForm) = op.assem
CellData.get_domains(op::ParamFEOpFromWeakForm) = op.domains

function ODEs.get_res(op::ParamFEOpFromWeakForm{LinearParamEq})
  (μ,u,v) -> op.jac(μ,u,u,v) - op.res(μ,v)
end

function ODEs.get_res(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq})
  res_lin = get_res(get_linear_part(op))
  res_nlin = get_res(get_nonlinear_part(op))
  (μ,u,v) -> res_lin(μ,u,v) + res_nlin(μ,u,v)
end

function get_jac(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq})
  jac_lin,jac_nlin = op.jac
  (μ,u,du,v) -> jac_lin(μ,du,v) + jac_nlin(μ,u,du,v)
end

"""
    get_linear_part(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq}) -> ParamFEOpFromWeakForm{LinearParamEq}
    get_nonlinear_part(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq}) -> ParamFEOpFromWeakForm{NonlinearParamEq}

Splits a `Bundle`-based linear+nonlinear operator back into its two plainly-tagged
halves
"""
function Utils.get_linear_part(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq})
  res = get_linear_part(op.res)
  jac_bilin = get_linear_part(op.jac)
  jac(μ,u,du,v) = jac_bilin(μ,du,v)
  domains = get_linear_part(op.domains)
  ParamFEOpFromWeakForm{LinearParamEq}(
    res,jac,op.pspace,op.assem,op.trial,op.test,domains)
end

function Utils.get_nonlinear_part(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq})
  res = get_nonlinear_part(op.res)
  jac = get_nonlinear_part(op.jac)
  domains = get_nonlinear_part(op.domains)
  ParamFEOpFromWeakForm{NonlinearParamEq}(
    res,jac,op.pspace,op.assem,op.trial,op.test,domains)
end

# triangulation utils

"""
    set_domains(op::ParamFEOperator,args...) -> ParamFEOperator

A no-op: residual/jacobian assembly no longer depends on whether triangulations
were explicitly tracked, so there is nothing left to "fix"
"""
set_domains(op::ParamFEOperator,args...) = op

"""
    change_domains(op::ParamFEOperator,args...) -> ParamFEOperator

Changes the triangulations for residual/jacobian, keeping everything else
unchanged
"""
function change_domains(op::ParamFEOpFromWeakForm{O},trians_res,trians_jac) where O
  domains′ = FEDomains(trians_res,trians_jac)
  ParamFEOpFromWeakForm{O}(
    op.res,op.jac,op.pspace,op.assem,op.trial,op.test,domains′)
end

# concrete types

"""
    struct GenericParamOperator{O<:UnEvalOperatorType} <: ParamOperator{O}
      op::ParamFEOperator{O}
    end

Wrapper that transforms a `ParamFEOperator` into an `ParamOperator`
"""
struct GenericParamOperator{O<:UnEvalOperatorType} <: ParamOperator{O}
  op::ParamFEOperator{O}
end

get_fe_operator(op::GenericParamOperator) = op.op
