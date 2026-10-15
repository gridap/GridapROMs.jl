"""
    abstract type ParamFEOperator{O<:UnEvalOperatorType,T<:WeakFormStyle} <: FEOperator end

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
abstract type ParamFEOperator{O<:UnEvalOperatorType,T<:WeakFormStyle} <: FEOperator end

"""
    const PlainParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,PlainForm}
"""
const PlainParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,PlainForm}

"""
    const BundleParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,BundleForm}
"""
const BundleParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,BundleForm}

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
    struct ParamFEOpFromWeakForm{O<:UnEvalOperatorType,T<:WeakFormStyle} <: ParamFEOperator{O,T}
      res::Function
      jac::Function
      pspace::ParamSpace
      assem::Assembler
      trial::FESpace
      test::FESpace
      domains::FEDomains
    end

Most standard instance of a parametric FE operator
"""
struct ParamFEOpFromWeakForm{O<:UnEvalOperatorType,T<:WeakFormStyle} <: ParamFEOperator{O,T}
  res::Union{Function,Bundle}
  jac::Union{Function,Bundle}
  pspace::ParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  domains::Union{FEDomains,Bundle}
end

const PlainParamFEOpFromWeakForm{O<:UnEvalOperatorType} = ParamFEOpFromWeakForm{O,PlainForm}
const BundleParamFEOpFromWeakForm{O<:UnEvalOperatorType} = ParamFEOpFromWeakForm{O,BundleForm}

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  ParamFEOpFromWeakForm{NonlinearParamEq,PlainForm}(
    res,jac,pspace,assem,trial,test,domains)
end

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test,domains::FEDomains)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{NonlinearParamEq,PlainForm}(
    res,jac,pspace,assem,trial,test,domains)
end

"""
    LinearParamFEOperator(rhs::Function,jac::Function,pspace,trial,test,args...
      ) -> ParamFEOperator{LinearParamEq,WeakFormStyle}

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
  ParamFEOpFromWeakForm{LinearParamEq,PlainForm}(
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
  ParamFEOpFromWeakForm{LinearParamEq,PlainForm}(
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
      ) -> ParamFEOperator{NonlinearParamEq,WeakFormStyle}

Returns a parametric FE operator that bundles a linear and a nonlinear part
"""
function ParamFEOperator(res::Bundle,jac::Bundle,pspace,trial,test,domains::Bundle)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{NonlinearParamEq,BundleForm}(
    res,jac,pspace,assem,trial,test,domains)
end

function ParamFEOperator(res::Bundle,jac::Bundle,pspace,trial,test)
  domains = Bundle(FEDomains(),FEDomains())
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{NonlinearParamEq,BundleForm}(
    res,jac,pspace,assem,trial,test,domains)
end

FESpaces.get_test(op::ParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::ParamFEOpFromWeakForm) = op.trial
get_param_space(op::ParamFEOpFromWeakForm) = op.pspace
ODEs.get_res(op::ParamFEOpFromWeakForm) = op.res
get_jac(op::ParamFEOpFromWeakForm) = op.jac
ODEs.get_assembler(op::ParamFEOpFromWeakForm) = op.assem
CellData.get_domains(op::ParamFEOpFromWeakForm) = op.domains

"""
    get_linear_part(op::ParamFEOpFromWeakForm{NonlinearParamEq,BundleForm}) -> ParamFEOpFromWeakForm{LinearParamEq,BundleForm}
    get_nonlinear_part(op::ParamFEOpFromWeakForm{NonlinearParamEq,BundleForm}) -> ParamFEOpFromWeakForm{NonlinearParamEq,BundleForm}

Splits a `Bundle`-based linear+nonlinear operator back into its two plainly-tagged
halves
"""
function Utils.get_linear_part(op::ParamFEOpFromWeakForm{NonlinearParamEq,BundleForm})
  ParamFEOpFromWeakForm{LinearParamEq,BundleForm}(
    op.res.linear,op.jac.linear,op.pspace,op.assem,op.trial,op.test,op.domains.linear)
end

function Utils.get_nonlinear_part(op::ParamFEOpFromWeakForm{NonlinearParamEq,BundleForm})
  ParamFEOpFromWeakForm{NonlinearParamEq,BundleForm}(
    op.res.nonlinear,op.jac.nonlinear,op.pspace,op.assem,op.trial,op.test,op.domains.nonlinear)
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
(including the [`WeakFormStyle`](@ref) trait) unchanged
"""
function change_domains(op::ParamFEOpFromWeakForm{O,T},trians_res,trians_jac) where {O,T}
  domains′ = FEDomains(trians_res,trians_jac)
  ParamFEOpFromWeakForm{O,T}(
    op.res,op.jac,op.pspace,op.assem,op.trial,op.test,domains′)
end

# concrete types

"""
    struct GenericParamOperator{O<:UnEvalOperatorType,T<:WeakFormStyle} <: ParamOperator{O,T}
      op::ParamFEOperator{O,T}
    end

Wrapper that transforms a `ParamFEOperator` into an `ParamOperator`
"""
struct GenericParamOperator{O<:UnEvalOperatorType,T<:WeakFormStyle} <: ParamOperator{O,T}
  op::ParamFEOperator{O,T}
end

get_fe_operator(op::GenericParamOperator) = op.op
