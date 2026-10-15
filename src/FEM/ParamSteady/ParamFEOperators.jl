"""
    abstract type ParamFEOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: FEOperator end

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
abstract type ParamFEOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: FEOperator end

"""
    const JointParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,JointDomains}
"""
const JointParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,JointDomains}

"""
    const SplitParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,SplitDomains}
"""
const SplitParamFEOperator{O<:UnEvalOperatorType} = ParamFEOperator{O,SplitDomains}

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
    struct ParamFEOpFromWeakForm{O<:UnEvalOperatorType,T<:TriangulationStyle} <: ParamFEOperator{O,T}
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
struct ParamFEOpFromWeakForm{O<:UnEvalOperatorType,T<:TriangulationStyle} <: ParamFEOperator{O,T}
  res::Union{Function,Bundle}
  jac::Union{Function,Bundle}
  pspace::ParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  domains::Union{FEDomains,Bundle}
end

const JointParamFEOpFromWeakForm{O<:UnEvalOperatorType} = ParamFEOpFromWeakForm{O,JointDomains}
const SplitParamFEOpFromWeakForm{O<:UnEvalOperatorType} = ParamFEOpFromWeakForm{O,SplitDomains}

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  ParamFEOpFromWeakForm{NonlinearParamEq,JointDomains}(
    res,jac,pspace,assem,trial,test,domains)
end

function ParamFEOperator(res::Function,jac::Function,pspace,trial,test,domains::FEDomains)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{NonlinearParamEq,SplitDomains}(
    res,jac,pspace,assem,trial,test,domains)
end

"""
    LinearParamFEOperator(rhs::Function,jac::Function,pspace,trial,test,args...
      ) -> ParamFEOperator{LinearParamEq,TriangulationStyle}

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
  ParamFEOpFromWeakForm{LinearParamEq,JointDomains}(
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
  ParamFEOpFromWeakForm{LinearParamEq,SplitDomains}(
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
      ) -> ParamFEOperator{LinearNonlinearParamEq,TriangulationStyle}

Returns a parametric FE operator that bundles a linear and a nonlinear part
"""
function ParamFEOperator(res::Bundle,jac::Bundle,pspace,trial,test,domains::Bundle)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{LinearNonlinearParamEq,SplitDomains}(
    res,jac,pspace,assem,trial,test,domains)
end

function ParamFEOperator(res::Bundle,jac::Bundle,pspace,trial,test)
  domains = Bundle(FEDomains(),FEDomains())
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{LinearNonlinearParamEq,JointDomains}(
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
    get_linear_part(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq}) -> ParamFEOpFromWeakForm{LinearParamEq}
    get_nonlinear_part(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq}) -> ParamFEOpFromWeakForm{NonlinearParamEq}
"""
function Utils.get_linear_part(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq,T}) where T
  ParamFEOpFromWeakForm{LinearParamEq,T}(
    op.res.linear,op.jac.linear,op.pspace,op.assem,op.trial,op.test,op.domains.linear)
end

function Utils.get_nonlinear_part(op::ParamFEOpFromWeakForm{LinearNonlinearParamEq,T}) where T
  ParamFEOpFromWeakForm{NonlinearParamEq,T}(
    op.res.nonlinear,op.jac.nonlinear,op.pspace,op.assem,op.trial,op.test,op.domains.nonlinear)
end

# triangulation utils

"""
    set_domains(op::ParamFEOperator,args...) -> JointParamFEOperator

Fixes the triangulations for residual/jacobian; the resulting operator will have
the trait [`JointDomains`](@ref) activated
"""
set_domains(op::JointParamFEOperator,args...) = op

"""
    change_domains(op::ParamFEOperator,args...) -> ParamFEOperator

Changes the triangulations for residual/jacobian; the resulting operator will have
the same [`TriangulationStyle`](@ref) trait as the one of the input
"""
change_domains(op::JointParamFEOperator,args...) = op

"""
    set_domains(op::SplitParamFEOperator) -> JointParamFEOperator

Retags a [`SplitParamFEOperator`](@ref) as a [`JointParamFEOperator`](@ref),
leaving `res`/`jac` untouched
"""
function set_domains(op::SplitParamFEOpFromWeakForm{O}) where O
  ParamFEOpFromWeakForm{O,JointDomains}(
    op.res,op.jac,op.pspace,op.assem,op.trial,op.test,op.domains)
end

function change_domains(op::SplitParamFEOpFromWeakForm{O},trians_res,trians_jac) where O
  domains′ = FEDomains(trians_res,trians_jac)
  ParamFEOpFromWeakForm{O,SplitDomains}(
    op.res,op.jac,op.pspace,op.assem,op.trial,op.test,domains′)
end

# concrete types

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
