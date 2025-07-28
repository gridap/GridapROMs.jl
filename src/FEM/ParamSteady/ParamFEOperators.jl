"""
    abstract type ParamFEOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: FEOperator end

Parametric extension of a `FEOperator` in [`Gridap`](@ref). Compared to
a standard FEOperator, there are the following novelties:

- a ParamSpace is provided, so that parametric realizations can be extracted
  directly from the ParamFEOperator
- a function representing a norm matrix is provided, so that errors in the
  desired norm can be automatically computed

Subtypes:

- [`ParamFEOpFromWeakForm`](@ref)
- [`LinearNonlinearParamFEOperator`](@ref)
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

ParamDataStructures.realization(op::ParamFEOperator;kwargs...) = realization(get_param_space(op);kwargs...)

get_param_assembler(op::ParamFEOperator,r::AbstractRealization) = parameterize(get_assembler(op),r)

CellData.get_domains(op::ParamFEOperator) = @abstractmethod
Utils.get_domains_res(op::ParamFEOperator) = get_domains_res(get_domains(op))
Utils.get_domains_jac(op::ParamFEOperator) = get_domains_jac(get_domains(op))

DofMaps.get_dof_map(op::ParamFEOperator) = get_dof_map(get_trial(op))
DofMaps.get_sparse_dof_map(op::ParamFEOperator) = get_sparse_dof_map(get_trial(op),get_test(op))

"""
    get_dof_map_at_domains(op::ParamFEOperator) -> Contribution

Returns the residual dof map restricted to every residual triangulation
"""
function get_dof_map_at_domains(op::ParamFEOperator)
  test = get_test(op)
  dof_map = get_dof_map(test)
  domains_res = get_domains_res(op)
  dof_map_at_domains = tfill(dof_map,Val(length(domains_res)))
  Contribution(dof_map_at_domains,domains_res)
end

"""
    get_sparse_dof_map_at_domains(op::ParamFEOperator) -> Contribution

Returns the Jacobian dof map restricted to every Jacobian triangulation
"""
function get_sparse_dof_map_at_domains(op::ParamFEOperator)
  trial = get_trial(op)
  test = get_test(op)
  domains_jac = get_domains_jac(op)

  sparse_dof_map_at_domains = ()
  for trian in domains_jac
    sparse_dof_map = get_sparse_dof_map(trial,test,trian)
    sparse_dof_map_at_domains = (sparse_dof_map_at_domains...,sparse_dof_map)
  end

  Contribution(sparse_dof_map_at_domains,domains_jac)
end

# used to build a (norm) matrix directly from the FE operator, instead of
# unpacking the trial and test spaces
function FESpaces.assemble_matrix(op::ParamFEOperator,form::Function)
  test = get_test(op)
  _assemble_matrix(form,test)
end

function _assemble_matrix(f,V::SingleFieldFESpace)
  V′ = get_configuration_space(V)
  assemble_matrix(f,V′,V′)
end

function _assemble_matrix(f,V::TProductFESpace)
  a = TProductSparseMatrixAssembler(V,V)
  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(V)
  assemble_matrix(a,collect_cell_matrix(V,V,f(u,v)))
end

function _assemble_matrix(f,V::MultiFieldFESpace)
  _assemble_matrix(f,V.spaces)
end

function _assemble_matrix(f,spaces::Vector{<:SingleFieldFESpace})
  spaces′ = map(get_configuration_space,spaces)
  V = MultiFieldFESpace(spaces′,style=BlockMultiFieldStyle())
  assemble_matrix(f,V,V)
end

function _assemble_matrix(f,spaces::Vector{<:TProductFESpace})
  V = MultiFieldFESpace(spaces,style=BlockMultiFieldStyle())
  a = TProductBlockSparseMatrixAssembler(V,V)
  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(V)
  assemble_matrix(a,collect_cell_matrix(V,V,f(u,v)))
end

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
  res::Function
  jac::Function
  pspace::ParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  domains::FEDomains
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
  res′,jac′ = _set_domains(res,jac,test,trial,domains)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{NonlinearParamEq,SplitDomains}(
    res′,jac′,pspace,assem,trial,test,domains)
end

"""
    LinearParamFEOperator(res::Function,jac::Function,pspace,trial,test,args...
      ) -> ParamFEOperator{LinearParamEq,TriangulationStyle}

Returns a linear parametric FE operator
"""
function LinearParamFEOperator(res::Function,jac::Function,pspace,trial,test)
  jac′(μ,u,du,v) = jac(μ,du,v)
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  ParamFEOpFromWeakForm{LinearParamEq,JointDomains}(
    res,jac′,pspace,assem,trial,test,domains)
end

function LinearParamFEOperator(res::Function,jac::Function,pspace,trial,test,domains::FEDomains)
  jac′(μ,u,du,v,args...) = jac(μ,du,v,args...)
  res′,jac′′ = _set_domains(res,jac′,test,trial,domains)
  assem = SparseMatrixAssembler(trial,test)
  ParamFEOpFromWeakForm{LinearParamEq,SplitDomains}(
    res′,jac′′,pspace,assem,trial,test,domains)
end

for f in (:ParamFEOperator,:LinearParamFEOperator)
  @eval begin
    function $f(res::Function,jac::Function,pspace,trial,test,trians...)
      domains = FEDomains(trians...)
      $f(res,jac,pspace,trial,test,domains)
    end
  end
end

FESpaces.get_test(op::ParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::ParamFEOpFromWeakForm) = op.trial
get_param_space(op::ParamFEOpFromWeakForm) = op.pspace
ODEs.get_res(op::ParamFEOpFromWeakForm) = op.res
Utils.get_jac(op::ParamFEOpFromWeakForm) = op.jac
ODEs.get_assembler(op::ParamFEOpFromWeakForm) = op.assem
CellData.get_domains(op::ParamFEOpFromWeakForm) = op.domains

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

for f in (:set_domains,:change_domains)
  T = f == :set_domains ? :JointDomains : :SplitDomains
  @eval begin
    function $f(op::SplitParamFEOperator)
      $f(op,get_domains(op))
    end

    function $f(op::SplitParamFEOperator,domains::FEDomains)
      $f(op,get_domains_res(domains),get_domains_jac(domains))
    end

    function $f(op::SplitParamFEOpFromWeakForm{O},trians_res,trians_jac) where O
      trian_res′ = order_domains(get_domains_res(op),trians_res)
      trian_jac′ = order_domains(get_domains_jac(op),trians_jac)
      res′,jac′ = _set_domains(op.res,op.jac,op.trial,op.test,trian_res′,trian_jac′)
      domains′ = FEDomains(trian_res′,trian_jac′)
      ParamFEOpFromWeakForm{O,$T}(
        res′,jac′,op.pspace,op.assem,op.trial,op.test,domains′)
    end
  end
end

function _set_domains(
  res::Function,
  jac::Function,
  test::FESpace,
  trial::FESpace,
  trian_res::Tuple{Vararg{Triangulation}},
  trian_jac::Tuple{Vararg{Triangulation}})

  polyn_order = get_polynomial_order(test)
  @check polyn_order == get_polynomial_order(trial)
  res′ = _set_domain_param_res(res,trian_res,polyn_order)
  jac′ = _set_domain_param_jac(jac,trian_jac,polyn_order)
  return res′,jac′
end

function _set_domains(
  res::Function,
  jac::Function,
  test::FESpace,
  trial::FESpace,
  domains::FEDomains)

  trian_res = get_domains_res(domains)
  trian_jac = get_domains_jac(domains)
  _set_domains(res,jac,test,trial,trian_res,trian_jac)
end

function _set_domain_param_jac(
  jac::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  jac′(μ,u,du,v,args...) = jac(μ,u,du,v,args...)
  jac′(μ,u,du,v) = jac′(μ,u,du,v,meas...)
  return jac′
end

function _set_domain_param_res(
  res::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  res′(μ,u,v,args...) = res(μ,u,v,args...)
  res′(μ,u,v) = res′(μ,u,v,meas...)
  return res′
end
