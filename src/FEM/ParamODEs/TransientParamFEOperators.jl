"""
    const TransientParamFEOperator{O<:ODEParamOperatorType,T<:TriangulationStyle} = ParamFEOperator{O,T}

Parametric extension of a `TransientFEOperator` in [`Gridap`](@ref). Compared to
a standard TransientFEOperator, there are the following novelties:

- a [`TransientParamSpace`](@ref) is provided, so that parametric realisations can be extracted
  directly from the `TransientParamFEOperator`
- a function representing a norm matrix is provided, so that errors in the
  desired norm can be automatically computed

Subtypes:

- [`TransientParamFEOpFromWeakForm`](@ref)
- [`TransientLinearParamFEOpFromWeakForm`](@ref)
"""
const TransientParamFEOperator{O<:ODEParamOperatorType,T<:TriangulationStyle} = ParamFEOperator{O,T}

"""
    const JointTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,JointDomains}
"""
const JointTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,JointDomains}

"""
    const SplitTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,SplitDomains}
"""
const SplitTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,SplitDomains}

function FESpaces.get_algebraic_operator(op::TransientParamFEOperator)
  GenericParamOperator(op)
end

ODEs.get_res(op::TransientParamFEOperator) = @abstractmethod

ODEs.get_jacs(op::TransientParamFEOperator) = @abstractmethod

function get_order(op::TransientParamFEOperator)
  @abstractmethod
end

"""
    struct TransientParamFEOpFromWeakForm{O<:ODEParamOperatorType,T} <: TransientParamFEOperator{O,T}
      res::Union{Function,Bundle}
      jacs::Union{Tuple{Vararg{Function}},Bundle}
      constant_forms::Tuple{Vararg{Bool}}
      tpspace::TransientParamSpace
      assem::Assembler
      trial::FESpace
      test::FESpace
      domains::Union{FEDomains,Bundle}
      order::Integer
    end

Instance of [`TransientParamFEOperator`](@ref), used for nonlinear (`O=NonlinearParamODE`),
linear (`O=LinearParamODE`), and linear+nonlinear split (`O=LinearNonlinearParamODE`,
`res`/`jacs`/`domains` each a [`Bundle`](@ref)) transient problems alike
"""
struct TransientParamFEOpFromWeakForm{O<:ODEParamOperatorType,T} <: TransientParamFEOperator{O,T}
  res::Union{Function,Bundle}
  jacs::Union{Tuple{Vararg{Function}},Bundle}
  constant_forms::Tuple{Vararg{Bool}}
  tpspace::TransientParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  domains::Union{FEDomains,Bundle}
  order::Integer
end

const JointTransientParamFEOpFromWeakForm{O<:ODEParamOperatorType} = TransientParamFEOpFromWeakForm{O,JointDomains}
const SplitTransientParamFEOpFromWeakForm{O<:ODEParamOperatorType} = TransientParamFEOpFromWeakForm{O,SplitDomains}

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test
  )

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{NonlinearParamODE,JointDomains}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test,domains::FEDomains
  )

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{NonlinearParamODE,SplitDomains}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

"""
    TransientParamFEOperator(res::Bundle,jacs::Bundle,tpspace,trial,test,domains::Bundle=Bundle(FEDomains(),FEDomains())
      ) -> TransientParamFEOperator{LinearNonlinearParamODE,TriangulationStyle}

Returns a transient parametric FE operator that bundles a linear and a nonlinear part
"""
function TransientParamFEOperator(res::Bundle,jacs::Bundle,tpspace,trial,test,domains::Bundle)
  order = length(jacs.nonlinear) - 1
  assem = SparseMatrixAssembler(trial,test)
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{LinearNonlinearParamODE,SplitDomains}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(res::Bundle,jacs::Bundle,tpspace,trial,test)
  domains = Bundle(FEDomains(),FEDomains())
  order = length(jacs.nonlinear) - 1
  assem = SparseMatrixAssembler(trial,test)
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{LinearNonlinearParamODE,JointDomains}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test,trians...
  )

  domains = FEDomains(trians...)
  TransientParamFEOperator(res,jacs,tpspace,trial,test,domains)
end

function TransientParamFEOperator(
  res::Function,jac::Function,tpspace,trial,test,args...
  )

  TransientParamFEOperator(res,(jac,),tpspace,trial,test,args...)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test,args...
  )

  TransientParamFEOperator(res,(jac,jac_t),tpspace,trial,test,args...)
end

function TransientParamFEOperator(
  res::Function,tpspace,trial,test,args...;order::Integer=1
  )

  function jac_0(μ,t,u,du,v)
    function res_0(y)
      u0 = TransientCellField(y,u.derivatives)
      res(μ,t,u0,v)
    end
    jacobian(res_0,u.cellfield)
  end
  jacs = (jac_0,)

  for k in 1:order
    function jac_k(μ,t,u,duk,v)
      function res_k(y)
        derivatives = (u.derivatives[1:k-1]...,y,u.derivatives[k+1:end]...)
        uk = TransientCellField(u.cellfield,derivatives)
        res(μ,t,uk,v)
      end
      jacobian(res_k,u.derivatives[k])
    end
    jacs = (jacs...,jac_k)
  end

  TransientParamFEOperator(res,jacs,tpspace,trial,test,args...)
end

FESpaces.get_test(op::TransientParamFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamFEOpFromWeakForm) = op.trial
get_order(op::TransientParamFEOpFromWeakForm) = op.order
ODEs.get_res(op::TransientParamFEOpFromWeakForm) = op.res
ODEs.get_jacs(op::TransientParamFEOpFromWeakForm) = op.jacs
ODEs.get_assembler(op::TransientParamFEOpFromWeakForm) = op.assem
ODEs.is_form_constant(op::TransientParamFEOpFromWeakForm,k::Integer) = op.constant_forms[k]
ParamSteady.get_param_space(op::TransientParamFEOpFromWeakForm) = op.tpspace
CellData.get_domains(op::TransientParamFEOpFromWeakForm) = op.domains

"""
    const TransientLinearParamFEOpFromWeakForm{T} = TransientParamFEOpFromWeakForm{LinearParamODE,T}
"""
const TransientLinearParamFEOpFromWeakForm{T} = TransientParamFEOpFromWeakForm{LinearParamODE,T}
const JointTransientLinearParamFEOpFromWeakForm = TransientLinearParamFEOpFromWeakForm{JointDomains}
const SplitTransientLinearParamFEOpFromWeakForm = TransientLinearParamFEOpFromWeakForm{SplitDomains}

"""
  TransientLinearParamFEOperator(res::Function,forms::Tuple{Vararg{Function}},
    tpspace,trial,test;kwargs...) -> TransientLinearParamFEOpFromWeakForm{TriangulationStyle}

Returns a linear parametric FE operator
"""
function TransientLinearParamFEOperator(
  res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms))
  )

  order = length(forms)-1
  jacs = ntuple(k -> ((μ,t,u,duk,v) -> forms[k](μ,t,duk,v)),length(forms))
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  TransientParamFEOpFromWeakForm{LinearParamODE,JointDomains}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientLinearParamFEOperator(
  res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test,domains::FEDomains;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms))
  )

  order = length(forms) - 1
  jacs = ntuple(k -> ((μ,t,u,duk,v) -> forms[k](μ,t,duk,v)),length(forms))
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOpFromWeakForm{LinearParamODE,SplitDomains}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientLinearParamFEOperator(
  res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test,trians...;kwargs...
  )

  domains = FEDomains(trians...)
  TransientLinearParamFEOperator(res,forms,tpspace,trial,test,domains;kwargs...)
end

function TransientLinearParamFEOperator(
  res::Function,mass::Function,tpspace,trial,test,args...;kwargs...
  )

  TransientLinearParamFEOperator(res,(mass,),tpspace,trial,test,args...;kwargs...)
end

function TransientLinearParamFEOperator(
  res::Function,stiffness::Function,mass::Function,tpspace,trial,test,args...;kwargs...
  )

  TransientLinearParamFEOperator(res,(stiffness,mass),tpspace,trial,test,args...;kwargs...)
end

function TransientLinearParamFEOperator(
  res::Function,stiffness::Function,damping::Function,mass::Function,
  tpspace,trial,test,args...;kwargs...
  )

  TransientLinearParamFEOperator(res,(stiffness,damping,mass),tpspace,trial,test,args...;kwargs...)
end

# triangulation utils

function ParamSteady.set_domains(op::SplitTransientParamFEOpFromWeakForm{O}) where O
  TransientParamFEOpFromWeakForm{O,JointDomains}(
    op.res,op.jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,op.domains,op.order)
end

function ParamSteady.change_domains(op::SplitTransientParamFEOpFromWeakForm{O},trian_res,trian_jacs) where O
  domains′ = FEDomains(trian_res,trian_jacs)
  TransientParamFEOpFromWeakForm{O,SplitDomains}(
    op.res,op.jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,domains′,op.order)
end

"""
    get_linear_part(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE}) -> TransientParamFEOpFromWeakForm{LinearParamODE}
    get_nonlinear_part(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE}) -> TransientParamFEOpFromWeakForm{NonlinearParamODE}
"""
function Utils.get_linear_part(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE,T}) where T
  jacs = op.jacs.linear
  TransientParamFEOpFromWeakForm{LinearParamODE,T}(
    op.res.linear,jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,op.domains.linear,length(jacs)-1)
end

function Utils.get_nonlinear_part(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE,T}) where T
  jacs = op.jacs.nonlinear
  TransientParamFEOpFromWeakForm{NonlinearParamODE,T}(
    op.res.nonlinear,jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,op.domains.nonlinear,length(jacs)-1)
end

function ODEs.get_res(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE})
  get_res(get_nonlinear_part(op))
end

function ODEs.get_jacs(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE})
  get_jacs(get_nonlinear_part(op))
end