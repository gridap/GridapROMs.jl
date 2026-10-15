"""
    const TransientParamFEOperator{O<:ODEParamOperatorType,T<:WeakFormStyle} = ParamFEOperator{O,T}

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
const TransientParamFEOperator{O<:ODEParamOperatorType,T<:WeakFormStyle} = ParamFEOperator{O,T}

"""
    const PlainTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,PlainForm}
"""
const PlainTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,PlainForm}

"""
    const BundleTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,BundleForm}
"""
const BundleTransientParamFEOperator{O<:ODEParamOperatorType} = TransientParamFEOperator{O,BundleForm}

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
      jacs::Tuple{Vararg{Union{Function,Bundle}}}
      constant_forms::Tuple{Vararg{Bool}}
      tpspace::TransientParamSpace
      assem::Assembler
      trial::FESpace
      test::FESpace
      domains::Union{FEDomains,Bundle}
      order::Integer
    end

Instance of [`TransientParamFEOperator`](@ref), used for nonlinear (`O=NonlinearParamODE`),
linear (`O=LinearParamODE`), and linear+nonlinear split transient problems alike. `jacs`
is always a tuple of length `order+1`; each entry is either a plain weak-form function
(that order's jacobian has no nonlinear part, e.g. a mass term) or a [`Bundle`](@ref) of
a linear and a nonlinear half (that order's jacobian is itself split)
"""
struct TransientParamFEOpFromWeakForm{O<:ODEParamOperatorType,T} <: TransientParamFEOperator{O,T}
  res::Union{Function,Bundle}
  jacs::Tuple{Vararg{Union{Function,Bundle}}}
  constant_forms::Tuple{Vararg{Bool}}
  tpspace::TransientParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  domains::Union{FEDomains,Bundle}
  order::Integer
end

const PlainTransientParamFEOpFromWeakForm{O<:ODEParamOperatorType} = TransientParamFEOpFromWeakForm{O,PlainForm}
const BundleTransientParamFEOpFromWeakForm{O<:ODEParamOperatorType} = TransientParamFEOpFromWeakForm{O,BundleForm}

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test
  )

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{NonlinearParamODE,PlainForm}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test,domains::FEDomains
  )

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{NonlinearParamODE,PlainForm}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

"""
    TransientParamFEOperator(res::Bundle,jacs::Tuple{Vararg{Union{Function,Bundle}}},tpspace,trial,test,domains::Bundle=Bundle(FEDomains(),FEDomains())
      ) -> TransientParamFEOperator{NonlinearParamODE,BundleForm}

Returns a transient parametric FE operator that bundles a linear and a nonlinear part.
`jacs` is a plain tuple; individual entries may themselves be a [`Bundle`](@ref) (see
[`TransientParamFEOpFromWeakForm`](@ref))
"""
function TransientParamFEOperator(
  res::Bundle,jacs::Tuple{Vararg{Union{Function,Bundle}}},tpspace,trial,test,domains::Bundle
  )

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{NonlinearParamODE,BundleForm}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(res::Bundle,jacs::Tuple{Vararg{Union{Function,Bundle}}},tpspace,trial,test)
  domains = Bundle(FEDomains(),FEDomains())
  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{NonlinearParamODE,BundleForm}(
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

"""
    wrap_bilinear_jacs(forms::Tuple{Vararg{Function}}) -> Tuple{Vararg{Function}}

Wraps a tuple of bilinear forms `forms[k](μ,t,duk,v)` (one per temporal derivative
order, Gridap's own convention for a linear operator) into the 5-argument
`jacs[k](μ,t,u,duk,v)` convention expected by [`TransientParamFEOpFromWeakForm`](@ref)
(the extra `u` argument is ignored, since a bilinear form does not depend on the
current state)
"""
wrap_bilinear_jacs(forms::Tuple{Vararg{Function}}) = ntuple(k -> ((μ,t,u,duk,v) -> forms[k](μ,t,duk,v)),length(forms))

"""
  TransientLinearParamFEOperator(res::Function,forms::Tuple{Vararg{Function}},
    tpspace,trial,test;kwargs...) -> TransientLinearParamFEOpFromWeakForm{WeakFormStyle}

Returns a linear parametric FE operator
"""
function TransientLinearParamFEOperator(
  res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms))
  )

  order = length(forms)-1
  jacs = wrap_bilinear_jacs(forms)
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  TransientParamFEOpFromWeakForm{LinearParamODE,PlainForm}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientLinearParamFEOperator(
  res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test,domains::FEDomains;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms))
  )

  order = length(forms) - 1
  jacs = wrap_bilinear_jacs(forms)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOpFromWeakForm{LinearParamODE,PlainForm}(
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

function ParamSteady.change_domains(op::TransientParamFEOpFromWeakForm{O,T},trian_res,trian_jacs) where {O,T}
  domains′ = FEDomains(trian_res,trian_jacs)
  TransientParamFEOpFromWeakForm{O,T}(
    op.res,op.jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,domains′,op.order)
end

"""
    get_linear_part(op::TransientParamFEOpFromWeakForm{NonlinearParamODE,BundleForm}) -> TransientParamFEOpFromWeakForm{LinearParamODE,BundleForm}
    get_nonlinear_part(op::TransientParamFEOpFromWeakForm{NonlinearParamODE,BundleForm}) -> TransientParamFEOpFromWeakForm{NonlinearParamODE,BundleForm}

`jacs` is a tuple whose entries may individually be a plain `Function` (that order
has no nonlinear part) or a [`Bundle`](@ref) (that order is itself split): a plain
entry is reused as-is for the linear half, and replaced with a zero contribution
(of the same shape) for the nonlinear half
"""
function Utils.get_linear_part(op::TransientParamFEOpFromWeakForm{NonlinearParamODE,BundleForm})
  res = get_linear_part(op.res)
  jacs = map(get_linear_part,op.jacs)
  domains = get_linear_part(op.domains)
  TransientParamFEOpFromWeakForm{LinearParamODE,PlainForm}(
    res,jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,domains,length(jacs)-1)
end

function Utils.get_nonlinear_part(op::TransientParamFEOpFromWeakForm{NonlinearParamODE,BundleForm})
  res = get_nonlinear_part(op.res)
  jacs = map(get_nonlinear_part,op.jacs)
  domains = get_nonlinear_part(op.domains)
  TransientParamFEOpFromWeakForm{NonlinearParamODE,PlainForm}(
    res,jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,domains,length(jacs)-1)
end

_zero_jac(jac::Function) = (args...) -> 0*jac(args...)