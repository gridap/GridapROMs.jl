"""
    const TransientParamFEOperator{O<:ODEParamOperatorType} = ParamFEOperator{O}

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
const TransientParamFEOperator{O<:ODEParamOperatorType} = ParamFEOperator{O}

function FESpaces.get_algebraic_operator(op::TransientParamFEOperator)
  GenericParamOperator(op)
end

ODEs.get_res(op::TransientParamFEOperator) = @abstractmethod

ODEs.get_jacs(op::TransientParamFEOperator) = @abstractmethod

function get_order(op::TransientParamFEOperator)
  @abstractmethod
end

"""
    struct TransientParamFEOpFromWeakForm{O<:ODEParamOperatorType} <: TransientParamFEOperator{O}
      res::Union{Function,Bundle}
      jacs::Tuple{Vararg{Union{Function,Bundle,Nothing}}}
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
`res`/`domains` a [`Bundle`](@ref)) transient problems alike. For a split operator,
`jacs` is always a tuple of [`Bundle`](@ref)s of length `order+1`, one per temporal
derivative order; after splitting into [`get_linear_part`](@ref)/[`get_nonlinear_part`](@ref),
an entry may be `nothing` (that order has no linear, resp. no nonlinear, part, e.g. a
mass term is purely linear)
"""
struct TransientParamFEOpFromWeakForm{O<:ODEParamOperatorType} <: TransientParamFEOperator{O}
  res::Union{Function,Bundle}
  jacs::Tuple{Vararg{Union{Function,Bundle,Nothing}}}
  constant_forms::Tuple{Vararg{Bool}}
  tpspace::TransientParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  domains::Union{FEDomains,Bundle}
  order::Integer
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test
  )

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{NonlinearParamODE}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test,domains::FEDomains
  )

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{NonlinearParamODE}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

"""
    TransientParamFEOperator(res::Bundle,jacs::Tuple{Vararg{Bundle}},tpspace,trial,test,domains::Bundle=Bundle(FEDomains(),FEDomains())
      ) -> TransientParamFEOperator{LinearNonlinearParamODE}

Returns a transient parametric FE operator that bundles a linear and a nonlinear part.
`jacs` is a plain tuple; individual entries may themselves be a [`Bundle`](@ref) (see
[`TransientParamFEOpFromWeakForm`](@ref))
"""
function TransientParamFEOperator(
  res::Bundle,jacs::Tuple{Vararg{Bundle}},tpspace,trial,test,domains::Bundle
  )

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{LinearNonlinearParamODE}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(res::Bundle,jacs::Tuple{Vararg{Bundle}},tpspace,trial,test)
  domains = Bundle(FEDomains(),FEDomains())
  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  constant_forms = ntuple(_ -> false,order+1)
  TransientParamFEOpFromWeakForm{LinearNonlinearParamODE}(
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
    const TransientLinearParamFEOpFromWeakForm = TransientParamFEOpFromWeakForm{LinearParamODE}
"""
const TransientLinearParamFEOpFromWeakForm = TransientParamFEOpFromWeakForm{LinearParamODE}

"""
  TransientLinearParamFEOperator(res::Function,forms::Tuple{Vararg{Function}},
    tpspace,trial,test;kwargs...) -> TransientLinearParamFEOpFromWeakForm

Returns a linear parametric FE operator
"""
function TransientLinearParamFEOperator(
  res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms))
  )

  order = length(forms)-1
  jacs = _wrap_jacobians(forms)
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  TransientParamFEOpFromWeakForm{LinearParamODE}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientLinearParamFEOperator(
  res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test,domains::FEDomains;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms))
  )

  order = length(forms) - 1
  jacs = _wrap_jacobians(forms)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOpFromWeakForm{LinearParamODE}(
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

function ParamSteady.change_domains(op::TransientParamFEOpFromWeakForm{O},trian_res,trian_jacs) where O
  domains′ = FEDomains(trian_res,trian_jacs)
  TransientParamFEOpFromWeakForm{O}(
    op.res,op.jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,domains′,op.order)
end

"""
    get_linear_part(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE}) -> TransientParamFEOpFromWeakForm{LinearParamODE}
    get_nonlinear_part(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE}) -> TransientParamFEOpFromWeakForm{NonlinearParamODE}

`jacs` is a tuple whose entries are each a [`Bundle`](@ref); either half of a given
order's `Bundle` may be `nothing` (that order has no linear, resp. no nonlinear,
part, e.g. a mass term is purely linear). `get_linear_part`/`get_nonlinear_part`
truncate `order` to the last non-`nothing` entry of their respective half, so
that code iterating `1:order+1` never indexes into a trailing `nothing`
"""
function Utils.get_linear_part(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE})
  res = get_linear_part(op.res)
  jacs = map(get_linear_part,op.jacs)
  domains = get_linear_part(op.domains)
  TransientParamFEOpFromWeakForm{LinearParamODE}(
    res,jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,domains,_find_order(jacs))
end

function Utils.get_nonlinear_part(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE})
  res = get_nonlinear_part(op.res)
  jacs = map(get_nonlinear_part,op.jacs)
  domains = get_nonlinear_part(op.domains)
  TransientParamFEOpFromWeakForm{NonlinearParamODE}(
    res,jacs,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,domains,_find_order(jacs))
end

# utils

function _wrap_jacobian(f::Function)
  (μ,t,u,duk,v) -> f(μ,t,duk,v)
end

function _wrap_jacobian(f::Bundle)
  flin,fnlin = f
  if isa(flin,Function) && isa(fnlin,Function)
    (μ,t,u,duk,v) -> flin(μ,t,duk,v) + fnlin(μ,t,u,duk,v)
  elseif isa(flin,Function) && isnothing(fnlin)
    (μ,t,u,duk,v) -> flin(μ,t,duk,v)
  elseif isnothing(flin) && isa(fnlin,Function)
    fnlin
  end
end

function _wrap_jacobians(forms::Tuple)
  ntuple(k -> _wrap_jacobian(forms[k]),length(forms))
end

function _find_order(jacs::Tuple)
  o = findlast(!isnothing,jacs)
  isnothing(o) ? 0 : o-1
end