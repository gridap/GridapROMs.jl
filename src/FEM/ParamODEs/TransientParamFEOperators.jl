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
  TransientParamFEOpFromWeakForm{NonlinearParamODE}(
    res,jacs,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test,domains::FEDomains
  )

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOpFromWeakForm{NonlinearParamODE}(
    res,jacs,tpspace,assem,trial,test,domains,order)
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
  TransientParamFEOpFromWeakForm{LinearNonlinearParamODE}(
    res,jacs,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(res::Bundle,jacs::Tuple{Vararg{Bundle}},tpspace,trial,test)
  domains = Bundle(FEDomains(),FEDomains())
  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOpFromWeakForm{LinearNonlinearParamODE}(
    res,jacs,tpspace,assem,trial,test,domains,order)
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

function ODEs.get_res(op::TransientParamFEOpFromWeakForm{LinearParamODE})
  jacs = op.jacs
  order = op.order
  function res(μ,t,u,v)
    ∂tku = u
    dc = jacs[1](μ,t,∂tku,v)
    for k in 1:order
      ∂tku = ∂t(∂tku)
      dc = dc + jacs[k+1](μ,t,∂tku,v)
    end
    dc - op.res(μ,t,v)
  end
  res
end

function ODEs.get_res(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE})
  res_lin = get_res(get_linear_part(op))
  res_nlin = get_res(get_nonlinear_part(op))
  (μ,t,u,v) -> res_lin(μ,t,u,v) + res_nlin(μ,t,u,v)
end

function ODEs.get_jacs(op::TransientParamFEOpFromWeakForm{LinearParamODE})
  ntuple(k -> (μ,t,u,du,v) -> op.jacs[k](μ,t,du,v),op.order+1)
end

function ODEs.get_jacs(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE})
  jac_lin = get_jacs(get_linear_part(op))
  jac_nlin = get_jacs(get_nonlinear_part(op))
  ntuple(k -> (μ,t,u,du,v) -> jac_lin[k](μ,t,du,v) + jac_nlin[k](μ,t,du,v),op.order+1)
end

ODEs.get_assembler(op::TransientParamFEOpFromWeakForm) = op.assem
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
function TransientLinearParamFEOperator(res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test)
  order = length(forms)-1
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  TransientParamFEOpFromWeakForm{LinearParamODE}(
    res,forms,tpspace,assem,trial,test,domains,order)
end

function TransientLinearParamFEOperator(res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test,domains::FEDomains)
  order = length(forms) - 1
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOpFromWeakForm{LinearParamODE}(
    res,forms,tpspace,assem,trial,test,domains,order)
end

function TransientLinearParamFEOperator(res::Function,forms::Tuple{Vararg{Function}},tpspace,trial,test,trians...;kwargs...)
  domains = FEDomains(trians...)
  TransientLinearParamFEOperator(res,forms,tpspace,trial,test,domains;kwargs...)
end

function TransientLinearParamFEOperator(res::Function,mass::Function,tpspace,trial,test,args...;kwargs...)
  TransientLinearParamFEOperator(res,(mass,),tpspace,trial,test,args...;kwargs...)
end

function TransientLinearParamFEOperator(res::Function,stiffness::Function,mass::Function,tpspace,trial,test,args...;kwargs...)
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
    op.res,op.jacs,op.tpspace,op.assem,op.trial,op.test,domains′,op.order)
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
    res,jacs,op.tpspace,op.assem,op.trial,op.test,domains,_find_order(jacs))
end

function Utils.get_nonlinear_part(op::TransientParamFEOpFromWeakForm{LinearNonlinearParamODE})
  res = get_nonlinear_part(op.res)
  jacs = map(get_nonlinear_part,op.jacs)
  domains = get_nonlinear_part(op.domains)
  TransientParamFEOpFromWeakForm{NonlinearParamODE}(
    res,jacs,op.tpspace,op.assem,op.trial,op.test,domains,_find_order(jacs))
end

# utils

function _find_order(jacs::Tuple)
  o = findlast(!isnothing,jacs)
  isnothing(o) ? 0 : o-1
end