"""
    const TransientParamFEOperator{O<:ODEParamOperatorType,T<:TriangulationStyle} = ParamFEOperator{O,T}

Parametric extension of a `TransientFEOperator` in [`Gridap`](@ref). Compared to
a standard TransientFEOperator, there are the following novelties:

- a [`TransientParamSpace`](@ref) is provided, so that parametric realizations can be extracted
  directly from the `TransientParamFEOperator`
- a function representing a norm matrix is provided, so that errors in the
  desired norm can be automatically computed

Subtypes:

- [`TransientParamFEOpFromWeakForm`](@ref)
- [`TransientParamLinearFEOpFromWeakForm`](@ref)
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

function ParamSteady.get_sparse_dof_map_at_domains(op::TransientParamFEOperator)
  trial = get_trial(op)
  test = get_test(op)
  domains_jacs = get_domains_jac(op)

  map(domains_jacs) do domains_jac
    sparse_dof_map_at_domains = ()
    for trian in domains_jac
      sparse_dof_map = get_sparse_dof_map(trial,test,trian)
      sparse_dof_map_at_domains = (sparse_dof_map_at_domains...,sparse_dof_map)
    end
    Contribution(sparse_dof_map_at_domains,domains_jac)
  end
end

"""
    struct TransientParamFEOpFromWeakForm{T} <: TransientParamFEOperator{NonlinearParamODE,T}
      res::Function
      jacs::Tuple{Vararg{Function}}
      tpspace::TransientParamSpace
      assem::Assembler
      trial::FESpace
      test::FESpace
      domains::FEDomains
      order::Integer
    end

Instance of [`TransientParamFEOperator`](@ref), to be used when the transient problem is
nonlinear
"""
struct TransientParamFEOpFromWeakForm{T} <: TransientParamFEOperator{NonlinearParamODE,T}
  res::Function
  jacs::Tuple{Vararg{Function}}
  tpspace::TransientParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  domains::FEDomains
  order::Integer
end

const JointTransientParamFEOpFromWeakForm = TransientParamFEOpFromWeakForm{JointDomains}
const SplitTransientParamFEOpFromWeakForm = TransientParamFEOpFromWeakForm{SplitDomains}

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test)

  order = length(jacs) - 1
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  TransientParamFEOpFromWeakForm{JointDomains}(
    res,jacs,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test,domains::FEDomains)

  order = length(jacs) - 1
  res′,jacs′ = set_domains(res,jacs,test,trial,domains)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamFEOpFromWeakForm{SplitDomains}(
    res′,jacs′,tpspace,assem,trial,test,domains,order)
end

function TransientParamFEOperator(
  res::Function,jacs::Tuple{Vararg{Function}},tpspace,trial,test,trians...)

  domains = FEDomains(trians...)
  TransientParamFEOperator(res,jacs,tpspace,trial,test,domains)
end

function TransientParamFEOperator(
  res::Function,jac::Function,tpspace,trial,test,args...)

  TransientParamFEOperator(res,(jac,),tpspace,trial,test,args...)
end

function TransientParamFEOperator(
  res::Function,jac::Function,jac_t::Function,tpspace,trial,test,args...)

  TransientParamFEOperator(res,(jac,jac_t),tpspace,trial,test,args...)
end

function TransientParamFEOperator(
  res::Function,tpspace,trial,test,args...;order::Integer=1)

  function jac_0(μ,t,u,du,v,args...)
    function res_0(y)
      u0 = TransientCellField(y,u.derivatives)
      res(μ,t,u0,v,args...)
    end
    jacobian(res_0,u.cellfield)
  end
  jacs = (jac_0,)

  for k in 1:order
    function jac_k(μ,t,u,duk,v,args...)
      function res_k(y)
        derivatives = (u.derivatives[1:k-1]...,y,u.derivatives[k+1:end]...)
        uk = TransientCellField(u.cellfield,derivatives)
        res(μ,t,uk,v,args...)
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
ParamSteady.get_param_space(op::TransientParamFEOpFromWeakForm) = op.tpspace
CellData.get_domains(op::TransientParamFEOpFromWeakForm) = op.domains

"""
    struct TransientParamLinearFEOpFromWeakForm{T} <: TransientParamFEOperator{LinearParamODE,T}
      res::Function
      jacs::Tuple{Vararg{Function}}
      constant_forms::Tuple{Vararg{Bool}}
      tpspace::TransientParamSpace
      assem::Assembler
      trial::FESpace
      test::FESpace
      domains::FEDomains
      order::Integer
    end

Instance of [`TransientParamFEOperator`](@ref), to be used when the transient problem is
linear
"""
struct TransientParamLinearFEOpFromWeakForm{T} <: TransientParamFEOperator{LinearParamODE,T}
  res::Function
  jacs::Tuple{Vararg{Function}}
  constant_forms::Tuple{Vararg{Bool}}
  tpspace::TransientParamSpace
  assem::Assembler
  trial::FESpace
  test::FESpace
  domains::FEDomains
  order::Integer
end

const JointTransientParamLinearFEOpFromWeakForm = TransientParamLinearFEOpFromWeakForm{JointDomains}

"""
  TransientParamLinearFEOperator(forms::Tuple{Vararg{Function}},res::Function,
    tpspace,trial,test;kwargs...) -> TransientParamLinearFEOpFromWeakForm{TriangulationStyle}

Returns a linear parametric FE operator
"""
function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,tpspace,trial,test;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms)))

  order = length(forms)-1
  jacs = ntuple(k -> ((μ,t,u,duk,v) -> forms[k](μ,t,duk,v)),length(forms))
  assem = SparseMatrixAssembler(trial,test)
  domains = FEDomains()
  TransientParamLinearFEOpFromWeakForm{JointDomains}(
    res,jacs,constant_forms,tpspace,assem,trial,test,domains,order)
end

const SplitTransientParamLinearFEOpFromWeakForm = TransientParamLinearFEOpFromWeakForm{SplitDomains}

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,tpspace,trial,test,domains::FEDomains;
  constant_forms::Tuple{Vararg{Bool}}=ntuple(_ -> false,length(forms)))

  order = length(forms) - 1
  jacs = ntuple(k -> ((μ,t,u,duk,v,args...) -> forms[k](μ,t,duk,v,args...)),length(forms))
  res′,jacs′ = set_domains(res,jacs,test,trial,domains)
  assem = SparseMatrixAssembler(trial,test)
  TransientParamLinearFEOpFromWeakForm{SplitDomains}(
    res′,jacs′,constant_forms,tpspace,assem,trial,test,domains,order)
end

function TransientParamLinearFEOperator(
  forms::Tuple{Vararg{Function}},res::Function,tpspace,trial,test,trians...;kwargs...)

  domains = FEDomains(trians...)
  TransientParamLinearFEOperator(forms,res,tpspace,trial,test,domains;kwargs...)
end

function TransientParamLinearFEOperator(
  mass::Function,res::Function,tpspace,trial,test,args...;kwargs...)

  TransientParamLinearFEOperator((mass,),res,tpspace,trial,test,args...;kwargs...)
end

function TransientParamLinearFEOperator(
  stiffness::Function,mass::Function,res::Function,tpspace,trial,test,args...;kwargs...)

  TransientParamLinearFEOperator((stiffness,mass),res,tpspace,trial,test,args...;kwargs...)
end

function TransientParamLinearFEOperator(
  stiffness::Function,damping::Function,mass::Function,res::Function,
  tpspace,trial,test,args...;kwargs...)

  TransientParamLinearFEOperator((stiffness,damping,mass),res,tpspace,trial,test,args...;kwargs...)
end

FESpaces.get_test(op::TransientParamLinearFEOpFromWeakForm) = op.test
FESpaces.get_trial(op::TransientParamLinearFEOpFromWeakForm) = op.trial
get_order(op::TransientParamLinearFEOpFromWeakForm) = op.order
ODEs.get_res(op::TransientParamLinearFEOpFromWeakForm) = op.res
ODEs.get_jacs(op::TransientParamLinearFEOpFromWeakForm) = op.jacs
ODEs.get_assembler(op::TransientParamLinearFEOpFromWeakForm) = op.assem
ODEs.is_form_constant(op::TransientParamLinearFEOpFromWeakForm,k::Integer) = op.constant_forms[k]
ParamSteady.get_param_space(op::TransientParamLinearFEOpFromWeakForm) = op.tpspace
CellData.get_domains(op::TransientParamLinearFEOpFromWeakForm) = op.domains

# triangulation utils

for f in (:set_domains,:change_domains)
  T = f == :set_domains ? :JointDomains : :SplitDomains
  @eval begin
    function $f(op::SplitTransientParamFEOpFromWeakForm,trian_res,trian_jacs)
      trian_res′ = order_domains(get_domains_res(op),trian_res)
      trian_jacs′ = map(order_domains,get_domains_jac(op),trian_jacs)
      res′,jacs′ = set_domains(op.res,op.jacs,op.test,op.trial,trian_res′,trian_jacs′)
      domains′ = FEDomains(trian_res′,trian_jacs′)
      TransientParamFEOpFromWeakForm{$T}(
        res′,jacs′,op.tpspace,op.assem,op.trial,op.test,domains′,op.order)
    end

    function $f(op::SplitTransientParamLinearFEOpFromWeakForm,trian_res,trian_jacs)
      trian_res′ = order_domains(get_domains_res(op),trian_res)
      trian_jacs′ = map(order_domains,get_domains_jac(op),trian_jacs)
      res′,jacs′ = set_domains(op.res,op.jacs,op.test,op.trial,trian_res′,trian_jacs′)
      domains′ = FEDomains(trian_res′,trian_jacs′)
      TransientParamLinearFEOpFromWeakForm{$T}(
        res′,jacs′,op.constant_forms,op.tpspace,op.assem,op.trial,op.test,domains′,op.order)
    end
  end
end

function _set_domain_jac(
  jac::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newjac(μ,t,u,du,v,args...) = jac(μ,t,u,du,v,args...)
  newjac(μ,t,u,du,v) = newjac(μ,t,u,du,v,meas...)
  return newjac
end

function _set_domain_jacs(
  jacs::Tuple{Vararg{Function}},
  trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}},
  order)

  newjacs = ()
  for (jac,trian) in zip(jacs,trians)
    newjacs = (newjacs...,_set_domain_jac(jac,trian,order))
  end
  return newjacs
end

function _set_domain_form(
  res::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  newres(μ,t,u,v,args...) = res(μ,t,u,v,args...)
  newres(μ,t,u,v) = newres(μ,t,u,v,meas...)
  return newres
end

function Utils.set_domains(
  res::Function,
  jacs::Tuple{Vararg{Function}},
  test::FESpace,
  trial::FESpace,
  trian_res::Tuple{Vararg{Triangulation}},
  trian_jacs::Tuple{Vararg{Tuple{Vararg{Triangulation}}}})

  polyn_order = get_polynomial_order(test)
  @check polyn_order == get_polynomial_order(trial)
  res′ = _set_domain_form(res,trian_res,polyn_order)
  jacs′ = _set_domain_jacs(jacs,trian_jacs,polyn_order)
  return res′,jacs′
end

function Utils.set_domains(
  res::Function,
  jacs::Tuple{Vararg{Function}},
  test::FESpace,
  trial::FESpace,
  domains::FEDomains)

  trian_res = get_domains_res(domains)
  trian_jacs = get_domains_jac(domains)
  set_domains(res,jacs,test,trial,trian_res,trian_jacs)
end

function LinearNonlinearTransientParamFEOperator(
  op_lin::TransientParamFEOperator,
  op_nlin::TransientParamFEOperator)

  LinearNonlinearParamFEOperator{LinearNonlinearParamODE}(op_lin,op_nlin)
end

function ODEs.get_res(op::LinearNonlinearParamFEOperator{LinearNonlinearParamODE})
  get_res(get_nonlinear_operator(op))
end

function ODEs.get_jacs(op::LinearNonlinearParamFEOperator{LinearNonlinearParamODE})
  get_jacs(get_nonlinear_operator(op))
end

function get_order(op::LinearNonlinearParamFEOperator{LinearNonlinearParamODE})
  get_order(get_nonlinear_operator(op))
end

function ParamSteady.set_domains(op::LinearNonlinearParamFEOperator{LinearNonlinearParamODE})
  op_lin = set_domains(get_linear_operator(op))
  op_nlin = set_domains(get_nonlinear_operator(op))
  LinearNonlinearTransientParamFEOperator(op_lin,op_nlin)
end

function ParamSteady.join_operators(
  op_lin::TransientParamFEOperator,
  op_nlin::TransientParamFEOperator)

  op_lin = set_domains(op_lin)
  op_nlin = set_domains(op_nlin)

  @check get_trial(op_lin) == get_trial(op_nlin)
  @check get_test(op_lin) == get_test(op_nlin)
  @check op_lin.tpspace === op_nlin.tpspace

  trial = get_trial(op_lin)
  test = get_test(op_lin)
  order = max(get_order(op_lin),get_order(op_nlin))

  res(μ,t,u,v) = get_res(op_lin)(μ,t,u,v) + get_res(op_nlin)(μ,t,u,v)

  order_lin = get_order(op_lin)
  order_nlin = get_order(op_nlin)

  jacs = ()
  for i = 1:order+1
    function jac_i(μ,t,u,du,v)
      if i <= order_lin+1 && i <= order_nlin+1
        get_jacs(op_lin)[i](μ,t,u,du,v) + get_jacs(op_nlin)[i](μ,t,u,du,v)
      elseif i <= order_lin+1
        get_jacs(op_lin)[i](μ,t,u,du,v)
      else i <= order_nlin+1
        get_jacs(op_nlin)[i](μ,t,u,du,v)
      end
    end
    jacs = (jacs...,jac_i)
  end

  TransientParamFEOperator(res,jacs,op_lin.tpspace,trial,test)
end
