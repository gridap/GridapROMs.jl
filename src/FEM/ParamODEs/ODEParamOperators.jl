"""
    abstract type ODEParamOperatorType <: UnEvalOperatorType end

Parametric extension of the type `ODEOperatorType` in [`Gridap`](@ref).

Subtypes:

- [`NonlinearParamODE`](@ref)
- [`LinearParamODE`](@ref)
- [`LinearNonlinearParamODE`](@ref)
"""
abstract type ODEParamOperatorType <: UnEvalOperatorType end

"""
    struct NonlinearParamODE <: ODEParamOperatorType end
"""
struct NonlinearParamODE <: ODEParamOperatorType end

"""
    struct LinearParamODE <: ODEParamOperatorType end
"""
struct LinearParamODE <: ODEParamOperatorType end

"""
    struct LinearNonlinearParamODE <: ODEParamOperatorType end
"""
struct LinearNonlinearParamODE <: ODEParamOperatorType end

"""
    const ODEParamOperator{T<:ODEParamOperatorType,T<:TriangulationStyle} = ParamOperator{O,T}

Transient extension of the type [`ParamOperator`](@ref).
"""
const ODEParamOperator{O<:ODEParamOperatorType,T<:TriangulationStyle} = ParamOperator{O,T}

"""
    const JointODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,JointDomains}
"""
const JointODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,JointDomains}

"""
    const SplitODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,SplitDomains}
"""
const SplitODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,SplitDomains}

ODEs.get_jacs(odeop::ODEParamOperator) = get_jacs(get_fe_operator(odeop))
get_order(odeop::ODEParamOperator) = get_order(get_fe_operator(odeop))
ODEs.is_form_constant(odeop::ODEParamOperator,k::Integer) = is_form_constant(get_fe_operator(odeop),k)

# basic interface

function ParamAlgebra.allocate_paramcache(
  odeop::ODEParamOperator,
  r::TransientRealization;
  evaluated=false)

  feop = get_fe_operator(odeop)
  order = get_order(odeop)
  pttrial = get_trial(feop)
  trial = evaluated ? evaluate(pttrial,r) : allocate_space(pttrial,r)
  pttrials = (pttrial,)
  trials = (trial,)
  for k in 1:order
    pttrials = (pttrials...,∂t(pttrials[k]))
    trialk = evaluated ? evaluate(pttrials[k+1],r) : allocate_space(pttrials[k+1],r)
    trials = (trials...,trialk)
  end
  ParamCache(trials,pttrials)
end

function ParamAlgebra.update_paramcache!(
  paramcache::ParamCache,
  odeop::ODEParamOperator,
  r::TransientRealization)

  trials = ()
  for k in 1:get_order(odeop)+1
    trials = (trials...,evaluate!(paramcache.trial[k],paramcache.ptrial[k],r))
  end
  paramcache.trial = trials
  paramcache
end

function ParamAlgebra.allocate_systemcache(
  odeop::ODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache::ParamCache)

  A = allocate_jacobian(odeop,r,us,paramcache)
  b = allocate_residual(odeop,r,us,paramcache)
  return SystemCache(A,b)
end

function Algebra.allocate_residual(
  odeop::JointODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop)
  vecdata = collect_cell_vector(test,res(μ,t,uh,v))
  allocate_vector(assem,vecdata)
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::JointODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop)
  dc = res(μ,t,uh,v)
  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.residual!(
  b::AbstractVector,
  odeop::JointODEParamOperator{LinearParamODE},
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  # Residual
  res = get_res(odeop)
  dc = res(μ,t,uh,v)

  vecdata = collect_cell_vector(test,dc)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  odeop::JointODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop)
  dc = DomainContribution()
  for k in 1:get_order(odeop)+1
    jac = jacs[k]
    dc = dc + jac(μ,t,uh,du,v)
  end
  matdata = collect_cell_matrix(trial,test,dc)
  allocate_matrix(assem,matdata)
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::JointODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop)
  dc = DomainContribution()
  for k in 1:get_order(odeop)+1
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    dc = dc + w * jac(μ,t,uh,du,v)
  end

  if num_domains(dc) > 0
    matdata = collect_cell_matrix(trial,test,dc)
    assemble_matrix_add!(A,assem,matdata)
  end

  A
end

function Algebra.allocate_residual(
  odeop::SplitODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)
  trian_res = get_domains_res(odeop)
  res = get_res(odeop)
  dc = res(μ,t,uh,v)

  b = contribution(trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end
  b
end

function Algebra.residual!(
  b::ArrayContribution,
  odeop::SplitODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  trian_res = get_domains_res(odeop)
  res = get_res(odeop)
  dc = res(μ,t,uh,v)

  map(b.values,trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector_add!(values,assem,vecdata)
  end
  b
end

function Algebra.residual(
  odeop::SplitODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)

  # Residual
  trian_res = get_domains_res(odeop)
  res = get_res(odeop)
  dc = res(μ,t,uh,v)

  contribution(trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector(assem,vecdata)
  end
end

function Algebra.allocate_jacobian(
  odeop::SplitODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)

  trian_jacs = get_domains_jac(odeop)
  jacs = get_jacs(odeop)
  As = ()
  for k in 1:get_order(odeop)+1
    jac = jacs[k]
    trian_jac = trian_jacs[k]
    dc = jac(μ,t,uh,du,v)
    A = contribution(trian_jac) do trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      allocate_matrix(assem,matdata)
    end
    As = (As...,A)
  end
  As
end

function ODEs.jacobian_add!(
  As::TupOfArrayContribution,
  odeop::SplitODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)

  trian_jacs = get_domains_jac(odeop)
  jacs = get_jacs(odeop)
  for k in 1:get_order(odeop)+1
    A = As[k]
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    trian_jac = trian_jacs[k]
    dc = w * jac(μ,t,uh,du,v)
    if num_domains(dc) > 0
      map(A.values,trian_jac) do values,trian
        matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
        assemble_matrix_add!(values,assem,matdata)
      end
    end
  end

  As
end

function Algebra.jacobian(
  odeop::SplitODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = evaluate(get_trial(odeop),nothing)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop)
  v = get_fe_basis(test)

  assem = get_param_assembler(odeop,r)
  μ,t = get_params(r),get_times(r)

  trian_jacs = get_domains_jac(odeop)
  jacs = get_jacs(odeop)
  As = ()
  for k in 1:get_order(odeop)+1
    w = ws[k]
    iszero(w) && continue
    jac = jacs[k]
    trian_jac = trian_jacs[k]
    dc = w * jac(μ,t,uh,du,v)
    A = contribution(trian_jac) do trian
      matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
      assemble_matrix(assem,matdata)
    end
    As = (As...,A)
  end

  As
end

# nonlinear interface

const LinearNonlinearODEParamOperator{T<:TriangulationStyle} = ODEParamOperator{LinearNonlinearParamODE,T}

ParamSteady.get_fe_operator(op::LinearNonlinearODEParamOperator) = get_fe_operator(get_nonlinear_operator(op))
ParamSteady.join_operators(op::LinearNonlinearODEParamOperator) = get_algebraic_operator(join_operators(get_fe_operator(op)))

function ParamAlgebra.allocate_paramcache(op::LinearNonlinearODEParamOperator,r::TransientRealization)
  op_nlin = get_nonlinear_operator(op)
  allocate_paramcache(op_nlin,r)
end

function ParamAlgebra.allocate_systemcache(op::LinearNonlinearODEParamOperator,u::AbstractVector)
  op_nlin = get_nonlinear_operator(op)
  allocate_systemcache(op_nlin,u)
end

function ParamAlgebra.update_paramcache!(
  paramcache::AbstractParamCache,
  op::LinearNonlinearODEParamOperator,
  r::TransientRealization)

  op_nlin = get_nonlinear_operator(op)
  update_paramcache!(paramcache,op_nlin,r)
end

function ParamDataStructures.parameterize(
  op::LinearNonlinearODEParamOperator,
  r::TransientRealization)

  op_lin = parameterize(get_linear_operator(op),r)
  op_nlin = parameterize(get_nonlinear_operator(op),r)
  syscache_lin = allocate_systemcache(op_lin)
  LinNonlinParamOperator(op_lin,op_nlin,syscache_lin)
end

function Algebra.allocate_residual(
  op::LinearNonlinearODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function Algebra.allocate_jacobian(
  op::LinearNonlinearODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function Algebra.residual!(
  b,
  op::LinearNonlinearODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  kwargs...)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

function ODEs.jacobian_add!(
  A,
  op::LinearNonlinearODEParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  @notimplemented "This is inefficient. Instead, assemble the nonlinear system
  by defining a [`LinearNonlinearParamOperator`](@ref)"
end

# constructors

function TransientLinearParamOperator(args...;kwargs...)
  feop = TransientParamLinearFEOperator(args...;kwargs...)
  get_algebraic_operator(feop)
end

function TransientParamOperator(args...;kwargs...)
  feop = TransientParamFEOperator(args...;kwargs...)
  get_algebraic_operator(feop)
end

function LinearNonlinearTransientParamOperator(op_lin::ParamOperator,op_nlin::ParamOperator)
  feop_lin = get_fe_operator(op_lin)
  feop_nlin = get_fe_operator(op_nlin)
  feop = LinearNonlinearTransientParamFEOperator(feop_lin,feop_nlin)
  get_algebraic_operator(feop)
end

const ODEParamNonlinearOperator = GenericParamNonlinearOperator{<:ODEParamOperator}

function ParamAlgebra.allocate_systemcache(nlop::ODEParamNonlinearOperator)
  xh = zero(first(nlop.paramcache.trial))
  x = get_free_dof_values(xh)
  allocate_systemcache(nlop,x)
end

# compute space-time residuals/jacobians (no time marching)

function Algebra.residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  u::AbstractParamVector,
  u0::AbstractParamVector)

  dt,θ = solver.dt,solver.θ
  x = copy(u)
  uθ = copy(u)

  shift!(r,dt*(θ-1))
  shift!(uθ,u0,θ,1-θ)
  shift!(x,u0,1/dt,-1/dt)
  us = (uθ,x)
  b = residual(odeop,r,us)
  shift!(r,dt*(1-θ))

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealization,
  u::AbstractParamVector,
  u0::AbstractParamVector)

  dt,θ = solver.dt,solver.θ
  x = copy(u)
  uθ = copy(u)

  shift!(r,dt*(θ-1))
  shift!(uθ,u0,θ,1-θ)
  shift!(x,u0,1/dt,-1/dt)
  us = (uθ,x)
  ws = (1,1)

  A = jacobian(odeop,r,us,ws)
  shift!(r,dt*(1-θ))

  return A
end

function Algebra.residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  u::AbstractParamVector,
  u0::AbstractParamVector)

  dt,θ = solver.dt,solver.θ
  x = copy(u)
  fill!(x,zero(eltype(x)))
  us = (x,x)

  shift!(r,dt*(θ-1))
  b = residual(odeop,r,us)
  shift!(r,dt*(1-θ))

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealization,
  u::AbstractParamVector,
  u0::AbstractParamVector)

  dt,θ = solver.dt,solver.θ
  ws = (1,1)
  x = copy(u)
  fill!(x,zero(eltype(x)))
  us = (x,x)

  shift!(r,dt*(θ-1))
  A = jacobian(odeop,r,us,ws)
  shift!(r,dt*(1-θ))

  return A
end

# utils

function ParamDataStructures.shift!(
  a::ConsecutiveParamVector,
  a0::ConsecutiveParamVector,
  α::Number,
  β::Number)

  data = get_all_data(a)
  data0 = get_all_data(a0)
  data′ = copy(data)
  np = param_length(a0)
  for ipt = param_eachindex(a)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        data[is,ipt] = α*data[is,ipt] + β*data0[is,ipt]
      end
    else
      for is in axes(data,1)
        data[is,ipt] = α*data[is,ipt] + β*data′[is,ipt-np]
      end
    end
  end
end

function ParamDataStructures.shift!(
  a::BlockParamVector,
  a0::BlockParamVector,
  α::Number,
  β::Number)

  @inbounds for (ai,a0i) in zip(blocks(a),blocks(a0))
    ParamDataStructures.shift!(ai,a0i,α,β)
  end
end
