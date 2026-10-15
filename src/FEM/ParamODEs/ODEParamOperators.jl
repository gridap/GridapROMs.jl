"""
    abstract type ODEParamOperatorType <: UnEvalOperatorType end

Parametric extension of the type `ODEOperatorType` in [`Gridap`](@ref).

Subtypes:

- [`NonlinearParamODE`](@ref)
- [`LinearParamODE`](@ref)
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
    const ODEParamOperator{O<:ODEParamOperatorType,T<:WeakFormStyle} = ParamOperator{O,T}

Transient extension of the type [`ParamOperator`](@ref).
"""
const ODEParamOperator{O<:ODEParamOperatorType,T<:WeakFormStyle} = ParamOperator{O,T}

"""
    const PlainODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,PlainForm}
"""
const PlainODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,PlainForm}

"""
    const BundleODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,BundleForm}
"""
const BundleODEParamOperator{O<:ODEParamOperatorType} = ODEParamOperator{O,BundleForm}

ODEs.get_jacs(odeop::ODEParamOperator) = get_jacs(get_fe_operator(odeop))
get_order(odeop::ODEParamOperator) = get_order(get_fe_operator(odeop))
ODEs.is_form_constant(odeop::ODEParamOperator,k::Integer) = is_form_constant(get_fe_operator(odeop),k)

# basic interface

function ParamAlgebra.allocate_paramcache(
  odeop::ODEParamOperator,
  r::TransientRealisation;
  evaluated=false
  )

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
  r::TransientRealisation
  )

  trials = ()
  for k in 1:get_order(odeop)+1
    trials = (trials...,evaluate!(paramcache.trial[k],paramcache.ptrial[k],r))
  end
  paramcache.trial = trials
  paramcache
end

function ParamAlgebra.allocate_systemcache(
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache::ParamCache
  )

  A = allocate_jacobian(odeop,r,us,paramcache)
  b = allocate_residual(odeop,r,us,paramcache)
  x = similar(first(us))
  return SystemCache(A,b,x)
end

function Algebra.allocate_residual(
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

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
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false
  )

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

"""
    residual!(b,odeop::ODEParamOperator,r,us,paramcache,trian)
    jacobian_add!(A,odeop::ODEParamOperator,r,us,paramcache,k,trian)

Single-triangulation variants of `residual!`/`jacobian_add!`: assemble into `b`/`A`
only the contribution relative to the triangulation `trian` (for `jacobian_add!`,
of the `k`-th temporal derivative's jacobian). Used by `residual_snapshots`/
`jacobian_snapshots` to compute hyper-reduction training data one triangulation at
a time, reusing a single cache
"""
function Algebra.residual!(
  b::AbstractVector,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache,
  trian;
  add::Bool=false
  )

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  !add && fill!(b,zero(eltype(b)))

  μ,t = get_params(r),get_times(r)

  res = get_res(odeop)
  dc = res(μ,t,uh,v)
  vecdata = collect_cell_vector_for_trian(test,dc,trian)
  assemble_vector_add!(b,assem,vecdata)

  b
end

function Algebra.allocate_jacobian(
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = get_trial(odeop)
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
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  )

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = get_trial(odeop)
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

"""
    allocate_jacobians(odeop::ODEParamOperator,r,us,paramcache) -> Tuple
    jacobians_add!(As::Tuple,odeop::ODEParamOperator,r,us,ws,paramcache) -> Tuple

Order-separated variants of `allocate_jacobian`/`jacobian_add!`: instead of summing
the `order+1` temporal-derivative jacobians (weighted by `ws`) into a single
matrix, keep one matrix per order. Used where each order needs its own reduced
basis/hyper-reduction (e.g. [`spacetime_jacobian`](@ref)), unlike the generic
`allocate_jacobian`/`jacobian_add!`, used for the actual time-marching solve, which
only ever needs the `ws`-weighted total
"""
function allocate_jacobians(
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = get_trial(odeop)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop)
  As = ()
  for k in 1:get_order(odeop)+1
    dc = jacs[k](μ,t,uh,du,v)
    matdata = collect_cell_matrix(trial,test,dc)
    As = (As...,allocate_matrix(assem,matdata))
  end
  As
end

function jacobians_add!(
  As::Tuple,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  )

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = get_trial(odeop)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)

  jacs = get_jacs(odeop)
  for k in 1:get_order(odeop)+1
    w = ws[k]
    iszero(w) && continue
    dc = w*jacs[k](μ,t,uh,du,v)
    matdata = collect_cell_matrix(trial,test,dc)
    assemble_matrix_add!(As[k],assem,matdata)
  end
  As
end

function ODEs.jacobian_add!(
  A::AbstractMatrix,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache,
  k::Integer,
  trian
  )

  uh = ODEs._make_uh_from_us(odeop,us,paramcache.trial)
  trial = get_trial(odeop)
  du = get_trial_fe_basis(trial)
  test = get_test(odeop)
  v = get_fe_basis(test)
  assem = get_param_assembler(odeop,r)

  μ,t = get_params(r),get_times(r)

  w = ws[k]
  iszero(w) && return A

  jac = get_jacs(odeop)[k]
  dc = w * jac(μ,t,uh,du,v)
  matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
  assemble_matrix_add!(A,assem,matdata)

  A
end

# constructors

function TransientLinearParamOperator(args...;kwargs...)
  feop = TransientLinearParamFEOperator(args...;kwargs...)
  get_algebraic_operator(feop)
end

function TransientParamOperator(args...;kwargs...)
  feop = TransientParamFEOperator(args...;kwargs...)
  get_algebraic_operator(feop)
end

const ODEParamNonlinearOperator = GenericParamNonlinearOperator{<:ODEParamOperator}

function ParamAlgebra.allocate_systemcache(nlop::ODEParamNonlinearOperator)
  xh = zero(first(nlop.paramcache.trial))
  x = get_free_dof_values(xh)
  allocate_systemcache(nlop,x)
end
