function ODEs.ode_start(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r0::TransientRealisation,
  us0::Tuple{Vararg{AbstractVector}})

  nstates = length(us0)
  state0 = ntuple(i -> copy(us0[i]),Val{nstates}())
  upcache = ntuple(i -> copy(us0[i]),Val{nstates}())
  paramcache = allocate_paramcache(odeop,r0)
  syscache = allocate_systemcache(odeop,r0,us0,paramcache)
  return state0,(upcache,paramcache,syscache)
end

function ODEs.ode_finish!(
  uf::AbstractVector,
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  statef::Tuple{Vararg{AbstractVector}},
  odecache)

  copy!(uf,first(statef))
  uf
end

# linear - nonlinear interface

function ODEs.ode_start(
  solver::ODESolver,
  odeop::LinearNonlinearODEParamOperator,
  r0::TransientRealisation,
  us0::Tuple{Vararg{AbstractVector}})

  nstates = length(us0)
  state0 = ntuple(i -> copy(us0[i]),Val{nstates}())
  # linear caches
  upcache_lin = ntuple(i -> copy(us0[i]),Val{nstates}())
  op_lin =  get_linear_operator(odeop)
  order_lin = get_order(op_lin)
  us0_lin = ntuple(i -> us0[i],Val{order_lin+1}())
  paramcache_lin = allocate_paramcache(op_lin,r0)
  syscache_lin = allocate_systemcache(op_lin,r0,us0_lin,paramcache_lin)
  # nonlinear caches
  upcache_nlin = ntuple(i -> copy(us0[i]),Val{nstates}())
  op_nlin =  get_nonlinear_operator(odeop)
  order_nlin = get_order(op_nlin)
  us0_nlin = ntuple(i -> us0[i],Val{order_nlin+1}())
  paramcache_nlin = allocate_paramcache(op_nlin,r0)
  _syscache_nlin = allocate_systemcache(op_nlin,r0,us0_nlin,paramcache_nlin)
  syscache_nlin = compatible_cache(_syscache_nlin,syscache_lin)
  return state0,(upcache_lin,upcache_nlin,paramcache_lin,paramcache_nlin,syscache_lin,syscache_nlin)
end

include("ThetaMethod.jl")
include("GeneralizedAlpha1.jl")
include("GeneralizedAlpha2.jl")