function stage_variable(solver::ODESolver,args...)
  @abstractmethod
end

function allocate_updatecache(solver::ODESolver,args...)
  stage_variable(solver,args...)
end

function ODEs.ode_start(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r0::TransientRealisation,
  u0::AbstractVector)

  state0 = stage_variable(solver,u0)
  upcache = allocate_updatecache(solver,u0)
  order = get_order(odeop)
  us0 = tfill(u0,Val{order+1}())
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
  u0::AbstractVector)

  state0 = stage_variable(solver,u0)
  # linear caches
  upcache_lin = allocate_updatecache(solver,u0)
  op_lin =  get_linear_operator(odeop)
  order_lin = get_order(op_lin)
  us0_lin = tfill(u0,Val{order_lin+1}())
  paramcache_lin = allocate_paramcache(op_lin,r0)
  syscache_lin = allocate_systemcache(op_lin,r0,us0_lin,paramcache_lin)
  # nonlinear caches
  upcache_nlin = allocate_updatecache(solver,u0)
  op_nlin =  get_nonlinear_operator(odeop)
  order_nlin = get_order(op_nlin)
  us0_nlin = tfill(u0,Val{order_nlin+1}())
  paramcache_nlin = allocate_paramcache(op_nlin,r0)
  _syscache_nlin = allocate_systemcache(op_nlin,r0,us0_nlin,paramcache_nlin)
  syscache_nlin = compatible_cache(_syscache_nlin,syscache_lin)
  return state0,(upcache_lin,upcache_nlin,paramcache_lin,paramcache_nlin,syscache_lin,syscache_nlin)
end

include("ThetaMethod.jl")
include("GeneralizedAlpha1.jl")
include("GeneralizedAlpha2.jl")