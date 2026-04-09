function ODEs.ode_start(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r0::TransientRealisation,
  us0::Tuple{Vararg{AbstractVector}})

  nstates = get_order(odeop)+1
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

function add_initial_conditions(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  _us0::Tuple{Vararg{AbstractVector}}
  )
  
  @abstractmethod
end

# linear - nonlinear interface

function ODEs.ode_start(
  solver::ODESolver,
  odeop::LinearNonlinearODEParamOperator,
  r0::TransientRealisation,
  us0::Tuple{Vararg{AbstractVector}})

  nstates = get_order(odeop)+1
  state0 = ntuple(i -> copy(us0[i]),Val{nstates}())

  # linear caches
  op_lin =  get_linear_operator(odeop)
  upcache_lin = ntuple(i -> copy(us0[i]),Val{nstates}())
  us0_lin = ntuple(i -> us0[i],Val{nstates}())
  paramcache_lin = allocate_paramcache(op_lin,r0)
  syscache_lin = allocate_systemcache(op_lin,r0,us0_lin,paramcache_lin)

  # nonlinear caches
  op_nlin =  get_nonlinear_operator(odeop)
  upcache_nlin = ntuple(i -> copy(us0[i]),Val{nstates}())
  us0_nlin = ntuple(i -> us0[i],Val{nstates}())
  paramcache_nlin = allocate_paramcache(op_nlin,r0)
  _syscache_nlin = allocate_systemcache(op_nlin,r0,us0_nlin,paramcache_nlin)
  syscache_nlin = compatible_cache(_syscache_nlin,syscache_lin)

  return state0,(upcache_lin,upcache_nlin,paramcache_lin,paramcache_nlin,syscache_lin,syscache_nlin)
end

include("ThetaMethod.jl")
include("GeneralizedAlpha1.jl")
include("GeneralizedAlpha2.jl")