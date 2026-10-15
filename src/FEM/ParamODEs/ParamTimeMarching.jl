function ODEs.ode_start(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r0::TransientRealisation,
  us0::Tuple{Vararg{AbstractVector}}
  )

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
  odecache
  )

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

include("ThetaMethod.jl")
include("GeneralizedAlpha1.jl")
include("GeneralizedAlpha2.jl")