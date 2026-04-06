function ode_start(
  odeslvr::GeneralizedAlpha2,
  odeop::ODEParamOperator,
  r0::TransientRealisation,
  us0::NTuple{2,AbstractVector}
  )

  nstates = length(us0)
  state0 = ntuple(i -> copy(us0[i]),Val{nstates}())
  upcache = ntuple(i -> copy(us0[i]),Val{nstates}())
  paramcache = allocate_paramcache(odeop,r0)
  syscache = allocate_systemcache(odeop,r0,us0,paramcache)

  ws = (0,0,1)
  
  u0,v0,x = state0[1]
  state_update(x) = (u0,v0,x)
  update_paramcache!(paramcache,odeop,r)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)

  return state0,(upcache,paramcache,syscache)
end

function ODEs.ode_march!(
  statef::NTuple{3,AbstractVector},
  solver::GeneralizedAlpha2,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  state::NTuple{3,AbstractVector},
  odecache
  )

  u0,v0,a0 = state0
  x = statef[3]

  (uα,vα,aα),paramcache,syscache = odecache
  dt = odeslvr.dt
  αf,αm,γ,β = odeslvr.αf,odeslvr.αm,odeslvr.γ,odeslvr.β
  ws = ((1 - αf) * β * dt^2,(1 - αf) * γ * dt,1 - αm)

  mid_shift!(solver,r)
  function state_update(x)
    copyto!(uα,u0)
    axpy!((1 - αf) * dt,v0,uα)
    axpy!((1 - αf) * (1 - 2 * β) * dt^2 / 2,a0,uα)
    axpy!((1 - αf) * β * dt^2,x,uα)

    copyto!(vα,v0)
    axpy!((1 - αf) * (1 - γ) * dt,a0,vα)
    axpy!((1 - αf) * γ * dt,x,vα)

    copyto!(aα,a0)
    rmul!(aα,αm)
    axpy!(1 - αm,x,aα)

    (uα,vα,aα)
  end
  update_paramcache!(paramcache,odeop,r)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)
  mid_front_shift!(solver,r)

  statef = ODEs._update_alpha2!(statef,state,dt,x,γ,β)
  (r,statef)
end

function ODEs.ode_march!(
  statef::NTuple{3,AbstractVector},
  solver::GeneralizedAlpha2,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  state::NTuple{3,AbstractVector},
  odecache
  )

  u0,v0,a0 = state0
  x = statef[3]

  (uα,vα,aα),paramcache,syscache = odecache
  dt = odeslvr.dt
  αf,αm,γ,β = odeslvr.αf,odeslvr.αm,odeslvr.γ,odeslvr.β
  ws = ((1 - αf) * β * dt^2,(1 - αf) * γ * dt,1 - αm)

  mid_shift!(solver,r)
  copyto!(uα,u0)
  axpy!((1 - αf) * dt,v0,uα)
  axpy!((1 - αf) * (1 - 2 * β) * dt^2 / 2,a0,uα)
  copyto!(vα,v0)
  axpy!((1 - αf) * (1 - γ) * dt,a0,vα)
  copyto!(aα,a0)
  rmul!(aα,αm)
  state_update(x) = (uα,vα,aα)
  update_paramcache!(paramcache,odeop,r)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)
  mid_front_shift!(solver,r)

  statef = ODEs._update_alpha2!(statef,state,dt,x,γ,β)
  (r,statef)
end

# linear - nonlinear case 

function ODEs.ode_march!(
  statef::NTuple{3,AbstractVector},
  solver::GeneralizedAlpha2,
  odeop::LinearNonlinearODEParamOperator,
  r::TransientRealisation,
  state::NTuple{3,AbstractVector},
  odecache
  )

  u0,v0 = state0
  x = statef[2]

  ((uα_lin,vα_lin,aα_lin),(uα_nlin,vα_nlin,aα_nlin),
  paramcache_lin,paramcache_nlin,
  syscache_lin,syscache_nlin) = odecache
  dt = odeslvr.dt
  αf,αm,γ,β = odeslvr.αf,odeslvr.αm,odeslvr.γ,odeslvr.β
  ws = ((1 - αf) * β * dt^2,(1 - αf) * γ * dt,1 - αm)

  mid_shift!(solver,r)

  # linear updates
  op_lin = get_linear_operator(odeop)
  copyto!(uα_lin,u0)
  axpy!((1 - αf) * dt,v0,uα_lin)
  axpy!((1 - αf) * (1 - 2 * β) * dt^2 / 2,a0,uα_lin)
  copyto!(vα_lin,v0)
  axpy!((1 - αf) * (1 - γ) * dt,a0,vα_lin)
  copyto!(aα_lin,a0)
  rmul!(aα_lin,αm)
  state_update_lin(x) = (uα_lin,vα_lin,aα_lin)
  update_paramcache!(paramcache_lin,op_lin,r)
  nlop_lin = ParamStageOperator(op_lin,r,state_update_lin,ws,paramcache_lin)

  # nonlinear updates
  op_nlin = get_nonlinear_operator(odeop)
  function state_update_nlin(x)
    copyto!(uα_nlin,u0)
    axpy!((1 - αf) * dt,v0,uα_nlin)
    axpy!((1 - αf) * (1 - 2 * β) * dt^2 / 2,a0,uα_nlin)
    axpy!((1 - αf) * β * dt^2,x,uα_nlin)

    copyto!(vα_nlin,v0)
    axpy!((1 - αf) * (1 - γ) * dt,a0,vα_nlin)
    axpy!((1 - αf) * γ * dt,x,vα_nlin)

    copyto!(aα_nlin,a0)
    rmul!(aα_nlin,αm)
    axpy!(1 - αm,x,aα_nlin)

    (uα_nlin,vα_nlin,aα_nlin)
  end
  update_paramcache!(paramcache_nlin,op_nlin,r)
  nlop_nlin = ParamStageOperator(op_nlin,r,state_update_nlin,ws,paramcache_nlin)

  nlop = LinNonlinParamOperator(nlop_lin,nlop_nlin,syscache_lin)
  solve!(x,solver.sysslvr,nlop,syscache_nlin)
  mid_front_shift!(solver,r)

  statef = ODEs._update_alpha2!(statef,state,dt,x,γ,β)
  (r,statef)
end

function mid_shift!(
  solver::GeneralizedAlpha2,
  r::TransientRealisation
  ) 

  δ = solver.dt*(1-solver.αf)
  shift!(r,δ)
end

function mid_front_shift!(
  solver::GeneralizedAlpha2,
  r::TransientRealisation
  ) 

  δ = solver.αf*solver.dt
  shift!(r,δ)
end