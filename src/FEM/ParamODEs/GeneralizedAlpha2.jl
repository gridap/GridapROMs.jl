function add_initial_conditions(
  solver::GeneralizedAlpha2,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  _us0::NTuple{2,AbstractVector}
  )
  
  u0,v0 = _us0
  x = copy(u0)
  us0 = (u0,v0,x)
  r0 = get_at_time(r,:initial)
  paramcache = allocate_paramcache(odeop,r0;evaluated=true)
  syscache = allocate_systemcache(odeop,r0,us0,paramcache)

  ws = (0,0,1)
  state_update(x) = (u0,v0,x)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)

  return us0
end

function ODEs.ode_march!(
  statef::NTuple{3,AbstractVector},
  solver::GeneralizedAlpha2,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  state0::NTuple{3,AbstractVector},
  odecache
  )

  u0,v0,a0 = state0
  x = statef[3]

  (uα,vα,aα),paramcache,syscache = odecache
  dt = solver.dt
  αf,αm,γ,β = solver.αf,solver.αm,solver.γ,solver.β
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

  statef = ODEs._update_alpha2!(statef,state0,dt,x,γ,β)
  (r,statef)
end

function ODEs.ode_march!(
  statef::NTuple{3,AbstractVector},
  solver::GeneralizedAlpha2,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  state0::NTuple{3,AbstractVector},
  odecache
  )

  u0,v0,a0 = state0
  x = statef[3]

  (uα,vα,aα),paramcache,syscache = odecache
  dt = solver.dt
  αf,αm,γ,β = solver.αf,solver.αm,solver.γ,solver.β
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

  statef = ODEs._update_alpha2!(statef,state0,dt,x,γ,β)
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