function ode_start(
  odeslvr::GeneralizedAlpha1, 
  odeop::ODEParamOperator,
  r0::TransientRealisation, 
  us0::NTuple{1,AbstractVector},
  odecache
  )

  # Unpack inputs
  u0 = us0[1]
  odeslvrcache, odeopcache = odecache
  uα, vα, sysslvrcache = odeslvrcache

  # Unpack solver
  sysslvr = odeslvr.sysslvr

  # Allocate state
  s0, s1 = copy(u0), copy(u0)

  # Define scheme
  x = s1
  tx = t0
  usx(x) = (u0, x)
  ws = (0, 1)

  # Update ODE operator cache
  update_odeopcache!(odeopcache, odeop, tx)

  # Create and solve stage operator
  stageop = NonlinearStageOperator(
    odeop, odeopcache,
    tx, usx, ws
  )

  sysslvrcache = solve!(x, sysslvr, stageop, sysslvrcache)

  # Update state
  state0 = (s0, s1)

  # Pack outputs
  odeslvrcache = (uα, vα, sysslvrcache)
  odecache = (odeslvrcache, odeopcache)
  (state0, odecache)
end

function ode_start(
  odeslvr::GeneralizedAlpha1, odeop::ODEOperator,
  t0::Real, us0::NTuple{2,AbstractVector},
  odecache
)
  # Unpack inputs
  u0, v0 = us0[1], us0[2]

  # Allocate state
  s0, s1 = copy(u0), copy(v0)

  # Update state
  state0 = (s0, s1)

  # Pack outputs
  (state0, odecache)
end

function ODEs.ode_march!(
  statef::NTuple{2,AbstractVector},
  solver::GeneralizedAlpha1,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  state::NTuple{2,AbstractVector},
  odecache
  )

  u0,v0 = state0
  x = statef[2]

  (uα,vα),paramcache,syscache = odecache
  dt,αf,αm,γ = odeslvr.dt,odeslvr.αf,odeslvr.αm,odeslvr.γ
  ws = (αf*γ*dt,αm)

  shift!(r,αf*dt)
  function state_update(x)
    copyto!(uα,u0)
    axpy!(αf * (1 - γ) * dt,v0,uα)
    axpy!(αf * γ * dt,x,uα)

    copyto!(vα,v0)
    rmul!(vα,1 - αm)
    axpy!(αm,x,vα)
    
    (uα,vα)
  end
  update_paramcache!(paramcache,odeop,r)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)
  shift!(r,dt*(1-αf))

  statef = ODEs._update_alpha1!(statef,state,dt,x,γ)
  (r,statef)
end

function ODEs.ode_march!(
  statef::NTuple{2,AbstractVector},
  solver::GeneralizedAlpha1,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  state::NTuple{2,AbstractVector},
  odecache
  )

  u0,v0 = state0
  x = statef[2]

  (uα,vα),paramcache,syscache = odecache
  dt,αf,αm,γ = odeslvr.dt,odeslvr.αf,odeslvr.αm,odeslvr.γ
  ws = (αf*γ*dt,αm)
  
  shift!(r,αf*dt)
  copyto!(uα,u0)
  axpy!(αf * (1 - γ) * dt,v0,uα)
  copyto!(vα,v0)
  rmul!(vα,1 - αm)
  state_update(x) = (uα,vα)
  update_paramcache!(paramcache,odeop,r)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)
  shift!(r,dt*(1-αf))

  statef = ODEs._update_alpha1!(statef,state,dt,x,γ)
  (r,statef)
end

# linear - nonlinear case 

function ODEs.ode_march!(
  statef::NTuple{2,AbstractVector},
  solver::GeneralizedAlpha1,
  odeop::LinearNonlinearODEParamOperator,
  r::TransientRealisation,
  state::NTuple{2,AbstractVector},
  odecache
  )

  u0,v0 = state0
  x = statef[2]

  ((uα_lin,vα_lin),(uα_nlin,vα_nlin),
  paramcache_lin,paramcache_nlin,
  syscache_lin,syscache_nlin) = odecache
  dt,αf,αm,γ = odeslvr.dt,odeslvr.αf,odeslvr.αm,odeslvr.γ
  ws = (αf*γ*dt,αm)

  shift!(r,αf*dt)

  # linear updates
  op_lin = get_linear_operator(odeop)
  copyto!(uα_lin,u0)
  axpy!(αf * (1 - γ) * dt,v0,uα_lin)
  copyto!(vα_lin,v0)
  rmul!(vα_lin,1 - αm)
  state_update_lin(x) = (uα_lin,vα_lin)
  update_paramcache!(paramcache_lin,op_lin,r)
  nlop_lin = ParamStageOperator(op_lin,r,state_update_lin,ws,paramcache_lin)

  # nonlinear updates
  op_nlin = get_nonlinear_operator(odeop)
  function state_update_nlin(x)
    copyto!(uα_nlin,u0)
    axpy!(αf * (1 - γ) * dt,v0,uα_nlin)
    axpy!(αf * γ * dt,x,uα_nlin)

    copyto!(vα_nlin,v0)
    rmul!(vα_nlin,1 - αm)
    axpy!(αm,x,vα_nlin)
    
    (uα_nlin,vα_nlin)
  end
  update_paramcache!(paramcache_nlin,op_nlin,r)
  nlop_nlin = ParamStageOperator(op_nlin,r,state_update_nlin,ws,paramcache_nlin)

  nlop = LinNonlinParamOperator(nlop_lin,nlop_nlin,syscache_lin)
  solve!(x,solver.sysslvr,nlop,syscache_nlin)
  shift!(r,dt*(1-αf))

  statef = ODEs._update_alpha1!(statef,state,dt,x,γ)
  (r,statef)
end
