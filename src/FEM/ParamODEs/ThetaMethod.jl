function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  state::NTuple{1,AbstractVector},
  odecache)

  u0 = state[1]
  x = statef[1]

  uθ,paramcache,syscache = odecache
  dt,θ = solver.dt,solver.θ
  ws = (dt*θ,1)

  mid_shift!(solver,r)
  function state_update(x)
    copy!(uθ,u0)
    axpy!(dt*θ,x,uθ)
    (uθ,x)
  end
  update_paramcache!(paramcache,odeop,r)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)
  mid_front_shift!(solver,r)

  statef = ODEs._udate_theta!(statef,state,dt,x)
  (r,statef)
end

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  state::NTuple{1,AbstractVector},
  odecache)

  u0 = state[1]
  x = statef[1]
  fill!(x,zero(eltype(x)))

  uθ,paramcache,syscache = odecache
  dt,θ = solver.dt,solver.θ
  ws = (dt*θ,1)

  mid_shift!(solver,r)
  state_update(x) = (u0,x)
  update_paramcache!(paramcache,odeop,r)
  nlop = ParamStageOperator(odeop,r,state_update,ws,paramcache)
  solve!(x,solver.sysslvr,nlop,syscache)
  mid_front_shift!(solver,r)

  statef = ODEs._udate_theta!(statef,state,dt,x)
  (r,statef)
end

# linear - nonlinear case 

function ODEs.ode_march!(
  statef::NTuple{1,AbstractVector},
  solver::ThetaMethod,
  odeop::LinearNonlinearODEParamOperator,
  r::TransientRealisation,
  state::NTuple{1,AbstractVector},
  odecache)

  u0 = state[1]
  x = statef[1]

  (uθ_lin,uθ_nlin,
  paramcache_lin,paramcache_nlin,
  syscache_lin,syscache_nlin) = odecache
  dt,θ = solver.dt,solver.θ
  ws = (dt*θ,1)

  mid_shift!(solver,r)

  # linear updates
  op_lin = get_linear_operator(odeop)
  state_update_lin(x) = (u0,x)
  update_paramcache!(paramcache_lin,op_lin,r)
  nlop_lin = ParamStageOperator(op_lin,r,state_update_lin,ws,paramcache_lin)

  # nonlinear updates
  op_nlin = get_nonlinear_operator(odeop)
  function state_update_nlin(x)
    copy!(uθ_nlin,u0)
    axpy!(dt*θ,x,uθ_nlin)
    (uθ_nlin,x)
  end
  update_paramcache!(paramcache_nlin,op_nlin,r)
  nlop_nlin = ParamStageOperator(op_nlin,r,state_update_nlin,ws,paramcache_nlin)

  nlop = LinNonlinParamOperator(nlop_lin,nlop_nlin,syscache_lin)
  solve!(x,solver.sysslvr,nlop,syscache_nlin)
  mid_front_shift!(solver,r)

  statef = ODEs._udate_theta!(statef,state,dt,x)
  (r,statef)
end

function mid_shift!(
  solver::ThetaMethod,
  r::TransientRealisation
  ) 

  δ = solver.θ*solver.dt
  shift!(r,δ)
end

function mid_front_shift!(
  solver::ThetaMethod,
  r::TransientRealisation
  ) 

  δ = solver.dt*(1-solver.θ)
  shift!(r,δ)
end