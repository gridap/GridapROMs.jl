"""
    add_initial_conditions(solver::ODESolver,odeop::ODEParamOperator,r::TransientRealisation,us0) -> Tuple

Completes the initial conditions `us0` with the additional states required by the
time-marching scheme of `solver`
"""
function add_initial_conditions(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  _us0::NTuple{1,AbstractVector}
  )
  
  u0, = _us0
  us0 = (u0,copy(u0))
  return us0
end

function ODEs.ode_march!(
  statef::NTuple{2,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  state::NTuple{2,AbstractVector},
  odecache
  )

  u0 = state[1]
  x = statef[1]

  (uθ,vθ),paramcache,syscache = odecache
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
  statef::NTuple{2,AbstractVector},
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  state::NTuple{2,AbstractVector},
  odecache
  )

  u0 = state[1]
  x = statef[1]

  (uθ,vθ),paramcache,syscache = odecache
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

function ODEs._udate_theta!(
  statef::NTuple{2,AbstractVector}, 
  state0::NTuple{2,AbstractVector},
  dt::Real, 
  x::AbstractVector
  )
  
  u0 = state0[1]
  uf,vf = statef
  rmul!(uf,dt)
  axpy!(1,u0,uf)
  (uf,vf)
end