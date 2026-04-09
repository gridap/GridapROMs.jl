include("TimeCombinations.jl")

function spacetime_residual(
  tcomb::TimeCombination,
  odeop::ODEParamOperator,
  s::AbstractSnapshots
  )

  r = get_realisation(s)
  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  spacetime_residual(tcomb,odeop,r,u,us0)
end

function spacetime_residual(
  tcomb::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )

  usx = get_time_combination(tcomb,u,us0)
  _prev_mid_shift!(tcomb,r)
  b = residual(odeop,r,usx)
  _cur_shift!(tcomb,r)
  return b
end

function allocate_spacetime_residual(
  tcomb::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )

  _prev_mid_shift!(tcomb,r)
  paramcache = allocate_paramcache(odeop,r;evaluated=true)
  _cur_shift!(tcomb,r)
  usx = allocate_time_combination(tcomb,u,us0)
  b = residual(odeop,r,usx)
  
  return b
end

function spacetime_residual!(
  b::AbstractParamVector,
  tcomb::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )

  get_time_combination!(usx,tcomb,u,us0)
  _prev_mid_shift!(tcomb,r)
  residual!(b,odeop,r,usx)
  _cur_shift!(tcomb,r)
  return b
end

function spacetime_jacobian(
  tcomb::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::AbstractSnapshots
  )

  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  usx = get_time_combination(tcomb,u,us0)
  ws = ntuple(_ -> 1,Val(length(us0)))

  _prev_mid_shift!(tcomb,r)
  A = jacobian(odeop,r,usx,ws)
  _cur_shift!(tcomb,r)

  return A
end

function spacetime_residual(
  tcomb::TimeCombination,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::AbstractSnapshots
  )

  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  usx = get_zero_time_combination(tcomb,u,us0)

  _prev_mid_shift!(tcomb,r)
  b = residual(odeop,r,usx)
  _cur_shift!(tcomb,r)

  return b
end

function spacetime_jacobian(
  tcomb::TimeCombination,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::AbstractSnapshots
  )

  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  usx = get_zero_time_combination(tcomb,u,us0)
  ws = ntuple(_ -> 1,Val(length(us0)))

  _prev_mid_shift!(tcomb,r)
  A = jacobian(odeop,r,usx,ws)
  _cur_shift!(tcomb,r)

  return A
end

# utils 

function _prev_mid_shift!(tcomb::TimeCombination,r::TransientRealisation)
  _prev_mid_shift!(tcomb.solver,r)
end

function _cur_shift!(tcomb::TimeCombination,r::TransientRealisation)
  _cur_shift!(tcomb.solver,r)
end

function _prev_mid_shift!(
  solver::ThetaMethod,
  r::TransientRealisation
  ) 

  δ = solver.dt*(1-solver.θ)
  shift!(r,-δ)
end

function _cur_shift!(
  solver::ThetaMethod,
  r::TransientRealisation
  ) 

  δ = solver.dt*(1-solver.θ)
  shift!(r,δ)
end

function _prev_mid_shift!(
  solver::GeneralizedAlpha1,
  r::TransientRealisation
  ) 

  δ = solver.dt*(1-solver.αf)
  shift!(r,-δ)
end

function _cur_shift!(
  solver::GeneralizedAlpha1,
  r::TransientRealisation
  ) 

  δ = solver.dt*(1-solver.αf)
  shift!(r,δ)
end

function _prev_mid_shift!(
  solver::GeneralizedAlpha2,
  r::TransientRealisation
  ) 

  δ = solver.dt*solver.αf
  shift!(r,-δ)
end

function _cur_shift!(
  solver::GeneralizedAlpha2,
  r::TransientRealisation
  ) 

  δ = solver.dt*solver.αf
  shift!(r,δ)
end