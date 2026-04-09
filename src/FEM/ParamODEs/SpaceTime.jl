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

  b,usx,paramcache = allocate_spacetime_residual(tcomb,odeop,r,u,us0)
  spacetime_residual!(b,tcomb,odeop,r,u,usx,us0,paramcache)
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
  b = allocate_residual(odeop,r,usx,paramcache)
  return b,usx,paramcache
end

function spacetime_residual!(
  b::AbstractParamVector,
  tcomb::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  usx::Tuple{Vararg{AbstractParamVector}},
  us0::Tuple{Vararg{AbstractParamVector}},
  paramcache
  )

  time_combination!(usx,tcomb,u,us0)
  _prev_mid_shift!(tcomb,r)
  residual!(b,odeop,r,usx,paramcache)
  _cur_shift!(tcomb,r)
  return b
end

function spacetime_residual!(
  b::AbstractParamVector,
  tcomb::TimeCombination,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  u::AbstractParamVector,
  usx::Tuple{Vararg{AbstractParamVector}},
  us0::Tuple{Vararg{AbstractParamVector}},
  paramcache
  )

  zero_time_combination!(usx,tcomb,u,us0)
  _prev_mid_shift!(tcomb,r)
  residual!(b,odeop,r,usx,paramcache)
  _cur_shift!(tcomb,r)
  return b
end

function spacetime_jacobian(
  tcomb::TimeCombination,
  odeop::ODEParamOperator,
  s::AbstractSnapshots
  )

  r = get_realisation(s)
  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  spacetime_jacobian(tcomb,odeop,r,u,us0)
end

function spacetime_jacobian(
  tcomb::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )

  A,usx,paramcache = allocate_spacetime_jacobian(tcomb,odeop,r,u,us0)
  spacetime_jacobian!(A,tcomb,odeop,r,u,usx,us0,paramcache)
  return A
end

function allocate_spacetime_jacobian(
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
  A = allocate_jacobian(odeop,r,usx,paramcache)
  return A,usx,paramcache
end

function spacetime_jacobian!(
  A::AbstractParamMatrix,
  tcomb::TimeCombination,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  usx::Tuple{Vararg{AbstractParamVector}},
  us0::Tuple{Vararg{AbstractParamVector}},
  paramcache
  )

  ws = ntuple(_ -> 1,Val(length(us0)))
  time_combination!(usx,tcomb,u,us0)
  _prev_mid_shift!(tcomb,r)
  jacobian!(A,odeop,r,usx,ws,paramcache)
  _cur_shift!(tcomb,r)
  return b
end

function spacetime_jacobian!(
  A::AbstractParamMatrix,
  tcomb::TimeCombination,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  u::AbstractParamVector,
  usx::Tuple{Vararg{AbstractParamVector}},
  us0::Tuple{Vararg{AbstractParamVector}},
  paramcache
  )

  ws = ntuple(_ -> 1,Val(length(us0)))
  zero_time_combination!(usx,tcomb,u,us0)
  _prev_mid_shift!(tcomb,r)
  jacobian!(A,odeop,r,usx,ws,paramcache)
  _cur_shift!(tcomb,r)
  return b
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