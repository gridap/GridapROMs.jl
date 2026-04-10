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
  b,
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
  b,
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
  A,
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
  return A
end

function spacetime_jacobian!(
  A,
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
  return A
end

# utils 

function _prev_mid_shift!(c::CombinationOrder,r::TransientRealisation)
  _prev_mid_shift!(c.combination,r)
end

function _cur_shift!(c::CombinationOrder,r::TransientRealisation)
  _cur_shift!(c.combination,r)
end

function _prev_mid_shift!(c::ThetaMethodCombination,r::TransientRealisation) 
  δ = c.dt*(1-c.θ)
  shift!(r,-δ)
end

function _cur_shift!(c::ThetaMethodCombination,r::TransientRealisation) 
  δ = c.dt*(1-c.θ)
  shift!(r,δ)
end

function _prev_mid_shift!(c::GenAlpha1Combination,r::TransientRealisation) 
  δ = c.dt*(1-c.αf)
  shift!(r,-δ)
end

function _cur_shift!(c::GenAlpha1Combination,r::TransientRealisation)
  δ = c.dt*(1-c.αf)
  shift!(r,δ)
end

function _prev_mid_shift!(c::GenAlpha2Combination,r::TransientRealisation)
  δ = c.dt*c.αf
  shift!(r,-δ)
end

function _cur_shift!(c::GenAlpha2Combination,r::TransientRealisation)
  δ = c.dt*c.αf
  shift!(r,δ)
end