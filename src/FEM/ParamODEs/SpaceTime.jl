include("TimeCombinations.jl")

function Algebra.residual(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::AbstractSnapshots
  )

  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  tcomb = TimeCombination(solver)
  usx = get_time_combination(tcomb,u,us0)

  _prev_mid_shift!(solver,r)
  b = residual(odeop,r,usx)
  _cur_shift!(solver,r)

  return b
end

function Algebra.jacobian(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::AbstractSnapshots
  )

  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  tcomb = TimeCombination(solver)
  usx = get_time_combination(tcomb,u,us0)
  ws = ntuple(_ -> 1,Val(length(us0)))

  _prev_mid_shift!(solver,r)
  b = jacobian(odeop,r,usx,ws)
  _cur_shift!(solver,r)

  return b
end

function Algebra.residual(
  solver::ODESolver,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::AbstractSnapshots
  )

  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  z = zero(eltype2(u))
  N = length(us0)
  usx = ntuple(_ -> fill!(similar(u),z),Val{N}())

  _prev_mid_shift!(solver,r)
  b = residual(odeop,r,usx)
  _cur_shift!(solver,r)

  return b
end

function Algebra.jacobian(
  solver::ODESolver,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::AbstractSnapshots
  )

  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  z = zero(eltype2(u))
  N = length(us0)
  usx = ntuple(_ -> fill!(similar(u),z),Val{N}())
  ws = ntuple(_ -> 1,Val{N}())

  _prev_mid_shift!(solver,r)
  b = jacobian(odeop,r,usx,ws)
  _cur_shift!(solver,r)

  return b
end

# utils 

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