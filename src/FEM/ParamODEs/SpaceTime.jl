function front_shift!(solver::ODESolver,r::TransientRealisation) 
  front_shift!(ShiftedSolver(solver),r)
end

function back_shift!(solver::ODESolver,r::TransientRealisation) 
  back_shift!(ShiftedSolver(solver),r)
end

# if shift=:spacetime => compute space-time residuals/jacobians (no time marching)
# if shift=:spaceonly => compute time-dependent residuals/jacobians (time marching)

function Algebra.residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  if shift==:spacetime 
    _spacetime_residual(solver,odeop,r,u,us0)
  else shift==:spaceonly
    _spaceonly_residual(solver,odeop,r,u,us0)
  end
end

function Algebra.jacobian(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  if shift==:spacetime 
    _spacetime_jacobian(solver,odeop,r,u,us0)
  else shift==:spaceonly
    _spaceonly_jacobian(solver,odeop,r,u,us0)
  end
end

function _spacetime_residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )
  
  shift!(uθ,u0,0,1)
  us = (uθ,x)

  _prev_mid_shift!(solver,r)
  b = residual(odeop,r,us)
  _cur_shift!(solver,r)

  return b 
end

function Algebra.residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  u = get_param_data(s)
  u0, = get_initial_param_data(s)

  dt,θ = solver.dt,solver.θ
  x = copy(u)
  uθ = copy(u)

  if shift==:spacetime
    shift!(uθ,u0,θ,1-θ)
    shift!(x,u0,1/dt,-1/dt)
    us = (uθ,x)
  else
    shift!(uθ,u0,0,1)
    us = (uθ,x)
  end

  _prev_mid_shift!(solver,r)
  b = residual(odeop,r,us)
  _cur_shift!(solver,r)

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  u = get_param_data(s)
  u0, = get_initial_param_data(s)

  dt,θ = solver.dt,solver.θ
  x = copy(u)
  uθ = copy(u)

  if shift==:spacetime
    shift!(uθ,u0,θ,1-θ)
    shift!(x,u0,1/dt,-1/dt)
    us = (uθ,x)
    ws = (1,1)
  else
    shift!(uθ,u0,0,1)
    us = (uθ,x)
    ws = (dt*θ,1)
  end

  _prev_mid_shift!(solver,r)
  A = jacobian(odeop,r,us,ws)
  _cur_shift!(solver,r)

  return A
end

function Algebra.residual(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  u = get_param_data(s)
  u0, = get_initial_param_data(s)

  dt,θ = solver.dt,solver.θ
  x = copy(u)
  fill!(x,zero(eltype(x)))

  if shift==:spacetime
    us = (x,x)
  else
    uθ = copy(u)
    shift!(uθ,u0,0,1)
    us = (uθ,x)
  end

  _prev_mid_shift!(solver,r)
  b = residual(odeop,r,us)
  _cur_shift!(solver,r)

  return b
end

function Algebra.jacobian(
  solver::ThetaMethod,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  u = get_param_data(s)
  u0, = get_initial_param_data(s)
  
  dt,θ = solver.dt,solver.θ
  x = copy(u)
  fill!(x,zero(eltype(x)))

  if shift==:spacetime
    ws = (1,1)
    us = (x,x)
  else
    ws = (dt*θ,1)
    uθ = copy(u)
    shift!(uθ,u0,0,1)
    us = (uθ,x)
  end

  _prev_mid_shift!(solver,r)
  A = jacobian(odeop,r,us,ws)
  _cur_shift!(solver,r)

  return A
end

function Algebra.residual(
  solver::GeneralizedAlpha1,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  u0,u̇0 = us0

  dt,αf,αm,γ = solver.dt,solver.αf,solver.αm,solver.γ
  x = copy(u)
  uθ = copy(u)

  if shift==:spacetime
    shift!(uθ,u0,αf,1-αf)
    shift!(x,u0,1/dt,-1/dt)
    us = (uθ,x)
  else
    shift!(uθ,u0,0,1)
    us = (uθ,x)
  end

  _prev_mid_shift!(solver,r)
  b = residual(odeop,r,us)
  _cur_shift!(solver,r)

  return b
end

function Algebra.jacobian(
  solver::GeneralizedAlpha1,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  
end

function Algebra.residual(
  solver::GeneralizedAlpha1,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  
end

function Algebra.jacobian(
  solver::GeneralizedAlpha1,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  
end


function Algebra.residual(
  solver::GeneralizedAlpha2,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  
end

function Algebra.jacobian(
  solver::GeneralizedAlpha2,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  
end

function Algebra.residual(
  solver::GeneralizedAlpha2,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  
end

function Algebra.jacobian(
  solver::GeneralizedAlpha2,
  odeop::ODEParamOperator{LinearParamODE},
  r::TransientRealisation,
  s::Snapshots;
  shift=:spacetime)

  
end

# utils

function ParamDataStructures.shift!(
  a::ConsecutiveParamVector,
  a0::ConsecutiveParamVector,
  α::Number,
  β::Number)

  data = get_all_data(a)
  data0 = get_all_data(a0)
  data′ = copy(data)
  np = param_length(a0)
  for ipt = param_eachindex(a)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        data[is,ipt] = α*data[is,ipt] + β*data0[is,ipt]
      end
    else
      for is in axes(data,1)
        data[is,ipt] = α*data[is,ipt] + β*data′[is,ipt-np]
      end
    end
  end
end

function ParamDataStructures.shift!(
  a::BlockParamVector,
  a0::BlockParamVector,
  α::Number,
  β::Number)

  @inbounds for (ai,a0i) in zip(blocks(a),blocks(a0))
    ParamDataStructures.shift!(ai,a0i,α,β)
  end
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