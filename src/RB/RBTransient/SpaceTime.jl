struct SpaceTimeOperator <: NonlinearOperator
  op::NonlinearOperator
  tcomb::TimeCombination
  usx::Tuple{Vararg{AbstractVector}}
  us0::Tuple{Vararg{AbstractVector}}
end

function SpaceTimeOperator(
  op::ParamOperator, 
  c::TimeCombination,
  r::TransientRealisation,
  us0::Tuple{Vararg{AbstractVector}}
  )

  pop = parameterise(op,r)
  trial = get_trial(op)(r)
  û = zero_free_values(trial)
  ûs = allocate_time_combination(c,û,us0)
  SpaceTimeOperator(pop,c,ûs,us0)
end

function SpaceTimeOperator(
  op::ParamOperator, 
  tcomb::TimeCombination,
  r::TransientRealisation,
  uhs0::Tuple{Vararg{Function}}
  )

  params = get_params(r)
  us0 = ()
  for uh0 in uhs0
    us0 = (us0...,get_free_dof_values(uh0(params)))
  end
  SpaceTimeOperator(op,tcomb,r,us0)
end

function Algebra.allocate_residual(
  nlop::SpaceTimeOperator,
  x::AbstractVector)

  allocate_residual(nlop.op.op,nlop.op.r,nlop.usx,nlop.op.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::SpaceTimeOperator,
  x::AbstractVector)

  time_combination!(nlop.usx,nlop.tcomb,x,nlop.us0)
  residual!(b,nlop.op.op,nlop.op.r,usx,nlop.op.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::SpaceTimeOperator,
  x::AbstractVector)

  allocate_jacobian(nlop.op.op,nlop.op.r,nlop.usx,nlop.op.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::SpaceTimeOperator,
  x::AbstractVector)

  time_combination!(nlop.usx,nlop.tcomb,x,nlop.us0)
  jacobian!(A,nlop.op.op,nlop.op.r,usx,nlop.op.ws,nlop.op.paramcache)
  A
end

struct SpaceTimeSolver <: NonlinearSolver
  sysslvr::ODESolver
end

function SpaceTimeSolver(rbsolver::RBSolver)
  SpaceTimeSolver(get_fe_solver(rbsolver))
end

get_shift(s::SpaceTimeSolver) = @notimplemented 

function get_shift(s::SpaceTimeSolver{<:ThetaMethod})
  s.sysslvr.dt*(s.sysslvr.θ-1)
end

function get_shift(s::SpaceTimeSolver{<:GeneralizedAlpha1})
  s.sysslvr.dt*(s.sysslvr.αf-1)
end

function get_shift(s::SpaceTimeSolver{<:GeneralizedAlpha2})
  s.sysslvr.dt*(1-s.sysslvr.αf)
end

front_shift!(s::SpaceTimeSolver,r::TransientRealisation) = shift!(r,get_shift(s))
back_shift!(s::SpaceTimeSolver,r::TransientRealisation) = shift!(r,-get_shift(s))

ParamODEs.TimeCombination(s::SpaceTimeSolver) = TimeCombination(s.sysslvr)

function Algebra.solve!(
  x̂::AbstractVector,
  solver::SpaceTimeSolver,
  nlop::NonlinearParamOperator,
  syscache
  )

  r = _get_realisation(nlop)
  front_shift!(solver,r)
  _update_paramcache!(nlop,r)
  solve!(x̂,solver.sysslvr,nlop,syscache)
  back_shift!(solver,r)
end

# utils 

function _insert_initial_condition!(
  u::RBParamVector,
  U::RBSpace,
  c::TimeCombination,
  us0::NTuple{N,AbstractParamVector}
  ) where N

  ParamODEs._zero_combination!(u.fe_data,c,us0)
  project!(u.data,U,u.fe_data)
  u
end

_get_realisation(nlop::NonlinearParamOperator) = @abstractmethod
_get_realisation(nlop::GenericParamNonlinearOperator) = nlop.μ
_get_realisation(nlop::LinNonlinParamOperator) = _get_realisation(nlop.op_nonlinear)

function _update_paramcache!(nlop::NonlinearParamOperator,r::TransientRealisation)
  @abstractmethod
end

function _update_paramcache!(nlop::GenericParamNonlinearOperator,r::TransientRealisation)
  update_paramcache!(nlop.paramcache,nlop.op,r)
end

function _update_paramcache!(nlop::LinNonlinParamOperator,r::TransientRealisation)
  _update_paramcache!(nlop.op_linear,r)
  _update_paramcache!(nlop.op_nonlinear,r)
end