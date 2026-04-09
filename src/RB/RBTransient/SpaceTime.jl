struct SpaceTimeSolver <: NonlinearSolver
  sysslvr::ODESolver
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

function Algebra.solve!(
  x̂::AbstractVector,
  solver::SpaceTimeSolver,
  nlop::NonlinearParamOperator,
  syscache)

  r = _get_realisation(nlop)
  front_shift!(solver,r)
  _update_paramcache!(nlop,r)
  solve!(x̂,solver.sysslvr,nlop,syscache)
  back_shift!(solver,r)
end

function get_reduced_combination(
  U::RBSpace,
  c::TimeCombination,
  u::RBParamVector,
  us0::NTuple{N,RBParamVector}
  ) where N

  fed = u.fe_data
  fed0 = map(x -> x.fe_data,us0)
  fe_data = get_time_combination(c,fed,fed0)
  data = project(U,fe_data)
  reduced_vector(data,fe_data)
end

function get_zero_reduced_combination(
  U::RBSpace,
  c::TimeCombination,
  u::RBParamVector,
  us0::NTuple{N,RBParamVector}
  ) where N

  fed = u.fe_data
  fed0 = map(x -> x.fe_data,us0)
  fe_data = get_zero_time_combination(c,fed,fed0)
  data = project(U,fe_data)
  reduced_vector(data,fe_data)
end

# utils 

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