struct ShiftedSolver <: NonlinearSolver
  sysslvr::NonlinearSolver
  δ::Real
end

function ShiftedSolver(odesolver::ODESolver)
  @notimplemented "For now, only theta methods are implemented"
end

function ShiftedSolver(odesolver::ThetaMethod)
  dt,θ = odesolver.dt,odesolver.θ
  δ = dt*(θ-1)
  sysslvr = odesolver.sysslvr
  ShiftedSolver(sysslvr,δ)
end

front_shift!(solver::ShiftedSolver,r::TransientRealization) = shift!(r,solver.δ)
back_shift!(solver::ShiftedSolver,r::TransientRealization) = shift!(r,-solver.δ)

_get_realization(nlop::NonlinearParamOperator) = @abstractmethod
_get_realization(nlop::GenericParamNonlinearOperator) = nlop.μ
_get_realization(nlop::LinNonlinParamOperator) = _get_realization(nlop.op_nonlinear)

function _update_paramcache!(nlop::NonlinearParamOperator,r::TransientRealization)
  @abstractmethod
end

function _update_paramcache!(nlop::GenericParamNonlinearOperator,r::TransientRealization)
  update_paramcache!(nlop.paramcache,nlop.op,r)
end

function _update_paramcache!(nlop::LinNonlinParamOperator,r::TransientRealization)
  _update_paramcache!(nlop.op_linear,r)
  _update_paramcache!(nlop.op_nonlinear,r)
end

function Algebra.solve!(
  x̂::AbstractVector,
  solver::ShiftedSolver,
  nlop::NonlinearParamOperator,
  syscache)

  r = _get_realization(nlop)
  front_shift!(solver,r)
  _update_paramcache!(nlop,r)
  t = @timed _update_paramcache!(nlop,r)
  solve!(x̂,solver.sysslvr,nlop,syscache)
  back_shift!(solver,r)
end
