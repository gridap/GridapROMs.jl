struct SpaceTimeSolver{A<:ODESolver,B<:NonlinearSolver} <: NonlinearSolver
  solver::A
  usx::Tuple{Vararg{AbstractParamVector}}
  us0::Tuple{Vararg{AbstractParamVector}}
  function SpaceTimeSolver(
    solver::A,
    usx::Tuple{Vararg{AbstractParamVector}},
    us0::Tuple{Vararg{AbstractParamVector}},
    ) where A 
    
    B = typeof(get_solver(solver))
    new{A,B}(solver,usx,us0)
  end
end

const SpaceTimeLinearSolver{A<:ODESolver} = SpaceTimeSolver{A,<:LinearSolver}

function SpaceTimeSolver(
  solver::ODESolver,
  u::AbstractParamVector,
  us0::Tuple{Vararg{AbstractParamVector}}
  )

  c = TimeCombination(solver)
  us = time_combination(c,u,us0)
  SpaceTimeSolver(c,us,us0)
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

get_solver(s::SpaceTimeSolver) = @notimplemented
get_solver(s::SpaceTimeSolver{<:ThetaMethod}) = s.solver.sysslvr
get_solver(s::SpaceTimeSolver{<:GeneralizedAlpha1}) = s.solver.sysslvr
get_solver(s::SpaceTimeSolver{<:GeneralizedAlpha2}) = s.solver.sysslvr

function Algebra.solve!(
  x̂::AbstractParamVector,
  solver::SpaceTimeSolver,
  nlop::NonlinearParamOperator,
  syscache
  )

  r = _get_realisation(nlop)
  front_shift!(solver,r)
  _update_paramcache!(nlop,r)
  _st_solve!(x̂,solver,nlop,syscache)
  back_shift!(solver,r)
end

# utils 

function _st_solve!(
  x::AbstractParamVector,
  s::SpaceTimeLinearSolver,
  op::NonlinearParamOperator,
  cache
  )
    
  solve!(x,get_solver(s),op,cache)
end

function _st_solve!(
  x::AbstractParamVector,
  s::SpaceTimeSolver,
  op::NonlinearParamOperator,
  cache::Nothing)

  cache = allocate_systemcache(op,x)
  _st_solve!(x,s,op,cache)
end

const SpaceTimeNewtonSolver{A<:ODESolver} = SpaceTimeSolver{A,<:NewtonSolver}

function _st_solve!(
  x::AbstractParamVector,
  s::SpaceTimeNewtonSolver,
  op::NonlinearParamOperator,
  cache::SystemCache
  ) 

  nls = get_solver(s)

  fill!(x,zero(eltype(x)))
  update_systemcache!(op,x)

  @unpack A,b = cache
  residual!(b,op,x)
  jacobian!(A,op,x)

  A_item = testitem(A)
  x_item = testitem(x)
  dx = allocate_in_domain(A_item)
  fill!(dx,zero(eltype(dx)))
  ss = symbolic_setup(nls.ls,A_item)
  ns = numerical_setup(ss,A_item,x_item)

  _st_solve_nr!(x,A,b,dx,ns,s,op)
  return NonlinearSolvers.NewtonCache(A,b,dx,ns)
end

function _st_solve!(
  x::AbstractParamVector,
  s::SpaceTimeNewtonSolver,
  op::NonlinearParamOperator,
  cache::NonlinearSolvers.NewtonCache
  ) 

  update_systemcache!(op,x)

  @unpack A,b,dx,ns = cache
  residual!(b,op,x)
  jacobian!(A,op,x)

  _st_solve_nr!(x,A,b,dx,ns,s,op)
  return cache
end

function _st_solve_nr!(
  x::AbstractParamVector,
  A::AbstractParamMatrix,
  b::AbstractParamVector,
  dx,ns,s,op)

  tcomb = TimeCombination(s.solver)
  nls = get_solver(s)
  log = nls.log

  res = norm(b)
  done = LinearSolvers.init!(log,res)

  while !done
    @inbounds for i in param_eachindex(x)
      xi = param_getindex(x,i)
      Ai = param_getindex(A,i)
      bi = param_getindex(b,i)
      numerical_setup!(ns,Ai)
      rmul!(bi,-1)
      solve!(dx,ns,bi)
      xi .+= dx
    end

    time_combination!(s.usx,tcomb,x,s.us0)
    residual!(b,op,x)
    res  = norm(b)
    done = LinearSolvers.update!(log,res)

    if !done
      jacobian!(A,op,x)
    end
  end

  LinearSolvers.finalize!(log,res)
  return x
end

function _st_solve_nr!(
  x::RBParamVector,
  A::AbstractParamMatrix,
  b::AbstractParamVector,
  dx,ns,s,op)

  tcomb = TimeCombination(s.solver)
  nls = get_solver(s)
  log = nls.log
  RBSteady.change_tols!(log)

  trial = RBSteady._get_trial(op)

  res = norm(b)
  done = LinearSolvers.init!(log,res)

  while !done
    @inbounds for i in param_eachindex(x)
      xi = param_getindex(x,i)
      Ai = param_getindex(A,i)
      bi = param_getindex(b,i)
      numerical_setup!(ns,Ai)
      rmul!(bi,-1)
      solve!(dx,ns,bi)
      xi .+= dx
    end

    inv_project(trial,x)
    time_combination!(s.usx,tcomb,x,s.us0)
    residual!(b,op,x)
    res  = norm(b)
    done = LinearSolvers.update!(log,res)

    if !done
      jacobian!(A,op,x)
    end
  end

  LinearSolvers.finalize!(log,res)
  return x
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