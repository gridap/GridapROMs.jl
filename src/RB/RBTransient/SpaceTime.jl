struct SpaceTimeParamOperator <: NonlinearParamOperator
  op::ODEParamOperator
  r::TransientRealisation
  usx::Tuple{Vararg{AbstractParamVector}}
  ws::Tuple{Vararg{Real}}
  paramcache::AbstractParamCache
end

function SpaceTimeParamOperator(op::ODEParamOperator,r::TransientRealisation)
  nstates = get_order(op)+1
  trial = get_trial(op)(r)
  u = zero_free_values(trial)
  us0 = ntuple(_ -> u,Val{nstates}())
  ws = ntuple(_ -> 1,Val{nstates}())

  us = allocate_time_combination(u,us0)
  paramcache = allocate_paramcache(op,r)
  SpaceTimeParamOperator(op,r,us,ws,paramcache)
end

function ParamDataStructures.parameterise(op::ODEParamOperator,r::TransientRealisation)
  SpaceTimeParamOperator(op,r)
end

function Algebra.allocate_residual(
  nlop::SpaceTimeParamOperator,
  x::AbstractVector
  )

  allocate_residual(nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

# Current state is unused in residual/jacobian evaluations. This is because the time combination 
# is evaluated BEFORE (see _set_initial_condition! at the very start of the solve, and time_combination!
# in the nonlinear solver loop).
function Algebra.residual!(
  b::AbstractVector,
  nlop::SpaceTimeParamOperator,
  x::AbstractVector
  )

  residual!(b,nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::SpaceTimeParamOperator,
  x::AbstractVector
  )

  allocate_jacobian(nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::SpaceTimeParamOperator,
  x::AbstractVector
  )

  jacobian!(A,nlop.op,nlop.r,nlop.usx,nlop.ws,nlop.paramcache)
  A
end

struct SpaceTimeSolver{A<:ODESolver,B<:NonlinearSolver} <: NonlinearSolver
  solver::A
  us0::Tuple{Vararg{AbstractParamVector}}
  function SpaceTimeSolver(solver::ODESolver,us0::Tuple{Vararg{AbstractParamVector}})
    A = typeof(solver)
    B = typeof(get_solver(solver))
    new{A,B}(solver,us0)
  end
end

function SpaceTimeSolver(solver::RBSolver,args...)
  SpaceTimeSolver(get_fe_solver(solver),args...)
end

get_shift(s::SpaceTimeSolver) = @notimplemented 

function get_shift(s::SpaceTimeSolver{<:ThetaMethod})
  s.solver.dt*(1-s.solver.θ)
end

function get_shift(s::SpaceTimeSolver{<:GeneralizedAlpha1})
  s.solver.dt*(1-s.solver.αf)
end

function get_shift(s::SpaceTimeSolver{<:GeneralizedAlpha2})
  s.solver.dt*s.solver.αf
end

back_shift!(s::SpaceTimeSolver,r::TransientRealisation) = shift!(r,-get_shift(s))
front_shift!(s::SpaceTimeSolver,r::TransientRealisation) = shift!(r,get_shift(s))

get_solver(s::NonlinearSolver) = s
get_solver(s::SpaceTimeSolver) = get_solver(s.solver)
get_solver(s::ODESolver) = @notimplemented
get_solver(s::ThetaMethod) = s.sysslvr
get_solver(s::GeneralizedAlpha1) = s.sysslvr
get_solver(s::GeneralizedAlpha2) = s.sysslvr

function Algebra.solve!(
  x̂::AbstractParamVector,
  solver::SpaceTimeSolver,
  nlop::SpaceTimeParamOperator,
  syscache
  )

  back_shift!(solver,nlop.r)
  update_paramcache!(nlop.paramcache,nlop.op,nlop.r)
  _set_initial_condition!(nlop,solver)
  _st_solve!(x̂,solver,nlop,syscache)
  front_shift!(solver,nlop.r)
end

function ParamODEs.time_combination!(
  usx::NTuple{N,AbstractParamVector},
  c::TimeCombination,
  u::RBParamVector,
  us0::NTuple{N,AbstractParamVector}
  ) where N

  time_combination!(usx,c,u.fe_data,us0)
end

function ParamODEs.allocate_time_combination(
  u::RBParamVector, 
  us0::NTuple{N,AbstractParamVector}
  ) where N

  allocate_time_combination(u.fe_data,us0)
end

# utils 

const SpaceTimeLinearSolver{A<:ODESolver} = SpaceTimeSolver{A,<:LinearSolver}

function _st_solve!(
  x::AbstractParamVector,
  s::SpaceTimeLinearSolver,
  op::SpaceTimeParamOperator,
  cache
  )
    
  solve!(x,get_solver(s),op,cache)
end

function _st_solve!(
  x::AbstractParamVector,
  s::SpaceTimeSolver,
  op::SpaceTimeParamOperator,
  cache::Nothing
  )

  cache = allocate_systemcache(op,x)
  _st_solve!(x,s,op,cache)
end

const SpaceTimeNewtonSolver{A<:ODESolver} = SpaceTimeSolver{A,<:NewtonSolver}

function _st_solve!(
  x::AbstractParamVector,
  s::SpaceTimeNewtonSolver,
  op::SpaceTimeParamOperator,
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
  op::SpaceTimeParamOperator,
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
  dx,ns,s,op
  )

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

    time_combination!(op.usx,tcomb,x,s.us0)
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
  dx,ns,s,op
  )

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
    time_combination!(op.usx,tcomb,x,s.us0)
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

function _set_initial_condition!(nlop::SpaceTimeParamOperator,solver::SpaceTimeSolver)
  c = TimeCombination(solver.solver)
  zero_time_combination!(nlop.usx,c,solver.us0)
end