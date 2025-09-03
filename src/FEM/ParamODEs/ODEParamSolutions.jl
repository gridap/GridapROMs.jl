"""
    struct ODEParamSolution{V} <: ODESolution
      solver::ODESolver
      odeop::ODEParamOperator
      r::TransientRealization
      us0::Tuple{Vararg{V}}
    end
"""
struct ODEParamSolution{V} <: ODESolution
  solver::ODESolver
  odeop::ODEParamOperator
  r::TransientRealization
  u0::V
end

function Base.iterate(sol::ODEParamSolution)
  # initialize
  r0 = get_at_time(sol.r,:initial)
  state0,odecache = ode_start(sol.solver,sol.odeop,r0,sol.u0)

  # march
  statef = copy.(state0)
  rf,statef = ode_march!(statef,sol.solver,sol.odeop,r0,state0,odecache)

  # finish
  uf = copy(sol.u0)
  uf = ode_finish!(uf,sol.solver,sol.odeop,rf,statef,odecache)

  state = (rf,statef,state0,uf,odecache)
  return (rf,uf),state
end

function Base.iterate(sol::ODEParamSolution,state)
  r0,state0,statef,uf,odecache = state

  if get_times(r0) >= get_final_time(sol.r) - eps()
    return nothing
  end

  # march
  rf,statef = ode_march!(statef,sol.solver,sol.odeop,r0,state0,odecache)

  # finish
  uf = ode_finish!(uf,sol.solver,sol.odeop,rf,statef,odecache)

  state = (rf,statef,state0,uf,odecache)
  return (rf,uf),state
end

function Base.collect(sol::ODEParamSolution)
  values = collect_param_solutions(sol)
  t = @timed values = collect_param_solutions(sol)
  tracker = CostTracker(t;name="FEM time marching",nruns=num_params(sol.r))
  return values,tracker
end

function Algebra.solve(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealization,
  u0::AbstractVector)

  ODEParamSolution(solver,odeop,r,u0)
end

function Algebra.solve(
  solver::ODESolver,
  odeop::SplitODEParamOperator,
  r::TransientRealization,
  u0::AbstractVector)

  solve(solver,set_domains(odeop),r,u0)
end

function Algebra.solve(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealization,
  uh0::Function)

  params = get_params(r)
  u0 = get_free_dof_values(uh0(params))
  solve(solver,odeop,r,u0)
end

# utils

initial_condition(sol::ODEParamSolution) = sol.u0

function collect_param_solutions(sol)
  @notimplemented
end

function collect_param_solutions(sol::ODEParamSolution{<:ConsecutiveParamVector{T}}) where T
  u0item = testitem(sol.u0)
  ncols = num_params(sol.r)*num_times(sol.r)
  sols = _allocate_solutions(sol.u0,ncols)
  for (k,(rk,uk)) in enumerate(sol)
    _collect_solutions!(sols,uk,k)
  end
  return sols
end

function collect_param_solutions(sol::ODEParamSolution{<:BlockParamVector{T}}) where T
  u0item = testitem(sol.u0)
  ncols = num_params(sol.r)*num_times(sol.r)
  sols = _allocate_solutions(sol.u0,ncols)
  for (k,(rk,uk)) in enumerate(sol)
    for i in 1:blocklength(u0item)
      _collect_solutions!(blocks(sols)[i],blocks(uk)[i],k)
    end
  end
  return sols
end

function _allocate_solutions(u0::ConsecutiveParamVector{T},ncols) where T
  data = similar(u0,T,(size(u0,1),ncols))
  return ConsecutiveParamArray(data)
end

function _allocate_solutions(u0::BlockParamVector,ncols)
  mortar(b -> _allocate_solutions(b,ncols),blocks(u0))
end

function _collect_solutions!(sols::ConsecutiveParamVector,ui::ConsecutiveParamVector,it::Int)
  _collect_solutions!(get_all_data(sols),ui,it)
end

function _collect_solutions!(
  values::AbstractMatrix,
  ui::ConsecutiveParamVector,
  it::Int)

  datai = get_all_data(ui)
  nparams = param_length(ui)
  for ip in 1:nparams
    itp = (it-1)*nparams+ip
    for is in axes(values,1)
      @inbounds v = datai[is,ip]
      @inbounds values[is,itp] = v
    end
  end
end
