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
  values = _collect_param_solutions(sol)
  t = @timed values = _collect_param_solutions(sol)
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

function _collect_param_solutions(sol)
  @notimplemented
end

function _collect_param_solutions(sol::ODEParamSolution{<:ConsecutiveParamVector{T}}) where T
  u0item = testitem(sol.u0)
  ncols = num_params(sol.r)*num_times(sol.r)
  values = similar(u0item,T,(size(u0item,1),ncols))
  for (k,(rk,uk)) in enumerate(sol)
    _collect_solutions!(values,uk,k)
  end
  return ConsecutiveParamArray(values)
end

function _collect_param_solutions(sol::ODEParamSolution{<:BlockParamVector{T}}) where T
  u0item = testitem(sol.u0)
  ncols = num_params(sol.r)*num_times(sol.r)
  values = map(b -> ConsecutiveParamArray(similar(b,T,(size(b,1),ncols))),blocks(u0item))
  for (k,(rk,uk)) in enumerate(sol)
    for i in 1:blocklength(u0item)
      _collect_solutions!(values[i].data,blocks(uk)[i],k)
    end
  end
  return mortar(values)
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
