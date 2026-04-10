"""
    struct ODEParamSolution{V} <: ODESolution
      solver::ODESolver
      odeop::ODEParamOperator
      r::TransientRealisation
      us0::Tuple{Vararg{V}}
    end
"""
struct ODEParamSolution{V} <: ODESolution
  solver::ODESolver
  odeop::ODEParamOperator
  r::TransientRealisation
  us0::Tuple{Vararg{V}}
end

function Base.iterate(sol::ODEParamSolution)
  # initialize
  r0 = get_at_time(sol.r,:initial)
  state0,odecache = ode_start(sol.solver,sol.odeop,r0,sol.us0)

  # march
  statef = copy.(state0)
  rf,statef = ode_march!(statef,sol.solver,sol.odeop,r0,state0,odecache)

  # finish
  uf = copy(first(sol.us0))
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
  t = @timed values = collect_param_solutions(sol)
  tracker = CostTracker(t;name="FEM time marching",nruns=num_params(sol.r))
  return values,tracker
end

function Algebra.solve(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  us0::Tuple{Vararg{AbstractVector}}
  )

  if length(us0) < get_order(odeop) + 1
    us0 = add_initial_conditions(solver,odeop,r,us0)
  end
  ODEParamSolution(solver,odeop,r,us0)
end

function Algebra.solve(
  solver::ODESolver,
  odeop::SplitODEParamOperator,
  r::TransientRealisation,
  us0::Tuple{Vararg{AbstractVector}}
  )

  solve(solver,set_domains(odeop),r,us0)
end

function Algebra.solve(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  uhs0::Tuple{Vararg{Function}}
  )

  params = get_params(r)
  us0 = ()
  for uh0 in uhs0
    us0 = (us0...,get_free_dof_values(uh0(params)))
  end
  solve(solver,odeop,r,us0)
end

function Algebra.solve(
  solver::ODESolver,
  odeop::ODEParamOperator,
  r::TransientRealisation,
  uh0::Any
  )

  solve(solver,odeop,r,(uh0,))
end

# utils

initial_conditions(sol::ODEParamSolution) = sol.us0

function collect_param_solutions(sol)
  @notimplemented
end

function collect_param_solutions(sol::ODEParamSolution{<:ConsecutiveParamVector{T}}) where T
  u0 = first(sol.us0)
  ncols = num_params(sol.r)*num_times(sol.r)
  sols = _allocate_solutions(u0,ncols)
  for (k,(rk,uk)) in enumerate(sol)
    _collect_solutions!(sols,uk,k)
  end
  return sols
end

function collect_param_solutions(sol::ODEParamSolution{<:BlockParamVector{T}}) where T
  u0 = first(sol.us0)
  u0item = testitem(u0)
  ncols = num_params(sol.r)*num_times(sol.r)
  sols = _allocate_solutions(u0,ncols)
  for (k,(rk,uk)) in enumerate(sol)
    for i in 1:blocklength(u0item)
      _collect_solutions!(blocks(sols)[i],blocks(uk)[i],k)
    end
  end
  return sols
end

function _allocate_solutions(u0::ConsecutiveParamVector{T},ncols) where T
  data = similar(u0,T,(innerlength(u0),ncols))
  return ConsecutiveParamArray(data)
end

function _allocate_solutions(u0::BlockParamVector,ncols)
  mortar(map(b -> _allocate_solutions(b,ncols),blocks(u0)))
end

function _collect_solutions!(
  sols::ConsecutiveParamVector,
  ui::ConsecutiveParamVector,
  it::Int
  )
  _collect_solutions!(get_all_data(sols),ui,it)
end

function _collect_solutions!(
  values::AbstractMatrix,
  ui::ConsecutiveParamVector,
  it::Int
  )

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
