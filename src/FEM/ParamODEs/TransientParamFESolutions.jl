"""
    struct TransientParamFESolution{V} <: TransientFESolution
      odesol::ODEParamSolution{V}
      trial
    end

Wrapper around a `TransientParamFEOperator` and `ODESolver` that represents the
parametric solution at a set of time steps. It is an iterator that computes the solution
at each time step in a lazy fashion when accessing the solution.
"""
struct TransientParamFESolution{V} <: TransientFESolution
  odesol::ODEParamSolution{V}
  trial
end

initial_condition(sol::TransientParamFESolution) = initial_condition(sol.odesol)

function TransientParamFESolution(
  solver::ODESolver,
  op::TransientParamFEOperator,
  r::TransientRealization,
  u0)

  odeop = get_algebraic_operator(op)
  odesol = solve(solver,odeop,r,u0)
  trial = get_trial(op)
  TransientParamFESolution(odesol,trial)
end

function Base.iterate(sol::TransientParamFESolution)
  ode_it = iterate(sol.odesol)
  if isnothing(ode_it)
    return nothing
  end
  uf,ode_it_state = ode_it
  timeid = first(ode_it_state)

  rf = sol.odesol.stageop[timeid].r
  Uh = allocate_space(sol.trial,rf)
  Uh = evaluate!(Uh,sol.trial,rf)
  uhf = FEFunction(Uh,uf)

  state = Uh,ode_it_state
  uhf,state
end

function Base.iterate(sol::TransientParamFESolution,state)
  Uh,ode_it_state = state
  ode_it = iterate(sol.odesol,ode_it_state)
  if isnothing(ode_it)
    return nothing
  end
  uf,ode_it_state = ode_it
  timeid = first(ode_it_state)

  rf = sol.odesol.stageop[timeid].r
  Uh = evaluate!(Uh,sol.trial,rf)
  uhf = FEFunction(Uh,uf)

  state = Uh,ode_it_state
  uhf,state
end

function Base.collect(sol::TransientParamFESolution{V}) where V
  odesol = sol.odesol
  ntimes = num_times(odesol.r)

  free_values = Vector{V}(undef,ntimes)
  for (k,uhk) in enumerate(sol)
    uk = get_free_dof_values(uhk)
    free_values[k] = copy(uhk)
  end

  return free_values,odesol.tracker
end
