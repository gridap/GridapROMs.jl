for S in (:NonlinearSolver,:LinearSolver)
  @eval begin
    function Algebra.solve!(u::AbstractVector,solver::$S,op::JointParamOperator,r::Realization)
      nlop = parameterize(op,r)
      syscache = allocate_systemcache(nlop,u)
      t = @timed solve!(u,solver,nlop,syscache)
      stats = CostTracker(t,name="Solver";nruns=num_params(r))
      stats
    end

    function Algebra.solve!(u::AbstractVector,solver::$S,op::SplitParamOperator,r::Realization)
      solve!(u,solver,set_domains(op),r)
    end
  end
end

function Algebra.solve(solver::NonlinearSolver,op::ParamOperator,r::Realization)
  U = get_trial(op)(r)
  u = zero_free_values(U)
  stats = solve!(u,solver,op,r)
  u,stats
end

function Algebra.solve!(uh,fesolver::FESolver,feop::ParamFEOperator,r::Realization)
  u = get_free_dof_values(uh)
  solver = get_solver(fesolver)
  op = get_algebraic_operator(feop)
  stats = solve!(uh,solver,op,r)
  uh,stats
end

function Algebra.solve(fesolver::FESolver,feop::ParamFEOperator,r::Realization)
  U = get_trial(op)(r)
  uh = zero(U)
  solve!(uh,fesolver,feop,r)
end

get_solver(solver::LinearFESolver) = solver.ls
get_solver(solver::NonlinearFESolver) = solver.nls
