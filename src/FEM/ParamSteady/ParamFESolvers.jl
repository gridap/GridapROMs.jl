for S in (:NonlinearSolver,:LinearSolver)
  @eval begin
    function Algebra.solve!(
      u::AbstractVector,
      solver::$S,
      op::JointParamOperator,
      r::Realisation
      )
      nlop = parameterise(op,r)
      syscache = allocate_systemcache(nlop,u)
      t = @timed solve!(u,solver,nlop,syscache)
      stats = CostTracker(t,name="Solver";nruns=num_params(r))
      stats
    end

    function Algebra.solve!(
      u::AbstractVector,
      solver::$S,
      op::SplitParamOperator,
      r::Realisation
      )
      solve!(u,solver,set_domains(op),r)
    end
  end
end

function Algebra.solve(solver::NonlinearSolver,op::ParamOperator,r::Realisation)
  u = zero_initial_guess(op,r)
  stats = solve!(u,solver,op,r)
  u,stats
end

for S in (:FESolver,:LinearFESolver)
  @eval begin
    function Algebra.solve!(uh,fesolver::$S,feop::ParamFEOperator,r::Realisation)
      u = get_free_dof_values(uh)
      solver = get_solver(fesolver)
      op = get_algebraic_operator(feop)
      stats = solve!(u,solver,op,r)
      uh,stats
    end
  end
end

function Algebra.solve(fesolver::FESolver,feop::ParamFEOperator,r::Realisation)
  U = get_trial(feop)(r)
  uh = zero(U)
  solve!(uh,fesolver,feop,r)
  uh
end

get_solver(solver::LinearFESolver) = solver.ls
get_solver(solver::NonlinearFESolver) = solver.nls
