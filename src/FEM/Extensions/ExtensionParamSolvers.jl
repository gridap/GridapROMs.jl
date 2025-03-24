function Algebra.solve(solver::NonlinearSolver,op::ExtensionParamOperator,r::Realization)
  U = get_trial(op)(r)
  u = zero_free_values(U)
  stats = solve!(u,solver,op.op,r)
  bg_u = extend_free_values(U,u)
  bg_u,stats
end
