function RBSteady.RBSolver(
  fesolver::ODESolver,
  reduction::Reduction;
  nparams_res=20,
  nparams_jacs=ntuple(_ -> 20,get_time_order(fesolver)+1),
  kwargs...
  )

  c = TimeCombination(fesolver)
  residual_reduction = HighDimHyperReduction(c,reduction;nparams=nparams_res,kwargs...)
  jacobian_reduction = ntuple(
    i -> HighDimHyperReduction(
      CombinationOrder{i}(c),reduction;
      nparams=nparams_jacs[i],kwargs...
    ),
    Val(get_time_order(fesolver)+1)
  )
  RBSolver(fesolver,reduction,residual_reduction,jacobian_reduction)
end

function RBSteady.RBSolver(
  fesolver::ODESolver,
  reduction::LocalReduction;
  nparams_res=20,
  nparams_jacs=ntuple(_ -> 20,get_time_order(fesolver)+1),
  kwargs...
  )

  c = TimeCombination(fesolver)
  residual_reduction = LocalHighDimHyperReduction(c,reduction;nparams=nparams_res,kwargs...)
  jacobian_reduction = ntuple(
    i -> LocalHighDimHyperReduction(
      CombinationOrder{i}(c),reduction;
      nparams=nparams_jacs[i],kwargs...),
    Val(get_time_order(fesolver)+1)
  )
  RBSolver(fesolver,reduction,residual_reduction,jacobian_reduction)
end

const TransientRBSolver{A<:ODESolver,B,C,D} = RBSolver{A,B,C,D}

ParamODEs.TimeCombination(s::TransientRBSolver) = TimeCombination(get_fe_solver(s))

RBSteady.num_jac_params(s::TransientRBSolver) = maximum(map(num_params,s.jacobian_reduction))

function RBSteady.solution_snapshots(
  solver::RBSolver,
  feop::ODEParamOperator,
  r::TransientRealisation,
  args...
  )

  fesolver = get_fe_solver(solver)
  sol = solve(fesolver,feop,r,args...)
  values,stats = collect(sol)
  initial_values = initial_conditions(sol)
  i = get_dof_map(feop)
  snaps = Snapshots(values,initial_values,i,r)
  return snaps,stats
end

# not needed
function RBSteady.solution_snapshots(
  fesolver::ODESolver,
  op::ODEParamOperator,
  r::TransientRealisation,
  args...
  )

  sol = solve(fesolver,op,r,args...)
  values,stats = collect(sol)
  initial_values = initial_conditions(sol)
  i = get_dof_map(op)
  snaps = Snapshots(values,initial_values,i,r)
  return snaps,stats
end

function RBSteady.residual_snapshots(
  solver::RBSolver,
  odeop::ODEParamOperator,
  s::AbstractSnapshots
  )

  c = TimeCombination(solver)
  sres = select_snapshots(s,RBSteady.res_params(solver))
  rres = get_realisation(sres)
  b = spacetime_residual(c,odeop,sres)
  ib = get_dof_map(odeop,b)
  return Snapshots(b,ib,rres)
end

function RBSteady.residual_snapshots(
  solver::RBSolver,
  op::ODEParamOperator{LinearNonlinearParamODE},
  s::AbstractSnapshots
  )

  res_lin = residual_snapshots(solver,get_linear_operator(op),s)
  res_nlin = residual_snapshots(solver,get_nonlinear_operator(op),s)
  return (res_lin,res_nlin)
end

function RBSteady.jacobian_snapshots(
  solver::RBSolver,
  odeop::ODEParamOperator,
  s::AbstractSnapshots
  )

  c = TimeCombination(solver)
  sjac = select_snapshots(s,RBSteady.jac_params(solver))
  rjac = get_realisation(sjac)
  A = spacetime_jacobian(c,odeop,sjac)
  jac_reduction = RBSteady.get_jacobian_reduction(solver)
  sA = ()
  for (reda,a) in zip(jac_reduction,A)
    ia = get_sparse_dof_map(odeop,a)
    sa = Snapshots(a,ia,rjac)
    sA = (sA...,select_snapshots(sa,1:num_params(reda)))
  end
  return sA
end

function RBSteady.jacobian_snapshots(
  solver::RBSolver,
  op::ODEParamOperator{LinearNonlinearParamODE},
  s::AbstractSnapshots
  )

  jac_lin = jacobian_snapshots(solver,get_linear_operator(op),s)
  jac_nlin = jacobian_snapshots(solver,get_nonlinear_operator(op),s)
  return (jac_lin,jac_nlin)
end

function Algebra.solve(
  solver::RBSolver,
  op::NonlinearOperator,
  r::TransientRealisation,
  us0::Tuple{Vararg{AbstractVector}}
  )

  trial = get_trial(op)(r)
  x̂ = zero_free_values(trial)

  fesolver = SpaceTimeSolver(solver,us0)
  nlop = parameterise(op,r)
  syscache = allocate_systemcache(nlop,x̂)

  t = @timed solve!(x̂,fesolver,nlop,syscache)
  stats = CostTracker(t,nruns=num_params(r),name="RB")

  return x̂,stats
end

function Algebra.solve(
  solver::RBSolver,
  op::NonlinearOperator,
  r::TransientRealisation,
  uhs0::Tuple{Vararg{Function}}
  )

  trial = get_trial(op)
  params = get_params(r)
  us0 = ()
  for uh0 in uhs0
    u0 = get_free_dof_values(uh0(params))
    û0 = space_project(trial,u0)
    us0 = (us0...,reduced_vector(û0,u0))
  end
  if length(us0) < get_order(op) + 1
    fesolver = get_fe_solver(solver)
    us0 = add_initial_conditions(fesolver,op,r,us0)
  end
  solve(solver,op,r,us0)
end

function Algebra.solve(
  solver::RBSolver,
  op::NonlinearOperator,
  r::TransientRealisation,
  u0::Any
  )

  solve(solver,op,r,(u0,))
end

# local solver

function Algebra.solve(
  solver::RBSolver,
  op::AbstractLocalRBOperator,
  r::TransientRealisation,
  ush0::Tuple{Vararg{AbstractVector}}
  )

  @notimplemented "When using local reduced operators, provide the initial condition as functions 
  of the parameters, so that they can be localised. See the other solve method for an example."
end

function Algebra.solve(
  solver::RBSolver,
  op::AbstractLocalRBOperator,
  r::TransientRealisation,
  ush0::Tuple{Vararg{Function}}
  )

  t = @timed x̂vec = map(get_params(r)) do μ
    opμt = get_local(op,μ)
    rμt = _to_realisation(r,μ)
    x̂, = solve(solver,opμt,rμt,ush0)
    x̂
  end
  x̂ = param_cat(x̂vec)
  stats = CostTracker(t,nruns=num_params(r),name="RB")
  return (x̂,stats)
end

function _to_realisation(r::TransientRealisation,μ::AbstractVector)
  all_times = [get_initial_time(r),get_times(r)...]
  TransientRealisation(Realisation([μ]),all_times)
end

# utils 

get_time_order(::ODESolver) = @abstractmethod
get_time_order(::ThetaMethod) = 1
get_time_order(::GeneralizedAlpha1) = 1
get_time_order(::GeneralizedAlpha2) = 2