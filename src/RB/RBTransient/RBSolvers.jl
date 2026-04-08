function RBSteady.RBSolver(
  fesolver::ODESolver,
  reduction::Reduction;
  nparams_res=20,
  nparams_jacs=ntuple(_ -> 20,get_time_order(fesolver)+1),
  kwargs...)

  tcomb = TimeCombination(fesolver)
  residual_reduction = HighDimHyperReduction(tcomb,reduction;nparams=nparams_res,kwargs...)
  jacobian_reduction = ntuple(
    i -> HighDimHyperReduction(
      CombinationOrder{i}(tcomb),reduction;
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
  kwargs...)

  tcomb = TimeCombination(fesolver)
  residual_reduction = LocalHighDimHyperReduction(tcomb,reduction;nparams=nparams_res,kwargs...)
  jacobian_reduction = ntuple(
    i -> LocalHighDimHyperReduction(
      CombinationOrder{i}(tcomb),reduction;
      nparams=nparams_jacs[i],kwargs...),
    Val(get_time_order(fesolver)+1)
  )
  RBSolver(fesolver,reduction,residual_reduction,jacobian_reduction)
end

const TransientRBSolver{A<:ODESolver,B,C,D} = RBSolver{A,B,C,D}

RBSteady.num_jac_params(s::TransientRBSolver) = num_params(first(s.jacobian_reduction))
get_system_solver(s::TransientRBSolver) = ShiftedSolver(s.fesolver)

function RBSteady.solution_snapshots(
  solver::RBSolver,
  feop::ODEParamOperator,
  r::TransientRealisation,
  args...)

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
  args...)

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
  s::AbstractSnapshots)

  fesolver = get_fe_solver(solver)
  sres = select_snapshots(s,RBSteady.res_params(solver))
  rres = get_realisation(sres)
  b = residual(fesolver,odeop,rres,sres)
  ib = get_dof_map_at_domains(odeop)
  return Snapshots(b,ib,rres)
end

function RBSteady.residual_snapshots(
  solver::RBSolver,
  op::ODEParamOperator{LinearNonlinearParamODE},
  s::AbstractSnapshots)

  res_lin = residual_snapshots(solver,get_linear_operator(op),s)
  res_nlin = residual_snapshots(solver,get_nonlinear_operator(op),s)
  return (res_lin,res_nlin)
end

function RBSteady.jacobian_snapshots(
  solver::RBSolver,
  odeop::ODEParamOperator,
  s::AbstractSnapshots)

  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,RBSteady.jac_params(solver))
  rjac = get_realisation(sjac)
  A = jacobian(fesolver,odeop,rjac,sjac)
  iA = get_sparse_dof_map_at_domains(odeop)
  jac_reduction = RBSteady.get_jacobian_reduction(solver)
  sA = ()
  for (reda,a,ia) in zip(jac_reduction,A,iA)
    sa = Snapshots(a,ia,rjac)
    sA = (sA...,select_snapshots(sa,1:num_params(reda)))
  end
  return sA
end

function RBSteady.jacobian_snapshots(
  solver::RBSolver,
  op::ODEParamOperator{LinearNonlinearParamODE},
  s::AbstractSnapshots)

  jac_lin = jacobian_snapshots(solver,get_linear_operator(op),s)
  jac_nlin = jacobian_snapshots(solver,get_nonlinear_operator(op),s)
  return (jac_lin,jac_nlin)
end

# Note: we do not need to touch the initial condition, as it is already modeled by the 
# ROM; for more details, check where the function `get_initial_param_data` (called 
# inside `residual_snapshots` and `jacobian_snapshots`) leads to!
function Algebra.solve(
  solver::RBSolver,
  op::NonlinearOperator,
  r::TransientRealisation,
  us0)

  trial = get_trial(op)(r)
  x̂ = zero_free_values(trial)

  nlop = parameterise(op,r)
  syscache = allocate_systemcache(nlop,x̂)

  fesolver = get_system_solver(solver)
  t = @timed solve!(x̂,fesolver,nlop,syscache)
  stats = CostTracker(t,nruns=num_params(r),name="RB")

  return x̂,stats
end

# local solver

function Algebra.solve(
  solver::RBSolver,
  op::AbstractLocalRBOperator,
  r::TransientRealisation,
  us0
  )

  t = @timed x̂vec = map(get_params(r)) do μ
    opμt = get_local(op,μ)
    rμt = _to_realisation(r,μ)
    x̂, = solve(solver,opμt,rμt,us0)
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