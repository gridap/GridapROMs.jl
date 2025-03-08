# check TransientMDEIMReduction for more details
time_combinations(fesolver::ODESolver) = @notimplemented

function time_combinations(fesolver::GeneralizedAlpha1)
  combine_res(x) = nothing
  combine_jac(x,y) = nothing
  combine_djac(x,y) = nothing
  return combine_res,combine_jac,combine_djac
end

function time_combinations(fesolver::GeneralizedAlpha2)
  combine_res(x) = nothing
  combine_jac(x,y) = nothing
  combine_djac(x,y) = nothing
  combine_ddjac(x,y) = nothing
  return combine_res,combine_jac,combine_djac,combine_ddjac
end

function time_combinations(fesolver::ThetaMethod)
  dt,θ = fesolver.dt,fesolver.θ
  combine_res(x) = x
  combine_jac(x,y) = θ*x+(1-θ)*y
  combine_djac(x,y) = (x-y)/dt
  return combine_res,combine_jac,combine_djac
end

function RBSteady.RBSolver(
  fesolver::ODESolver,
  state_reduction::Reduction;
  nparams_res=20,
  nparams_jac=20,
  nparams_djac=nparams_jac)

  red_style = ReductionStyle(state_reduction)
  cres,cjac,cdjac = time_combinations(fesolver)

  residual_reduction = TransientMDEIMReduction(cres,red_style;nparams=nparams_res)
  jac_reduction = TransientMDEIMReduction(cjac,red_style;nparams=nparams_jac)
  djac_reduction = TransientMDEIMReduction(cdjac,red_style;nparams=nparams_djac)
  jacobian_reduction = (jac_reduction,djac_reduction)

  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

RBSteady.num_jac_params(s::RBSolver{<:ODESolver}) = num_params(first(s.jacobian_reduction))
get_system_solver(s::RBSolver{<:ODESolver}) = s.fesolver.sysslvr

function RBSteady.solution_snapshots(
  solver::RBSolver,
  feop::ODEParamOperator,
  r::TransientRealization,
  args...)

  fesolver = get_fe_solver(solver)
  sol = solve(fesolver,feop,r,args...)
  values,stats = collect(sol)
  initial_values = initial_condition(sol)
  i = get_dof_map(feop)
  snaps = Snapshots(values,initial_values,i,r)
  return snaps,stats
end

function RBSteady.residual_snapshots(
  solver::RBSolver,
  odeop::ODEParamOperator,
  s::AbstractSnapshots;
  nparams=RBSteady.res_params(solver))

  fesolver = get_fe_solver(solver)
  sres = select_snapshots(s,nparams)
  us_res = get_param_data(sres)
  us0_res = get_initial_data(sres)
  r_res = get_realization(sres)
  b = residual(fesolver,odeop,r_res,us_res,us0_res)
  ib = get_dof_map_at_domains(odeop)
  return Snapshots(b,ib,r_res)
end

function RBSteady.residual_snapshots(
  solver::RBSolver,
  op::ODEParamOperator{LinearNonlinearParamODE},
  s::AbstractSnapshots;
  kwargs...)

  res_lin = residual_snapshots(solver,get_linear_operator(op),s;kwargs...)
  res_nlin = residual_snapshots(solver,get_nonlinear_operator(op),s;kwargs...)
  return (res_lin,res_nlin)
end

function RBSteady.jacobian_snapshots(
  solver::RBSolver,
  odeop::ODEParamOperator,
  s::AbstractSnapshots;
  nparams=RBSteady.jac_params(solver))

  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,nparams)
  us_jac = get_param_data(sjac)
  us0_jac = get_initial_data(sjac)
  r_jac = get_realization(sjac)
  A = jacobian(fesolver,odeop,r_jac,us_jac,us0_jac)
  iA = get_sparse_dof_map_at_domains(odeop)
  jac_reduction = RBSteady.get_jacobian_reduction(solver)
  sA = ()
  for (reda,a,ia) in zip(jac_reduction,A,iA)
    sa = Snapshots(a,ia,r_jac)
    sA = (sA...,select_snapshots(sa,1:num_params(reda)))
  end
  return sA
end

function RBSteady.jacobian_snapshots(
  solver::RBSolver,
  op::ODEParamOperator{LinearNonlinearParamODE},
  s::AbstractSnapshots;
  kwargs...)

  jac_lin = jacobian_snapshots(solver,get_linear_operator(op),s;kwargs...)
  jac_nlin = jacobian_snapshots(solver,get_nonlinear_operator(op),s;kwargs...)
  return (jac_lin,jac_nlin)
end

# Solve a POD-MDEIM problem

function Algebra.solve(
  solver::RBSolver,
  op::NonlinearOperator,
  r::TransientRealization,
  xh0::Union{Function,AbstractVector})

  trial = get_trial(op)(r)
  x̂ = zero_free_values(trial)

  nlop = parameterize(op,r)
  syscache = allocate_systemcache(nlop,x̂)

  fesolver = get_system_solver(solver)
  t = @timed solve!(x̂,fesolver,nlop,syscache)
  stats = CostTracker(t,nruns=num_params(r),name="RB")

  return x̂,stats
end
