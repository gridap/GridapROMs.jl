"""
    struct RBSolver{A<:GridapType,B} <: GridapType
      fesolver::A
      state_reduction::Reduction
      residual_reduction::Reduction
      jacobian_reduction::B
    end

Wrapper around a FE solver (e.g. `NonlinearSolver` or `ODESolver` in `Gridap`) with
additional information on the reduced basis (RB) method employed to solve a given
problem dependent on a set of parameters. A RB method is a projection-based
reduced order model where

1. a suitable subspace of a FESpace is sought, of dimension n ≪ Nₕ
2. a matrix-based discrete empirical interpolation method (MDEIM) is performed
  to approximate the manifold of the parametric residuals and jacobians
3. the EIM approximations are compressed with (Petrov-)Galerkin projections
  onto the subspace
4. for every desired choice of parameters, numerical integration is performed, and
  the resulting n × n system of equations is cheaply solved

In particular:

- ϵ: tolerance used in the projection-based truncated proper orthogonal
  decomposition (TPOD) or in the tensor train singular value decomposition (TT-SVD),
  where a basis spanning the reduced subspace is computed; the value of ϵ is
  responsible for selecting the dimension of the subspace, i.e. n = n(ϵ)
- nparams_state: number of snapshots considered when running TPOD or TT-SVD
- nparams_res: number of snapshots considered when running MDEIM for the residual
- nparams_jac: number of snapshots considered when running MDEIM for the jacobian
- nparams_test:  number of snapshots considered when computing the error the RB
  method commits with respect to the FE procedure
"""
struct RBSolver{A<:GridapType,B} <: GridapType
  fesolver::A
  state_reduction::Reduction
  residual_reduction::Reduction
  jacobian_reduction::B
end

function RBSolver(
  fesolver::GridapType,
  state_reduction::Reduction;
  nparams_res=20,
  nparams_jac=20)

  red_style = ReductionStyle(state_reduction)
  residual_reduction = MDEIMReduction(red_style;nparams=nparams_res)
  jacobian_reduction = MDEIMReduction(red_style;nparams=nparams_jac)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

"""
    get_fe_solver(s::RBSolver) -> NonlinearSolver

Returns the underlying `NonlinearSolver` from a [`RBSolver`](@ref) `s`
"""
get_fe_solver(s::RBSolver) = s.fesolver
get_state_reduction(s::RBSolver) = s.state_reduction
get_residual_reduction(s::RBSolver) = s.residual_reduction
get_jacobian_reduction(s::RBSolver) = s.jacobian_reduction

num_state_params(s::RBSolver) = num_params(s.state_reduction)
num_res_params(s::RBSolver) = num_params(s.residual_reduction)
num_jac_params(s::RBSolver) = num_params(s.jacobian_reduction)

num_offline_params(s::RBSolver) = max(num_state_params(s),num_res_params(s),num_jac_params(s))
offline_params(s::RBSolver) = 1:num_offline_params(s)
res_params(s::RBSolver) = 1:num_res_params(s)
jac_params(s::RBSolver) = 1:num_jac_params(s)

"""
    solution_snapshots(solver::NonlinearSolver,feop::ParamOperator,r::Realization) -> SteadySnapshots
    solution_snapshots(solver::ODESolver,feop::TransientParamOperator,r::TransientRealization,u0) -> TransientSnapshots

The problem encoded in the FE operator `feop` is solved several times, and the solution
snapshots are returned along with the information related to the computational
cost of the FE method. In transient settings, an initial condition `u0` should be
provided.
"""
function solution_snapshots(
  solver::RBSolver,
  feop::ParamOperator,
  args...;
  nparams=num_offline_params(solver),
  r=realization(feop;nparams))

  solution_snapshots(solver,feop,r,args...)
end

function solution_snapshots(
  solver::RBSolver,
  feop::ParamOperator,
  r::AbstractRealization,
  args...)

  fesolver = get_fe_solver(solver)
  dof_map = get_dof_map(feop)
  values,stats = solve(fesolver,feop,r)
  snaps = Snapshots(values,dof_map,r)
  return snaps,stats
end

# not needed
function solution_snapshots(
  fesolver::NonlinearSolver,
  op::ParamOperator,
  r::Realization)

  dof_map = get_dof_map(op)
  values,stats = solve(fesolver,op,r)
  snaps = Snapshots(values,dof_map,r)
  return snaps,stats
end

"""
    residual_snapshots(solver::RBSolver,op::ParamOperator,s::AbstractSnapshots) -> Contribution
    residual_snapshots(solver::RBSolver,op::ODEParamOperator,s::AbstractSnapshots) -> Contribution

Returns a residual `Contribution` relative to the FE operator `op`. The
quantity `s` denotes the solution snapshots in which we evaluate the residual
"""
function residual_snapshots(
  solver::RBSolver,
  op::ParamOperator,
  s::AbstractSnapshots)

  sres = select_snapshots(s,res_params(solver))
  us_res = get_param_data(sres)
  r_res = get_realization(sres)
  b = residual(op,r_res,us_res)
  ib = get_dof_map_at_domains(op)
  return Snapshots(b,ib,r_res)
end

function residual_snapshots(
  solver::RBSolver,
  op::ParamOperator{LinearParamEq},
  s::AbstractSnapshots)

  sres = select_snapshots(s,res_params(solver))
  us_res = get_param_data(sres) |> similar
  fill!(us_res,zero(eltype2(us_res)))
  r_res = get_realization(sres)
  b = residual(op,r_res,us_res)
  ib = get_dof_map_at_domains(op)
  return Snapshots(b,ib,r_res)
end

function residual_snapshots(solver::RBSolver,op::ParamOperator{LinearNonlinearParamEq},args...)
  res_lin = residual_snapshots(solver,get_linear_operator(op),args...)
  res_nlin = residual_snapshots(solver,get_nonlinear_operator(op),args...)
  return (res_lin,res_nlin)
end

"""
    jacobian_snapshots(solver::RBSolver,op::ParamOperator,s::AbstractSnapshots) -> Contribution
    jacobian_snapshots(solver::RBSolver,op::ODEParamOperator,s::AbstractSnapshots) -> Tuple{Vararg{Contribution}}

Returns a Jacobian `Contribution` relative to the FE operator `op`. The
quantity `s` denotes the solution snapshots in which we evaluate the jacobian.
In transient settings, the output is a tuple whose `n`th element is the Jacobian
relative to the `n`th temporal derivative
"""
function jacobian_snapshots(
  solver::RBSolver,
  op::ParamOperator,
  s::AbstractSnapshots)

  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,jac_params(solver))
  us_jac = get_param_data(sjac)
  r_jac = get_realization(sjac)
  A = jacobian(op,r_jac,us_jac)
  iA = get_sparse_dof_map_at_domains(op)
  return Snapshots(A,iA,r_jac)
end

function jacobian_snapshots(
  solver::RBSolver,
  op::ParamOperator{LinearParamEq},
  s::AbstractSnapshots)

  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,jac_params(solver))
  us_jac = get_param_data(sjac) |> similar
  fill!(us_jac,zero(eltype2(us_jac)))
  r_jac = get_realization(sjac)
  A = jacobian(op,r_jac,us_jac)
  iA = get_sparse_dof_map_at_domains(op)
  return Snapshots(A,iA,r_jac)
end

function jacobian_snapshots(
  solver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  s::AbstractSnapshots)

  jac_lin = jacobian_snapshots(solver,get_linear_operator(op),s)
  jac_nlin = jacobian_snapshots(solver,get_nonlinear_operator(op),s)
  return (jac_lin,jac_nlin)
end

# Solve a POD-MDEIM problem

function Algebra.solve(
  solver::RBSolver,
  op::NonlinearOperator,
  r::AbstractRealization)

  trial = get_trial(op)(r)
  x̂ = zero_free_values(trial)

  nlop = parameterize(op,r)
  syscache = allocate_systemcache(nlop,x̂)

  fesolver = get_fe_solver(solver)
  t = @timed solve!(x̂,fesolver,nlop,syscache)
  stats = CostTracker(t,nruns=num_params(r),name="RB")

  return x̂,stats
end

function Algebra._solve_nr!(
  x::RBParamVector,
  A::AbstractParamMatrix,
  b::AbstractParamVector,
  dx,ns,nls,op)

  log = nls.log
  change_tols!(log)

  trial = _get_trial(op)

  res = norm(b)
  done = LinearSolvers.init!(log,res)

  while !done
    @inbounds for i in param_eachindex(x)
      xi = param_getindex(x,i)
      Ai = param_getindex(A,i)
      bi = param_getindex(b,i)
      numerical_setup!(ns,Ai)
      rmul!(bi,-1)
      solve!(dx,ns,bi)
      xi .+= dx
    end

    inv_project(trial,x)
    residual!(b,op,x)
    res  = norm(b)
    done = LinearSolvers.update!(log,res)

    if !done
      jacobian!(A,op,x)
    end
  end

  LinearSolvers.finalize!(log,res)
  return x
end

_get_trial(op) = @notimplemented

function _get_trial(op::LinNonlinParamOperator)
  μ = op.op_nonlinear.μ
  evaluate(get_trial(op.op_nonlinear.op),μ)
end

function change_tols!(log::ConvergenceLog)
  log.tols.rtol = 1e-5
  log
end
