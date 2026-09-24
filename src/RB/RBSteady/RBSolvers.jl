abstract type SolverContext end
struct GlobalContext <: SolverContext end
struct LocalContext <: SolverContext end

abstract type ROMSolver <: GridapType end

"""
    struct RBSolver{A,B,C,D,E,F} <: ROMSolver
      fesolver::A
      context::B
      state_reduction::C
      residual_reduction::D
      jacobian_reduction::E
      tracker::F
    end

Wrapper around a FE solver (e.g. `NonlinearSolver` or `ODESolver` in [`Gridap`](@ref)) with
additional information on the reduced basis (RB) method employed to solve a given
problem dependent on a set of parameters. A RB method is a projection-based
reduced order model where

1. a suitable subspace of a FESpace is sought, of dimension n << Nₕ
2. a matrix-based discrete empirical interpolation method (e.g. DEIM) is performed
  to approximate the manifold of the parametric residuals and jacobians
3. the EIM approximations are compressed with (Petrov-)Galerkin projections
  onto the subspace
4. for every desired choice of parameters, numerical integration is performed, and
  the resulting n × n system of equations is cheaply solved

Fields:

- `fesolver`: solver used to compute the full-order (FE) solutions/snapshots
- `context`: either `GlobalContext` (a [`GlobalRBSolver`](@ref)) or `LocalContext`
  (a [`LocalRBSolver`](@ref), used for cluster-local ROMs)
- `state_reduction`: `Reduction` strategy compressing state snapshots into the
  reduced subspace (e.g. `tol`/`nparams` for TPOD or TT-SVD live here)
- `residual_reduction`, `jacobian_reduction`: `HyperReduction` strategies
  hyper-reducing the residual/Jacobian, respectively; the number of snapshots
  used to train each is set through the `nparams_res`/`nparams_jac` keywords
  of the constructor below
- `tracker`: a [`RBPerformanceTracker`](@ref) accumulating timing/memory/error
  information across the offline (subspace, jacobian and residual
  hyper-reduction) and online (reduced solve) phases

    RBSolver(fesolver,reduction::Reduction;nparams_res=20,nparams_jac=20,verbose=default_verbose(),kwargs...)
    RBSolver(fesolver,style::ReductionStyle,args...;nparams=100,kwargs...)

The most convenient way to build a `RBSolver`: `reduction` (or `style`, from which
a `Reduction` is built using `nparams` snapshots) governs the state subspace,
while `nparams_res`/`nparams_jac` set the number of snapshots used for hyper-reduction.
`verbose` controls whether `solver.tracker` prints cost information as it is
populated during the offline/online phases (through [`set_subspace_tracker!`](@ref),
[`set_jacobian_tracker!`](@ref), [`set_residual_tracker!`](@ref) and
[`update_rom_tracker!`](@ref)); the (relative) error of an online solve against a
full-order reference is instead computed on demand with [`rom_performance`](@ref).
"""
struct RBSolver{A,B,C,D,E,F} <: ROMSolver
  fesolver::A
  context::B
  state_reduction::C
  residual_reduction::D
  jacobian_reduction::E
  tracker::F
end

function RBSolver(
  fesolver,
  context::SolverContext,
  state_reduction,
  residual_reduction,
  jacobian_reduction;
  tracker=RBPerformanceTracker()
  )

  RBSolver(fesolver,context,state_reduction,residual_reduction,jacobian_reduction,tracker)
end

const GlobalRBSolver{A,C,D,E} = RBSolver{A,GlobalContext,C,D,E}

function GlobalRBSolver(fesolver,args...;kwargs...)
  RBSolver(fesolver,GlobalContext(),args...;kwargs...)
end

function RBSolver(
  fesolver,
  state_reduction,
  residual_reduction,
  jacobian_reduction;
  kwargs...
  )

  GlobalRBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction;kwargs...)
end

const LocalRBSolver{A,C,D,E} = RBSolver{A,LocalContext,C,D,E}

function LocalRBSolver(fesolver,args...;kwargs...)
  RBSolver(fesolver,LocalContext(),args...;kwargs...)
end

function RBSolver(
  fesolver,
  state_reduction::Union{LocalReduction,SupremizerReduction{A,B,<:LocalReduction} where {A,B}},
  residual_reduction,
  jacobian_reduction;
  kwargs...
  )

  LocalRBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction;kwargs...)
end

function RBSolver(
  fesolver::GridapType,
  reduction::Reduction;
  nparams_res=20,
  nparams_jac=20,
  verbose=default_verbose(),
  kwargs...
  )

  residual_reduction = HyperReduction(reduction;nparams=nparams_res,kwargs...)
  jacobian_reduction = HyperReduction(reduction;nparams=nparams_jac,kwargs...)
  RBSolver(fesolver,reduction,residual_reduction,jacobian_reduction;tracker=RBPerformanceTracker(;verbose))
end

function RBSolver(
  fesolver::GridapType,
  style::ReductionStyle,
  args...;
  nparams=100,
  kwargs...
  )

  reduction = Reduction(style;nparams)
  RBSolver(fesolver,reduction,args...;kwargs...)
end

"""
    get_fe_solver(s::RBSolver) -> NonlinearSolver

Returns the underlying `NonlinearSolver` from a [`RBSolver`](@ref) `s`
"""
get_fe_solver(s::RBSolver) = s.fesolver
get_reduced_solver(s::RBSolver) = LUSolver()
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

function change_context(s::GlobalRBSolver)
  LocalRBSolver(get_fe_solver(s),get_state_reduction(s),get_residual_reduction(s),get_jacobian_reduction(s))
end

function change_context(s::LocalRBSolver)
  GlobalRBSolver(get_fe_solver(s),get_state_reduction(s),get_residual_reduction(s),get_jacobian_reduction(s))
end

for f in (:set_fom_tracker!,:set_rom_tracker!,:set_subspace_tracker!,:set_jacobian_tracker!,:set_residual_tracker!)
  @eval begin
    function Utils.$f(s::RBSolver,args...;kwargs...)
      Utils.$f(s.tracker,args...;kwargs...)
    end
  end
end

update_rom_tracker!(s::RBSolver,args...;kwargs...) = set_rom_tracker!(s,args...;kwargs...)

"""
    solution_snapshots(solver::NonlinearSolver,feop::ParamOperator,r::Realisation) -> SteadySnapshots
    solution_snapshots(solver::ODESolver,feop::TransientParamOperator,r::TransientRealisation,u0) -> TransientSnapshots

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
  r=realisation(feop;nparams)
  )

  solution_snapshots(solver,feop,r,args...)
end

function solution_snapshots(
  solver::RBSolver,
  feop::ParamOperator,
  r::AbstractRealisation,
  args...
  )

  fesolver = get_fe_solver(solver)
  solution_snapshots(fesolver,feop,r,args...)
end

function solution_snapshots(
  fesolver::NonlinearSolver,
  op::ParamOperator,
  r::Realisation
  )

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
  s::AbstractSnapshots
  )

  sres = select_snapshots(s,res_params(solver))
  us_res = get_param_data(sres)
  r_res = get_realisation(sres)
  b = Algebra.residual(op,r_res,us_res)
  ib = get_dof_map(op,b)
  return Snapshots(b,ib,r_res)
end

function residual_snapshots(
  solver::RBSolver,
  op::ParamOperator{LinearParamEq},
  s::AbstractSnapshots
  )

  sres = select_snapshots(s,res_params(solver))
  us_res = get_param_data(sres) |> similar
  fill!(us_res,zero(eltype2(us_res)))
  r_res = get_realisation(sres)
  b = Algebra.residual(op,r_res,us_res)
  ib = get_dof_map(op,b)
  return Snapshots(b,ib,r_res)
end

function residual_snapshots(
  solver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  args...
  )

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
  s::AbstractSnapshots
  )

  sjac = select_snapshots(s,jac_params(solver))
  us_jac = get_param_data(sjac)
  r_jac = get_realisation(sjac)
  A = Algebra.jacobian(op,r_jac,us_jac)
  iA = get_sparse_dof_map(op,A)
  return Snapshots(A,iA,r_jac)
end

function jacobian_snapshots(
  solver::RBSolver,
  op::ParamOperator{LinearParamEq},
  s::AbstractSnapshots
  )

  sjac = select_snapshots(s,jac_params(solver))
  us_jac = get_param_data(sjac) |> similar
  fill!(us_jac,zero(eltype2(us_jac)))
  r_jac = get_realisation(sjac)
  A = Algebra.jacobian(op,r_jac,us_jac)
  iA = get_sparse_dof_map(op,A)
  return Snapshots(A,iA,r_jac)
end

function jacobian_snapshots(
  solver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  s::AbstractSnapshots
  )

  jac_lin = jacobian_snapshots(solver,get_linear_operator(op),s)
  jac_nlin = jacobian_snapshots(solver,get_nonlinear_operator(op),s)
  return (jac_lin,jac_nlin)
end

# solvers 

function Algebra.solve(solver::GlobalRBSolver,op::NonlinearOperator,r::Realisation)
  trial = get_trial(op)(r)
  x̂ = zero_free_values(trial)

  nlop = parameterise(op,r)
  syscache = allocate_systemcache(nlop,x̂)

  s = get_reduced_solver(solver)
  t = @timed solve!(x̂,s,nlop,syscache)
  update_rom_tracker!(solver,t,nruns=num_params(r))

  inv_project!(x̂,trial)

  return x̂
end

function Algebra.solve(solver::LocalRBSolver,op::NonlinearOperator,r::AbstractRealisation,args...)
  gsolver = change_context(solver)
  t = @timed x̂vec = map(get_params(r)) do _μ
    opμ = get_local(op,_μ)
    μ = to_realisation(r,_μ)
    solve(gsolver,opμ,μ,args...)
  end
  x̂ = to_param_array(r,x̂vec)
  update_rom_tracker!(solver,t,nruns=num_params(r))
  return x̂
end

to_realisation(r::Realisation,μ) = Realisation([μ])
to_param_array(r::Realisation,x) = param_cat(x)

# nonlinear solver

function Algebra._solve_nr!(
  x::RBParamVector,
  A::AbstractParamMatrix,
  b::AbstractParamVector,
  dx,ns,nls,op
  )

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

    inv_project!(x,trial)
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

_get_param(op::NonlinearParamOperator) = @notimplemented
_get_param(op::GenericParamNonlinearOperator) = op.μ

_get_trial(op) = @notimplemented

function _get_trial(op::LinNonlinParamOperator)
  μ = _get_param(op.op_nonlinear)
  evaluate(get_trial(op.op_nonlinear.op),μ)
end

function change_tols!(log::ConvergenceLog)
  log.tols.rtol = 1e-6
  log
end