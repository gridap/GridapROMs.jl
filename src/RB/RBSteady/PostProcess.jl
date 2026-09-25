const SNAPSHOTS_LABEL = "snaps"
const RESIDUALS_LABEL = "res"
const JACOBIANS_LABEL = "jac"
const RHS_LABEL = "rhs"
const LHS_LABEL = "lhs"
const TEST_LABEL = "test"
const TRIAL_LABEL = "trial"
const STATISTICS_LABEL = "stats"
const RESULTS_LABEL = "results"
const PROJECTION_LABEL = "projection"
const CONTRIBUTIONS_LABEL = "contributions"
const LINEAR_LABEL = "lin"
const NONLINEAR_LABEL = "nlin"
const OFFLINE_LABEL = "offline"
const ONLINE_LABEL = "online"

_get_label(name::String,label) = _get_label(name,string(label))

function _get_label(name::String,label::String)
  label == "" && return name
  name == "" && return label
  return name * "_" * label
end

function _get_label(name,labels...)
  first_lab,last_labs... = labels
  _get_label(_get_label(name,first_lab...),last_labs...)
end

function get_filename(dir::String,name::String,labels...;extension=".jld")
  joinpath(dir,_get_label(name,labels...)*extension)
end

function save(dir,s::AbstractSnapshots;label="")
  snaps_dir = get_filename(dir,SNAPSHOTS_LABEL,label)
  serialize(snaps_dir,s)
end

function load(dir,base=SNAPSHOTS_LABEL;label="")
  stats_dir = get_filename(dir,base,label)
  deserialize(stats_dir)
end

"""
    load_snapshots(dir;label="") -> AbstractSnapshots

Load the snapshots at the directory `dir`. Throws an error if the snapshots
have not been previously saved to file
"""
function load_snapshots(dir;label="")
  load(dir,SNAPSHOTS_LABEL;label)
end

function save(dir,stats::PerformanceTracker;label="")
  stats_dir = get_filename(dir,STATISTICS_LABEL,label)
  serialize(stats_dir,stats)
end

function load_stats(dir;label="")
  load(dir,STATISTICS_LABEL;label)
end

function save(dir,b::Projection;label="")
  proj_dir = get_filename(dir,PROJECTION_LABEL,label)
  serialize(proj_dir,b)
end

function load_projection(dir;label="")
  load(dir,PROJECTION_LABEL;label)
end

function save(dir,r::RBSpace;label="")
  save(dir,get_reduced_subspace(r);label)
end

"""
"""
function load_subspace(dir,f::FESpace;label="")
  basis = load_projection(dir;label)
  reduced_subspace(f,basis)
end

function save(dir,contrib::Contribution;label="")
  contrib_dir = get_filename(dir,CONTRIBUTIONS_LABEL,label)
  serialize(contrib_dir,get_contributions(contrib))
end

function _setup_contribution(vals::Tuple{Vararg{Any}},trian)
  @check length(trian)==length(vals)
  Contribution(vals,trian)
end

function _setup_contribution(vals::Tuple{Vararg{HRProjection}},trian)
  @check length(trian)==length(vals)
  redtrian = ()
  for i in eachindex(trian)
    redtrian = (redtrian...,reduced_triangulation(trian[i],vals[i]))
  end
  Contribution(vals,redtrian)
end

"""
"""
function load_contribution(
  dir,
  trian::Tuple;
  label=""
  )

  contrib_dir = get_filename(dir,CONTRIBUTIONS_LABEL,label)
  vals = deserialize(contrib_dir)
  _setup_contribution(vals,trian)
end

function save(dir,op::ROMOperator;label="")
  save(dir,get_test(op);label=_get_label(label,TEST_LABEL))
  save(dir,get_trial(op);label=_get_label(label,TRIAL_LABEL))
  save(dir,get_rhs(op);label=_get_label(label,RHS_LABEL))
  save(dir,get_lhs(op);label=_get_label(label,LHS_LABEL))
end

"""
    load_operator(dir,feop::ParamOperator;kwargs...) -> ROMOperator

Given a FE operator `feop`, load its reduced counterpart stored in the
directory `dir`. Throws an error if the reduced operator has not been previously
saved to file
"""
function load_operator(dir,feop::ParamOperator;label="")
  test = load_subspace(dir,get_test(feop);label=_get_label(label,TEST_LABEL))
  trial = load_subspace(dir,get_trial(feop);label=_get_label(label,TRIAL_LABEL))
  trian_res = get_domains_res(feop)
  trian_jac = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res;label=_get_label(label,RHS_LABEL))
  red_lhs = load_contribution(dir,trian_jac;label=_get_label(label,LHS_LABEL))
  return ROMOperator(feop,trial,test,red_lhs,red_rhs)
end

function save(dir,feop::LinearNonlinearROMOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  # test and trial are the same for both the linear and nonlinear operators
  save(dir,get_test(feop_lin);label=_get_label(label,TEST_LABEL))
  save(dir,get_trial(feop_lin);label=_get_label(label,TRIAL_LABEL))
  # weakform-related quantities are different for the linear and nonlinear operators
  save(dir,get_rhs(feop_lin);label=_get_label(label,LINEAR_LABEL,RHS_LABEL))
  save(dir,get_lhs(feop_lin);label=_get_label(label,LINEAR_LABEL,LHS_LABEL))
  save(dir,get_rhs(feop_nlin);label=_get_label(label,NONLINEAR_LABEL,RHS_LABEL))
  save(dir,get_lhs(feop_nlin);label=_get_label(label,NONLINEAR_LABEL,LHS_LABEL))
end

function load_operator(dir,feop::LinearNonlinearParamOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  # test and trial are the same for both the linear and nonlinear operators
  test = load_subspace(dir,get_test(feop_lin);label=_get_label(label,TEST_LABEL))
  trial = load_subspace(dir,get_trial(feop_lin);label=_get_label(label,TRIAL_LABEL))
  # weakform-related quantities are different for the linear and nonlinear operators
  red_rhs_lin = load_contribution(dir,get_domains_res(feop_lin);label=_get_label(label,LINEAR_LABEL,RHS_LABEL))
  red_lhs_lin = load_contribution(dir,get_domains_jac(feop_lin);label=_get_label(label,LINEAR_LABEL,LHS_LABEL))
  red_rhs_nlin = load_contribution(dir,get_domains_res(feop_nlin);label=_get_label(label,NONLINEAR_LABEL,RHS_LABEL))
  red_lhs_nlin = load_contribution(dir,get_domains_jac(feop_nlin);label=_get_label(label,NONLINEAR_LABEL,LHS_LABEL))
  op_lin = ROMOperator(feop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = ROMOperator(feop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearROMOperator(op_lin,op_nlin)
end

"""
    compute_error!(
      solver::RBSolver,
      op::ParamOperator,
      x::AbstractSnapshots
      x̂::AbstractSnapshots,
      ) -> RBPerformanceTracker

Updates `solver.tracker.error` in place with the (relative) error between the
full-order snapshots `fesnaps` and the reduced approximation `x̂`
"""
function compute_error!(solver::RBSolver,op::ParamOperator,x::AbstractSnapshots,x̂::AbstractSnapshots)
  feop = get_fe_operator(op)
  tracker = solver.tracker
  tracker.error = compute_relative_error(solver,feop,x,x̂)
  return tracker
end

"""
    rom_performance(
      solver::RBSolver,
      op::ROMOperator,
      x::AbstractSnapshots,
      x̂
      ) -> RBPerformanceTracker

Updates `solver.tracker.error` from the (relative) error between the full-order
snapshots `fesnaps` and the reduced approximation `x̂`, and returns `solver.tracker`
"""
function rom_performance(solver::RBSolver,op::ParamOperator,x::AbstractSnapshots,x̂)
  _to_snaps(x̂) = @abstractmethod 
  _to_snaps(x̂::AbstractSnapshots) = x̂
  _to_snaps(x̂::RBParamVector) = Snapshots(_fe_data(x̂),get_dof_map(x),get_realisation(x)) 
  compute_error!(solver,op,x,_to_snaps(x̂))
  return solver.tracker
end

function save(dir,perf::RBPerformanceTracker;label="")
  results_dir = get_filename(dir,RESULTS_LABEL,label)
  serialize(results_dir,perf)
end

"""
"""
function load_results(dir;label="")
  results_dir = get_filename(dir,RESULTS_LABEL,label)
  deserialize(results_dir)
end

function Utils.compute_relative_error(solver::RBSolver,feop,sol,sol_approx)
  state_red = get_state_reduction(solver)
  norm_style = NormStyle(state_red)
  compute_relative_error(norm_style,feop,sol,sol_approx)
end

function Utils.compute_relative_error(norm_style::EuclideanNorm,feop,sol,sol_approx)
  compute_relative_error(sol,sol_approx)
end

function Utils.compute_relative_error(norm_style::AssembleOperator,feop,sol,sol_approx)
  X = assemble_operator(norm_style,feop)
  compute_relative_error(sol,sol_approx,X)
end

function Utils.compute_relative_error(
  sol::BlockSnapshots{<:Any,N},
  sol_approx::BlockSnapshots{<:Any,N},
  args...
  ) where N

  @check size(sol) == size(sol_approx)
  T = eltype2(sol)
  error = Array{T,N}(undef,size(sol))
  for i in eachindex(sol)
    error[i] = compute_relative_error(sol[i],sol_approx[i])
  end
  error
end

function Utils.compute_relative_error(
  sol::BlockSnapshots,
  sol_approx::BlockSnapshots,
  X::MatrixOrTensor
  )

  @check size(sol) == size(sol_approx)
  error = zeros(size(sol))
  for i in eachindex(sol)
    error[i] = compute_relative_error(sol[i],sol_approx[i],X[Block(i,i)])
  end
  error
end

include("Diagnostics.jl")