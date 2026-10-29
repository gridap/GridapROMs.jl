"""
    SNAPSHOTS_LABEL

Label of the files storing snapshots
"""
const SNAPSHOTS_LABEL = "snaps"
"""
    RESIDUALS_LABEL

Label of the files storing residual snapshots
"""
const RESIDUALS_LABEL = "res"
"""
    JACOBIANS_LABEL

Label of the files storing Jacobian snapshots
"""
const JACOBIANS_LABEL = "jac"
"""
    RHS_LABEL

Label of the files storing hyper-reduced residuals
"""
const RHS_LABEL = "rhs"
"""
    LHS_LABEL

Label of the files storing hyper-reduced Jacobians
"""
const LHS_LABEL = "lhs"
"""
    TEST_LABEL

Label of the files storing reduced test spaces
"""
const TEST_LABEL = "test"
"""
    TRIAL_LABEL

Label of the files storing reduced trial spaces
"""
const TRIAL_LABEL = "trial"
"""
    STATISTICS_LABEL

Label of the files storing performance statistics
"""
const STATISTICS_LABEL = "stats"
"""
    RESULTS_LABEL

Label of the files storing results of [`rom_performance`](@ref)
"""
const RESULTS_LABEL = "results"
"""
    PROJECTION_LABEL

Label of the files storing projections
"""
const PROJECTION_LABEL = "projection"
"""
    CONTRIBUTIONS_LABEL

Label of the files storing contributions
"""
const CONTRIBUTIONS_LABEL = "contributions"
"""
    LINEAR_LABEL

Label of the files storing linear parts of a `Bundle`
"""
const LINEAR_LABEL = "lin"
"""
    NONLINEAR_LABEL

Label of the files storing nonlinear parts of a `Bundle`
"""
const NONLINEAR_LABEL = "nlin"
"""
    OFFLINE_LABEL

Label of the files storing offline quantities
"""
const OFFLINE_LABEL = "offline"
"""
    ONLINE_LABEL

Label of the files storing online quantities
"""
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

"""
    load_stats(dir;label="") -> PerformanceTracker

Loads the performance statistics stored in `dir`
"""
function load_stats(dir;label="")
  load(dir,STATISTICS_LABEL;label)
end

function save(dir,b::Projection;label="")
  proj_dir = get_filename(dir,PROJECTION_LABEL,label)
  serialize(proj_dir,b)
end

"""
    load_projection(dir;label="") -> Projection

Loads the projection stored in `dir`
"""
function load_projection(dir;label="")
  load(dir,PROJECTION_LABEL;label)
end

function save(dir,r::RBSpace;label="")
  save(dir,get_reduced_subspace(r);label)
end

"""
    load_subspace(dir,f::FESpace;label="") -> RBSpace

Loads the reduced subspace of `f` stored in `dir`
"""
function load_subspace(dir,f::FESpace;label="")
  basis = load_projection(dir;label)
  reduced_subspace(f,basis)
end

function save(dir,contrib::Contribution;label="")
  contrib_dir = get_filename(dir,CONTRIBUTIONS_LABEL,label)
  serialize(contrib_dir,get_contributions(contrib))
end

function save(dir,contrib::Bundle;label="")
  save(dir,contrib.linear;label=_get_label(label,LINEAR_LABEL))
  save(dir,contrib.nonlinear;label=_get_label(label,NONLINEAR_LABEL))
end

function _setup_contribution(vals::Tuple{Vararg{Any}},trian)
  @check length(trian)==length(vals)
  Contribution(vals,trian)
end

"""
    load_contribution(dir,trians;label="") -> Contribution

Loads the hyper-reduced contributions stored in `dir`, and associates them with the
triangulations `trians` (a tuple, or a [`Bundle`](@ref) of tuples)
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

function load_contribution(dir,trians::Bundle;label="")
  lin = load_contribution(dir,trians.linear;label=_get_label(label,LINEAR_LABEL))
  nlin = load_contribution(dir,trians.nonlinear;label=_get_label(label,NONLINEAR_LABEL))
  Bundle(lin,nlin)
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

"""
    rom_performance(
      solver::RBSolver,
      op::ROMOperator,
      x::AbstractSnapshots,
      x̂::AbstractSnapshots
      ) -> RBPerformanceTracker

Updates `solver.tracker.error` from the (relative) error between the full-order
snapshots `fesnaps` and the reduced approximation `x̂`, and returns `solver.tracker`
"""
function rom_performance(solver::RBSolver,op::ParamOperator,x::AbstractSnapshots,x̂::AbstractSnapshots)
  trial = get_trial(op)
  tracker = solver.tracker
  tracker.error = compute_relative_error(trial,x,x̂)
  return tracker
end

function save(dir,perf::RBPerformanceTracker;label="")
  results_dir = get_filename(dir,RESULTS_LABEL,label)
  serialize(results_dir,perf)
end

"""
    load_results(dir;label="") -> RBPerformanceTracker

Loads the results of [`rom_performance`](@ref) stored in `dir`
"""
function load_results(dir;label="")
  results_dir = get_filename(dir,RESULTS_LABEL,label)
  deserialize(results_dir)
end

function Utils.compute_relative_error(trial::RBSpace,sol,sol_approx)
  X = get_norm_matrix(trial)
  compute_relative_error(sol,sol_approx,X)
end

function Utils.compute_relative_error(sol::BlockSnapshots,sol_approx::BlockSnapshots)
  @check size(sol) == size(sol_approx)
  map(i -> compute_relative_error(sol[i],sol_approx[i]),eachindex(sol))
end

function Utils.compute_relative_error(sol::BlockSnapshots,sol_approx::BlockSnapshots,X)
  @check size(sol) == size(sol_approx)
  map(i -> compute_relative_error(sol[i],sol_approx[i],X[i]),eachindex(sol))
end

include("Diagnostics.jl")