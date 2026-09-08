"""
    create_dir(dir::String) -> Nothing

Recursive creation of a directory `dir`; does not do anything if `dir` exists
"""
function create_dir(dir::String)
  if !isdir(dir)
    parent_dir, = splitdir(dir)
    create_dir(parent_dir)
    mkdir(dir)
  end
  return
end

const SNAPSHOTS_LABEL = "snaps"
const RESIDUALS_LABEL = "res"
const JACOBIANS_LABEL = "jac"
const RHS_LABEL = "rhs"
const LHS_LABEL = "lhs"
const TEST_LABEL = "test"
const TRIAL_LABEL = "trial"
const STATISTICS_LABEL = "stats"
const RESULTS_LABEL = "results"
const PROJECTION_LABEL = "basis"
const CONTRIBUTIONS_LABEL = "contrib"
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
function load_reduced_subspace(dir,f::FESpace;label="")
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

function _save_fixed_operator_parts(dir,op;label="")
  save(dir,get_test(op);label=_get_label(label,TEST_LABEL))
  save(dir,get_trial(op);label=_get_label(label,TRIAL_LABEL))
end

function _save_trian_operator_parts(dir,op::ReducedOperator;label="")
  save(dir,get_rhs(op);label=_get_label(label,RHS_LABEL))
  save(dir,get_lhs(op);label=_get_label(label,LHS_LABEL))
end

function save(dir,op::ReducedOperator;kwargs...)
  _save_fixed_operator_parts(dir,op;kwargs...)
  _save_trian_operator_parts(dir,op;kwargs...)
end

function _load_fixed_operator_parts(dir,feop;label="")
  test = load_reduced_subspace(dir,get_test(feop);label=_get_label(label,TEST_LABEL))
  trial = load_reduced_subspace(dir,get_trial(feop);label=_get_label(label,TRIAL_LABEL))
  return trial,test
end

function _load_trian_operator_parts(dir,feop::ParamOperator;label="")
  trian_res = get_domains_res(feop)
  trian_jac = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res;label=_get_label(label,RHS_LABEL))
  red_lhs = load_contribution(dir,trian_jac;label=_get_label(label,LHS_LABEL))
  return red_lhs,red_rhs
end

"""
    load_operator(dir,feop::ParamOperator;kwargs...) -> ReducedOperator

Given a FE operator `feop`, load its reduced counterpart stored in the
directory `dir`. Throws an error if the reduced operator has not been previously
saved to file
"""
function load_operator(dir,feop::ParamOperator;kwargs...)
  trial,test = _load_fixed_operator_parts(dir,feop;kwargs...)
  red_lhs,red_rhs = _load_trian_operator_parts(dir,feop;kwargs...)
  return ReducedOperator(feop,trial,test,red_lhs,red_rhs)
end

function save(dir,feop::LinearNonlinearReducedOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  _save_fixed_operator_parts(dir,feop_lin;label)
  _save_trian_operator_parts(dir,feop_lin;label=_get_label(label,LINEAR_LABEL))
  _save_trian_operator_parts(dir,feop_nlin;label=_get_label(label,NONLINEAR_LABEL))
end

function load_operator(dir,feop::LinearNonlinearParamOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  trial,test = _load_fixed_operator_parts(dir,feop_lin;label)
  red_lhs_lin,red_rhs_lin = _load_trian_operator_parts(
    dir,feop_lin;label=_get_label(LINEAR_LABEL,label))
  red_lhs_nlin,red_rhs_nlin = _load_trian_operator_parts(
    dir,feop_nlin;label=_get_label(NONLINEAR_LABEL,label))
  op_lin = ReducedOperator(feop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = ReducedOperator(feop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearReducedOperator(op_lin,op_nlin)
end

"""
    struct ROMPerformance
      error
      speedup
    end

Allows to compute errors and computational speedups to compare the properties of
the algorithm with the FE performance.
"""
struct ROMPerformance
  error
  speedup
end

get_error(perf::ROMPerformance) = perf.error
get_speedup(perf::ROMPerformance) = perf.speedup

function Base.show(io::IO,k::MIME"text/plain",perf::ROMPerformance)
  println(io," -------------------- ROMPerformance -------------------------")
  println(io," > error: $(perf.error)")
  println(io," > speedup in time: $(perf.speedup.speedup_time)")
  println(io," > speedup in memory: $(perf.speedup.speedup_memory)")
  println(io," -------------------------------------------------------------")
end

function Base.show(io::IO,perf::ROMPerformance)
  show(io,MIME"text/plain"(),perf)
end

function mean(perfs::AbstractVector{<:ROMPerformance})
  mean_err = mean(map(get_error,perfs))
  mean_su = mean(map(get_speedup,perfs))
  ROMPerformance(mean_err,mean_su)
end

"""
    eval_performance(
      solver::RBSolver,
      rbop::ReducedOperator,
      fesnaps::AbstractSnapshots,
      rbsnaps::AbstractSnapshots,
      festats::CostTracker,
      rbstats::CostTracker
      ) -> ROMPerformance

Arguments:
  - `solver`: solver for the reduced problem
  - `rbop`: reduced operator representing the PDE
  - `fesnaps`: online snapshots of the FE solution
  - `rbsnaps`: reduced approximation of `fesnaps`
  - `festats`: time and memory consumption needed to compute `fesnaps`
  - `rbstats`: time and memory consumption needed to compute `rbsnaps`

Returns the performance of the reduced algorithm, in terms of the (relative) error
between `rbsnaps` and `fesnaps`, and the computational speedup between `rbstats`
and `festats`
"""
function eval_performance(
  solver::RBSolver,
  rbop::ReducedOperator,
  fesnaps::AbstractSnapshots,
  rbsnaps::AbstractSnapshots,
  festats::CostTracker,
  rbstats::CostTracker
  )

  feop = get_fe_operator(rbop)
  error = compute_relative_error(solver,feop,fesnaps,rbsnaps)
  speedup = compute_speedup(festats,rbstats)
  ROMPerformance(error,speedup)
end

function eval_performance(
  solver::RBSolver,
  rbop::ReducedOperator,
  fesnaps::AbstractSnapshots,
  x̂::RBParamVector,
  festats::CostTracker,
  rbstats::CostTracker
  )

  r = get_realisation(fesnaps)
  i = get_dof_map(fesnaps)
  rbsnaps = Snapshots(_fe_data(x̂),i,r)
  eval_performance(solver,rbop,fesnaps,rbsnaps,festats,rbstats)
end

function save(dir,perf::ROMPerformance;label="")
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
  sol::SteadySnapshots{T,N},
  sol_approx::SteadySnapshots{T,N},
  args...
  ) where {T,N}

  @check size(sol) == size(sol_approx) 
  errors = zeros(num_params(sol))
  @inbounds for ip = 1:num_params(sol)
    solip = param_getindex(sol,ip)
    solip_approx = param_getindex(sol_approx,ip)
    err_norm = induced_norm(solip-solip_approx,args...)
    sol_norm = induced_norm(solip,args...)
    errors[ip] = err_norm / sol_norm
  end
  return mean(errors)
end

function Utils.compute_relative_error(
  sol::BlockSnapshots{N},
  sol_approx::BlockSnapshots{N},
  args...
  ) where N

  @check sol.touched == sol_approx.touched
  T = eltype2(sol)
  error = Array{T,N}(undef,size(sol))
  for i in eachindex(sol)
    if sol.touched[i]
      error[i] = compute_relative_error(sol[i],sol_approx[i])
    end
  end
  error
end

function Utils.compute_relative_error(
  sol::BlockSnapshots,
  sol_approx::BlockSnapshots,
  X::MatrixOrTensor
  )

  @check sol.touched == sol_approx.touched
  error = zeros(size(sol))
  for i in eachindex(sol)
    if sol.touched[i]
      error[i] = compute_relative_error(sol[i],sol_approx[i],X[Block(i,i)])
    end
  end
  error
end

include("Diagnostics.jl")