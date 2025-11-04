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

function DrWatson.save(dir,args::Tuple)
  map(a->save(dir,a),args)
end

_get_label(name::String,label) = @abstractmethod
_get_label(name::String,label::Union{Number,Symbol}) = _get_label(name,string(label))
function _get_label(name::String,label::String)
  if label==""
    name
  else
    if name==""
      label
    else
      name * "_" * label
    end
  end
end

function _get_label(name,labels...)
  first_lab,last_labs... = labels
  _get_label(name,_get_label(first_lab,last_labs...))
end

function get_filename(dir::String,name::String,labels...;extension=".jld")
  joinpath(dir,_get_label(name,labels...)*extension)
end

function DrWatson.save(dir,s::AbstractSnapshots;label="")
  snaps_dir = get_filename(dir,"snaps",label)
  serialize(snaps_dir,s)
end

"""
    load_snapshots(dir;label="") -> AbstractSnapshots

Load the snapshots at the directory `dir`. Throws an error if the snapshots
have not been previously saved to file
"""
function load_snapshots(dir;label="")
  snaps_dir = get_filename(dir,"snaps",label)
  deserialize(snaps_dir)
end

function DrWatson.save(dir,stats::PerformanceTracker;label="")
  stats_dir = get_filename(dir,"stats",label)
  serialize(stats_dir,stats)
end

function load_stats(dir;label="")
  stats_dir = get_filename(dir,"stats",label)
  deserialize(stats_dir)
end

function DrWatson.save(dir,b::Projection;label="")
  proj_dir = get_filename(dir,"basis",label)
  serialize(proj_dir,b)
end

function load_projection(dir;label="")
  proj_dir = get_filename(dir,"basis",label)
  deserialize(proj_dir)
end

function DrWatson.save(dir,r::RBSpace;label="")
  save(dir,get_reduced_subspace(r);label)
end

"""
"""
function load_reduced_subspace(dir,f::FESpace;label="")
  basis = load_projection(dir;label)
  reduced_subspace(f,basis)
end

function DrWatson.save(dir,contrib::Contribution;label="")
  contrib_dir = get_filename(dir,"contrib",label)
  serialize(contrib_dir,get_contributions(contrib))
end

function DrWatson.save(dir,lproj::LocalProjection{<:Contribution};label="")
  _lproj = LocalProjection(map(get_contributions,lproj.projections),lproj.k)
  contrib_dir = get_filename(dir,"contrib",label)
  serialize(contrib_dir,_lproj)
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

function _setup_contribution(vals::LocalProjection,trian)
  projections = map(p -> _setup_contribution(p,trian),vals.projections)
  LocalProjection(projections,vals.k)
end

"""
"""
function load_contribution(
  dir,
  trian::Tuple{Vararg{Triangulation}};
  label="")

  contrib_dir = get_filename(dir,"contrib",label)
  vals = deserialize(contrib_dir)
  _setup_contribution(vals,trian)
end

function _save_fixed_operator_parts(dir,op;label="")
  save(dir,get_test(op);label=_get_label(label,"test"))
  save(dir,get_trial(op);label=_get_label(label,"trial"))
end

function _save_trian_operator_parts(dir,op::RBOperator;label="")
  save(dir,get_rhs(op);label=_get_label(label,"rhs"))
  save(dir,get_lhs(op);label=_get_label(label,"lhs"))
end

function DrWatson.save(dir,op::RBOperator;kwargs...)
  _save_fixed_operator_parts(dir,op;kwargs...)
  _save_trian_operator_parts(dir,op;kwargs...)
end

function _load_fixed_operator_parts(dir,feop;label="")
  test = load_reduced_subspace(dir,get_test(feop);label=_get_label(label,"test"))
  trial = load_reduced_subspace(dir,get_trial(feop);label=_get_label(label,"trial"))
  return trial,test
end

function _load_trian_operator_parts(dir,feop::ParamOperator;label="")
  trian_res = get_domains_res(feop)
  trian_jac = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res;label=_get_label(label,"rhs"))
  red_lhs = load_contribution(dir,trian_jac;label=_get_label(label,"lhs"))
  return red_lhs,red_rhs
end

"""
    load_operator(dir,feop::ParamOperator;kwargs...) -> RBOperator

Given a FE operator `feop`, load its reduced counterpart stored in the
directory `dir`. Throws an error if the reduced operator has not been previously
saved to file
"""
function load_operator(dir,feop::ParamOperator;kwargs...)
  trial,test = _load_fixed_operator_parts(dir,feop;kwargs...)
  red_lhs,red_rhs = _load_trian_operator_parts(dir,feop;kwargs...)
  return RBOperator(feop,trial,test,red_lhs,red_rhs)
end

function DrWatson.save(dir,feop::LinearNonlinearRBOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  _save_fixed_operator_parts(dir,feop_lin;label)
  _save_trian_operator_parts(dir,feop_lin;label=_get_label(label,"lin"))
  _save_trian_operator_parts(dir,feop_nlin;label=_get_label(label,"nlin"))
end

function load_operator(dir,feop::LinearNonlinearParamOperator;label="")
  feop_lin = get_linear_operator(feop)
  feop_nlin = get_nonlinear_operator(feop)
  trial,test = _fixed_operator_parts(dir,feop_lin;label)
  red_lhs_lin,red_rhs_lin = _load_trian_operator_parts(
    dir,feop_lin;label=_get_label("lin",label))
  red_lhs_nlin,red_rhs_nlin = _load_trian_operator_parts(
    dir,feop_nlin;label=_get_label("nlin",label))
  op_lin = RBOperator(feop_lin,trial,test,red_lhs_lin,red_rhs_lin)
  op_nlin = RBOperator(feop_nlin,trial,test,red_lhs_nlin,red_rhs_nlin)
  return LinearNonlinearRBOperator(op_lin,op_nlin)
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
      feop::ParamOperator,
      fesnaps::AbstractSnapshots,
      rbsnaps::AbstractSnapshots,
      festats::CostTracker,
      rbstats::CostTracker
      ) -> ROMPerformance

Arguments:
  - `solver`: solver for the reduced problem
  - `feop`: FE operator representing the PDE
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
  feop::ParamOperator,
  fesnaps::AbstractSnapshots,
  rbsnaps::AbstractSnapshots,
  festats::CostTracker,
  rbstats::CostTracker
  )

  state_red = get_state_reduction(solver)
  norm_style = NormStyle(state_red)
  error = compute_relative_error(norm_style,feop,fesnaps,rbsnaps)
  speedup = compute_speedup(festats,rbstats)
  ROMPerformance(error,speedup)
end

function eval_performance(
  solver::RBSolver,
  feop::ParamOperator,
  rbop::RBOperator,
  fesnaps::AbstractSnapshots,
  x̂::AbstractParamVector,
  festats::CostTracker,
  rbstats::CostTracker
  )

  r = get_realization(fesnaps)
  rbsnaps = to_snapshots(rbop,x̂,r)
  eval_performance(solver,feop,fesnaps,rbsnaps,festats,rbstats)
end

function DrWatson.save(dir,perf::ROMPerformance;label="")
  results_dir = get_filename(dir,"results",label)
  serialize(results_dir,perf)
end

"""
"""
function load_results(dir;label="")
  results_dir = get_filename(dir,"results",label)
  deserialize(results_dir,perf)
end

function Utils.compute_relative_error(norm_style::EnergyNorm,feop,sol,sol_approx)
  X = assemble_matrix(feop,get_norm(norm_style))
  compute_relative_error(sol,sol_approx,X)
end

function Utils.compute_relative_error(norm_style::EuclideanNorm,feop,sol,sol_approx)
  compute_relative_error(sol,sol_approx)
end

function Utils.compute_relative_error(
  sol::SteadySnapshots{T,N},
  sol_approx::SteadySnapshots{T,N},
  args...) where {T,N}

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
  sol::BlockSnapshots,
  sol_approx::BlockSnapshots,
  args...)

  @check sol.touched == sol_approx.touched
  T = eltype2(sol)
  error = Array{T,ndims(sol)}(undef,size(sol))
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
  X::MatrixOrTensor)

  @check sol.touched == sol_approx.touched
  error = zeros(size(sol))
  for i in eachindex(sol)
    if sol.touched[i]
      error[i] = compute_relative_error(sol[i],sol_approx[i],X[Block(i,i)])
    end
  end
  error
end

function _writevtk(::Type{T},Ω,dir,uh,ûh,eh) where T
  writevtk(Ω,dir,cellfields=["uh"=>uh,"ûh"=>ûh,"eh"=>eh])
end

function _writevtk(::Type{T},Ω,dir,uh,ûh,eh) where T<:Complex
  writevtk(Ω,dir,cellfields=["uh"=>abs2(uh),"ûh"=>abs2(ûh),"eh"=>abs2(eh)])
end

function plot_a_solution(dir,Ω,uh,ûh,r::Realization)
  T = eltype2(get_free_dof_values(uh))
  uh1 = param_getindex(uh,1)
  ûh1 = param_getindex(ûh,1)
  eh1 = uh1 - ûh1
  _writevtk(T,Ω,dir*".vtu",uh1,ûh1,eh1)
end

function plot_a_solution(
  dir::String,
  trial::UnEvalTrialFESpace,
  sol::Snapshots,
  sol_approx::Snapshots;
  trian=get_triangulation(trial),
  field=1)

  r = get_realization(sol)
  Ur = trial(r)
  uh = FEFunction(Ur,get_param_data(sol))
  ûh = FEFunction(Ur,get_param_data(sol_approx))
  dirfield = joinpath(dir,"var$field")
  plot_a_solution(dirfield,trian,uh,ûh,r)
end

"""
    plot_a_solution(
      dir::String,
      feop::ParamOperator,
      sol::AbstractSnapshots,
      sol_approx::AbstractSnapshots,
      args...;
      kwargs...
      ) -> Nothing

Plots a single FE solution, RB solution, and the point-wise error between the two,
by selecting the first FE snapshot in `sol` and the first reduced snapshot in `sol_approx`
"""
function plot_a_solution(
  dir::String,
  feop::ParamOperator,
  sol::Snapshots,
  sol_approx::Snapshots;
  kwargs...)

  trial = get_trial(feop)
  plot_a_solution(dir,trial,sol,sol_approx;kwargs...)
end

function plot_a_solution(
  dir::String,
  feop::ParamOperator,
  sol::BlockSnapshots,
  sol_approx::BlockSnapshots;
  kwargs...)

  @check sol.touched == sol_approx.touched
  trials = get_trial(feop)
  for i in eachindex(sol)
    if sol.touched[i]
      plot_a_solution(dir,trials[i],sol[i],sol_approx[i];field=i,kwargs...)
    end
  end
end

function plot_a_solution(
  dir::String,
  feop::ParamOperator,
  rbop::RBOperator,
  fesnaps::AbstractSnapshots,
  x̂::AbstractParamVector,
  r::AbstractRealization;
  kwargs...)

  rbsnaps = to_snapshots(rbop,x̂,r)
  plot_a_solution(dir,feop,fesnaps,rbsnaps;kwargs...)
end

# utils

function to_snapshots(rbop::RBOperator,x̂::AbstractParamVector,r::AbstractRealization)
  to_snapshots(get_trial(rbop),x̂,r)
end

function to_snapshots(rbop::AbstractLocalRBOperator,x̂::AbstractParamVector,r::AbstractRealization)
  xvec = map(enumerate(r)) do (i,μ)
    x̂μ = param_getindex(x̂,i)
    opμ = get_local(rbop,μ)
    trialμ = get_trial(opμ)
    inv_project(trialμ,x̂μ)
  end
  x = ParamArray(xvec)
  i = get_dof_map(rbop)
  Snapshots(x,i,r)
end
