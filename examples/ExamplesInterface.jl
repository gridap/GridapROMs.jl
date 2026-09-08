using DrWatson
using Gridap
using Plots
using Serialization
using Test

using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.RBTransient

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

import Gridap.FESpaces: get_trial
import Gridap.Helpers: @abstractmethod
import Gridap.MultiField: BlockMultiFieldStyle
import GridapROMs.ParamFESpaces: UnEvalTrialFESpace
import GridapROMs.ParamSteady: get_fe_operator
import GridapROMs.RBSteady: get_state_reduction,get_residual_reduction,get_jacobian_reduction,get_error,_fe_data

function try_loading_fe_snapshots(dir,rbsolver,feop,args...;label="",kwargs...)
  try
    fesnaps = load_snapshots(dir;label)
    festats = load_stats(dir;label)
    println("Load snapshots at $dir succeeded!")
    return fesnaps,festats
  catch
    println("Load snapshots at $dir failed, must compute them")
    fesnaps,festats = solution_snapshots(rbsolver,feop,args...;kwargs...)
    save(dir,fesnaps;label)
    save(dir,festats;label)
    return fesnaps,festats
  end
end

function try_loading_online_fe_snapshots(
  dir,rbsolver,feop,args...;
  nparams=10,reuse_online=false,sampling=:uniform,label=ONLINE_LABEL,kwargs...
  )

  if reuse_online
    try
      x = load_snapshots(dir;label)
      festats = load_stats(dir;label)
      μon = get_realisation(x)
      println("Load online snapshots at $dir succeeded!")
      return x,festats,μon
    catch
      println("Load online snapshots at $dir failed, must compute them")
    end
  end
  μon = realisation(feop;nparams,sampling)
  x,festats = solution_snapshots(rbsolver,feop,μon,args...;kwargs...)
  save(dir,x;label)
  save(dir,festats;label)
  return x,festats,μon
end

function try_loading_fe_jac_res(dir,rbsolver,feop,fesnaps)
  try
    jac = load_jacobians(dir,feop)
    res = load_residuals(dir,feop)
    println("Load res/jac at $dir succeeded!")
    return jac,res
  catch
    println("Load res/jac at $dir failed, must compute them")
    jac = jacobian_snapshots(rbsolver,feop,fesnaps)
    res = residual_snapshots(rbsolver,feop,fesnaps)
    save_jacobians(dir,feop,jac)
    save_residuals(dir,feop,res)
    return jac,res
  end
end

function try_loading_reduced_operator(dir_tolrank,rbsolver,feop,fesnaps,jac,res)
  try
    rbop = load_operator(dir_tolrank,feop)
    println("Load reduced operator at $dir_tolrank succeeded!")
    return rbop
  catch
    println("Load reduced operator at $dir_tolrank failed, must run offline phase")
    rbop = reduced_operator(rbsolver,feop,fesnaps,jac,res)
    save(dir_tolrank,rbop)
    return rbop
  end
end

update_redstyle(rs::SearchSVDRank,tolrank) = SearchSVDRank(tolrank)
update_redstyle(rs::LRApproxRank,tolrank) = LRApproxRank(tolrank)
update_redstyle(rs::TTSVDRanks,tolrank) = TTSVDRanks(map(s->update_redstyle(s,tolrank),rs.style))

function update_reduction(red::Reduction,tolrank)
  @abstractmethod
end

function update_reduction(red::TrivialHyperReduction,tolrank)
  red
end

function update_reduction(red::PODReduction,tolrank)
  PODReduction(update_redstyle(red.red_style,tolrank),red.norm_style,red.nparams)
end

function update_reduction(red::TTSVDReduction,tolrank)
  TTSVDReduction(update_redstyle(red.red_style,tolrank),red.norm_style,red.nparams)
end

function update_reduction(red::LocalReduction,tolrank)
  LocalReduction(update_reduction(red.reduction,tolrank),red.ncentroids)
end

function update_reduction(red::SupremizerReduction,tolrank)
  SupremizerReduction(update_reduction(red.reduction,tolrank),red.coupling,red.supr_tol)
end

function update_reduction(red::DEIMHyperReduction,tolrank)
  DEIMHyperReduction(update_reduction(red.reduction,tolrank))
end

function update_reduction(red::SOPTHyperReduction,tolrank)
  SOPTHyperReduction(update_reduction(red.reduction,tolrank))
end

function update_reduction(red::RBFHyperReduction,tolrank)
  RBFHyperReduction(update_reduction(red.reduction,tolrank),red.strategy)
end

function update_reduction(red::SteadyReduction,tolrank)
  SteadyReduction(update_reduction(red.reduction,tolrank))
end

function update_reduction(red::KroneckerReduction,tolrank)
  KroneckerReduction(
    map(r->update_reduction(r,tolrank),red.reductions)
  )
end

function update_reduction(red::SequentialReduction,tolrank)
  SequentialReduction(update_reduction(red.reduction,tolrank))
end

function update_reduction(red::HighDimTrivialHyperReduction,tolrank)
  red
end

function update_reduction(red::HighDimDEIMHyperReduction,tolrank)
  HighDimDEIMHyperReduction(update_reduction(red.reduction,tolrank),red.combination)
end

function update_reduction(red::HighDimSOPTHyperReduction,tolrank)
  HighDimSOPTHyperReduction(update_reduction(red.reduction,tolrank),red.combination)
end

function update_reduction(red::HighDimRBFHyperReduction,tolrank)
  HighDimRBFHyperReduction(update_reduction(red.reduction,tolrank),red.combination,red.strategy)
end

function update_reduction(red::NTuple{N,Reduction},tolrank) where N
  map(r->update_reduction(r,tolrank),red)
end

function update_solver(rbsolver::RBSolver,rank::Int)
  fesolver = get_fe_solver(rbsolver)
  state_reduction = update_reduction(get_state_reduction(rbsolver),rank)
  residual_reduction = update_reduction(get_residual_reduction(rbsolver),rank+5)
  jacobian_reduction = update_reduction(get_jacobian_reduction(rbsolver),rank+5)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function update_solver(rbsolver::RBSolver,tol)
  fesolver = get_fe_solver(rbsolver)
  state_reduction = update_reduction(get_state_reduction(rbsolver),tol)
  residual_reduction = update_reduction(get_residual_reduction(rbsolver),tol*1e-2)
  jacobian_reduction = update_reduction(get_jacobian_reduction(rbsolver),tol*1e-2)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function plot_solutions(
  dir::String,
  trial::UnEvalTrialFESpace,
  sol::Snapshots,
  sol_approx::Snapshots;
  trian=get_triangulation(trial),
  kwargs...
  )

  r = get_realisation(sol)
  Ur = trial(r)
  uh = FEFunction(Ur,get_param_data(sol))
  ûh = FEFunction(Ur,get_param_data(sol_approx))
  _plot_solutions(dir,trian,uh,ûh,r;kwargs...)
end

function plot_solutions(
  dir::String,
  rbop::ReducedOperator,
  sol::Snapshots,
  sol_approx::Snapshots;
  kwargs...
  )

  feop = get_fe_operator(rbop)
  trial = get_trial(feop)
  plot_solutions(dir,trial,sol,sol_approx;kwargs...)
end

function plot_solutions(
  dir::String,
  rbop::ReducedOperator,
  sol::BlockSnapshots,
  sol_approx::BlockSnapshots;
  kwargs...
  )
  
  feop = get_fe_operator(rbop)
  trials = get_trial(feop)
  for i in eachindex(sol)
    if sol.touched[i]
      plot_solutions(dir,trials[i],sol[i],sol_approx[i];field=i,kwargs...)
    end
  end
end

function plot_solutions(
  dir::String,
  rbop::ReducedOperator,
  fesnaps::AbstractSnapshots,
  x̂::AbstractParamVector;
  kwargs...
  )

  i = get_dof_map(fesnaps)
  r = get_realisation(fesnaps)
  rbsnaps = Snapshots(_fe_data(x̂),i,r)
  plot_solutions(dir,rbop,fesnaps,rbsnaps;kwargs...)
end

function _plot_solutions(dir,trian,uh,ûh,r::Realisation;field=1)
  T = eltype2(get_free_dof_values(uh))
  fields = Pair{String}[]
  for ip in 1:num_params(r)
    uhip  = param_getindex(uh,ip)
    ûhip  = param_getindex(ûh,ip)
    ehip  = uhip - ûhip
    if T <: Complex
      push!(fields,"uh_param_$ip"  => abs2(uhip))
      push!(fields,"ûh_param_$ip" => abs2(ûhip))
      push!(fields,"eh_param_$ip"  => abs2(ehip))
    else
      push!(fields,"uh_param_$ip"  => uhip)
      push!(fields,"ûh_param_$ip" => ûhip)
      push!(fields,"eh_param_$ip"  => ehip)
    end
  end
  writevtk(trian,dir*"_field_$field",cellfields=fields)
end

function _plot_solutions(dir,trian,uh,ûh,r::TransientRealisation;field=1)
  T = eltype2(get_free_dof_values(uh))
  fields = Pair{String}[]
  np = num_params(r)
  for it in 1:num_times(r), ip in 1:num_params(r)
    uhipt  = param_getindex(uh,(it-1)*np+ip)
    ûhipt  = param_getindex(ûh,(it-1)*np+ip)
    ehipt  = uhipt - ûhipt
    if T <: Complex
      push!(fields,"uh_param_$(ip)_$(it)"  => abs2(uhipt))
      push!(fields,"ûh_param_$(ip)_$(it)" => abs2(ûhipt))
      push!(fields,"eh_param_$(ip)_$(it)"  => abs2(ehipt))
    else
      push!(fields,"uh_param_$(ip)_$(it)"  => uhipt)
      push!(fields,"ûh_param_$(ip)_$(it)" => ûhipt)
      push!(fields,"eh_param_$(ip)_$(it)"  => ehipt)
    end
  end
  writevtk(trian,dir*"_field_$field",cellfields=fields)
end

function plot_errors(dir,tolranks,perfs::AbstractVector{<:ROMPerformance})
  errs = map(get_error,perfs)
  n = length(first(errs))
  errvec = hcat(map(i -> getindex.(errs,i),1:n)...)
  labvec = n==1 ? "Error" : hcat(["Error $i" for i in 1:n])

  file = joinpath(dir,"convergence.png")
  p = plot(tolranks,tolranks,lw=3,label="Tol.")
  scatter!(tolranks,errvec,lw=3,label=labvec)
  plot!(xscale=:log10,yscale=:log10)
  xlabel!("Tolerance")
  ylabel!("Error")
  title!("Average relative error")
  savefig(p,file)
end

# function run_offline(
#   dir::String,rbsolver::RBSolver,feop::ParamOperator,
#   tolranks=[1e-1,1e-2,1e-3,1e-4,1e-5],args...;kwargs...
#   )
  
#   # fesnaps, = try_loading_fe_snapshots(dir,rbsolver,feop,args...)
#   # jac = load_jacobians(dir,feop)
#   res = load_residuals(dir,feop)

#   for tolrank in tolranks
#     println("Running test $dir with tolrank = $tolrank")

#     dir_tolrank = joinpath(dir,string(tolrank))
#     create_dir(dir_tolrank)

#     rbsolver = update_solver(rbsolver,tolrank)
#     # red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
#     # save(dir_tolrank,red_trial;label=TRIAL_LABEL)
#     # save(dir_tolrank,red_test;label=TEST_LABEL)
#     red_trial,red_test = RBSteady._load_fixed_operator_parts(dir_tolrank,feop)
#     # jac_red = get_jacobian_reduction(rbsolver)
#     # red_jac = reduced_jacobian(jac_red,red_trial,red_test,jac)
#     res_red = get_residual_reduction(rbsolver)
#     red_res = reduced_residual(res_red,red_test,res)
#     save(dir_tolrank,red_res;label=RHS_LABEL)
#   end
# end

function run_test(
  dir::String,rbsolver::RBSolver,feop::ParamOperator,tolranks=[1e-1,1e-2,1e-3,1e-4,1e-5],
  args...;nparams=10,reuse_online=false,sampling=:uniform,kwargs...)

  fesnaps, = try_loading_fe_snapshots(dir,rbsolver,feop,args...)
  jac,res = try_loading_fe_jac_res(dir,rbsolver,feop,fesnaps)
  x,festats,μon = try_loading_online_fe_snapshots(
    dir,rbsolver,feop,args...;nparams,reuse_online,sampling)

  perfs = ROMPerformance[]

  for tolrank in tolranks
    println("Running test $dir with tolrank = $tolrank")

    dir_tolrank = joinpath(dir,string(tolrank))
    create_dir(dir_tolrank)

    rbsolver = update_solver(rbsolver,tolrank)
    rbop = try_loading_reduced_operator(dir_tolrank,rbsolver,feop,fesnaps,jac,res)
    # rbop = load_operator(dir_tolrank,feop)

    x̂,rbstats = solve(rbsolver,rbop,μon,args...)
    perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)
    println(perf)
    push!(perfs,perf)
  end

  results_dir = joinpath(dir,"results")
  create_dir(results_dir)

  # plot_errors(results_dir,tolranks,perfs)
  serialize(joinpath(results_dir,"performance.jld"),(tolrank => perf for (tolrank,perf) in zip(tolranks,perfs)))

  return perfs
end