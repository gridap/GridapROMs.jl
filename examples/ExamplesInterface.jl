using DrWatson
using Gridap
using Plots
using Serialization
using Test

using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.ParamDataStructures

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

import Gridap.Helpers: @abstractmethod
import Gridap.MultiField: BlockMultiFieldStyle
import GridapROMs.ParamAlgebra: get_linear_operator,get_nonlinear_operator
import GridapROMs.ParamDataStructures: ReshapedSnapshots,TransientSnapshotsWithIC,get_realisation
import GridapROMs.RBSteady: TrivialHyperReduction,get_state_reduction,get_residual_reduction,get_jacobian_reduction,get_error,_get_label
import GridapROMs.Utils: Contribution,TupOfArrayContribution

function change_dof_map(s::GenericSnapshots,dof_map)
  pdata = get_param_data(s)
  r = get_realisation(s)
  Snapshots(pdata,dof_map,r)
end

function change_dof_map(s::ReshapedSnapshots,dof_map)
  pdata = get_param_data(s)
  r = get_realisation(s)
  Snapshots(pdata,dof_map,r)
end

function change_dof_map(s::TransientSnapshotsWithIC,dof_map)
  TransientSnapshotsWithIC(s.initial_data,change_dof_map(s.snaps,dof_map))
end

function change_dof_map(resjac::Contribution,dof_map::Contribution)
  resjac′ = ()
  for i in eachindex(resjac)
    resjac′ = (resjac′...,change_dof_map(resjac[i],dof_map[i]))
  end
  return Contribution(resjac′,resjac.trians)
end

function change_dof_map(jac::TupOfArrayContribution,dof_map::TupOfArrayContribution)
  jac′ = ()
  for i in eachindex(jac)
    jac′ = (jac′...,change_dof_map(jac[i],dof_map[i]))
  end
  return jac′
end

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
  dir,rbsolver,feop,args...;nparams=10,reuse_online=false,sampling=:uniform,label="online",kwargs...)

  if reuse_online
    x,festats = try_loading_fe_snapshots(dir,rbsolver,feop,args...;nparams,label)
    μon = get_realisation(x)
  else
    μon = realisation(feop;nparams,sampling=:uniform)
    x,festats = solution_snapshots(rbsolver,feop,μon,args...;kwargs...)
    save(dir,x;label)
    save(dir,festats;label)
  end
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
  SupremizerReduction(update_reduction(red.reduction,tolrank),red.supr_op,red.supr_tol)
end

function update_reduction(red::MDEIMHyperReduction,tolrank)
  MDEIMHyperReduction(update_reduction(red.reduction,tolrank))
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

function update_reduction(red::TrivialHyperReduction,tolrank)
  red
end

function update_reduction(red::SequentialReduction,tolrank)
  SequentialReduction(update_reduction(red.reduction,tolrank))
end

function update_reduction(red::HighDimMDEIMHyperReduction,tolrank)
  HighDimMDEIMHyperReduction(update_reduction(red.reduction,tolrank),red.combination)
end

function update_reduction(red::NTuple{N,HighDimMDEIMHyperReduction},tolrank) where N
  map(r->update_reduction(r,tolrank),red)
end

function update_reduction(red::HighDimSOPTHyperReduction,tolrank)
  HighDimSOPTHyperReduction(update_reduction(red.reduction,tolrank),red.combination)
end

function update_reduction(red::NTuple{N,HighDimSOPTHyperReduction},tolrank) where N
  map(r->update_reduction(r,tolrank),red)
end

function update_reduction(red::HighDimRBFHyperReduction,tolrank)
  HighDimRBFHyperReduction(update_reduction(red.reduction,tolrank),red.combination,red.strategy)
end

function update_reduction(red::NTuple{N,HighDimRBFHyperReduction},tolrank) where N
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

    plot_dir_tolrank = joinpath(dir_tolrank,"plot")
    create_dir(plot_dir_tolrank)

    rbsolver = update_solver(rbsolver,tolrank)
    rbop = try_loading_reduced_operator(dir_tolrank,rbsolver,feop,fesnaps,jac,res)

    x̂,rbstats = solve(rbsolver,rbop,μon,args...)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
    println(perf)
    push!(perfs,perf)

    plot_a_solution(plot_dir_tolrank,feop,rbop,x,x̂,μon;kwargs...)
  end

  results_dir = joinpath(dir,"results")
  create_dir(results_dir)

  plot_errors(results_dir,tolranks,perfs)
  serialize(joinpath(results_dir,"performance.jld"),(tolrank => perf for (tolrank,perf) in zip(tolranks,perfs)))

  return perfs
end
