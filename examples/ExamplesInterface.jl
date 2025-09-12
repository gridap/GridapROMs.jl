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

import Gridap.CellData: get_domains
import Gridap.Helpers: @abstractmethod
import Gridap.MultiField: BlockMultiFieldStyle
import GridapROMs.ParamAlgebra: get_linear_operator,get_nonlinear_operator
import GridapROMs.ParamDataStructures: AbstractSnapshots,ReshapedSnapshots,TransientSnapshotsWithIC,GenericTransientRealization,get_realization
import GridapROMs.ParamSteady: ParamOperator
import GridapROMs.RBSteady: get_state_reduction,get_residual_reduction,get_jacobian_reduction,load_stats,get_filename,_get_label
import GridapROMs.Utils: Contribution,TupOfArrayContribution,change_domains

function change_dof_map(s::GenericSnapshots,dof_map)
  pdata = get_param_data(s)
  r = get_realization(s)
  Snapshots(pdata,dof_map,r)
end

function change_dof_map(s::ReshapedSnapshots,dof_map)
  pdata = get_param_data(s)
  r = get_realization(s)
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

for (f,l) in zip((:save_residuals,:save_jacobians),(:res,:jac))
  @eval begin
    function $f(dir,resjac,feop::ParamOperator;label="")
      save(dir,resjac;label=_get_label(label,string($l)))
    end

    function $f(dir,resjac,feop::LinearNonlinearParamOperator;label="")
      @assert length(resjac) == 2
      resjac_lin,resjac_nlin = resjac
      res_lin = $f(dir,resjac_lin,get_linear_operator(feop);label=_get_label(label,"lin"))
      res_nlin = $f(dir,resjac_nlin,get_nonlinear_operator(feop);label=_get_label(label,"nlin"))
      return (res_lin,res_nlin)
    end
  end
end

for (f,g,l) in zip((:load_residuals,:load_jacobians),(:get_domains_res,:get_domains_jac),(:res,:jac))
  @eval begin
    function $f(dir,feop::ParamOperator;label="")
      load_contribution(dir,$g(feop);label=_get_label(label,string($l)))
    end

    function $f(dir,feop::LinearNonlinearParamOperator;label="")
      res_lin = $f(dir,get_linear_operator(feop);label=_get_label(label,"lin"))
      res_nlin = $f(dir,get_nonlinear_operator(feop);label=_get_label(label,"nlin"))
      return (res_lin,res_nlin)
    end
  end
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
  dir,rbsolver,feop,args...;nparams=10,reuse_online=false,sampling=:uniform,label="",kwargs...)

  label = "online"
  if reuse_online
    x,festats = try_loading_fe_snapshots(dir,rbsolver,feop,args...;nparams,label)
    μon = get_realization(x)
  else
    μon = realization(feop;nparams,sampling=:uniform)
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
    save_jacobians(dir,jac,feop)
    save_residuals(dir,res,feop)
    return jac,res
  end
end

function try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps,jac,res)
  try
    rbop = load_operator(dir_tol,feop)
    println("Load reduced operator at $dir_tol succeeded!")
    return rbop
  catch
    println("Load reduced operator at $dir_tol failed, must run offline phase")
    rbop = reduced_operator(rbsolver,feop,fesnaps,jac,res)
    save(dir_tol,rbop)
    return rbop
  end
end

get_error(perf::ROMPerformance) = perf.error

function plot_errors(dir,tols,perfs::Vector{ROMPerformance})
  errs = map(get_error,perfs)
  n = length(first(errs))
  errvec = map(i -> getindex.(errs,i),1:n)
  labvec = n==1 ? "Error" : ["Error $i" for i in 1:n]

  file = joinpath(dir,"convergence.png")
  p = plot(tols,tols,lw=3,label="Tol.")
  scatter!(tols,errvec,lw=3,label=labvec)
  plot!(xscale=:log10,yscale=:log10)
  xlabel!("Tolerance")
  ylabel!("Error")
  title!("Average relative error")

  savefig(p,file)
end

update_redstyle(rs::SearchSVDRank,tol) = SearchSVDRank(tol)
update_redstyle(rs::LRApproxRank,tol) = LRApproxRank(tol)
update_redstyle(rs::TTSVDRanks,tol) = TTSVDRanks(map(s->update_redstyle(s,tol),rs.style))

function update_reduction(red::Reduction,tol)
  @abstractmethod
end

function update_reduction(red::AffineReduction,tol)
  AffineReduction(update_redstyle(red.red_style,tol),red.norm_style)
end

function update_reduction(red::PODReduction,tol)
  PODReduction(update_redstyle(red.red_style,tol),red.norm_style,red.nparams)
end

function update_reduction(red::TTSVDReduction,tol)
  TTSVDReduction(update_redstyle(red.red_style,tol),red.norm_style,red.nparams)
end

function update_reduction(red::SupremizerReduction,tol)
  SupremizerReduction(update_reduction(red.reduction,tol),red.supr_op,red.supr_tol)
end

function update_reduction(red::MDEIMHyperReduction,tol)
  MDEIMHyperReduction(update_reduction(red.reduction,tol))
end

function update_reduction(red::KroneckerReduction,tol)
  KroneckerReduction(
    update_reduction(red.reduction_space,tol),
    update_reduction(red.reduction_time,tol)
    )
end

function update_reduction(red::SequentialReduction,tol)
  SequentialReduction(update_reduction(red.reduction,tol))
end

function update_reduction(red::HighDimMDEIMHyperReduction,tol)
  HighDimMDEIMHyperReduction(update_reduction(red.reduction,tol),red.combine)
end

function update_reduction(red::NTuple{N,HighDimMDEIMHyperReduction},tol) where N
  map(r->update_reduction(r,tol),red)
end

function update_solver(rbsolver::RBSolver,tol)
  fesolver = get_fe_solver(rbsolver)
  state_reduction = update_reduction(get_state_reduction(rbsolver),tol)
  residual_reduction = update_reduction(get_residual_reduction(rbsolver),tol)
  jacobian_reduction = update_reduction(get_jacobian_reduction(rbsolver),tol)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function run_test(
  dir::String,rbsolver::RBSolver,feop::ParamOperator,tols=[1e-1,1e-2,1e-3,1e-4,1e-5],
  args...;nparams=10,reuse_online=false,sampling=:uniform,kwargs...)

  fesnaps, = try_loading_fe_snapshots(dir,rbsolver,feop,args...)
  jac,res = try_loading_fe_jac_res(dir,rbsolver,feop,fesnaps)
  x,festats,μon = try_loading_online_fe_snapshots(
    dir,rbsolver,feop,args...;nparams,reuse_online,sampling)

  perfs = ROMPerformance[]

  for tol in tols
    println("Running test $dir with tol = $tol")

    dir_tol = joinpath(dir,string(tol))
    create_dir(dir_tol)

    plot_dir_tol = joinpath(dir_tol,"plot")
    create_dir(plot_dir_tol)

    rbsolver = update_solver(rbsolver,tol)
    rbop = try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps,jac,res)

    x̂,rbstats = solve(rbsolver,rbop,μon,args...)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
    println(perf)
    push!(perfs,perf)

    plot_a_solution(plot_dir_tol,feop,rbop,x,x̂,μon;kwargs...)
  end

  results_dir = joinpath(dir,"results")
  create_dir(results_dir)

  plot_errors(results_dir,tols,perfs)
  serialize(joinpath(results_dir,"performance.jld"),(tol => perf for (tol,perf) in zip(tols,perfs)))

  return perfs
end

function run_cost(
  dir::String,rbsolver::RBSolver,feop::ParamOperator,
  tols=[1e-1,1e-2,1e-3,1e-4,1e-5],args...)

  fesnaps, = try_loading_fe_snapshots(dir,rbsolver,feop,args...)
  for tol in tols
    reduced_spaces(rbsolver,feop,fesnaps)
  end
end
