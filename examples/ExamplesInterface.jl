module ExamplesInterface

using DrWatson
using Gridap
using Plots
using Serialization
using Test

using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.ParamDataStructures

import Gridap.CellData: get_domains
import Gridap.Helpers: @abstractmethod
import Gridap.MultiField: BlockMultiFieldStyle
import GridapROMs.ParamAlgebra: get_linear_operator,get_nonlinear_operator
import GridapROMs.ParamDataStructures: AbstractSnapshots,ReshapedSnapshots,TransientSnapshotsWithIC,GenericTransientRealization,get_realization
import GridapROMs.ParamSteady: ParamOperator,LinearNonlinearParamEq
import GridapROMs.ParamODEs: ODEParamOperator,LinearNonlinearParamODE
import GridapROMs.RBSteady: reduced_operator,get_state_reduction,get_residual_reduction,get_jacobian_reduction,load_stats
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

function try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps)
  try
    rbop = load_operator(dir_tol,feop)
    println("Load reduced operator at $dir_tol succeeded!")
    return rbop
  catch
    println("Load reduced operator at $dir_tol failed, must run offline phase")
    dir = joinpath(splitpath(dir_tol)[1:end-1])
    local res,jac
    try
      res = load_residuals(dir,feop)
      jac = load_jacobians(dir,feop)
    catch
      res = residual_snapshots(rbsolver,feop,fesnaps)
      jac = jacobian_snapshots(rbsolver,feop,fesnaps)
      save(dir,res,feop;label="res")
      save(dir,jac,feop;label="jac")
    end
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

function update_reduction(red::HighOrderMDEIMHyperReduction,tol)
  HighOrderMDEIMHyperReduction(update_reduction(red.reduction,tol),red.combine)
end

function update_reduction(red::NTuple{N,HighOrderMDEIMHyperReduction},tol) where N
  map(r->update_reduction(r,tol),red)
end

function update_solver(rbsolver::RBSolver,tol)
  fesolver = get_fe_solver(rbsolver)
  state_reduction = update_reduction(get_state_reduction(rbsolver),tol)
  residual_reduction = update_reduction(get_residual_reduction(rbsolver),tol)
  jacobian_reduction = update_reduction(get_jacobian_reduction(rbsolver),tol)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function reduced_operator(rbsolver::RBSolver,feop::ParamOperator,red_trial,red_test,jac,res)
  jac_red = get_jacobian_reduction(rbsolver)
  red_lhs = reduced_jacobian(jac_red,red_trial,red_test,jac)
  res_red = get_residual_reduction(rbsolver)
  red_rhs = reduced_residual(res_red,red_test,res)
  RBOperator(feop,red_trial,red_test,red_lhs,red_rhs)
end

function reduced_operator(rbsolver::RBSolver,odeop::ODEParamOperator,red_trial,red_test,jac,res)
  jac_red = get_jacobian_reduction(rbsolver)
  red_lhs = reduced_jacobian(jac_red,red_trial,red_test,jac)
  res_red = get_residual_reduction(rbsolver)
  red_rhs = reduced_residual(res_red,red_test,res)
  RBOperator(odeop,red_trial,red_test,red_lhs,red_rhs)
end

for T in (:LinearNonlinearParamEq,:LinearNonlinearParamODE)
  @eval begin
    function reduced_operator(
      rbsolver::RBSolver,
      feop::ParamOperator{$T},
      red_trial,
      red_test,
      (jac_lin,jac_nlin),
      (res_lin,res_nlin)
      )

      rbop_lin = reduced_operator(rbsolver,get_linear_operator(feop),red_trial,red_test,jac_lin,res_lin)
      rbop_nlin = reduced_operator(rbsolver,get_nonlinear_operator(feop),red_trial,red_test,jac_nlin,res_nlin)
      LinearNonlinearRBOperator(rbop_lin,rbop_nlin)
    end
  end
end

function reduced_operator(rbsolver::RBSolver,feop::ParamOperator,sol::AbstractSnapshots,args...)
  red_trial,red_test = reduced_spaces(rbsolver,feop,sol)
  reduced_operator(rbsolver,feop,red_trial,red_test,args...)
end

function run_test(
  dir::String,rbsolver::RBSolver,feop::ParamOperator,tols=[1e-1,1e-2,1e-3,1e-4,1e-5],
  args...;nparams=10,reuse_online=false,sampling=:uniform,kwargs...)

  fesnaps, = try_loading_fe_snapshots(dir,rbsolver,feop,args...)
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
    rbop = try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps)

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

export DrWatson
export Gridap
export Plots
export Serialization
export Test

export GridapROMs

export BlockMultiFieldStyle

export run_test

end
