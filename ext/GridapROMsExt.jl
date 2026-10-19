module GridapROMsExt

using DrWatson
using Gridap
using Plots
using Serialization

using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.RBTransient

import Gridap.FESpaces: get_trial
import Gridap.Helpers: @abstractmethod
import GridapROMs.ParamDataStructures: get_sampling_style
import GridapROMs.ParamFESpaces: UnEvalTrialFESpace
import GridapROMs.ParamSteady: get_fe_operator,get_param_space
import GridapROMs.RBSteady: get_state_reduction,get_residual_reduction,get_jacobian_reduction,num_offline_params

function GridapROMs.try_loading_fe_snapshots(
  dir,rbsolver,feop,args...;
  label="",
  sampling=get_sampling_style(get_param_space(feop)),
  nparams=num_offline_params(rbsolver)
  )

  try
    fesnaps = load_snapshots(dir;label)
    println("Load snapshots at $dir succeeded!")
    return fesnaps
  catch
    println("Load snapshots at $dir failed, must compute them")
    r = realisation(feop;sampling,nparams)
    fesnaps = solution_snapshots(rbsolver,feop,r,args...)
    save(dir,fesnaps;label)
    return fesnaps
  end
end

function GridapROMs.try_loading_reduced_operator(dir_tolrank,rbsolver,feop,fesnaps)
  try
    rbop = load_operator(dir_tolrank,feop)
    println("Load reduced operator at $dir_tolrank succeeded!")
    return rbop
  catch
    println("Load reduced operator at $dir_tolrank failed, must run offline phase")
    rbop = reduced_operator(rbsolver,feop,fesnaps)
    save(dir_tolrank,rbop)
    return rbop
  end
end

update_redstyle(rs::SearchSVDRank,tolrank) = SearchSVDRank(tolrank)
update_redstyle(rs::TTSVDRanks,tolrank) = TTSVDRanks(map(s->update_redstyle(s,tolrank),rs.style))

function GridapROMs.update_reduction(red::Reduction,tolrank)
  @abstractmethod
end

function GridapROMs.update_reduction(red::TrivialHyperReduction,tolrank)
  red
end

function GridapROMs.update_reduction(red::PODReduction,tolrank)
  PODReduction(update_redstyle(red.red_style,tolrank),red.norm_style,red.nparams)
end

function GridapROMs.update_reduction(red::TTSVDReduction,tolrank)
  TTSVDReduction(update_redstyle(red.red_style,tolrank),red.norm_style,red.nparams)
end

function GridapROMs.update_reduction(red::LocalReduction,tolrank)
  LocalReduction(update_reduction(red.reduction,tolrank),red.ncentroids)
end

function GridapROMs.update_reduction(red::SupremizerReduction,tolrank)
  SupremizerReduction(update_reduction(red.reduction,tolrank),red.coupling,red.supr_tol)
end

function GridapROMs.update_reduction(red::DEIMHyperReduction,tolrank)
  DEIMHyperReduction(update_reduction(red.reduction,tolrank))
end

function GridapROMs.update_reduction(red::SOPTHyperReduction,tolrank)
  SOPTHyperReduction(update_reduction(red.reduction,tolrank))
end

function GridapROMs.update_reduction(red::RBFHyperReduction,tolrank)
  RBFHyperReduction(update_reduction(red.reduction,tolrank),red.strategy)
end

function GridapROMs.update_reduction(red::SteadyReduction,tolrank)
  SteadyReduction(update_reduction(red.reduction,tolrank))
end

function GridapROMs.update_reduction(red::KroneckerReduction,tolrank)
  KroneckerReduction(
    map(r->update_reduction(r,tolrank),red.reductions)
  )
end

function GridapROMs.update_reduction(red::SequentialReduction,tolrank)
  SequentialReduction(update_reduction(red.reduction,tolrank))
end

function GridapROMs.update_reduction(red::TransientTrivialHyperReduction,tolrank)
  red
end

function GridapROMs.update_reduction(red::TransientDEIMHyperReduction,tolrank)
  TransientDEIMHyperReduction(update_reduction(red.reduction,tolrank),red.combination)
end

function GridapROMs.update_reduction(red::TransientSOPTHyperReduction,tolrank)
  TransientSOPTHyperReduction(update_reduction(red.reduction,tolrank),red.combination)
end

function GridapROMs.update_reduction(red::TransientRBFHyperReduction,tolrank)
  TransientRBFHyperReduction(update_reduction(red.reduction,tolrank),red.combination,red.strategy)
end

function GridapROMs.update_reduction(red::NTuple{N,Reduction},tolrank) where N
  map(r->update_reduction(r,tolrank),red)
end

function GridapROMs.update_solver(rbsolver::RBSolver,rank::Int)
  fesolver = get_fe_solver(rbsolver)
  state_reduction = update_reduction(get_state_reduction(rbsolver),rank)
  residual_reduction = update_reduction(get_residual_reduction(rbsolver),rank+5)
  jacobian_reduction = update_reduction(get_jacobian_reduction(rbsolver),rank+5)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function GridapROMs.update_solver(rbsolver::RBSolver,tol)
  fesolver = get_fe_solver(rbsolver)
  state_reduction = update_reduction(get_state_reduction(rbsolver),tol)
  residual_reduction = update_reduction(get_residual_reduction(rbsolver),tol*1e-2)
  jacobian_reduction = update_reduction(get_jacobian_reduction(rbsolver),tol*1e-2)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function GridapROMs.run_test(
  dir::String,rbsolver::RBSolver,feop::ParamOperator,tolranks=[1e-1,1e-2,1e-3,1e-4,1e-5],
  args...;nparams=10,reuse_online=false,sampling=:uniform,kwargs...)

  fesnaps = try_loading_fe_snapshots(dir,rbsolver,feop,args...)
  x = try_loading_fe_snapshots(
    dir,rbsolver,feop,args...;nparams,sampling,label=ONLINE_LABEL)

  perfs = map(tolranks) do tolrank
    println("Running test $dir with tolrank = $tolrank")

    dir_tolrank = joinpath(dir,string(tolrank))
    mkpath(dir_tolrank)

    rbsolver = update_solver(rbsolver,tolrank)
    rbop = try_loading_reduced_operator(dir_tolrank,rbsolver,feop,fesnaps)

    μ = get_realisation(x)
    x̂ = solution_snapshots(rbsolver,rbop,μ,args...)
    rom_performance(rbsolver,rbop,x,x̂)
  end

  results_dir = joinpath(dir,"results")
  mkpath(results_dir)

  plot_errors(results_dir,tolranks,perfs)
  serialize(joinpath(results_dir,"performance.jld"),(tolrank => perf for (tolrank,perf) in zip(tolranks,perfs)))

  return perfs
end

function GridapROMs.plot_solutions(
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

function GridapROMs.plot_solutions(
  dir::String,
  rbop::ROMOperator,
  sol::Snapshots,
  sol_approx::Snapshots;
  kwargs...
  )

  feop = get_fe_operator(rbop)
  trial = get_trial(feop)
  plot_solutions(dir,trial,sol,sol_approx;kwargs...)
end

function GridapROMs.plot_solutions(
  dir::String,
  rbop::ROMOperator,
  sol::BlockSnapshots,
  sol_approx::BlockSnapshots;
  kwargs...
  )

  feop = get_fe_operator(rbop)
  trials = get_trial(feop)
  for i in eachindex(sol)
    plot_solutions(dir,trials[i],sol[i],sol_approx[i];field=i,kwargs...)
  end
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

function GridapROMs.plot_errors(dir,tolranks,perfs::AbstractVector{<:RBPerformanceTracker})
  errs = map(p->p.error,perfs)
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

end
