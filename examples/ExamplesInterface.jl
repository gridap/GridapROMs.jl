using DrWatson
using Gridap
using Makie
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
  nparams=10,reuse_online=false,sampling=:uniform,label=online_label,kwargs...
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

function update_reduction(red::SequentialReduction,tolrank)
  SequentialReduction(update_reduction(red.reduction,tolrank))
end

function update_reduction(red::HighDimTrivialHyperReduction,tolrank)
  red
end

function update_reduction(red::HighDimMDEIMHyperReduction,tolrank)
  HighDimMDEIMHyperReduction(update_reduction(red.reduction,tolrank),red.combination)
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
  rbop::RBOperator,
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
  rbop::RBOperator,
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
  rbop::RBOperator,
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
  nparams = num_params(r)
  ptrian = num_point_dims(trian) < 3 ? trian : BoundaryTriangulation(get_background_model(trian))
  for ip in 1:nparams
    uhip = param_getindex(uh,ip)
    ûhip = param_getindex(ûh,ip)
    ehip = uhip - ûhip
    uplot = T <: Complex ? abs2(uhip) : uhip
    ûplot = T <: Complex ? abs2(ûhip) : ûhip
    eplot = T <: Complex ? abs2(ehip) : ehip
    fig = Makie.Figure()
    Makie.plot(fig[1,1],ptrian,uplot)
    Makie.plot(fig[1,2],ptrian,ûplot)
    Makie.plot(fig[1,3],ptrian,eplot)
    dir_param = joinpath(dir,"param$ip")
    create_dir(dir_param)
    Makie.save(joinpath(dir_param,"field_$(field).png"),fig)
  end
end

function _plot_solutions(dir,trian,uh,ûh,r::TransientRealisation;field=1)
  T = eltype2(get_free_dof_values(uh))
  np = num_params(r)
  nt = num_times(r)
  ptrian = num_point_dims(trian) < 3 ? trian : BoundaryTriangulation(get_background_model(trian))
  for ip in 1:np
    dir_param = joinpath(dir,"param$ip")
    create_dir(dir_param)
    ufields = [param_getindex(uh,(it-1)*np+ip) for it in 1:nt]
    ûfields = [param_getindex(ûh,(it-1)*np+ip) for it in 1:nt]
    efields = [ufields[it]-ûfields[it] for it in 1:nt]
    if T <: Complex
      ufields = abs2.(ufields)
      ûfields = abs2.(ûfields)
      efields = abs2.(efields)
    end
    it_obs = Makie.Observable(1)
    uplot = Makie.lift(i->ufields[i],it_obs)
    ûplot = Makie.lift(i->ûfields[i],it_obs)
    eplot = Makie.lift(i->efields[i],it_obs)
    fig = Makie.Figure()
    Makie.plot(fig[1,1],ptrian,uplot)
    Makie.plot(fig[1,2],ptrian,ûplot)
    Makie.plot(fig[1,3],ptrian,eplot)
    Makie.record(fig,joinpath(dir_param,"field_$(field).gif"),1:nt) do it
      it_obs[] = it
    end
  end
end

function plot_errors(dir,tolranks,perfs::AbstractVector{<:ROMPerformance})
  errs = map(get_error,perfs)
  n = length(first(errs))
  errvec = hcat(map(i -> getindex.(errs,i),1:n)...)

  file = joinpath(dir,"convergence.png")
  fig = Makie.Figure()
  ax = Makie.Axis(fig[1,1],xscale=log10,yscale=log10,xlabel="Tolerance",ylabel="Error",title="Average relative error")
  Makie.lines!(ax,tolranks,tolranks,label="Tol.",linewidth=3)
  
  for i in 1:n
    label = n==1 ? "Error" : "Error $i"
    Makie.scatter!(ax,tolranks,errvec[:,i],label=label)
  end
  
  Makie.axislegend(ax)
  Makie.save(file,fig)
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

    rbsolver = update_solver(rbsolver,tolrank)
    rbop = try_loading_reduced_operator(dir_tolrank,rbsolver,feop,fesnaps,jac,res)

    x̂,rbstats = solve(rbsolver,rbop,μon,args...)
    perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)
    println(perf)
    push!(perfs,perf)

    serialize(joinpath(dir_tolrank,"rb_solution.jld"),x̂)
  end

  results_dir = joinpath(dir,"results")
  create_dir(results_dir)

  # plot_errors(results_dir,tolranks,perfs)
  serialize(joinpath(results_dir,"performance.jld"),(tolrank => perf for (tolrank,perf) in zip(tolranks,perfs)))

  return perfs
end

struct Problem{A,B,C,D,E}
  rbsolver::A
  feop::B
  r::C
  name::D
  ic::E
end

const SteadyProblem{A,B,C,D} = Problem{A,B,C,D,Nothing}

Problem(rbsolver,feop,r,name::AbstractString) = Problem(rbsolver,feop,r,name,nothing)

function run_problem(prob::Problem;outdir::String=default_outdir())
  fesnaps,stats = _solve(prob)
  dir = joinpath(outdir,prob.name)
  create_dir(dir)
  save(dir,fesnaps)
  save(dir,stats)
  return fesnaps
end

function _solve(prob::SteadyProblem)
  solution_snapshots(prob.rbsolver,prob.feop,prob.r)
end

function _solve(prob::Problem)
  solution_snapshots(prob.rbsolver,prob.feop,prob.r,prob.ic)
end

function default_outdir()
  projdir = get(ENV,"PROJDIR",nothing)
  @assert !isnothing(projdir) 
  projdir
end

function save_gathered_snapshots(name::String;outdir::String=default_outdir())
  i = 1 
  dirs = String[]
  while isdir(joinpath(outdir,name*"_$i"))
    push!(dirs,joinpath(outdir,name*"_$i"))
    i += 1
  end
  isempty(dirs) && error("No gathered snapshot directories found for $name in $outdir")
  svec = map(dirs) do path
    load_snapshots(path)
  end
  s_offline = param_cat(svec[1:end-1])
  s_online = svec[end]
  target_dir = joinpath(outdir,name)
  (isdir(target_dir) || isfile(target_dir)) && error("Refusing to overwrite existing gathered snapshots at $target_dir")
  create_dir(target_dir)
  save(target_dir,s_offline)
  save(target_dir,s_online;label=online_label)
end