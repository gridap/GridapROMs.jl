module MeshTests

using DrWatson
using Serialization
using Gridap
using GridapROMs

using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.RBTransient

import Gridap.Helpers: @abstractmethod
import GridapROMs.Utils: Contribution,TupOfArrayContribution
import GridapROMs.ParamDataStructures: ReshapedSnapshots,TransientSnapshotsWithIC,GenericTransientRealization,_get_params
import GridapROMs.ParamSteady: get_domains_res,get_domains_jac,get_dof_map_at_domains,get_sparse_dof_map_at_domains
import GridapROMs.RBSteady: load_stats

include("ExamplesInterface.jl")

get_id(path::String) = split(path,"_")[end][1:end-4]

function get_ids(dir::String)
  dirs = readdir(dir,join=true)
  ids = Int[]
  for path in dirs
    id = get_id(path)
    try
      push!(ids,parse(Int,id))
    catch
      @assert id == "online" "$(id)"
    end
  end
  sort(ids)
end

get_parent_dir(dir::String) = join(split(dir,"/")[1:end-1],"/")

function get_offline_dirs_sol(dir::String)
  ids = get_ids(dir)
  offline_dirs = String[]
  for id in ids
    push!(offline_dirs,joinpath(dir,"snaps_"*string(id)*".jld"))
  end
  return offline_dirs
end

function get_online_dir_sol(dir::String)
  dirs = readdir(dir,join=true)
  online_dir = joinpath(dir,"snaps_online.jld")
  @assert online_dir ∈ dirs "Online snaps have not been generated"
  return online_dir
end

function get_offline_dirs_resjac(dir::String;label="1")
  parent_dir = get_parent_dir(dir)
  sol_dir = joinpath(parent_dir,"sol")
  ids = get_ids(sol_dir)
  offline_dirs = String[]
  for id in ids
    push!(offline_dirs,joinpath(dir,"snaps_"*string(id)*"_"*label*".jld"))
  end
  return offline_dirs
end

function get_offline_dirs(dir::String;kwargs...)
  parts = split(dir,"/")
  if last(parts) == "sol"
    get_offline_dirs_sol(dir)
  else
    get_offline_dirs_resjac(dir;kwargs...)
  end
end

function merge_realizations(rvec::Vector{<:Realization})
  Realization(vcat(map(_get_params,rvec)...))
end

function merge_realizations(rvec::Vector{<:GenericTransientRealization})
  r1 = first(rvec)
  GenericTransientRealization(
    merge_realizations(map(get_params,rvec)),
    get_times(r1),
    r1.t0
    )
end

function merge_param_data(pvec::Vector{<:ConsecutiveParamVector})
  ConsecutiveParamVector(hcat(map(get_all_data,pvec)...))
end

function merge_param_data(pvec::Vector{<:ConsecutiveParamSparseMatrixCSC})
  nzval = hcat(map(get_all_data,pvec)...)
  p1 = first(pvec)
  ConsecutiveParamSparseMatrixCSC(p1.m,p1.n,p1.colptr,p1.rowval,nzval)
end

function merge_snapshots(svec::Vector{<:Snapshots})
  @abstractmethod
end

function merge_snapshots(svec::Vector{<:GenericSnapshots})
  dvec = map(get_all_data,svec)
  rvec = map(get_realization,svec)
  d = hcat(dvec...)
  r = merge_realizations(rvec)
  i = get_dof_map(first(svec))
  GenericSnapshots(d,i,r)
end

function merge_snapshots(svec::Vector{<:ReshapedSnapshots})
  dvec = map(get_all_data,svec)
  pvec = map(get_param_data,svec)
  rvec = map(get_realization,svec)
  d = hcat(dvec...)
  p = merge_param_data(pvec)
  r = merge_realizations(rvec)
  i = get_dof_map(first(svec))
  ReshapedSnapshots(d,p,i,r)
end

function merge_snapshots(svec::Vector{<:TransientSnapshotsWithIC})
  get_snaps(s::TransientSnapshotsWithIC) = s.snaps
  snaps = merge_snapshots(map(get_snaps,svec))
  d0vec = map(get_initial_data,svec)
  TransientSnapshotsWithIC(hcat(d0vec...),snaps)
end

function get_offline_snapshots(dir::String;kwargs...)
  svec = map(get_offline_dirs(dir;kwargs...)) do path
    deserialize(path)
  end
  merge_snapshots(svec)
end

function get_online_snapshots(dir::String)
  parts = split(dir,"/")
  @assert last(parts) == "sol"
  fesnaps = deserialize(get_online_dir_sol(dir))
  festats = load_stats(joinpath(dir,"..");label="online")
  fesnaps,festats
end

function get_offline_snapshots(
  dir,
  trian::Tuple{Vararg{Triangulation}};
  label="")

  dec = ()
  for (i,t) in enumerate(trian)
    deci = get_offline_snapshots(dir;label=RBSteady._get_label(label,i))
    dec = (dec...,deci)
  end
  return Contribution(dec,trian)
end

function get_offline_snapshots(
  dir,
  trian::Tuple{Vararg{Tuple{Vararg{Triangulation}}}};
  label="")

  dec = ()
  for (i,t) in enumerate(trian)
    deci = get_offline_snapshots(dir,t;label="$i")
    dec = (dec...,deci)
  end
  return dec
end

function get_offline_snapshots(dir::String,feop::ParamOperator;kwargs...)
  parts = split(dir,"/")
  @assert last(parts) ∈ ("res","jac")
  trians = last(parts) == "res" ? get_domains_res(feop) : get_domains_jac(feop)
  get_offline_snapshots(dir,trians;kwargs...)
end

function change_snaps_dof_map(s::GenericSnapshots,dof_map)
  pdata = get_param_data(s)
  r = get_realization(s)
  Snapshots(pdata,dof_map,r)
end

function change_snaps_dof_map(s::ReshapedSnapshots,dof_map)
  pdata = get_param_data(s)
  r = get_realization(s)
  Snapshots(pdata,dof_map,r)
end

function change_snaps_dof_map(s::TransientSnapshotsWithIC,dof_map)
  TransientSnapshotsWithIC(s.initial_data,change_snaps_dof_map(s.snaps,dof_map))
end

function change_dof_map(resjac::Contribution,dof_map::Contribution)
  resjac′ = ()
  for i in eachindex(resjac)
    resjac′ = (resjac′...,change_snaps_dof_map(resjac[i],dof_map[i]))
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

function get_offline_online_solutions(dir::String,feop::ParamOperator,method=:pod)
  fesnaps = get_offline_snapshots(joinpath(dir,"sol"))
  x,festats = get_online_snapshots(joinpath(dir,"sol"))
  if method != :pod
    dof_map = get_dof_map(feop)
    fesnaps = change_snaps_dof_map(fesnaps,dof_map)
    x = change_snaps_dof_map(x,dof_map)
  end
  fesnaps,(x,festats)
end

function get_2d_poisson_info(M,method=:pod;nparams=5,nparams_res=5,nparams_jac=2,sampling=:halton,start=1)
  order = 1
  degree = 2*order

  domain = (0,1,0,1)
  partition = (M,M)
  model = method==:pod ? CartesianDiscreteModel(domain,partition) : TProductDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  pdomain = (1,5,1,5,1,5,1,5,1,5)
  pspace = sampling==:halton ? ParamSpace(pdomain;sampling,start) : ParamSpace(pdomain;sampling)

  ν(μ) = x -> μ[1]#*exp(-μ[2]*x[1])
  νμ(μ) = parameterize(ν,μ)

  f(μ) = x -> μ[3]
  fμ(μ) = parameterize(f,μ)

  g(μ) = x -> μ[4]#exp(-μ[4]*x[2])
  gμ(μ) = parameterize(g,μ)

  h(μ) = x -> μ[5]
  hμ(μ) = parameterize(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
  state_reduction = method==:pod ? Reduction(1e-4,energy;nparams) : Reduction(fill(1e-4,2),energy;nparams)

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  return feop,rbsolver
end

function get_3d_poisson_info(M,method=:pod;nparams=5,nparams_res=5,nparams_jac=2,sampling=:halton,start=1)
  order = 1
  degree = 2*order

  domain = (0,1,0,1,0,1)
  partition = (M,M,M)
  model = method==:pod ? CartesianDiscreteModel(domain,partition) : TProductDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[26])
  dΓn = Measure(Γn,degree)

  pdomain = (1,5,1,5,1,5,1,5,1,5)
  pspace = sampling==:halton ? ParamSpace(pdomain;sampling,start) : ParamSpace(pdomain;sampling)

  ν(μ) = x -> μ[1]*exp(-μ[2]*x[1])
  νμ(μ) = parameterize(ν,μ)

  f(μ) = x -> μ[3]
  fμ(μ) = parameterize(f,μ)

  g(μ) = x -> exp(-μ[4]*x[2])
  gμ(μ) = parameterize(g,μ)

  h(μ) = x -> μ[5]
  hμ(μ) = parameterize(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
  state_reduction = method==:pod ? Reduction(1e-4,energy;nparams) : Reduction(fill(1e-4,3),energy;nparams)

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  return feop,rbsolver
end

function get_2d_heateq_info(M,method=:pod;nparams=5,nparams_res=5,nparams_jac=2,nparams_djac=1,sampling=:halton,start=1)
  order = 1
  degree = 2*order

  domain = (0,1,0,1)
  partition = (M,M)
  model = method==:pod ? CartesianDiscreteModel(domain,partition) : TProductDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  θ = 0.5
  dt = 0.0025
  t0 = 0.0
  tf = 10*dt

  pdomain = (1,5,1,5,1,5,1,5,1,5,1,5)
  tdomain = t0:dt:tf
  ptspace = sampling==:halton ? TransientParamSpace(pdomain,tdomain;sampling,start) : TransientParamSpace(pdomain,tdomain;sampling)

  ν(μ,t) = x -> μ[1]*exp(-μ[2]*sin(2pi*t/tf)*x[1])
  νμt(μ,t) = parameterize(ν,μ,t)

  f(μ,t) = x -> μ[3]
  fμt(μ,t) = parameterize(f,μ,t)

  g(μ,t) = x -> exp(-μ[4]*x[2])*(1-cos(2pi*t/tf)+sin(2pi*t/tf)/μ[5])
  gμt(μ,t) = parameterize(g,μ,t)

  h(μ,t) = x -> μ[6]
  hμt(μ,t) = parameterize(h,μ,t)

  u0(μ) = x -> 0.0
  u0μ(μ) = ParamFunction(u0,μ)

  stiffness(μ,t,u,v,dΩ) = ∫(νμt(μ,t)*∇(v)⋅∇(u))dΩ
  mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
  rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
  res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientLinearParamOperator((stiffness,mass),res,ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  state_reduction = method==:pod ? HighDimReduction(1e-4,energy;nparams) : HighDimReduction(fill(1e-4,3),energy;nparams)

  fesolver = ThetaMethod(LUSolver(),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  return feop,rbsolver,uh0μ
end

function get_3d_heateq_info(M,method=:pod;nparams=5,nparams_res=5,nparams_jac=2,nparams_djac=1,sampling=:halton,start=1)
  order = 1
  degree = 2*order

  domain = (0,1,0,1,0,1)
  partition = (M,M,M)
  model = method==:pod ? CartesianDiscreteModel(domain,partition) : TProductDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[26])
  dΓn = Measure(Γn,degree)

  θ = 0.5
  dt = 0.0025
  t0 = 0.0
  tf = 10*dt

  pdomain = (1,5,1,5,1,5,1,5,1,5,1,5)
  tdomain = t0:dt:tf
  ptspace = sampling==:halton ? TransientParamSpace(pdomain,tdomain;sampling,start) : TransientParamSpace(pdomain,tdomain;sampling)

  ν(μ,t) = x -> μ[1]*exp(-μ[2]*sin(2pi*t/tf)*x[1])
  νμt(μ,t) = parameterize(ν,μ,t)

  f(μ,t) = x -> μ[3]
  fμt(μ,t) = parameterize(f,μ,t)

  g(μ,t) = x -> exp(-μ[4]*x[2])*(1-cos(2pi*t/tf)+sin(2pi*t/tf)/μ[5])
  gμt(μ,t) = parameterize(g,μ,t)

  h(μ,t) = x -> μ[6]
  hμt(μ,t) = parameterize(h,μ,t)

  u0(μ) = x -> 0.0
  u0μ(μ) = ParamFunction(u0,μ)

  stiffness(μ,t,u,v,dΩ) = ∫(νμt(μ,t)*∇(v)⋅∇(u))dΩ
  mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
  rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
  res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientLinearParamOperator((stiffness,mass),res,ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  state_reduction = method==:pod ? HighDimReduction(1e-4,energy;nparams) : HighDimReduction(fill(1e-4,4),energy;nparams)

  fesolver = ThetaMethod(LUSolver(),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  return feop,rbsolver,uh0μ
end

function get_elasticity_info(M,method=:pod;nparams=5,nparams_res=5,nparams_jac=2,nparams_djac=1,sampling=:halton,start=1)
  order = 1
  degree = 2*order

  Myz = Int(M/8)
  domain = (0,1,0,1/8,0,1/8)
  partition = (M,Myz,Myz)
  model = method==:pod ? CartesianDiscreteModel(domain,partition) : TProductDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn1 = BoundaryTriangulation(model,tags=[22])
  dΓn1 = Measure(Γn1,degree)
  Γn2 = BoundaryTriangulation(model,tags=[24])
  dΓn2 = Measure(Γn2,degree)
  Γn3 = BoundaryTriangulation(model,tags=[26])
  dΓn3 = Measure(Γn3,degree)

  θ = 0.5
  dt = 0.01
  t0 = 0.0
  tf = 10*dt
  tdomain = t0:dt:tf

  pdomain = (1e10,9*1e10,0.25,0.42,-4*1e5,4*1e5,-4*1e5,4*1e5,-4*1e5,4*1e5)
  ptspace = TransientParamSpace(pdomain,tdomain)

  λ(μ) = μ[1]*μ[2]/((1+μ[2])*(1-2*μ[2]))
  p(μ) = μ[1]/(2(1+μ[2]))

  σ(μ,t) = ε -> exp(sin(2pi*t/tf))*(λ(μ)*tr(ε)*one(ε) + 2*p(μ)*ε)
  σμt(μ,t) = TransientParamFunction(σ,μ,t)

  g(μ,t) = x -> VectorValue(0.0,0.0,0.0)
  gμt(μ,t) = parameterize(g,μ,t)

  h1(μ,t) = x -> VectorValue(μ[3]*sin(2pi*t/tf),0.0,0.0)
  h1μt(μ,t) = parameterize(h1,μ,t)

  h2(μ,t) = x -> VectorValue(0.0,μ[4]*sin(2pi*t/tf),0.0)
  h2μt(μ,t) = parameterize(h2,μ,t)

  h3(μ,t) = x -> VectorValue(0.0,0.0,μ[5]*t)
  h3μt(μ,t) = parameterize(h3,μ,t)

  u0(μ) = x -> VectorValue(0.0,0.0,0.0)
  u0μ(μ) = ParamFunction(u0,μ)

  stiffness(μ,t,u,v,dΩ) = ∫( ε(v) ⊙ (σμt(μ,t)∘ε(u)) )dΩ
  mass(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
  rhs(μ,t,v,dΓn1,dΓn2,dΓn3) = ∫(h1μt(μ,t)⋅v)dΓn1 + ∫(h2μt(μ,t)⋅v)dΓn2 + ∫(h3μt(μ,t)⋅v)dΓn3
  res(μ,t,u,v,dΩ,dΓn1,dΓn2,dΓn3) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΓn1,dΓn2,dΓn3)

  trian_res = (Ω,Γn1,Γn2,Γn3)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientLinearParamOperator((stiffness,mass),res,ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  energy(du,v) = ∫(v⋅du)dΩ + ∫(∇(v)⊙∇(du))dΩ

  state_reduction = method==:pod ? HighDimReduction(1e-4,energy;nparams) : HighDimReduction(fill(1e-4,5),energy;nparams)

  fesolver = ThetaMethod(LUSolver(),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  return feop,rbsolver,uh0μ
end

function get_test_info(args...;label="2d_heateq",nparams_djac=1,kwargs...)
  if label=="2d_poisson"
    get_2d_poisson_info(args...;kwargs...)
  elseif label=="3d_poisson"
    get_3d_poisson_info(args...;kwargs...)
  elseif label=="2d_heateq"
    get_2d_heateq_info(args...;nparams_djac,kwargs...)
  elseif label=="3d_heateq"
    get_3d_heateq_info(args...;nparams_djac,kwargs...)
  else label=="elasticity"
    get_elasticity_info(args...;nparams_djac,kwargs...)
  end
end

function generate_snaps(M;label="2d_heateq",id=string(Int(rand(1:1e4))),kwargs...)
  if id == "online"
    sampling = :uniform
    start = nothing
  else
    sampling = :halton
    start = parse(Int,id)
  end

  feop,rbsolver,args... = get_test_info(M;label,sampling,start,kwargs...)

  println("Generating snapshots for problem: $label, id: $id, M: $M")
  fesnaps,festats = solution_snapshots(rbsolver,feop,args...)
  ressnaps = residual_snapshots(rbsolver,feop,fesnaps)
  jacsnaps = jacobian_snapshots(rbsolver,feop,fesnaps)

  dir = datadir(label*"_$M")
  create_dir(dir)

  save(dir,festats;label=id)

  dir_sol = joinpath(dir,"sol")
  create_dir(dir_sol)
  save(dir_sol,fesnaps;label=id)

  dir_res = joinpath(dir,"res")
  create_dir(dir_res)
  save(dir_res,ressnaps;label=id)

  dir_jac = joinpath(dir,"jac")
  create_dir(dir_jac)
  save(dir_jac,jacsnaps,feop;label=id)
end

function main_snapshots(;
  M_poisson=(50,100,200),
  M_heateq=(50,100,200),
  M_elasticity=(48,64,80),
  kwargs...
  )

  for M in M_poisson
    generate_snaps(M;label="2d_poisson",kwargs...)
  end

  for M in M_poisson
    generate_snaps(M;label="3d_poisson",kwargs...)
  end

  for M in M_elasticity
    generate_snaps(M;label="elasticity",kwargs...)
  end
end

function try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps,method=:pod)
  try
    rbop = load_operator(dir_tol,feop)
    println("Load reduced operator at $dir_tol succeeded!")
    return rbop
  catch
    println("Load reduced operator at $dir_tol failed, must run offline phase")
    dir = joinpath(splitpath(dir_tol)[1:end-1])
    res = get_offline_snapshots(joinpath(dir,"res"),feop)
    jac = get_offline_snapshots(joinpath(dir,"jac"),feop)
    if method != :pod
      res = change_dof_map(res,get_dof_map_at_domains(feop))
      jac = change_dof_map(jac,get_sparse_dof_map_at_domains(feop))
    end
    rbop = reduced_operator(rbsolver,feop,fesnaps,jac,res)
    save(dir_tol,rbop)
    return rbop
  end
end

function main_rb(;method=:pod,M_test=(25,50,100),tols=(1e-1,1e-2,1e-3,1e-4,1e-5),label="2d_heateq")
  for M in M_test
    feop,rbsolver,args... = get_test_info(M,method;label,nparams=80,nparams_res=80,nparams_jac=20,nparams_djac=1)

    dir = datadir(label*"_$M")
    fesnaps,(x,festats) = get_offline_online_solutions(dir,feop,method)
    μ = get_realization(x)

    for tol in tols
      println("Running test $dir with tol = $tol")

      dir_tol = joinpath(dir,string(method)*"_"*string(tol))
      create_dir(dir_tol)

      rbsolver = update_solver(rbsolver,tol)
      rbop = try_loading_reduced_operator(dir_tol,rbsolver,feop,fesnaps,method)
      x̂,rbstats = solve(rbsolver,rbop,μ,args...)

      perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
      println(perf)
    end
  end
end

function run_rb(
  M_poisson=(25,50,100),
  M_heateq=(25,50,100),
  M_elasticity=(48,64,80)
  )

  main_rb(;method=:pod,M_test=M_poisson,label="2d_poisson",kwargs...)
  main_rb(;method=:ttsvd,M_test=M_poisson,label="2d_poisson",kwargs...)

  main_rb(;method=:pod,M_test=M_poisson,label="3d_poisson",kwargs...)
  main_rb(;method=:ttsvd,M_test=M_poisson,label="3d_poisson",kwargs...)

  main_rb(;method=:pod,M_test=M_heateq,label="3d_heateq",kwargs...)
  main_rb(;method=:ttsvd,M_test=M_heateq,label="3d_heateq",kwargs...)

  main_rb(;method=:pod,M_test=M_elasticity,label="elasticity",kwargs...)
  main_rb(;method=:ttsvd,M_test=M_elasticity,label="elasticity",kwargs...)

end

M_test = (320,700)
for M in M_test
  for id in (string.(1:5:80)...,"online")
    generate_snaps(M;id,label="2d_poisson")
  end
end
end
