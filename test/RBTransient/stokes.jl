module TransientStokes

using Gridap
using Gridap.MultiField
using GridapROMs

function main(
  method=:pod,compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=2
  )

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :mdeim

  println("Running test with $compression ($method, $hypred_strategy) strategy")

  pdomain = (1,10,-1,5,1,2)

  domain = (0,1,0,1)
  partition = (10,10)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(μ,t) = x -> μ[1]*exp(sin(t))
  aμt(μ,t) = parameterise(a,μ,t)

  g(μ,t) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2])*t,0.0)*(x[1]==0.0)
  gμt(μ,t) = parameterise(g,μ,t)

  u0(μ) = x -> VectorValue(0.0,0.0)
  u0μ(μ) = parameterise(u0,μ)
  p0(μ) = x -> 0.0
  p0μ(μ) = parameterise(p0,μ)

  stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
  res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  trial_u = TransientTrialParamFESpace(test_u,gμt)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_p = TransientTrialParamFESpace(test_p)
  test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  coupling((du,dp),(v,q)) = method==:pod ? ∫(dp*(∇⋅(v)))dΩ : ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
  if method == :pod
    state_reduction = HighDimReduction(coupling,tol,energy;nparams,sketch,compression,ncentroids)
  elseif method == :ttsvd
    state_reduction = HighDimReduction(coupling,fill(tol,4),energy;nparams,sketch,compression,ncentroids)
  end

  θ = 0.5
  dt = 0.01
  t0 = 0.0
  tf = 10*dt
  tdomain = t0:dt:tf

  ptspace = TransientParamSpace(pdomain,tdomain)

  fesolver = ThetaMethod(LUSolver(),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jacs=(nparams_jac,nparams_jac),hypred_strategy)

  feop = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop,xh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon,xh0μ)
  x,festats = solution_snapshots(rbsolver,feop,μon,xh0μ)
  perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)

  println(perf)
end

for method in (:pod,:ttsvd), compression in (:local,:global), hypred_strategy in (:mdeim,:sopt,:rbf,:none,:affine)
  main(method,compression,hypred_strategy)
end

end

using DrWatson
using Gridap
using Gridap.MultiField
using GridapROMs

include("../../examples/ExamplesInterface.jl")

method=:pod
compression=:local
hypred_strategy=:mdeim
  tol=1e-4
  nparams=50
  nparams_res=floor(Int,nparams/3)
  nparams_jac=floor(Int,nparams/4)
  sketch=:sprn
  ncentroids=2

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :mdeim

  println("Running test with $compression ($method, $hypred_strategy) strategy")

  pdomain = (1,10,-1,5,1,2)

  domain = (0,1,0,1)
  partition = (10,10)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(μ,t) = x -> μ[1]*exp(sin(t))
  aμt(μ,t) = parameterise(a,μ,t)

  g(μ,t) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2])*t,0.0)*(x[1]==0.0)
  gμt(μ,t) = parameterise(g,μ,t)

  u0(μ) = x -> VectorValue(0.0,0.0)
  u0μ(μ) = parameterise(u0,μ)
  p0(μ) = x -> 0.0
  p0μ(μ) = parameterise(p0,μ)

  stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
  res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  trial_u = TransientTrialParamFESpace(test_u,gμt)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_p = TransientTrialParamFESpace(test_p)
  test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  coupling((du,dp),(v,q)) = method==:pod ? ∫(dp*(∇⋅(v)))dΩ : ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
  if method == :pod
    state_reduction = HighDimReduction(coupling,tol,energy;nparams,sketch,compression,ncentroids)
  elseif method == :ttsvd
    state_reduction = HighDimReduction(coupling,fill(tol,4),energy;nparams,sketch,compression,ncentroids)
  end

  θ = 0.5
  dt = 0.01
  t0 = 0.0
  tf = 10*dt
  tdomain = t0:dt:tf

  ptspace = TransientParamSpace(pdomain,tdomain)

  fesolver = ThetaMethod(LUSolver(),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jacs=(nparams_jac,nparams_jac),hypred_strategy)
  
  feop = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains)

  dir = datadir("diagnostics")
  isdir(dir) && rm(dir;recursive=true)
  create_dir(dir)
  
  tols = [1e-1,1e-3,1e-5]
  run_test(dir,rbsolver,feop,tols,xh0μ)

  dgn = rom_diagnostics(dir,rbsolver,feop,xh0μ)

  # fesnaps, = solution_snapshots(rbsolver,feop,xh0μ)
  # rbop = reduced_operator(rbsolver,feop,fesnaps)

  # μon = realisation(feop;nparams=8,sampling=:uniform)
  # x̂,rbstats = solve(rbsolver,rbop,μon,xh0μ)
  # x,festats = solution_snapshots(rbsolver,feop,μon,xh0μ)
  # perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)

  # println(perf)

  # rhs = rbop.rhs.values[1]
  # trian = rbop.rhs.trians[1]
  # ids = trian.tface_to_mface

  # union(map(i->get_integration_cells(rhs[1].reductions[i].interpolation),1:4)...)
  
  # red = rbsolver.residual_reduction
  # ress = residual_snapshots(rbsolver,feop,fesnaps)
  # s = ress.values[1]
  # trian = ress.trians[1]
  # hyper_reds = map(eachindex(s)) do i
  #   hyper_red, = RBSteady.reduced_form(red,s[i],trian,rbop.test[i])
  #   hyper_red
  # end

  # hyper_red = BlockHRProjection(hyper_reds,s.touched)
  # red_trian = reduced_triangulation(trian,hyper_red)

  # interp = get_interpolation(hyper_red)
  using GridapROMs.RBSteady
  
  s,jacs,ress = load_problem_snapshots(dir,rbsolver,feop,xh0μ;label=online_label)

  name = first(sort(readdir(dir)))
  subdir = joinpath(dir,name)

  rbop = load_operator(subdir,feop)
  diagnostics = RBSteady.offline_diagnostics(rbop)

  proj_err = projection_error(rbsolver,rbop,s)
  # err_res,err_jac = hr_error(rbsolver,rbop,ress,jacs,s)
  i = 1
  μi = first(get_realisation(s))
  opi = get_local(rbop,μi)
  si = select_snapshots(s,i)
  resi = select_snapshots(ress,i)
  jaci = select_snapshots(jacs,i)
  gsolver = RBSteady.change_context(rbsolver)
  # err_res_i,err_jac_i = RBSteady.hr_error(gsolver,opi,resi,jaci,si)
  μ = get_realisation(s)
  u = get_param_data(s)
  # err_res = RBSteady.hr_error_res(opi,ress,μ,u)
  rs = ress[1]