module DiagnosticTests

using Gridap
using Gridap.MultiField
using Test
using DrWatson

using GridapROMs

include("../../examples/ExamplesInterface.jl")

function main(
  method=:pod,compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=2
  )

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:none,:affine,:mdeim,:sopt,:rbf) ? hypred_strategy : :mdeim

  println("Running test with compression $method, $compression compressions, and $hypred_strategy hyper-reduction")

  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (20,20)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(μ) = x -> μ[1]*exp(-x[1])
  aμ(μ) = parameterise(a,μ)

  g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
  gμ(μ) = parameterise(g,μ)

  stiffness(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  res(μ,(u,p),(v,q),dΩ) = stiffness(μ,(u,p),(v,q),dΩ)

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  trial_u = ParamTrialFESpace(test_u,gμ)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_p = ParamTrialFESpace(test_p)
  test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

  energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ
  coupling((du,dp),(v,q)) = method == :pod ? ∫(dp*(∇⋅(v)))dΩ : ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ

  if method == :pod
    state_reduction = SupremizerReduction(coupling,tol,energy;nparams,sketch,compression,ncentroids)
  elseif method == :ttsvd
    state_reduction = SupremizerReduction(coupling,fill(tol,3),energy;nparams,sketch,compression,ncentroids)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  
  dir = datadir("diagnostics")
  rm(dir;recursive=true)
  create_dir(dir)

  tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
  run_test(dir,rbsolver,feop,tols)

  dgn = rom_diagnostics(dir,rbsolver,feop)
  println(dgn)
end

for method in (:pod,:ttsvd), compression in (:local,:global), hypred_strategy in (:none,:affine,:mdeim,:sopt,:rbf)
  main(method,compression,hypred_strategy)
end

end

using Gridap
using Gridap.MultiField
using Test
using DrWatson

using GridapROMs

include("../../examples/ExamplesInterface.jl")

method=:pod
compression=:local
hypred_strategy=:none 
  tol=1e-4
  nparams=50
  nparams_res=floor(Int,nparams/3)
  nparams_jac=floor(Int,nparams/4)
  sketch=:sprn
  ncentroids=2

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:none,:affine,:mdeim,:sopt,:rbf) ? hypred_strategy : :mdeim

  println("Running test with compression $method, $compression compressions, and $hypred_strategy hyper-reduction")

  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (20,20)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(μ) = x -> μ[1]*exp(-x[1])
  aμ(μ) = parameterise(a,μ)

  g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
  gμ(μ) = parameterise(g,μ)

  stiffness(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  res(μ,(u,p),(v,q),dΩ) = stiffness(μ,(u,p),(v,q),dΩ)

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  trial_u = ParamTrialFESpace(test_u,gμ)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_p = ParamTrialFESpace(test_p)
  test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

  energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ
  coupling((du,dp),(v,q)) = method == :pod ? ∫(dp*(∇⋅(v)))dΩ : ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ

  if method == :pod
    state_reduction = SupremizerReduction(coupling,tol,energy;nparams,sketch,compression,ncentroids)
  elseif method == :ttsvd
    state_reduction = SupremizerReduction(coupling,fill(tol,3),energy;nparams,sketch,compression,ncentroids)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)


  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

  dir = datadir("diagnostics")
  rm(dir;recursive=true)
  create_dir(dir)

  tols = [1e-1,1e-2]
  run_test(dir,rbsolver,feop,tols)

  dgn = rom_diagnostics(dir,rbsolver,feop)
  println(dgn)

  using GridapROMs.ParamAlgebra 
  using GridapROMs.ParamSteady 
  using GridapROMs.RBSteady 
  using Gridap.Algebra
  using Gridap.FESpaces

  fesnaps = load_snapshots(dir,label="online")
  jacs = load_jacobians(dir,feop)
  ress = load_residuals(dir,feop)
  op = RBSteady.get_local(rbop,first(μon))
  i = 1
  si = select_snapshots(fesnaps,[i])
  resi = select_snapshots(ress,[i])
  jaci = select_snapshots(jacs,[i])
  # err_res_i,err_jac_i = hr_error(opi,resi,jaci,si)
  μ = get_realisation(si)
  u = get_param_data(si)|> similar
  fill!(u,zero(eltype2(u)))
  # err_res = RBSteady.hr_error_res(op,resi,μ,u)
  nlop = parameterise(op,μ)
  b = RBSteady.allocate_diagnostic_residual(nlop,u)

  using Gridap.FESpaces
  uh = EvaluationFunction(nlop.paramcache.trial,u)
  v = get_fe_basis(get_test(op))

  using GridapROMs.Utils
  _trian_res = Utils.get_domains_res(op)
  using Gridap.ODEs
  rhs = RBSteady.get_rhs(op)
  dc = ODEs.get_res(op)(μ,uh,v)