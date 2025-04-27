module NewTests

using DrWatson
using Gridap
using GridapROMs

import GridapROMs.RBTransient: time_combinations

include("ExamplesInterface.jl")

function poisson_2d(M,method=:pod,sketch=:sprn)
  println("Running 2d poisson test with M = $M, method = $method")

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
  pspace = ParamSpace(pdomain)

  ν(μ) = x -> μ[1]+μ[2]*x[1]
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
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
  nparams = 80
  tol = method==:pod ? 1e-4 : fill(1e-4,2)

  state_reduction = Reduction(tol,energy;nparams,sketch)
  res_reduction = MDEIMReduction(Reduction(tol;nparams=50))
  jac_reduction = MDEIMReduction(Reduction(tol;nparams=20))

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction,res_reduction,jac_reduction)

  dir = datadir("2d_poisson_$(M)_$(method)")
  create_dir(dir)

  tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
  ExamplesInterface.run_test(dir,rbsolver,feop,tols)
end

function poisson_3d(M,method=:pod,sketch=:sprn)
  println("Running 3d poisson test with M = $M, method = $method")

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
  pspace = ParamSpace(pdomain)

  ν(μ) = x -> μ[1]+μ[2]*x[1]
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
  nparams = 80
  tol = method==:pod ? 1e-4 : fill(1e-4,3)

  state_reduction = Reduction(tol,energy;nparams,sketch)
  res_reduction = MDEIMReduction(Reduction(tol;nparams=50))
  jac_reduction = MDEIMReduction(Reduction(tol;nparams=20))

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction,res_reduction,jac_reduction)

  dir = datadir("3d_poisson_$(M)_$(method)")
  create_dir(dir)

  tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
  ExamplesInterface.run_test(dir,rbsolver,feop,tols)
end

function heateq_3d(M,method=:pod,sketch=:sprn)
  println("Running 3d heateq test with M = $M, method = $method")

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
  dt = 0.01
  t0 = 0.0
  tf = 10*dt
  tdomain = t0:dt:tf

  pdomain = (1,5,1,5,1,5,1,5,1,5,1,5)
  ptspace = TransientParamSpace(pdomain,tdomain)

  ν(μ,t) = x -> (μ[1]+μ[2]*x[1])*exp(sin(2pi*t/tf))
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

  fesolver = ThetaMethod(LUSolver(),dt,θ)

  nparams = 50
  tol = method==:pod ? 1e-4 : fill(1e-4,4)
  cres,cjac,cdjac = time_combinations(fesolver)

  state_reduction = TransientReduction(tol,energy;nparams,sketch)
  res_reduction = TransientMDEIMReduction(cres,TransientReduction(tol;nparams=50))
  jac_reduction = TransientMDEIMReduction(cjac,TransientReduction(tol;nparams=20))
  djac_reduction = TransientMDEIMReduction(cdjac,TransientReduction(tol;nparams=1))

  rbsolver = RBSolver(fesolver,state_reduction,res_reduction,(jac_reduction,djac_reduction))

  dir = datadir("3d_heateq_$(M)_$(method)")
  create_dir(dir)

  tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
  ExamplesInterface.run_test(dir,rbsolver,feop,tols,uh0μ)
end

function elasticity_3d(M,method=:pod,sketch=:sprn)
  println("Running 3d elasticity test with M = $M, method = $method")

  order = 2
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

  h1(μ,t) = x -> VectorValue(μ[3]*exp(sin(2pi*t/tf)),0.0,0.0)
  h1μt(μ,t) = parameterize(h1,μ,t)

  h2(μ,t) = x -> VectorValue(0.0,μ[4]*exp(sin(2pi*t/tf)),0.0)
  h2μt(μ,t) = parameterize(h2,μ,t)

  h3(μ,t) = x -> VectorValue(0.0,0.0,μ[5]*(1+t))
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

  fesolver = ThetaMethod(LUSolver(),dt,θ)

  nparams = 50
  tol = method==:pod ? 1e-4 : fill(1e-4,5)
  cres,cjac,cdjac = time_combinations(fesolver)

  state_reduction = TransientReduction(tol,energy;nparams,sketch)
  res_reduction = TransientMDEIMReduction(cres,TransientReduction(tol;nparams=50))
  jac_reduction = TransientMDEIMReduction(cjac,TransientReduction(tol;nparams=20))
  djac_reduction = TransientMDEIMReduction(cdjac,TransientReduction(tol;nparams=1))

  rbsolver = RBSolver(fesolver,state_reduction,res_reduction,(jac_reduction,djac_reduction))

  dir = datadir("3d_elasticity_$(M)_$(method)")
  create_dir(dir)

  tols = [1e-4,]
  ExamplesInterface.run_test(dir,rbsolver,feop,tols,uh0μ)
end

# for M in (250,350,460)
#   # poisson_2d(M,:pod)
#   poisson_2d(M,:ttsvd)
# end

# for M in (40,50,60)
#   # poisson_3d(M,:pod)
#   poisson_3d(M,:ttsvd)
# end

# for M in (40,)
#   # heateq_3d(M,:pod)
#   heateq_3d(M,:ttsvd)
# end

for M in (56,)
  # elasticity_3d(M,:pod)
  elasticity_3d(M,:ttsvd)
end

end
