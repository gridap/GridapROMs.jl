module TimeAffineTests

using DrWatson
using Gridap
using Gridap.MultiField
using GridapROMs
using GridapROMs.ParamODEs
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.Utils
using Plots
using Random
using Test

function stokes(method,strategies;tol=1e-4,nparams=15,nparams_res=5,nparams_jac=6)
  Random.seed!(1234)

  pdomain = (1,10,-1,5,1,2)

  domain = (0,1,0,1)
  partition = (5,5)
  model = method==:ttsvd ? TProductDiscreteModel(domain,partition) : CartesianDiscreteModel(domain,partition)

  order = 2
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)

  a(μ,t) = x -> 1+exp(-μ[1]*x[1])
  aμ(μ,t) = parameterise(a,μ,t)

  g(μ,t) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2])*t,0.0)*(x[1]==0.0)
  gμt(μ,t) = parameterise(g,μ,t)

  u0(μ) = x -> VectorValue(0.0,0.0)
  u0μ(μ) = parameterise(u0,μ)
  p0(μ) = x -> 0.0
  p0μ(μ) = parameterise(p0,μ)

  stiffness(μ,t,(u,p),(v,q)) = ∫(aμ(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  mass(μ,t,(uₜ,pₜ),(v,q)) = ∫(v⋅uₜ)dΩ
  rhs(μ,t,(v,q)) = ∫(v⋅VectorValue(0.0,0.0))dΩ
  domains = FEDomains((Ω,),((Ω,),(Ω,)))

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_u = TransientTrialParamFESpace(test_u,gμt)
  trial_p = TransientTrialParamFESpace(test_p)
  test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

  dt = 0.01
  t0 = 0.0
  tdomain = t0:dt:10*dt
  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  energy = BlockNorm((H1(),L2()))
  tolrank = method==:ttsvd ? fill(tol,4) : tol
  state_reduction = TransientReduction(DivCoupling(),tolrank,energy;nparams,sketch=:none)

  fesolver = ThetaMethod(LUSolver(),dt,0.5)
  c = TimeCombination(fesolver)
  residual_reduction = TransientHyperReduction(c,state_reduction;nparams=nparams_res)
  jacobian_reduction = ntuple(2) do k
    TransientHyperReduction(CombinationOrder{k}(c),state_reduction;nparams=nparams_jac,hypred_strategy=strategies[k])
  end
  rbsolver = RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)

  feop = TransientLinearParamOperator(rhs,(stiffness,mass),TransientParamSpace(pdomain,tdomain),trial,test,domains)
  fesnaps = solution_snapshots(rbsolver,feop,xh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=5,sampling=:uniform)
  x̂ = solution_snapshots(rbsolver,rbop,μon,xh0μ)
  x = solution_snapshots(rbsolver,feop,μon,xh0μ)
  err = rom_performance(rbsolver,rbop,x,x̂).error
  return rbop,err,feop,μon,x̂,xh0μ,rbsolver
end

function heat(;tol=1e-4)
  model = CartesianDiscreteModel((0,1,0,1),(8,8))
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2)

  a(μ,t) = x -> 1+μ[1]*x[1]
  aμ(μ,t) = parameterise(a,μ,t)
  f(μ,t) = x -> μ[2]
  fμ(μ,t) = parameterise(f,μ,t)
  u0(μ) = x -> 0.0
  u0μ(μ) = parameterise(u0,μ)

  stiffness(μ,t,u,v) = ∫(aμ(μ,t)*∇(v)⋅∇(u))dΩ
  mass(μ,t,uₜ,v) = ∫(v*uₜ)dΩ
  rhs(μ,t,v) = ∫(fμ(μ,t)*v)dΩ
  domains = FEDomains((Ω,),((Ω,),(Ω,)))

  reffe = ReferenceFE(lagrangian,Float64,1)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags="boundary")
  trial = TransientTrialParamFESpace(test)

  dt = 0.01
  t0 = 0.0
  ptspace = TransientParamSpace((1,10,1,10),t0:dt:10*dt)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  state_reduction = TransientReduction(tol,H1();nparams=15,sketch=:none)
  fesolver = ThetaMethod(LUSolver(),dt,0.5)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res=5,nparams_jacs=(5,5),hypred_strategy=:time_affine)

  feop = TransientLinearParamOperator(rhs,(stiffness,mass),ptspace,trial,test,domains)
  fesnaps = solution_snapshots(rbsolver,feop,uh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=5,sampling=:uniform)
  x̂ = solution_snapshots(rbsolver,rbop,μon,uh0μ)
  x = solution_snapshots(rbsolver,feop,μon,uh0μ)
  err = rom_performance(rbsolver,rbop,x,x̂).error
  return rbop,err
end

time_ids(a) = RBTransient.get_common_time_domain(a)

@testset "time-independent jacobians, $method" for method in (:pod,:ttsvd)
  _,err_deim, = stokes(method,(:deim,:deim))
  rbop,err,feop,μon,x̂,xh0μ,rbsolver = stokes(method,(:time_affine,:time_affine))

  lhs = RBSteady.get_lhs(rbop)
  @test all(lhsk -> all(c -> c isa TimeAffineHRProjection,get_contributions(lhsk)),lhs)
  @test time_ids(lhs) == [1]
  @test num_reduced_dofs(first(get_contributions(lhs[1]))[1,1]) > 1
  @test maximum(err) < 1e-2
  @test err ≈ err_deim rtol=1e-6

  _,err_mixed, = stokes(method,(:time_affine,:deim))
  @test err_mixed ≈ err_deim rtol=1e-6

  dir = datadir("time_affine_save")
  mkpath(dir)
  save(dir,rbop)
  rbop_load = load_operator(dir,feop)
  x̂_load = solution_snapshots(rbsolver,rbop_load,μon,xh0μ)
  @test collect(x̂_load) ≈ collect(x̂)
  rm(dir;recursive=true)

  dir = datadir("time_affine_diagnostics")
  mkpath(dir)
  run_test(dir,rbsolver,feop,[1e-2,1e-4],xh0μ)
  println(rom_diagnostics(dir,rbsolver,feop,xh0μ))
  rm(dir;recursive=true)
end

@testset "time-independent residual" begin
  rbop,err = heat()
  @test time_ids(RBSteady.get_rhs(rbop)) == [1]
  @test time_ids(RBSteady.get_lhs(rbop)) == [1]
  @test maximum(err) < 1e-2
end

end