module StokesEmbedded

using Gridap
using Gridap.MultiField
using GridapEmbedded
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function coupling(method=:pod)
  if method == :pod
    ((du,dp),(v,q)) -> ∫(dp*(∇⋅(v)))dΩ
  else method == :ttsvd
    ((du,dp),(v,q)) -> ∫(dp*∂₁(v))dΩbg + ∫(dp*∂₂(v))dΩbg
  end
end

function main(
  method=:pod,n=20;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,x0=Point(0.5,0.5),R=0.3
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  pmin = Point(0,0)
  pmax = Point(1,1)
  dp = pmax - pmin
  h = dp[1]/n

  geo = !disk(R,x0=x0)

  domain = (0,1,0,1)
  partition = (n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(domain,partition)
  else
    bgmodel = CartesianDiscreteModel(domain,partition)
  end

  order = 2
  degree = 2*order

  cutgeo = cut(bgmodel,geo)

  Ωbg = Triangulation(bgmodel)
  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL_IN)
  Γ = EmbeddedBoundary(cutgeo)
  Γn = BoundaryTriangulation(Ωact,tags="boundary")

  dΩbg = Measure(Ωbg,degree)
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)
  dΓn = Measure(Γn,degree)

  nΓ = get_normal_vector(Γ)
  nΓn = get_normal_vector(Γn)

  u(μ) = x -> VectorValue(x[1]*exp(-μ[1]*x[1]),μ[2]*x[2])
  p(μ) = x -> x[1] - sin(μ[3]*x[2])
  f(μ) = x -> - Δ(u(μ))(x) + ∇(p(μ))(x)
  g(μ) = x -> (∇⋅u(μ))(x)

  uμ(μ) = parameterize(u,μ)
  pμ(μ) = parameterize(p,μ)
  fμ(μ) = parameterize(f,μ)
  gμ(μ) = parameterize(g,μ)

  γ = order*(order+1)

  a(μ,(u,p),(v,q),dΩ,dΓ) =
    ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ +
    ∫( (γ/h)*v⋅u - v⋅(nΓ⋅∇(u)) - (nΓ⋅∇(v))⋅u + (p*nΓ)⋅v + (q*nΓ)⋅u ) * dΓ

  l(μ,(v,q),dΩ,dΓ,dΓn) =
    ∫( v⋅fμ(μ) - q*gμ(μ) ) * dΩ +
    ∫( (γ/h)*v⋅uμ(μ) - (nΓ⋅∇(v))⋅uμ(μ) + (q*nΓ)⋅uμ(μ) ) * dΓ +
    ∫( v⋅(nΓn⋅∇(uμ(μ))) - (nΓn⋅v)*pμ(μ) ) * dΓn

  res(μ,(u,p),(v,q),dΩ,dΓ,dΓn) = a(μ,(u,p),(v,q),dΩ,dΓ) - l(μ,(v,q),dΩ,dΓ,dΓn)

  trian_res = (Ω,Γ,Γn)
  trian_jac = (Ω,Γ)
  domains = FEDomains(trian_res,trian_jac)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)

  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testbg_u = FESpace(Ωbg,reffe_u,conformity=:H1)
  testbg_p = FESpace(Ωbg,reffe_p,conformity=:H1)
  testact_u = FESpace(Ωact,reffe_u,conformity=:H1)
  testact_p = FESpace(Ωact,reffe_p,conformity=:H1)
  testagg_u = AgFEMSpace(testact_u,aggregates)
  testagg_p = AgFEMSpace(testact_p,aggregates)

  test_u = DirectSumFESpace(testbg_u,testagg_u)
  test_p = DirectSumFESpace(testbg_p,testagg_p)
  trial_u = ParamTrialFESpace(test_u,gμ)
  trial_p = ParamTrialFESpace(test_p)
  test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = ExtensionLinearParamOperator(res,a,pspace,trial,test,domains)

  tolrank = tol_or_rank(tol,rank)
  energy((du,dp),(v,q)) = ∫(du⋅v)dΩbg  + ∫(dp*q)dΩbg + ∫(∇(v)⊙∇(du))dΩbg
  state_reduction = SupremizerReduction(coupling(method),tolrank,energy;nparams,sketch)

  extension = BlockExtension([HarmonicExtension(),ZeroExtension()])
  fesolver = ExtensionSolver(LUSolver(),extension)
  rbsolver = RBSolver(fesolver,state_reduction)

  # offline
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  # online
  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  # test
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
  println(perf)
end


function main_transient(
  method=:pod,n=20;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,x0=Point(0.5,0.5),R=0.3
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

  θ = 1
  dt = 0.01
  t0 = 0.0
  tf = 30*dt
  tdomain = t0:dt:tf

  pdomain = (1,10,1,10,1,10)
  ptspace = TransientParamSpace(pdomain,tdomain)

  pmin = Point(0,0)
  pmax = Point(1,1)
  dp = pmax - pmin
  h = dp[1]/n

  geo = !disk(R,x0=x0)

  domain = (0,1,0,1)
  partition = (n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(domain,partition)
  else
    bgmodel = CartesianDiscreteModel(domain,partition)
  end

  order = 2
  degree = 2*order

  cutgeo = cut(bgmodel,geo)

  Ωbg = Triangulation(bgmodel)
  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL_IN)
  Γ = EmbeddedBoundary(cutgeo)
  Γn = BoundaryTriangulation(Ωact,tags="boundary")

  dΩbg = Measure(Ωbg,degree)
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)
  dΓn = Measure(Γn,degree)

  nΓ = get_normal_vector(Γ)
  nΓn = get_normal_vector(Γn)

  u(μ,t) = x -> t*VectorValue(x[1]*exp(-μ[1]*x[1]),μ[2]*x[2])
  p(μ,t) = x -> t*x[1] - sin(μ[3]*x[2])
  f(μ,t) = x -> - Δ(u(μ,t))(x) + ∇(p(μ))(x)
  g(μ,t) = x -> (∇⋅u(μ,t))(x)
  u0(μ) = x -> VectorValue(0.0,0.0)
  p0(μ) = x -> 0.0

  uμt(μ,t) = parameterize(u,μ,t)
  pμt(μ,t) = parameterize(p,μ,t)
  fμt(μ,t) = parameterize(f,μ,t)
  gμt(μ,t) = parameterize(g,μ,t)
  u0μ(μ) = parameterize(u0,μ)
  p0μ(μ) = parameterize(p0,μ)

  γ = order*(order+1)

  a(μ,t,(u,p),(v,q),dΩ,dΓ) =
    ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ +
    ∫( (γ/h)*v⋅u - v⋅(nΓ⋅∇(u)) - (nΓ⋅∇(v))⋅u + (p*nΓ)⋅v + (q*nΓ)⋅u ) * dΓ

  m(μ,t,(u,p),(v,q),dΩ) = ∫( v⋅u ) * dΩ

  l(μ,t,(v,q),dΩ,dΓ,dΓn) =
    ∫( v⋅fμt(μ,t) - q*gμt(μ,t) ) * dΩ +
    ∫( (γ/h)*v⋅uμt(μ,t) - (nΓ⋅∇(v))⋅uμt(μ,t) + (q*nΓ)⋅uμt(μ,t) ) * dΓ +
    ∫( v⋅(nΓn⋅∇(uμt(μ,t))) - (nΓn⋅v)*pμt(μ,t) ) * dΓn

  res(μ,t,(u,p),(v,q),dΩ,dΓ,dΓn) = ∫( v⋅∂t(u) )*dΩ + a(μ,t,(u,p),(v,q),dΩ,dΓ) - l(μ,t,(v,q),dΩ,dΓ,dΓn)

  trian_res = (Ω,Γ,Γn)
  trian_a = (Ω,Γ)
  trian_m = (Ω,)
  domains = FEDomains(trian_res,(trian_a,trian_m))

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)

  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testbg_u = FESpace(Ωbg,reffe_u,conformity=:H1)
  testbg_p = FESpace(Ωbg,reffe_p,conformity=:H1)
  testact_u = FESpace(Ωact,reffe_u,conformity=:H1)
  testact_p = FESpace(Ωact,reffe_p,conformity=:H1)
  testagg_u = AgFEMSpace(testact_u,aggregates)
  testagg_p = AgFEMSpace(testact_p,aggregates)

  test_u = DirectSumFESpace(testbg_u,testagg_u)
  test_p = DirectSumFESpace(testbg_p,testagg_p)
  trial_u = TransientTrialParamFESpace(test_u,gμt)
  trial_p = TransientTrialParamFESpace(test_p)
  test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = TransientExtensionLinearParamOperator((a,m),res,ptspace,trial,test,domains)

  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  tolrank = method==:ttsvd ? fill(tol_or_rank(tol,rank),4) : tol_or_rank(tol,rank)
  energy((du,dp),(v,q)) = ∫(du⋅v)dΩbg  + ∫(dp*q)dΩbg + ∫(∇(v)⊙∇(du))dΩbg
  state_reduction = TransientReduction(coupling(method),tolrank,energy;nparams,sketch)

  extension = BlockExtension([HarmonicExtension(),ZeroExtension()])
  fesolver = ThetaMethod(ExtensionSolver(LUSolver(),extension),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction)

  # offline
  fesnaps, = solution_snapshots(rbsolver,feop,xh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  # online
  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon,xh0μ)

  # test
  x,festats = solution_snapshots(rbsolver,feop,μon,xh0μ)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
  println(perf)
end

main_transient(:pod)
main_transient(:ttsvd)

end
