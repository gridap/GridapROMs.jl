module TransientNavierStokes

using Gridap
using Gridap.MultiField
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),nparams_djac=1,sketch=:sprn
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
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
  aμt(μ,t) = TransientParamFunction(a,μ,t)

  g(μ,t) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2])*t,0.0)*(x[1]==0.0)
  gμt(μ,t) = TransientParamFunction(g,μ,t)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
  dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

  u0(μ) = x -> VectorValue(0.0,0.0)
  u0μ(μ) = ParamFunction(u0,μ)
  p0(μ) = x -> 0.0
  p0μ(μ) = ParamFunction(p0,μ)

  stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
  res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

  res_nlin(μ,t,(u,p),(v,q),dΩ) = c(u,v,dΩ)
  jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

  trian_res = (Ω,)
  trian_jac = (Ω,)
  trian_jac_t = (Ω,)
  domains_lin = FEDomains(trian_res,(trian_jac,trian_jac_t))
  domains_nlin = FEDomains(trian_res,(trian_jac,))

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

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
    state_reduction = HighOrderReduction(coupling,tolrank,energy;nparams,sketch)
  else method == :ttsvd
    tolranks = fill(tolrank,4)
    ttcoupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
    state_reduction = HighOrderReduction(ttcoupling,tolranks,energy;nparams)
  end

  θ = 0.5
  dt = 0.01
  t0 = 0.0
  tf = 10*dt
  tdomain = t0:dt:tf

  fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  ptspace_uniform = TransientParamSpace(pdomain,tdomain;sampling=:uniform)
  feop_lin_uniform = TransientLinearParamOperator((stiffness,mass),res,ptspace_uniform,
    trial,test,domains_lin;constant_forms=(false,true))
  feop_nlin_uniform = TransientParamOperator(res_nlin,jac_nlin,ptspace_uniform,
    trial,test,domains_nlin)
  feop_uniform = LinearNonlinearTransientParamOperator(feop_lin_uniform,feop_nlin_uniform)
  μon = realization(feop_uniform;nparams=10)
  x,festats = solution_snapshots(rbsolver,feop_uniform,μon,xh0μ)

  for sampling in (:uniform,:halton,:latin_hypercube,:tensorial_uniform)
    println("Running $method test with sampling strategy $sampling")
    ptspace = TransientParamSpace(pdomain,tdomain;sampling)
    feop_lin = TransientLinearParamOperator((stiffness,mass),res,ptspace,
      trial,test,domains_lin)
    feop_nlin = TransientParamOperator(res_nlin,jac_nlin,ptspace,
      trial,test,domains_nlin)
    feop = LinearNonlinearTransientParamOperator(feop_lin,feop_nlin)

    fesnaps, = solution_snapshots(rbsolver,feop,xh0μ)
    rbop = reduced_operator(rbsolver,feop,fesnaps)
    x̂,rbstats = solve(rbsolver,rbop,μon,xh0μ)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

    println(perf)
  end

end

main(:pod)
main(:ttsvd)

end
