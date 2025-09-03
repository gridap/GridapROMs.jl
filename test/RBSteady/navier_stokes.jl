module NavierStokesEquation

using Gridap
using Gridap.MultiField
using GridapSolvers
using GridapSolvers.NonlinearSolvers
using Test
using DrWatson

using GridapROMs

function main(
  method=:pod,compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=2
  )

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:rbf) ? hypred_strategy : :mdeim

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

  Re = 100
  a(μ) = x -> μ[1]/Re*exp(-x[1])
  aμ(μ) = parameterize(a,μ)

  g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
  gμ(μ) = parameterize(g,μ)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v,dΩ) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
  dc(u,du,v,dΩ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

  jac_lin(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  res_lin(μ,(u,p),(v,q),dΩ) = jac_lin(μ,(u,p),(v,q),dΩ)

  res_nlin(μ,(u,p),(v,q),dΩ) = c(u,v,dΩ)
  jac_nlin(μ,(u,p),(du,dp),(v,q),dΩ) = dc(u,du,v,dΩ)

  trian_res = (Ω,)
  trian_jac = (Ω,)
  domains_lin = FEDomains(trian_res,trian_jac)
  domains_nlin = FEDomains(trian_res,trian_jac)

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
  else method == :ttsvd
    state_reduction = SupremizerReduction(coupling,fill(tol,3),energy;nparams,sketch,compression,ncentroids)
  end

  fesolver = NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop_lin = LinearParamOperator(res_lin,jac_lin,pspace,trial,test,domains_lin)
  feop_nlin = ParamOperator(res_nlin,jac_nlin,pspace,trial,test,domains_nlin)
  feop = LinearNonlinearParamOperator(feop_lin,feop_nlin)
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

  println(perf)
end

for method in (:pod,:ttsvd), compression in (:local,:global), hypred_strategy in (:mdeim,)
  main(method,compression,hypred_strategy)
end

end
