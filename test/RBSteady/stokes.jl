module StokesEquation

using Gridap
using Gridap.MultiField
using Test
using DrWatson

using GridapROMs

function main(
  method=:pod,compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=15,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=2
  )

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :mdeim

  println("Running test with $compression ($method, $hypred_strategy) strategy")

  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (8,8)
  model = CartesianDiscreteModel(domain,partition)

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

  if method == :ttsvd
    test_u = TProductFESpace(Ω,(lagrangian,(VectorValue{2,Float64},order),(;));conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
    test_p = TProductFESpace(Ω,(lagrangian,(Float64,order-1),(;));conformity=:H1)
  else
    reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
    reffe_p = ReferenceFE(lagrangian,Float64,order-1)
    test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  end
  trial_u = ParamTrialFESpace(test_u,gμ)
  trial_p = ParamTrialFESpace(test_p)
  test = MultiFieldFESpace([test_u,test_p])
  trial = MultiFieldFESpace([trial_u,trial_p])

  energy = BlockOperator((H1(),L2()))
  coupling = DivCoupling()

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
  perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)

  println(perf)
end

for method in (:pod,:ttsvd), compression in (:local,:global), hypred_strategy in (:mdeim,:sopt,:rbf,:none,:affine)
  main(method,compression,hypred_strategy)
end

end