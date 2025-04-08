module NavierStokesEquation

using Gridap
using Gridap.MultiField
using GridapSolvers
using GridapSolvers.NonlinearSolvers
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
  pdomain = (1,10,-1,5,1,2)

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
  aμ(μ) = ParamFunction(a,μ)

  g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
  gμ(μ) = ParamFunction(g,μ)

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

  energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  trial_u = ParamTrialFESpace(test_u,gμ)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_p = ParamTrialFESpace(test_p)
  test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
    state_reduction = SupremizerReduction(coupling,tolrank,energy;nparams,sketch)
  else method == :ttsvd
    tolranks = fill(tolrank,4)
    ttcoupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
    state_reduction = SupremizerReduction(ttcoupling,tolranks,energy;nparams)
  end

  fesolver = NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  pspace_uniform = ParamSpace(pdomain;sampling=:uniform)
  feop_lin_uniform = LinearParamOperator(res_lin,jac_lin,pspace_uniform,trial,test,domains_lin)
  feop_nlin_uniform = ParamOperator(res_nlin,jac_nlin,pspace_uniform,trial,test,domains_nlin)
  feop_uniform = LinearNonlinearParamOperator(feop_lin_uniform,feop_nlin_uniform)
  μon = realization(feop_uniform;nparams=10)
  x,festats = solution_snapshots(rbsolver,feop_uniform,μon)

  for sampling in (:uniform,:halton,:latin_hypercube,:tensorial_uniform)
    println("Running $method test with sampling strategy $sampling")
    pspace = ParamSpace(pdomain;sampling)
    feop_lin = LinearParamOperator(res_lin,jac_lin,pspace,trial,test,domains_lin)
    feop_nlin = ParamOperator(res_nlin,jac_nlin,pspace,trial,test,domains_nlin)
    feop = LinearNonlinearParamOperator(feop_lin,feop_nlin)

    fesnaps, = solution_snapshots(rbsolver,feop)
    rbop = reduced_operator(rbsolver,feop,fesnaps)
    x̂,rbstats = solve(rbsolver,rbop,μon)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

    println(perf)
  end

end

main(:pod)
main(:ttsvd)

end
