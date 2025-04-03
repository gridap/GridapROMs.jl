module StokesEmbedded

using Gridap
using Gridap.MultiField
using GridapEmbedded
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  n=20,method=:pod;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,unsafe=false
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  R = 0.3
  pmin = Point(0,0)
  pmax = Point(1,1)
  n = 20
  partition = (n,n)

  geo = !disk(R,x0=Point(0.5,0.5))

  domain = (0,1,0,1)
  partition = (n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(domain,partition)
  else
    bgmodel = CartesianDiscreteModel(domain,partition)
  end
  labels = get_face_labeling(bgmodel)
  add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])

  order = 2
  degree = 2*order

  cutgeo = cut(bgmodel,geo)

  Ωbg = Triangulation(bgmodel)
  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL_IN)
  Γ = EmbeddedBoundary(cutgeo)

  dΩbg = Measure(Ωbg,degree)
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  nΓ = get_normal_vector(Γ)

  ν(μ) = x -> μ[1]
  νμ(μ) = ParamFunction(ν,μ)

  g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
  gμ(μ) = ParamFunction(g,μ)

  f(μ) = x -> VectorValue(0.0,0.0)
  fμ(μ) = ParamFunction(f,μ)

  g_0(μ) = x -> VectorValue(0.0,0.0)
  gμ_0(μ) = ParamFunction(g_0,μ)

  a(μ,(u,p),(v,q),dΩ,dΓ) = (
    ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ +
    ∫( - v⋅(nΓ⋅∇(u))*νμ(μ) + (nΓ⋅∇(v))⋅u*νμ(μ) + (p*nΓ)⋅v + (q*nΓ)⋅u )dΓ
  )

  l(μ,(u,p),(v,q),dΩ) = ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ

  trian_res = (Ω,)
  trian_jac = (Ω,Γ)
  domains = FEDomains(trian_res,trian_jac)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    test_u = TestFESpace(Ωact,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
    test_p = TestFESpace(Ωact,reffe_p;conformity=:H1)
    energy((du,dp),(v,q)) = ∫(du⋅v)dΩ  + ∫(dp*q)dΩ + ∫(∇(v)⊙∇(du))dΩ
    coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
    state_reduction = SupremizerReduction(coupling,tolrank,energy;nparams,sketch)
  else method == :ttsvd
    test_u = TProductFESpace(Ωact,Ωbg,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
    test_p = TProductFESpace(Ωact,Ωbg,reffe_p;conformity=:H1)
    tolranks = fill(tolrank,3)
    ttenergy((du,dp),(v,q)) = ∫(du⋅v)dΩbg  + ∫(dp*q)dΩbg + ∫(∇(v)⊙∇(du))dΩbg
    ttcoupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩbg + ∫(dp*∂₂(v))dΩbg
    state_reduction = SupremizerReduction(ttcoupling,tolranks,ttenergy;nparams,unsafe)
  end

  trial_u = ParamTrialFESpace(test_u,gμ)
  trial_p = ParamTrialFESpace(test_p)
  test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  feop = LinearParamOperator(l,a,pspace,trial,test,domains)

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

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

main(:pod)
main(:ttsvd)

end
