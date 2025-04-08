module PoissonEmbedded

using Gridap
using GridapEmbedded
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod,n=20;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  R = 0.5
  L = 0.8*(2*R)
  p1 = Point(0.0,0.0)
  p2 = p1 + VectorValue(L,0.0)

  geo1 = disk(R,x0=p1)
  geo2 = disk(R,x0=p2)
  geo = setdiff(geo1,geo2)

  t = 1.01
  pmin = p1-t*R
  pmax = p1+t*R
  dp = pmax - pmin

  partition = (n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(pmin,pmax,partition)
  else
    bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
  end

  order = 1
  degree = 2*order

  cutgeo = cut(bgmodel,geo)
  Ωbg = Triangulation(bgmodel)
  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩbg = Measure(Ωbg,degree)
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  nΓ = get_normal_vector(Γ)

  ν(μ) = x->μ[3]
  νμ(μ) = ParamFunction(ν,μ)

  f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
  fμ(μ) = ParamFunction(f,μ)

  g(μ) = x->0
  gμ(μ) = ParamFunction(g,μ)

  # non-symmetric formulation

  a(μ,u,v,dΩ,dΓ) = ∫( νμ(μ)*∇(v)⋅∇(u) )dΩ - ∫( νμ(μ)*v*(nΓ⋅∇(u)) - νμ(μ)*(nΓ⋅∇(v))*u )dΓ
  b(μ,u,v,dΩ,dΓ) = a(μ,u,v,dΩ,dΓ) - ∫( v*fμ(μ) )dΩ - ∫( νμ(μ)*(nΓ⋅∇(v))*gμ(μ) )dΓ

  trian_a = (Ω,Γ)
  trian_b = (Ω,Γ)
  domains = FEDomains(trian_b,trian_a)

  reffe = ReferenceFE(lagrangian,Float64,order)

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    test = TestFESpace(Ωact,reffe;conformity=:H1)
    energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ
    state_reduction = PODReduction(tolrank,energy;nparams,sketch)
  else method == :ttsvd
    test = TProductFESpace(Ωact,Ωbg,reffe;conformity=:H1)
    tolranks = fill(tolrank,3)
    ttenergy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
    state_reduction = Reduction(tolranks,ttenergy;nparams)
  end

  trial = ParamTrialFESpace(test)
  feop = LinearParamOperator(b,a,pspace,trial,test,domains)

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
