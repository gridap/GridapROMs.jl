using DrWatson
using LinearAlgebra
using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Algebra
using Gridap.FESpaces
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.ParamAlgebra
using GridapROMs.Utils

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:pod
n=20
tol=1e-4
rank=nothing
nparams=200
nparams_res=100
nparams_jac=100

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

pdomain = (0.5,1.5)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(2,2)
dp = pmax - pmin

partition = (n,n)
if method==:ttsvd
  bgmodel = TProductDiscreteModel(pmin,pmax,partition)
else
  bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
end

f(x) = 1.0
g(x) = x[1]-x[2]

order = 2
degree = 2*order

Ωbg = Triangulation(bgmodel)
dΩbg = Measure(Ωbg,degree)

reffe = ReferenceFE(lagrangian,Float64,order)
testbg = FESpace(Ωbg,reffe,conformity=:H1)

energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
tolrank = tol_or_rank(tol,rank)
tolrank = method == :ttsvd ? fill(tolrank,2) : tolrank
ncentroids = 16
state_reduction = LocalReduction(tolrank,energy;nparams,ncentroids)

ncentroids_res = ncentroids_jac = 4
fesolver = ExtensionSolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,ncentroids_res,ncentroids_jac,interp=true)

const γd = 10.0
const hd = dp[1]/n

function def_fe_operator(μ)
  x0 = Point(μ[1],μ[1])
  geo = !disk(R,x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)

  a(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(v,dΩ,dΓ) = ∫(f⋅v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
  res(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ - l(v,dΩ,dΓ)

  domains = FEDomains((Ω,Γ),(Ω,Γ))

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test,domains)
end

μ = realization(pspace;nparams)

feop = param_operator(μ) do μ
  println("------------------")
  def_fe_operator(μ)
end

fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(pspace;nparams=1,sampling=:uniform)

x̂,rbstats = solve(rbsolver,rbop,μon)

feopon = param_operator(μon) do μ
  println("------------------")
  def_fe_operator(μ)
end
x,festats = solution_snapshots(rbsolver,feopon,μon)
perf = eval_performance(rbsolver,feopon,rbop,x,x̂,festats,rbstats)

# rbsnaps = RBSteady.to_snapshots(rbop,x̂,μon)
