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
nparams=25#0
nparams_res=25#0
nparams_jac=25#0

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
ncentroids = 5
state_reduction = LocalReduction(tolrank,energy;nparams,ncentroids)

fesolver = ExtensionSolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=nparams,nparams_jac=nparams,interp=true)

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

# μon = realization(pspace;nparams=1,sampling=:uniform)
μon = Realization([[0.684117676992132]])

x̂,rbstats = solve(rbsolver,rbop,μon)

feopon = param_operator(μon) do μ
  println("------------------")
  def_fe_operator(μ)
end
x,festats = solution_snapshots(rbsolver,feopon,μon)
perf = eval_performance(rbsolver,feopon,rbop,x,x̂,festats,rbstats)

rbsnaps = RBSteady.to_snapshots(rbop,x̂,μon)

# # TT

_bgmodel = TProductDiscreteModel(pmin,pmax,partition)
_Ωbg = Triangulation(_bgmodel)
_dΩbg = Measure(_Ωbg,degree)

_testbg = FESpace(_Ωbg,reffe,conformity=:H1)

_energy(du,v) = ∫(v*du)_dΩbg + ∫(∇(v)⋅∇(du))_dΩbg
_tolrank = fill(tolrank,2)
_state_reduction = LocalReduction(_tolrank,_energy;nparams,ncentroids)
_rbsolver = RBSolver(fesolver,_state_reduction;nparams_res=nparams,nparams_jac=nparams,interp=true)

function _def_fe_operator(μ)
  x0 = Point(μ[1],μ[1])
  geo = !disk(R,x0=x0)
  cutgeo = cut(_bgmodel,geo)

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

  test = DirectSumFESpace(_testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test,domains)
end

_feop = param_operator(μ) do μ
  println("------------------")
  _def_fe_operator(μ)
end

_fesnaps, = solution_snapshots(_rbsolver,_feop)
_rbop = reduced_operator(_rbsolver,_feop,_fesnaps)

_x̂, = solve(_rbsolver,_rbop,μon)

_feopon = param_operator(μon) do μ
  println("------------------")
  _def_fe_operator(μ)
end

_x, = solution_snapshots(_rbsolver,_feopon,μon)
_perf = eval_performance(_rbsolver,_feopon,_rbop,_x,_x̂,festats,rbstats)

#

bgmodel = TProductDiscreteModel(Point(0,0),Point(1,1),(2,2))
Ωbg = Triangulation(bgmodel)
Ωact = view(Ωbg,[2,3])
testbg = FESpace(Ωbg,reffe,conformity=:H1)
testact = FESpace(Ωact,reffe,conformity=:H1)
test = DirectSumFESpace(testbg,testact)

space = test.space
compl = test.complementary
