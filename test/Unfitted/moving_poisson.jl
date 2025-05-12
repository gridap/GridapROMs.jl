using DrWatson
using LinearAlgebra
using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Algebra
using GridapROMs.RBSteady
using GridapROMs.Extensions

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:pod
n=40
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

pdomain = (0.4,0.6,0.4,0.6,1.,5.)
pspace = ParamSpace(pdomain)

R = 0.2
pmin = Point(0,0)
pmax = Point(1,1)
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

X = assemble_matrix(energy,testbg,testbg)

fesolver = ExtensionSolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=nparams,nparams_jac=nparams)

const γd = 10.0
const hd = dp[1]/n

function def_fe_operator(μ)
  x0 = Point(μ[1],μ[2])
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

  trian_a = (Ω,Γ)
  trian_res = (Ω,Γ)
  domains = FEDomains(trian_res,trian_a)

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test,domains)
end

μ = realization(pspace;nparams=50)

feop = param_operator(μ) do μ
  println("------------------")
  def_fe_operator(μ)
end

using Gridap.FESpaces
using Gridap.Algebra
using GridapROMs.ParamAlgebra
using GridapROMs.Utils
using GridapROMs.RBSteady

fesnaps, = solution_snapshots(rbsolver,feop)
red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
red_lhs,red_rhs = RBSteady.reduced_weak_form(rbsolver,feop,red_trial,red_test,fesnaps)
rbop = RBOperator(feop,red_trial,red_test,red_lhs,red_rhs)

μ = realization(pspace;sampling=:uniform)
opμ = RBSteady.get_local(rbop,μ.params[1])

trial = get_trial(opμ)
x̂ = zero_free_values(trial)
A = allocate_jacobian(opμ,x̂)
b = allocate_residual(opμ,x̂)
solve!(x̂,fesolver.solver,A,b)
ff = get_fe_operator(opμ)
syscache = allocate_systemcache(opμ[1],x̂)
t = @timed solve!(x̂,fesolver,opμ,syscache)
stats = CostTracker(t,nruns=num_params(r),name="RB")
