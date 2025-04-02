using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using GridapEmbedded
using ROManifolds
using ROManifolds.ParamAlgebra
using ROManifolds.ParamDataStructures
using ROManifolds.Extensions
using ROManifolds.DofMaps
using ROManifolds.RBSteady
using ROManifolds.Utils
using SparseArrays
using DrWatson
using Test
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 10
partition = (n,n)

dp = pmax - pmin

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

model = CartesianDiscreteModel(pmin,pmax,partition)
cutgeo = cut(model,geo2)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ωbg = Triangulation(model)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)
Ωinact = Ωout.b
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)

order = 1
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩinact = Measure(Ωinact,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
dΓ = Measure(Γ,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

V = FESpace(Ωbg,reffe,conformity=:H1)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)
Vout = FESpace(Ωinact,reffe,conformity=:H1)

const γd = 10.0
const hd = dp[1]/n

ν(μ) = x->μ[3]
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

h(μ) = x->1
hμ(μ) = ParamFunction(h,μ)

g(μ) = x->μ[3]*x[1]-x[2]
gμ(μ) = ParamFunction(g,μ)

a(μ,u,v,dΩ,dΓ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(μ,v,dΩ,dΓ,dΓn) = ∫(fμ(μ)⋅v)dΩ + ∫(hμ(μ)⋅v)dΓn + ∫( (γd/hd)*v*gμ(μ) - (n_Γ⋅∇(v))*gμ(μ) )dΓ
res(μ,u,v,dΩ,dΓ,dΓn) =  a(μ,u,v,dΩ,dΓ) - l(μ,v,dΩ,dΓ,dΓn)

trian_a = (Ω,Γ)
trian_res = (Ω,Γ,Γn)
domains = FEDomains(trian_res,trian_a)

Vext = DirectSumFESpace(V,Vagg,Vout)
Uext = ParamTrialFESpace(Vext,gμ)

feop = ExtensionLinearParamOperator(res,a,pspace,Uext,Vext,domains)

solver = ExtensionSolver(LUSolver())
energy(u,v) = ∫(∇(v)⋅∇(u))dΩbg
state_reduction = PODReduction(1e-4,energy;nparams=50)
rbsolver = RBSolver(solver,state_reduction;nparams_res=50,nparams_jac=20)

μ = realization(feop;nparams=50)
fesnaps, = solution_snapshots(rbsolver,feop,μ)

rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

writevtk(Ωbg,datadir("plts/sol_harm_ok"),cellfields=["uh"=>FEFunction(V,bg_u[1])])
writevtk(Ωbg,datadir("plts/sol_harm"),cellfields=["uh"=>FEFunction(V,new_bg_u[1])])
