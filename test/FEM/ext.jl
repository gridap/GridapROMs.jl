using Gridap
using Gridap.Arrays
using Gridap.FESpaces
using GridapEmbedded
using ROManifolds
using ROManifolds.Extensions
using ROManifolds.DofMaps
using SparseArrays
using DrWatson
using Test
using DrWatson

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 30
partition = (n,n)

dp = pmax - pmin

const γd = 10.0
const hd = dp[1]/n

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

model = CartesianDiscreteModel(pmin,pmax,partition)
cutgeo = cut(model,geo2)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ωbg = Triangulation(model)
Ωact = Triangulation(cutgeo,ACTIVE)
Ωactout = Triangulation(cutgeo,ACTIVE_OUT)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)

order = 1
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωout,degree)
dΩactout = Measure(Ωactout,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

g(x) = x[1]-x[2]

reffe = ReferenceFE(lagrangian,Float64,order)

Vact = FESpace(Ωact,reffe,conformity=:H1)
Vactout = FESpace(Ωactout,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)

V = FESpace(model,reffe,conformity=:H1)

a(u,v) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(v) = ∫(v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ

space = EmbeddedFESpace(Vagg,V)

Ωinact = Ωout.b
dΩinact = Measure(Ωinact,degree)
Vout = FESpace(Ωinact,reffe,conformity=:H1)

Vsum = DirectSumFESpace(V,Vagg,Vout)
Usum = Vsum

solver = ExtensionSolver(LUSolver())
opsum = AffineFEOperator(a,l,Vsum,Usum)
# solve(solver,opsum.op)
u = solve(solver.solver,opsum.op)
f = opsum.trial
fin = Extensions.get_space(f)
fout = Extensions.get_out_space(f)
fbg = Extensions.get_bg_space(f)

_Ωout = get_triangulation(fout)
_dΩout = Measure(_Ωout,degree)

uh_in = FEFunction(fin,u)
uh_in_bg = Extensions.ExtendedFEFunction(f,u)

aout(u,v) = ∫(∇(u)⊙∇(v))_dΩout
lout(v) = (-1)*∫(∇(uh_in_bg)⊙∇(v))_dΩout
oop = AffineFEOperator(aout,lout,fout,fout)
uh_out = solve(oop)

uh_bg = uh_in ⊕ uh_out
result = get_free_dof_values(uh_bg)


op = AffineFEOperator(a,l,Vagg,Vagg)
_uh_ag = solve(op)
uh_ag = FEFunction(Vsum.space,_uh_ag.free_values)

_uh_out = zero(Vsum.complementary)
_uh = uh_ag ⊕ _uh_out

aout(u,v) = ∫(∇(v)⋅∇(u))dΩinact
lout(v) = ∫(∇(v)⋅∇(_uh))dΩinact

op_out = AffineFEOperator(aout,lout,Vsum.complementary,Vsum.complementary)
uh_out = solve(op_out)

uh = uh_ag ⊕ uh_out

plt_dir = datadir("plts")
writevtk(Ωbg,joinpath(plt_dir,"sol"),cellfields=["uh"=>uh])

fv = get_free_dof_values(uh)
dv = get_dirichlet_dof_values(uh)

uhbg = FEFunction(V,fv,dv)
writevtk(Ωbg,joinpath(plt_dir,"sol"),cellfields=["uh"=>uhbg])
writevtk(Ωinact,joinpath(plt_dir,"sol_out"),cellfields=["uh"=>uhbg])
