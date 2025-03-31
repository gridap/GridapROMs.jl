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
n = 10
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

aout(u,v) = ∫(∇(v)⋅∇(u))dΩinact
lout(v) = ∫(∇(v)⋅∇(uh_ag))dΩinact

# complementary = EmbeddedFESpace(Vout,V)
complementary = Extensions.get_complementary(space,Vout)


# cell_dof_ids = get_cell_dof_ids(complementary)
# fv = rand(num_free_dofs(complementary))
# dv = rand(num_dirichlet_dofs(complementary))
# k = Broadcasting(PosZeroNegReindex(fv,dv))
# dofs1 = cell_dof_ids[1]
# c = return_cache(k,dofs1)
# evaluate!(c,k,dofs1)
# # lazy_map(Broadcasting(PosZeroNegReindex(fv,dv)),cell_dof_ids)
# kk = Broadcasting(PosNegReindex(fv,dv))
# cc = return_cache(kk,dofs1)
# evaluate!(cc,kk,dofs1)
