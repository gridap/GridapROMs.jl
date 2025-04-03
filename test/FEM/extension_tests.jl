using Gridap
using GridapEmbedded
using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.TProduct
using DrWatson
using Test

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
Ωactout = Triangulation(cutgeo,ACTIVE_OUT)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)

order = 1
degree = 2*order

dΩ = Measure(Ω,degree)
dΩout = Measure(Ωactout,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

f(x) = x[1]
h(x) = x[2]
g(x) = x[2]-x[1]

const γd = 10.0
const hd = dp[1]/n

a(u,v) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(v) =  ∫(f⋅v)dΩ + ∫(h⋅v)dΓn + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ

reffe = ReferenceFE(lagrangian,Float64,2)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)
Uagg = TrialFESpace(Vagg,g)

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(v) =  ∫(∇(v)⋅∇(g))dΩout

Vout = FESpace(Ωactout,reffe,conformity=:H1)
Uout = TrialFESpace(Vout,g)

V = FESpace(model,reffe,conformity=:H1)

agg_dof_to_bgdof = get_dof_to_bg_dof(V,Vagg)
act_out_dof_to_bgdof = get_dof_to_bg_dof(V,Vout)
agg_out_dof_to_bgdof = setdiff(act_out_dof_to_bgdof,agg_dof_to_bgdof)

op = AffineFEOperator(a,l,Uagg,Vagg)

# extend by g
gV = interpolate_everywhere(g,V)
gV_out = view(gV.free_values,agg_out_dof_to_bgdof)
ext = FunctionalExtension(gV_out,agg_out_dof_to_bgdof)
solver = ExtensionSolver(LUSolver(),ext)
u = solve(solver,op.op)
uh = FEFunction(V,u)
writevtk(Ωbg,datadir("sol_g.vtu"),cellfields=["uh"=>uh])

# extend by 0
ext = ZeroExtension(agg_out_dof_to_bgdof)
solver = ExtensionSolver(LUSolver(),ext)
u = solve(solver,op.op)
uh = FEFunction(V,u)
writevtk(Ωbg,datadir("sol_0.vtu"),cellfields=["uh"=>uh])

# harmonic extension
A = assemble_matrix(aout,Uout,Vout)
b = assemble_vector(lout,Vout)
agg_out_dof_to_act_out_dof = get_bg_dof_to_dof(V,Vout,agg_out_dof_to_bgdof)
ext = HarmonicExtension(A,b,agg_out_dof_to_bgdof,agg_out_dof_to_act_out_dof)
solver = ExtensionSolver(LUSolver(),ext)
u = solve(solver,op.op)
uh = FEFunction(V,u)
writevtk(Ωbg,datadir("sol_harm.vtu"),cellfields=["uh"=>uh])
