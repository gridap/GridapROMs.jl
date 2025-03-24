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
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)

cutgeoout = cut(model,!geo2)
aggregatesout = aggregate(strategy,cutgeoout)
Voutact = FESpace(Ωactout,reffe,conformity=:H1)
Voutagg = AgFEMSpace(Voutact,aggregatesout)

g(x) = x[2]-x[1]

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(v) = ∫(∇(v)⋅∇(g))dΩout

V = FESpace(model,reffe,conformity=:H1)
Vfun = FunctionExtensionFESpace(V,Vagg,Voutagg,g)
Vharm = HarmonicExtensionFESpace(V,Vagg,Voutagg,aout,lout)

gh = interpolate_everywhere(g,V)
gh_fun = Extensions.extended_interpolate_everywhere(g,Vfun)
@assert gh_fun.free_values ≈ gh.free_values

gh_harm = Extensions.extended_interpolate_everywhere(g,Vharm)
writevtk(Ωbg,datadir("plts/sol_harm"),cellfields=["uh"=>gh_harm])

Vext = Vharm
assem = SparseMatrixAssembler(Vext,Vext)

ν(x) = x[2]-x[1]
a(u,v) = ∫(ν*∇(v)⋅∇(u))dΩ + ∫( 100*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(v) = ∫(ν*v)dΩ + ∫( 100*v*g - (n_Γ⋅∇(v))*g )dΓ
@assert assemble_matrix(a,assem,Vext,Vext) ≈ assemble_matrix(a,assem,Vagg,Vagg)
@assert assemble_vector(l,assem,Vext) ≈ assemble_vector(l,assem,Vagg)

ext_assem = ExtensionAssembler(Vext,Vext)

A = assemble_matrix(a,ext_assem,Vext,Vext)
in_A = assemble_matrix(a,ext_assem.assem,Vagg,Vagg)
out_A = assemble_matrix(aout,ext_assem.assem,Voutagg,Voutagg)
norm(A)^2 ≈ norm(in_A)^2 + norm(out_A)^2

b = assemble_vector(l,ext_assem,Vext)
in_b = assemble_vector(l,ext_assem.assem,Vagg)
out_b = assemble_vector(lout,ext_assem.assem,Voutagg)
norm(b)^2 ≈ norm(in_b)^2 + norm(out_b)^2

solver = LUSolver()

in_u = zero_free_values(Vext)
in_A = assemble_matrix(a,Vext,Vext)
in_b = assemble_vector(l,Vext)
solve!(in_u,solver,in_A,in_b)
uh = ExtendedFEFunction(Vext,in_u)
u = extend_free_values(Vext,in_u)

@assert u[Vext.fdof_to_bg_fdofs] ≈ in_u
@assert u[Vext.extension.fdof_to_bg_fdofs] ≈ Vext.extension.values.free_values

u_alg = similar(u)
solve!(u_alg,solver,A,b)

@assert u_alg ≈ u

# order 2

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωout,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)

cutgeoout = cut(model,!geo2)
aggregatesout = aggregate(strategy,cutgeoout)
Voutact = FESpace(Ωactout,reffe,conformity=:H1)
Voutagg = Voutact
