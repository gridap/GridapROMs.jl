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

ν(μ) = x->μ[3]
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

h(μ) = x->1
hμ(μ) = ParamFunction(h,μ)

g(μ) = x->μ[3]*x[1]-x[2]
gμ(μ) = ParamFunction(g,μ)

const γd = 10.0
const hd = dp[1]/n

a(μ,u,v,dΩ,dΓ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(μ,v,dΩ,dΓ,dΓn) = ∫(fμ(μ)⋅v)dΩ + ∫(hμ(μ)⋅v)dΓn + ∫( (γd/hd)*v*gμ(μ) - (n_Γ⋅∇(v))*gμ(μ) )dΓ
res(μ,u,v,dΩ,dΓ,dΓn) =  a(μ,u,v,dΩ,dΓ) - l(μ,v,dΩ,dΓ,dΓn)

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(μ,v) = ∫(∇(v)⋅∇(gμ(μ)))dΩout

V = FESpace(model,reffe,conformity=:H1)
Vext = ParamHarmonicExtensionFESpace(V,Vagg,Voutagg,aout,lout)
Uext = ParamTrialFESpace(Vext,gμ)

μ = realization(pspace;nparams=50)
Uμ = Uext(μ)

u = zero_free_values(Uμ)
uh = FEFunction(Uμ,u)

afun(u,v) = a(μ,u,v,dΩ,dΓ)
lfun(v) = res(μ,uh,v,dΩ,dΓ,dΓn)

solver = LUSolver()

A = assemble_matrix(afun,Uμ,Vext)
b = assemble_vector(lfun,Uμ)
solve!(u,solver,A,b)
uh = ExtendedFEFunction(Uμ,u)

uext = extend_free_values(Uμ,u)

norm(uext[1])^2 ≈ norm(u[1])^2 + norm(Uμ.space.extension.values.free_values[1])^2
uext[1][Vext.dof_to_bg_dofs] ≈ u[1]
uext[1][Vext.extension.dof_to_bg_dofs] ≈ Uμ.space.extension.values.free_values[1]

dof_map = get_dof_map(V)
fesnaps = Snapshots(uext,dof_map,μ)

energy(u,v) = ∫(∇(v)⋅∇(u))dΩbg
X = assemble_matrix(energy,V,V)

state_reduction = PODReduction(1e-4,energy;nparams=50)
basis = reduced_basis(state_reduction,fesnaps,X)

jacs =
