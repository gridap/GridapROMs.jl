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

V = FESpace(Ωbg,reffe,conformity=:H1)

Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(Vact,aggregates)

cutgeoout = cut(model,!geo2)
aggregatesout = aggregate(strategy,cutgeoout)
Voutact = FESpace(Ωactout,reffe,conformity=:H1)
Voutagg = AgFEMSpace(Voutact,aggregatesout)

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

aout(u,v) = ∫(∇(v)⋅∇(u))dΩout
lout(μ,v) = ∫(∇(v)⋅∇(gμ(μ)))dΩout

Vext = ParamHarmonicExtensionFESpace(V,Vagg,Voutagg,aout,lout)
Uext = ParamTrialFESpace(Vext,gμ)

feop = LinearParamOperator(res,a,pspace,Uext,Vext,domains)

solver = LUSolver()

μ = realization(pspace;nparams=50)
u, = solve(solver,feop,μ)
Uμ = Uext(μ)
uext = extend_free_values(Uμ,u)

dof_map = get_dof_map(Vext)
fesnaps = Snapshots(uext,dof_map,μ)

energy(u,v) = ∫(∇(v)⋅∇(u))dΩbg
# X = assemble_matrix(energy,V,V)

state_reduction = PODReduction(1e-4,energy;nparams=50)
rbsolver = RBSolver(solver,state_reduction;nparams_res=50,nparams_jac=50)

# basis = reduced_basis(state_reduction,fesnaps,X)

red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)

dm = get_dof_map(Vext)
sin = get_sparse_dof_map(Vext,Vext,Ω)
sΓ = get_sparse_dof_map(Vext,Vext,Γ)
sΩout = get_sparse_dof_map(Vext,Vext,Ωout)
sΩactout = get_sparse_dof_map(Vext,Vext,Ωactout)
"NOTE: this sparsity on Ωactout won't work long term, since I will be inputing
Ωout in aout"

extop = ExtensionParamOperator(feop)
Aallin = jacobian_snapshots(rbsolver,extop,fesnaps)
assemallout = Extensions.ExtensionAssemblerInsertOut(Uμ,Uμ)
_Aallout = assemble_matrix(assemallout,Uμ.space.extension.matdata)
Aallout = Snapshots(_Aallout,sΩactout,μ)
Ainin = jacobian_snapshots(rbsolver,feop,fesnaps)

ballin = residual_snapshots(rbsolver,extop,fesnaps)
_ballout = assemble_vector(assemallout,Uμ.space.extension.vecdata)
ballout = Snapshots(_ballout,dm,μ)
binin = residual_snapshots(rbsolver,feop,fesnaps)

hr_reduction = rbsolver.residual_reduction
hr_lhs = reduced_jacobian(hr_reduction,red_trial,red_test,Aallin)
hr_rhs = reduced_residual(hr_reduction,red_test,ballin)
hr_lhs_out,t_lhs_out = RBSteady.reduced_form(hr_reduction,Aallout,Ωout,red_trial,red_test)
hr_rhs_out,t_rhs_out = RBSteady.reduced_form(hr_reduction,ballout,Ωout,red_test)

trians_rhs = get_domains(hr_rhs)
trians_lhs = get_domains(hr_lhs)
feop′ = change_domains(feop,trians_rhs,trians_lhs)
rbop = GenericRBOperator(feop′,red_trial,red_test,hr_lhs,hr_rhs)

r = realization(pspace;nparams=10,sampling=:uniform)
x = Extensions.zero_bg_free_values(Uext(r))
xrb = project(get_trial(rbop)(r),x)
x̂ = RBParamVector(xrb,x)

nlop = parameterize(rbop,r)
syscache = allocate_systemcache(nlop,x̂)

hr_lhs_out_cache = allocate_hypred_cache(hr_lhs_out,r)

A,b = syscache.A,syscache.b
fill!(x̂,zero(eltype(x̂)))
residual!(b,nlop,x̂)
jacobian!(A,nlop,x̂)
ns = solve!(x̂,solver,A,b)

rbsnaps = RBSteady.to_snapshots(rbop.trial,x̂,r)

uon, = solve(solver,feop,r)
uexton = extend_free_values(Uext(r),uon)
fesnapson = Snapshots(uexton,dof_map,r)

fesnapson - rbsnaps
