using DrWatson
using LinearAlgebra
using Serialization

using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Geometry
using Gridap.Arrays
using Gridap.Algebra
using Gridap.CellData
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.ODEs
using GridapROMs.DofMaps
using GridapROMs.Uncommon
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.Utils

method=:pod
tol=1e-4
rank=nothing
nparams=100
nparams_res=nparams_jac=nparams

domain = (0,2,0,2)
n = 40
partition = (n,n)
bgmodel = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2

pspace = ParamSpace((-0.5,0.5))

const γd = 10.0
const hd = 2/n

# quantities on the base configuration

μ0 = (1.0,)
x0 = Point(μ0[1],μ0[1])
geo = !disk(0.3,x0=x0)
cutgeo = cut(bgmodel,geo)

Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)

dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

n_Γ = get_normal_vector(Γ)
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

const E = 1
const ν = 0.33
const λ = E*ν/((1+ν)*(1-2*ν))
const p = E/(2(1+ν))

function get_deformation_map(μ)
  φ(μ) = x -> VectorValue(μ[1],μ[1])
  φμ(μ) = parameterize(φ,μ)

  σ(ε) = λ*tr(ε)*one(ε) + 2*p*ε

  dΩ = Measure(Ω,2*degree)
  dΓ = Measure(Γ,2*degree)

  a(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ + ∫( (γd/hd)*v⋅u - v⋅(n_Γ⋅(σ∘ε(u))) - (n_Γ⋅(σ∘ε(v)))⊙u )dΓ
  l(μ,v) = ∫( (γd/hd)*v⋅φμ(μ) - (n_Γ⋅(σ∘ε(v)))⋅φμ(μ) )dΓ
  res(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ - l(μ,v)

  reffeφ = ReferenceFE(lagrangian,VectorValue{2,Float64},2*order)
  Vφact = FESpace(Ωact,reffeφ,conformity=:H1,dirichlet_tags="boundary")
  Vφ = AgFEMSpace(Vφact,aggregates)
  Uφ = ParamTrialFESpace(Vφ)

  feop = LinearParamOperator(res,a,pspace,Uφ,Vφ)
  φ, = solve(LUSolver(),feop,μ)
  FEFunction(Uφ(μ),φ)
end

reffe = ReferenceFE(lagrangian,Float64,order)

function def_fe_operator(μ)
  φh = get_deformation_map(μ)
  Ωactφ = mapped_grid(Ωact,φh)
  Ωφ = mapped_grid(Ω,φh)
  Γφ = mapped_grid(Γ,φh)

  dΩφ = Measure(Ωφ,degree)
  dΓφ = Measure(Γφ,degree)

  n_Γφ = get_normal_vector(Γφ)

  g(μ) = x -> x[1]-x[2]
  gμ(μ) = parameterize(g,μ)

  a(u,v,dΩφ,dΓφ) = ∫(∇(v)⋅∇(u))dΩφ + ∫( (γd/hd)*v*u  - v*(n_Γφ⋅∇(u)) - (n_Γφ⋅∇(v))*u )dΓφ
  l(v,dΩφ,dΓφ) = ∫(f⋅v)dΩφ + ∫( (γd/hd)*v*g - (n_Γφ⋅∇(v))*g )dΓφ
  res(u,v,dΩφ,dΓφ) = ∫(∇(v)⋅∇(u))dΩφ - l(v,dΩφ,dΓφ)

  domains = FEDomains((Ωφ,Γφ),(Ωφ,Γφ))

  testact = FESpace(Ωactφ,reffe,conformity=:H1,dirichlet_tags="boundary")
  test = AgFEMSpace(testact,aggregates)
  trial = ParamTrialFESpace(test,gμ)

  LinearParamOperator(res,a,pspace,trial,test,domains)
end

μ = realization(pspace;nparams)
feop = def_fe_operator(μ)

μon = realization(pspace;nparams,sampling=:uniform)
feopon = def_fe_operator(μon)

fesolver = LUSolver()
state_reduction = Reduction(tol;nparams,sketch=:sprn) #energy
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

fesnaps, = solution_snapshots(rbsolver,feop,μ)
rbop′ = reduced_operator(rbsolver,feop,fesnaps)
rbop = RBSteady.change_operator(rbop′,feopon)

x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feopon,μon)
perf = eval_performance(rbsolver,feopon,rbop,x,x̂,festats,rbstats)

μ = realization(pspace)
φ = get_deformation_map(μ)
Ωactφ = MappedGrid(Ωact,φ)
testact = FESpace(Ωactφ,reffe,conformity=:H1,dirichlet_tags="boundary")
test = AgFEMSpace(testact,aggregates)
trial = ParamTrialFESpace(test,gμ)

μ = realization(pspace;nparams=10)
φh = get_deformation_map(μ)
Ωactφ = mapped_grid(Ωact,φh)
Ωφ = mapped_grid(Ω,φh)
Γφ = mapped_grid(Γ,φh)
