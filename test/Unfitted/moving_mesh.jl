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

domain = (-1,1,-1,1)
n=40
partition = (n,n)
bgmodel = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2

pspace = ParamSpace((-0.5,0.5,0.1,0.3))

const γd = 10.0
const hd = 2/n

# quantities on the base configuration

μ0 = (0.0,0.2)
x0 = Point(μ0[1],μ0[1])
geo = !disk(μ0[2],x0=x0)
cutgeo = cut(bgmodel,geo)

Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)

dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

n_Γ = get_normal_vector(Γ)
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

function get_deformation_map(μ)
  φ(μ) = x -> VectorValue((μ[1]-μ0[1]) + (μ[2]/μ0[2])*x[1], (μ[1]-μ0[1]) + (μ[2]/μ0[2])*x[2])
  φμ(μ) = parameterize(φ,μ)

  λ = 0.33/((1+0.33)*(1-2*0.33))
  p = 1/(2(1+0.33))
  σ(ε) = λ*tr(ε)*one(ε) + 2*p*ε

  a(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ + ∫( (γd/hd)*v⋅u - v⋅(n_Γ⋅(σ∘ε(u))) - (n_Γ⋅(σ∘ε(v)))⊙u )dΓ
  l(μ,v) = ∫( (γd/hd)*v⋅φμ(μ) - (n_Γ⋅(σ∘ε(v)))⋅φμ(μ) )dΓ
  res(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ - l(μ,v)

  reffeφ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  Vφact = FESpace(Ωact,reffeφ,conformity=:H1,dirichlet_tags="boundary")
  Vφ = AgFEMSpace(Vφact,aggregates)
  Uφ = ParamTrialFESpace(Vφ)

  feop = LinearParamOperator(res,a,pspace,Uφ,Vφ)
  φ, = solve(LUSolver(),feop,μ)
  FEFunction(Uφ(μ),φ)
end

reffe = ReferenceFE(lagrangian,Float64,order)

Ω = Triangulation(bgmodel)
dΩ = Measure(Ω,degree)
energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

function def_fe_operator(μ)
  φ = get_deformation_map(μ)
  Ωactφ = MappedGrid(Ωact,φ)

  g(μ) = x -> x[1]-x[2]
  gμ(μ) = parameterize(g,μ)

  a(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(v,dΩ,dΓ) = ∫(f⋅v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
  res(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ - l(v,dΩ,dΓ)

  domains = FEDomains((Ω,Γ),(Ω,Γ))

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

#

μ = realization(pspace)
φ = get_deformation_map(μ)
Ωactφ = MappedGrid(Ωact,φ)
testact = FESpace(Ωactφ,reffe,conformity=:H1,dirichlet_tags="boundary")

f = testact
trian_a = get_triangulation(f)
bgcell_to_bgcellin = aggregates
shfns_g = get_fe_basis(testact)
dofs_g = get_fe_dof_basis(testact)
bgcell_to_gcell=1:length(bgcell_to_bgcellin)
D = num_cell_dims(trian_a)
glue = get_glue(trian_a,Val(D))
acell_to_bgcell = glue.tface_to_mface
bgcell_to_acell = glue.mface_to_tface
acell_to_bgcellin = collect(lazy_map(Reindex(bgcell_to_bgcellin),acell_to_bgcell))
acell_to_acellin = collect(lazy_map(Reindex(bgcell_to_acell),acell_to_bgcellin))
acell_to_gcell = lazy_map(Reindex(bgcell_to_gcell),acell_to_bgcell)

# Build shape funs of g by replacing local funs in cut cells by the ones at the root
# This needs to be done with shape functions in the physical domain
# otherwise shape funs in cut and root cells are the same
acell_phys_shapefuns_g = get_array(change_domain(shfns_g,PhysicalDomain()))
acell_phys_root_shapefuns_g = lazy_map(Reindex(acell_phys_shapefuns_g),acell_to_acellin)
root_shfns_g = GenericCellField(acell_phys_root_shapefuns_g,trian_a,PhysicalDomain())

# Compute data needed to compute the constraints
dofs_f = get_fe_dof_basis(f)
shfns_f = get_fe_basis(f)
acell_to_coeffs = dofs_f(root_shfns_g)
acell_to_proj = dofs_g(shfns_f)
acell_to_dof_ids = get_cell_dof_ids(f)

# c = return_cache(dofs_f,root_shfns_g)
# evaluate!(c,dofs_f,root_shfns_g)
for i in 1:length(get_data(dofs_f))
  println(i)
  a = get_data(dofs_f)[i]
  b = get_data(root_shfns_g)[i]
  c = return_cache(a,b)
  evaluate!(c,a,b)
end

i = 13
a = get_data(dofs_f)[i]
b = get_data(root_shfns_g)[i]
c = return_cache(a,b)
evaluate!(c,a,b)
