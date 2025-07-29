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
sketch=:sprn
compression=:local

domain = (0,2,0,2)
n = 40
partition = (n,n)
bgmodel = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2

pspace = ParamSpace((-0.1,0.1))

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

  f(μ) = x -> 1.0
  fμ(μ) = parameterize(f,μ)
  g(μ) = x -> x[1]-x[2]
  gμ(μ) = parameterize(g,μ)

  a(μ,u,v,dΩφ,dΓφ) = ∫(∇(v)⋅∇(u))dΩφ + ∫( (γd/hd)*v*u  - v*(n_Γφ⋅∇(u)) - (n_Γφ⋅∇(v))*u )dΓφ
  l(μ,v,dΩφ) = ∫(fμ(μ)⋅v)dΩφ
  res(μ,u,v,dΩφ) = ∫(∇(v)⋅∇(u))dΩφ - l(μ,v,dΩφ)

  domains = FEDomains((Ωφ,),(Ωφ,Γφ))

  testact = FESpace(Ωactφ,reffe,conformity=:H1,dirichlet_tags=[1,3,7])
  test = AgFEMSpace(testact,aggregates)
  trial = ParamTrialFESpace(test,gμ)

  LinearParamOperator(res,a,pspace,trial,test,domains)
end

μ = realization(pspace;nparams)
feop = def_fe_operator(μ)

μon = realization(pspace;nparams,sampling=:uniform)
feopon = def_fe_operator(μon)

fesolver = LUSolver()

function rb_solver(tol,nparams)
  tol = method == :ttsvd ? fill(tol,3) : tol
  tolhr = tol.*1e-2
  state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids=1)
  residual_reduction = LocalHyperReduction(tolhr;nparams,ncentroids=8)
  jacobian_reduction = LocalHyperReduction(tolhr;nparams,ncentroids=8)
  # residual_reduction = HyperReduction(tolhr;nparams)
  # jacobian_reduction = HyperReduction(tolhr;nparams)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

rbsolver = rb_solver(tol,nparams)

fesnaps, = solution_snapshots(rbsolver,feop,μ)
rbop′ = reduced_operator(rbsolver,feop,fesnaps)
rbop = change_operator(rbop′,feopon)

x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feopon,μon)
perf = eval_performance(rbsolver,feopon,rbop,x,x̂,festats,rbstats)

# μend = Realization([[-0.0875]])
# feopend = def_fe_operator(μend)
# φendh = get_deformation_map(μend)
# Ωφ = mapped_grid(Ω,φendh)
# uh = FEFunction(param_getindex(get_trial(feopend)(μend),1),fesnaps[:,8])
# writevtk(Ωφ,datadir("plts/uh"),cellfields=["uh"=>uh])
# writevtk(Ωφ,datadir("plts/Ωφ"))

# jx = jacobian_snapshots(rbsolver,feopon,x)
# rx = residual_snapshots(rbsolver,feopon,x)

# rbtest = get_test(rbop)
# rbtrial = get_trial(rbop)

# jxdata = recast(jx[1])
# rxdata = get_param_data(rx[1])

# ĵ1 = galerkin_projection(get_basis(rbtrial),recast(jx[1]),get_basis(rbtest))
# ĵ2 = galerkin_projection(get_basis(rbtrial),recast(jx[2]),get_basis(rbtest))
# ĵ = ĵ1+ĵ2

# r̂1 = galerkin_projection(get_basis(rbtest),rx[1])
# r̂2 = galerkin_projection(get_basis(rbtest),rx[2])
# r̂ = r̂1+r̂2
# # r̂ - syscache.b.hypred.data

# x̂ = zero_free_values(rbtrial(μon))
# nlop = parameterize(rbop,μon)
# syscache = allocate_systemcache(nlop,x̂)

# jacobian!(syscache.A,nlop,x̂)
# residual!(syscache.b,nlop,x̂)

# norm(ĵ - permutedims(syscache.A.hypred.data,(1,3,2)))
# norm(r̂ - syscache.b.hypred.data)

# jx1 = recast(jx[1])[1,1]
# interp = rbop.lhs[1].interpolation
# domain = interp.domain
# rows,cols = domain.metadata
# Jx1_red = [jx1[i,j] for (i,j) in zip(rows,cols)]
# syscache.A.fecache[1][1] - Jx1_red

# jx2 = recast(jx[2])[1,1]
# interp = rbop.lhs[2].interpolation
# domain = interp.domain
# rows,cols = domain.metadata
# Jx2_red = [jx2[i,j] for (i,j) in zip(rows,cols)]
# syscache.A.fecache[2][1] - Jx2_red

# i = rbop.lhs[1].interpolation.interpolation
# θ1 = zeros(4)
# ldiv!(θ1,i,Jx1_red)
# norm(θ1 - syscache.A.coeff[1][1])
# norm(sum([rbop.lhs[1].basis.basis[:,i,:]*θ1[i] for i = 1:4]) - ĵ1[:,1,:])

# rx1 = get_param_data(rx[1])[1]
# interp = rbop.rhs[1].interpolation
# domain = interp.domain
# rows = domain.metadata
# rx1_red = rx1[rows]
# syscache.b.fecache[1][1] - rx1_red

# basis = projection(get_reduction(rbsolver.residual_reduction.reduction),ress[1])
# rx[1] - basis.basis*basis.basis'*rx[1]

# i = rbop.rhs[1].interpolation.interpolation
# θ1 = zeros(3)
# ldiv!(θ1,i,rx1_red)
# norm(θ1 - syscache.b.coeff[1][1])
# norm(rbop.rhs[1].basis.basis*θ1 - r̂1[:,1])

# rx2 = get_param_data(rx[2])[1]
# interp = rbop.rhs[2].interpolation
# domain = interp.domain
# rows = domain.metadata
# rx2_red = rx2[rows]
# i = rbop.rhs[2].interpolation.interpolation
# θ2 = zeros(4)
# ldiv!(θ2,i,rx2_red)
# norm(θ2 - syscache.b.coeff[2][1])
# norm(rbop.rhs[2].basis.basis*θ2 - r̂2[:,1])

# basis = projection(get_reduction(rbsolver.residual_reduction.reduction),ress[1])
# rx[1] - basis.basis*basis.basis'*rx[1]
