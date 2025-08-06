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
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.Utils

import Gridap.Geometry: push_normal

method=:ttsvd
tol=1e-4
rank=nothing
nparams=200
nparams_res=nparams_jac=nparams
sketch=:sprn
compression=:local
ncentroids=16

domain = (0,2,0,2)
n = 40
partition = (n,n)
bgmodel = method==:ttsvd ? TProductDiscreteModel(domain,partition) : CartesianDiscreteModel(domain,partition)

order = 1
degree = 2

# # case 1: translation
# pdomain = (0.6,1.4)
# case 2: translation and dilation
pdomain = (0.6,1.4,0.25,0.35)
pspace = ParamSpace(pdomain)

const γd = 10.0
const hd = 2/n

# quantities on the base configuration

μ0 = (1.0,0.3)
x0 = Point(μ0[1],μ0[1])
geo = !disk(μ0[2],x0=x0)
cutgeo = cut(bgmodel,geo)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

energy(du,v) = method==:ttsvd ? ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg : ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ + ∫((γd/hd)*v⋅du)dΓ

n_Γ = get_normal_vector(Γ)
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

domains = FEDomains((Ω,),(Ω,Γ))

f(μ) = x -> 1.0
fμ(μ) = parameterize(f,μ)
g(μ) = x -> 0.0
gμ(μ) = parameterize(g,μ)

reffe = ReferenceFE(lagrangian,Float64,order)
testact = FESpace(Ωact,reffe,conformity=:H1,dirichlet_tags=[1,3,7])
test = AgFEMSpace(testact,aggregates)
trial = ParamTrialFESpace(test,gμ)

function get_deformation_map(μ)
  # # case 1: translation
  # φ(μ) = x -> VectorValue(μ[1]-μ0[1],μ[1]-μ0[1])
  # case 2: translation and dilation
  φ(μ) = x -> VectorValue(μ[1]-x[1] + (μ[2]/μ0[2])*(x[1]-μ0[1]),μ[1]-x[2] + (μ[2]/μ0[2])*(x[2]-μ0[1]))
  φμ(μ) = parameterize(φ,μ)

  E = 1
  ν = 0.33
  λ = E*ν/((1+ν)*(1-2*ν))
  p = E/(2(1+ν))
  σ(ε) = λ*tr(ε)*one(ε) + 2*p*ε

  dΩ = Measure(Ω,2*degree)
  dΓ = Measure(Γ,2*degree)

  a(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ + ∫( (γd/hd)*v⋅u - v⋅(n_Γ⋅(σ∘ε(u))) - (n_Γ⋅(σ∘ε(v)))⋅u )dΓ
  l(μ,v) = ∫( (γd/hd)*v⋅φμ(μ) - (n_Γ⋅(σ∘ε(v)))⋅φμ(μ) )dΓ
  res(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ - l(μ,v)

  reffeφ = ReferenceFE(lagrangian,VectorValue{2,Float64},2*order)
  Vφact = FESpace(Ωact,reffeφ,conformity=:H1,dirichlet_tags="boundary")
  Vφ = AgFEMSpace(Vφact,aggregates)
  Uφ = ParamTrialFESpace(Vφ)

  feop = LinearParamOperator(res,a,pspace,Uφ,Vφ)
  d, = solve(LUSolver(),feop,μ)
  FEFunction(Uφ(μ),d)
end

function def_fe_operator(μ)
  φh = get_deformation_map(μ)
  Ωactφ = mapped_grid(Ωact,φh)

  ϕ = get_cell_map(Ωactφ)
  ϕh = GenericCellField(ϕ,Ωact,ReferenceDomain())
  dJ = det∘(∇(ϕh))
  dJΓn(j,c,n) = j*√(n⋅inv(c)⋅n)
  C = (j->j⋅j')∘(∇(ϕh))
  invJt = inv∘∇(ϕh)
  ∇_I(a) = dot∘(invJt,∇(a))
  _n_Γ = push_normal∘(invJt,n_Γ)
  dJΓ = dJΓn∘(dJ,C,_n_Γ)
  ∫_Ω(a) = ∫(a*dJ)
  ∫_Γ(a) = ∫(a*dJΓ)

  a(μ,u,v,dΩ,dΓ) = ∫_Ω(∇_I(v)⋅∇_I(u))dΩ + ∫_Γ( (γd/hd)*v*u - v*(_n_Γ⋅∇_I(u)) - (_n_Γ⋅∇_I(v))*u )dΓ
  l(μ,v,dΩ) = ∫_Ω(fμ(μ)⋅v)dΩ
  res(μ,u,v,dΩ) = ∫_Ω(∇_I(v)⋅∇_I(u))dΩ - l(μ,v,dΩ)

  LinearParamOperator(res,a,pspace,trial,test,domains)
end

function def_extended_fe_operator(μ)
  φh = get_deformation_map(μ)
  Ωactφ = mapped_grid(Ωact,φh)

  ϕ = get_cell_map(Ωactφ)
  ϕh = GenericCellField(ϕ,Ωact,ReferenceDomain())
  dJ = det∘(∇(ϕh))
  dJΓn(j,c,n) = j*√(n⋅inv(c)⋅n)
  C = (j->j⋅j')∘(∇(ϕh))
  invJt = inv∘∇(ϕh)
  ∇_I(a) = dot∘(invJt,∇(a))
  _n_Γ = push_normal∘(invJt,n_Γ)
  dJΓ = dJΓn∘(dJ,C,_n_Γ)
  ∫_Ω(a) = ∫(a*dJ)
  ∫_Γ(a) = ∫(a*dJΓ)

  a(μ,u,v,dΩ,dΓ) = ∫_Ω(∇_I(v)⋅∇_I(u))dΩ + ∫_Γ( (γd/hd)*v*u - v*(_n_Γ⋅∇_I(u)) - (_n_Γ⋅∇_I(v))*u )dΓ
  l(μ,v,dΩ) = ∫_Ω(fμ(μ)⋅v)dΩ
  res(μ,u,v,dΩ) = ∫_Ω(∇_I(v)⋅∇_I(u))dΩ - l(μ,v,dΩ)

  testbg = FESpace(Ωbg,reffe,conformity=:H1,dirichlet_tags=[1,3,7])
  testext = DirectSumFESpace(testbg,test)
  trialext = ParamTrialFESpace(testext,gμ)
  ExtensionLinearParamOperator(res,a,pspace,trialext,testext,domains)
end

get_feop = method==:ttsvd ? def_extended_fe_operator : def_fe_operator

function rb_solver(tol,nparams)
  fesolver = LUSolver()
  if method == :ttsvd
    tol = fill(tol,3)
    fesolver = ExtensionSolver(fesolver)
  end
  hr = compression == :global ? HyperReduction : LocalHyperReduction
  tolhr = tol.*1e-2
  state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids=16)
  residual_reduction = hr(tolhr;nparams,ncentroids=16)
  jacobian_reduction = hr(tolhr;nparams,ncentroids=16)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function plot_sol(rbop,x,x̂,μon,dir,i=1)
  φh = get_deformation_map(μon)
  φhi = param_getindex(φh,i)
  Ωφ = mapped_grid(Ω,φhi)
  U = param_getindex(trial(μon),i)

  if method==:ttsvd
    u,û = vec(x[:,:,i]),vec(x̂[:,:,i])
  else
    u,û = x[:,i],x̂[:,i]
  end

  uh = FEFunction(U,u)
  ûh = FEFunction(U,û)
  eh = FEFunction(test,abs.(u-û))

  writevtk(Ωφ,dir*".vtu",cellfields=["uh"=>uh,"ûh"=>ûh,"eh"=>eh])
end

function postprocess(
  dir,
  solver,
  feop,
  rbop,
  fesnaps,
  x̂,
  festats,
  rbstats)

  μon = get_realization(fesnaps)
  rbsnaps = RBSteady.to_snapshots(rbop,x̂,μon)
  plot_sol(rbop,fesnaps,rbsnaps,μon,dir)
  return eval_performance(rbsolver,feopon,fesnaps,rbsnaps,festats,rbstats)
end

function local_solver(dir,rbsolver,rbop,μ,x,festats)
  k, = get_clusters(rbop.test)
  μsplit = cluster(μ,k)
  xsplit = cluster(x,k)
  perfs = ROMPerformance[]
  for (μi,xi) in zip(μsplit,xsplit)
    feopi = def_fe_operator(μi)
    rbopi = change_operator(get_local(rbop,first(μi)),feopi)
    x̂,rbstats = solve(rbsolver,rbopi,μi)
    perf = postprocess(dir,rbsolver,feopi,rbopi,xi,x̂,festats,rbstats)
    push!(perfs,perf)
  end
  return mean(perfs)
end

max_subspace_size(op::RBOperator) = max_subspace_size(get_test(op))
max_subspace_size(r::RBSpace) = max_subspace_size(r.subspace)
max_subspace_size(a::Projection) = num_reduced_dofs(a)
max_subspace_size(a::NormedProjection) = max_subspace_size(a.projection)
max_subspace_size(a::AffineContribution) = maximum(map(max_subspace_size,a.values))

function max_subspace_size(a::LocalProjection)
  maxsize = 0
  for proj in a.projections
    maxsize = max(maxsize,max_subspace_size(proj))
  end
  return maxsize
end

test_dir = joinpath(datadir("moving_poisson"),string(method))
create_dir(test_dir)

μ = realization(pspace;nparams)
feop = get_feop(μ)

μon = realization(pspace;nparams=10,sampling=:uniform)
feopon = get_feop(μon)

rbsolver = rb_solver(tol,nparams)
fesnaps, = solution_snapshots(rbsolver,feop,μ)
jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

x,festats = solution_snapshots(rbsolver,feopon,μon)

perfs = ROMPerformance[]
maxsizes = Int[]

for tol in (1e-1,1e-2,1e-3,1e-4)
  rbsolver = rb_solver(tol,nparams)
  dir = joinpath(test_dir,string(compression)*"_"*string(tol))
  rbop = reduced_operator(rbsolver,feop,fesnaps,jacs,ress)
  maxsize = max_subspace_size(rbop)
  if compression == :global
    rbop′ = change_operator(rbop,feopon)
    x̂,rbstats = solve(rbsolver,rbop′,μon)
    perf = postprocess(dir,rbsolver,feopon,rbop′,x,x̂,festats,rbstats)
  else
    perf = local_solver(dir,rbsolver,rbop,μon,x,festats)
  end
  push!(perfs,perf)
  push!(maxsizes,maxsize)
  println(perf)
end

serialize(joinpath(test_dir,"results"),perfs)
serialize(joinpath(test_dir,"maxsizes"),maxsizes)
