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

import Gridap.Geometry: push_normal

method=:pod
tol=1e-4
rank=nothing
nparams=100
nparams_res=nparams_jac=nparams
sketch=:sprn
compression=:local
ncentroids=8

domain = (0,0.4,0,0.2,0,0.05)
partition = (40,20,5)
bgmodel = method==:ttsvd ? TProductDiscreteModel(domain,partition) : CartesianDiscreteModel(domain,partition)

order = 2
degree = 2*order

pdomain = (0.06,0.14,0.06,0.14,0.26,0.34,0.06,0.14)
pspace = ParamSpace(pdomain)

const γd = 10.0
const hd = 0.1

const E = 1
const ν = 0.33
const λ = E*ν/((1+ν)*(1-2*ν))
const p = E/(2(1+ν))
σ(ε) = λ*tr(ε)*one(ε) + 2*p*ε

# quantities on the base configuration

const R = 0.03
μ0 = (0.1,0.1,0.3,0.1)
x1 = Point(μ0[1],μ0[2],0.0)
x2 = Point(μ0[3],μ0[4],0.0)
v = VectorValue(0.0,0.0,0.05)
geo = cylinder(R,x0=x1,v=v)∪cylinder(R,x0=x2,v=v)
cutgeo = cut(bgmodel,!geo)
cutgeo1 = cut(bgmodel,!cylinder(R,x0=x1,v=v))
cutgeo2 = cut(bgmodel,!cylinder(R,x0=x2,v=v))

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)
Γ1 = EmbeddedBoundary(cutgeo1)
Γ2 = EmbeddedBoundary(cutgeo2)

n_Γ = get_normal_vector(Γ)
n_Γ1 = get_normal_vector(Γ1)
n_Γ2 = get_normal_vector(Γ2)

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓ1 = Measure(Γ1,degree)
dΓ2 = Measure(Γ2,degree)

labels = get_face_labeling(bgmodel)
topbottom = [21,22]
bnd = setdiff(1:26,topbottom)
add_tag_from_tags!(labels,"bnd",bnd)
add_tag_from_tags!(labels,"topbottom",topbottom)

energy(du,v) = method==:ttsvd ? ∫(v⋅du)dΩbg + ∫(∇(v)⊙∇(du))dΩbg : ∫(v⋅du)dΩ + ∫(∇(v)⊙∇(du))dΩ

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

domains = FEDomains((Ω,),(Ω,Γ1,Γ2))

f(μ) = x -> VectorValue(2*x[1]*x[2],2*x[1]*x[2],0.0)
fμ(μ) = parameterize(f,μ)

reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
testact = FESpace(Ωact,reffe,conformity=:H1,dirichlet_tags="bnd")
test = AgFEMSpace(testact,aggregates)
trial = ParamTrialFESpace(test)

function get_deformation_map(μ)
  φ1(μ) = x -> VectorValue(μ[1]-μ0[1],μ[2]-μ0[2],0.0)
  φ1μ(μ) = parameterize(φ1,μ)
  φ2(μ) = x -> VectorValue(μ[3]-μ0[3],μ[4]-μ0[4],0.0)
  φ2μ(μ) = parameterize(φ2,μ)

  a(μ,u,v) = (
    ∫( ε(v)⊙(σ∘ε(u)) )*dΩ +
    ∫( (γd/hd)*v⋅u - v⋅(n_Γ1⋅(σ∘ε(u))) - (n_Γ1⋅(σ∘ε(v)))⋅u )dΓ1 +
    ∫( (γd/hd)*v⋅u - v⋅(n_Γ2⋅(σ∘ε(u))) - (n_Γ2⋅(σ∘ε(v)))⋅u )dΓ2
  )
  l(μ,v) = (
    ∫( (γd/hd)*v⋅φ1μ(μ) - (n_Γ1⋅(σ∘ε(v)))⋅φ1μ(μ) )dΓ1 +
    ∫( (γd/hd)*v⋅φ2μ(μ) - (n_Γ2⋅(σ∘ε(v)))⋅φ2μ(μ) )dΓ2
  )
  res(μ,u,v) = ∫( ε(v)⊙(σ∘ε(u)) )*dΩ - l(μ,v)

  maskbnd = [true,true,true]
  masktopbottom = [false,false,true]

  reffeφ = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  Vφact = FESpace(Ωact,reffeφ,conformity=:H1,
    dirichlet_tags=["bnd","topbottom"],
    dirichlet_masks=[maskbnd,masktopbottom])
  Vφ = AgFEMSpace(Vφact,aggregates)
  Uφ = ParamTrialFESpace(Vφ)

  feop = LinearParamOperator(res,a,pspace,Uφ,Vφ)
  dφ, = solve(LUSolver(),feop,μ)
  FEFunction(Uφ(μ),dφ)
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
  _n_Γ1 = push_normal∘(invJt,n_Γ1)
  _n_Γ2 = push_normal∘(invJt,n_Γ2)
  dJΓ1 = dJΓn∘(dJ,C,_n_Γ1)
  dJΓ2 = dJΓn∘(dJ,C,_n_Γ2)
  ∫_Ω(a) = ∫(a*dJ)
  ∫_Γ1(a) = ∫(a*dJΓ1)
  ∫_Γ2(a) = ∫(a*dJΓ2)

  a(μ,u,v,dΩ,dΓ1,dΓ2) = (
    ∫_Ω( ε(v)⊙(σ∘ε(u)) )*dΩ +
    ∫_Γ1( (γd/hd)*v⋅u - v⋅(_n_Γ1⋅(σ∘ε(u))) - (_n_Γ1⋅(σ∘ε(v)))⋅u )dΓ1 +
    ∫_Γ2( (γd/hd)*v⋅u - v⋅(_n_Γ2⋅(σ∘ε(u))) - (_n_Γ2⋅(σ∘ε(v)))⋅u )dΓ2
  )
  l(μ,v,dΩ) = ∫_Ω(fμ(μ)⋅v)dΩ
  res(μ,u,v,dΩ) = ∫_Ω( ε(v)⊙(σ∘ε(u)) )*dΩ - l(μ,v,dΩ)

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
  _n_Γ1 = push_normal∘(invJt,n_Γ1)
  _n_Γ2 = push_normal∘(invJt,n_Γ2)
  dJΓ1 = dJΓn∘(dJ,C,_n_Γ1)
  dJΓ2 = dJΓn∘(dJ,C,_n_Γ2)
  ∫_Ω(a) = ∫(a*dJ)
  ∫_Γ1(a) = ∫(a*dJΓ1)
  ∫_Γ2(a) = ∫(a*dJΓ2)

  a(μ,u,v,dΩ,dΓ1,dΓ2) = (
    ∫_Ω( ε(v)⊙(σ∘ε(u)) )*dΩ +
    ∫_Γ1( (γd/hd)*v⋅u - v⋅(_n_Γ1⋅(σ∘ε(u))) - (_n_Γ1⋅(σ∘ε(v)))⋅u )dΓ1 +
    ∫_Γ2( (γd/hd)*v⋅u - v⋅(_n_Γ2⋅(σ∘ε(u))) - (_n_Γ2⋅(σ∘ε(v)))⋅u )dΓ2
  )
  l(μ,v,dΩ) = ∫_Ω(fμ(μ)⋅v)dΩ
  res(μ,u,v,dΩ) = ∫_Ω( ε(v)⊙(σ∘ε(u)) )*dΩ - l(μ,v,dΩ)

  testbg = FESpace(Ωbg,reffe,conformity=:H1,dirichlet_tags="bnd")
  testext = DirectSumFESpace(testbg,test)
  trialext = ParamTrialFESpace(testext)
  ExtensionLinearParamOperator(res,a,pspace,trialext,testext,domains)
end

get_feop = method==:ttsvd ? def_extended_fe_operator : def_fe_operator

function rb_solver(tol,nparams)
  fesolver = LUSolver()
  if method == :ttsvd
    tol = fill(tol,4)
    fesolver = ExtensionSolver(fesolver)
  end
  hr = compression == :global ? HyperReduction : LocalHyperReduction
  tolhr = tol.*1e-2
  state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids)
  residual_reduction = hr(tolhr;nparams,ncentroids)
  jacobian_reduction = hr(tolhr;nparams,ncentroids)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function plot_sol(rbop,x,x̂,μon,dir,i=1)
  φh = get_deformation_map(μon)
  φhi = param_getindex(φh,i)
  Ωφ = mapped_grid(Ω,φhi)
  U = param_getindex(trial(μon),i)

  if method==:ttsvd
    u,û = vec(x[:,:,:,:,i]),vec(x̂[:,:,:,:,i])
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

test_dir = joinpath(datadir("moving_elasticity"),string(method))
create_dir(test_dir)

μ = realization(pspace;nparams)
feop = get_feop(μ)

μon = realization(pspace;nparams=10,sampling=:uniform)
feopon = get_feop(μon)

rbsolver = rb_solver(tol,nparams)
fesnaps, = solution_snapshots(rbsolver,feop,μ)
save(test_dir,fesnaps)
jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

x,festats = solution_snapshots(rbsolver,feopon,μon)
save(test_dir,x;label="online")
save(test_dir,festats;label="online")

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
