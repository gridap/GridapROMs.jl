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
nparams=200
nparams_res=nparams_jac=nparams
sketch=:sprn
compression=:local
ncentroids=16

domain = (0,2,0,2)
n = 40
partition = (n,n)
bgmodel = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2

# case 1: translation
pdomain = (0.6,1.4)
# # case 2: translation and dilation
# pdomain = (0.6,1.4,0.25,0.35)
pspace = ParamSpace(pdomain)

const γd = 10.0
const hd = 2/n

# quantities on the base configuration

μ0 = (1.0,0.3)
x0 = Point(μ0[1],μ0[1])
geo = !disk(μ0[2],x0=x0)
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
  # case 1: translation
  φ(μ) = x -> VectorValue(μ[1]-μ0[1],μ[1]-μ0[1])
  # # case 2: translation and dilation
  # φ(x) = VectorValue(μrand[1]-x[1] + (μrand[2]/μ0[2])*(x[1]-μ0[1]),
  #                    μrand[1]-x[2] + (μrand[2]/μ0[2])*(x[2]-μ0[1]))
  φμ(μ) = parameterize(φ,μ)

  E = 1
  ν = 0.33
  λ = E*ν/((1+ν)*(1-2*ν))
  p = E/(2(1+ν))
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

function def_fe_operator(μ)
  φh = get_deformation_map(μ)
  Ωactφ = mapped_grid(Ωact,φh)
  Ωφ = mapped_grid(Ω,φh)
  Γφ = mapped_grid(Γ,φh)

  dΩ₀(meas) = ReferenceMeasure(meas,Ωφ)
  dΓ₀(meas) = ReferenceMeasure(meas,Γφ)

  ∇act₀(f) = ∇₀(f,Ωactφ)
  nΓ₀ = n₀(n_Γ,Ωactφ)

  a(μ,u,v,dΩ,dΓ) = ∫(∇act₀(v)⋅∇act₀(u))dΩ₀(dΩ) + ∫( (γd/hd)*v*u - v*(nΓ₀⋅∇act₀(u)) - (nΓ₀⋅∇act₀(v))*u )dΓ₀(dΓ)
  l(μ,v,dΩ) = ∫(fμ(μ)⋅v)dΩ₀(dΩ)
  res(μ,u,v,dΩ) = ∫(∇(v)⋅∇(u))dΩ₀(dΩ) - l(μ,v,dΩ)

  LinearParamOperator(res,a,pspace,trial,test,domains)
end

function rb_solver(tol,nparams)
  tol = method == :ttsvd ? fill(tol,3) : tol
  hr = compression == :global ? HyperReduction : LocalHyperReduction
  tolhr = tol.*1e-2
  state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids=16)
  residual_reduction = hr(tolhr;nparams,ncentroids=16)
  jacobian_reduction = hr(tolhr;nparams,ncentroids=16)
  fesolver = LUSolver()
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
  maxsizes = Int[]
  for (μi,xi) in zip(μsplit,xsplit)
    feopi = def_fe_operator(μi)
    rbopi = change_operator(get_local(rbop,first(μi)),feopi)
    x̂,rbstats = solve(rbsolver,rbopi,μi)
    perf = postprocess(dir,rbsolver,feopi,rbopi,xi,x̂,festats,rbstats)
    maxsize = max_subspace_size(rbopi)
    push!(perfs,perf)
    push!(maxsizes,maxsize)
  end
  return mean(perfs),maximum(maxsizes)
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

test_dir = datadir("moving_poisson")
create_dir(test_dir)

μ = realization(pspace;nparams)
feop = def_fe_operator(μ)

μon = realization(pspace;nparams=10,sampling=:uniform)
feopon = def_fe_operator(μon)

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
  if compression == :global
    rbop′ = change_operator(rbop,feopon)
    x̂,rbstats = solve(rbsolver,rbop′,μon)
    perf = postprocess(dir,rbsolver,feopon,rbop′,x,x̂,festats,rbstats)
    maxsize = max_subspace_size(rbop′)
  else
    perf,maxsize = local_solver(dir,rbsolver,rbop,μon,x,festats)
  end
  push!(perfs,perf)
  push!(maxsizes,maxsize)
  println(perf)
end

serialize(joinpath(test_dir,"results"),perfs)
serialize(joinpath(test_dir,"maxsizes"),maxsizes)

# k, = get_clusters(rbop.test)
# μsplit = cluster(μon,k)
# xsplit = cluster(x,k)
# (μi,xi) = μsplit[7],xsplit[7]
# feopi = def_fe_operator(μi)
# rbopi = change_operator(get_local(rbop,first(μi)),feopi)
# x̂,rbstats = solve(rbsolver,rbopi,μi)
# perf = postprocess(dir,rbsolver,feopi,rbopi,xi,x̂,festats,rbstats)
