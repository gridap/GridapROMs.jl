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

pspace = ParamSpace((-0.4,0.4))

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
  φ(μ) = x -> VectorValue(μ[1],μ[1])
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

  dΩ₀ = ReferenceMeasure(Ωφ,degree)
  dΓ₀ = ReferenceMeasure(Γφ,degree)

  ∇act₀(f) = ∇₀(f,Ωactφ)
  nΓ₀ = n₀(n_Γ,Ωactφ)

  a(μ,u,v,dΩ₀,dΓ₀) = ∫(∇act₀(v)⋅∇act₀(u))*dΩ₀ + ∫( (γd/hd)*v*u - v*(nΓ₀⋅∇act₀(u)) - (nΓ₀⋅∇act₀(v))*u )*dΓ₀
  l(μ,v,dΩ₀) = ∫(fμ(μ)⋅v)dΩ₀
  res(μ,u,v,dΩ₀) = ∫(∇(v)⋅∇(u))dΩ₀ - l(μ,v,dΩ₀)

  LinearParamOperator(res,a,pspace,trial,test,domains)
end

μ = realization(pspace;nparams)
feop = def_fe_operator(μ)

μon = realization(pspace;nparams=1,sampling=:uniform)
feopon = def_fe_operator(μon)

function rb_solver(tol,nparams)
  tol = method == :ttsvd ? fill(tol,3) : tol
  tolhr = tol.*1e-2
  state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids=8)
  residual_reduction = LocalHyperReduction(tolhr;nparams,ncentroids=8)
  jacobian_reduction = LocalHyperReduction(tolhr;nparams,ncentroids=8)
  # residual_reduction = HyperReduction(tolhr;nparams)
  # jacobian_reduction = HyperReduction(tolhr;nparams)
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

  writevtk(Ωφ,dir*"_Ω",cellfields=["uh"=>uh,"ûh"=>ûh,"eh"=>eh])
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
  return eval_performance(rbsolver,feopon,rbop,x,x̂,festats,rbstats)
end

function local_solver(rbop::RBOperator,μ::Realization)
  k, = get_clusters(rbop.test)
  μsplit = cluster_realizatons(μ,k)
  perfs = ROMPerformance[]
  for μi in μsplit
    feopi = def_fe_operator(μi)
    rbopi = change_operator(rbop,feopi)
    x̂,rbstats = solve(rbsolver,rbopi,μi)
    perf = postprocess(dir,rbsolver,feopi,rbopi,x,x̂,festats,rbstats)
    push!(perfs,perf)
  end
  return mean(perfs)
end

test_dir = datadir("moving_poisson")
create_dir(test_dir)

rbsolver = rb_solver(tol,nparams)
fesnaps, = solution_snapshots(rbsolver,feop)
jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

x,festats = solution_snapshots(rbsolver,feopon,μon)

for tol in (1e-2,1e-3,1e-4,1e-5,1e-6)
  rbsolver = rb_solver(tol,nparams)

  dir = joinpath(test_dir,string(tol))

  rbop′ = reduced_operator(rbsolver,feop,fesnaps,jacs,ress)
  rbop = change_operator(rbop′,feopon)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  perf = postprocess(dir,rbsolver,feopon,rbop,x,x̂,festats,rbstats)
  println(perf)
  # perf = local_solver(rbop,μon)
  # println(perf)
end
