module MovingPoisson

using DrWatson
using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Geometry
using Gridap.CellData
using Gridap.FESpaces
using GridapROMs.RBSteady
using GridapROMs.Extensions

import Gridap.Geometry: push_normal

method = :pod
tol = 1e-4
nparams = 100
compression = :local
ncentroids = 8

const L = 2
const W = 2
const n = 40
const γd = 10.0
const hd = max(L,W)/n

domain = (0,L,0,W)
partition = (n,n)
bgmodel = method==:ttsvd ? TProductDiscreteModel(domain,partition) : CartesianDiscreteModel(domain,partition)

order = 1
degree = 2*order

pdomain = (0.6,1.4,0.25,0.35)
pspace = ParamSpace(pdomain)

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

energy(du,v) = method==:ttsvd ? ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg : ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ + ∫((γd/hd)*v*du)dΓ

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

function local_solver(rbsolver,rbop,μ,x,festats)
  k, = get_clusters(rbop.test)
  μsplit = cluster(μ,k)
  xsplit = cluster(x,k)
  perfs = ROMPerformance[]
  for (μi,xi) in zip(μsplit,xsplit)
    feopi = get_feop(μi)
    rbopi = change_operator(get_local(rbop,first(μi)),feopi)
    x̂,rbstats = solve(rbsolver,rbopi,μi)
    perf = eval_performance(rbsolver,feopi,rbopi,xi,x̂,festats,rbstats)
    push!(perfs,perf)
  end
  return mean(perfs)
end

fesolver = LUSolver()
if method == :ttsvd
  tol = fill(tol,2)
  fesolver = ExtensionSolver(fesolver)
end
hr = compression == :global ? HyperReduction : LocalHyperReduction
state_reduction = Reduction(tol,energy;nparams,sketch=:sprn,compression,ncentroids)
residual_reduction = hr(tol.*1e-2;nparams,ncentroids)
jacobian_reduction = hr(tol.*1e-2;nparams,ncentroids)
rbsolver = RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)

μ = realization(pspace;nparams)
feop = get_feop(μ)
fesnaps, = solution_snapshots(rbsolver,feop,μ)

μon = realization(pspace;nparams=10,sampling=:uniform)
feopon = get_feop(μon)
x,festats = solution_snapshots(rbsolver,feopon,μon)

rbop = reduced_operator(rbsolver,feop,fesnaps)

if compression == :global
  rbop′ = change_operator(rbop,feopon)
  x̂,rbstats = solve(rbsolver,rbop′,μon)
  perf = eval_performance(rbsolver,feop,rbop′,x,x̂,festats,rbstats)
else
  perf = local_solver(rbsolver,rbop,μon,x,festats)
end

println(perf)

end
