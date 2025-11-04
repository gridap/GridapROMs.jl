module MovingNavierStokes

using DrWatson
using Gridap
using GridapEmbedded
using GridapROMs
using GridapSolvers

using Gridap.Geometry
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapSolvers.NonlinearSolvers

import Gridap.Geometry: push_normal
import Gridap.Helpers: @notimplementedif

method = :pod
@notimplementedif method != :pod "Must still implement TT-SVD version of this test"
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
bgmodel = CartesianDiscreteModel(domain,partition)

order = 2
degree = 2*order

pdomain = (0.7,1.3,0.7,1.3)
pspace = ParamSpace(pdomain)

# quantities on the base configuration

μ0 = (1.0,1.0)
x0 = Point(μ0[1],μ0[2])
geo = !disk(0.3,x0=x0)
cutgeo = cut(bgmodel,geo)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(dp*q)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫((γd/hd)*du⋅v)dΓ
coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ

n_Γ = get_normal_vector(Γ)
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

domains_lin = FEDomains((Ω,),(Ω,Γ))
domains_nlin = FEDomains((Ω,),(Ω,))

const Re = 10

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

g(μ) = x -> VectorValue(x[2]*(W-x[2]),0.0)*(x[1]≈0.0)
gμ(μ) = parameterize(g,μ)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)

testact_u = FESpace(Ωact,reffe_u,conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
testact_p = FESpace(Ωact,reffe_p,conformity=:H1)
test_u = AgFEMSpace(testact_u,aggregates)
trial_u = ParamTrialFESpace(test_u,gμ)
test_p = AgFEMSpace(testact_p,aggregates)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

function get_deformation_map(μ)
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

  jac_lin(μ,(u,p),(v,q),dΩ,dΓ) =
    ∫_Ω( (1/Re)*∇_I(v)⊙∇_I(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ +
    ∫_Γ( (γd/hd)*v⋅u - (1/Re)*v⋅(_n_Γ⋅∇_I(u)) - (1/Re)*(_n_Γ⋅∇_I(v))⋅u + (p*_n_Γ)⋅v + (q*_n_Γ)⋅u ) * dΓ
  res_lin(μ,(u,p),(v,q),dΩ) = ∫_Ω( (1/Re)*∇_I(v)⊙∇_I(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ

  jac_nlin(μ,(u,p),(du,dp),(v,q),dΩ) = ∫_Ω( v⊙(dconv∘(du,∇_I(du),u,∇_I(u))) )dΩ
  res_nlin(μ,(u,p),(v,q),dΩ) = ∫_Ω( v⊙(conv∘(u,∇_I(u))) )dΩ

  feop_lin = LinearParamOperator(res_lin,jac_lin,pspace,trial,test,domains_lin)
  feop_nlin = ParamOperator(res_nlin,jac_nlin,pspace,trial,test,domains_nlin)
  LinearNonlinearParamOperator(feop_lin,feop_nlin)
end

function local_solver(rbsolver,rbop,μ,x,festats)
  k, = get_clusters(rbop.test)
  μsplit = cluster(μ,k)
  xsplit = cluster(x,k)
  perfs = ROMPerformance[]
  for (μi,xi) in zip(μsplit,xsplit)
    feopi = def_fe_operator(μi)
    rbopi = change_operator(get_local(rbop,first(μi)),feopi)
    x̂,rbstats = solve(rbsolver,rbopi,μi)
    perf = eval_performance(rbsolver,feopi,rbopi,xi,x̂,festats,rbstats)
    push!(perfs,perf)
  end
  return mean(perfs)
end

fesolver = NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true)
hr = compression == :global ? HyperReduction : LocalHyperReduction
state_reduction = SupremizerReduction(coupling,tol,energy;nparams,sketch=:sprn,compression,ncentroids)
residual_reduction = hr(tol.*1e-2;nparams,ncentroids)
jacobian_reduction = hr(tol.*1e-2;nparams,ncentroids)
rbsolver = RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)

μ = realization(pspace;nparams)
feop = def_fe_operator(μ)
fesnaps, = solution_snapshots(rbsolver,feop,μ)

μon = realization(pspace;nparams=10,sampling=:uniform)
feopon = def_fe_operator(μon)
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
