module PoissonEmbedded

using Gridap
using GridapEmbedded
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod,n=20;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  R = 0.5
  L = 0.8*(2*R)
  p1 = Point(0.0,0.0)
  p2 = p1 + VectorValue(L,0.0)

  geo1 = disk(R,x0=p1)
  geo2 = disk(R,x0=p2)
  geo = setdiff(geo1,geo2)

  t = 1.01
  pmin = p1-t*R
  pmax = p1+t*R
  dp = pmax - pmin

  partition = (n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(pmin,pmax,partition)
  else
    bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
  end

  order = 2
  degree = 2*order

  cutgeo = cut(bgmodel,geo)
  Ωbg = Triangulation(bgmodel)
  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩbg = Measure(Ωbg,degree)
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  nΓ = get_normal_vector(Γ)

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

  reffe = ReferenceFE(lagrangian,Float64,order)

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testbg = FESpace(Ωbg,reffe,conformity=:H1)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = ParamTrialFESpace(test,gμ)
  feop = ExtensionLinearParamOperator(res,a,pspace,trial,test,domains)

  energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
  tolrank = tol_or_rank(tol,rank)
  state_reduction = Reduction(tolrank,energy;nparams,sketch)

  trial = ParamTrialFESpace(test)
  feop = LinearParamOperator(b,a,pspace,trial,test,domains)

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  # offline
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  # online
  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  # test
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
  println(perf)
end

main(:pod)
main(:ttsvd)

end

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using GridapEmbedded
using GridapROMs
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.Extensions
using GridapROMs.DofMaps
using GridapROMs.RBSteady
using GridapROMs.Utils
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
n = 20
partition = (n,n)

dp = pmax - pmin

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

model = TProductDiscreteModel(pmin,pmax,partition)
cutgeo = cut(model,geo2)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ωbg = Triangulation(model)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)

order = 1
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
dΓ = Measure(Γ,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

V = FESpace(Ωbg,reffe,conformity=:H1)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(V,Vact,aggregates)

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

Vext = DirectSumFESpace(V,Vagg)
Uext = ParamTrialFESpace(Vext,gμ)

feop = ExtensionLinearParamOperator(res,a,pspace,Uext,Vext,domains)

solver = ExtensionSolver(LUSolver())
energy(u,v) = ∫(v*u)dΩbg + ∫(∇(v)⋅∇(u))dΩbg
state_reduction = Reduction(fill(1e-4,2),energy;nparams=50)
rbsolver = RBSolver(solver,state_reduction;nparams_res=50,nparams_jac=20)

fesnaps, = solution_snapshots(rbsolver,feop)

rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)
reduction = get_reduction(rbsolver.jacobian_reduction)
# basis = projection(reduction,jacs[1])
basis = projection(reduction,ress[2])
# (rows,cols),interp = empirical_interpolation(basis)
rows,interp = empirical_interpolation(basis)
# trian = jacs.trians[1]
trian = ress.trians[2]
# cells_trial = RBSteady.reduced_cells(red_trial,trian,cols)
cells = RBSteady.reduced_cells(red_test,trian,rows)
# cells = union(cells_trial,cells_test)
# icols = RBSteady.reduced_idofs(red_trial,trian,cells,cols)
RBSteady.reduced_idofs(red_test,trian,cells,rows)
cell_dof_ids = Extensions.get_bg_cell_dof_ids(red_test.space,trian)
# RBSteady.get_cells_to_idofs(cell_dof_ids,cells,rows)
RBSteady.get_idof_correction(cell_dof_ids)
cellids = Vext.space.bg_cell_dof_ids
