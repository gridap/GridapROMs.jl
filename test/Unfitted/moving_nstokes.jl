using DrWatson
using BlockArrays
using LinearAlgebra
using Serialization

using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Geometry
using Gridap.Arrays
using Gridap.Algebra
using Gridap.CellData
using Gridap.MultiField
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.ODEs
using GridapROMs.DofMaps
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.Utils

using GridapSolvers
using GridapSolvers.NonlinearSolvers

import Gridap.Geometry: push_normal

method=:pod
tol=1e-4
rank=nothing
nparams=100
nparams_res=nparams_jac=nparams
sketch=:sprn
compression=:local
ncentroids=8

const L = 2
const W = 1

const n = 40

domain = (0,L,0,W)
partition = Int.((L,W) .* n)
bgmodel = CartesianDiscreteModel(domain,partition)

const R = 0.3
x0 = Point(3L/4,W/2)
geo = disk(R,x0=x0)
cutgeo = cut(bgmodel,!geo)

Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)

order = 2
degree = 2*order

const γd = 10.0
const hd = 0.01

const E = 1
const ν = 0.33
const λ = E*ν/((1+ν)*(1-2*ν))
const p = E/(2(1+ν))
σ(ε) = λ*tr(ε)*one(ε) + 2*p*ε

const Re = 100

# params
μ0 = (0.0,0.0)
pdomain = (0.0,L/2,0.0,W/2)
pspace = ParamSpace(pdomain)

dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(dp*q)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫((γd/hd)*du⋅v)dΓ
coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

domains_lin = FEDomains((Ω,),(Ω,Γ))
domains_nlin = FEDomains((Ω,),(Ω,))

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)

g(μ) = x -> VectorValue(-x[2]*(x[2]-W),0.0)*(x[1]≈0.0)
gμ(μ) = parameterize(g,μ)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

testact_u = FESpace(Ωact,reffe_u,conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
testact_p = FESpace(Ωact,reffe_p,conformity=:H1)
test_u = AgFEMSpace(testact_u,aggregates)
trial_u = ParamTrialFESpace(test_u,gμ)
test_p = AgFEMSpace(testact_p,aggregates)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

function get_deformation_map(μ)
  is_corner_x(x,μ) = x[1] ≈ 0.0
  is_corner_y(x,μ) = x[2] ≈ 0.0 && x[1] < μ[1]
  is_corner(x,μ) = is_corner_x(x,μ)||is_corner_y(x,μ)
  sigmoid(x,μ) = (1+exp(0.1/μ[1]))/(1+exp(0.1/(μ[1]-x[1])))

  φ(μ) = x -> VectorValue(0.0,μ[2]*(1-x[2]/W)*sigmoid(x,μ)*is_corner(x,μ))
  φμ(μ) = parameterize(φ,μ)

  a(μ,u,v) = (
    ∫( ε(v)⊙(σ∘ε(u)) )*dΩ +
    ∫( (γd/hd)*v⋅u - v⋅(n_Γ⋅(σ∘ε(u))) - (n_Γ⋅(σ∘ε(v)))⋅u )dΓ
  )
  res(μ,u,v) = ∫( ε(v)⊙(σ∘ε(u)) )*dΩ - ∫((n_Γ⋅(σ∘ε(v)))⋅φμ(μ))dΓ

  reffeφ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  Vφact = FESpace(Ωact,reffeφ,conformity=:H1,dirichlet_tags="boundary")
  Vφ = AgFEMSpace(Vφact,aggregates)
  Uφ = ParamTrialFESpace(Vφ,φμ)

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

function rb_solver(tol,nparams)
  fesolver = NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true)
  hr = compression == :global ? HyperReduction : LocalHyperReduction
  tolhr = tol.*1e-2
  state_reduction = SupremizerReduction(coupling,tol,energy;nparams,sketch,compression,ncentroids)
  residual_reduction = hr(tolhr;nparams,ncentroids)
  jacobian_reduction = hr(tolhr;nparams,ncentroids)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function plot_sol(rbop,x,x̂,μon,dir,i=1)
  φh = get_deformation_map(μon)
  φhi = param_getindex(φh,i)
  Ωφ = mapped_grid(Ω,φhi)
  X = trial(μon)
  U = param_getindex(X[1],i)
  P = param_getindex(X[2],i)

  uu,pp = blocks(x)
  ûû,p̂p̂ = blocks(x̂)

  u,û = uu[:,i],ûû[:,i]
  p,p̂ = pp[:,i],p̂p̂[:,i]

  uh = FEFunction(U,u)
  ûh = FEFunction(U,û)
  euh = FEFunction(test_u,abs.(u-û))
  ph = FEFunction(P,p)
  p̂h = FEFunction(P,p̂)
  eph = FEFunction(test_p,abs.(p-p̂))

  cellfields = ["uh"=>uh,"ûh"=>ûh,"euh"=>euh,"ph"=>ph,"p̂h"=>p̂h,"eph"=>eph]
  writevtk(Ωφ,dir*".vtu",cellfields=cellfields)
end

function postprocess(
  dir,
  rbsolver,
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
  k, = get_clusters(get_test(rbop))
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
max_subspace_size(a::Union{BlockProjection,BlockHRProjection}) = sum(map(max_subspace_size,a.array))

function max_subspace_size(a::LocalProjection)
  maxsize = 0
  for proj in a.projections
    maxsize = max(maxsize,max_subspace_size(proj))
  end
  return maxsize
end

test_dir = joinpath(datadir("moving_nstokes"),string(method))
create_dir(test_dir)

rbsolver = rb_solver(tol,nparams)

# fesnaps,x,festats,μ,feop,μon,feopon = try
#   fesnaps = load_snapshots(test_dir)
#   x = load_snapshots(test_dir;label="online")
#   festats = load_stats(test_dir;label="online")

#   μ = get_realization(fesnaps)
#   feop = def_fe_operator(μ)

#   μon = get_realization(x)
#   feopon = def_fe_operator(μon)

#   fesnaps,x,festats,μ,feop,μon,feopon
# catch
#   μ = realization(pspace;nparams)
#   feop = def_fe_operator(μ)

#   μon = realization(pspace;nparams=10,sampling=:uniform)
#   feopon = def_fe_operator(μon)

#   fesnaps, = solution_snapshots(rbsolver,feop,μ)
#   save(test_dir,fesnaps)

#   x,festats = solution_snapshots(rbsolver,feopon,μon)
#   save(test_dir,x;label="online")
#   save(test_dir,festats;label="online")

#   fesnaps,x,festats,μ,feop,μon,feopon
# end

# jacs_lin = jacobian_snapshots(rbsolver,get_linear_operator(feop),fesnaps)
# ress_lin = residual_snapshots(rbsolver,get_linear_operator(feop),fesnaps)
# jacs_nlin = jacobian_snapshots(rbsolver,get_nonlinear_operator(feop),fesnaps)
# ress_nlin = residual_snapshots(rbsolver,get_nonlinear_operator(feop),fesnaps)
# jacs = (jacs_lin,jacs_nlin)
# ress = (ress_lin,ress_nlin)

# perfs = ROMPerformance[]
# maxsizes = Int[]

# for tol in (1e-1,1e-2,1e-3,1e-4)
#   rbsolver = rb_solver(tol,nparams)
#   dir = joinpath(test_dir,string(compression)*"_"*string(tol))
#   rbop = reduced_operator(rbsolver,feop,fesnaps,jacs,ress)
#   maxsize = max_subspace_size(rbop)
#   if compression == :global
#     rbop′ = change_operator(rbop,feopon)
#     x̂,rbstats = solve(rbsolver,rbop′,μon)
#     perf = postprocess(dir,rbsolver,feopon,rbop′,x,x̂,festats,rbstats)
#   else
#     perf = local_solver(dir,rbsolver,rbop,μon,x,festats)
#   end
#   push!(perfs,perf)
#   push!(maxsizes,maxsize)
#   println(perf)
# end

# serialize(joinpath(test_dir,"results"),perfs)
# serialize(joinpath(test_dir,"maxsizes"),maxsizes)

μ = realization(pspace;nparams,start=101)
feop = def_fe_operator(μ)

fesnaps, = solution_snapshots(rbsolver,feop,μ)
save(test_dir,fesnaps;label="new")
