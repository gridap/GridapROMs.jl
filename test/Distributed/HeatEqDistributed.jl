module HeatEqDistributed

using DrWatson
using Gridap
using GridapDistributed
using GridapROMs
using GridapPETSc
using PartitionedArrays
using Test

using Gridap.Algebra
using Gridap.FESpaces
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.Distributed
using GridapROMs.RBSteady
using GridapROMs.Utils

method=:pod
compression=:global
hypred_strategy=:deim
tol=1e-4
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
ncentroids=2

method = method ∈ (:pod,:ttsvd) ? method : :pod
compression = compression ∈ (:global,:local) ? compression : :global
hypred_strategy = hypred_strategy ∈ (:deim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :deim

domain = (0,1,0,1)
partition = (8,8)

pdomain = (1,10,1,10,1,10)
θ = 0.5
dt = 0.01
t0 = 0.0
tf = 2*dt
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
aμt(μ,t) = parameterise(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterise(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = parameterise(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
gμt(μ,t) = parameterise(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = parameterise(u0,μ)

order = 1
degree = 2*order

state_reduction = TransientReduction(tol,H1();nparams,compression,ncentroids)

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks,parts,domain,partition)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
  mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
  rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
  res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = TransientTrialParamFESpace(test,gμt)

  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  fesolver = ThetaMethod(LUSolver(),dt,θ)#PETScLinearSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jacs=(nparams_jac,nparams_jac),hypred_strategy)

  feop = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)

  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,start=nparams+1)
  x̂ = solve(rbsolver,rbop,μon,uh0μ)
  x, = solution_snapshots(rbsolver,feop,μon,uh0μ)
  println(rom_performance(rbsolver,rbop,x,x̂))

  perr = RBSteady.projection_error(rbsolver,rbop,fesnaps)
  println("diagnostic | projection error (basis + project/inv_project, no HR): ", perr)

  rbsolverx = RBSteady.set_params(rbsolver;nparams=num_params(x))
  res = residual_snapshots(rbsolverx,feop,x)
  jac = jacobian_snapshots(rbsolverx,feop,x)
  err_res,err_jac = RBSteady.hr_error(rbsolverx,rbop,res,jac,x)
  println("diagnostic | hr error residual (per trian): ", err_res)
  println("diagnostic | hr error jacobian (per trian): ", err_jac)

  # per-rank save / load round-trip of the FE snapshots (distributed)
  diagdir = mkpath(joinpath(@__DIR__,"boh_diag_heateq"))
  save(diagdir,fesnaps)
  fesnaps_loaded = load_snapshots(diagdir,ranks)
  println("diagnostic | snapshots save/load round-trip ok: ",
    compute_relative_error(fesnaps,fesnaps_loaded) < 1e-12)
end

end