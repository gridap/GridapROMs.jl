module PoissonDistributed

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
pspace = ParamSpace(pdomain)

a(μ) = x -> exp(-x[1]/sum(μ))
aμ(μ) = parameterise(a,μ)

f(μ) = x -> 1.
fμ(μ) = parameterise(f,μ)

g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
gμ(μ) = parameterise(g,μ)

h(μ) = x -> abs(cos(μ[3]*x[2]))
hμ(μ) = parameterise(h,μ)

order = 1
degree = 2*order

state_reduction = Reduction(tol,H1();nparams,compression,ncentroids)

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks,parts,domain,partition)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)

  fesolver = LUSolver()#PETScLinearSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy,verbose=i_am_main(ranks))

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,start=nparams+1)
  x̂ = solve(rbsolver,rbop,μon)
  x, = solution_snapshots(rbsolver,feop,μon)
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
  diagdir = mkpath(joinpath(@__DIR__,"boh_diag"))
  save(diagdir,fesnaps)
  fesnaps_loaded = load_snapshots(diagdir,ranks)
  println("diagnostic | snapshots save/load round-trip ok: ",
    compute_relative_error(fesnaps,fesnaps_loaded) < 1e-12)
end

end