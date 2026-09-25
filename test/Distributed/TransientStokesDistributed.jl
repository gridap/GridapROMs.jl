module TransientStokesDistributed

using DrWatson
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using GridapDistributed
using GridapPETSc
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers
using MPI
using PartitionedArrays
using Plots
using Test

using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.Distributed

# reuses `build_fesolver`/`ASMSolver`/`PressureSolver` and the custom
# `Gridap.Algebra.solve!` override for param-batched distributed systems
include("StokesDistributed.jl")

function main(
  distribute,parts,
  compression=:global,hypred_strategy=:deim;
  tol=1e-4,nparams=15,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),ncentroids=2
  )

  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:deim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :deim

  println("Running test with $compression (pod, $hypred_strategy) strategy"); flush(stdout)

  ranks = distribute(LinearIndices((prod(parts),)))

  domain = (0,1,0,1)
  partition = (8,8)
  model = CartesianDiscreteModel(ranks,parts,domain,partition)
  println("  model built"); flush(stdout)

  pdomain = (1,10,1,10)

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a(μ,t) = x -> μ[1]*exp(sin(t))
  aμt(μ,t) = parameterise(a,μ,t)

  g(μ,t) = x -> VectorValue(-μ[2]*x[2]*(1.0-x[2])*t,0.0)*(x[1]==0.0)
  gμt(μ,t) = parameterise(g,μ,t)

  u0(μ) = x -> VectorValue(0.0,0.0)
  u0μ(μ) = parameterise(u0,μ)
  p0(μ) = x -> 0.0
  p0μ(μ) = parameterise(p0,μ)

  stiffness(μ,t,(u,p),(v,q),dΩ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  mass(μ,t,(uₜ,pₜ),(v,q),dΩ) = ∫(v⋅uₜ)dΩ
  res(μ,t,(u,p),(v,q),dΩ) = ∫(v⋅∂t(u))dΩ + stiffness(μ,t,(u,p),(v,q),dΩ)

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
  trial_u = TransientTrialParamFESpace(test_u,gμt)
  trial_p = TransientTrialParamFESpace(test_p)
  test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
  println("  fe spaces built"); flush(stdout)

  θ = 0.5
  dt = 0.01
  t0 = 0.0
  tf = 10*dt
  tdomain = t0:dt:tf
  ptspace = TransientParamSpace(pdomain,tdomain)

  xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

  energy = BlockNorm((H1(),L2()))
  coupling = DivCoupling()
  state_reduction = TransientReduction(coupling,tol,energy;nparams,compression,ncentroids)

  fesolver = ThetaMethod(StokesDistributed.build_fesolver(test_p,dΩ),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jacs=(nparams_jac,nparams_jac),hypred_strategy)

  feop = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains)
  println("  feop built, starting run_test"); flush(stdout)

  dir = datadir("diagnostics_transient_stokes_distributed")
  if i_am_main(ranks)
    isdir(dir) && rm(dir;recursive=true)
    create_dir(dir)
  end
  MPI.Initialized() && MPI.Barrier(MPI.COMM_WORLD)

  tols = [1e-1,1e-3,1e-5]
  run_test(dir,rbsolver,feop,tols,xh0μ)
  println("  run_test done"); flush(stdout)

  dgn = rom_diagnostics(dir,rbsolver,feop,xh0μ)
  println(dgn); flush(stdout)
end

end
