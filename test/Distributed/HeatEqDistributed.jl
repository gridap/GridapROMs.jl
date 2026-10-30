module HeatEqDistributed

using DrWatson
using Gridap
using GridapDistributed
using GridapROMs
using PartitionedArrays
using Random
using Test

using Gridap.Algebra
using Gridap.FESpaces
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.Distributed
using GridapROMs.RBSteady
using GridapROMs.Utils

import PartitionedArrays: getany

function main(
  distribute,parts,
  compression=:global,hypred_strategy=:deim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),ncentroids=2
  )

  ranks = distribute(LinearIndices((prod(parts),)))
  seed = getany(emit(map(_ -> rand(UInt32),ranks)))
  Random.seed!(seed) # set a seed for the parameters

  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:deim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :deim

  i_am_main(ranks) && println("Running test with $compression (pod, $hypred_strategy) strategy")

  domain = (0,1,0,1)
  partition = (8,8)
  model = CartesianDiscreteModel(ranks,parts,domain,partition)

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

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  stiffness(μ,t,u,v) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
  mass(μ,t,uₜ,v) = ∫(v*uₜ)dΩ
  rhs(μ,t,v) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn

  trian_rhs = (Ω,Γn)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_rhs,(trian_stiffness,trian_mass))

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = TransientTrialParamFESpace(test,gμt)

  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  state_reduction = TransientReduction(tol,H1();nparams,compression,ncentroids)

  fesolver = ThetaMethod(LUSolver(),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jacs=(nparams_jac,nparams_jac),hypred_strategy,verbose=i_am_main(ranks))

  feop = TransientLinearParamOperator(rhs,(stiffness,mass),ptspace,trial,test,domains)
  fesnaps = solution_snapshots(rbsolver,feop,uh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,sampling=:uniform)
  x̂ = solution_snapshots(rbsolver,rbop,μon,uh0μ)
  x = solution_snapshots(rbsolver,feop,μon,uh0μ)
  perf = rom_performance(rbsolver,rbop,x,x̂)
  i_am_main(ranks) && println(perf)
  err = perf.error

  dir = datadir("heateq_distributed_save")
  if i_am_main(ranks)
    isdir(dir) && rm(dir;recursive=true)
    mkpath(dir)
  end
  PartitionedArrays.barrier(ranks)

  save(dir,rbop)
  for rbop_load in (load_operator(dir,feop,ranks),load_operator(dir,feop))
    x̂_load = solution_snapshots(rbsolver,rbop_load,μon,uh0μ)
    @test rom_performance(rbsolver,rbop_load,x,x̂_load).error ≈ err
  end
end

end
