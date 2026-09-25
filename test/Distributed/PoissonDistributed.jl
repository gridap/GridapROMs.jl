module PoissonDistributed

using DrWatson
using Gridap
using GridapDistributed
using GridapROMs
using PartitionedArrays
using Test

using Gridap.Algebra
using Gridap.FESpaces
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.Distributed
using GridapROMs.RBSteady

function main(
  distribute,parts,
  compression=:global,hypred_strategy=:deim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),ncentroids=2
  )

  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:deim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :deim

  println("Running test with $compression (pod, $hypred_strategy) strategy")

  ranks = distribute(LinearIndices((prod(parts),)))

  domain = (0,1,0,1)
  partition = (8,8)
  model = CartesianDiscreteModel(ranks,parts,domain,partition)

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

  state_reduction = Reduction(tol,H1();nparams,compression,ncentroids)

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,sampling=:uniform)
  x̂ = solve(rbsolver,rbop,μon)
  x, = solution_snapshots(rbsolver,feop,μon)
  println(rom_performance(rbsolver,rbop,x,x̂))
end

end