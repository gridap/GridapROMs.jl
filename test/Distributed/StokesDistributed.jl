module StokesDistributed

using DrWatson
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using GridapDistributed
using GridapPETSc
using GridapROMs
using GridapROMs.RBSteady
using GridapSolvers
using MPI
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers
using PartitionedArrays
using Plots
using Test

using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.Distributed

function petsc_asm_setup(ksp)
  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPCG)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCASM)
end

ASMSolver() = PETScLinearSolver(petsc_asm_setup)

function petsc_cg_jacobi_setup(ksp)
  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPCG)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCJACOBI)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[],1.e-6,1e-14,GridapPETSc.PETSC.PETSC_DEFAULT,20)
end

PressureSolver() = PETScLinearSolver(petsc_cg_jacobi_setup)

function Gridap.Algebra.solve!(
  x::GridapROMs.Distributed.AbstractParamPVector,
  ls::Gridap.Algebra.LinearSolver,
  A::GridapROMs.Distributed.AbstractParamPSparseMatrix,
  b::GridapROMs.Distributed.AbstractParamPVector
  )

  x_fixed = allocate_in_domain(A)
  fill!(x_fixed,zero(eltype(x_fixed)))

  A_item = Gridap.Arrays.testitem(A)
  x_item = Gridap.Arrays.testitem(x_fixed)
  ss = symbolic_setup(ls,A_item)
  ns = numerical_setup(ss,A_item,x_item)
  solve!(x_fixed,ns,A,b)
  copy!(x,x_fixed)
  ns
end

function build_fesolver(Q,dΩ)
  solver_u = ASMSolver()
  solver_p = PressureSolver()

  blocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(p*q)dΩ,Q,Q)]
  prec = BlockTriangularSolver(blocks,[solver_u,solver_p])
  FGMRESSolver(30,prec;rtol=1.e-6,verbose=false)
end

function main(
  distribute,parts,
  compression=:global,hypred_strategy=:deim;
  tol=1e-4,nparams=12,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),ncentroids=2
  )

  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:deim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :deim

  println("Running test with $compression (pod, $hypred_strategy) strategy")

  ranks = distribute(LinearIndices((prod(parts),)))

  domain = (0,1,0,1)
  partition = (8,8)
  model = CartesianDiscreteModel(ranks,parts,domain,partition)

  pdomain = (1,10,1,10)
  pspace = ParamSpace(pdomain)

  a(μ) = x -> μ[1]*exp(-x[1])
  aμ(μ) = parameterise(a,μ)

  g(μ) = x -> VectorValue(-μ[2]*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
  gμ(μ) = parameterise(g,μ)

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  stiffness(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  res(μ,(u,p),(v,q),dΩ) = stiffness(μ,(u,p),(v,q),dΩ)

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)
  V = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  Q = TestFESpace(Ω,reffe_p;conformity=:H1)
  U = ParamTrialFESpace(V,gμ)
  P = ParamTrialFESpace(Q)
  X = MultiFieldFESpace([U,P];style=BlockMultiFieldStyle())
  Y = MultiFieldFESpace([V,Q];style=BlockMultiFieldStyle())

  energy = BlockNorm((H1(),L2()))
  coupling = DivCoupling()
  state_reduction = SupremizerReduction(coupling,tol,energy;nparams,compression,ncentroids)

  fesolver = build_fesolver(Q,dΩ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,X,Y,domains)

  dir = datadir("diagnostics_stokes_distributed")
  if i_am_main(ranks)
    isdir(dir) && rm(dir;recursive=true)
    mkpath(dir)
  end
  MPI.Initialized() && MPI.Barrier(MPI.COMM_WORLD)

  tols = [1e-1,1e-3,1e-5]
  run_test(dir,rbsolver,feop,tols)

  dgn = rom_diagnostics(dir,rbsolver,feop)
  println(dgn)
end

end
