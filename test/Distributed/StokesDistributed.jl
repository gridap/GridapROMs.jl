module StokesDistributed

using DrWatson
using LinearAlgebra
using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using GridapDistributed
using GridapPETSc
using GridapROMs
using GridapSolvers
using PartitionedArrays
using Test

using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady

using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers

tol=1e-4
nparams=12
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
ncentroids=2
hypred_strategy=:deim

domain = (0,1,0,1)
partition = (8,8)

pdomain = (1,10,1,10)
pspace = ParamSpace(pdomain)

a(μ) = x -> μ[1]*exp(-x[1])
aμ(μ) = parameterise(a,μ)

g(μ) = x -> VectorValue(-μ[2]*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
gμ(μ) = parameterise(g,μ)

order = 2

energy = BlockNorm((H1(),L2()))
coupling = DivCoupling()
state_reduction = SupremizerReduction(coupling,tol,energy;nparams,ncentroids)

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

function build_rbsolver(Q,dΩ,ranks)
  solver_u = ASMSolver()
  solver_p = PressureSolver()

  blocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(p*q)dΩ,Q,Q)]
  prec = BlockTriangularSolver(blocks,[solver_u,solver_p])
  fesolver = FGMRESSolver(30,prec;rtol=1.e-6,verbose=false)

  RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)
end

function build_spaces(Ω)
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)

  V = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  Q = TestFESpace(Ω,reffe_p;conformity=:H1)
  U = ParamTrialFESpace(V,gμ)
  P = ParamTrialFESpace(Q)

  X = MultiFieldFESpace([U,P];style=BlockMultiFieldStyle())
  Y = MultiFieldFESpace([V,Q];style=BlockMultiFieldStyle())
  return X,Y,Q
end

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks,parts,domain,partition)

  Ω = Triangulation(model)
  degree = 2*order
  dΩ = Measure(Ω,degree)

  stiffness(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  res(μ,(u,p),(v,q),dΩ) = stiffness(μ,(u,p),(v,q),dΩ)

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  X,Y,Q = build_spaces(Ω)
  feop = LinearParamOperator(res,stiffness,pspace,X,Y,domains)

  rbsolver = build_rbsolver(Q,dΩ,ranks)

  fesnaps, = solution_snapshots(rbsolver,feop)
  println("diagnostic | fesnaps built ok")
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  println("diagnostic | rbop (with supremizer enrichment) built ok")

  # 2) velocity basis H1-orthogonality after supremizer enrichment
  μ = get_realisation(fesnaps)
  trial = get_trial(rbop)(μ)
  rsub = RBSteady.get_reduced_subspace(trial)
  a_primal = rsub[1]
  Φ = RBSteady.get_basis(a_primal)
  Xp = RBSteady.get_norm_matrix(a_primal)
  G = Φ'*(Xp*Φ)
  n = size(G,1)
  orth_err = maximum(abs.(G .- I(n)))
  println("diagnostic | velocity basis H1-orthogonality max|Φ'HΦ - I|: ", orth_err)

  # 1) MDEIM approximation quality (HR errors for residual/jacobian)
  res = residual_snapshots(rbsolver,feop,fesnaps)
  jac = jacobian_snapshots(rbsolver,feop,fesnaps)
  err_res,err_jac = RBSteady.hr_error(rbsolver,rbop,res,jac,fesnaps)
  println("diagnostic | hr error residual (per trian): ", err_res)
  println("diagnostic | hr error jacobian (per trian): ", err_jac)

  perr = RBSteady.projection_error(rbsolver,rbop,fesnaps)
  println("diagnostic | projection error (basis + project/inv_project, no HR): ", perr)
end

end