using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using GridapDistributed
using GridapPETSc
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.Distributed
using GridapROMs.RBSteady
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers
using PartitionedArrays
using Test

tol=1e-4
nparams=2
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
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=false)

  blocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(p*q)dΩ,Q,Q)]
  prec = BlockTriangularSolver(blocks,[solver_u,solver_p])
  fesolver = FGMRESSolver(30,prec;rtol=1.e-6,verbose=false)

  RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)
end

# NOTE: FE spaces are built from `model` (not `Triangulation(model)`), and the
# `MultiFieldFESpace`s use `BlockMultiFieldStyle()` -- both required to avoid
# a severe Julia type-inference blowup that otherwise hits `TestFESpace` on a
# distributed triangulation (confirmed empirically: dropping either one
# reintroduces a compile-time hang that can run indefinitely).
function build_spaces(model)
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1)

  V = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  Q = TestFESpace(model,reffe_p;conformity=:H1)
  U = ParamTrialFESpace(V,gμ)
  P = ParamTrialFESpace(Q)

  X = MultiFieldFESpace([U,P];style=BlockMultiFieldStyle())
  Y = MultiFieldFESpace([V,Q];style=BlockMultiFieldStyle())
  return X,Y,Q
end

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  model = CartesianDiscreteModel(ranks,parts,domain,partition)

  # `dΩ` is captured as a closure (not passed as an explicit weak-form
  # argument via `FEDomains`) since there is only one triangulation -- this
  # also avoids the compile-time blowup mentioned above.
  Ω = Triangulation(model)
  degree = 2*order
  dΩ = Measure(Ω,degree)

  stiffness(μ,(u,p),(v,q)) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
  res(μ,(u,p),(v,q)) = stiffness(μ,(u,p),(v,q))

  X,Y,Q = build_spaces(model)
  feop = LinearParamOperator(res,stiffness,pspace,X,Y)

  rbsolver = build_rbsolver(Q,dΩ,ranks)

  fesnaps, = solution_snapshots(rbsolver,feop)
  println("diagnostic | fesnaps built ok")
  rbop = reduced_operator(rbsolver,feop,fesnaps)
  println("diagnostic | rbop (with supremizer enrichment) built ok")

  perr = RBSteady.projection_error(rbsolver,rbop,fesnaps)
  println("diagnostic | projection error (basis + project/inv_project, no HR): ", perr)
end

petsc_options = "-ksp_error_if_not_converged true"

with_debug() do distribute
  GridapPETSc.with(;args=split(petsc_options)) do
    main(distribute,(2,2))
    GridapPETSc.gridap_petsc_gc()
  end
end