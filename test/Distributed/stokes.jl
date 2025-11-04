using DrWatson

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField

using GridapDistributed

using GridapPETSc

using GridapROMs
using GridapROMs.ParamAlgebra

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers

using PartitionedArrays

function petsc_asm_setup(ksp)
  rtol = PetscScalar(1.e-9)
  atol = GridapPETSc.PETSC.PETSC_DEFAULT
  dtol = GridapPETSc.PETSC.PETSC_DEFAULT
  maxits = GridapPETSc.PETSC.PETSC_DEFAULT

  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPCG)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCASM)
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

function petsc_asm_solve(solver,op,μ)
  trial = get_trial(op)
  y = zero_free_values(trial(μ))
  nlop = parameterize(op,μ)
  syscache = allocate_systemcache(nlop,y)
  A = get_matrix(syscache)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,solver,nlop,syscache)
  return x
end

function build_snapshots(solver,op,μ)
  x = petsc_asm_solve(solver,op,μ)
  i = get_dof_map(op)
  return Snapshots(x,i,μ)
end

ASMSolver() = PETScLinearSolver(petsc_asm_setup)

function add_labels!(labels)
  add_tag_from_tags!(labels,"top",[6])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,5,7,8])
end

np = (2,2)
nc = (4,4)
domain = (0,1,0,1)
parts = with_mpi() do distribute
  distribute(LinearIndices((prod(np),)))
end

model = CartesianDiscreteModel(parts,np,domain,nc)
map(add_labels!,local_views(get_face_labeling(model)))

function main(parts,model)
  pspace = ParamSpace((0,1,0,1))

  # Weak formulation
  f(μ) = x -> VectorValue(μ[1],1.0)
  fμ(μ) = parameterize(f,μ)

  g(μ) = x -> VectorValue(μ[2],0.0)
  gμ(μ) = parameterize(g,μ)
  g0(μ) = x -> VectorValue(0.0,0.0)
  g0μ(μ) = parameterize(g0,μ)

  a(μ,(u,p),(v,q)) = ∫(∇(v)⊙∇(u))dΩ - ∫((∇⋅v)*p)dΩ - ∫((∇⋅u)*q)dΩ
  res(μ,(u,p),(v,q)) = a(μ,(u,p),(v,q)) - ∫(v⋅fμ(μ))dΩ

  # FE spaces
  order = 2
  degree = 2*(order+1)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

  V = TestFESpace(model,reffe_u,dirichlet_tags=["walls","top"])
  U = ParamTrialFESpace(V,[g0μ,gμ])
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
  P = ParamTrialFESpace(Q)

  X = MultiFieldFESpace([U,P];style=BlockMultiFieldStyle())
  Y = MultiFieldFESpace([V,Q];style=BlockMultiFieldStyle())

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  op = LinearParamOperator(res,a,pspace,X,Y)

  solver_u = ASMSolver()
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-10,verbose=i_am_main(parts))

  # Block triangular preconditioner
  blocks = [LinearSystemBlock() LinearSystemBlock();
            LinearSystemBlock() BiformBlock((p,q) -> ∫(p*q)dΩ,Q,Q)]
  Prec = BlockTriangularSolver(blocks,[solver_u,solver_p])
  solver = FGMRESSolver(30,Prec;rtol=1.e-8,verbose=i_am_main(parts))

  μ = realization(pspace;nparams=2)
  build_snapshots(solver,op,μ)
end

petsc_options = "-ksp_monitor -ksp_error_if_not_converged true -ksp_converged_reason"

GridapPETSc.with(;args=split(petsc_options)) do
  block_part_snaps = main(parts,model)
  part_block_snaps = local_views(block_part_snaps)
  map(linear_indices(part_block_snaps),part_block_snaps) do i,s
    save(pwd(),s;label="part_$i")
  end
  GridapPETSc.gridap_petsc_gc()
end
