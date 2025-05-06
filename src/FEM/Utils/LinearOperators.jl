"""
"""
struct LinearOperator <: NonlinearOperator
  trial::FESpace
  test::FESpace
  matrix::AbstractMatrix
  vector::AbstractVector
  function LinearOperator(trial::FESpace,test::FESpace,matrix::AbstractMatrix,vector::AbstractVector)
    @assert num_free_dofs(trial) == size(matrix,2) "Incompatible trial space and matrix"
    @assert num_free_dofs(test) == size(matrix,1) "Incompatible test space and matrix"
    new(trial,test,matrix,vector)
  end
end

function LinearOperator(
  weakform::Function,trial::FESpace,test::FESpace,assem::Assembler)
  @assert ! isa(test,TrialFESpace) """\n
  It is not allowed to build an LinearOperator with a test space of type TrialFESpace.

  Make sure that you are writing first the trial space and then the test space when
  building an LinearOperator or a FEOperator.
  """

  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)

  uhd = zero(trial)
  matcontribs,veccontribs = weakform(u,v)
  data = collect_cell_matrix_and_vector(trial,test,matcontribs,veccontribs,uhd)
  A,b = assemble_matrix_and_vector(assem,data)
  LinearOperator(trial,test,A,b)
end

function LinearOperator(weakform::Function,args...)
  assem = SparseMatrixAssembler(args...)
  trial,test, = args
  LinearOperator(weakform,trial,test,assem)
end

function LinearOperator(a::Function,ℓ::Function,args...)
  LinearOperator(args...) do u,v
    a(u,v),ℓ(v)
  end
end

FESpaces.get_test(op::LinearOperator) = op.test
FESpaces.get_trial(op::LinearOperator) = op.trial
Algebra.get_matrix(op::LinearOperator) = op.matrix
Algebra.get_vector(op::LinearOperator) = op.vector

function Algebra.residual!(b::AbstractVector,op::LinearOperator,x::AbstractVector)
  mul!(b,op.matrix,x)
  b .-= op.vector
  b
end

function Algebra.jacobian!(A::AbstractMatrix,op::LinearOperator,x::AbstractVector)
  copy_entries!(A,op.matrix)
  A
end

function Algebra.jacobian(op::LinearOperator,x::AbstractVector)
  op.matrix
end

function Algebra.zero_initial_guess(op::LinearOperator)
  x = allocate_in_domain(typeof(op.vector),op.matrix)
  fill!(x,zero(eltype(x)))
  x
end

function Algebra.allocate_residual(op::LinearOperator,x::AbstractVector)
  allocate_in_range(typeof(op.vector),op.matrix)
end

function Algebra.allocate_jacobian(op::LinearOperator,x::AbstractVector)
  op.matrix
end

function Algebra.solve!(x::AbstractVector,ls::LinearSolver,op::LinearOperator,cache::Nothing)
  A = op.matrix
  b = op.vector
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A)
  solve!(x,ns,b)
  ns
end

function Algebra.solve!(x::AbstractVector,ls::LinearSolver,op::LinearOperator,cache)
  newmatrix = true
  solve!(x,ls,op,cache,newmatrix)
  cache
end

function Algebra.solve!(
  x::AbstractVector,ls::LinearSolver,op::LinearOperator,cache,newmatrix::Bool)
  A = op.matrix
  b = op.vector
  ns = cache
  if newmatrix
    numerical_setup!(ns,A)
  end
  solve!(x,ns,b)
  cache
end

function Algebra.solve!(
  x::AbstractVector,ls::LinearSolver,op::LinearOperator,cache::Nothing,newmatrix::Bool)
  solve!(x,ls,op,cache)
end
