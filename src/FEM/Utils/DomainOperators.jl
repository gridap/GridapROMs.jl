"""
"""
abstract type DomainOperator{O<:OperatorType,T<:TriangulationStyle} <: NonlinearOperator end

"""
"""
const JointOperator{O<:OperatorType} = DomainOperator{O,JointDomains}

"""
"""
const SplitOperator{O<:OperatorType} = DomainOperator{O,SplitDomains}

get_fe_operator(op::DomainOperator) = @abstractmethod
FESpaces.get_test(op::DomainOperator) = get_test(get_fe_operator(op))
FESpaces.get_trial(op::DomainOperator) = get_trial(get_fe_operator(op))
ODEs.get_res(op::DomainOperator) = get_res(get_fe_operator(op))
get_jac(op::DomainOperator) = get_jac(get_fe_operator(op))
ODEs.get_assembler(op::DomainOperator) = get_assembler(get_fe_operator(op))

CellData.get_domains(op::DomainOperator) = get_domains(get_fe_operator(op))
get_domains_res(op::DomainOperator) = get_domains_res(get_domains(op))
get_domains_jac(op::DomainOperator) = get_domains_jac(get_domains(op))

set_domains(op::DomainOperator,args...) = get_algebraic_operator(set_domains(get_fe_operator(op),args...))
change_domains(op::DomainOperator,args...) = get_algebraic_operator(change_domains(get_fe_operator(op),args...))

function Algebra.zero_initial_guess(op::DomainOperator)
  trial = get_trial(op)
  zero_free_values(trial)
end

# algebra: dispatch on split/joint

function Algebra.allocate_residual(op::JointOperator,u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh = EvaluationFunction(trial,u)
  v = get_fe_basis(test)
  res = get_res(op)
  vecdata = collect_cell_vector(test,res(uh,v))

  assem = get_assembler(op)
  allocate_vector(assem,vecdata)
end

function Algebra.residual!(b::AbstractVector,op::JointOperator,u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh = EvaluationFunction(trial,u)
  v = get_fe_basis(test)
  res = get_res(op)
  vecdata = collect_cell_vector(test,res(uh,v))

  assem = get_assembler(op)
  assemble_vector!(b,assem,vecdata)
  b
end

function Algebra.residual!(b::AbstractVector,op::JointOperator{LinearEq},u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh0 = zero(trial)
  v = get_fe_basis(test)
  res = get_res(op)
  vecdata = collect_cell_vector(test,res(uh0,v))

  assem = get_assembler(op)
  assemble_vector!(b,assem,vecdata)
  rmul!(b,-1)
  b
end

function Algebra.allocate_jacobian(op::JointOperator,u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh = EvaluationFunction(trial,u)
  du = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  jac = get_jac(op)
  matdata = collect_cell_matrix(trial,test,jac(uh,du,v))

  assem = get_assembler(op)
  allocate_matrix(assem,matdata)
end

function Algebra.jacobian!(A::AbstractMatrix,op::JointOperator,u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh = EvaluationFunction(trial,u)
  du = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  jac = get_jac(op)
  matdata = collect_cell_matrix(trial,test,jac(uh,du,v))

  assem = get_assembler(op)
  assemble_matrix!(A,assem,matdata)
end

function Algebra.allocate_residual(op::SplitOperator,u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh = EvaluationFunction(trial,u)
  v = get_fe_basis(test)
  res = get_res(op)
  trian_res = get_domains_res(op)
  dc = res(uh,v)

  assem = get_assembler(op)

  contribution(trian_res) do trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    allocate_vector(assem,vecdata)
  end
end

function Algebra.residual!(b::Contribution,op::SplitOperator,u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh = EvaluationFunction(trial,u)
  v = get_fe_basis(test)
  res = get_res(op)
  trian_res = get_domains_res(op)
  dc = res(uh,v)

  assem = get_assembler(op)

  map(b.values,trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector!(values,assem,vecdata)
  end
  b
end

function Algebra.residual!(b::Contribution,op::SplitOperator{LinearEq},u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh0 = zero(trial)
  v = get_fe_basis(test)
  res = get_res(op)
  trian_res = get_domains_res(op)
  dc = res(uh0,v)

  assem = get_assembler(op)

  map(b.values,trian_res) do values,trian
    vecdata = collect_cell_vector_for_trian(test,dc,trian)
    assemble_vector!(values,assem,vecdata)
  end
  b
end

function Algebra.allocate_jacobian(op::SplitOperator,u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh = EvaluationFunction(trial,u)
  du = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  jac = get_jac(op)
  trian_jac = get_domains_jac(op)
  dc = jac(uh,du,v)

  assem = get_assembler(op)

  contribution(trian_jac) do trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    allocate_matrix(assem,matdata)
  end
end

function Algebra.jacobian!(A::Contribution,op::SplitOperator,u::AbstractVector)
  trial = get_trial(op)
  test = get_test(op)

  uh = EvaluationFunction(trial,u)
  du = get_trial_fe_basis(trial)
  v = get_fe_basis(test)
  jac = get_jac(op)
  trian_jac = get_domains_jac(op)
  dc = jac(uh,du,v)

  assem = get_assembler(op)

  map(A.values,trian_jac) do values,trian
    matdata = collect_cell_matrix_for_trian(trial,test,dc,trian)
    assemble_matrix_add!(values,assem,matdata)
  end
  A
end

# concrete implementation

struct GenericDomainOperator{O<:OperatorType,T<:TriangulationStyle} <: DomainOperator{O,T}
  feop::FEDomainOperator{O,T}
end

get_fe_operator(op::GenericDomainOperator) = op.feop

function LinearDomainOperator(args...;kwargs...)
  feop = LinearFEOperator(args...;kwargs...)
  GenericDomainOperator(feop)
end

function DomainOperator(args...;kwargs...)
  feop = FEOperator(args...;kwargs...)
  GenericDomainOperator(feop)
end

# utils

"""
    function collect_cell_matrix_for_trian(
      trial::FESpace,
      test::FESpace,
      a::DomainContribution,
      strian::Triangulation
      ) -> Tuple{Vector{<:Any},Vector{<:Any},Vector{<:Any}}

Computes the cell-wise data needed to assemble a global sparse matrix for a given
input triangulation `strian`
"""
function collect_cell_matrix_for_trian(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  rows = get_cell_dof_ids(test,trian)
  cols = get_cell_dof_ids(trial,trian)
  Any[cell_mat_rc],Any[rows],Any[cols]
end

"""
    function collect_cell_vector_for_trian(
      test::FESpace,
      a::DomainContribution,
      strian::Triangulation
      ) -> Tuple{Vector{<:Any},Vector{<:Any}}

Computes the cell-wise data needed to assemble a global vector for a given
input triangulation `strian`
"""
function collect_cell_vector_for_trian(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  rows = get_cell_dof_ids(test,trian)
  Any[cell_vec_r],Any[rows]
end

for T in (:LinearEq,:NonlinearEq)
  @eval begin
    function Algebra.solve!(
      x::AbstractVector,
      solver::LinearSolver,
      op::SplitOperator{$T}
      )

      solve!(x,solver,set_domains(op))
    end
  end
end

function Algebra.solve!(
  x::AbstractVector,
  solver::LinearSolver,
  op::DomainOperator{LinearEq}
  )

  u = zero_initial_guess(op)
  A = jacobian(op,u)
  b = residual(op,u)
  solve!(x,solver,A,b)
end
