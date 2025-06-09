"""
    reduced_operator(solver::RBSolver,feop::ParamOperator,args...;kwargs...) -> RBOperator
    reduced_operator(solver::RBSolver,feop::TransientParamOperator,args...;kwargs...) -> TransientRBOperator

Computes a RB operator from the FE operator `feop`
"""
function reduced_operator(
  dir::String,
  solver::RBSolver,
  feop::ParamOperator,
  args...;
  kwargs...)

  fesnaps,festats = solution_snapshots(solver,feop,args...;kwargs...)
  rbop = reduced_operator(solver,feop,fesnaps)
  save(dir,fesnaps)
  save(dir,rbop)
  rbop
end

function reduced_operator(
  solver::RBSolver,
  feop::ParamOperator,
  args...;
  kwargs...)

  fesnaps,festats = solution_snapshots(solver,feop,args...;kwargs...)
  reduced_operator(solver,feop,fesnaps)
end

function reduced_operator(
  solver::RBSolver,
  feop::ParamOperator,
  s::AbstractSnapshots)

  red_trial,red_test = reduced_spaces(solver,feop,s)
  reduced_operator(solver,feop,red_trial,red_test,s)
end

function reduced_operator(
  solver::RBSolver,
  feop::ParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots
  )

  red_lhs,red_rhs = reduced_weak_form(solver,feop,red_trial,red_test,s)
  RBOperator(feop,red_trial,red_test,red_lhs,red_rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots
  )

  red_op_lin = reduced_operator(solver,get_linear_operator(op),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),red_trial,red_test,s)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

"""
    abstract type RBOperator{O} <: ParamOperator{O,SplitDomains} end

Type representing reduced algebraic operators used within a reduced order modelling
framework in steady applications. A RBOperator should contain the following information:

- a reduced test and trial space, computed according to [`reduced_spaces`](@ref)
- a hyper-reduced residual and jacobian, computed according to [`reduced_weak_form`](@ref)

Subtypes:

- [`GenericRBOperator`](@ref)
- [`LinearNonlinearRBOperator`](@ref)
"""
abstract type RBOperator{O} <: ParamOperator{O,SplitDomains} end

"""
    struct GenericRBOperator{O,A} <: RBOperator{O}
      op::ParamOperator{O}
      trial::RBSpace
      test::RBSpace
      lhs::A
      rhs::AffineContribution
    end

Fields:

- `op`: underlying high dimensional FE operator
- `trial`: reduced trial space
- `test`: reduced trial space
- `lhs`: hyper-reduced left hand side
- `rhs`: hyper-reduced right hand side
"""
struct GenericRBOperator{O,A} <: RBOperator{O}
  op::ParamOperator{O}
  trial::RBSpace
  test::RBSpace
  lhs::A
  rhs::AffineContribution
end

function RBOperator(
  op::ParamOperator,
  trial::RBSpace,
  test::RBSpace,
  lhs::AffineContribution,
  rhs::AffineContribution)

  trians_rhs = get_domains(rhs)
  trians_lhs = get_domains(lhs)
  op′ = change_domains(op,trians_rhs,trians_lhs)
  GenericRBOperator(op′,trial,test,lhs,rhs)
end

Utils.get_fe_operator(op::GenericRBOperator) = op.op
FESpaces.get_trial(op::GenericRBOperator) = op.trial
FESpaces.get_test(op::GenericRBOperator) = op.test
get_lhs(op::GenericRBOperator) = op.lhs
get_rhs(op::GenericRBOperator) = op.lhs

function Algebra.allocate_residual(op::GenericRBOperator,u::AbstractVector)
  allocate_hypred_cache(op.rhs)
end

function Algebra.allocate_jacobian(op::GenericRBOperator,u::AbstractVector)
  allocate_hypred_cache(op.lhs)
end

function Algebra.residual!(b::HRArray,op::GenericRBOperator,u::AbstractVector)
  fill!(b,zero(eltype(b)))

  trial = get_trial(op.op)
  test = get_test(op.op)
  uh = EvaluationFunction(trial,u)
  v = get_fe_basis(test)

  trian_res = get_domains_res(op.op)
  res = get_res(op.op)
  dc = res(uh,v)

  for strian in trian_res
    b_strian = b.fecache[strian]
    rhs_strian = op.rhs[strian]
    vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian)
    assemble_hr_vector_add!(b_strian,vecdata...)
  end

  inv_project!(b,op.rhs)
end

function Algebra.jacobian!(A::HRArray,op::GenericRBOperator,u::AbstractVector)
  fill!(A,zero(eltype(A)))

  trial = get_trial(op.op)
  test = get_test(op.op)
  uh = EvaluationFunction(trial,u)
  du = get_trial_fe_basis(trial)
  v = get_fe_basis(test)

  trian_jac = get_domains_jac(op.op)
  jac = get_jac(op.op)
  dc = jac(r,uh,du,v)

  for strian in trian_jac
    A_strian = A.fecache[strian]
    lhs_strian = op.lhs[strian]
    matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian)
    assemble_hr_matrix_add!(A_strian,matdata...)
  end

  inv_project!(A,op.lhs)
end

function Algebra.allocate_residual(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  allocate_hypred_cache(op.rhs,r)
end

function Algebra.allocate_jacobian(
  op::GenericRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  allocate_hypred_cache(op.lhs,r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op.op)
  v = get_fe_basis(test)

  trian_res = get_domains_res(op.op)
  res = get_res(op.op)
  dc = res(r,uh,v)

  for strian in trian_res
    b_strian = b.fecache[strian]
    rhs_strian = op.rhs[strian]
    vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian)
    assemble_hr_vector_add!(b_strian,vecdata...)
  end

  inv_project!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  fill!(A,zero(eltype(A)))

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op.op)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)

  trian_jac = get_domains_jac(op.op)
  jac = get_jac(op.op)
  dc = jac(r,uh,du,v)

  for strian in trian_jac
    A_strian = A.fecache[strian]
    lhs_strian = op.lhs[strian]
    matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian)
    assemble_hr_matrix_add!(A_strian,matdata...)
  end

  inv_project!(A,op.lhs)
end

struct InterpRBOperator{O,A} <: RBOperator{O}
  op::ParamOperator{O}
  trial::RBSpace
  test::RBSpace
  lhs::A
  rhs::AbstractHRProjection
end

function RBOperator(
  op::ParamOperator,trial::RBSpace,test::RBSpace,lhs,rhs::AbstractHRProjection)

  InterpRBOperator(op,trial,test,lhs,rhs)
end

Utils.get_fe_operator(op::InterpRBOperator) = op.op
FESpaces.get_trial(op::InterpRBOperator) = op.trial
FESpaces.get_test(op::InterpRBOperator) = op.test
get_lhs(op::InterpRBOperator) = op.lhs
get_rhs(op::InterpRBOperator) = op.lhs

function Algebra.allocate_residual(op::InterpRBOperator,u::AbstractVector)
  allocate_hypred_cache(op.rhs)
end

function Algebra.allocate_jacobian(op::InterpRBOperator,u::AbstractVector)
  allocate_hypred_cache(op.lhs)
end

function Algebra.residual!(b::HRArray,op::InterpRBOperator,u::AbstractVector)
  fill!(b,zero(eltype(b)))
  inv_project!(b,op.rhs)
end

function Algebra.jacobian!(A::HRArray,op::InterpRBOperator,u::AbstractVector)
  fill!(A,zero(eltype(A)))
  inv_project!(A,op.lhs)
end

function Algebra.allocate_residual(
  op::InterpRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  allocate_hypred_cache(op.rhs,r)
end

function Algebra.allocate_jacobian(
  op::InterpRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  allocate_hypred_cache(op.lhs,r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::InterpRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  fill!(b,zero(eltype(b)))
  inv_project!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::InterpRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  fill!(A,zero(eltype(A)))
  inv_project!(A,op.lhs,r)
end

struct AlgebraicRBOperator{O} <: RBOperator{O}
  op::ParamOperator{O}
  trial::RBSpace
  test::RBSpace
  lhs::AbstractHRArray
  rhs::AbstractHRArray
end

function RBOperator(
  op::ParamOperator,trial::RBSpace,test::RBSpace,lhs::AbstractHRArray,rhs::AbstractHRArray)

  AlgebraicRBOperator(op,trial,test,lhs,rhs)
end

Utils.get_fe_operator(op::AlgebraicRBOperator) = op.op
FESpaces.get_trial(op::AlgebraicRBOperator) = op.trial
FESpaces.get_test(op::AlgebraicRBOperator) = op.test
get_lhs(op::AlgebraicRBOperator) = op.lhs
get_rhs(op::AlgebraicRBOperator) = op.lhs

function Algebra.allocate_residual(op::AlgebraicRBOperator,u::AbstractVector)
  copy(op.rhs)
end

function Algebra.allocate_jacobian(op::AlgebraicRBOperator,u::AbstractVector)
  copy(op.lhs)
end

function Algebra.residual!(b::HRArray,op::AlgebraicRBOperator,u::AbstractVector)
  copyto!(b,op.rhs)
end

function Algebra.jacobian!(A::HRArray,op::AlgebraicRBOperator,u::AbstractVector)
  copyto!(A,op.lhs)
end

function Algebra.allocate_residual(
  op::AlgebraicRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  copy(op.rhs)
end

function Algebra.allocate_jacobian(
  op::AlgebraicRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  copy(op.lhs)
end

function Algebra.residual!(
  b::HRParamArray,
  op::AlgebraicRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  copyto!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::AlgebraicRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  copyto!(A,op.lhs)
end

struct LocalRBOperator{O} <: RBOperator{O}
  operators::Vector{<:RBOperator{O}}
  k::KmeansResult
end

function RBOperator(
  op::ParamOperator,trial::RBSpace,test::RBSpace,lhs::LocalProjection,rhs::LocalProjection)

  operators = map(local_values(trial),local_values(test),local_values(lhs),local_values(rhs)
  ) do trial,test,lhs,rhs
    RBOperator(op,trial,test,lhs,rhs)
  end
  k = get_clusters(lhs)
  LocalRBOperator(operators,k)
end

function get_local(op::LocalRBOperator,μ::AbstractVector)
  lab = get_label(op.k,μ)
  op.operators[lab]
end

function Algebra.solve(
  solver::RBSolver,
  op::LocalRBOperator,
  r::Realization)

  fesolver = get_fe_solver(solver)

  t = @timed x̂vec = map(r) do μ
    opμ = get_local(op,μ)
    x̂,stats = solve(solver,opμ,Realization(μ))
    testitem(x̂)
  end
  x̂ = GenericParamVector(x̂vec)
  stats = CostTracker(t,nruns=num_params(r),name="RB")
  return (x̂,stats)
end

"""
    struct LinearNonlinearRBOperator <: RBOperator{LinearNonlinearParamEq}
      op_linear::RBOperator
      op_nonlinear::RBOperator
    end

Extends the concept of [`GenericRBOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications
"""
struct LinearNonlinearRBOperator <: RBOperator{LinearNonlinearParamEq}
  op_linear::RBOperator
  op_nonlinear::RBOperator
end

ParamAlgebra.get_linear_operator(op::LinearNonlinearRBOperator) = op.op_linear
ParamAlgebra.get_nonlinear_operator(op::LinearNonlinearRBOperator) = op.op_nonlinear
FESpaces.get_trial(op::LinearNonlinearRBOperator) = get_trial(get_nonlinear_operator(op))
FESpaces.get_test(op::LinearNonlinearRBOperator) = get_test(get_nonlinear_operator(op))

Utils.get_fe_operator(op::LinearNonlinearRBOperator) = get_fe_operator(get_nonlinear_operator(op))

function ParamAlgebra.allocate_paramcache(op::LinearNonlinearRBOperator,μ::AbstractRealization)
  op_nlin = get_nonlinear_operator(op)
  allocate_paramcache(op_nlin,μ)
end

function ParamAlgebra.allocate_systemcache(op::LinearNonlinearRBOperator,u::AbstractVector)
  op_nlin = get_nonlinear_operator(op)
  allocate_systemcache(op_nlin,u)
end

function ParamAlgebra.update_paramcache!(
  paramcache::AbstractParamCache,
  op::LinearNonlinearRBOperator,
  μ::AbstractRealization)

  op_nlin = get_nonlinear_operator(op)
  update_paramcache!(paramcache,op_nlin,μ)
end

function ParamDataStructures.parameterize(op::LinearNonlinearRBOperator,μ::AbstractRealization)
  op_lin = parameterize(get_linear_operator(op),μ)
  op_nlin = parameterize(get_nonlinear_operator(op),μ)
  syscache_lin = allocate_systemcache(op_lin)
  LinNonlinParamOperator(op_lin,op_nlin,syscache_lin)
end

# utils

function to_snapshots(rbop::RBOperator,x̂::AbstractParamVector,r::AbstractRealization)
  to_snapshots(get_trial(rbop),x̂,r)
end

function to_snapshots(rbop::LocalRBOperator,x̂::AbstractParamVector,r::AbstractRealization)
  xvec = map(x̂,r) do x̂,μ
    opμ = get_local(rbop,μ)
    trial = get_trial(opμ)
    inv_project(trial,x̂)
  end
  @check all(size(x) == size(first(xvec)) for x in xvec)
  i = get_global_dof_map(rbop)
  Snapshots(stack(xvec),i,r)
end

get_global_dof_map(rbop::RBOperator) = get_global_dof_map(get_trial(rbop))
get_global_dof_map(rbop::LocalRBOperator) = get_global_dof_map(first(rbop.operators))
