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

function reduced_operator(rbsolver::RBSolver,feop::ParamOperator,s,jac,res)
  red_trial,red_test = reduced_spaces(rbsolver,feop,s)
  reduced_operator(rbsolver,feop,red_trial,red_test,jac,res)
end

function reduced_operator(
  rbsolver::RBSolver,
  feop::ParamOperator,
  red_trial::RBSpace,
  red_test::RBSpace,
  jac,
  res
  )

  jac_red = get_jacobian_reduction(rbsolver)
  red_lhs = reduced_jacobian(jac_red,red_trial,red_test,jac)
  res_red = get_residual_reduction(rbsolver)
  red_rhs = reduced_residual(res_red,red_test,res)
  RBOperator(feop,red_trial,red_test,red_lhs,red_rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  red_trial::RBSpace,
  red_test::RBSpace,
  jac,
  res
  )

  jac_lin,jac_nlin = jac
  res_lin,res_nlin = res
  red_op_lin = reduced_operator(solver,get_linear_operator(op),red_trial,red_test,jac_lin,res_lin)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(op),red_trial,red_test,jac_nlin,res_nlin)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

"""
    abstract type RBOperator{O,T} <: ParamOperator{O,T} end

Type representing reduced algebraic operators used within a reduced order modelling
framework in steady applications. A RBOperator should contain the following information:

- a reduced test and trial space, computed according to [`reduced_spaces`](@ref)
- a hyper-reduced residual and jacobian, computed according to [`reduced_weak_form`](@ref)

Subtypes:

- [`GenericRBOperator`](@ref)
- [`LinearNonlinearRBOperator`](@ref)
"""
abstract type RBOperator{O,T} <: ParamOperator{O,T} end

const JointRBOperator{O} = RBOperator{O,JointDomains}
const SplitRBOperator{O} = RBOperator{O,SplitDomains}

Utils.get_fe_operator(op::RBOperator) = @abstractmethod
FESpaces.get_trial(op::RBOperator) = @abstractmethod
FESpaces.get_test(op::RBOperator) = @abstractmethod
get_lhs(op::RBOperator) = @abstractmethod
get_rhs(op::RBOperator) = @abstractmethod

function Algebra.allocate_residual(
  op::RBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  allocate_hypred_cache(get_rhs(op),r)
end

function Algebra.allocate_jacobian(
  op::RBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  allocate_hypred_cache(get_lhs(op),r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::SplitRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)

  trian_res = get_domains_res(op)
  rhs = get_rhs(op)
  res = get_res(op)
  dc = res(r,uh,v)

  for strian in trian_res
    b_strian = b.fecache[strian]
    rhs_strian = get_interpolation(rhs[strian])
    vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian)
    assemble_hr_array_add!(b_strian,vecdata...)
  end

  interpolate!(b,rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::SplitRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  fill!(A,zero(eltype(A)))

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)

  trian_jac = get_domains_jac(op)
  lhs = get_lhs(op)
  jac = get_jac(op)
  dc = jac(r,uh,du,v)

  for strian in trian_jac
    A_strian = A.fecache[strian]
    lhs_strian = get_interpolation(lhs[strian])
    matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian)
    assemble_hr_array_add!(A_strian,matdata...)
  end

  interpolate!(A,lhs)
end

function Algebra.residual!(
  b::HRParamArray,
  op::JointRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)

  rhs = get_rhs(op)
  bg_trian = first(get_domains(rhs))
  res = get_res(op)
  dc = res(r,uh,v)

  for strian in get_domains(dc)
    rhs_strian = move_interpolation(rhs[bg_trian],test,strian)
    vecdata = collect_reduced_cell_hr_vector(test,dc,strian,rhs_strian)
    assemble_hr_array_add!(b.fecache[bg_trian],vecdata...)
  end

  interpolate!(b,rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::JointRBOperator,
  r::Realization,
  u::AbstractVector,
  paramcache)

  fill!(A,zero(eltype(A)))

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)

  lhs = get_lhs(op)
  bg_trian = first(get_domains(lhs))
  jac = get_jac(op)
  dc = jac(r,uh,du,v)

  for strian in get_domains(dc)
    lhs_strian = move_interpolation(lhs[bg_trian],trial,test,strian)
    matdata = collect_reduced_cell_hr_matrix(trial,test,dc,strian,lhs_strian)
    assemble_hr_array_add!(A.fecache[bg_trian],matdata...)
  end

  interpolate!(A,lhs)
end

function change_operator(op::JointRBOperator,op′::ParamOperator)
  rhs,lhs = get_rhs(op),get_lhs(op)
  RBOperator(op′,op.trial,op.test,lhs,rhs)
end

function change_operator(op::SplitRBOperator,op′::ParamOperator)
  rhs,lhs = get_rhs(op),get_lhs(op)
  trians_rhs′ = change_triangulation(get_domains_res(op′),get_domains(rhs))
  trians_lhs′ = change_triangulation(get_domains_jac(op′),get_domains(lhs))
  rhs′ = change_domains(rhs,trians_rhs′)
  lhs′ = change_domains(lhs,trians_lhs′)
  RBOperator(op′,op.trial,op.test,lhs′,rhs′)
end

"""
    struct GenericRBOperator{O,T,A,B} <: RBOperator{O,T}
      op::ParamOperator{O,T}
      trial::RBSpace
      test::RBSpace
      lhs::A
      rhs::B
    end

Fields:

- `op`: underlying high dimensional FE operator
- `trial`: reduced trial space
- `test`: reduced test space
- `lhs`: hyper-reduced left hand side
- `rhs`: hyper-reduced right hand side
"""
struct GenericRBOperator{O,T,A,B} <: RBOperator{O,T}
  op::ParamOperator{O,T}
  trial::RBSpace
  test::RBSpace
  lhs::A
  rhs::B
end

function RBOperator(
  op::SplitParamOperator,
  trial::RBSpace,
  test::RBSpace,
  lhs::AffineContribution,
  rhs::AffineContribution)

  trians_rhs = get_domains(rhs)
  trians_lhs = get_domains(lhs)
  op′ = change_domains(op,trians_rhs,trians_lhs)
  GenericRBOperator(op′,trial,test,lhs,rhs)
end

function RBOperator(
  op::JointParamOperator,
  trial::RBSpace,
  test::RBSpace,
  lhs::AffineContribution,
  rhs::AffineContribution)

  GenericRBOperator(op,trial,test,lhs,rhs)
end

Utils.get_fe_operator(op::GenericRBOperator) = op.op
FESpaces.get_trial(op::GenericRBOperator) = op.trial
FESpaces.get_test(op::GenericRBOperator) = op.test
get_lhs(op::GenericRBOperator) = op.lhs
get_rhs(op::GenericRBOperator) = op.rhs

function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator{O,T,A,<:RBFContribution},
  r::Realization,
  u::AbstractVector,
  paramcache) where {O,T,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator{O,T,<:RBFContribution,B},
  r::Realization,
  u::AbstractVector,
  paramcache) where {O,T,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs,r)
end

"""
    struct LocalRBOperator{O,T,A,B} <: RBOperator{O,T}
      op::ParamOperator{O,T}
      trial::RBSpace
      test::RBSpace
      lhs::A
      rhs::B
    end

Fields:

- `op`: underlying high dimensional FE operator
- `trial`: local reduced trial spaces
- `test`: local reduced test spaces
- `lhs`: local hyper-reduced left hand sides
- `rhs`: local hyper-reduced right hand sides
"""
struct LocalRBOperator{O,T,A,B} <: RBOperator{O,T}
  op::ParamOperator{O,T}
  trial::RBSpace
  test::RBSpace
  lhs::A
  rhs::B
end

Utils.get_fe_operator(op::LocalRBOperator) = op.op
FESpaces.get_trial(op::LocalRBOperator) = op.trial
FESpaces.get_test(op::LocalRBOperator) = op.test
get_lhs(op::LocalRBOperator) = op.lhs
get_rhs(op::LocalRBOperator) = op.rhs

function RBOperator(
  op::ParamOperator,trial::RBSpace,test::RBSpace,lhs::LocalProjection,rhs::LocalProjection)

  LocalRBOperator(op,trial,test,lhs,rhs)
end

function get_local(op::LocalRBOperator,μ::AbstractVector)
  opμ = get_local(op.op,μ)
  trialμ = get_local(op.trial,μ)
  testμ = get_local(op.test,μ)
  lhsμ = get_local(op.lhs,μ)
  rhsμ = get_local(op.rhs,μ)
  RBOperator(opμ,trialμ,testμ,lhsμ,rhsμ)
end

"""
    struct LinearNonlinearRBOperator{A<:RBOperator,B<:RBOperator,T} <: RBOperator{LinearNonlinearParamEq,T}
      op_linear::A
      op_nonlinear::B
    end

Extends the concept of [`GenericRBOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications
"""
struct LinearNonlinearRBOperator{A<:RBOperator,B<:RBOperator,T} <: RBOperator{LinearNonlinearParamEq,T}
  op_linear::A
  op_nonlinear::B
  function LinearNonlinearRBOperator(op_linear::RBOperator{OL,T},op_nonlinear::RBOperator{ON,T}) where {OL,ON,T}
    A = typeof(op_linear)
    B = typeof(op_nonlinear)
    new{A,B,T}(op_linear,op_nonlinear)
  end
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

function change_operator(op::LinearNonlinearRBOperator,op′::LinearNonlinearParamOperator)
  op_lin′ = change_operator(get_linear_operator(op),get_linear_operator(op′))
  op_nlin′ = change_operator(get_nonlinear_operator(op),get_nonlinear_operator(op′))
  LinearNonlinearRBOperator(op_lin′,op_nlin′)
end

const LinearNonlinearGenericRBOperator{T} = LinearNonlinearRBOperator{<:GenericRBOperator,<:GenericRBOperator,T}

const LinearNonlinearLocalRBOperator{T} = LinearNonlinearRBOperator{<:LocalRBOperator,<:LocalRBOperator,T}

const AbstractLocalRBOperator = Union{LocalRBOperator,LinearNonlinearLocalRBOperator}

function get_local(op::LinearNonlinearLocalRBOperator,μ::AbstractVector)
  opμ_linear = get_local(get_linear_operator(op),μ)
  opμ_nlinear = get_local(get_nonlinear_operator(op),μ)
  LinearNonlinearRBOperator(opμ_linear,opμ_nlinear)
end

# local solver

function Algebra.solve(
  solver::RBSolver,
  op::AbstractLocalRBOperator,
  r::Realization)

  t = @timed x̂vec = map(r) do μ
    opμ = get_local(op,μ)
    x̂, = solve(solver,opμ,Realization([μ]))
    x̂
  end
  x̂ = param_cat(x̂vec)
  stats = CostTracker(t,nruns=num_params(r),name="RB")
  return (x̂,stats)
end
