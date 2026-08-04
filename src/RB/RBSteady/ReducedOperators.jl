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
  kwargs...
  )

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
  kwargs...
  )

  fesnaps,festats = solution_snapshots(solver,feop,args...;kwargs...)
  reduced_operator(solver,feop,fesnaps)
end

function reduced_operator(
  solver::RBSolver,
  feop::ParamOperator,
  s::AbstractSnapshots
  )

  feop′ = _setup(feop)
  red_trial,red_test = reduced_spaces(solver,feop′,s)
  reduced_operator(solver,feop′,red_trial,red_test,s)
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

function reduced_operator(
  solver::NeuralOpSolver,
  feop::ParamOperator,
  s::AbstractSnapshots
  )
  
  reduction = get_state_reduction(solver)
  model,ps,st,max_u = train_neural_operator(reduction,feop,s)
  NeuralRBOperator(feop,model,ps,st,max_u)
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

ParamSteady.get_fe_operator(op::RBOperator) = @abstractmethod
FESpaces.get_trial(op::RBOperator) = @abstractmethod
FESpaces.get_test(op::RBOperator) = @abstractmethod
get_lhs(op::RBOperator) = @abstractmethod
get_rhs(op::RBOperator) = @abstractmethod

function ParamSteady.set_domains(op::RBOperator,args...) 
  feop = set_domains(get_fe_operator(op),args...)
  RBOperator(feop,get_trial(op),get_test(op),get_lhs(op),get_rhs(op))
end

function ParamSteady.change_domains(op::RBOperator,args...) 
  feop = set_domains(get_fe_operator(op),args...)
  RBOperator(feop,get_trial(op),get_test(op),get_lhs(op),get_rhs(op))
end

function Algebra.allocate_residual(
  op::RBOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

  allocate_hypred_cache(get_rhs(op),r)
end

function Algebra.allocate_jacobian(
  op::RBOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

  allocate_hypred_cache(get_lhs(op),r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::RBOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

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
  op::RBOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

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
  rhs::AffineContribution
  )

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
  rhs::AffineContribution
  )

  GenericRBOperator(op,trial,test,lhs,rhs)
end

ParamSteady.get_fe_operator(op::GenericRBOperator) = op.op
FESpaces.get_trial(op::GenericRBOperator) = op.trial
FESpaces.get_test(op::GenericRBOperator) = op.test
get_lhs(op::GenericRBOperator) = op.lhs
get_rhs(op::GenericRBOperator) = op.rhs

function Algebra.allocate_residual(
  op::GenericRBOperator{O,T,B,<:NoHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  b = allocate_residual(op.op,r,u,paramcache)
  b̂ = allocate_hypred_cache(get_rhs(op),r)
  HRParamArray(b,b̂.coeff,b̂.hypred)
end

function Algebra.allocate_jacobian(
  op::GenericRBOperator{O,T,<:NoHRContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  A = allocate_jacobian(op.op,r,u,paramcache)
  Â = allocate_hypred_cache(get_lhs(op),r)
  HRParamArray(A,Â.coeff,Â.hypred)
end

function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator{O,SplitDomains,A,<:NoHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,A}

  fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)

  rhs = get_rhs(op)
  res = get_res(op)
  dc = res(r,uh,v)
  assem = get_param_assembler(op,r)

  for strian in get_domains(rhs)
    vecdata = collect_cell_vector_for_trian(test,dc,strian)
    assemble_vector_add!(b.fecache[strian],assem,vecdata)
    galerkin_projection!(b.coeff[strian],test,b.fecache[strian])
  end

  interpolate!(b,rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator{O,SplitDomains,<:NoHRContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,B}

  fill!(A,zero(eltype(A)))

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)

  lhs = get_lhs(op)
  jac = get_jac(op)
  dc = jac(r,uh,du,v)
  assem = get_param_assembler(op,r)

  for strian in get_domains(lhs)
    matdata = collect_cell_matrix_for_trian(trial,test,dc,strian)
    assemble_matrix_add!(A.fecache[strian],assem,matdata)
    galerkin_projection!(A.coeff[strian],test,A.fecache[strian],trial)
  end

  interpolate!(A,lhs)
end

function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator{O,T,A,<:AffineHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator{O,T,<:AffineHRContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs)
end

function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator{O,T,A,<:RBFContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator{O,T,<:RBFContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs,r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator{O,T,A,<:NNContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,A}

  fill!(b.hypred,zero(eltype(b.hypred)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator{O,T,<:NNContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  fill!(A.hypred,zero(eltype(A.hypred)))
  interpolate!(A,op.lhs,r)
end

"""
    struct LinearNonlinearRBOperator{O,T} <: RBOperator{O,T}
      op_linear::RBOperator
      op_nonlinear::RBOperator
    end

Extends the concept of [`GenericRBOperator`](@ref) to accommodate the linear/nonlinear
splitting of terms in nonlinear applications
"""
struct LinearNonlinearRBOperator{O,T} <: RBOperator{O,T}
  op_linear::RBOperator
  op_nonlinear::RBOperator

  function LinearNonlinearRBOperator(
    op_linear::RBOperator{OL,T},
    op_nonlinear::RBOperator{ON,T}
    ) where {OL,ON,T}

    new{LinearNonlinearParamEq,T}(op_linear,op_nonlinear)
  end

  function LinearNonlinearRBOperator(
    op_linear::RBOperator{OL,T},
    op_nonlinear::RBOperator{ON,T}
    ) where {OL<:ODEParamOperatorType,ON<:ODEParamOperatorType,T}

    new{LinearNonlinearParamODE,T}(op_linear,op_nonlinear)
  end
end

ParamAlgebra.get_linear_operator(op::LinearNonlinearRBOperator) = op.op_linear
ParamAlgebra.get_nonlinear_operator(op::LinearNonlinearRBOperator) = op.op_nonlinear
FESpaces.get_trial(op::LinearNonlinearRBOperator) = get_trial(get_nonlinear_operator(op))
FESpaces.get_test(op::LinearNonlinearRBOperator) = get_test(get_nonlinear_operator(op))

function ParamSteady.get_fe_operator(op::LinearNonlinearRBOperator)
  feop_lin = get_fe_operator(get_linear_operator(op))
  feop_nlin = get_fe_operator(get_nonlinear_operator(op))
  LinearNonlinearParamOperator(feop_lin,feop_nlin)
end 

function ParamAlgebra.allocate_paramcache(
  op::LinearNonlinearRBOperator,
  μ::AbstractRealisation
  )

  op_nlin = get_nonlinear_operator(op)
  allocate_paramcache(op_nlin,μ)
end

function ParamAlgebra.allocate_systemcache(
  op::LinearNonlinearRBOperator,
  u::AbstractVector
  )

  op_nlin = get_nonlinear_operator(op)
  allocate_systemcache(op_nlin,u)
end

function ParamAlgebra.update_paramcache!(
  paramcache::AbstractParamCache,
  op::LinearNonlinearRBOperator,
  μ::AbstractRealisation
  )

  op_nlin = get_nonlinear_operator(op)
  update_paramcache!(paramcache,op_nlin,μ)
end

function ParamDataStructures.parameterise(
  op::LinearNonlinearRBOperator,
  μ::AbstractRealisation
  )

  op_lin = parameterise(get_linear_operator(op),μ)
  op_nlin = parameterise(get_nonlinear_operator(op),μ)
  syscache_lin = allocate_systemcache(op_lin)
  LinNonlinParamOperator(op_lin,op_nlin,syscache_lin)
end

function change_operator(op::LinearNonlinearRBOperator,op′::LinearNonlinearParamOperator)
  op_lin′ = change_operator(get_linear_operator(op),get_linear_operator(op′))
  op_nlin′ = change_operator(get_nonlinear_operator(op),get_nonlinear_operator(op′))
  LinearNonlinearRBOperator(op_lin′,op_nlin′)
end

"""
    struct NeuralRBOperator{O,T,M,S,C,U} <: RBOperator{O,T}
      op::ParamOperator{O,T}
      model_weights::M
      model_states::S
      strategy::Strat
      max_u::U
    end

Reduced operator for Neural Operators. It replaces the Galerkin 
matrices (lhs, rhs) with the trained weights and states,
along with the model configuration and normalization factors.
"""
struct NeuralRBOperator{O,T,Mod,M,S,U} <: RBOperator{O,T}
  op::ParamOperator{O,T}
  model::Mod
  model_weights::M
  model_states::S
  max_u::U
end

ParamSteady.get_fe_operator(op::NeuralRBOperator) = op.op

# local

function get_local(op::RBOperator,μ::AbstractVector)
  trialμ = get_local(op.trial,μ)
  testμ = get_local(op.test,μ)
  lhsμ = get_local(op.lhs,μ)
  rhsμ = get_local(op.rhs,μ)
  RBOperator(op.op,trialμ,testμ,lhsμ,rhsμ)
end

function get_local(op::LinearNonlinearRBOperator,μ::AbstractVector)
  opμ_linear = get_local(get_linear_operator(op),μ)
  opμ_nlinear = get_local(get_nonlinear_operator(op),μ)
  LinearNonlinearRBOperator(opμ_linear,opμ_nlinear)
end

# snapshots 

function solution_snapshots(
  solver::RBSolver,
  op::RBOperator,
  r::AbstractRealisation,
  args...
  )

  x̂, = solve(solver,op,r,args...)
  i = get_dof_map(op)
  Snapshots(_fe_data(x̂),i,r)
end


# utils 

function _setup(op)
  V = get_test(op)
  U = get_trial(op)
  _convert_to_block(op,V,U)
end

function _setup(op::LinearNonlinearParamOperator)
  op_lin = _setup(get_linear_operator(op))
  op_nlin = _setup(get_nonlinear_operator(op))
  LinearNonlinearParamOperator(op_lin,op_nlin)
end

function _convert_to_block(op,V,U)
  op
end

function _convert_to_block(op::ParamOperator,V::T,U::T) where T<:MultiFieldFESpace{ConsecutiveMultiFieldStyle}
  Vb = _convert_to_block(V)
  Ub = _convert_to_block(U)
  feop = get_fe_operator(op)
  feopb = _convert_to_block(feop,Ub,Vb)
  typeof(op)(feopb)
end

function _convert_to_block(V::MultiFieldFESpace)
  MultiFieldFESpace(V.spaces;style=BlockMultiFieldStyle())
end

function _convert_to_block(feop::ParamFEOperator,U,V) 
  assem = SparseMatrixAssembler(U,V)
  typeof(feop)(feop.res,feop.jac,feop.pspace,assem,U,V,feop.domains)
end