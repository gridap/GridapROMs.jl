"""
    reduced_operator(solver::RBSolver,feop::ParamOperator,args...;kwargs...) -> ROMOperator
    reduced_operator(solver::RBSolver,feop::TransientParamOperator,args...;kwargs...) -> TransientROMOperator

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
  ROMOperator(feop,red_trial,red_test,red_lhs,red_rhs)
end

function reduced_operator(
  solver::RBSolver,
  op::ParamOperator{LinearNonlinearParamEq},
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots
  )

  red_op_lin = reduced_operator(solver,get_linear_part(op),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_part(op),red_trial,red_test,s)
  lhs = Bundle(get_lhs(red_op_lin),get_lhs(red_op_nlin))
  rhs = Bundle(get_rhs(red_op_lin),get_rhs(red_op_nlin))
  RBOperator(op,red_trial,red_test,lhs,rhs)
end

function reduced_operator(rbsolver::RBSolver,feop::ParamOperator,s,jac,res)
  feop′ = _setup(feop)
  red_trial,red_test = reduced_spaces(rbsolver,feop′,s)
  reduced_operator(rbsolver,feop′,red_trial,red_test,jac,res)
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
  ROMOperator(feop,red_trial,red_test,red_lhs,red_rhs)
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
  red_op_lin = reduced_operator(solver,get_linear_part(op),red_trial,red_test,jac_lin,res_lin)
  red_op_nlin = reduced_operator(solver,get_nonlinear_part(op),red_trial,red_test,jac_nlin,res_nlin)
  LinearNonlinearROMOperator(red_op_lin,red_op_nlin)
end

"""
    abstract type ROMOperator{O,T} <: ParamOperator{O,T} end

Type representing reduced algebraic operators used within a reduced order modelling
framework in steady applications. A ROMOperator should contain the following information:

- a reduced test and trial space, computed according to [`reduced_spaces`](@ref)
- a hyper-reduced residual and jacobian, computed according to [`reduced_weak_form`](@ref)

Subtypes:

- [`RBOperator`](@ref)
- [`LinearNonlinearROMOperator`](@ref)
"""
abstract type ROMOperator{O,T} <: ParamOperator{O,T} end

const JointROMOperator{O} = ROMOperator{O,JointDomains}
const SplitROMOperator{O} = ROMOperator{O,SplitDomains}

ParamSteady.get_fe_operator(op::ROMOperator) = @abstractmethod
FESpaces.get_trial(op::ROMOperator) = @abstractmethod
FESpaces.get_test(op::ROMOperator) = @abstractmethod
get_lhs(op::ROMOperator) = @abstractmethod
get_rhs(op::ROMOperator) = @abstractmethod

function ParamSteady.set_domains(op::ROMOperator,args...)
  feop = set_domains(get_fe_operator(op),args...)
  ROMOperator(feop,get_trial(op),get_test(op),get_lhs(op),get_rhs(op))
end

function ParamSteady.change_domains(op::ROMOperator,args...)
  feop = change_domains(get_fe_operator(op),args...)
  ROMOperator(feop,get_trial(op),get_test(op),get_lhs(op),get_rhs(op))
end

function Algebra.allocate_residual(
  op::ROMOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

  allocate_hypred_cache(get_rhs(op),r)
end

function Algebra.allocate_jacobian(
  op::ROMOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

  allocate_hypred_cache(get_lhs(op),r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::ROMOperator,
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
  dc = ParamSteady._eval_res(op,r,uh,v)

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
  op::ROMOperator,
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

function change_operator(op::JointROMOperator,op′::ParamOperator)
  rhs,lhs = get_rhs(op),get_lhs(op)
  ROMOperator(op′,op.trial,op.test,lhs,rhs)
end

function change_operator(op::SplitROMOperator,op′::ParamOperator)
  rhs,lhs = get_rhs(op),get_lhs(op)
  rhs′ = change_domains(rhs,get_domains_res(op′))
  lhs′ = change_domains(lhs,get_domains_jac(op′))
  ROMOperator(op′,op.trial,op.test,lhs′,rhs′)
end

"""
    struct RBOperator{O,T,A,B} <: ROMOperator{O,T}
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
struct RBOperator{O,T,A,B} <: ROMOperator{O,T}
  op::ParamOperator{O,T}
  trial::RBSpace
  test::RBSpace
  lhs::A
  rhs::B
end

function ROMOperator(
  op::ParamOperator,
  trial::RBSpace,
  test::RBSpace,
  lhs::AffineContribution,
  rhs::AffineContribution
  )

  RBOperator(op,trial,test,lhs,rhs)
end

function ROMOperator(
  op::ParamOperator,
  trial::RBSpace,
  test::RBSpace,
  lhs::Bundle,
  rhs::Bundle
  )

  RBOperator(op,trial,test,lhs,rhs)
end

ParamSteady.get_fe_operator(op::RBOperator) = op.op
FESpaces.get_trial(op::RBOperator) = op.trial
FESpaces.get_test(op::RBOperator) = op.test
get_lhs(op::RBOperator) = op.lhs
get_rhs(op::RBOperator) = op.rhs

function Algebra.allocate_residual(
  op::RBOperator{O,T,B,<:NoHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  b = allocate_residual(op.op,r,u,paramcache)
  b̂ = allocate_hypred_cache(get_rhs(op),r)
  HRParamArray(b,b̂.coeff,b̂.hypred)
end

function Algebra.allocate_jacobian(
  op::RBOperator{O,T,<:NoHRContribution,B},
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
  op::RBOperator{O,SplitDomains,A,<:NoHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,A}

  fill!(b,zero(eltype(b)))

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)

  rhs = get_rhs(op)
  dc = ParamSteady._eval_res(op,r,uh,v)
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
  op::RBOperator{O,SplitDomains,<:NoHRContribution,B},
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
  op::RBOperator{O,T,A,<:AffineHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::RBOperator{O,T,<:AffineHRContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs)
end

function Algebra.residual!(
  b::HRParamArray,
  op::RBOperator{O,T,A,<:RBFContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::RBOperator{O,T,<:RBFContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs,r)
end

"""
    const LinearNonlinearROMOperator{T} = RBOperator{LinearNonlinearParamEq,T}

A linear+nonlinear split ROM operator is just a [`RBOperator`](@ref) whose
`op`/`lhs`/`rhs` fields are each a [`Bundle`](@ref) of the linear and nonlinear
halves, sharing the same reduced `trial`/`test`
"""
const LinearNonlinearROMOperator{T} = RBOperator{LinearNonlinearParamEq,T}

"""
    const AnyLinearNonlinearROMOperator{T} = Union{RBOperator{LinearNonlinearParamEq,T},RBOperator{LinearNonlinearParamODE,T}}

Covers both the steady ([`LinearNonlinearROMOperator`](@ref)) and transient
(`TransientLinearNonlinearROMOperator`, defined in `RBTransient`) linear+nonlinear
split ROM operators, so that the paramcache/`parameterise`/`change_operator`/
`get_local` machinery below is written once for both
"""
const AnyLinearNonlinearROMOperator{T} = Union{RBOperator{LinearNonlinearParamEq,T},RBOperator{LinearNonlinearParamODE,T}}

"""
    LinearNonlinearROMOperator(op_lin::ROMOperator,op_nlin::ROMOperator) -> LinearNonlinearROMOperator

Merges two plainly-tagged, freshly-built reduced operators (sharing the same
reduced trial/test) into a single [`Bundle`](@ref)-based `RBOperator`, tagged
`LinearNonlinearParamEq` or `LinearNonlinearParamODE` as appropriate. Used when
`op_lin`/`op_nlin` were built independently (e.g. loaded separately from disk);
when a bundled operator is already in hand, prefer rebuilding its `lhs`/`rhs`
`Bundle`s directly (see `change_operator`/`get_local` below) instead of calling this
"""
function LinearNonlinearROMOperator(
  op_lin::ROMOperator{OL},op_nlin::ROMOperator{ON}
  ) where {OL<:Union{LinearParamEq,NonlinearParamEq},ON<:Union{LinearParamEq,NonlinearParamEq}}

  @check get_trial(op_lin) === get_trial(op_nlin)
  @check get_test(op_lin) === get_test(op_nlin)
  feop_lin = get_fe_operator(get_fe_operator(op_lin))
  feop_nlin = get_fe_operator(get_fe_operator(op_nlin))
  res = Bundle(get_res(feop_lin),get_res(feop_nlin))
  jac = Bundle(get_jac(feop_lin),get_jac(feop_nlin))
  domains = Bundle(get_domains(feop_lin),get_domains(feop_nlin))
  feop = ParamFEOperator(
    res,jac,get_param_space(feop_nlin),get_trial(feop_nlin),get_test(feop_nlin),domains)
  op = get_algebraic_operator(feop)
  lhs = Bundle(get_lhs(op_lin),get_lhs(op_nlin))
  rhs = Bundle(get_rhs(op_lin),get_rhs(op_nlin))
  RBOperator(op,get_trial(op_lin),get_test(op_lin),lhs,rhs)
end

function LinearNonlinearROMOperator(
  op_lin::ROMOperator{OL},op_nlin::ROMOperator{ON}
  ) where {OL<:ODEParamOperatorType,ON<:ODEParamOperatorType}

  @check get_trial(op_lin) === get_trial(op_nlin)
  @check get_test(op_lin) === get_test(op_nlin)
  feop_lin = get_fe_operator(get_fe_operator(op_lin))
  feop_nlin = get_fe_operator(get_fe_operator(op_nlin))
  res = Bundle(get_res(feop_lin),get_res(feop_nlin))
  jacs = Bundle(get_jacs(feop_lin),get_jacs(feop_nlin))
  domains = Bundle(get_domains(feop_lin),get_domains(feop_nlin))
  feop = TransientParamFEOperator(
    res,jacs,get_param_space(feop_nlin),get_trial(feop_nlin),get_test(feop_nlin),domains)
  op = get_algebraic_operator(feop)
  lhs = Bundle(get_lhs(op_lin),get_lhs(op_nlin))
  rhs = Bundle(get_rhs(op_lin),get_rhs(op_nlin))
  RBOperator(op,get_trial(op_lin),get_test(op_lin),lhs,rhs)
end

function Utils.get_linear_part(op::AnyLinearNonlinearROMOperator)
  RBOperator(get_linear_part(get_fe_operator(op)),op.trial,op.test,op.lhs.linear,op.rhs.linear)
end

function Utils.get_nonlinear_part(op::AnyLinearNonlinearROMOperator)
  RBOperator(get_nonlinear_part(get_fe_operator(op)),op.trial,op.test,op.lhs.nonlinear,op.rhs.nonlinear)
end

function ParamAlgebra.allocate_paramcache(
  op::AnyLinearNonlinearROMOperator,
  μ::AbstractRealisation
  )

  op_nlin = get_nonlinear_part(op)
  allocate_paramcache(op_nlin,μ)
end

function ParamAlgebra.allocate_systemcache(
  op::AnyLinearNonlinearROMOperator,
  u::AbstractVector
  )

  op_nlin = get_nonlinear_part(op)
  allocate_systemcache(op_nlin,u)
end

function ParamAlgebra.update_paramcache!(
  paramcache::AbstractParamCache,
  op::AnyLinearNonlinearROMOperator,
  μ::AbstractRealisation
  )

  op_nlin = get_nonlinear_part(op)
  update_paramcache!(paramcache,op_nlin,μ)
end

function ParamDataStructures.parameterise(
  op::AnyLinearNonlinearROMOperator,
  μ::AbstractRealisation
  )

  op_lin = parameterise(get_linear_part(op),μ)
  op_nlin = parameterise(get_nonlinear_part(op),μ)
  syscache_lin = allocate_systemcache(op_lin)
  LinNonlinParamOperator(op_lin,op_nlin,syscache_lin)
end

# local

# change_operator/get_local for a Bundle-based linear+nonlinear ROM operator
# are *not* special-cased here: the generic change_operator(::JointROMOperator/
# ::SplitROMOperator,...) and get_local(::ROMOperator,...) below already work,
# as long as get_local/change_domains/get_domains_res/get_domains_jac know how
# to recurse into a Bundle (see Utils/FEDomains.jl and the extension below)
get_local(b::Bundle,μ::AbstractVector) = Bundle(get_local(b.linear,μ),get_local(b.nonlinear,μ))

function get_local(op::ROMOperator,μ::AbstractVector)
  trialμ = get_local(op.trial,μ)
  testμ = get_local(op.test,μ)
  lhsμ = get_local(op.lhs,μ)
  rhsμ = get_local(op.rhs,μ)
  ROMOperator(op.op,trialμ,testμ,lhsμ,rhsμ)
end

# snapshots 

function solution_snapshots(
  solver::RBSolver,
  op::ROMOperator,
  r::AbstractRealisation,
  args...
  )

  x̂ = solve(solver,op,r,args...)
  i = get_dof_map(op)
  Snapshots(_fe_data(x̂),i,r)
end


# utils 

function _setup(op)
  V = get_test(op)
  U = get_trial(op)
  _convert_to_block(op,V,U)
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

function _convert_to_block(feop::ParamFEOperator,U,V) 
  assem = SparseMatrixAssembler(U,V)
  typeof(feop)(feop.res,feop.jac,feop.pspace,assem,U,V,feop.domains)
end

_convert_to_block(V::FESpace) = V

function _convert_to_block(V::MultiFieldFESpace)
  MultiFieldFESpace(V.spaces;style=BlockMultiFieldStyle())
end