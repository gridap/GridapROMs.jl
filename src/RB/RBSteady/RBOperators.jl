"""
    reduced_operator(solver::RBSolver,feop::ParamOperator,args...;kwargs...) -> ROMOperator
    reduced_operator(solver::RBSolver,feop::ParamOperator,s::AbstractSnapshots) -> ROMOperator
    reduced_operator(dir::String,solver::RBSolver,feop::ParamOperator,args...;kwargs...) -> ROMOperator

Offline phase: computes the reduced operator of the FE operator `feop`, i.e. the reduced
spaces ([`reduced_spaces`](@ref)) and the hyper-reduced residual and Jacobian
([`reduced_weak_form`](@ref)). The snapshots `s` are computed with
[`solution_snapshots`](@ref) if not provided (`args` and `kwargs` are passed on to it).
When `dir` is provided, snapshots and reduced operator are saved there
"""
function reduced_operator(
  dir::String,
  solver::RBSolver,
  feop::ParamOperator,
  args...;
  kwargs...
  )

  fesnaps = solution_snapshots(solver,feop,args...;kwargs...)
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

  fesnaps = solution_snapshots(solver,feop,args...;kwargs...)
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

"""
    abstract type ROMOperator{O} <: ParamOperator{O} end

Type representing reduced algebraic operators used within a reduced order modelling
framework in steady applications. A ROMOperator should contain the following information:

- a reduced test and trial space, computed according to [`reduced_spaces`](@ref)
- a hyper-reduced residual and jacobian, computed according to [`reduced_weak_form`](@ref)

Subtypes:

- [`RBOperator`](@ref)
"""
abstract type ROMOperator{O} <: ParamOperator{O} end

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

"""
    change_operator(op::ROMOperator,op′::ParamOperator) -> ROMOperator

Rebinds `op`'s hyper-reduced `lhs`/`rhs` to a new underlying FE operator `op′`
(e.g. the same problem at a different geometry, in a moving-domain setting),
adapting their triangulations to `op′`'s. When `op′`'s domains coincide with
`op`'s own (the common, non-moving-domain case), `change_domains` is a no-op and
this just rebinds `op.op` to `op′`
"""
function change_operator(op::ROMOperator,op′::ParamOperator)
  rhs,lhs = get_rhs(op),get_lhs(op)
  rhs′ = change_domains(rhs,get_domains_res(op′))
  lhs′ = change_domains(lhs,get_domains_jac(op′))
  ROMOperator(op′,op.trial,op.test,lhs′,rhs′)
end

"""
    struct RBOperator{O,A,B} <: ROMOperator{O}
      op::ParamOperator{O}
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
struct RBOperator{O,A,B} <: ROMOperator{O}
  op::ParamOperator{O}
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
  op::RBOperator{O,A,<:NoHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,A}

  b0 = allocate_residual(op.op,r,u,paramcache)
  b = contribution(get_domains(get_rhs(op))) do trian
    similar(b0)
  end
  b̂ = allocate_hypred_cache(get_rhs(op),r)
  HRParamArray(b,b̂.coeff,b̂.hypred)
end

function Algebra.allocate_jacobian(
  op::RBOperator{O,<:NoHRContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,B}

  A0 = allocate_jacobian(op.op,r,u,paramcache)
  A = contribution(get_domains(get_lhs(op))) do trian
    similar(A0)
  end
  Â = allocate_hypred_cache(get_lhs(op),r)
  HRParamArray(A,Â.coeff,Â.hypred)
end

function Algebra.residual!(
  b::HRParamArray,
  op::RBOperator{O,A,<:NoHRContribution},
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
  op::RBOperator{O,<:NoHRContribution,B},
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
  op::RBOperator{O,A,<:AffineHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::RBOperator{O,<:AffineHRContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs)
end

function Algebra.residual!(
  b::HRParamArray,
  op::RBOperator{O,A,<:RBFContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::RBOperator{O,<:RBFContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs,r)
end

"""
    get_linear_part(op::RBOperator) -> RBOperator
    get_nonlinear_part(op::RBOperator) -> RBOperator

Splits a Bundle-`lhs`/`rhs` `RBOperator` (built from a [`LinearNonlinearParamOperator`](@ref))
back into its two plainly-tagged halves, sharing the same reduced `trial`/`test`
"""
function Utils.get_linear_part(op::RBOperator)
  RBOperator(
    get_linear_part(get_fe_operator(op)),
    op.trial,op.test,
    get_linear_part(op.lhs),
    get_linear_part(op.rhs)
  )
end

function Utils.get_nonlinear_part(op::RBOperator)
  RBOperator(
    get_nonlinear_part(get_fe_operator(op)),
    op.trial,op.test,
    get_nonlinear_part(op.lhs),
    get_nonlinear_part(op.rhs)
  )
end

function ParamDataStructures.parameterise(op::RBOperator{LinearNonlinearParamEq},μ::Realisation)
  op_lin = parameterise(get_linear_part(op),μ)
  op_nlin = parameterise(get_nonlinear_part(op),μ)
  syscache_lin = allocate_systemcache(op_lin)
  LinNonlinParamOperator(op_lin,op_nlin,syscache_lin)
end

# local

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
  r::Realisation
  )

  dof_map = get_dof_map(op)
  stats = @timed values = solve(solver,op,r)
  set_rom_tracker!(solver,stats,nruns=num_params(r))
  snaps = Snapshots(values,dof_map,r)
  return snaps
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