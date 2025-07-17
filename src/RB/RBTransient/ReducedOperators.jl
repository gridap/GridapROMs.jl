function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots)

  red_op_lin = reduced_operator(solver,get_linear_operator(odeop),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(odeop),red_trial,red_test,s)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

function RBSteady.RBOperator(
  odeop::ODEParamOperator,
  trial::RBSpace,
  test::RBSpace,
  lhs::TupOfAffineContribution,
  rhs::AffineContribution)

  trians_rhs = get_domains(rhs)
  trians_lhs = map(get_domains,lhs)
  odeop′ = change_domains(odeop,trians_rhs,trians_lhs)
  GenericRBOperator(odeop′,trial,test,lhs,rhs)
end

function RBSteady.RBOperator(
  odeop::ODEParamOperator,
  trial::RBSpace,
  test::RBSpace,
  lhs::Tuple{Vararg{LocalProjection}},
  rhs::LocalProjection
  )

  LocalRBOperator(odeop,trial,test,lhs,rhs)
end

const TransientRBOperator{O<:ODEParamOperatorType} = RBOperator{O}
const TransientGenericRBOperator{O<:ODEParamOperatorType,B} = GenericRBOperator{O,<:TupOfAffineContribution,B}
const TransientLocalRBOperator{O<:ODEParamOperatorType,B} = LocalRBOperator{O,<:Tuple{Vararg{LocalProjection}},B}

function Algebra.allocate_residual(
  op::TransientRBOperator,
  r::TransientRealization,
  u::AbstractVector,
  paramcache)

  allocate_hypred_cache(op.rhs,r)
end

function Algebra.allocate_jacobian(
  op::TransientRBOperator,
  r::TransientRealization,
  u::AbstractVector,
  paramcache)

  allocate_hypred_cache(op.lhs,r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientGenericRBOperator,
  r::TransientRealization,
  u::AbstractVector,
  paramcache
  )

  fill!(b,zero(eltype(b)))

  np = num_params(r)
  hr_time_ids = get_common_time_domain(op.rhs)
  hr_param_time_ids = range_1d(1:np,hr_time_ids,np)
  hr_uh = _make_hr_uh_from_us(op.op,u,paramcache.trial,hr_param_time_ids)

  test = get_test(op.op)
  v = get_fe_basis(test)

  trian_res = get_domains_res(op.op)
  μ = get_params(r)
  hr_t = view(get_times(r),hr_time_ids)
  res = get_res(op.op)
  dc = res(μ,hr_t,hr_uh,v)

  for strian in trian_res
    b_strian = b.fecache[strian]
    rhs_strian = get_interpolation(op.rhs[strian])
    vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian,hr_param_time_ids)
    assemble_hr_vector_add!(b_strian,vecdata...)
  end

  interpolate!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientGenericRBOperator,
  r::TransientRealization,
  u::AbstractVector,
  paramcache
  )

  fill!(A,zero(eltype(A)))

  np = num_params(r)
  hr_time_ids = get_common_time_domain(op.lhs)
  hr_param_time_ids = range_1d(1:np,hr_time_ids,np)
  hr_uh = _make_hr_uh_from_us(op.op,u,paramcache.trial,hr_param_time_ids)

  trial = get_trial(op.op)
  du = get_trial_fe_basis(trial)
  test = get_test(op.op)
  v = get_fe_basis(test)

  trian_jacs = get_domains_jac(op.op)
  μ = get_params(r)
  hr_t = view(get_times(r),hr_time_ids)
  jacs = get_jacs(op.op)

  for k in 1:get_order(op.op)+1
    Ak = A.fecache[k]
    lhs = op.lhs[k]
    jac = jacs[k]
    dc = jac(μ,hr_t,hr_uh,du,v)
    trian_jac = trian_jacs[k]
    for strian in trian_jac
      A_strian = Ak[strian]
      lhs_strian = get_interpolation(lhs[strian])
      matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian,hr_param_time_ids)
      assemble_hr_matrix_add!(A_strian,matdata...)
    end
  end

  interpolate!(A,op.lhs)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientGenericRBOperator{O,<:RBFContribution},
  r::TransientRealization,
  u::AbstractVector,
  paramcache) where O

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientGenericRBOperator{O,<:RBFContribution},
  r::TransientRealization,
  u::AbstractVector,
  paramcache) where O

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs,r)
end

function RBSteady.get_local(op::TransientLocalRBOperator,μ)
  trialμ = get_local(op.trial,μ)
  testμ = get_local(op.test,μ)
  lhsμ = map(lhs->get_local(lhs,μ),op.lhs)
  rhsμ = get_local(op.rhs,μ)
  RBOperator(op.op,trialμ,testμ,lhsμ,rhsμ)
end

# utils

function _reduce_vector(u::ConsecutiveParamVector,hr_ids::AbstractVector)
  ConsecutiveParamArray(view(u.data,:,hr_ids))
end

function _reduce_vector(u::BlockConsecutiveParamVector,hr_ids::AbstractVector)
  mortar(map(b -> _reduce_vector(b,hr_ids),blocks(u)))
end

function _reduce_vector(u::RBParamVector,hr_ids::AbstractVector)
  RBParamVector(u.data,_reduce_vector(u.fe_data,hr_ids))
end

function _reduce_trial(trial::TrialParamFESpace,hr_ids::AbstractVector)
  dv = trial.dirichlet_values
  dv′ = _reduce_vector(trial.dirichlet_values,hr_ids)
  trial′ = TrialParamFESpace(dv′,trial.space)
  return trial′
end

function _reduce_trial(trial::TrivialParamFESpace,hr_ids::AbstractVector)
  trial′ = TrivialParamFESpace(trial.space,length(hr_ids))
  return trial′
end

function _reduce_trial(trial::MultiFieldFESpace,hr_ids::AbstractVector)
  vec_trial′ = map(f -> _reduce_trial(f,hr_ids),trial.spaces)
  trial′ = MultiFieldFESpace(trial.vector_type,vec_trial′,trial.multi_field_style)
  return trial′
end

function _reduce_arguments(
  u::AbstractVector,
  trial::Tuple{Vararg{FESpace}},
  hr_ids::AbstractVector)

  N = length(trial)
  u′ = _reduce_vector(u,hr_ids)
  us′ = tfill(u′,Val{N}())
  trial′ = ()
  for i = eachindex(trial)
    trial′ = (trial′...,_reduce_trial(trial[i],hr_ids))
  end
  return us′,trial′
end

function _make_hr_uh_from_us(
  odeop::ODEParamOperator,
  u::AbstractVector,
  trial::Tuple{Vararg{FESpace}},
  hr_param_time_ids)

  hr_us,hr_trial = _reduce_arguments(u,trial,hr_param_time_ids)
  ODEs._make_uh_from_us(odeop,hr_us,hr_trial)
end
