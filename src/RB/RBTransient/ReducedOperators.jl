function RBSteady.reduced_operator(
  solver::RBSolver,
  odeop::ODEParamOperator{LinearNonlinearParamODE},
  red_trial::RBSpace,
  red_test::RBSpace,
  s::AbstractSnapshots
  )

  red_op_lin = reduced_operator(solver,get_linear_operator(odeop),red_trial,red_test,s)
  red_op_nlin = reduced_operator(solver,get_nonlinear_operator(odeop),red_trial,red_test,s)
  LinearNonlinearRBOperator(red_op_lin,red_op_nlin)
end

function RBSteady.RBOperator(
  odeop::ODEParamOperator,
  trial::RBSpace,
  test::RBSpace,
  lhs::TupOfAffineContribution,
  rhs::AffineContribution
  )

  trians_rhs = get_domains(rhs)
  trians_lhs = map(get_domains,lhs)
  odeop′ = change_domains(odeop,trians_rhs,trians_lhs)
  GenericRBOperator(odeop′,trial,test,lhs,rhs)
end

const TransientRBOperator{O<:ODEParamOperatorType,T} = RBOperator{O,T}
const JointTransientRBOperator{O<:ODEParamOperatorType} = TransientRBOperator{O,JointDomains}
const SplitTransientRBOperator{O<:ODEParamOperatorType} = TransientRBOperator{O,SplitDomains}
const TransientGenericRBOperator{O<:ODEParamOperatorType,T,A,B} = GenericRBOperator{O,T,A,B}

function Algebra.allocate_residual(
  op::TransientRBOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  allocate_hypred_cache(get_rhs(op),r)
end

function Algebra.allocate_jacobian(
  op::TransientRBOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  allocate_hypred_cache(get_lhs(op),r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientRBOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  fill!(b,zero(eltype(b)))

  np = num_params(r)
  rhs = get_rhs(op)
  hr_time_ids = get_common_time_domain(rhs)
  hr_param_time_ids = range_1d(1:np,hr_time_ids,np)
  hr_uh = _make_hr_uh_from_us(op,us,paramcache.trial,hr_param_time_ids)

  test = get_test(op)
  v = get_fe_basis(test)

  trian_res = get_domains_res(op)
  μ = get_params(r)
  hr_t = view(get_times(r),hr_time_ids)
  res = get_res(op)
  dc = res(μ,hr_t,hr_uh,v)

  for strian in trian_res
    b_strian = b.fecache[strian]
    rhs_strian = get_interpolation(rhs[strian])
    vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian,hr_param_time_ids)
    assemble_hr_array_add!(b_strian,vecdata...)
  end

  interpolate!(b,rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientRBOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  )

  fill!(A,zero(eltype(A)))

  np = num_params(r)
  lhss = get_lhs(op)
  hr_time_ids = get_common_time_domain(lhss)
  hr_param_time_ids = range_1d(1:np,hr_time_ids,np)
  hr_uh = _make_hr_uh_from_us(op,us,paramcache.trial,hr_param_time_ids)

  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)

  trian_jacs = get_domains_jac(op)
  μ = get_params(r)
  hr_t = view(get_times(r),hr_time_ids)
  jacs = get_jacs(op)

  for k in 1:get_order(op)+1
    Ak = A.fecache[k]
    lhs = lhss[k]
    jac = jacs[k]
    w = ws[k]
    iszero(w) && continue
    dc = w * jac(μ,hr_t,hr_uh,du,v)
    trian_jac = trian_jacs[k]
    for strian in trian_jac
      A_strian = Ak[strian]
      lhs_strian = get_interpolation(lhs[strian])
      matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian,hr_param_time_ids)
      assemble_hr_array_add!(A_strian,matdata...)
    end
  end

  interpolate!(A,lhss)
end

function Algebra.allocate_residual(
  op::TransientGenericRBOperator{O,T,B,<:HighDimNoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,B}

  b = allocate_residual(op.op,r,us,paramcache)
  b̂ = allocate_hypred_cache(get_rhs(op),r)
  HRParamArray(b,b̂.coeff,b̂.hypred)
end

function Algebra.allocate_jacobian(
  op::TransientGenericRBOperator{O,T,<:TupOfHighDimNoHRContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,B}

  A = allocate_jacobian(op.op,r,us,paramcache)
  Â = allocate_hypred_cache(get_lhs(op),r)
  HRParamArray(A,Â.coeff,Â.hypred)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientGenericRBOperator{O,T,A,<:HighDimNoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,A}

  fill!(b,zero(eltype(b)))

  uh = ODEs._make_uh_from_us(op,us,paramcache.trial)
  test = get_test(op)
  v = get_fe_basis(test)

  rhs = get_rhs(op)
  μ = get_params(r)
  t = get_times(r)
  res = get_res(op)
  dc = res(μ,t,uh,v)
  assem = get_param_assembler(op,r)

  for strian in get_domains(rhs)
    red = get_style(rhs[strian])
    c = get_time_combination(red)
    vecdata = collect_cell_vector_for_trian(test,dc,strian)
    assemble_vector_add!(b.fecache[strian],assem,vecdata)
    galerkin_projection!(b.coeff[strian],test,b.fecache[strian],c)
  end

  interpolate!(b,rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientGenericRBOperator{O,T,<:TupOfHighDimNoHRContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B}

  fill!(A,zero(eltype(A)))

  uh = ODEs._make_uh_from_us(op,us,paramcache.trial)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)

  lhss = get_lhs(op)
  μ = get_params(r)
  t = get_times(r)
  jacs = get_jacs(op)
  trian_jacs = get_domains_jac(op)
  assem = get_param_assembler(op,r)

  for k in 1:get_order(op)+1
    Ak = A.fecache[k]
    Ark = A.coeff[k]
    lhs = lhss[k]
    jac = jacs[k]
    w = ws[k]
    iszero(w) && continue
    dc = w * jac(μ,t,uh,du,v)
    for strian in trian_jacs[k]
      red = get_style(lhs[strian])
      c = get_time_combination(red)
      matdata = collect_cell_matrix_for_trian(trial,test,dc,strian)
      assemble_matrix_add!(Ak[strian],assem,matdata)
      galerkin_projection!(Ark[strian],test,Ak[strian],trial,c)
    end
  end

  interpolate!(A,lhss)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientGenericRBOperator{O,T,A,<:HighDimAffineHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientGenericRBOperator{O,T,<:TupOfHighDimAffineHRContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientGenericRBOperator{O,T,A,<:HighDimRBFContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,A}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientGenericRBOperator{O,T,<:TupOfHighDimRBFContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs,r)
end

function Algebra.residual!(
  b::HRParamArray,
  op::GenericRBOperator{O,T,A,<:NNContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,A}

  fill!(b.hypred,zero(eltype(b.hypred)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::GenericRBOperator{O,T,<:TupOfHighDimNNContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B}

  fill!(A.hypred,zero(eltype(A.hypred)))
  interpolate!(A,op.lhs,r)
end

const TransientLinearNonlinearRBOperator{A<:TransientRBOperator,B<:TransientRBOperator} = LinearNonlinearRBOperator{A,B}

function get_order(op::TransientLinearNonlinearRBOperator)
  max(get_order(get_linear_operator(op)),get_order(get_nonlinear_operator(op)))
end

function ParamODEs.add_initial_conditions(
  solver::ODESolver,
  op::TransientLinearNonlinearRBOperator,
  args...
  )
  
  add_initial_conditions(solver,get_nonlinear_operator(op),args...)
end

# snapshots 

function RBSteady.solution_snapshots(
  solver::TransientRBSolver{A,B,<:SteadyReduction},
  op::TransientRBOperator,
  r::TransientRealisation,
  args...
  ) where {A,B}

  fesolver = get_fe_solver(solver)
  x̂,stats = solve(fesolver,op,r,args...) |> collect
  i = get_dof_map(op)
  snaps = Snapshots(_fe_data(x̂),i,r)
  return snaps,stats
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
  us::Tuple{Vararg{AbstractVector}},
  trial::Tuple{Vararg{FESpace}},
  hr_ids::AbstractVector
  )

  us′ = ()
  trial′ = ()
  for (u,trial) in zip(us,trial)
    us′ = (us′...,_reduce_vector(u,hr_ids))
    trial′ = (trial′...,_reduce_trial(trial,hr_ids))
  end
  return us′,trial′
end

function _make_hr_uh_from_us(
  odeop::ODEParamOperator,
  us::Tuple{Vararg{AbstractVector}},
  trial::Tuple{Vararg{FESpace}},
  hr_param_time_ids::AbstractVector
  )

  hr_us,hr_trial = _reduce_arguments(us,trial,hr_param_time_ids)
  ODEs._make_uh_from_us(odeop,hr_us,hr_trial)
end