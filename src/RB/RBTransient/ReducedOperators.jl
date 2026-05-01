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

function RBSteady.RBOperator(
  odeop::ODEParamOperator,
  trial::RBSpace,
  test::RBSpace,
  lhs::Tuple{Vararg{LocalProjection}},
  rhs::LocalProjection
  )

  LocalRBOperator(odeop,trial,test,lhs,rhs)
end

const TransientRBOperator{O<:ODEParamOperatorType,T} = RBOperator{O,T}
const JointTransientRBOperator{O<:ODEParamOperatorType} = TransientRBOperator{O,JointDomains}
const SplitTransientRBOperator{O<:ODEParamOperatorType} = TransientRBOperator{O,SplitDomains}
const TransientGenericRBOperator{O<:ODEParamOperatorType,T,B} = GenericRBOperator{O,<:TupOfAffineContribution,T,B}
const TransientLocalRBOperator{O<:ODEParamOperatorType,T,B} = LocalRBOperator{O,<:Tuple{Vararg{LocalProjection}},T,B}

for T in (:JointTransientRBOperator,:SplitTransientRBOperator)
  @eval begin
    function Algebra.allocate_residual(
      op::$T,
      r::TransientRealisation,
      us::Tuple{Vararg{AbstractVector}},
      paramcache
      )

      allocate_hypred_cache(get_rhs(op),r)
    end

    function Algebra.allocate_jacobian(
      op::$T,
      r::TransientRealisation,
      us::Tuple{Vararg{AbstractVector}},
      paramcache
      )

      allocate_hypred_cache(get_lhs(op),r)
    end

    function Algebra.residual!(
      b::HRParamArray,
      op::$T,
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
        assemble_hr_vector_add!(b_strian,vecdata...)
      end

      interpolate!(b,rhs)
    end

    function Algebra.jacobian!(
      A::HRParamArray,
      op::$T,
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
          assemble_hr_matrix_add!(A_strian,matdata...)
        end
      end

      interpolate!(A,lhss)
    end
  end
end

function Algebra.residual!(
  b::NoHRParamArray,
  op::TransientGenericRBOperator{O,T,<:NoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T}
function Algebra.allocate_residual(
  op::TransientGenericRBOperator{O,T,<:NoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T}

  b_fe = allocate_residual(op.op,r,us,paramcache)
  b_rb = allocate_hypred_cache(get_rhs(op),r)
  coeff = allocate_hypred_cache(get_rhs(op),r)
  nohr_array(b_fe,b_rb,coeff,b_rb)
end

function Algebra.allocate_jacobian(
  op::TransientGenericRBOperator{O,T,<:NoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T}

  A_fe = allocate_jacobian(op.op,r,us,paramcache)
  A_rb = allocate_hypred_cache(get_lhs(op),r)
  coeff = allocate_hypred_cache(get_lhs(op),r)
  nohr_array(A_fe,A_rb,coeff,A_rb)
end

function Algebra.residual!(
  b::NoHRParamArray,
  op::TransientGenericRBOperator{O,T,<:NoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T}

  fill!(b,zero(eltype(b)))

  uh = ODEs._make_uh_from_us(op,us,paramcache.trial)
  test = get_test(op)
  v = get_fe_basis(test)

  rhs = get_rhs(op)
  μ = get_params(r)
  t = get_times(r)
  res = get_res(op)
  dc = res(μ,t,uh,v)
  assem = get_param_assembler(get_fe_operator(op),r)

  for strian in get_domains(rhs)
    vecdata = collect_cell_vector_for_trian(get_fe_space(test),dc,strian)
    b_fe = allocate_vector(assem,vecdata)
    fill!(b_fe,zero(eltype(b_fe)))
    assemble_vector_add!(b_fe,assem,vecdata)
    copyto!(b.fecache[strian],b_fe)
    project!(b.rbfecache[strian],test,b.fecache[strian])
  end

  interpolate!(b,rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientGenericRBOperator{O,T,<:NoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T}

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
  assem = get_param_assembler(get_fe_operator(op),r)
  Φ_test = get_basis(test)
  Φ_trial = get_basis(trial)

  for k in 1:get_order(op)+1
    jac = jacs[k]
    w = ws[k]
    iszero(w) && continue
    dc = w * jac(μ,t,uh,du,v)
    for strian in trian_jacs[k]
      matdata = collect_cell_matrix_for_trian(get_fe_space(trial),get_fe_space(test),dc,strian)
      A_fe = allocate_matrix(assem,matdata)
      fill!(A_fe,zero(eltype(A_fe)))
      assemble_matrix_add!(A_fe,assem,matdata)
      for i in 1:param_length(r)
        Ai = param_getindex(A_fe,i)
        Ci = param_getindex(A.fecache[k][strian],i)
        Ci .= Φ_test' * Ai * Φ_trial
      end
    end
  end

  interpolate!(A,lhss)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientGenericRBOperator{O,T,<:AffineHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientGenericRBOperator{O,T,<:AffineHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B<:RBSteady.AffineHRContribution}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs)
end

function Algebra.residual!(
  b::HRParamArray,
  op::TransientGenericRBOperator{O,T,<:RBFContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T}

  fill!(b,zero(eltype(b)))
  interpolate!(b,op.rhs,r)
end

function Algebra.jacobian!(
  A::HRParamArray,
  op::TransientGenericRBOperator{O,T,<:RBFContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T}

  fill!(A,zero(eltype(A)))
  interpolate!(A,op.lhs,r)
end

function RBSteady.get_local(op::TransientLocalRBOperator,μ::AbstractVector)
  trialμ = get_local(op.trial,μ)
  testμ = get_local(op.test,μ)
  lhsμ = map(lhs->get_local(lhs,μ),op.lhs)
  rhsμ = get_local(op.rhs,μ)
  RBOperator(op.op,trialμ,testμ,lhsμ,rhsμ)
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

function Algebra.jacobian!(
  A::NoHRParamArray,
  op::TransientGenericRBOperator{O,T,<:NoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T}

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
  assem = get_param_assembler(get_fe_operator(op),r)

  for k in 1:get_order(op)+1
    jac = jacs[k]
    w = ws[k]
    iszero(w) && continue
    dc = w * jac(μ,t,uh,du,v)
    for strian in trian_jacs[k]
      matdata = collect_cell_matrix_for_trian(get_fe_space(trial),get_fe_space(test),dc,strian)
      A_fe = allocate_matrix(assem,matdata)
      fill!(A_fe,zero(eltype(A_fe)))
      assemble_matrix_add!(A_fe,assem,matdata)
      copyto!(A.fecache[k][strian],A_fe)
      project!(A.rbfecache[k][strian],trial,test,A.fecache[k][strian])
    end
  end

  interpolate!(A,lhss)
end
