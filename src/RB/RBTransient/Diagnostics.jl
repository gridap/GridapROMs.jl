function RBSteady.allocate_diagnostic_residual(nlop::SpaceTimeParamOperator,u)
  rhs = get_rhs(nlop.op) 
  RBSteady.allocate_dcontribution(rhs,nlop.r)
end

function RBSteady.allocate_diagnostic_jacobian(nlop::SpaceTimeParamOperator,u)
  lhs = get_lhs(nlop.op)
  RBSteady.allocate_dcontribution(lhs,nlop.r)
end

function RBSteady.diagnostic_interpolate!(
  cache::Tuple{Vararg{DiagnosticsContribution}},
  a::TupOfAffineContribution
  )

  for (ai,bi,ci,di) in zip(cache.hypred,cache.coeff,a,cache.fecache)
    RBSteady.diagnostic_interpolate!(ai,bi,ci,di)
  end
end

function RBSteady.diagnostic_residual!(
  b::DiagnosticsContribution,
  op::SplitTransientRBOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

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

  RBSteady.diagnostic_interpolate!(b,rhs)
end

function RBSteady.diagnostic_residual!(b,nlop::SpaceTimeParamOperator,u)
  RBSteady.diagnostic_residual!(b,nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

function RBSteady.diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::SplitTransientRBOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  )

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

  RBSteady.diagnostic_interpolate!(A,lhss)
end

function RBSteady.diagnostic_jacobian!(A,nlop::SpaceTimeParamOperator,u)
  RBSteady.diagnostic_jacobian!(A,nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

function RBSteady.hr_diagnostics(c::TupOfAffineContribution)
  Tuple(RBSteady.hr_diagnostics(v) for v in c)
end

function RBSteady.hr_error_jac(
  op::RBOperator,
  jac::TupOfArrayContribution,
  μ::AbstractRealisation
  )

  test  = get_test(op)
  trial = get_trial(op)
  lhs = get_lhs(op)
  nlop = parameterise(op,μ)
  red_jac = diagnostic_jacobian(nlop,u)

  err = ()
  for (i,(jaci,lhsi)) in enumerate(zip(jac,lhs))
    erri = ()
    for (jaci_t,ai_t,fecache_t,hypred_t) in zip(
      get_contributions(jaci),
      get_contributions(lhsi),
      get_contributions(red_jac.fecache),
      get_contributions(red_jac.hypred)
      )

      erri = (erri...,hr_error_jac(trial,test,jaci_t,ai_t,fecache_t,hypred_t))
    end 
    err = (err...,erri)
  end
  
  return err
end

function RBSteady.save_jacobians(dir,feop::ODEParamOperator,jacs::Tuple;label="")
  for (i,jac) in enumerate(jacs)
    save_jacobians(dir,feop,jac;label=_get_label(label,i))
  end
end

function RBSteady.save_jacobians(dir,feop::LinearNonlinearODEParamOperator,jacs::Tuple;label="")
  @assert length(jacs) == 2
  save_jacobians(dir,feop.linear_op,jacs[1];label=string(label,"_lin"))
  save_jacobians(dir,feop.nonlinear_op,jacs[2];label=string(label,"_nonlin"))
end

# utils 

function RBSteady.set_params(red::SteadyReduction;kwargs...)
  SteadyReduction(RBSteady.set_params(red.reduction;kwargs...))
end

function RBSteady.set_params(red::KroneckerReduction;kwargs...)
  KroneckerReduction(
    map(r->RBSteady.set_params(r;kwargs...),red.reductions)
  )
end

function RBSteady.set_params(red::SequentialReduction;kwargs...)
  SequentialReduction(RBSteady.set_params(red.reduction;kwargs...))
end

for T in (:HighDimMDEIMHyperReduction,:HighDimSOPTHyperReduction,)
  @eval begin
    function RBSteady.set_params(red::$T;kwargs...)
      $T(RBSteady.set_params(red.reduction;kwargs...),red.combination)
    end

    function RBSteady.set_params(red::NTuple{N,$T};kwargs...) where N
      map(r->RBSteady.set_params(r;kwargs...),red)
    end
  end
end

function RBSteady.set_params(red::HighDimRBFHyperReduction;kwargs...)
  HighDimRBFHyperReduction(RBSteady.set_params(red.reduction;kwargs...),red.combination,red.strategy)
end

function RBSteady.set_params(red::NTuple{N,HighDimRBFHyperReduction};kwargs...) where N
  map(r->RBSteady.set_params(r;kwargs...),red)
end


