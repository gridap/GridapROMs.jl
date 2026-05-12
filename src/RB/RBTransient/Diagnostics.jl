function RBSteady.allocate_diagnostic_residual(nlop::SpaceTimeParamOperator,u)
  RBSteady.allocate_diagnostic_residual(nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

function RBSteady.allocate_diagnostic_residual(
  op::TransientGenericRBOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  rhs = get_rhs(op)
  RBSteady.allocate_dcontribution(rhs,r)
end

function RBSteady.allocate_diagnostic_residual(
  op::GenericRBOperator{O,T,B,<:HighDimNoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,B}

  rhs = get_rhs(op)
  b = allocate_residual(op,r,us,paramcache)
  b̂ = RBSteady.allocate_dcontribution(rhs,r)
  DiagnosticsContribution(b,b̂.coeff,b̂.hypred)
end

function RBSteady.allocate_dcontribution(
  a::TupOfAffineContribution,
  r::AbstractRealisation
  )

  fecache = map(ai -> RBSteady.allocate_coefficient(ai,r),a)
  coeff = map(ai -> RBSteady.allocate_coefficient(ai,r),a)
  hypred = map(ai -> RBSteady.allocate_hyper_reduction(ai,r),a)
  DiagnosticsContribution(fecache,coeff,hypred)
end

function RBSteady.allocate_diagnostic_jacobian(nlop::SpaceTimeParamOperator,u)
  RBSteady.allocate_diagnostic_jacobian(nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

function RBSteady.allocate_diagnostic_jacobian(
  op::TransientGenericRBOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  lhs = get_lhs(op)
  RBSteady.allocate_dcontribution(lhs,r)
end

function RBSteady.allocate_diagnostic_jacobian(
  op::GenericRBOperator{O,T,<:TupOfHighDimNoHRContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,B}

  lhs = get_lhs(op)
  A = allocate_jacobian(op,r,us,paramcache)
  Â = RBSteady.allocate_dcontribution(lhs,r)
  DiagnosticsContribution(A,Â.coeff,Â.hypred)
end

function RBSteady.diagnostic_interpolate!(
  cache::DiagnosticsContribution,
  a::TupOfAffineContribution
  )

  for (hi,ci,ai,fi) in zip(cache.hypred,cache.coeff,a,cache.fecache)
    RBSteady.diagnostic_interpolate!(DiagnosticsContribution(fi,ci,hi),ai)
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

function RBSteady.diagnostic_residual!(
  b::DiagnosticsContribution,
  op::GenericRBOperator{O,T,A,<:HighDimNoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,A}

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

  RBSteady.diagnostic_interpolate!(b,rhs)
end

function RBSteady.diagnostic_residual!(
  b::DiagnosticsContribution,
  op::GenericRBOperator{O,T,A,<:HighDimAffineHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,A}

  interpolate!(b.hypred,b.coeff,op.rhs,b.fecache)
end

function RBSteady.diagnostic_residual!(
  b::DiagnosticsContribution,
  op::GenericRBOperator{O,T,A,<:HighDimRBFContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,A}

  interpolate!(b.hypred,b.coeff,op.rhs,r)
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
  RBSteady.diagnostic_jacobian!(A,nlop.op,nlop.r,nlop.usx,nlop.ws,nlop.paramcache)
end

function RBSteady.diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::GenericRBOperator{O,T,<:TupOfHighDimNoHRContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B}

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

  RBSteady.diagnostic_interpolate!(A,lhss)
end

function RBSteady.diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::GenericRBOperator{O,T,<:TupOfHighDimAffineHRContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B}

  interpolate!(A.hypred,A.coeff,op.lhs,A.fecache)
end

function RBSteady.diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::GenericRBOperator{O,T,<:TupOfHighDimRBFContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B}

  for (hi,ci,ai) in zip(A.hypred,A.coeff,op.lhs)
    interpolate!(hi,ci,ai,r)
  end
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

for T in (:HighDimMDEIMProjection,:HighDimSOPTProjection)
  @eval begin
    function RBSteady.check_interpolation(res,a::$T{<:ReducedVecProjection},fecache)
      msg = "fecache mismatch at interpolation points"
      interp = get_interpolation(a)
      rows = get_interpolation_rows(interp)
      indices_time = get_indices_time(interp)
      style = get_domain_style(interp)

      bdata = if style isa KroneckerDomain
        get_at_kron_domain(res,rows,indices_time)
      else
        @check style isa SequentialDomain "Unsupported transient domain style"
        get_at_seq_domain(res,rows,indices_time)
      end

      @check isapprox(get_all_data(fecache),get_all_data(bdata);rtol=1e-8) msg
      return true
    end

    function RBSteady.check_interpolation(jac,a::$T{<:ReducedMatProjection},fecache)
      msg = "fecache mismatch at interpolation points"
      interp = get_interpolation(a)
      rows = get_interpolation_rows(interp)
      cols = get_interpolation_cols(interp)
      indices_time = get_indices_time(interp)
      style = get_domain_style(interp)

      Adata = if style isa KroneckerDomain
        get_at_kron_domain(jac,(rows,cols),indices_time)
      else
        @check style isa SequentialDomain "Unsupported transient domain style"
        get_at_seq_domain(jac,(rows,cols),indices_time)
      end

      @check isapprox(get_all_data(fecache),get_all_data(Adata);rtol=1e-8) msg
      return true
    end
  end
end

function RBSteady.set_params(red::SteadyReduction;kwargs...)
  SteadyReduction(RBSteady.set_params(red.reduction;kwargs...))
end

function RBSteady.set_params(red::KroneckerReduction;kwargs...)
  KroneckerReduction(map(r->RBSteady.set_params(r;kwargs...),red.reductions))
end

function RBSteady.set_params(red::SequentialReduction;kwargs...)
  SequentialReduction(RBSteady.set_params(red.reduction;kwargs...))
end

function RBSteady.set_params(red::HighDimMDEIMHyperReduction;kwargs...)
  HighDimMDEIMHyperReduction(RBSteady.set_params(red.reduction;kwargs...),red.combination)
end

function RBSteady.set_params(red::HighDimSOPTHyperReduction;kwargs...)
  HighDimSOPTHyperReduction(RBSteady.set_params(red.reduction;kwargs...),red.combination)
end

function RBSteady.set_params(red::HighDimRBFHyperReduction;kwargs...)
  HighDimRBFHyperReduction(RBSteady.set_params(red.reduction;kwargs...),red.combination,red.strategy)
end

function RBSteady.set_params(red::NTuple{N,Reduction};kwargs...) where N
  map(r->RBSteady.set_params(r;kwargs...),red)
end


