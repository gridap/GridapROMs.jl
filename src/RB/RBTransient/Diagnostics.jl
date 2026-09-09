function RBSteady.allocate_diagnostic_residual(nlop::SpaceTimeParamOperator,u)
  RBSteady.allocate_diagnostic_residual(nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

function RBSteady.allocate_diagnostic_residual(
  op::TransientReducedOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  rhs = get_rhs(op)
  RBSteady.allocate_dcontribution(rhs,r)
end

function RBSteady.allocate_diagnostic_residual(
  op::TransientRBOperator{O,T,B,<:HighDimNoHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,B}

  rhs = get_rhs(op)
  b = allocate_residual(op.op,r,us,paramcache)
  b̂ = RBSteady.allocate_dcontribution(rhs,r)
  DiagnosticsContribution(b,b̂.coeff,b̂.hypred)
end

function RBSteady.allocate_dcontribution(
  a::TupOfAffineContribution,
  r::AbstractRealisation
  )

  fecache = map(ai -> RBSteady.allocate_coefficient(ai,r),a)
  coeff = map(ai -> RBSteady.allocate_coefficient(ai,r),a)
  hypred = ()
  for ai in a 
    hypredi = contribution(get_domains(ai)) do trian
      RBSteady.allocate_hyper_reduction(ai[trian],r)
    end
    hypred = (hypred...,hypredi)
  end
  DiagnosticsContribution(fecache,coeff,hypred)
end

function RBSteady.allocate_diagnostic_jacobian(nlop::SpaceTimeParamOperator,u)
  RBSteady.allocate_diagnostic_jacobian(nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

function RBSteady.allocate_diagnostic_jacobian(
  op::TransientReducedOperator,
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  )

  lhs = get_lhs(op)
  RBSteady.allocate_dcontribution(lhs,r)
end

function RBSteady.allocate_diagnostic_jacobian(
  op::TransientRBOperator{O,T,<:TupOfHighDimNoHRContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,B}

  lhs = get_lhs(op)
  A = allocate_jacobian(op.op,r,us,paramcache)
  Â = RBSteady.allocate_dcontribution(lhs,r)
  DiagnosticsContribution(A,Â.coeff,Â.hypred)
end

function RBSteady.diagnostic_interpolate!(
  cache::DiagnosticsContribution,
  a::TupOfAffineContribution
  )

  for (hi,ci,ai,fi) in zip(cache.hypred,cache.coeff,a,cache.fecache)
    for (ât,ft,ct,ht) in zip(
      get_contributions(ai),
      get_contributions(fi),
      get_contributions(ci),
      get_contributions(hi)
      )

      interpolate!(ht,ct,ât,ft)
    end
  end
end

function RBSteady.diagnostic_interpolate!(
  cache::DiagnosticsContribution,
  a::TupOfAffineContribution,
  r::TransientRealisation
  )

  for (hi,ci,ai) in zip(cache.hypred,cache.coeff,a)
    for (ât,ct,ht) in zip(
      get_contributions(ai),
      get_contributions(ci),
      get_contributions(hi)
      )

      interpolate!(ht,ct,ât,r)
    end
  end
end

function RBSteady.diagnostic_residual!(
  b::DiagnosticsContribution,
  op::SplitTransientReducedOperator,
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
    assemble_hr_array_add!(b_strian,vecdata)
  end

  RBSteady.diagnostic_interpolate!(b,rhs)
end

function RBSteady.diagnostic_residual!(b,nlop::SpaceTimeParamOperator,u)
  RBSteady.diagnostic_residual!(b,nlop.op,nlop.r,nlop.usx,nlop.paramcache)
end

function RBSteady.diagnostic_residual!(
  b::DiagnosticsContribution,
  op::TransientRBOperator{O,T,A,<:HighDimNoHRContribution},
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
  op::TransientRBOperator{O,T,A,<:HighDimAffineHRContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,A}

  RBSteady.diagnostic_interpolate!(b,op.rhs)
end

function RBSteady.diagnostic_residual!(
  b::DiagnosticsContribution,
  op::TransientRBOperator{O,T,A,<:HighDimRBFContribution},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  paramcache
  ) where {O,T,A}

  RBSteady.diagnostic_interpolate!(b,op.rhs,r)
end

function RBSteady.diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::SplitTransientReducedOperator,
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
      assemble_hr_array_add!(A_strian,matdata)
    end
  end

  RBSteady.diagnostic_interpolate!(A,lhss)
end

function RBSteady.diagnostic_jacobian!(A,nlop::SpaceTimeParamOperator,u)
  RBSteady.diagnostic_jacobian!(A,nlop.op,nlop.r,nlop.usx,nlop.ws,nlop.paramcache)
end

function RBSteady.diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::TransientRBOperator{O,T,<:TupOfHighDimNoHRContribution,B},
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

  RBSteady.diagnostic_interpolate!(A,lhss)
end

function RBSteady.diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::TransientRBOperator{O,T,<:TupOfHighDimAffineHRContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B}

  RBSteady.diagnostic_interpolate!(A,op.lhs)
end

function RBSteady.diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::TransientRBOperator{O,T,<:TupOfHighDimRBFContribution,B},
  r::TransientRealisation,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache
  ) where {O,T,B}

  RBSteady.diagnostic_interpolate!(A,op.lhs,r)
end

function RBSteady.hr_diagnostics(c::TupOfAffineContribution)
  Tuple(RBSteady.hr_diagnostics(v) for v in c)
end

function RBSteady.hr_error_res(
  test::SingleFieldRBSpace,
  res::TransientSnapshots,
  a::HRProjection,
  fecache::AbstractParamArray,
  hypred::AbstractParamVector
  )
  
  RBSteady.check_interpolation(res,a,fecache)

  red = get_style(a)
  c = get_time_combination(red)
  b̂ = get_basis(galerkin_projection(test,res,c))
  hrb̂ = get_all_data(hypred)

  compute_relative_error(b̂,hrb̂)
end

function RBSteady.hr_error_jac(
  trial::SingleFieldRBSpace,
  test::SingleFieldRBSpace,
  jac::TransientSnapshots,
  a::HRProjection,
  fecache::AbstractParamArray,
  hypred::AbstractParamMatrix
  )
  
  RBSteady.check_interpolation(jac,a,fecache)

  μ = get_realisation(jac)
  red = get_style(a)
  c = get_time_combination(red)
  Â = get_basis(galerkin_projection(test,jac,trial,c))
  Â = reshape(permutedims(Â,(1,3,2)),:,num_params(μ))
  hrÂ = reshape(get_all_data(hypred),:,num_params(μ))

  compute_relative_error(Â,hrÂ)
end

function RBSteady.hr_error_res(
  c::TimeCombination,
  op::TransientReducedOperator{O},
  res::ArrayContribution,
  r::AbstractRealisation,
  u::AbstractVector,
  us0::Tuple{Vararg{AbstractVector}}
  ) where O

  test = get_test(op)
  rhs = get_rhs(op)

  ParamODEs.to_stencil!(r,c)
  paramcache = allocate_paramcache(op,r;evaluated=true)
  usx = if O == LinearParamODE
    zero_time_combination(c,u,us0)
  else
    time_combination(c,u,us0)
  end
  red_res = RBSteady.allocate_diagnostic_residual(op,r,usx,paramcache)
  RBSteady.diagnostic_residual!(red_res,op,r,usx,paramcache)
  ParamODEs.from_stencil!(r,c)

  err = ()
  for (res_t,a_t,fecache_t,hypred_t) in zip(
    get_contributions(res),
    get_contributions(rhs),
    get_contributions(red_res.fecache),
    get_contributions(red_res.hypred)
    )

    err = (err...,RBSteady.hr_error_res(test,res_t,a_t,fecache_t,hypred_t))
  end 
  
  return err
end

function RBSteady.hr_error_jac(
  c::TimeCombination,
  op::TransientReducedOperator{O},
  jac::TupOfArrayContribution,
  r::AbstractRealisation,
  u::AbstractVector,
  us0::Tuple{Vararg{AbstractVector}}
  ) where O

  test  = get_test(op)
  trial = get_trial(op)
  lhs = get_lhs(op)

  ws = ntuple(_ -> 1,Val{get_order(op)+1}())
  ParamODEs.to_stencil!(r,c)
  paramcache = allocate_paramcache(op,r;evaluated=true)
  usx = if O == LinearParamODE
    zero_time_combination(c,u,us0)
  else
    time_combination(c,u,us0)
  end
  red_jac = RBSteady.allocate_diagnostic_jacobian(op,r,usx,paramcache)
  RBSteady.diagnostic_jacobian!(red_jac,op,r,usx,ws,paramcache)
  ParamODEs.from_stencil!(r,c)

  err = ()
  for (jaci,lhsi,fecachei,hypredi) in zip(jac,lhs,red_jac.fecache,red_jac.hypred)
    erri = ()
    for (jaci_t,ai_t,fecachei_t,hypredi_t) in zip(
      get_contributions(jaci),
      get_contributions(lhsi),
      get_contributions(fecachei),
      get_contributions(hypredi)
      )
      
      erri = (erri...,RBSteady.hr_error_jac(trial,test,jaci_t,ai_t,fecachei_t,hypredi_t))
    end 
    err = (err...,erri)
  end
  
  return err
end

function RBSteady.hr_error(solver::GlobalRBSolver,op::TransientReducedOperator,res,jac,s)
  c = TimeCombination(solver)
  μ = get_realisation(s)
  u = get_param_data(s)
  us0 = get_initial_param_data(s)
  err_res = RBSteady.hr_error_res(c,op,res,μ,u,us0)
  err_jac = RBSteady.hr_error_jac(c,op,jac,μ,u,us0)
  return err_res,err_jac
end

function RBSteady.save_jacobians(dir,feop::ODEParamOperator,jacs::Tuple;label="")
  for (i,jac) in enumerate(jacs)
    save_jacobians(dir,feop,jac;label=_get_label(label,i))
  end
end

function RBSteady.load_jacobians(dir,feop::ODEParamOperator;label="")
  jacs = ()
  for i in 1:get_order(feop)+1
    dom_jaci = get_domains_jac(feop)[i]
    labi = _get_label(label,i,JACOBIANS_LABEL)
    jaci = load_contribution(dir,dom_jaci;label=labi)
    jacs = (jacs...,jaci)
  end
  return jacs
end

# utils 

for T in (:HighDimDEIMHyperReduction,:HighDimSOPTHyperReduction)
  @eval begin
    function RBSteady.check_interpolation(res,a::HRVecProjection{<:$T},fecache)
      msg = "fecache mismatch at interpolation points"
      interp = get_interpolation(a)
      rows = get_interpolation_dofs(interp)
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

    function RBSteady.check_interpolation(jac,a::HRMatProjection{<:$T},fecache)
      msg = "fecache mismatch at interpolation points"
      interp = get_interpolation(a)
      rows,cols = get_interpolation_dofs(interp)
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

for S in (:HRVecProjection,:HRMatProjection), T in (:HighDimRBFHyperReduction,:HighDimTrivialHyperReduction)
  @eval function RBSteady.check_interpolation(resjac,a::$S{<:$T},fecache)
    return true
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

function RBSteady.set_params(red::HighDimDEIMHyperReduction;kwargs...)
  HighDimDEIMHyperReduction(RBSteady.set_params(red.reduction;kwargs...),red.combination)
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
