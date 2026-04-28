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

# utils 

function RBSteady.set_params(red::SteadyReduction;kwargs...)
  SteadyReduction(RBSteady.set_params(red.reduction;kwargs...))
end

function RBSteady.set_params(red::KroneckerReduction;kwargs...)
  KroneckerReduction(
    RBSteady.set_params(red.reduction_space;kwargs...),
    RBSteady.set_params(red.reduction_time;kwargs...)
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


