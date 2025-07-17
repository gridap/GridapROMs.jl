function RBSteady.HRProjection(
  red::HighOrderHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial,get_combine(red))
  interp = Interpolation(red,basis,trian,trial,test)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.HRProjection(
  red::HighOrderRBFHyperReduction,
  s::Snapshots,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial,get_combine(red))
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.reduced_jacobian(
  red::Tuple{Vararg{Reduction}},
  trial::RBSpace,
  test::RBSpace,
  contribs::Tuple{Vararg{Any}})

  a = ()
  for i in eachindex(contribs)
    a = (a...,reduced_jacobian(red[i],trial,test,contribs[i]))
  end
  return a
end

const TupOfAffineContribution = Tuple{Vararg{AffineContribution}}

function RBSteady.allocate_coefficient(a::TupOfAffineContribution,b::TupOfArrayContribution)
  @check length(a) == length(b)
  coeffs = ()
  for (a,b) in zip(a,b)
    coeffs = (coeffs...,RBSteady.allocate_coefficient(a,b))
  end
  return coeffs
end

function FESpaces.interpolate!(
  b̂::AbstractParamArray,
  coeff::TupOfArrayContribution,
  a::TupOfAffineContribution,
  b::TupOfArrayContribution)

  @check length(coeff) == length(a) == length(b)
  fill!(b̂,zero(eltype(b̂)))
  for (ai,bi,ci) in zip(a,b,coeff)
    for (aval,bval,cval) in zip(get_contributions(ai),get_contributions(bi),get_contributions(ci))
      interpolate!(b̂,cval,aval,bval)
    end
  end
  return b̂
end

function FESpaces.interpolate!(cache::HRParamArray,a::TupOfAffineContribution)
  interpolate!(cache.hypred,cache.coeff,a,cache.fecache)
end

function RBSteady.allocate_hypred_cache(a::TupOfAffineContribution,args...)
  fecache = map(ai -> RBSteady.allocate_coefficient(ai,args...),a)
  coeffs = map(ai -> RBSteady.allocate_coefficient(ai,args...),a)
  hypred = RBSteady.allocate_hyper_reduction(first(a),args...)
  return hr_array(fecache,coeffs,hypred)
end

function get_common_time_domain(a::HRProjection...)
  time_ids = ()
  for ai in a
    interpi = get_interpolation(ai)
    time_ids = (time_ids...,get_indices_time(interpi))
  end
  union(time_ids...)
end

function get_common_time_domain(a::AffineContribution)
  get_common_time_domain(get_contributions(a)...)
end

function get_common_time_domain(a::TupOfAffineContribution)
  union(map(get_common_time_domain,a)...)
end

function get_common_time_domain(a::BlockHRProjection...)
  time_ids = ()
  for ai in a
    for i in eachindex(ai)
      if ai.touched[i]
        time_ids = (time_ids...,get_indices_time(ai[i]))
      end
    end
  end
  union(time_ids...)
end
