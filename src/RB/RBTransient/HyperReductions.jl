function RBSteady.HRProjection(red::TransientHyperReduction,s,trian,trial,test)
  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial,get_time_combination(red))
  interp = Interpolation(red,basis,trian,trial,test)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.HRProjection(red::TransientNoHyperReduction,s,trian,test)
  T = get_dof_value_type(test)
  nrows = num_reduced_dofs(test)
  basis = ReducedProjection(zeros(T,nrows,1))
  interp = Interpolation(red,trian)
  return HRProjection(basis,red,interp)
end

function RBSteady.HRProjection(red::TransientNoHyperReduction,s,trian,trial,test)
  T = get_dof_value_type(trial)
  nrows = num_reduced_dofs(test)
  ncols = num_reduced_dofs(trial)
  basis = ReducedProjection(zeros(T,nrows,ncols,1))
  interp = Interpolation(red,trian)
  return HRProjection(basis,red,interp)
end

function RBSteady.HRProjection(red::TransientAffineHyperReduction,s,trian,test)
  basis = GalerkinProjectable(s)
  proj_basis = project(test,basis)
  interp = Interpolation(red)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.HRProjection(red::TransientAffineHyperReduction,s,trian,trial,test)
  basis = GalerkinProjectable(s)
  proj_basis = project(test,basis,trial,get_time_combination(red))
  interp = Interpolation(red)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.HRProjection(red::TransientRBFHyperReduction,s,trian,test)
  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis)
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.HRProjection(red::TransientRBFHyperReduction,s,trian,trial,test)
  basis = projection(get_reduction(red),s)
  proj_basis = project(test,basis,trial,get_time_combination(red))
  interp = Interpolation(red,basis,s)
  return HRProjection(proj_basis,red,interp)
end

function RBSteady.reduced_jacobian(
  red::Tuple{Vararg{Reduction}},
  trial::RBSpace,
  test::RBSpace,
  contribs::Tuple{Vararg{Any}}
  )

  a = ()
  for i in eachindex(contribs)
    a = (a...,reduced_jacobian(red[i],trial,test,contribs[i]))
  end
  return ContributionTuple(a)
end

const TransientNoHRProjection{A<:Projection} = HRProjection{<:TransientNoHyperReduction,A}
const TransientAffineHRProjection{A<:Projection} = HRProjection{<:TransientAffineHyperReduction,A}
const TransientDEIMProjection{A<:Projection} = HRProjection{<:TransientDEIMHyperReduction,A}
const TransientSOPTProjection{A<:Projection} = HRProjection{<:TransientSOPTHyperReduction,A}
const TransientRBFProjection{A<:Projection} = HRProjection{<:TransientRBFHyperReduction,A}

function FESpaces.interpolate!(
  b̂::AbstractArray,
  coeff::AbstractArray,
  a::TransientNoHRProjection,
  x::AbstractArray
  )

  o = one(eltype2(b̂))
  axpy!(o,coeff,b̂)
  return b̂
end

RBSteady.allocate_coefficient(a::TransientNoHRProjection) = RBSteady.allocate_hyper_reduction(a)

function FESpaces.interpolate!(
  b̂::AbstractArray,
  coeff::AbstractArray,
  a::TransientAffineHRProjection,
  x::Any
  )

  o = one(eltype2(b̂))
  L = param_length(b̂)
  ϕ = get_basis(get_basis(a))
  axpy!(o,parameterise(ϕ,L),b̂)
  return b̂
end

const TransientNoHRContribution = AffineContribution{<:TransientNoHRProjection}
const TransientAffineHRContribution = AffineContribution{<:TransientAffineHRProjection}
const TransientDEIMContribution = AffineContribution{<:TransientDEIMProjection}
const TransientSOPTContribution = AffineContribution{<:TransientSOPTProjection}
const TransientRBFContribution = AffineContribution{<:TransientRBFProjection}

"""
    const AffineContributionTuple = ContributionTuple{<:AffineContribution,T} where T

Concrete (see [`ContributionTuple`](@ref)) replacement for what used to be a
raw `Tuple{Vararg{AffineContribution}}` -- one entry per time derivative
order in unsteady settings.
"""
const AffineContributionTuple = ContributionTuple{<:AffineContribution,T} where T
const TransientNoHRContributionTuple = ContributionTuple{<:TransientNoHRContribution,T} where T
const TransientAffineHRContributionTuple = ContributionTuple{<:TransientAffineHRContribution,T} where T
const TransientDEIMContributionTuple = ContributionTuple{<:TransientDEIMContribution,T} where T
const TransientSOPTContributionTuple = ContributionTuple{<:TransientSOPTContribution,T} where T
const TransientRBFContributionTuple = ContributionTuple{<:TransientRBFContribution,T} where T

function RBSteady.allocate_coefficient(a::AffineContributionTuple,b::ArrayContributionTuple)
  @check length(a) == length(b)
  coeffs = ()
  for (a,b) in zip(a,b)
    coeffs = (coeffs...,RBSteady.allocate_coefficient(a,b))
  end
  return ContributionTuple(coeffs)
end

function FESpaces.interpolate!(
  b̂::AbstractParamArray,
  coeff::ArrayContributionTuple,
  a::AffineContributionTuple,
  b::ArrayContributionTuple
  )

  @check length(coeff) == length(a) == length(b)
  fill!(b̂,zero(eltype(b̂)))
  for (ai,bi,ci) in zip(a,b,coeff)
    for (aval,bval,cval) in zip(get_contributions(ai),get_contributions(bi),get_contributions(ci))
      interpolate!(b̂,cval,aval,bval)
    end
  end
  return b̂
end

function FESpaces.interpolate!(
  b̂::AbstractParamArray,
  coeff::ArrayContributionTuple,
  a::AffineContributionTuple,
  r::AbstractRealisation
  )

  @check length(coeff) == length(a)
  fill!(b̂,zero(eltype(b̂)))
  for (ai,ci) in zip(a,coeff)
    for (aval,cval) in zip(get_contributions(ai),get_contributions(ci))
      interpolate!(b̂,cval,aval,r)
    end
  end
  return b̂
end

function FESpaces.interpolate!(cache::HRParamArray,a::AffineContributionTuple)
  interpolate!(cache.hypred,cache.coeff,a,cache.fecache)
end

function FESpaces.interpolate!(cache::HRParamArray,a::AffineContributionTuple,r::AbstractRealisation)
  interpolate!(cache.hypred,cache.coeff,a,r)
end

function RBSteady.allocate_hypred_cache(a::AffineContributionTuple,args...)
  fecache = map(ai -> RBSteady.allocate_coefficient(ai,args...),a)
  coeffs = map(ai -> RBSteady.allocate_coefficient(ai,args...),a)
  hypred = RBSteady.allocate_hyper_reduction(first(a),args...)
  return HRParamArray(fecache,coeffs,hypred)
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

function get_common_time_domain(a::AffineContributionTuple)
  union(map(get_common_time_domain,a)...)
end

function get_common_time_domain(a::BlockHRProjection...)
  time_ids = ()
  for ai in a
    for i in eachindex(ai)
      interpi = get_interpolation(ai[i])
      time_ids = (time_ids...,get_indices_time(interpi))
    end
  end
  union(time_ids...)
end
