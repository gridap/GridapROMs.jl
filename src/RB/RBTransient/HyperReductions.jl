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

function RBSteady.reduced_jacobian!(cache,red,op::ODEParamOperator,red_trial,red_test)
  ContributionTuple(ntuple(length(cache)) do k
    contribution(get_domains_jac(op)[k]) do trian
      jac = RBSteady._jacobian_for_trian!(cache[k],op,trian)
      RBSteady.reduced_form(red[k],jac,trian,red_trial,red_test)
    end
  end)
end

for f in (:reduced_residual!,:reduced_jacobian!)
  @eval begin
    function RBSteady.$f(cache,red,op::LinearNonlinearODEParamOperator,args...)
      op_lin = get_linear_part(op)
      op_nlin = get_nonlinear_part(op)
      cache_lin,cache_nlin = cache
      red_lin = RBSteady.$f(cache_lin,red,op_lin,args...)
      red_nlin = RBSteady.$f(cache_nlin,red,op_nlin,args...)
      Bundle(red_lin,red_nlin)
    end
  end
end

# utils

function RBSteady._prepare_residual(
  op::ODEParamOperator,
  c::TimeCombination,
  r::TransientRealisation,
  u::AbstractVector,
  us0::Tuple{Vararg{AbstractVector}}
  )

  usx,paramcache = ParamODEs.allocate_spacetime_cache(c,op,r,u,us0)

  _time_combination!(usx,op,c,u,us0)
  ParamODEs.to_stencil!(r,c)
  uh = ODEs._make_uh_from_us(op,usx,paramcache.trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,r)
  μ,t = get_params(r),get_times(r)
  dc = get_res(op)(μ,t,uh,v)
  vecdata = collect_cell_vector(test,dc)
  b = allocate_vector(assem,vecdata)
  ParamODEs.from_stencil!(r,c)

  return b,dc,assem,r
end

function RBSteady._prepare_residual(
  op::LinearNonlinearODEParamOperator,
  c::TimeCombination,
  r::TransientRealisation,
  u::AbstractVector,
  us0::Tuple{Vararg{AbstractVector}}
  )

  op_lin = get_linear_part(op)
  op_nlin = get_nonlinear_part(op)

  usx,paramcache = ParamODEs.allocate_spacetime_cache(c,op,r,u,us0)

  time_combination!(usx,c,u,us0)
  ParamODEs.to_stencil!(r,c)
  uh = ODEs._make_uh_from_us(op,usx,paramcache.trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,r)
  μ,t = get_params(r),get_times(r)
  dc_lin = get_res(op_lin)(μ,t,uh,v)
  dc_nlin = get_res(op_nlin)(μ,t,uh,v)
  vecdata = collect_cell_vector(test,dc_lin+dc_nlin)
  b = allocate_vector(assem,vecdata)
  ParamODEs.from_stencil!(r,c)

  data_lin = b,dc_lin,assem,r
  data_nlin = b,dc_nlin,assem,r

  return data_lin,data_nlin
end

function RBSteady._prepare_residual(
  solver::RBSolver,
  op::ODEParamOperator,
  s::AbstractSnapshots
  )

  c = TimeCombination(solver)
  sres = select_snapshots(s,RBSteady.res_params(solver))
  r = get_realisation(sres)
  u = get_param_data(sres)
  us0 = get_initial_param_data(sres)
  RBSteady._prepare_residual(op,c,r,u,us0)
end

function RBSteady._prepare_jacobian(
  op::ODEParamOperator,
  c::TimeCombination,
  r::TransientRealisation,
  u::AbstractVector,
  us0::Tuple{Vararg{AbstractVector}}
  )

  usx,paramcache = ParamODEs.allocate_spacetime_cache(c,op,r,u,us0)

  _time_combination!(usx,op,c,u,us0)
  ParamODEs.to_stencil!(r,c)
  uh = ODEs._make_uh_from_us(op,usx,paramcache.trial)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,r)
  μ,t = get_params(r),get_times(r)
  jacs = get_jacs(op)

  dcs = ntuple(k -> jacs[k](μ,t,uh,du,v),get_order(op)+1)
  matdata = collect_cell_matrix(trial,test,sum(dcs))
  A = allocate_matrix(assem,matdata)
  data = ntuple(k -> (A,dcs[k],assem,r),get_order(op)+1)
  ParamODEs.from_stencil!(r,c)

  return data
end

function RBSteady._prepare_jacobian(
  op::LinearNonlinearODEParamOperator,
  c::TimeCombination,
  r::TransientRealisation,
  u::AbstractVector,
  us0::Tuple{Vararg{AbstractVector}}
  )

  op_lin = get_linear_part(op)
  op_nlin = get_nonlinear_part(op)

  usx,paramcache = ParamODEs.allocate_spacetime_cache(c,op,r,u,us0)

  time_combination!(usx,c,u,us0)
  ParamODEs.to_stencil!(r,c)
  uh = ODEs._make_uh_from_us(op,usx,paramcache.trial)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)
  assem = get_param_assembler(op,r)
  μ,t = get_params(r),get_times(r)
  jacs_lin = get_jacs(op_lin)
  jacs_nlin = get_jacs(op_nlin)

  dcs_lin = ntuple(k -> jacs_lin[k](μ,t,uh,du,v),get_order(op_lin)+1)
  dcs_nlin = ntuple(k -> jacs_nlin[k](μ,t,uh,du,v),get_order(op_nlin)+1)
  matdata = collect_cell_matrix(trial,test,sum(dcs_lin)+sum(dcs_nlin))
  A = allocate_matrix(assem,matdata)

  data_lin = ntuple(k -> (A,dcs_lin[k],assem,r),get_order(op_lin)+1)
  data_nlin = ntuple(k -> (A,dcs_nlin[k],assem,r),get_order(op_nlin)+1)
  ParamODEs.from_stencil!(r,c)

  return data_lin,data_nlin
end

function RBSteady._prepare_jacobian(
  solver::RBSolver,
  op::ODEParamOperator,
  s::AbstractSnapshots
  )

  c = TimeCombination(solver)
  sjac = select_snapshots(s,RBSteady.jac_params(solver))
  r = get_realisation(sjac)
  u = get_param_data(sjac)
  us0 = get_initial_param_data(sjac)
  RBSteady._prepare_jacobian(op,c,r,u,us0)
end

function _time_combination!(usx,op::ODEParamOperator,c,u,us0)
  time_combination!(usx,c,u,us0)
end

function _time_combination!(usx,op::ODEParamOperator{LinearParamODE},c,u,us0)
  zero_time_combination!(usx,c,us0)
end