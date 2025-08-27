RBSteady.Interpolation(::HighDimMDEIMHyperReduction,args...) = MDEIMInterpolation(args...)
RBSteady.Interpolation(::HighDimRBFHyperReduction,args...) = RBFInterpolation(args...)

get_domain_style(a::Interpolation) = get_domain_style(get_integration_domain(a))

get_indices_time(a::Interpolation) = get_indices_time(get_integration_domain(a))
get_itimes(a::Interpolation,args...) = get_itimes(get_integration_domain(a),args...)
get_param_itimes(a::Interpolation,args...) = get_param_itimes(get_integration_domain(a),args...)

get_itimes(a::Interpolation,common_ids::Range1D) = error("should not be here")
get_param_itimes(a::Interpolation,common_ids::Range1D) = get_param_itimes(a,common_ids.parent)

function FESpaces.interpolate!(cache::AbstractArray,a::Interpolation,b::AbstractMatrix)
  ldiv!(cache,a,vec(b))
  cache
end

# EIM interpolation

const TransientMDEIMInterpolation{A,B} = MDEIMInterpolation{A,B}

function RBSteady.MDEIMInterpolation(
  basis::TransientProjection,
  trian::Triangulation,
  test::RBSpace
  )

  (rows,indices_time),interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = IntegrationDomain(typeof(basis),trian,test,rows,indices_time)
  MDEIMInterpolation(factor,domain)
end

function RBSteady.MDEIMInterpolation(
  basis::TransientProjection,
  trian::Triangulation,
  trial::RBSpace,
  test::RBSpace
  )

  ((rows,cols),indices_time),interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = IntegrationDomain(typeof(basis),trian,trial,test,rows,cols,indices_time)
  MDEIMInterpolation(factor,domain)
end

function get_param_itimes(a::TransientMDEIMInterpolation,common_ids::Range2D)
  common_param_ids = common_ids.axis1
  common_time_ids = common_ids.axis2
  local_time_ids = get_indices_time(a)
  local_itime_ids = get_itimes(a,common_time_ids)
  locations = range_2d(common_param_ids,local_itime_ids,length(common_param_ids))
  return locations
end

# RBF interpolation

const TransientRBFInterpolation{A} = RBFInterpolation{A}

function RBSteady.RBFInterpolation(
  strategy::AbstractRadialBasis,
  a::KroneckerProjection,
  s::TransientSnapshots
  )

  inds,interp = empirical_interpolation(a)
  factor = lu(interp)
  r = get_realization(s)
  red_data = get_at_kron_domain(s,inds...)
  coeff = allocate_coefficient(a,r)
  ldiv!(coeff,factor,red_data)
  interp = Interpolator(get_params(r),coeff,strategy)
  RBFInterpolation(interp)
end

function RBSteady.RBFInterpolation(
  strategy::AbstractRadialBasis,
  a::SequentialProjection,
  s::TransientSnapshots
  )

  inds,interp = empirical_interpolation(a)
  factor = lu(interp)
  r = get_realization(s)
  red_data = get_at_seq_domain(s,inds...)
  coeff = allocate_coefficient(a,r)
  ldiv!(coeff,factor,red_data)
  interp = Interpolator(get_params(r),coeff,strategy)
  RBFInterpolation(interp)
end

function get_at_kron_domain(
  s::TransientSnapshots,
  rows::AbstractVector{<:Integer},
  indices_time::AbstractVector{<:Integer}
  )

  data = reshape(get_all_data(s),:,num_params(s),num_times(s))
  datav = zeros(eltype(s),length(rows)*length(indices_time),num_params(s))
  for (j,itime) in enumerate(indices_time)
    for (i,row) in enumerate(rows)
      for k in 1:num_params(s)
        datav[(j-1)*length(indices_time)+i,k] = data[row,k,itime]
      end
    end
  end
  ConsecutiveParamArray(datav)
end

function get_at_kron_domain(
  s::TransientSparseSnapshots,
  rowscols::Tuple,
  indices_time::AbstractVector{<:Integer}
  )

  rows,cols = rowscols
  sparsity = get_sparsity(get_dof_map(s))
  inds = sparsify_split_indices(rows,cols,sparsity)
  data = get_all_data(s)
  datav = zeros(eltype(s),length(inds)*length(indices_time),num_params(s))
  for (j,itime) in enumerate(indices_time)
    for (i,ind) in enumerate(inds)
      for k in 1:num_params(s)
        datav[(j-1)*length(indices_time)+i,k] = data[ind,k,itime]
      end
    end
  end
  ConsecutiveParamArray(datav)
end

function get_at_seq_domain(
  s::TransientSnapshots,
  rows::AbstractVector{<:Integer},
  indices_time::AbstractVector{<:Integer}
  )

  @check length(rows) == length(indices_time)
  data = reshape(get_all_data(s),:,num_params(s),num_times(s))
  datav = zeros(eltype(s),length(rows),num_params(s))
  for i in CartesianIndices(datav)
    datav[i] = data[rows[i.I[1]],i.I[2],indices_time[i.I[1]]]
  end
  ConsecutiveParamArray(datav)
end

function get_at_seq_domain(
  s::TransientSparseSnapshots,
  rowscols::Tuple,
  indices_time::AbstractVector{<:Integer}
  )

  @check length(rowscols) == length(indices_time)
  rows,cols = rowscols
  sparsity = get_sparsity(get_dof_map(s))
  inds = sparsify_split_indices(rows,cols,sparsity)
  data = get_all_data(s)
  datav = zeros(eltype(s),length(inds),num_params(s))
  for i in CartesianIndices(datav)
    datav[i] = data[inds[i.I[1]],i.I[2],indices_time[i.I[1]]]
  end
  ConsecutiveParamArray(datav)
end

# multi field

const TransientBlockInterpolation{I,N} = BlockInterpolation{I,N}

get_domain_style(a::TransientBlockInterpolation) = get_domain_style(testitem(a))

function Arrays.return_cache(::typeof(get_indices_time),a::TransientBlockInterpolation)
  cache = get_indices_time(testitem(a))
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function get_indices_time(a::TransientBlockInterpolation)
  cache = return_cache(get_itimes,a)
  for i in eachindex(a)
    if a.touched[i]
      cache[i] = get_itimes(a[i])
    end
  end
  return ArrayBlock(cache,a.touched)
end

for (f,T) in zip((:get_itimes,:get_param_itimes),(:AbstractVector,:Range2D))
  @eval begin
    function Arrays.return_cache(::typeof($f),a::TransientBlockInterpolation,ids::$T)
      cache = $f(testitem(a),ids)
      block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::TransientBlockInterpolation,ids::$T)
      cache = return_cache($f,a,ids)
      for i in eachindex(a)
        if a.touched[i]
          cache[i] = $f(a[i],ids)
        end
      end
      return ArrayBlock(cache,a.touched)
    end
  end
end
