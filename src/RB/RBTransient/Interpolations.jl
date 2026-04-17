get_indices_time(a::Interpolation) = Int[]
get_itimes(a::Interpolation,ids::AbstractVector) = Int[]
get_param_itimes(a::Interpolation,ids::Range2D) = range_2d(ids.axis1,Int[])

get_itimes(a::Interpolation,common_ids::Range1D) = error("should not be here")
get_param_itimes(a::Interpolation,common_ids::Range1D) = get_param_itimes(a,common_ids.parent)

function FESpaces.interpolate!(cache::AbstractArray,a::Interpolation,b::AbstractMatrix)
  ldiv!(cache,a,vec(b))
  cache
end

# EIM interpolation

const TransientGreedyInterpolation{A,B} = GreedyInterpolation{A,B}

for (T,f) in zip((:HighDimMDEIMHyperReduction,:HighDimSOPTHyperReduction),
                 (:empirical_interpolation,:s_opt))
  @eval begin
    function RBSteady.Interpolation(
      red::$T,
      basis::TransientProjection,
      trian::Triangulation,
      test::RBSpace
      )

      (rows,indices_time),interp = $f(basis)
      factor = lu(interp)
      domain = IntegrationDomain(typeof(basis),trian,test,rows,indices_time)
      GreedyInterpolation(factor,domain)
    end

    function RBSteady.Interpolation(
      red::$T,
      basis::TransientProjection,
      trian::Triangulation,
      trial::RBSpace,
      test::RBSpace
      )

      ((rows,cols),indices_time),interp = $f(basis)
      factor = lu(interp)
      domain = IntegrationDomain(typeof(basis),trian,trial,test,rows,cols,indices_time)
      GreedyInterpolation(factor,domain)
    end
  end
end

get_domain_style(a::TransientGreedyInterpolation) = get_domain_style(a.domain)
get_indices_time(a::TransientGreedyInterpolation) = get_indices_time(a.domain)
get_itimes(a::TransientGreedyInterpolation,args...) = get_itimes(a.domain,args...)
get_param_itimes(a::TransientGreedyInterpolation,args...) = get_param_itimes(a.domain,args...)

function get_param_itimes(a::TransientGreedyInterpolation,common_ids::Range2D)
  common_param_ids = common_ids.axis1
  common_time_ids = common_ids.axis2
  local_itime_ids = get_itimes(a,common_time_ids)
  locations = range_2d(common_param_ids,local_itime_ids,length(common_param_ids))
  return locations
end

# RBF interpolation

const TransientRBFInterpolation{A} = RBFInterpolation{A}

for (T,f) in zip((:KroneckerProjection,:SequentialProjection),
                 (:get_at_kron_domain,:get_at_seq_domain))
  @eval begin
    function RBSteady.Interpolation(
      strategy::AbstractRadialBasis,
      a::$T,
      s::TransientSnapshots
      )

      inds,interp = empirical_interpolation(a)
      factor = lu(interp)
      r = get_realisation(s)
      red_data = $f(s,inds...)
      coeff = allocate_coefficient(a,r)
      ldiv!(coeff,factor,red_data)
      interp = Interpolator(get_params(r),coeff,strategy)
      RBFInterpolation(interp)
    end
  end
end

# multi field

const TransientBlockInterpolation{N} = BlockInterpolation{N}

get_domain_style(a::TransientBlockInterpolation) = get_domain_style(testitem(a))

function get_indices_time(a::TransientBlockInterpolation{N}) where N
  array = Array{Any,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      array[i] = get_indices_time(a.interp[i])
    end
  end
  ArrayBlock(array,a.touched)
end

function get_itimes(a::TransientBlockInterpolation{N},ids::AbstractVector) where N
  array = Array{Any,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      array[i] = get_itimes(a.interp[i],ids)
    end
  end
  ArrayBlock(array,a.touched)
end

function get_param_itimes(a::TransientBlockInterpolation{N},ids::Range2D) where N
  array = Array{Any,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      array[i] = get_param_itimes(a.interp[i],ids)
    end
  end
  ArrayBlock(array,a.touched)
end

# API

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
        datav[(j-1)*length(rows)+i,k] = data[row,k,itime]
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
        datav[(j-1)*length(inds)+i,k] = data[ind,k,itime]
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

  rows,cols = rowscols
  @check length(rows) == length(indices_time)
  sparsity = get_sparsity(get_dof_map(s))
  inds = sparsify_split_indices(rows,cols,sparsity)
  data = get_all_data(s)
  datav = zeros(eltype(s),length(inds),num_params(s))
  for i in CartesianIndices(datav)
    datav[i] = data[inds[i.I[1]],i.I[2],indices_time[i.I[1]]]
  end
  ConsecutiveParamArray(datav)
end