abstract type TransientInterpolation <: Interpolation end

RBSteady.Interpolation(::HighOrderMDEIMHyperReduction,args...) = TransientMDEIMInterpolation(args...)
RBSteady.Interpolation(::HighOrderRBFHyperReduction,args...) = TransientRBFInterpolation(args...)

get_domain_style(a::TransientInterpolation) = get_domain_style(get_integration_domain(a))

get_indices_time(a::TransientInterpolation) = get_indices_time(get_integration_domain(a))
get_itimes(a::TransientInterpolation,args...) = get_itimes(get_integration_domain(a),args...)
get_param_itimes(a::TransientInterpolation,args...) = get_param_itimes(get_integration_domain(a),args...)

get_itimes(a::TransientInterpolation,common_ids::Range1D) = get_itimes(a,common_ids.parent)
get_param_itimes(a::TransientInterpolation,common_ids::Range1D) = get_param_itimes(a,common_ids.parent)

function FESpaces.interpolate!(cache::AbstractArray,a::TransientInterpolation,b::AbstractMatrix)
  ldiv!(cache,a,vec(b))
  cache
end

# EIM interpolation

struct TransientMDEIMInterpolation{A,B} <: TransientInterpolation
  interpolation::A
  domain::B
end

TransientMDEIMInterpolation() = TransientMDEIMInterpolation(nothing,nothing)

function TransientMDEIMInterpolation(basis::Projection,trian::Triangulation,test::RBSpace)
  (rows,indices_time),interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = vector_domain(typeof(basis),trian,test,rows,indices_time)
  MDEIMInterpolation(interp,domain)
end

function TransientMDEIMInterpolation(basis::Projection,trian::Triangulation,trial::RBSpace,test::RBSpace)
  ((rows,cols),indices_time),interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = matrix_domain(typeof(basis),trian,trial,test,rows,cols,indices_time)
  TransientMDEIMInterpolation(interp,domain)
end

RBSteady.get_interpolation(a::TransientMDEIMInterpolation) = a.interpolation
RBSteady.get_integration_domain(a::TransientMDEIMInterpolation) = a.domain

function RBSteady.reduced_triangulation(trian::Triangulation,a::TransientMDEIMInterpolation)
  red_cells = get_integration_cells(a)
  red_trian = view(trian,red_cells)
  return red_trian
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

struct TransientRBFInterpolation{A} <: TransientInterpolation
  interpolation::A
end

TransientRBFInterpolation() = TransientRBFInterpolation(nothing)

function TransientRBFInterpolation(
  red::RBFHyperReduction,
  a::TransientProjection,
  s::TransientSnapshots
  )

  TransientRBFInterpolation(interp_strategy(red),a,s)
end

function TransientRBFInterpolation(
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
  TransientRBFInterpolation(interp)
end

function TransientRBFInterpolation(
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
  TransientRBFInterpolation(interp)
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

struct TransientBlockInterpolation{I<:TransientInterpolation,N} <: TransientInterpolation
  interp::BlockInterpolation{I,N}
end

Base.ndims(a::TransientBlockInterpolation) = ndims(a.interp)
Base.size(a::TransientBlockInterpolation,args...) = size(a.interp,args...)
Base.axes(a::TransientBlockInterpolation,args...) = axes(a.interp,args...)
Base.length(a::TransientBlockInterpolation) = length(a.interp)
Base.eachindex(a::TransientBlockInterpolation) = eachindex(a.interp)

Base.getindex(a::TransientBlockInterpolation,i...) = getindex(a.interp,i...)
Base.setindex!(a::TransientBlockInterpolation,i...) = setindex!(a.interp,v,i...)

Arrays.testitem(a::TransientBlockInterpolation) = testitem(a.interp)

for f in (:(RBSteady.get_cellids_rows),:(RBSteady.get_cellids_cols))
  @eval begin
    function Arrays.return_cache(::typeof($f),a::TransientBlockInterpolation)
      block_cache = Array{Table,ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::TransientBlockInterpolation)
      cache = return_cache($f,a)
      for i in eachindex(a)
        if a.touched[i]
          cache[i] = $f(a[i])
        end
      end
      return ArrayBlock(cache,a.touched)
    end
  end
end

function Arrays.return_cache(::typeof(get_integration_cells),a::TransientBlockInterpolation,args...)
  ntouched = length(findall(a.touched))
  cache = get_integration_cells(testitem(a),args...)
  block_cache = Vector{typeof(cache)}(undef,ntouched)
  return block_cache
end

function RBSteady.get_integration_cells(a::TransientBlockInterpolation,args...)
  _union(a) = a
  _union(a,b) = union(a,b)
  _union(a::AppendedArray,b::AppendedArray) = lazy_append(union(a.a,b.a),union(a.b,b.b))
  _union(a,b,c...) = _union(_union(a,b),c...)

  cache = return_cache(get_integration_cells,a,args...)
  count = 0
  for i in eachindex(a)
    if a.touched[i]
      count += 1
      cache[count] = get_integration_cells(a[i],args...)
    end
  end
  return _union(cache...)
end

function RBSteady.get_owned_icells(a::TransientBlockInterpolation,args...)
  cells = get_integration_cells(a,args...)
  get_owned_icells(a,cells)
end

function Arrays.return_cache(::typeof(get_owned_icells),a::TransientBlockInterpolation,cells)
  cache = get_owned_icells(testitem(a),cells)
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function RBSteady.get_owned_icells(a::TransientBlockInterpolation,cells::AbstractVector)
  cache = return_cache(get_owned_icells,a,cells)
  for i in eachindex(a)
    if a.touched[i]
      cache[i] = get_owned_icells(a[i],cells)
    end
  end
  return ArrayBlock(cache,a.touched)
end

function Arrays.return_cache(::typeof(get_indices_time),a::BlockHRProjection)
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

for f in (:get_itimes,:get_param_itimes)
  @eval begin
    function Arrays.return_cache(::typeof($f),a::TransientBlockInterpolation,ids::AbstractArray)
      cache = $f(testitem(a),ids)
      block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::BlockHRProjection,ids::AbstractArray)
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

const AbstractTransientMDEIMInterp = Union{TransientMDEIMInterpolation,TransientBlockInterpolation{<:TransientMDEIMInterpolation}}

const AbstractTransientRBFInterp = Union{TransientRBFInterpolation,TransientBlockInterpolation{<:TransientRBFInterpolation}}
