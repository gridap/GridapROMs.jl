abstract type TransientInterpolation <: Interpolation end

Interpolation(::HighOrderMDEIMHyperReduction,args...) = MDEIMInterpolation(args...)
Interpolation(::HighOrderRBFHyperReduction,args...) = RBFInterpolation(args...)

get_domain_style(a::TransientInterpolation) = get_domain_style(get_integration_domain(a))

get_indices_time(a::TransientInterpolation) = get_indices_time(get_integration_domain(a))
get_itimes(a::TransientInterpolation,args...) = get_itimes(get_integration_domain(a),args...)
get_param_itimes(a::TransientInterpolation,args...) = get_param_itimes(get_integration_domain(a),args...)

get_itimes(a::TransientInterpolation,common_ids::Range1D) = get_itimes(a,common_ids.parent)
get_param_itimes(a::TransientInterpolation,common_ids::Range1D) = get_param_itimes(a,common_ids.parent)

# EIM interpolation

struct TransientMDEIMInterpolation{A,B} <: TransientInterpolation
  interpolation::A
  domain::B
end

TransientMDEIMInterpolation() = TransientMDEIMInterpolation(nothing,nothing)

function TransientMDEIMInterpolation(basis::Projection,trian::Triangulation,test::RBSpace)
  (rows,indices_time),interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = vector_domain(reduction,trian,test,rows,indices_time)
  MDEIMInterpolation(interp,domain)
end

function TransientMDEIMInterpolation(basis::Projection,trian::Triangulation,trial::RBSpace,test::RBSpace)
  ((rows,cols),indices_time),interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = matrix_domain(reduction,trian,trial,test,rows,cols,indices_time)
  TransientMDEIMInterpolation(interp,domain)
end

RBSteady.get_interpolation(a::TransientMDEIMInterpolation) = a.interpolation
RBSteady.get_integration_domain(a::TransientMDEIMInterpolation) = a.domain

function FESpaces.interpolate!(cache::AbstractArray,a::TransientMDEIMInterpolation,b::AbstractParamMatrix)
  ldiv!(cache,a,vec(b))
  cache
end

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

# RBF interpolation: #TODO

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

for f in (:get_cellids_rows,:get_cellids_cols)
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

function get_integration_cells(a::TransientBlockInterpolation,args...)
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

function get_owned_icells(a::TransientBlockInterpolation,args...)
  cells = get_integration_cells(a,args...)
  get_owned_icells(a,cells)
end

function Arrays.return_cache(::typeof(get_owned_icells),a::TransientBlockInterpolation,cells)
  cache = get_owned_icells(testitem(a),cells)
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function get_owned_icells(a::TransientBlockInterpolation,cells::AbstractVector)
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

const AbstractMDEIMInterp = Union{MDEIMInterpolation,TransientBlockInterpolation{<:MDEIMInterpolation}}

const AbstractRBFInterp = Union{RBFInterpolation,TransientBlockInterpolation{<:RBFInterpolation}}
