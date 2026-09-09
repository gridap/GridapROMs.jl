get_indices_time(a::Interpolation) = Int[]
get_itimes(a::Interpolation,ids::AbstractVector) = Int[]
get_locations(a::Interpolation,ids::Range2D) = range_2d(ids.axis1,Int[])

get_itimes(a::Interpolation,ids::Range1D) = error("should not be here")
get_locations(a::Interpolation,ids::Range1D) = get_locations(a,ids.parent)

function FESpaces.interpolate!(cache::AbstractArray,a::Interpolation,b::AbstractMatrix)
  ldiv!(cache,a,vec(b))
  cache
end

function RBSteady.Interpolation(red::HighDimNoHyperReduction,trian,args...)
  n = num_cells(trian)
  cells = collect(Int32,1:n)
  FullInterpolation(cells)
end

# EIM interpolation

const TransientGreedyInterpolation{A,B} = GreedyInterpolation{A,B}

for (T,f) in zip((:HighDimDEIMHyperReduction,:HighDimSOPTHyperReduction),
                 (:DEIM,:SOPT))
  @eval begin
    function RBSteady.Interpolation(red::$T,a::TransientProjection,trian,test)
      (rows,indices_time),interp = $f(a)
      factor = lu(interp)
      domain = IntegrationDomain(typeof(a),trian,test,rows,indices_time)
      GreedyInterpolation(factor,domain)
    end

    function RBSteady.Interpolation(red::$T,a::TransientProjection,trian,trial,test)
      ((rows,cols),indices_time),interp = $f(a)
      factor = lu(interp)
      domain = IntegrationDomain(typeof(a),trian,trial,test,rows,cols,indices_time)
      GreedyInterpolation(factor,domain)
    end
  end
end

get_domain_style(a::TransientGreedyInterpolation) = get_domain_style(a.domain)
get_indices_time(a::TransientGreedyInterpolation) = get_indices_time(a.domain)
get_itimes(a::TransientGreedyInterpolation,ids::Union{Vector,Range2D}) = get_itimes(a.domain,ids)
get_locations(a::TransientGreedyInterpolation,ids::Vector) = get_locations(a.domain,ids)

function get_locations(a::TransientGreedyInterpolation{A,<:KroneckerIntegrationDomain},ids::Range2D) where A
  common_param_ids = ids.axis1
  common_time_ids = ids.axis2
  local_itime_ids = get_itimes(a,common_time_ids)
  locations = range_2d(common_param_ids,local_itime_ids,length(common_param_ids))
  return locations
end

function get_locations(a::TransientGreedyInterpolation{A,<:SequentialIntegrationDomain},ids::Range2D) where A
  dofs = get_interpolation_dofs(a)
  slocations = get_iudof_to_idof(dofs)
  common_param_ids = ids.axis1
  common_time_ids = ids.axis2
  local_itime_ids = get_itimes(a,common_time_ids)
  tlocations = range_2d(common_param_ids,local_itime_ids,length(common_param_ids))
  return (slocations,tlocations)
end

# RBF interpolation

const TransientRBFInterpolation{A} = RBFInterpolation{A}

for (T,f) in zip(
  (:KroneckerProjection,:SequentialProjection),
  (:get_at_kron_domain,:get_at_seq_domain)
  )
  @eval begin
    function RBSteady.Interpolation(red::HighDimRBFHyperReduction,a::$T,s::TransientSnapshots)
      strategy = RBSteady.interp_strategy(red)
      inds,interp = DEIM(a)
      factor = lu(interp)
      r = get_params(get_realisation(s))
      red_data = $f(s,inds...)
      coeff = parameterise(allocate_in_domain(a),r)
      ldiv!(coeff,factor,red_data)
      interp = Interpolator(r,coeff,strategy)
      RBFInterpolation(interp)
    end
  end
end

# multi field

const TransientBlockInterpolation{N} = BlockInterpolation{N}

function get_domain_style(a::TransientBlockInterpolation)
  get_domain_style(first(a.interp))
end

function get_indices_time(a::TransientBlockInterpolation{N}) where N
  map(get_indices_time,a.interp)
end

function get_itimes(a::TransientBlockInterpolation{N},ids::Union{Vector,Range2D}) where N
  map(itp -> get_itimes(itp,ids),a.interp)
end

function get_locations(a::TransientBlockInterpolation{N},ids::Range2D) where N
  map(itp -> get_locations(itp,ids),a.interp)
end

# API

function get_at_kron_domain(
  s::TransientSnapshots,
  rows::AbstractVector{<:Integer},
  indices_time::AbstractVector{<:Integer}
  )

  data = flatten(s)
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

function get_at_seq_domain(
  s::TransientSnapshots,
  rows::AbstractVector{<:Integer},
  indices_time::AbstractVector{<:Integer}
  )

  @check length(rows) == length(indices_time)
  data = flatten(s)
  datav = zeros(eltype(s),length(rows),num_params(s))
  for i in CartesianIndices(datav)
    datav[i] = data[rows[i.I[1]],i.I[2],indices_time[i.I[1]]]
  end
  ConsecutiveParamArray(datav)
end

for f in (:get_at_kron_domain,:get_at_seq_domain)
  @eval begin
    function $f(
      s::TransientSparseSnapshots,
      rowscols::Tuple,
      indices_time::AbstractVector{<:Integer}
      )

      rows,cols = rowscols
      sparsity = get_sparsity(get_dof_map(s))
      inds = sparsify_split_indices(rows,cols,sparsity)
      $f(s,inds,indices_time)
    end
  end
end