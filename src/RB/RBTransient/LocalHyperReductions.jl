function get_domain_style(a::LocalInterpolation)
  vals = local_vals(a)
  isempty(vals) && return KroneckerDomain()
  style = get_domain_style(vals[1])
  @check all(get_domain_style(v) == style for v in vals)
  style
end

function get_indices_time(a::LocalInterpolation)
  vals = local_vals(a)
  isempty(vals) && return Int[]
  time_ids = get_indices_time(vals[1])
  @check all(get_indices_time(v) == time_ids for v in vals)
  time_ids
end

function get_itimes(a::LocalInterpolation,ids::Union{Vector,Range2D})
  vals = local_vals(a)
  isempty(vals) && return Int[]
  itime_ids = get_itimes(vals[1],ids)
  @check all(get_itimes(v,ids) == itime_ids for v in vals)
  itime_ids
end

function get_locations(a::LocalInterpolation,ids::Range2D)
  vals = local_vals(a)
  isempty(vals) && return range_2d(ids.axis1,Int[])
  locations = get_locations(vals[1],ids)
  @check all(get_locations(v,ids) == locations for v in vals)
  locations
end

const TupOfLocalHRContribution = Tuple{Vararg{LocalHRContribution}}

function RBSteady.get_local(a::TupOfAffineContribution,μ::AbstractVector)
  map(a -> get_local(a,μ),a)
end