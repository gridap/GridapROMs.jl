get_indices_time(a::Interpolation) = Int[]
get_itimes(a::Interpolation,ids::AbstractVector) = Int[]
get_locations(a::Interpolation,ids::Range2D) = range_2d(ids.axis1,Int[])
get_interpolation_style(a::Interpolation) = KroneckerStyle()
get_itimes(a::Interpolation,ids::Range1D) = error("should not be here")
get_locations(a::Interpolation,ids::Range1D) = get_locations(a,ids.parent)

function RBSteady.Interpolation(red::TransientNoHyperReduction,trian,args...)
  n = num_cells(trian)
  cells = collect(Int32,1:n)
  FullInterpolation(cells)
end

abstract type InterpolationStyle end
struct KroneckerStyle <: InterpolationStyle end
struct SequentialStyle <: InterpolationStyle end

InterpolationStyle(x) = InterpolationStyle(typeof(x))
InterpolationStyle(::Type{T}) where T = @abstractmethod
InterpolationStyle(::Type{<:KroneckerProjection}) = KroneckerStyle()
InterpolationStyle(::Type{<:SequentialProjection}) = SequentialStyle()

struct TransientInterpolation{A,B,C} <: Interpolation
  style::A
  interp_space::B
  indices_time::C
end

RBSteady.get_integration_cells(a::TransientInterpolation) = get_integration_cells(a.interp_space)
RBSteady.get_cell_idofs(a::TransientInterpolation) = get_cell_idofs(a.interp_space)
RBSteady.get_interpolation_dofs(a::TransientInterpolation) = get_interpolation_dofs(a.interp_space)
RBSteady.get_owned_integration_cells(a::TransientInterpolation,args...) = get_owned_integration_cells(a.interp_space,args...)

function FESpaces.interpolate!(cache::AbstractArray,a::TransientInterpolation,b::AbstractArray)
  interpolate!(cache,a.interp_space,b)
end

get_interpolation_style(a::TransientInterpolation) = a.style
get_indices_time(a::TransientInterpolation) = a.indices_time

function get_itimes(i::TransientInterpolation,ids::AbstractVector)::Vector{Int}
  idsi = get_indices_time(i)
  filter(!isnothing,indexin(idsi,ids))
end

function get_itimes(i::TransientInterpolation,ids::Range2D)::Vector{Int}
  idsi = get_indices_time(i)
  filter(!isnothing,indexin(idsi,ids))
end

function get_locations(a::TransientInterpolation{KroneckerStyle},ids::Range2D)
  common_param_ids = ids.axis1
  common_time_ids = ids.axis2
  local_itime_ids = get_itimes(a,common_time_ids)
  locations = range_2d(common_param_ids,local_itime_ids,length(common_param_ids))
  return locations
end

function get_locations(a::TransientInterpolation{SequentialStyle},ids::Range2D)
  dofs = get_interpolation_dofs(a)
  slocations = get_iudof_to_idof(dofs)
  common_param_ids = ids.axis1
  common_time_ids = ids.axis2
  local_itime_ids = get_itimes(a,common_time_ids)
  tlocations = range_2d(common_param_ids,local_itime_ids,length(common_param_ids))
  return (slocations,tlocations)
end

# EIM interpolation

for (T,f) in zip((:TransientDEIMHyperReduction,:TransientSOPTHyperReduction),(:DEIM,:SOPT))
  @eval begin
    function RBSteady.Interpolation(red::$T,a::TransientProjection,args...)
      isnull(a) && return EmptyInterpolation()
      GreedyInterpolation(red,a,args...)
    end

    function RBSteady.GreedyInterpolation(red::$T,a::TransientProjection,trian,test)
      style = InterpolationStyle(a)
      (rows,indices_time),interp = $f(a)
      factor = lu(interp)
      domain = IntegrationDomain(trian,test,rows)
      interp = GreedyInterpolation(factor,domain)
      TransientInterpolation(style,interp,indices_time)
    end

    function RBSteady.GreedyInterpolation(red::$T,a::TransientProjection,trian,trial,test)
      style = InterpolationStyle(a)
      ((rows,cols),indices_time),interp = $f(a)
      factor = lu(interp)
      domain = IntegrationDomain(trian,trial,test,rows,cols)
      interp = GreedyInterpolation(factor,domain)
      TransientInterpolation(style,interp,indices_time)
    end
  end
end

# RBF interpolation

const TransientRBFInterpolation{A} = RBFInterpolation{A}

function RBSteady.Interpolation(red::TransientRBFHyperReduction,a::TransientProjection,args...)
  isnull(a) && return EmptyInterpolation()
  RBFInterpolation(red,a,args...)
end

for (T,f) in zip(
  (:KroneckerProjection,:SequentialProjection),
  (:get_at_kron_domain,:get_at_seq_domain)
  )
  @eval begin
    function RBSteady.RBFInterpolation(red::TransientRBFHyperReduction,a::$T,s::TransientSnapshots)
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

function RBSteady.get_at_domain(s::TransientSnapshots,i::Interpolation)
  dofs = get_interpolation_dofs(i)
  indices_time = get_indices_time(i)
  style = get_interpolation_style(i)
  if style isa KroneckerStyle
    get_at_kron_domain(s,dofs,indices_time)
  else
    get_at_seq_domain(s,dofs,indices_time)
  end
end

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
      dof_map = get_dof_map(s)
      inds = sparsify_split_indices(rows,cols,dof_map)
      $f(s,inds,indices_time)
    end
  end
end

# iurow_to_irow[id of a unique row] = ids of the entries of that row
# e.g. get_iurow_to_irow([1,10,100,10]) = [[1],[2,4],[3],[2,4]]
function get_iurow_to_irow(rows::AbstractVector)
  isempty(rows) && return Table(Int32[],Int32[1])
  rows_to_count = zeros(Int32,maximum(rows))
  for row in rows
    rows_to_count[row] += 1
  end

  ptrs = Vector{Int32}(undef,length(rows)+1)
  for (irow,row) in enumerate(rows)
    ptrs[irow+1] = rows_to_count[row]
  end
  length_to_ptrs!(ptrs)

  data = Vector{Int32}(undef,ptrs[end]-1)
  for (irow,row) in enumerate(rows)
    pini = ptrs[irow]
    count = 0
    for (jrow,_row) in enumerate(rows)
      if _row == row
        count += 1
        data[pini+count-1] = jrow
      end
    end
  end

  return Table(data,ptrs)
end

function get_iurowcol_to_irowcol(
  rows::AbstractVector,
  cols::AbstractVector,
  nrows::Int=(isempty(rows) ? 0 : maximum(rows))
  )

  @assert length(rows) == length(cols)
  isempty(rows) && return Table(Int32[],Int32[1])

  rowcols_to_count = zeros(Int32,maximum(rows)+nrows*(maximum(cols)-1))
  for (row,col) in zip(rows,cols)
    rowcols_to_count[row+nrows*(col-1)] += 1
  end

  ptrs = Vector{Int32}(undef,length(rows)+1)
  for (irowcol,rowcol) in enumerate(zip(rows,cols))
    row,col = rowcol
    ptrs[irowcol+1] = rowcols_to_count[row+nrows*(col-1)]
  end
  length_to_ptrs!(ptrs)

  data = Vector{Int32}(undef,ptrs[end]-1)
  for (irowcols,rowcols) in enumerate(zip(rows,cols))
    row,col = rowcols
    pini = ptrs[irowcols]
    count = 0
    for (jrowcols,_rowcols) in enumerate(zip(rows,cols))
      _row,_col = _rowcols
      if _row == row && _col == col
        count += 1
        data[pini+count-1] = jrowcols
      end
    end
  end

  return Table(data,ptrs)
end

function get_iudof_to_idof(rows::AbstractVector)
  get_iurow_to_irow(rows)
end

function get_iudof_to_idof(rowcols::Tuple{<:AbstractVector,<:AbstractVector})
  rows,cols = rowcols
  get_iurowcol_to_irowcol(rows,cols)
end

# multi field

function get_interpolation_style(a::BlockInterpolation)
  get_interpolation_style(first(a.interp))
end

function get_indices_time(a::BlockInterpolation{N}) where N
  map(get_indices_time,a.interp)
end

function get_itimes(a::BlockInterpolation{N},ids::Vector) where N
  map(itp -> get_itimes(itp,ids),a.interp)
end

function get_itimes(a::BlockInterpolation{N},ids::Range2D) where N
  map(itp -> get_itimes(itp,ids),a.interp)
end

function get_locations(a::BlockInterpolation{N},ids::Range2D) where N
  map(itp -> get_locations(itp,ids),a.interp)
end