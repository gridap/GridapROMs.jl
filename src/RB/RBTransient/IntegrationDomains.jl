# iurow_to_irow[id of a unique row] = ids of the entries of that row
# e.g. get_iurow_to_irow([1,10,100,10]) = [[1],[2,4],[3],[2,4]]
function get_iurow_to_irow(rows::AbstractVector)
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
  nrows::Int=maximum(rows)
  )

  @assert length(rows) == length(cols)

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

function get_max_offset(ptrs::Vector{<:Integer})
  offsets = zeros(Int32,length(ptrs)-1)
  for i in eachindex(offsets)
    offsets[i] = ptrs[i+1]-ptrs[i]
  end
  return maximum(offsets)
end

function get_max_offset(a::Table)
  get_max_offset(a.ptrs)
end

struct CellsToSpacetimeIrowsMap{N,A,B,C,D} <: Map
  cell_row_ids::A
  cells::B
  rows::C
  iurow_to_irow::D

  function CellsToSpacetimeIrowsMap{N}(
    a::A,b::B,c::C,d::D
    ) where {N,A,B,C,D}

    new{N,A,B,C,D}(a,b,c,d)
  end
end

function Arrays.return_cache(k::CellsToSpacetimeIrowsMap{N},icell::Int) where N
  row_cache = array_cache(k.cell_row_ids)
  rows_s = getindex!(row_cache,k.cell_row_ids,icell)
  irow_cache = _st_cache(rows_s,Val{N}())
  unwrap_cache = return_cache(unwrap_and_setsize!,irow_cache,rows_s)
  ucache = array_cache(k.iurow_to_irow)
  return row_cache,irow_cache,unwrap_cache,ucache
end

function Arrays.evaluate!(cache,k::CellsToSpacetimeIrowsMap{N},icell::Int) where N
  row_cache,irow_cache,unwrap_cache,ucache = cache
  cell = k.cells[icell]
  cellrows = getindex!(row_cache,k.cell_row_ids,cell)
  a = evaluate!(unwrap_cache,unwrap_and_setsize!,irow_cache,cellrows)
  _st_evaluate!(a,cellrows,k.rows,k.iurow_to_irow,ucache)
end

function get_spacetime_irows(cell_row_ids,cells,rows)
  iurow_to_irow = get_iurow_to_irow(rows)
  N = get_max_offset(iurow_to_irow)
  k = CellsToSpacetimeIrowsMap{N}(cell_row_ids,cells,rows,iurow_to_irow)
  lazy_map(k,1:length(cells))
end

abstract type TransientIntegrationDomainStyle end
struct KroneckerDomain <: TransientIntegrationDomainStyle end
struct SequentialDomain <: TransientIntegrationDomainStyle end

"""
    struct TransientIntegrationDomain{A<:TransientIntegrationDomainStyle,Ti<:Integer} <: IntegrationDomain
      domain_style::A
      domain_space::IntegrationDomain
      indices_time::Vector{Ti}
    end

Integration domain for a projection operator in a transient problem
"""
struct TransientIntegrationDomain{A<:TransientIntegrationDomainStyle,Ti<:Integer} <: IntegrationDomain
  domain_style::A
  domain_space::IntegrationDomain
  indices_time::Vector{Ti}
end

get_domain_style(a::TransientIntegrationDomain) = a.domain_style

function RBSteady.IntegrationDomain(
  ::Type{<:KroneckerProjection},
  trian::Triangulation,
  test::FESpace,
  rows::AbstractVector,
  indices_time::AbstractVector
  )

  domain_space = IntegrationDomain(trian,test,rows)
  TransientIntegrationDomain(KroneckerDomain(),domain_space,indices_time)
end

function RBSteady.IntegrationDomain(
  ::Type{<:KroneckerProjection},
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::AbstractVector,
  cols::AbstractVector,
  indices_time::AbstractVector
  )

  domain_space = IntegrationDomain(trian,trial,test,rows,cols)
  TransientIntegrationDomain(KroneckerDomain(),domain_space,indices_time)
end

function RBSteady.IntegrationDomain(
  ::Type{<:SequentialProjection},
  trian::Triangulation,
  test::FESpace,
  rows::AbstractVector,
  indices_time::AbstractVector
  )

  cell_row_ids = get_cell_dof_ids(test,trian)
  cells = RBSteady.get_rows_to_cells(cell_row_ids,rows)
  irows = get_spacetime_irows(cell_row_ids,cells,rows)
  domain_space = VectorDomain(cells,irows,rows)
  TransientIntegrationDomain(SequentialDomain(),domain_space,indices_time)
end

function RBSteady.IntegrationDomain(
  ::Type{<:SequentialProjection},
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::AbstractVector,
  cols::AbstractVector,
  indices_time::AbstractVector
  )

  cell_row_ids = get_cell_dof_ids(test,trian)
  cell_col_ids = get_cell_dof_ids(trial,trian)
  cells = RBSteady.get_rowcols_to_cells(cell_row_ids,cell_col_ids,rows,cols)
  irows = get_spacetime_irows(cell_row_ids,cells,rows)
  icols = get_spacetime_irows(cell_col_ids,cells,cols)
  domain_space = MatrixDomain(cells,irows,icols,rows,cols)
  TransientIntegrationDomain(SequentialDomain(),domain_space,indices_time)
end

RBSteady.get_integration_cells(i::TransientIntegrationDomain) = get_integration_cells(i.domain_space)
RBSteady.get_cell_irows(i::TransientIntegrationDomain) = get_cell_irows(i.domain_space)
RBSteady.get_cell_icols(i::TransientIntegrationDomain) = get_cell_icols(i.domain_space)
get_integration_domain_space(i::TransientIntegrationDomain) = i.domain_space
get_indices_time(i::TransientIntegrationDomain) = i.indices_time

function get_itimes(i::TransientIntegrationDomain,ids::AbstractVector)::Vector{Int}
  idsi = get_indices_time(i)
  filter(!isnothing,indexin(idsi,ids))
end

# utils 

_st_eltype(::Val{1}) = Int32
_st_eltype(::Val{N}) where N = VectorValue{N,Int32}

_zero(::Type{Int32}) = zero(Int32)
_zero(::Type{<:VectorValue{D,T}}) where {D,T} = VectorValue(tfill(zero(T),Val{D}()))

_setindex!(a::AbstractArray{Int32},i,v,_) = (a[i] = Int32(v))
_setindex!(a,i,v,j) = (a[i] = VectorValue(Base.setindex(a[i].data,Int32(v),j)))

function _st_cache(ids::AbstractArray,::Val{N}) where N
  T = _st_eltype(Val(N))
  array = fill(zero(T),size(ids))
  CachedArray(array)
end

function _st_cache(ids::ArrayBlock,::Val{N}) where N
  ai = _st_cache(testitem(ids),Val{N}())
  array = Array{typeof(ai),ndims(ids)}(undef,size(ids))
  for i in eachindex(ids.array)
    if ids.touched[i]
      array[i] = _st_cache(ids.array[i],Val{N}())
    end
  end
  ArrayBlock(array,ids.touched)
end

function _st_evaluate!(a,cellrows,rows,iurow_to_irow,ucache)
  fill!(a,_zero(eltype(a)))
  for iurow in eachindex(iurow_to_irow)
    irows = getindex!(ucache,iurow_to_irow,iurow)
    for (iuirow,irow) in enumerate(irows)
      row = rows[irow]
      for (icellrow,cellrow) in enumerate(cellrows)
        if row == cellrow
          _setindex!(a,icellrow,irow,iuirow)
        end
      end
    end
  end
  a
end

function _st_evaluate!(a,cellrows::OIdsToIds,rows,iurow_to_irow,ucache)
  fill!(a,_zero(eltype(a)))
  for iurow in eachindex(iurow_to_irow)
    irows = getindex!(ucache,iurow_to_irow,iurow)
    for (iuirow,irow) in enumerate(irows)
      row = rows[irow]
      for (_icellrow,cellrow) in enumerate(cellrows)
        if row == cellrow
          icellrow = cellrows.terms[_icellrow]
          _setindex!(a,icellrow,irow,iuirow)
        end
      end
    end
  end
  a
end

function _st_evaluate!(a::VectorBlock,cellrows::VectorBlock,rows,iurow_to_irow,ucache)
  @check a.touched == cellrows.touched
  for i in eachindex(a)
    if a.touched[i]
      _st_evaluate!(a.array[i],cellrows.array[i],rows,iurow_to_irow,ucache)
    end
  end
  a
end

