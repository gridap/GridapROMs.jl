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

struct CellsToSpacetimeIrowsMap{N,A,B,C,D} <: CellsToIdsMap
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

struct CellsToSpacetimeIrowcolsMap{N,A,B,C,D,E,F} <: CellsToIdsMap
  cell_row_ids::A
  cell_col_ids::B
  cells::C
  rows::D
  cols::E
  iurowcol_to_irowcol::F

  function CellsToSpacetimeIrowcolsMap{N}(
    a::A,b::B,c::C,d::D,e::E,f::F
    ) where {N,A,B,C,D,E,F}

    new{N,A,B,C,D,E,F}(a,b,c,d,e,f)
  end
end

function Arrays.return_cache(k::CellsToSpacetimeIrowcolsMap{N},icell::Int) where N
  row_cache = array_cache(k.cell_row_ids)
  col_cache = array_cache(k.cell_col_ids)
  rowcols_s = getindex!(row_cache,k.cell_row_ids,icell)
  colcols_s = getindex!(col_cache,k.cell_col_ids,icell)
  irowcol_cache = _st_cache(rowcols_s,Val{N}())
  unwrap_cache = return_cache(unwrap_and_setsize!,irowcol_cache,rowcols_s,colcols_s)
  ucache = array_cache(k.iurowcol_to_irowcol)
  return row_cache,col_cache,irowcol_cache,unwrap_cache,ucache
end

function Arrays.evaluate!(cache,k::CellsToSpacetimeIrowcolsMap{N},icell::Int) where N
  row_cache,col_cache,irowcol_cache,unwrap_cache,ucache = cache
  cell = k.cells[icell]
  cellrows = getindex!(row_cache,k.cell_row_ids,cell)
  cellcols = getindex!(col_cache,k.cell_col_ids,cell)
  a = evaluate!(unwrap_cache,unwrap_and_setsize!,irowcol_cache,cellrows,cellcols)
  _st_evaluate!(a,cellrows,cellcols,k.rows,k.cols,k.iurowcol_to_irowcol,ucache)
end

function get_spacetime_irowcols(cell_row_ids,cell_col_ids,cells,rows,cols)
  iurowcol_to_irowcol = get_iurowcol_to_irowcol(rows,cols)
  N = get_max_offset(iurowcol_to_irowcol)
  k = CellsToSpacetimeIrowcolsMap{N}(cell_row_ids,cell_col_ids,cells,rows,cols,iurowcol_to_irowcol)
  lazy_map(k,1:length(cells))
end

# function get_spacetime_irows(
#   cell_row_ids::AbstractArray{<:AbstractArray},
#   cells::AbstractVector,
#   rows::AbstractVector
#   )

#   correct_irow = RBSteady.get_idof_correction(cell_row_ids)
#   cache = array_cache(cell_row_ids)

#   ncells = length(cells)
#   ptrs = Vector{Int32}(undef,ncells+1)
#   @inbounds for (icell,cell) in enumerate(cells)
#     cellrows = getindex!(cache,cell_row_ids,cell)
#     ptrs[icell+1] = length(cellrows)
#   end
#   length_to_ptrs!(ptrs)

#   # count number of occurrences
#   iurow_to_irow = get_iurow_to_irow(rows)
#   ucache = array_cache(iurow_to_irow)
#   N = get_max_offset(iurow_to_irow)

#   z = zeros(Int32,N)
#   data = map(_ -> copy(z),1:ptrs[end]-1)
#   for (icell,cell) in enumerate(cells)
#     cellrows = getindex!(cache,cell_row_ids,cell)
#     for iurow in eachindex(iurow_to_irow)
#       irows = getindex!(ucache,iurow_to_irow,iurow)
#       for (iuirow,irow) in enumerate(irows)
#         row = rows[irow]
#         for (_icellrow,cellrow) in enumerate(cellrows)
#           if row == cellrow
#             icellrow = correct_irow(_icellrow,cellrows)
#             data[ptrs[icell]-1+icellrow][iuirow] = irow
#           end
#         end
#       end
#     end
#   end

#   Table(map(VectorValue,data),ptrs)
# end

# function get_spacetime_irowcols(
#   cell_row_ids::AbstractArray{<:AbstractArray},
#   cell_col_ids::AbstractArray{<:AbstractArray},
#   cells::AbstractVector,
#   rows::AbstractVector,
#   cols::AbstractVector
#   )

#   correct_irow = RBSteady.get_idof_correction(cell_row_ids)
#   correct_icol = RBSteady.get_idof_correction(cell_col_ids)
#   rowcache = array_cache(cell_row_ids)
#   colcache = array_cache(cell_col_ids)

#   ncells = length(cells)
#   ptrs = Vector{Int32}(undef,ncells+1)
#   @inbounds for (icell,cell) in enumerate(cells)
#     cellrows = getindex!(rowcache,cell_row_ids,cell)
#     cellcols = getindex!(colcache,cell_col_ids,cell)
#     ptrs[icell+1] = length(cellrows)*length(cellcols)
#   end
#   length_to_ptrs!(ptrs)

#   # count number of occurrences
#   nrows = length(rows)
#   iurowcol_to_irowcol = get_iurowcol_to_irowcol(rows,cols)
#   ucache = array_cache(iurowcol_to_irowcol)
#   N = get_max_offset(iurowcol_to_irowcol)

#   z = zeros(Int32,N)
#   data = map(_ -> copy(z),1:ptrs[end]-1)
#   for (icell,cell) in enumerate(cells)
#     cellrows = getindex!(rowcache,cell_row_ids,cell)
#     cellcols = getindex!(colcache,cell_col_ids,cell)
#     ncellrows = length(cellrows)
#     for iurowcol in eachindex(iurowcol_to_irowcol)
#       irowcols = getindex!(ucache,iurowcol_to_irowcol,iurowcol)
#       for (iuirowcol,irowcol) in enumerate(irowcols)
#         row,col = rows[irowcol],cols[irowcol]
#         for (_icellrow,cellrow) in enumerate(cellrows)
#           for (_icellcol,cellcol) in enumerate(cellcols)
#             if row == cellrow && col == cellcol
#               icellrow = correct_irow(_icellrow,cellrows)
#               icellcol = correct_icol(_icellcol,cellcols)
#               icellrowcol = icellrow + (icellcol-1)*ncellrows
#               data[ptrs[icell]-1+icellrowcol][iuirowcol] = irowcol
#             end
#           end
#         end
#       end
#     end
#   end

#   Table(map(VectorValue,data),ptrs)
# end

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
  domain_space = GenericDomain(cells,irows,rows)
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
  irowcols = get_spacetime_irowcols(cell_row_ids,cell_col_ids,cells,rows,cols)
  domain_space = GenericDomain(cells,irowcols,(rows,cols))
  TransientIntegrationDomain(SequentialDomain(),domain_space,indices_time)
end

RBSteady.get_integration_cells(i::TransientIntegrationDomain) = get_integration_cells(i.domain_space)
RBSteady.get_cell_idofs(i::TransientIntegrationDomain) = get_cell_idofs(i.domain_space)
get_integration_domain_space(i::TransientIntegrationDomain) = i.domain_space
get_indices_time(i::TransientIntegrationDomain) = i.indices_time

function get_itimes(i::TransientIntegrationDomain,ids::AbstractVector)::Vector{Int}
  idsi = get_indices_time(i)
  filter(!isnothing,indexin(idsi,ids))
end

# utils 

function _st_cache(ids::AbstractArray,::Val{N}) where N
  _getids(a) = a 
  _getids(a::OIdsToIds) = a.indices
  CachedArray(similar(_getids(ids),VectorValue{N,Int32}))
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
  fill!(a,zero(eltype(a)))
  for iurow in eachindex(iurow_to_irow)
    irows = getindex!(ucache,iurow_to_irow,iurow)
    for (iuirow,irow) in enumerate(irows)
      row = rows[irow]
      for (icellrow,cellrow) in enumerate(cellrows)
        if row == cellrow
          a[icellrow][iuirow] = irow
        end
      end
    end
  end
  a
end

function _st_evaluate!(a,cellrows::OIdsToIds,rows,iurow_to_irow,ucache)
  fill!(a,zero(eltype(a)))
  for iurow in eachindex(iurow_to_irow)
    irows = getindex!(ucache,iurow_to_irow,iurow)
    for (iuirow,irow) in enumerate(irows)
      row = rows[irow]
      for (_icellrow,cellrow) in enumerate(cellrows)
        if row == cellrow
          icellrow = cellrows.terms[_icellrow]
          a[icellrow][iuirow] = irow
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

function _st_evaluate!(a,cellrows,cellcols,rows,cols,iurowcol_to_irowcol,ucache)
  fill!(a,zero(eltype(a)))
  ncellrows = length(cellrows)
  for iurowcol in eachindex(iurowcol_to_irowcol)
    irowcols = getindex!(ucache,iurowcol_to_irowcol,iurowcol)
    for (iuirowcol,irowcol) in enumerate(irowcols)
      row,col = rows[irowcol],cols[irowcol]
      for (icellrow,cellrow) in enumerate(cellrows)
        for (icellcol,cellcol) in enumerate(cellcols)
          if row == cellrow && col == cellcol
            icellrowcol = icellrow + (icellcol-1)*ncellrows
            a[icellrowcol][iuirowcol] = irowcol
          end
        end
      end
    end
  end
  a
end

function _st_evaluate!(a,cellrows::OIdsToIds,cellcols::OIdsToIds,rows,cols,iurowcol_to_irowcol,ucache)
  fill!(a,zero(eltype(a)))
  ncellrows = length(cellrows)
  for iurowcol in eachindex(iurowcol_to_irowcol)
    irowcols = getindex!(ucache,iurowcol_to_irowcol,iurowcol)
    for (iuirowcol,irowcol) in enumerate(irowcols)
      row,col = rows[irowcol],cols[irowcol]
      for (_icellrow,cellrow) in enumerate(cellrows)
        for (_icellcol,cellcol) in enumerate(cellcols)
          if row == cellrow && col == cellcol
            icellrow = cellrows.terms[_icellrow] 
            icellcol = cellcols.terms[_icellcol]
            icellrowcol = icellrow + (icellcol-1)*ncellrows
            a[icellrowcol][iuirowcol] = irowcol
          end
        end
      end
    end
  end
  a
end

function _st_evaluate!(a::VectorBlock,cellrows::VectorBlock,cellcols::VectorBlock,rows,cols,iurowcol_to_irowcol,ucache)
  @check a.touched == cellrows.touched == cellcols.touched
  for i in eachindex(a)
    if a.touched[i]
      _st_evaluate!(a.array[i],cellrows.array[i],cellcols.array[i],rows,cols,iurowcol_to_irowcol,ucache)
    end
  end
  a
end