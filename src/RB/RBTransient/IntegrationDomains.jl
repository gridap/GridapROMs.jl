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

function get_spacetime_irows(
  cell_row_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  rows::AbstractVector
  )

  correct_irow = RBSteady.get_idof_correction(cell_row_ids)
  cache = array_cache(cell_row_ids)

  ncells = length(cells)
  ptrs = Vector{Int32}(undef,ncells+1)
  @inbounds for (icell,cell) in enumerate(cells)
    cellrows = getindex!(cache,cell_row_ids,cell)
    ptrs[icell+1] = length(cellrows)
  end
  length_to_ptrs!(ptrs)

  # count number of occurrences
  iurow_to_irow = get_iurow_to_irow(rows)
  ucache = array_cache(iurow_to_irow)
  N = get_max_offset(iurow_to_irow)

  z = zeros(Int32,N)
  data = map(_ -> copy(z),1:ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    cellrows = getindex!(cache,cell_row_ids,cell)
    for iurow in eachindex(iurow_to_irow)
      irows = getindex!(ucache,iurow_to_irow,iurow)
      for (iuirow,irow) in enumerate(irows)
        row = rows[irow]
        for (_icellrow,cellrow) in enumerate(cellrows)
          if row == cellrow
            icellrow = correct_irow(_icellrow,cellrows)
            data[ptrs[icell]-1+icellrow][iuirow] = irow
          end
        end
      end
    end
  end

  Table(map(VectorValue,data),ptrs)
end

function get_spacetime_irowcols(
  cell_row_ids::AbstractArray{<:AbstractArray},
  cell_col_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  rows::AbstractVector,
  cols::AbstractVector
  )

  correct_irow = RBSteady.get_idof_correction(cell_row_ids)
  correct_icol = RBSteady.get_idof_correction(cell_col_ids)
  rowcache = array_cache(cell_row_ids)
  colcache = array_cache(cell_col_ids)

  ncells = length(cells)
  ptrs = Vector{Int32}(undef,ncells+1)
  @inbounds for (icell,cell) in enumerate(cells)
    cellrows = getindex!(rowcache,cell_row_ids,cell)
    cellcols = getindex!(colcache,cell_col_ids,cell)
    ptrs[icell+1] = length(cellrows)*length(cellcols)
  end
  length_to_ptrs!(ptrs)

  # count number of occurrences
  nrows = length(rows)
  iurowcol_to_irowcol = get_iurowcol_to_irowcol(rows,cols)
  ucache = array_cache(iurowcol_to_irowcol)
  N = get_max_offset(iurowcol_to_irowcol)

  z = zeros(Int32,N)
  data = map(_ -> copy(z),1:ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    cellrows = getindex!(rowcache,cell_row_ids,cell)
    cellcols = getindex!(colcache,cell_col_ids,cell)
    ncellrows = length(cellrows)
    for iurowcol in eachindex(iurowcol_to_irowcol)
      irowcols = getindex!(ucache,iurowcol_to_irowcol,iurowcol)
      for (iuirowcol,irowcol) in enumerate(irowcols)
        row,col = rows[irowcol],cols[irowcol]
        for (_icellrow,cellrow) in enumerate(cellrows)
          for (_icellcol,cellcol) in enumerate(cellcols)
            if row == cellrow && col == cellcol
              icellrow = correct_irow(_icellrow,cellrows)
              icellcol = correct_icol(_icellcol,cellcols)
              icellrowcol = icellrow + (icellcol-1)*ncellrows
              data[ptrs[icell]-1+icellrowcol][iuirowcol] = irowcol
            end
          end
        end
      end
    end
  end

  Table(map(VectorValue,data),ptrs)
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
  indices_time::AbstractVector)

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
  indices_time::AbstractVector)

  domain_space = IntegrationDomain(trian,trial,test,rows,cols)
  TransientIntegrationDomain(KroneckerDomain(),domain_space,indices_time)
end

function RBSteady.IntegrationDomain(
  ::Type{<:SequentialProjection},
  trian::Triangulation,
  test::FESpace,
  rows::AbstractVector,
  indices_time::AbstractVector)

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
