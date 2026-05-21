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

abstract type TransientIntegrationDomainStyle end
struct KroneckerDomain <: TransientIntegrationDomainStyle end
struct SequentialDomain <: TransientIntegrationDomainStyle end

TransientIntegrationDomainStyle(args...) = @abstractmethod
TransientIntegrationDomainStyle(::Type{<:KroneckerProjection}) = KroneckerDomain()
TransientIntegrationDomainStyle(::Type{<:SequentialProjection}) = SequentialDomain()

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

const KroneckerIntegrationDomain{Ti<:Integer} = TransientIntegrationDomain{KroneckerDomain,Ti}
const SequentialIntegrationDomain{Ti<:Integer} = TransientIntegrationDomain{SequentialDomain,Ti}

get_domain_style(a::TransientIntegrationDomain) = a.domain_style

function RBSteady.IntegrationDomain(
  proj_style,
  trian::Triangulation,
  test::FESpace,
  rows::AbstractVector,
  indices_time::AbstractVector
  )

  domain_style = TransientIntegrationDomainStyle(proj_style)
  domain_space = IntegrationDomain(trian,test,rows)
  TransientIntegrationDomain(domain_style,domain_space,indices_time)
end

function RBSteady.IntegrationDomain(
  proj_style,
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::AbstractVector,
  cols::AbstractVector,
  indices_time::AbstractVector
  )

  domain_style = TransientIntegrationDomainStyle(proj_style)
  domain_space = IntegrationDomain(trian,trial,test,rows,cols)
  TransientIntegrationDomain(domain_style,domain_space,indices_time)
end

RBSteady.get_integration_cells(i::TransientIntegrationDomain) = get_integration_cells(i.domain_space)
RBSteady.get_cell_idofs(i::TransientIntegrationDomain) = get_cell_idofs(i.domain_space)
RBSteady.get_interpolation_dofs(a::TransientIntegrationDomain) = get_interpolation_dofs(a.domain_space)
get_integration_domain_space(i::TransientIntegrationDomain) = i.domain_space
get_indices_time(i::TransientIntegrationDomain) = i.indices_time

function get_itimes(i::TransientIntegrationDomain,ids::AbstractVector)::Vector{Int}
  idsi = get_indices_time(i)
  filter(!isnothing,indexin(idsi,ids))
end

function get_iudof_to_idof(rows::AbstractVector)
  get_iurow_to_irow(rows)
end

function get_iudof_to_idof(rowcols::Tuple{<:AbstractVector,<:AbstractVector})
  rows,cols = rowcols
  get_iurowcol_to_irowcol(rows,cols)
end