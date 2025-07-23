function empirical_interpolation(basis::AbstractMatrix)
  n = size(basis,2)
  I = zeros(Int,n)
  basisI = zeros(eltype(basis),n,n)
  @inbounds @views begin
    res = abs.(basis[:,1])
    I[1] = argmax(res)
    basisI[1,:] = basis[I[1],:]
    for l = 2:n
      U = basis[:,1:l-1]
      P = I[1:l-1]
      PᵀU = U[P,:]
      uₗ = basis[:,l]
      Pᵀuₗ = uₗ[P,:]
      c = vec(PᵀU \ Pᵀuₗ)
      mul!(res,U,c)
      @. res = abs(uₗ - res)
      I[l] = argmax(res)
      basisI[l,:] = basis[I[l],:]
    end
  end
  return I,basisI
end

function empirical_interpolation(A::ParamSparseMatrix)
  I,AI = empirical_interpolation(A.data)
  R′,C′ = recast_split_indices(I,param_getindex(A,1))
  return (R′,C′),AI
end

"""
    get_dofs_to_cells(
      cell_dof_ids::AbstractArray{<:AbstractArray},
      dofs::AbstractVector
      ) -> AbstractVector

Returns the list of cells containing the dof ids `dofs`
"""
function get_dofs_to_cells(
  cell_dof_ids::AbstractArray{<:AbstractArray},
  dofs::AbstractVector)

  ncells = length(cell_dof_ids)
  cells = fill(false,ncells)
  cache = array_cache(cell_dof_ids)
  for cell in 1:ncells
    celldofs = getindex!(cache,cell_dof_ids,cell)
    stop = false
    if !stop
      for dof in celldofs
        for _dof in dofs
          if dof == _dof
            cells[cell] = true
            stop = true
            break
          end
        end
      end
    end
  end
  Int32.(findall(cells))
end

get_idof_correction(a) = (idof,celldofs) -> idof
get_idof_correction(a::OTable) = (idof,celldofs) -> celldofs.terms[idof]
get_idof_correction(a::LazyArray{<:Fill{<:Reindex}}) = get_idof_correction(a.maps[1].values)
get_idof_correction(a::AppendedArray) = get_idof_correction(a.a)

function get_cells_to_irows(
  cell_row_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  rows::AbstractVector)

  correct_irow = get_idof_correction(cell_row_ids)
  cache = array_cache(cell_row_ids)

  ncells = length(cells)
  ptrs = Vector{Int32}(undef,ncells+1)
  @inbounds for (icell,cell) in enumerate(cells)
    cellrows = getindex!(cache,cell_row_ids,cell)
    ptrs[icell+1] = length(cellrows)
  end
  length_to_ptrs!(ptrs)

  data = fill(zero(Int32),ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    cellrows = getindex!(cache,cell_row_ids,cell)
    for (irow,row) in enumerate(rows)
      for (_icellrow,cellrow) in enumerate(cellrows)
        if row == cellrow
          icellrow = correct_irow(_icellrow,cellrows)
          data[ptrs[icell]-1+icellrow] = irow
        end
      end
    end
  end

  Table(data,ptrs)
end

function get_cells_to_irowcols(
  cell_row_ids::AbstractArray{<:AbstractArray},
  cell_col_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  rows::AbstractVector,
  cols::AbstractVector)

  correct_irow = get_idof_correction(cell_row_ids)
  correct_icol = get_idof_correction(cell_col_ids)
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

  data = fill(zero(Int32),ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    cellrows = getindex!(rowcache,cell_row_ids,cell)
    cellcols = getindex!(colcache,cell_col_ids,cell)
    ncellrows = length(cellrows)
    for (irowcol,rowcol) in enumerate(zip(rows,cols))
      row,col = rowcol
      for (_icellrow,cellrow) in enumerate(cellrows)
        for (_icellcol,cellcol) in enumerate(cellcols)
          if row == cellrow && col == cellcol
            icellrow = correct_irow(_icellrow,cellrows)
            icellcol = correct_icol(_icellcol,cellcols)
            icellrowcol = icellrow + (icellcol-1)*ncellrows
            data[ptrs[icell]-1+icellrowcol] = irowcol
          end
        end
      end
    end
  end

  Table(data,ptrs)
end

function reduced_cells(
  f::FESpace,
  trian::Triangulation,
  dofs::AbstractVector)

  cell_dof_ids = get_cell_dof_ids(f,trian)
  cells = get_dofs_to_cells(cell_dof_ids,dofs)
  return cells
end

function reduced_irows(
  test::FESpace,
  trian::Triangulation,
  cells::AbstractVector,
  rows::AbstractVector)

  cell_row_ids = get_cell_dof_ids(test,trian)
  irows = get_cells_to_irows(cell_row_ids,cells,rows)
  return irows
end

function reduced_irowcols(
  trial::FESpace,
  test::FESpace,
  trian::Triangulation,
  cells::AbstractVector,
  rows::AbstractVector,
  cols::AbstractVector)

  cell_col_ids = get_cell_dof_ids(trial,trian)
  cell_row_ids = get_cell_dof_ids(test,trian)
  irowcols = get_cells_to_irowcols(cell_row_ids,cell_col_ids,cells,rows,cols)
  return irowcols
end

"""
    abstract type IntegrationDomain end

Type representing the set of interpolation rows of a `Projection` subjected
to a EIM approximation with `empirical_interpolation`.
Subtypes:
- [`GenericDomain`](@ref)
- [`TransientIntegrationDomain`](@ref)
"""
abstract type IntegrationDomain end

get_integration_cells(i::IntegrationDomain,args...) = @abstractmethod
get_cell_idofs(i::IntegrationDomain) = @abstractmethod

function get_owned_icells(i::IntegrationDomain,cells::AbstractVector)::Vector{Int}
  cellsi = get_integration_cells(i)
  filter(!isnothing,indexin(cellsi,cells))
end

function get_integration_cells(i::IntegrationDomain,trian::Triangulation)
  get_integration_cells(i)
end

function get_integration_cells(i::IntegrationDomain,trian::AppendedTriangulation)
  cells = get_integration_cells(i)
  parent = get_parent(trian)
  parent_ncellsa = num_cells(parent.a)
  cell_to_istriana = zeros(Bool,num_cells(trian))
  for (icell,cell) in enumerate(cells)
    cell_to_istriana[icell] = cell <= parent_ncellsa
  end
  ncellsa = length(findall(cell_to_istriana))
  a = Vector{eltype(cells)}(undef,ncellsa)
  b = Vector{eltype(cells)}(undef,length(cells)-ncellsa)
  na = 0
  nb = 0
  for (icell,cell) in enumerate(cells)
    if cell_to_istriana[icell]
      na += 1
      a[na] = cell
    else
      nb += 1
      b[nb] = cell
    end
  end
  lazy_append(a,b)
end

"""
    struct GenericDomain{T,A} <: IntegrationDomain
      cells::Vector{Int32}
      cell_idofs::Table{T,Vector{T},Vector{Int32}}
      metadata::A
    end

Integration domain for a projection operator in a steady problem
"""
struct GenericDomain{T,A} <: IntegrationDomain
  cells::Vector{Int32}
  cell_idofs::Table{T,Vector{T},Vector{Int32}}
  metadata::A
end

get_integration_cells(i::GenericDomain) = i.cells
get_cell_idofs(i::GenericDomain) = i.cell_idofs

function IntegrationDomain(args...)
  @abstractmethod
end

function IntegrationDomain(
  trian::Triangulation,
  test::FESpace,
  rows::Vector{<:Number}
  )

  cells = reduced_cells(test,trian,rows)
  irows = reduced_irows(test,trian,cells,rows)
  GenericDomain(cells,irows,rows)
end

function IntegrationDomain(
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::Vector{<:Number},
  cols::Vector{<:Number}
  )

  cells_trial = reduced_cells(trial,trian,cols)
  cells_test = reduced_cells(test,trian,rows)
  cells = union(cells_trial,cells_test)
  irowcols = reduced_irowcols(trial,test,trian,cells,rows,cols)
  GenericDomain(cells,irowcols,(rows,cols))
end

function move_integration_domain(
  i::GenericDomain,
  test::FESpace,
  ttrian::Triangulation,
  )

  rows = i.metadata
  IntegrationDomain(ttrian,test,rows)
end

function move_integration_domain(
  i::GenericDomain,
  trial::FESpace,
  test::FESpace,
  ttrian::Triangulation,
  )

  rows,cols = i.metadata
  IntegrationDomain(ttrian,trial,test,rows,cols)
end
