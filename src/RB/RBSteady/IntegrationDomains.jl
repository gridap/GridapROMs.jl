function empirical_interpolation(A::AbstractMatrix)
  m,n = size(A)
  res = zeros(eltype(A),m)
  I = zeros(Int,n)
  @views I[1] = argmax(abs.(A[:,1]))
  if n > 1
    @inbounds for i = 2:n
      @views Ai = A[:,i]
      @views Bi = A[:,1:i-1]
      @views Ci = A[I[1:i-1],1:i-1]
      @views Di = A[I[1:i-1],i]
      @views res = Ai - Bi*(Ci \ Di)
      I[i] = argmax(map(abs,res))
    end
  end
  Ai = view(A,I,:)
  return I,Ai
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

function get_cells_to_idofs(
  cell_dof_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  dofs::AbstractVector)

  _correct_idof(is,li) = li
  _correct_idof(is::OIdsToIds,li) = is.terms[li]

  cache = array_cache(cell_dof_ids)

  ncells = length(cells)
  ptrs = Vector{Int32}(undef,ncells+1)
  @inbounds for (icell,cell) in enumerate(cells)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    ptrs[icell+1] = length(celldofs)
  end
  length_to_ptrs!(ptrs)

  data = fill(zero(Int32),ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    for (idof,dof) in enumerate(dofs)
      for (_icelldof,celldof) in enumerate(celldofs)
        if dof == celldof
          icelldof = _correct_idof(celldofs,_icelldof)
          data[ptrs[icell]-1+icelldof] = idof
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

function reduced_idofs(
  f::FESpace,
  trian::Triangulation,
  cells::AbstractVector,
  dofs::AbstractVector)

  cell_dof_ids = get_cell_dof_ids(f,trian)
  idofs = get_cells_to_idofs(cell_dof_ids,cells,dofs)
  return idofs
end

"""
    abstract type IntegrationDomain end

Type representing the set of interpolation rows of a `Projection` subjected
to a EIM approximation with `empirical_interpolation`.
Subtypes:
- [`VectorDomain`](@ref)
- [`MatrixDomain`](@ref)
- [`TransientIntegrationDomain`](@ref)
"""
abstract type IntegrationDomain end

get_integration_cells(i::IntegrationDomain) = @abstractmethod
get_cellids_rows(i::IntegrationDomain) = @abstractmethod
get_cellids_cols(i::IntegrationDomain) = @abstractmethod

function get_owned_icells(i::IntegrationDomain,cells::AbstractVector)::Vector{Int}
  cellsi = get_integration_cells(i)
  filter(!isnothing,indexin(cellsi,cells))
end

"""
    struct VectorDomain{T} <: IntegrationDomain
      cells::Vector{Int32}
      cell_irows::Table{T,Vector{T},Vector{Int32}}
    end

Integration domain for a projection vector operator in a steady problem
"""
struct VectorDomain{T} <: IntegrationDomain
  cells::Vector{Int32}
  cell_irows::Table{T,Vector{T},Vector{Int32}}
end

get_integration_cells(i::VectorDomain) = i.cells
get_cellids_rows(i::VectorDomain) = i.cell_irows

function vector_domain(args...)
  @abstractmethod
end

function vector_domain(
  trian::Triangulation,
  test::FESpace,
  rows::Vector{<:Number})

  cells = reduced_cells(test,trian,rows)
  irows = reduced_idofs(test,trian,cells,rows)
  VectorDomain(cells,irows)
end

"""
    struct MatrixDomain{T} <: IntegrationDomain{Int,1}
      cells::Vector{Int32}
      cell_irows::Table{T,Vector{T},Vector{Int32}}
      cell_icols::Table{T,Vector{T},Vector{Int32}}
    end

Integration domain for a projection vector operator in a steady problem
"""
struct MatrixDomain{T} <: IntegrationDomain
  cells::Vector{Int32}
  cell_irows::Table{T,Vector{T},Vector{Int32}}
  cell_icols::Table{T,Vector{T},Vector{Int32}}
end

function matrix_domain(args...)
  @abstractmethod
end

function matrix_domain(
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::Vector{<:Number},
  cols::Vector{<:Number})

  cells_trial = reduced_cells(trial,trian,cols)
  cells_test = reduced_cells(test,trian,rows)
  cells = union(cells_trial,cells_test)
  icols = reduced_idofs(trial,trian,cells,cols)
  irows = reduced_idofs(test,trian,cells,rows)
  MatrixDomain(cells,irows,icols)
end

get_integration_cells(i::MatrixDomain) = i.cells
get_cellids_rows(i::MatrixDomain) = i.cell_irows
get_cellids_cols(i::MatrixDomain) = i.cell_icols
