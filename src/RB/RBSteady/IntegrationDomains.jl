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

get_idof_correction(a) = (idof,celldofs) -> idof
get_idof_correction(a::OTable) = (idof,celldofs) -> celldofs.terms[idof]
get_idof_correction(a::LazyArray{<:Fill{<:Reindex}}) = get_idof_correction(a.maps[1].values)
get_idof_correction(a::AppendedArray) = get_idof_correction(a.a)

function get_cells_to_idofs(
  cell_dof_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  dofs::AbstractVector)

  correct_idof = get_idof_correction(cell_dof_ids)
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
          icelldof = correct_idof(_icelldof,celldofs)
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

get_integration_cells(i::IntegrationDomain,args...) = @abstractmethod
get_cellids_rows(i::IntegrationDomain) = @abstractmethod
get_cellids_cols(i::IntegrationDomain) = @abstractmethod

function get_owned_icells(i::IntegrationDomain,cells::AbstractVector)::Vector{Int}
  cellsi = get_integration_cells(i)
  filter(!isnothing,indexin(cellsi,cells))
end

"""
    struct VectorDomain{T,A} <: IntegrationDomain
      cells::Vector{Int32}
      cell_irows::Table{T,Vector{T},Vector{Int32}}
      metadata::A
    end

Integration domain for a projection vector operator in a steady problem
"""
struct VectorDomain{T,A} <: IntegrationDomain
  cells::Vector{Int32}
  cell_irows::Table{T,Vector{T},Vector{Int32}}
  metadata::A
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
  VectorDomain(cells,irows,rows)
end

"""
    struct MatrixDomain{T,S,A} <: IntegrationDomain
      cells::Vector{Int32}
      cell_irows::Table{T,Vector{T},Vector{Int32}}
      cell_icols::Table{S,Vector{S},Vector{Int32}}
      metadata::A
    end

Integration domain for a projection vector operator in a steady problem
"""
struct MatrixDomain{T,S,A} <: IntegrationDomain
  cells::Vector{Int32}
  cell_irows::Table{T,Vector{T},Vector{Int32}}
  cell_icols::Table{S,Vector{S},Vector{Int32}}
  metadata::A
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
  MatrixDomain(cells,irows,icols,(rows,cols))
end

get_integration_cells(i::MatrixDomain) = i.cells
get_cellids_rows(i::MatrixDomain) = i.cell_irows
get_cellids_cols(i::MatrixDomain) = i.cell_icols

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

function move_integration_domain(
  i::VectorDomain,
  test::FESpace,
  strian::Triangulation,
  ttrian::Triangulation,
  )

  rows = i.metadata
  scells = get_integration_cells(i)
  scell_to_tcell = strian_to_ttrian_cells(strian,ttrian)
  tcells = lazy_map(Reindex(scell_to_tcell),scells)
  tirows = reduced_idofs(test,ttrian,tcells,rows)
  VectorDomain(tcells,tirows,rows)
end

function move_integration_domain(
  i::MatrixDomain,
  trial::FESpace,
  test::FESpace,
  strian::Triangulation,
  ttrian::Triangulation,
  )

  rows,cols = i.metadata
  scells = get_integration_cells(i)
  scell_to_tcell = strian_to_ttrian_cells(strian,ttrian)
  tcells = lazy_map(Reindex(scell_to_tcell),scells)
  ticols = reduced_idofs(trial,ttrian,tcells,cols)
  tirows = reduced_idofs(test,ttrian,tcells,rows)
  MatrixDomain(tcells,tirows,ticols,i.metadata)
end
