struct MissingDofsFESpace{V} <: SingleFieldFESpace
  vector_type::Type{V}
  nfree::Int
  ndirichlet::Int
  cell_dofs_ids::AbstractArray
  fe_basis::CellField
  fe_dof_basis::CellDof
  cell_is_dirichlet::AbstractArray{Bool}
  dirichlet_dof_tag::Vector{Int8}
  dirichlet_cells::Vector{Int32}
  ntags::Int
end

FESpaces.ConstraintStyle(::Type{<:MissingDofsFESpace}) = UnConstrained()
FESpaces.get_free_dof_ids(f::MissingDofsFESpace) = Base.OneTo(f.nfree)
FESpaces.get_fe_basis(f::MissingDofsFESpace) = f.fe_basis
FESpaces.get_fe_dof_basis(f::MissingDofsFESpace) = f.fe_dof_basis
FESpaces.get_cell_dof_ids(f::MissingDofsFESpace) = f.cell_dofs_ids
FESpaces.get_triangulation(f::MissingDofsFESpace) = get_triangulation(f.fe_basis)
FESpaces.get_dof_value_type(f::MissingDofsFESpace{V}) where V = eltype(V)
FESpaces.get_vector_type(f::MissingDofsFESpace{V}) where V = V
FESpaces.get_cell_is_dirichlet(f::MissingDofsFESpace) = f.cell_is_dirichlet

# SingleFieldFESpace interface

FESpaces.get_dirichlet_dof_ids(f::MissingDofsFESpace) = Base.OneTo(f.ndirichlet)
FESpaces.num_dirichlet_tags(f::MissingDofsFESpace) = f.ntags
FESpaces.get_dirichlet_dof_tag(f::MissingDofsFESpace) = f.dirichlet_dof_tag

function FESpaces.scatter_free_and_dirichlet_values(f::MissingDofsFESpace,free_values,dirichlet_values)
  @check eltype(free_values) == eltype(dirichlet_values) """\n
  The entries stored in free_values and dirichlet_values should be of the same type.

  This error shows up e.g. when trying to build a FEFunction from a vector of integers
  if the Dirichlet values of the underlying space are of type Float64, or when the
  given free values are Float64 and the Dirichlet values ComplexF64.
  """
  cell_dof_ids = get_cell_dof_ids(f)
  lazy_map(Broadcasting(PosZeroNegReindex(free_values,dirichlet_values)),cell_dof_ids)
end

function FESpaces.gather_free_and_dirichlet_values!(free_vals,dirichlet_vals,f::MissingDofsFESpace,cell_vals)
  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  cells = 1:length(cell_vals)

  _free_and_dirichlet_nonmissing_values_fill!(
    free_vals,
    dirichlet_vals,
    cache_vals,
    cache_dofs,
    cell_vals,
    cell_dofs,
    cells)

  (free_vals,dirichlet_vals)
end

function FESpaces.gather_dirichlet_values!(dirichlet_vals,f::MissingDofsFESpace,cell_vals)
  cell_dofs = get_cell_dof_ids(f)
  cache_vals = array_cache(cell_vals)
  cache_dofs = array_cache(cell_dofs)
  free_vals = zero_free_values(f)
  cells = f.dirichlet_cells

  _free_and_dirichlet_nonmissing_values_fill!(
    free_vals,
    dirichlet_vals,
    cache_vals,
    cache_dofs,
    cell_vals,
    cell_dofs,
    cells)

  dirichlet_vals
end

function _free_and_dirichlet_nonmissing_values_fill!(
  free_vals,
  dirichlet_vals,
  cache_vals,
  cache_dofs,
  cell_vals,
  cell_dofs,
  cells)

  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    for (i,dof) in enumerate(dofs)
      val = vals[i]
      if dof > 0
        free_vals[dof] = val
      elseif dof < 0
        dirichlet_vals[-dof] = val
      end
    end
  end
end

function _free_and_dirichlet_nonmissing_values_fill!(
  free_vals::AbstractParamVector,
  dirichlet_vals::AbstractParamVector,
  cache_vals,
  cache_dofs,
  cell_vals,
  cell_dofs,
  cells)

  @check param_length(free_vals) == param_length(dirichlet_vals)
  free_data = get_all_data(free_vals)
  diri_data = get_all_data(dirichlet_vals)
  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    for k in param_eachindex(free_vals)
      val = param_getindex(vals,k)
      for (i,dof) in enumerate(dofs)
        if dof > 0
          free_data[dof,k] = val[i]
        elseif dof < 0
          diri_data[-dof,k] = val[i]
        end
      end
    end
  end
end
