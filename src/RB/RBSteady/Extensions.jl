function reduced_cells(
  f::ExtensionFESpace,
  trian::Triangulation,
  dofs::AbstractVector
  )

  cell_dof_ids = get_bg_cell_dof_ids(f,trian)
  cells = get_dofs_to_cells(cell_dof_ids,dofs)
  return cells
end

function reduced_cells(
  r::RBSpace{<:ExtensionFESpace},
  trian::Triangulation,
  dofs::AbstractVector)

  reduced_cells(get_fe_space(r),trian,dofs)
end

function reduced_idofs(
  f::ExtensionFESpace,
  trian::Triangulation,
  cells::AbstractVector,
  dofs::AbstractVector)

  cell_dof_ids = get_bg_cell_dof_ids(f,trian)
  idofs = get_cells_to_idofs(cell_dof_ids,cells,dofs)
  return idofs
end

function reduced_idofs(
  r::RBSpace{<:ExtensionFESpace},
  trian::Triangulation,
  cells::AbstractVector,
  dofs::AbstractVector)

  reduced_idofs(get_fe_space(r),trian,cells,dofs)
end

for T in (:SingleFieldParamFESpace,:UnEvalTrialFESpace,:TransientTrialFESpace,:TrialFESpace)
  @eval begin
    function reduced_cells(
      f::$T{<:ExtensionFESpace},
      trian::Triangulation,
      dofs::AbstractVector)

      reduced_cells(Extensions.get_ext_space(f),trian,dofs)
    end

    function reduced_cells(
      r::RBSpace{<:$T{<:ExtensionFESpace}},
      trian::Triangulation,
      dofs::AbstractVector)

      reduced_cells(get_fe_space(r),trian,dofs)
    end

    function reduced_idofs(
      f::$T{<:ExtensionFESpace},
      trian::Triangulation,
      cells::AbstractVector,
      dofs::AbstractVector)

      reduced_idofs(Extensions.get_ext_space(f),trian,cells,dofs)
    end

    function reduced_idofs(
      r::RBSpace{<:$T{<:ExtensionFESpace}},
      trian::Triangulation,
      cells::AbstractVector,
      dofs::AbstractVector)

      reduced_idofs(get_fe_space(r),trian,cells,dofs)
    end
  end
end
