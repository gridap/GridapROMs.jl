

function FESpaces.zero_free_values(r::RBSpace{<:ExtensionFESpace})
  zero_free_values(Extensions.get_bg_space(r))
end

function FESpaces.zero_free_values(r::RBSpace{<:SingleFieldParamFESpace{<:ExtensionFESpace}})
  zero_free_values(_get_rb_bg_space(r))
end

function to_snapshots(
  r::RBSpace{<:UnEvalTrialFESpace{<:ExtensionFESpace}},
  x̂::AbstractParamVector,
  μ::AbstractRealization
  )

  rμ = r(μ)
  x = inv_project(Extensions.get_bg_space(rμ),x̂)
  fill_out_free_values!(x,get_fe_space(rμ))
  i = get_dof_map(rμ)
  Snapshots(x,i,μ)
end

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

# utils

function Extensions.get_ext_space(r::RBSpace)
  fext = Extensions.get_ext_space(get_fe_space(r))
  reduced_subspace(fext,get_reduced_subspace(r))
end

function Extensions.get_bg_space(r::RBSpace)
  fbg = Extensions.get_bg_space(get_fe_space(r))
  reduced_subspace(fbg,get_reduced_subspace(r))
end

function _get_rb_bg_space(r::RBSpace{<:SingleFieldParamFESpace{<:ExtensionFESpace}})
  fextμ = get_fe_space(r)
  fbgμ = Extensions.get_bg_space(fextμ)
  reduced_subspace(fbgμ,get_reduced_subspace(r))
end
