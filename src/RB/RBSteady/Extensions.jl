function FESpaces.zero_free_values(r::RBSpace{<:DirectSumFESpace})
  zero_free_values(Extensions.get_bg_space(r))
end

function FESpaces.zero_free_values(r::RBSpace{<:SingleFieldParamFESpace{<:DirectSumFESpace}})
  zero_free_values(_get_rb_bg_space(r))
end

function to_snapshots(
  r::RBSpace{<:UnEvalTrialFESpace{<:DirectSumFESpace}},
  x̂::AbstractParamVector,
  μ::AbstractRealization
  )

  rμ = r(μ)
  x = inv_project(Extensions.get_bg_space(rμ),x̂)
  # pad_solution!(x,get_fe_space(rμ))
  i = get_dof_map(rμ)
  Snapshots(x,i,μ)
end

function reduced_cells(
  f::DirectSumFESpace,
  trian::Triangulation,
  dofs::AbstractVector
  )

  cell_dof_ids = get_bg_cell_dof_ids(f,trian)
  cells = get_dofs_to_cells(cell_dof_ids,dofs)
  return cells
end

function reduced_cells(
  r::RBSpace{<:DirectSumFESpace},
  trian::Triangulation,
  dofs::AbstractVector)

  reduced_cells(get_fe_space(r),trian,dofs)
end

function reduced_idofs(
  f::DirectSumFESpace,
  trian::Triangulation,
  cells::AbstractVector,
  dofs::AbstractVector)

  cell_dof_ids = get_bg_cell_dof_ids(f,trian)
  idofs = get_cells_to_idofs(cell_dof_ids,cells,dofs)
  return idofs
end

function reduced_idofs(
  r::RBSpace{<:DirectSumFESpace},
  trian::Triangulation,
  cells::AbstractVector,
  dofs::AbstractVector)

  reduced_idofs(get_fe_space(r),trian,cells,dofs)
end

for T in (:SingleFieldParamFESpace,:UnEvalTrialFESpace,:TransientTrialFESpace,:TrialFESpace)
  @eval begin
    function reduced_cells(
      f::$T{<:DirectSumFESpace},
      trian::Triangulation,
      dofs::AbstractVector)

      reduced_cells(f.space,trian,dofs)
    end

    function reduced_cells(
      r::RBSpace{<:$T{<:DirectSumFESpace}},
      trian::Triangulation,
      dofs::AbstractVector)

      reduced_cells(get_fe_space(r),trian,dofs)
    end

    function reduced_idofs(
      f::$T{<:DirectSumFESpace},
      trian::Triangulation,
      cells::AbstractVector,
      dofs::AbstractVector)

      reduced_idofs(f.space,trian,cells,dofs)
    end

    function reduced_idofs(
      r::RBSpace{<:$T{<:DirectSumFESpace}},
      trian::Triangulation,
      cells::AbstractVector,
      dofs::AbstractVector)

      reduced_idofs(get_fe_space(r),trian,cells,dofs)
    end
  end
end

# utils

function Extensions.get_bg_space(r::RBSpace)
  fbg = Extensions.get_bg_space(get_fe_space(r))
  reduced_subspace(fbg,get_reduced_subspace(r))
end

function _get_rb_bg_space(r::RBSpace{<:SingleFieldParamFESpace{<:DirectSumFESpace}})
  fμ = get_fe_space(r)
  fbgμ = Extensions.get_bg_space(fμ)
  reduced_subspace(fbgμ,get_reduced_subspace(r))
end
