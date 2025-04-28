function FESpaces.zero_free_values(r::RBSpace{<:DirectSumFESpace})
  zero_free_values(Extensions.get_bg_space(r))
end

function FESpaces.zero_free_values(r::RBSpace{<:SingleFieldParamFESpace{<:DirectSumFESpace}})
  zero_free_values(_get_rb_bg_space(r))
end

function Extensions.get_bg_cell_dof_ids(r::RBSpace{<:DirectSumFESpace},args...)
  get_bg_cell_dof_ids(get_fe_space(r),args...)
end

function Extensions.get_bg_cell_dof_ids(r::RBSpace{<:AbstractTrialFESpace{<:DirectSumFESpace}},args...)
  get_bg_cell_dof_ids(_get_dsum_fe_space(r),args...)
end

function to_snapshots(
  r::RBSpace{<:UnEvalTrialFESpace{<:DirectSumFESpace}},
  x̂::AbstractParamVector,
  μ::AbstractRealization
  )

  rμ = r(μ)
  x = inv_project(Extensions.get_bg_space(rμ),x̂)
  i = get_dof_map(rμ)
  Snapshots(x,i,μ)
end

for T in (
  :(RBSpace{<:DirectSumFESpace}),
  :(RBSpace{<:SingleFieldParamFESpace{<:DirectSumFESpace}})
  )
  @eval begin
    function reduced_cells(
      f::$T,
      trian::Triangulation,
      dofs::AbstractVector
      )

      cell_dof_ids = get_bg_cell_dof_ids(f,trian)
      cells = get_dofs_to_cells(cell_dof_ids,dofs)
      return cells
    end

    function reduced_idofs(
      f::$T,
      trian::Triangulation,
      cells::AbstractVector,
      dofs::AbstractVector)

      cell_dof_ids = get_bg_cell_dof_ids(f,trian)
      idofs = get_cells_to_idofs(cell_dof_ids,cells,dofs)
      return idofs
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

_get_dsum_fe_space(r::RBSpace{<:DirectSumFESpace}) = get_fe_space(r)
_get_dsum_fe_space(r::RBSpace{<:AbstractTrialFESpace{<:DirectSumFESpace}}) = _get_dsum_fe_space(get_fe_space(r))
