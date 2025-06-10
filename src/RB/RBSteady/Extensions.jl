const DirectSumRBSpace = Union{RBSpace{<:DirectSumFESpace},RBSpace{<:AbstractTrialFESpace{<:DirectSumFESpace}}}

function FESpaces.zero_free_values(r::DirectSumRBSpace)
  zero_free_values(get_bg_space(r))
end

function Extensions.get_bg_cell_dof_ids(r::DirectSumRBSpace,args...)
  get_bg_cell_dof_ids(get_fe_space(r),args...)
end

function reduced_cells(
  f::DirectSumRBSpace,
  trian::Triangulation,
  dofs::AbstractVector
  )

  cell_dof_ids = get_bg_cell_dof_ids(f,trian)
  cells = get_dofs_to_cells(cell_dof_ids,dofs)
  return cells
end

function reduced_idofs(
  f::DirectSumRBSpace,
  trian::Triangulation,
  cells::AbstractVector,
  dofs::AbstractVector)

  cell_dof_ids = get_bg_cell_dof_ids(f,trian)
  idofs = get_cells_to_idofs(cell_dof_ids,cells,dofs)
  return idofs
end

# utils

function Extensions.get_bg_space(r::RBSpace)
  fbg = get_bg_space(get_fe_space(r))
  reduced_subspace(fbg,get_reduced_subspace(r))
end

function Extensions.get_bg_space(r::RBSpace{<:SingleFieldParamFESpace{<:DirectSumFESpace}})
  fμ = get_fe_space(r)
  fbgμ = get_bg_space(fμ)
  reduced_subspace(fbgμ,get_reduced_subspace(r))
end

get_global_dof_map(r::RBSpace) = get_dof_map(get_fe_space(r))
get_global_dof_map(r::DirectSumRBSpace) = get_dof_map(get_bg_space(get_fe_space(r)))
