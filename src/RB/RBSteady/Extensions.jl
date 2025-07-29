const DirectSumRBSpace = Union{RBSpace{<:DirectSumFESpace},RBSpace{<:AbstractTrialFESpace{<:DirectSumFESpace}}}

function FESpaces.zero_free_values(r::DirectSumRBSpace)
  zero_free_values(get_bg_space(r))
end

function Extensions.get_bg_cell_dof_ids(r::DirectSumRBSpace,args...)
  get_bg_cell_dof_ids(get_fe_space(r),args...)
end

function IntegrationDomain(
  trian::Triangulation,
  test::DirectSumRBSpace,
  rows::Vector{<:Number}
  )

  cell_row_ids = get_bg_cell_dof_ids(test,trian)
  cells = get_rows_to_cells(cell_row_ids,rows)
  irows = get_cells_to_irows(cell_row_ids,cells,rows)
  GenericDomain(cells,irows,rows)
end

function IntegrationDomain(
  trian::Triangulation,
  trial::DirectSumRBSpace,
  test::DirectSumRBSpace,
  rows::Vector{<:Number},
  cols::Vector{<:Number}
  )

  cell_row_ids = get_bg_cell_dof_ids(test,trian)
  cell_col_ids = get_bg_cell_dof_ids(trial,trian)
  cells = get_rowcols_to_cells(cell_row_ids,cell_col_ids,rows,cols)
  irowcols = get_cells_to_irowcols(cell_row_ids,cell_col_ids,cells,rows,cols)
  GenericDomain(cells,irowcols,(rows,cols))
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
