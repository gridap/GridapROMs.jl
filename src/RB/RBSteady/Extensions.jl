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

# post process

function Utils.compute_relative_error(
  norm_style::EuclideanNorm,feop::ExtensionParamOperator,extsol,extsol_approx
  )

  trial = get_trial(feop)
  sol,sol_approx = remove_extension(trial,extsol,extsol_approx)
  compute_relative_error(sol,sol_approx)
end

function Utils.compute_relative_error(
  norm_style::EnergyNorm,feop::ExtensionParamOperator,extsol,extsol_approx
  )

  trial = get_trial(feop)
  test = get_test(feop)
  sol,sol_approx = remove_extension(trial,extsol,extsol_approx)
  X = assemble_matrix(get_norm(norm_style),trial,test)
  compute_relative_error(sol,sol_approx,X)
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

function _remove_extension(s::Snapshots,ids::AbstractVector)
  data = reshape(get_all_data(s),:,num_params(s))
  fdata = view(data,ids,:)
  fdof_map = VectorDofMap(length(ids))
  r = get_realization(s)
  Snapshots(fdata,fdof_map,r)
end

function remove_extension(f::SingleFieldFESpace,exts::Snapshots,aexts::Snapshots)
  fdofs = get_fdof_to_bg_fdof(f)
  s = _remove_extension(exts,fdofs)
  as = _remove_extension(aexts,fdofs)
  return (s,as)
end

function remove_extension(f::MultiFieldFESpace,exts::BlockSnapshots,aexts::BlockSnapshots)
  cache = Vector{Snapshots}(undef,size(exts))
  acache = Vector{Snapshots}(undef,size(aexts))
  for i in eachindex(cache)
    if exts.touched[i]
      fdofs = get_fdof_to_bg_fdof(f[i])
      cache[i] = _remove_extension(exts[i],fdofs)
      acache[i] = _remove_extension(aexts[i],fdofs)
    end
  end
  BlockSnapshots(cache,exts.touched),BlockSnapshots(acache,aexts.touched)
end
