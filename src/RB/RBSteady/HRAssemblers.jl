function collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation)

  cell_idofs = get_cell_idofs(interp)
  icells = get_owned_icells(interp,strian)
  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_idofs,icells)
end

function collect_reduced_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation)

  cell_idofs = get_cell_idofs(interp)
  cells = get_integration_cells(interp)
  icells = get_owned_icells(interp,strian)
  scell_mat = lazy_map(Reindex(get_contribution(a,strian)),cells)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2 "$(ndims(eltype(cell_mat)))"
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_idofs,icells)
end

function collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation)

  cell_idofs = get_cell_idofs(interp)
  icells = get_owned_icells(interp,strian)
  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_idofs,icells)
end

function collect_reduced_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation)

  cell_idofs = get_cell_idofs(interp)
  cells = get_integration_cells(interp)
  icells = get_owned_icells(interp,strian)
  scell_vec = lazy_map(Reindex(get_contribution(a,strian)),cells)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_idofs,icells)
end

struct BlockReindex{A} <: Map
  values::A
  blockid::Int
end

function Arrays.return_cache(k::BlockReindex,i...)
  array_cache(k.values)
end

function Arrays.evaluate!(cache,k::BlockReindex,i...)
  a = getindex!(cache,k.values,i...)
  a.array[k.blockid]
end

function assemble_hr_array_add!(b::ArrayBlock,cellvec,cellidsrows::ArrayBlock,icells::ArrayBlock)
  @check cellidsrows.touched == icells.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellveci = lazy_map(BlockReindex(cellvec,i),icells.array[i])
      assemble_hr_array_add!(b.array[i],cellveci,cellidsrows.array[i],icells.array[i])
    end
  end
end

function assemble_hr_array_add!(b,cellvec,cellidsrows,icells)
  if length(cellvec) > 0
    rows_cache = array_cache(cellidsrows)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache,cellvec,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    add! = Arrays.AddEntriesMap(+)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add!,add_cache,vals_cache,rows_cache
    _numeric_loop_hr_array!(b,caches,cellvec,cellidsrows)
  end
  b
end

@noinline function _numeric_loop_hr_array!(vec,caches,cell_vals,cell_rows)
  add!,add_cache,vals_cache,rows_cache = caches
  @assert length(cell_vals) == length(cell_rows)
  for cell in 1:length(cell_rows)
    rows = getindex!(rows_cache,cell_rows,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,vec,vals,rows)
  end
end
