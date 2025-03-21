function get_dof_to_odof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  bg_dof_to_dof = zeros(Int,num_unconstrained_free_dofs(bg_f))
  bg_cell_ids = get_cell_odof_ids(bg_f)
  cell_ids = get_cell_odof_ids(f)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  cell_to_bg_cell = get_cell_to_bg_cell(f)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    for (bg_dof,dof) in zip(bg_dofs,dofs)
      if bg_dof > 0
        bg_dof_to_dof[bg_dof] = dof
      end
    end
  end
  return bg_dof_to_dof
end

function get_bg_odof_to_odof(bg_f::SingleFieldFESpace,agg_f::FESpaceWithLinearConstraints)
  act_dof_to_agg_dof = get_dof_to_mdof(agg_f)
  act_odof_to_agg_odof = get_dof_to_odof(bg_f,agg_f)
  bg_odof_to_act_odof = get_bg_dof_to_dof(bg_f,agg_f.space)
  bg_odof_to_agg_odof = compose_index(bg_dof_to_act_dof,act_dof_to_agg_dof)
  return bg_odof_to_agg_odof
end

for f in (:get_bg_odof_to_odof,:get_odof_to_bg_odof)
  @eval begin
    function $f(
      bg_f::SingleFieldFESpace,
      f::SingleFieldFESpace,
      i_to_bg_dof::AbstractVector
      )

      bg_dof_to_j = $f(bg_f,f)
      i_to_j = similar(i_to_bg_dof)
      for (i,bg_dof) in enumerate(i_to_bg_dof)
        i_to_j[i] = bg_dof_to_j[bg_dof]
      end
      return i_to_j
    end
  end
end

function get_odof_to_bg_odof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  dof_to_bg_dof = zeros(Int,num_free_dofs(f))
  bg_cell_ids = get_cell_odof_ids(bg_f)
  cell_ids = get_cell_odof_ids(f)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  cell_to_bg_cell = get_cell_to_bg_cell(f)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    for (bg_dof,dof) in zip(bg_dofs,dofs)
      if dof > 0
        dof_to_bg_dof[dof] = bg_dof
      end
    end
  end
  return dof_to_bg_dof
end

function get_odof_to_bg_odof(bg_f::SingleFieldFESpace,agg_f::FESpaceWithLinearConstraints)
  agg_dof_to_act_dof = get_mdof_to_dof(agg_f)
  agg_odof_to_act_odof = get_dof_to_odof(bg_f,agg_dof_to_act_dof)
  act_odof_to_bg_odof = get_odof_to_bg_odof(bg_f,agg_f.space)
  agg_odof_to_bg_odof = compose_index(agg_odof_to_act_odof,act_odof_to_bg_odof)
  return agg_odof_to_bg_odof
end

function get_ag_out_odof_to_bg_odof(
  f_bg::UnconstrainedFESpace,
  f_ag::FESpaceWithLinearConstraints,
  f_out::UnconstrainedFESpace)

  ag_dof_to_bg_dof = get_dof_to_bg_dof(f_bg,f_ag)
  act_out_dof_to_bg_dof = get_dof_to_bg_dof(f_bg,f_out)
  ag_out_dof_to_bg_dof = setdiff(act_out_dof_to_bg_dof,agg_dof_to_bg_dof)
  return ag_out_dof_to_bg_dof
end

function reorder_dofs(f::FESpace,cell_odofs_ids::AbstractVector)
  odof_to_dof = zeros(Int32,num_free_dofs(f))
  cell_dofs_ids = get_cell_dof_ids(f)
  cache = array_cache(cell_dofs_ids)
  ocache = array_cache(cell_odofs_ids)
  for cell in 1:length(cell_dofs_ids)
    dofs = getindex!(cache,cell_dofs_ids,cell)
    odofs = getindex!(ocache,cell_odofs_ids,cell)
    iodof_to_idof = odofs.terms
    for iodof in eachindex(odofs)
      idof = iodof_to_idof[iodof]
      dof = dofs[idof]
      odof = odofs[iodof]
      if odof > 0
        odof_to_dof[odof] = dof
      end
    end
  end
  return odof_to_dof
end

function reorder_dof_map(dof2_to_dof1::AbstractVector,dofs2::AbstractVector)
  dofs1 = similar(dofs2)
  for (i,dof2) in enumerate(dofs2)
    dofs1[i] = dof2_to_dof1[dof2]
  end
  return dofs1
end
