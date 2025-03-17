function get_bg_dof_to_dof(
  bg_f::SingleFieldFESpace,
  f::SingleFieldFESpace,
  dof_to_bg_dof::AbstractVector
  )

  bg_dof_to_all_dof = get_bg_dof_to_dof(bg_f,f)
  bg_dof_to_dof = similar(dof_to_bg_dof)
  for (i,bg_dof) in enumerate(dof_to_bg_dof)
    bg_dof_to_dof[i] = bg_dof_to_all_dof[bg_dof]
  end
  return bg_dof_to_dof
end

function get_bg_dof_to_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  bg_dof_to_dof = zeros(Int,num_unconstrained_free_dofs(bg_f))
  bg_cell_ids = get_cell_dof_ids(bg_f)
  cell_ids = get_cell_dof_ids(f)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  cell_to_bg_cell = get_bg_cell_to_act_cell(f)
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

function get_bg_dof_to_dof(bg_f::SingleFieldFESpace,agg_f::FESpaceWithLinearConstraints)
  act_dof_to_agg_dof = get_dof_to_mdof(agg_f)
  bg_dof_to_act_dof = get_bg_dof_to_dof(bg_f,agg_f.space)
  bg_dof_to_agg_dof = compose_index(bg_dof_to_act_dof,act_dof_to_agg_dof)
  return bg_dof_to_agg_dof
end

function get_dof_to_bg_dof(
  bg_f::SingleFieldFESpace,
  f::SingleFieldFESpace,
  bg_dof_to_dof::AbstractVector
  )

  dof_to_all_bg_dof = get_dof_to_bg_dof(bg_f,f)
  dof_to_bg_dof = similar(bg_dof_to_dof)
  for (i,dof) in enumerate(bg_dof_to_dof)
    dof_to_bg_dof[i] = dof_to_all_bg_dof[dof]
  end
  return dof_to_bg_dof
end

function get_dof_to_bg_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  dof_to_bg_dof = zeros(Int,num_free_dofs(f))
  bg_cell_ids = get_cell_dof_ids(bg_f)
  cell_ids = get_cell_dof_ids(f)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  cell_to_bg_cell = get_bg_cell_to_act_cell(f)
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

function get_dof_to_bg_dof(bg_f::SingleFieldFESpace,agg_f::FESpaceWithLinearConstraints)
  agg_dof_to_act_dof = get_mdof_to_dof(agg_f)
  act_dof_to_bg_dof = get_dof_to_bg_dof(bg_f,agg_f.space)
  agg_dof_to_bg_dof = compose_index(agg_dof_to_act_dof,act_dof_to_bg_dof)
  return agg_dof_to_bg_dof
end

function get_ag_out_dof_to_bg_dof(
  f_bg::UnconstrainedFESpace,
  f_ag::FESpaceWithLinearConstraints,
  f_out::UnconstrainedFESpace)

  agg_dof_to_bg_dof = get_dof_to_bg_dof(f_bg,f_ag)
  act_out_dof_to_bg_dof = get_dof_to_bg_dof(f_bg,f_out)
  agg_out_dof_to_bg_dof = setdiff(act_out_dof_to_bg_dof,agg_dof_to_bg_dof)
  return agg_out_dof_to_bg_dof
end

function get_dof_to_mdof(f::FESpaceWithLinearConstraints)
  T = eltype(f.mDOF_to_DOF)
  dof_to_mdof = zeros(T,f.n_fdofs)
  cache = array_cache(f.DOF_to_mDOFs)
  for DOF in eachindex(dof_to_mdof)
    mDOFs = getindex!(cache,f.DOF_to_mDOFs,DOF)
    dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
    for mDOF in mDOFs
      mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
      if dof > 0
        dof_to_mdof[dof] = mdof
      end
    end
  end
  return dof_to_mdof
end

function get_mdof_to_dof(f::FESpaceWithLinearConstraints)
  T = eltype(f.mDOF_to_DOF)
  mdof_to_dof = zeros(T,f.n_fmdofs)
  for mDOF in eachindex(mdof_to_dof)
    DOF = f.mDOF_to_DOF[mDOF]
    mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
    dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
    if mdof > 0
      mdof_to_dof[mdof] = dof
    end
  end
  return mdof_to_dof
end

function compose_index(i1_to_i2,i2_to_i3)
  T_i3 = eltype(i2_to_i3)
  n_i1 = length(i1_to_i2)
  i1_to_i3 = zeros(T_i3,n_i1)
  for (i1,i2) in enumerate(i1_to_i2)
    if i2 > 0
      i1_to_i3[i1] = i2_to_i3[i2]
    end
  end
  return i1_to_i3
end

function get_dof_to_cells(cell_dofs::AbstractVector)
  inverse_table(Table(cell_dofs))
end

function inverse_table(cell_dofs::Table)
  ndofs = maximum(cell_dofs.data)
  ptrs = zeros(Int32,ndofs+1)
  for dof in cell_dofs.data
    ptrs[dof+1] += 1
  end
  length_to_ptrs!(ptrs)

  data = Vector{Int32}(undef,ptrs[end]-1)
  for cell in 1:length(cell_dofs)
    pini = cell_dofs.ptrs[cell]
    pend = cell_dofs.ptrs[cell+1]-1
    for p in pini:pend
      dof = cell_dofs.data[p]
      data[ptrs[dof]] = cell
      ptrs[dof] += 1
    end
  end
  rewind_ptrs!(ptrs)

  Table(data,ptrs)
end

num_unconstrained_free_dofs(f::SingleFieldFESpace) = num_free_dofs(f)
num_unconstrained_free_dofs(f::ZeroMeanFESpace) = num_unconstrained_free_dofs(f.space)
num_unconstrained_free_dofs(f::FESpaceWithConstantFixed) = num_unconstrained_free_dofs(f.space)
