function get_bg_fdof_to_fdof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  bg_fdof_to_fdof,_ = get_bg_dof_to_dof(bg_f,f)
  bg_fdof_to_fdof
end

function get_bg_ddof_to_ddof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  _,bg_ddof_to_ddof = get_bg_dof_to_dof(bg_f,f)
  bg_ddof_to_ddof
end

function get_fdof_to_bg_fdof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  fdof_to_bg_fdof,_ = get_dof_to_bg_dof(bg_f,f)
  fdof_to_bg_fdof
end

function get_ddof_to_bg_ddof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  _,ddof_to_bg_ddof = get_dof_to_bg_dof(bg_f,f)
  ddof_to_bg_ddof
end

function get_bg_dof_to_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  bg_fdof_to_fdof = zeros(Int,num_unconstrained_free_dofs(bg_f))
  bg_ddof_to_ddof = zeros(Int,num_dirichlet_dofs(bg_f))
  bg_cell_ids = get_cell_dof_ids(bg_f)
  cell_ids = get_cell_dof_ids(f)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  cell_to_bg_cell = get_cell_to_bg_cell(f)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    for (bg_dof,dof) in zip(bg_dofs,dofs)
      if bg_dof > 0
        @check dof > 0
        bg_fdof_to_fdof[bg_dof] = dof
      else
        @check dof < 0
        bg_ddof_to_ddof[-bg_dof] = -dof
      end
    end
  end
  return bg_fdof_to_fdof,bg_ddof_to_ddof
end

function get_bg_dof_to_dof(bg_f::SingleFieldFESpace,agg_f::FESpaceWithLinearConstraints)
  act_fdof_to_agg_fdof,act_ddof_to_agg_ddof = get_dof_to_mdof(agg_f)
  bg_fdof_to_act_fdof,bg_ddof_to_act_ddof = get_bg_dof_to_dof(bg_f,agg_f.space)
  bg_fdof_to_agg_fdof = compose_index(bg_fdof_to_act_fdof,act_fdof_to_agg_fdof)
  bg_ddof_to_agg_ddof = compose_index(bg_ddof_to_act_ddof,act_ddof_to_agg_ddof)
  return bg_fdof_to_agg_fdof,bg_ddof_to_agg_ddof
end

function get_dof_to_bg_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  fdof_to_bg_fdof = zeros(Int,num_free_dofs(f))
  ddof_to_bg_ddof = zeros(Int,num_dirichlet_dofs(f))
  bg_cell_ids = get_cell_dof_ids(bg_f)
  cell_ids = get_cell_dof_ids(f)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  cell_to_bg_cell = get_cell_to_bg_cell(f)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    for (bg_dof,dof) in zip(bg_dofs,dofs)
      if dof > 0
        @check bg_dof > 0
        fdof_to_bg_fdof[dof] = bg_dof
      else
        @check bg_dof < 0
        ddof_to_bg_ddof[-dof] = -bg_dof
      end
    end
  end
  return fdof_to_bg_fdof,ddof_to_bg_ddof
end

function get_dof_to_bg_dof(bg_f::SingleFieldFESpace,agg_f::FESpaceWithLinearConstraints)
  agg_fdof_to_act_fdof,agg_ddof_to_act_ddof = get_mdof_to_dof(agg_f)
  act_fdof_to_bg_fdof,act_ddof_to_bg_ddof = get_dof_to_bg_dof(bg_f,agg_f.space)
  agg_fdof_to_bg_fdof = compose_index(agg_fdof_to_act_fdof,act_fdof_to_bg_fdof)
  agg_ddof_to_bg_ddof = compose_index(agg_ddof_to_act_ddof,act_ddof_to_bg_ddof)
  return agg_fdof_to_bg_fdof,agg_ddof_to_bg_ddof
end

function get_dof_to_mdof(f::FESpaceWithLinearConstraints)
  T = eltype(f.mDOF_to_DOF)
  fdof_to_mfdof = zeros(T,num_free_dofs(f.space))
  ddof_to_mddof = zeros(T,num_dirichlet_dofs(f.space))
  cache = array_cache(f.DOF_to_mDOFs)
  for DOF in 1:length(f.DOF_to_mDOFs)
    mDOFs = getindex!(cache,f.DOF_to_mDOFs,DOF)
    dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
    for mDOF in mDOFs
      mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
      if dof > 0
        fdof_to_mfdof[dof] = mdof
      else
        ddof_to_mddof[-dof] = -mdof
      end
    end
  end
  return fdof_to_mfdof,ddof_to_mddof
end

function get_mdof_to_dof(f::FESpaceWithLinearConstraints)
  T = eltype(f.mDOF_to_DOF)
  mfdof_to_fdof = zeros(T,num_free_dofs(f))
  mddof_to_ddof = zeros(T,num_dirichlet_dofs(f))
  for mDOF in 1:length(f.mDOF_to_DOF)
    DOF = f.mDOF_to_DOF[mDOF]
    mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
    dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
    if mdof > 0
      mfdof_to_fdof[mdof] = dof
    else
      mddof_to_ddof[-mdof] = -dof
    end
  end
  return mfdof_to_fdof,mddof_to_ddof
end

function get_bg_dof_to_active_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  get_bg_dof_to_dof(bg_f,f)
end

function get_bg_dof_to_active_dof(bg_f::SingleFieldFESpace,f::FESpaceWithLinearConstraints)
  get_bg_dof_to_dof(bg_f,f.space)
end

function get_active_dof_to_bg_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  get_dof_to_bg_dof(bg_f,f)
end

function get_active_dof_to_bg_dof(bg_f::SingleFieldFESpace,f::FESpaceWithLinearConstraints)
  get_dof_to_bg_dof(bg_f,f.space)
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
      if dof > 0
        data[ptrs[dof]] = cell
        ptrs[dof] += 1
      end
    end
  end
  rewind_ptrs!(ptrs)

  Table(data,ptrs)
end

num_unconstrained_free_dofs(f::SingleFieldFESpace) = num_free_dofs(f)
num_unconstrained_free_dofs(f::ZeroMeanFESpace) = num_unconstrained_free_dofs(f.space)
num_unconstrained_free_dofs(f::FESpaceWithConstantFixed) = num_unconstrained_free_dofs(f.space)
