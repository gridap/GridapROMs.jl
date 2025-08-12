"""
    get_dof_map(space::FESpace) -> VectorDofMap

Returns the active dofs sorted by coordinate order, for every dimension. If `space` is a
D-dimensional, scalar `FESpace`, the output index map will be a subtype of
`AbstractDofMap{<:Integer,D}`. If `space` is a D-dimensional, vector-valued `FESpace`,
the output index map will be a subtype of `AbstractDofMap{D+1}`.
"""
function get_dof_map(f::SingleFieldFESpace,args...)
  n = num_free_dofs(f)
  VectorDofMap(n)
end

function get_dof_map(f::MultiFieldFESpace,args...)
  map(f -> get_dof_map(f,args...),f.spaces)
end

function get_sparse_dof_map(a::SparsityPattern,U::FESpace,V::FESpace,args...)
  TrivialSparseMatrixDofMap(a)
end

function get_sparse_dof_map(a::TProductSparsity,U::FESpace,V::FESpace,args...)
  Tu = get_dof_eltype(U)
  Tv = get_dof_eltype(V)
  try
    full_ids = get_d_sparse_dofs_to_full_dofs(Tu,Tv,a)
    sparse_ids = sparsify_indices(full_ids)
    SparseMatrixDofMap(sparse_ids,full_ids,a)
  catch
    msg = "Could not build sparse tensor-product dof mapping. Must represent the
    jacobian using a linear dof map"
    println(msg)
    get_sparse_dof_map(a.sparsity,U,V,args...)
  end
end

"""
    get_sparse_dof_map(trial::FESpace,test::FESpace,args...) -> AbstractDofMap

Returns the index maps related to Jacobiansin a FE problem. The default output
is a `TrivialSparseMatrixDofMap`; when the trial and test spaces are of type
`TProductFESpace`, a `SparseMatrixDofMap` is returned.
"""
function get_sparse_dof_map(trial::SingleFieldFESpace,test::SingleFieldFESpace,args...)
  sparsity = get_sparsity(trial,test,args...)
  get_sparse_dof_map(sparsity,trial,test,args...)
end

function get_sparse_dof_map(trial::MultiFieldFESpace,test::MultiFieldFESpace,args...)
  ntest = num_fields(test)
  ntrial = num_fields(trial)
  map(Iterators.product(1:ntest,1:ntrial)) do (i,j)
    get_sparse_dof_map(trial[j],test[i],args...)
  end
end

# utils

function get_cell_to_bg_cell(f::SingleFieldFESpace)
  get_cell_to_bg_cell(get_triangulation(f))
end

function get_bg_cell_to_cell(f::SingleFieldFESpace)
  get_bg_cell_to_cell(get_triangulation(f))
end

function get_cell_to_bg_cell(trian::Triangulation{Dt,Dp}) where {Dt,Dp}
  glue = get_glue(trian,Val(Dp))
  glue.tface_to_mface
end

function get_bg_cell_to_cell(trian::Triangulation{Dt,Dp}) where {Dt,Dp}
  cell_to_bg_cell = get_cell_to_bg_cell(trian)
  bgmodel = get_background_model(trian)
  ncells = num_cells(trian)
  nbgcells = num_cells(bgmodel)
  bg_cell_to_cell = zeros(eltype(cell_to_bg_cell),nbgcells)
  bg_cell_to_cell[cell_to_bg_cell] .= 1:ncells
  return bg_cell_to_cell
end

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
  cell_to_bg_cell = get_cell_to_bg_cell(f)
  get_bg_dof_to_dof!(bg_fdof_to_fdof,bg_ddof_to_ddof,bg_cell_ids,cell_ids,cell_to_bg_cell)
end

function get_bg_dof_to_dof!(
  bg_fdof_to_fdof,bg_ddof_to_ddof,
  bg_cell_ids::AbstractArray,
  cell_ids::AbstractArray,
  cell_to_bg_cell::AbstractVector
  )

  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    for (bg_dof,dof) in zip(bg_dofs,dofs)
      if bg_dof > 0
        @check dof > 0
        bg_fdof_to_fdof[bg_dof] = dof
      else
        @check dof < 0
        bg_ddof_to_ddof[-bg_dof] = dof
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
  cell_to_bg_cell = get_cell_to_bg_cell(f)
  get_dof_to_bg_dof!(fdof_to_bg_fdof,ddof_to_bg_ddof,bg_cell_ids,cell_ids,cell_to_bg_cell)
end

function get_dof_to_bg_dof!(
  fdof_to_bg_fdof,ddof_to_bg_ddof,
  bg_cell_ids::AbstractArray,
  cell_ids::AbstractArray,
  cell_to_bg_cell::AbstractVector
  )

  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    for (bg_dof,dof) in zip(bg_dofs,dofs)
      if dof > 0
        @check bg_dof > 0
        fdof_to_bg_fdof[dof] = bg_dof
      else
        @check bg_dof < 0
        ddof_to_bg_ddof[-dof] = bg_dof
      end
    end
  end
  fdof_to_bg_fdof,ddof_to_bg_ddof
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
        @check mdof > 0
        fdof_to_mfdof[dof] = mdof
      else
        @check mdof < 0
        ddof_to_mddof[-dof] = mdof
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
      @check dof > 0
      mfdof_to_fdof[mdof] = dof
    else
      @check dof < 0
      mddof_to_ddof[-mdof] = dof
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

function get_dof_to_cells(cell_dofs::AbstractVector)
  inverse_table(Table(cell_dofs))
end

num_unconstrained_free_dofs(f::SingleFieldFESpace) = num_free_dofs(f)
num_unconstrained_free_dofs(f::ZeroMeanFESpace) = num_unconstrained_free_dofs(f.space)
num_unconstrained_free_dofs(f::FESpaceWithConstantFixed) = num_unconstrained_free_dofs(f.space)
