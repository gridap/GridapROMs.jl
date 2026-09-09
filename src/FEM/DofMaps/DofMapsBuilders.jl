"""
    get_dof_map(space::FESpace,args...) -> VectorDofMap

Returns the free dofs sorted by coordinate order, for every dimension. If `space` is a
D-dimensional, scalar `FESpace`, the output index map will be a subtype of
`AbstractDofMap{<:Integer,D}`. If `space` is a D-dimensional, vector-valued `FESpace`,
the output index map will be a subtype of `AbstractDofMap{D+1}`.
"""
function get_dof_map(f::FESpace,args...)
  _get_dof_map(f,args...)
end

function get_dof_map(f::SingleFieldFESpace)
  n = num_free_dofs(f)
  VectorDofMap(n)
end

function get_dof_map(f::MultiFieldFESpace)
  map(get_dof_map,f.spaces)
end

function get_sparse_dof_map(a::SparsityPattern,U::FESpace,V::FESpace)
  TrivialSparseMatrixDofMap(a)
end

function get_sparse_dof_map(a::TProductSparsity,U::FESpace,V::FESpace)
  Tu = get_dof_eltype(U)
  Tv = get_dof_eltype(V)
  try
    full_ids = get_d_sparse_dofs_to_full_dofs(Tu,Tv,a)
    sparse_ids = sparsify_indices(full_ids)
    SparseMatrixDofMap(sparse_ids,full_ids,a)
  catch
    @warn "Could not build sparse tensor-product dof mapping. Must represent the
    jacobian using a linear dof map"
    get_sparse_dof_map(a.sparsity,U,V)
  end
end

"""
    get_sparse_dof_map(trial::FESpace,test::FESpace,args...) -> AbstractDofMap

Returns the index maps related to Jacobians in a FE problem. The default output
is a `TrivialSparseMatrixDofMap`; when the trial and test spaces are of type
`TProductFESpace`, a `SparseMatrixDofMap` is returned.
"""
function get_sparse_dof_map(trial::FESpace,test::FESpace,args...)
  _get_sparse_dof_map(trial,test,args...)
end

function get_sparse_dof_map(trial::SingleFieldFESpace,test::SingleFieldFESpace)
  sparsity = get_sparsity(trial,test)
  get_sparse_dof_map(sparsity,trial,test)
end

function get_sparse_dof_map(trial::MultiFieldFESpace,test::MultiFieldFESpace)
  ntest = num_fields(test)
  ntrial = num_fields(trial)
  map(Iterators.product(1:ntest,1:ntrial)) do (i,j)
    get_sparse_dof_map(trial[j],test[i])
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

"""
    get_bg_fdof_to_fdof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace) -> AbstractVector

Given a background FE space `bg_f`, e.g. defined on a background discrete model,
and a FE space `f` defined on an active portion of such model, defines an index
mapping relating each free dof of `bg_f` to a free dof in `f`. In practice, it
returns a map `v` of length `bg_nfdofs` such that `v[bg_fdof] = fdof` if `bg_fdof`
is located in the active portion, and `v[bg_fdof] = 0` otherwise
"""
function get_bg_fdof_to_fdof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  bg_fdof_to_fdof,_ = get_bg_dof_to_dof(bg_f,f)
  bg_fdof_to_fdof
end

"""
    get_bg_ddof_to_ddof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace) -> AbstractVector

Given a background FE space `bg_f`, e.g. defined on a background discrete model,
and a FE space `f` defined on an active portion of such model, defines an index
mapping relating each Dirichlet dof of `bg_f` to a Dirichlet dof in `f`. In practice, it
returns a map `v` of length `bg_nddofs` such that `v[bg_ddof] = ddof` if `bg_ddof`
is located in the active portion, and `v[bg_ddof] = 0` otherwise
"""
function get_bg_ddof_to_ddof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  _,bg_ddof_to_ddof = get_bg_dof_to_dof(bg_f,f)
  bg_ddof_to_ddof
end

"""
    get_fdof_to_bg_fdof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace) -> AbstractVector

Given a background FE space `bg_f`, e.g. defined on a background discrete model,
and a FE space `f` defined on an active portion of such model, defines an index
mapping relating each free dof of `f` to a free dof in `bg_f`. In practice, it
returns a map `v` of length `nfdofs` such that `v[fdof] = bg_fdof`
"""
function get_fdof_to_bg_fdof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  fdof_to_bg_fdof,_ = get_dof_to_bg_dof(bg_f,f)
  fdof_to_bg_fdof
end

"""
    get_ddof_to_bg_ddof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace) -> AbstractVector

Given a background FE space `bg_f`, e.g. defined on a background discrete model,
and a FE space `f` defined on an active portion of such model, defines an index
mapping relating each Dirichlet dof of `f` to a Dirichlet dof in `bg_f`. In practice, it
returns a map `v` of length `nddofs` such that `v[ddof] = bg_ddof`
"""
function get_ddof_to_bg_ddof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  _,ddof_to_bg_ddof = get_dof_to_bg_dof(bg_f,f)
  ddof_to_bg_ddof
end

"""
    get_bg_dof_to_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace
      ) -> (AbstractVector,AbstractVector)

Given a background FE space `bg_f`, e.g. defined on a background discrete model,
and a FE space `f` defined on an active portion of such model, returns the results
of [`get_bg_fdof_to_fdof`](@ref) and [`get_bg_ddof_to_ddof`](@ref)
"""
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

"""
    get_dof_to_bg_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace
      ) -> (AbstractVector,AbstractVector)

Given a background FE space `bg_f`, e.g. defined on a background discrete model,
and a FE space `f` defined on an active portion of such model, returns the results
of [`get_fdof_to_bg_fdof`](@ref) and [`get_ddof_to_bg_ddof`](@ref)
"""
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

"""
    get_bg_dof_to_active_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace
      ) -> (AbstractVector,AbstractVector)

Same as [`get_bg_dof_to_dof`](@ref), unless `f` is a FESpaceWithLinearConstraints,
in which case it calls `get_bg_dof_to_dof` on the underlying, unconstrained space
"""
function get_bg_dof_to_active_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace)
  get_bg_dof_to_dof(bg_f,f)
end

function get_bg_dof_to_active_dof(bg_f::SingleFieldFESpace,f::FESpaceWithLinearConstraints)
  get_bg_dof_to_dof(bg_f,f.space)
end

"""
    get_active_dof_to_bg_dof(bg_f::SingleFieldFESpace,f::SingleFieldFESpace
      ) -> (AbstractVector,AbstractVector)

Same as [`get_dof_to_bg_dof`](@ref), unless `f` is a FESpaceWithLinearConstraints,
in which case it calls `get_dof_to_bg_dof` on the underlying, unconstrained space
"""
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

# utils 

function _get_dof_map(f::SingleFieldFESpace,b::AbstractVector{<:Number})
  @check num_free_dofs(f) == length(b)
  get_dof_map(f)
end

function _get_dof_map(f::SingleFieldFESpace,b::AbstractVector{<:AbstractVector})
  _get_dof_map(f,testitem(b))
end

function _get_dof_map(f::MultiFieldFESpace,b::AbstractVector)
  map(1:num_fields(f)) do i
    bi = restrict_to_field(f,b,i)
    _get_dof_map(f[i],bi)
  end
end

function _get_dof_map(f::FESpace,b::Contribution)
  contribution(b.trians) do trian 
    _get_dof_map(f,b[trian])
  end
end

function _get_sparse_dof_map(f::SingleFieldFESpace,g::SingleFieldFESpace,A::AbstractMatrix{<:Number})
  sparsity = get_sparsity(f,g,A)
  get_sparse_dof_map(sparsity,f,g)
end

function _get_sparse_dof_map(f::SingleFieldFESpace,g::SingleFieldFESpace,A::AbstractMatrix{<:AbstractMatrix})
  _get_sparse_dof_map(f,g,testitem(A))
end

function _get_sparse_dof_map(
  trial::MultiFieldFESpace,
  test::MultiFieldFESpace,
  A::AbstractMatrix
  )

  ntest = num_fields(test)
  ntrial = num_fields(trial)
  map(Iterators.product(1:ntest,1:ntrial)) do (i,j)
    Aij = restr_to_fields(A,i,j)
    t = !iszero(Aij)
    v = _get_sparse_dof_map(trial[j],test[i],Aij)
  end
end

function _get_sparse_dof_map(f::FESpace,g::FESpace,A::Contribution)
  contribution(A.trians) do trian
    _get_sparse_dof_map(f,g,A[trian])
  end
end

restr_to_fields(A::AbstractMatrix{<:Number},i,j) = _restrict_to_fields(A,test,trial,i,j)
restr_to_fields(A::BlockMatrix{<:Number},i,j) = A.blocks[i,j]
restr_to_fields(A::AbstractMatrix{<:AbstractMatrix},i,j) = restr_to_fields(testitem(A),i,j)

function _restrict_to_fields(
  A::AbstractMatrix{<:Number},
  test::MultiFieldFESpace,
  trial::MultiFieldFESpace,
  test_field::Int,
  trial_field::Int
  )

  row_offsets = _compute_field_offsets(test.spaces)
  col_offsets = _compute_field_offsets(trial.spaces)
  row_ini = row_offsets[test_field] + 1
  row_end = row_offsets[test_field] + num_free_dofs(test[test_field])
  col_ini = col_offsets[trial_field] + 1
  col_end = col_offsets[trial_field] + num_free_dofs(trial[trial_field])
  fast_getindex(A,row_ini:row_end,col_ini:col_end)
end

fast_getindex(A::AbstractMatrix,rows::AbstractUnitRange,cols::AbstractUnitRange) = view(A,rows,cols)

function fast_getindex(A::SparseMatrixCSC{Tv,Ti},rows::AbstractUnitRange,cols::AbstractUnitRange{Int}) where {Tv,Ti}
  m = length(rows)
  n = length(cols)
  row_first = first(rows)
  row_last = last(rows)
  colptrA = SparseArrays.getcolptr(A)
  rowvalA = rowvals(A)
  nzvalA = nonzeros(A)

  new_colptr = Vector{Ti}(undef,n+1)
  new_colptr[1] = 1

  # First pass: count non-zeros in each output column via binary search on sorted rowvals
  @inbounds for j_new in 1:n
    col = cols[j_new]
    lo = Int(colptrA[col])
    hi = Int(colptrA[col+1]) - 1
    if lo <= hi
      r_lo = searchsortedfirst(rowvalA,row_first,lo,hi,Base.Order.Forward)
      r_hi = searchsortedlast(rowvalA,row_last,lo,hi,Base.Order.Forward)
      new_colptr[j_new+1] = new_colptr[j_new] + max(0,r_hi-r_lo+1)
    else
      new_colptr[j_new+1] = new_colptr[j_new]
    end
  end

  nnzS = new_colptr[n+1] - 1
  new_rowval = Vector{Ti}(undef,nnzS)
  new_nzval = Vector{Tv}(undef,nnzS)

  # Second pass: fill values, shifting row indices to 1-based local numbering
  @inbounds for j_new in 1:n
    col = cols[j_new]
    lo = Int(colptrA[col])
    hi = Int(colptrA[col+1]) - 1
    if lo <= hi
      r_lo = searchsortedfirst(rowvalA,row_first,lo,hi,Base.Order.Forward)
      r_hi = searchsortedlast(rowvalA,row_last,lo,hi,Base.Order.Forward)
      ptr = new_colptr[j_new]
      for k in r_lo:r_hi
        new_rowval[ptr] = rowvalA[k] - row_first + 1
        new_nzval[ptr] = nzvalA[k]
        ptr += 1
      end
    end
  end

  SparseMatrixCSC(m,n,new_colptr,new_rowval,new_nzval)
end