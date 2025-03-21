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
  full_ids = get_d_sparse_dofs_to_full_dofs(Tu,Tv,a)
  sparse_ids = sparsify_indices(full_ids)
  SparseMatrixDofMap(sparse_ids,full_ids,a)
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

for f in (:get_bg_dof_to_mask,:get_bg_dof_to_act_dof)
  @eval begin
    function $f(f::MultiFieldFESpace,args...)
      map(space -> $f(space,args...),f.spaces)
    end
  end
end

"""
    get_bg_dof_to_mask(f::FESpace,args...) -> Vector{Bool}

Associates a boolean mask to each background DOF of a `FESpace` `f`. If the DOF
is active, the mask is `false`. If the DOF is inactive (e.g., because it is
constrained), the mask is `true`
"""
function get_bg_dof_to_mask(f::SingleFieldFESpace,args...)
  map(iszero,get_bg_dof_to_act_dof(f,args...))
end

function get_bg_dof_to_act_dof(f::SingleFieldFESpace)
  IdentityVector(num_free_dofs(f))
end

"""
    get_bg_dof_to_act_dof(f::FESpace,args...) -> AbstractVector{<:Integer}

Associates an active DOF to each background DOF of a `FESpace` `f`. This is done
by computing the cumulative sum of the output of [`get_bg_dof_to_mask`](@ref) on `f`
"""
function get_bg_dof_to_act_dof(f::FESpaceWithLinearConstraints)
  bg_space = f.space
  ndofs = num_free_dofs(bg_space)
  sdof_to_bdof = setdiff(1:ndofs,f.mDOF_to_DOF)
  get_mask_to_act_dof(sdof_to_bdof,ndofs)
end

function get_bg_dof_to_act_dof(f::FESpaceWithConstantFixed)
  bg_space = f.space
  ndofs = num_free_dofs(bg_space)
  get_mask_to_act_dof(f.dof_to_fix,ndofs)
end

function get_bg_dof_to_act_dof(f::ZeroMeanFESpace)
  bg_space = f.space
  get_bg_dof_to_act_dof(bg_space)
end

function get_bg_dof_to_act_dof(f::FESpace,ttrian::Triangulation)
  strian = get_triangulation(f)
  bg_bg_dof_to_bg_dof = get_bg_dof_to_act_dof(f) # potential underlying constraints
  if ttrian ≈ strian
    bg_dof_to_act_dof = bg_bg_dof_to_bg_dof
  else
    bg_dof_to_bg_bg_dof = findall(!iszero,bg_bg_dof_to_bg_dof)
    bg_bg_dof_to_act_dof = zeros(Int,length(bg_bg_dof_to_bg_dof))
    bg_cell_ids = get_cell_dof_ids(f)
    act_cell_ids = get_cell_dof_ids(f,ttrian)
    bg_cache = array_cache(bg_cell_ids)
    act_cache = array_cache(act_cell_ids)
    act_to_bg_cell = Utils.get_tface_to_mface(ttrian)
    for (act_cell,bg_cell) in enumerate(act_to_bg_cell)
      bg_dofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
      act_dofs = getindex!(act_cache,act_cell_ids,act_cell)
      for (bg_dof,act_dof) in zip(bg_dofs,act_dofs)
        if bg_dof > 0
          bg_bg_dof = bg_dof_to_bg_bg_dof[bg_dof]
          bg_bg_dof_to_act_dof[bg_bg_dof] = act_dof
        end
      end
    end
  end
  return bg_bg_dof_to_act_dof
end

function get_mask_to_act_dof(masked_dofs::AbstractVector{<:Integer},ndofs::Int)
  bg_dof_to_act_dof = ones(Int,ndofs)
  _hide_masked_dofs!(bg_dof_to_act_dof,masked_dofs)
  cumsum!(bg_dof_to_act_dof)
  _hide_masked_dofs!(bg_dof_to_act_dof,masked_dofs)
  return bg_dof_to_act_dof
end

function get_mask_to_act_dof(masked_dofs::AbstractVector{<:Bool},ndofs::Int)
  @check length(masked_dofs) == ndofs
  bg_dof_to_act_dof = zeros(Int,ndofs)
  count = 0
  for (dof,mask) in enumerate(masked_dofs)
    if !mask
      count += 1
      bg_dof_to_act_dof[dof] = count
    end
  end
  return bg_dof_to_act_dof
end

function _hide_masked_dofs!(v,masked_dofs)
  for dof in masked_dofs
    v[dof] = 0
  end
end

"""
    get_polynomial_order(f::FESpace) -> Integer

Retrieves the polynomial order of `f`
"""
get_polynomial_order(f::SingleFieldFESpace) = get_polynomial_order(get_fe_basis(f))
get_polynomial_order(f::MultiFieldFESpace) = maximum(map(get_polynomial_order,f.spaces))

function get_polynomial_order(basis)
  cell_basis = get_data(basis)
  shapefun = testitem(cell_basis)
  get_order(shapefun.fields)
end

"""
    get_polynomial_orders(fs::FESpace) -> Integer

Retrieves the polynomial order of `fs` for every dimension
"""
get_polynomial_orders(fs::SingleFieldFESpace) = get_polynomial_orders(get_fe_basis(fs))
get_polynomial_orders(fs::MultiFieldFESpace) = maximum.(map(get_polynomial_orders,fs.spaces))

function get_polynomial_orders(basis)
  cell_basis = get_data(basis)
  shapefun = testitem(cell_basis)
  get_orders(shapefun.fields)
end

function get_cell_to_bg_cell(f::SingleFieldFESpace)
  trian = get_triangulation(f)
  D = num_cell_dims(trian)
  glue = get_glue(trian,Val(D))
  glue.tface_to_mface
end

function get_bg_cell_to_cell(f::SingleFieldFESpace)
  trian = get_triangulation(f)
  D = num_cell_dims(trian)
  glue = get_glue(trian,Val(D))
  glue.mface_to_tface
end

function get_cell_to_bg_cell(trian::Triangulation)
  Utils.get_tface_to_mface(trian)
end

function Base.cumsum!(a::AbstractVector{T}) where T
  s = zero(T)
  for (i,ai) in enumerate(a)
    s += ai
    a[i] = s
  end
end
