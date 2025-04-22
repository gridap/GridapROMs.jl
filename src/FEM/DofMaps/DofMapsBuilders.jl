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
