"""
    abstract type SparsityPattern end

Type used to represent the sparsity pattern of a sparse matrix, usually the
Jacobian in a FE problem.

Subtypes:

- [`SparsityCSC`](@ref)
- [`CartesianSparsity`](@ref)
- [`TProductSparsity`](@ref)
"""
abstract type SparsityPattern end

"""
    get_sparsity(U::FESpace,V::FESpace,trian=_get_common_domain(U,V)) -> SparsityPattern

Builds a [`SparsityPattern`](@ref) from two `FESpace`s `U` and `V`, via integration
on a triangulation `trian`
"""
function get_sparsity(U::FESpace,V::FESpace,trian=_get_common_domain(U,V))
  SparsityPattern(U,V,trian)
end

function SparsityPattern(U::FESpace,V::FESpace,trian=_get_common_domain(U,V))
  a = SparseMatrixAssembler(U,V)
  m1 = nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  cellidsrows = get_cell_dof_ids(V,trian)
  cellidscols = get_cell_dof_ids(U,trian)
  trivial_symbolic_loop_matrix!(m1,cellidsrows,cellidscols)
  m2 = nz_allocation(m1)
  trivial_symbolic_loop_matrix!(m2,cellidsrows,cellidscols)
  m3 = create_from_nz(m2)
  SparsityPattern(m3)
end

function SparsityPattern(a::AbstractSparseMatrix)
  @abstractmethod
end

get_background_matrix(a::SparsityPattern) = @abstractmethod
get_background_sparsity(a::SparsityPattern) = SparsityPattern(get_background_matrix(a))

num_rows(a::SparsityPattern) = size(get_background_matrix(a),1)
num_cols(a::SparsityPattern) = size(get_background_matrix(a),2)
SparseArrays.findnz(a::SparsityPattern) = findnz(get_background_matrix(a))
SparseArrays.nnz(a::SparsityPattern) = nnz(get_background_matrix(a))
SparseArrays.nonzeros(a::SparsityPattern) = nonzeros(get_background_matrix(a))
SparseArrays.nzrange(a::SparsityPattern,row::Integer) = nzrange(get_background_matrix(a),row)
SparseArrays.rowvals(a::SparsityPattern) = rowvals(get_background_matrix(a))
SparseArrays.getcolptr(a::SparsityPattern) = SparseArrays.getcolptr(get_background_matrix(a))
Algebra.nz_index(a::SparsityPattern,row::Integer,col::Integer) = nz_index(get_background_matrix(a),row,col)

for f in (:recast,:recast_indices,:recast_split_indices,:sparsify_indices)
  @eval begin
    $f(A::AbstractArray,a::SparsityPattern) = $f(A,get_background_matrix(a))
  end
end

"""
    struct SparsityCSC{Tv,Ti} <: SparsityPattern
      matrix::SparseMatrixCSC{Tv,Ti}
    end

Sparsity pattern associated to a compressed sparse column matrix `matrix`
"""
struct SparsityCSC{Tv,Ti} <: SparsityPattern
  matrix::SparseMatrixCSC{Tv,Ti}
end

function SparsityPattern(a::SparseMatrixCSC)
  SparsityCSC(a)
end

get_background_matrix(a::SparsityCSC) = a.matrix

"""
    struct CartesianSparsity{A<:SparsityPattern,B<:AbstractVector,C<:AbstractVector} <: SparsityPattern
      sparsity::A
      bg_rows_to_act_rows::B
      bg_cols_to_act_cols::C
    end

Fields:

  - `sparsity`: the [`SparsityPattern`](@ref) of a matrix assembled on a (active) geometry
    which is either Cartesian, or it is defined from a (background) Cartesian geometry
  - `bg_rows_to_act_rows`: a vector that maps a row of the Cartesian background
    geometry, to a row of the active geometry
  - `bg_cols_to_act_cols`: a vector that maps a column of the Cartesian background
    geometry, to a column of the active geometry
"""
struct CartesianSparsity{A<:SparsityPattern,B<:AbstractVector,C<:AbstractVector} <: SparsityPattern
  sparsity::A
  bg_rows_to_act_rows::B
  bg_cols_to_act_cols::C
end

CartesianSparsity(a::SparsityPattern,::IdentityVector,::IdentityVector) = a

get_background_matrix(a::CartesianSparsity) = get_background_matrix(a.sparsity)

num_bg_rows(a::CartesianSparsity) = length(a.bg_rows_to_act_rows)
num_bg_cols(a::CartesianSparsity) = length(a.bg_cols_to_act_cols)

function bg_findnz(a::CartesianSparsity)
  I,J,V = findnz(a.sparsity)
  bg_rows = findall(!iszero,a.bg_rows_to_act_rows)
  bg_cols = findall(!iszero,a.bg_cols_to_act_cols)
  for k in eachindex(I)
    I[k] = bg_rows[I[k]]
    J[k] = bg_cols[J[k]]
  end
  return I,J,V
end

"""
    struct TProductSparsity{A<:SparsityPattern,B<:SparsityPattern} <: SparsityPattern
      sparsity::A
      sparsities_1d::Vector{B}
    end

Fields:
- `sparsity`: a [`SparsityPattern`](@ref) of a matrix assembled on a Cartesian
  geometry of dimension `D`
- `sparsities_1d`: a vector of `D` 1D sparsity patterns

Structure used to represent a `SparsityPattern` of a matrix obtained by integrating a
bilinear form on a triangulation that can be obtained as the tensor product of `D`
1D triangulations. For example, this can be done on a Cartesian mesh composed of
D-cubes
"""
struct TProductSparsity{A<:SparsityPattern,B<:SparsityPattern} <: SparsityPattern
  sparsity::A
  sparsities_1d::Vector{B}
end

get_background_matrix(a::TProductSparsity) = get_background_matrix(a.sparsity)

univariate_num_rows(a::TProductSparsity) = Tuple(num_rows.(a.sparsities_1d))
univariate_num_cols(a::TProductSparsity) = Tuple(num_cols.(a.sparsities_1d))
univariate_findnz(a::TProductSparsity) = tuple_of_arrays(findnz.(a.sparsities_1d))
univariate_nnz(a::TProductSparsity) = Tuple(nnz.(a.sparsities_1d))

"""
    get_d_sparse_dofs_to_full_dofs(Tu,Tv,a::TProductSparsity) -> AbstractArray{<:Integer,D}

Input:
- `Tu`, `Tv`: DOF types of a trial `FESpace` `U` and test `FESpace` `V`, respecively
- `a`: a [`TProductSparsity`](@ref) representing the sparsity of the matrix
  assembled from `U` and `V`

Output:
- a D-array `d_sparse_dofs_to_full_dofs`, which represents a map from
  `Nnz_1 × … × Nnz_{D}` to `M⋅N`, where `Nnz_i` represents the number of
  nonzero entries of the `i`th 1D sparsity contained in `a`, and `M⋅N` is
  the total length of the tensor product sparse matrix in `a`. For vector-valued
  `FESpace`s, an additional axis is added to `d_sparse_dofs_to_full_dofs` representing
  the number of components. In particular, the component axis has a length equal to
  `num_components(Tu)⋅num_components(Tv)`
"""
function get_d_sparse_dofs_to_full_dofs(Tu,Tv,a::TProductSparsity)
  I,J, = findnz(a)
  nrows = num_rows(a)
  ncols = num_cols(a)
  get_d_sparse_dofs_to_full_dofs(Tu,Tv,a,I,J,nrows,ncols)
end

function get_d_sparse_dofs_to_full_dofs(Tu,Tv,a::TProductSparsity{<:CartesianSparsity})
  I_bg,J_bg, = bg_findnz(a.sparsity)
  nrows_bg = num_bg_rows(a.sparsity)
  ncols_bg = num_bg_cols(a.sparsity)
  nrows = num_rows(a)
  dsd2sd = get_d_sparse_dofs_to_full_dofs(Tu,Tv,a,I_bg,J_bg,nrows_bg,ncols_bg)
  for (k,sdk) in enumerate(dsd2sd)
    if sdk > 0
      Ik_bg = fast_index(sdk,nrows_bg)
      Jk_bg = slow_index(sdk,nrows_bg)
      Ik = a.sparsity.bg_rows_to_act_rows[Ik_bg]
      Jk = a.sparsity.bg_cols_to_act_cols[Jk_bg]
      dsd2sd[k] = Ik+(Jk-1)*nrows
    end
  end
  return dsd2sd
end

function get_d_sparse_dofs_to_full_dofs(
  ::Type{<:Real},::Type{<:Real},a::TProductSparsity,I,J,nrows,ncols)
  _scalar_d_sparse_dofs_to_full_dofs(a,I,J,nrows,ncols)
end

function get_d_sparse_dofs_to_full_dofs(
  ::Type{Tu},::Type{Tv},a::TProductSparsity,I,J,nrows,ncols) where {Tu,Tv}
  _multivalue_d_sparse_dofs_to_full_dofs(a,I,J,nrows,ncols,num_components(Tu),num_components(Tv))
end

function _scalar_d_sparse_dofs_to_full_dofs(a::TProductSparsity,I,J,nrows,ncols)
  nnz_sizes = univariate_nnz(a)
  rows = univariate_num_rows(a)
  cols = univariate_num_cols(a)

  i,j, = univariate_findnz(a)
  d_to_nz_pairs = map((id,jd)->map(CartesianIndex,id,jd),i,j)

  D = length(a.sparsities_1d)
  cache = zeros(Int,D)
  dsd2sd = zeros(Int,nnz_sizes...)

  for k in eachindex(I)
    rows_1d = _index_to_d_indices(I[k],rows)
    cols_1d = _index_to_d_indices(J[k],cols)
    _row_col_pair_to_nz_index!(cache,rows_1d,cols_1d,d_to_nz_pairs)
    dsd2sd[cache...] = I[k]+(J[k]-1)*nrows
  end

  return dsd2sd
end

function _multivalue_d_sparse_dofs_to_full_dofs(a::TProductSparsity,I,J,nrows,ncols,ncomps_col,ncomps_row)
  nnz_sizes = univariate_nnz(a)
  rows_no_comps = univariate_num_rows(a)
  cols_no_comps = univariate_num_cols(a)
  nrows_no_comps = prod(rows_no_comps)
  ncols_no_comps = prod(cols_no_comps)
  ncomps = ncomps_row*ncomps_col

  i,j, = univariate_findnz(a)
  d_to_nz_pairs = map((id,jd)->map(CartesianIndex,id,jd),i,j)

  D = length(a.sparsities_1d)
  cache = zeros(Int,D)
  dsd2sd = zeros(Int,nnz_sizes...,ncomps)

  for k in eachindex(I)
    I_node,I_comp = _fast_and_slow_index(I[k],nrows_no_comps)
    J_node,J_comp = _fast_and_slow_index(J[k],ncols_no_comps)
    comp = I_comp+(J_comp-1)*ncomps_row
    rows_1d = _index_to_d_indices(I_node,rows_no_comps)
    cols_1d = _index_to_d_indices(J_node,cols_no_comps)
    _row_col_pair_to_nz_index!(cache,rows_1d,cols_1d,d_to_nz_pairs)
    dsd2sd[cache...,comp] = I[k]+(J[k]-1)*nrows
  end

  return dsd2sd
end

# utils

"""
    get_dof_eltype(f::FESpace) -> Type

Fetches the DOF eltype for a `FESpace` `f`
"""
get_dof_eltype(f::FESpace) = get_dof_eltype(get_fe_dof_basis(f))
get_dof_eltype(b) = @abstractmethod
get_dof_eltype(b::LagrangianDofBasis{P,V}) where {P,V} = change_eltype(V,Float64)
get_dof_eltype(dof::CellDof) = get_dof_eltype(testitem(get_data(dof)))

function trivial_symbolic_loop_matrix!(A,cellidsrows,cellidscols)
  mat1 = nothing
  rows_cache = array_cache(cellidsrows)
  cols_cache = array_cache(cellidscols)

  rows1 = getindex!(rows_cache,cellidsrows,1)
  cols1 = getindex!(cols_cache,cellidscols,1)

  touch! = FESpaces.TouchEntriesMap()
  touch_cache = return_cache(touch!,A,mat1,rows1,cols1)
  caches = touch_cache,rows_cache,cols_cache

  for cell in 1:length(cellidscols)
    rows = getindex!(rows_cache,cellidsrows,cell)
    cols = getindex!(cols_cache,cellidscols,cell)
    evaluate!(touch_cache,touch!,A,mat1,rows,cols)
  end

  return A
end

function _get_common_domain(U::FESpace,V::FESpace)
  msg = """\n
  Cannot define a sparsity pattern object between the `FESpace`s given as input,
  as they are defined on incompatible triangulations
  """

  trian_U = get_triangulation(U)
  trian_V = get_triangulation(V)
  sa_tV = is_change_possible(trian_U,trian_V)
  sb_tU = is_change_possible(trian_V,trian_U)
  if sa_tV && sb_tU
    target_trian = best_target(trian_U,trian_V)
  elseif !sa_tV && sb_tU
    target_trian = trian_U
  elseif sa_tV && !sb_tU
    target_trian = trian_V
  else
    @notimplemented msg
  end
end

function _row_col_pair_to_nz_index!(
  cache,
  rows_1d::NTuple{D,Int},
  cols_1d::NTuple{D,Int},
  d_to_nz_pairs::Vector{Vector{CartesianIndex{2}}}
  ) where D

  for d in 1:D
    index = 0
    row_d = rows_1d[d]
    col_d = cols_1d[d]
    index_d = CartesianIndex((row_d,col_d))
    nz_pairs = d_to_nz_pairs[d]
    for (i,nzi) in enumerate(nz_pairs)
      if nzi == index_d
        index = i
        break
      end
    end
    @assert !iszero(index) "Could not build sparse dof mapping"
    cache[d] = index
  end
  return cache
end

function _fast_and_slow_index(i::Integer,s::Integer)
  fast_index(i,s),slow_index(i,s)
end

function _index_to_d_indices(i::Integer,s2::NTuple{2,Integer})
  _fast_and_slow_index(i,s2[1])
end

function _index_to_d_indices(i::Integer,sD::NTuple{D,Integer}) where D
  sD_minus_1 = sD[1:end-1]
  nD_minus_1 = prod(sD_minus_1)
  iD = slow_index(i,nD_minus_1)
  i′ = fast_index(i,nD_minus_1)
  (_index_to_d_indices(i′,sD_minus_1)...,iD)
end
