"""
    recast_indices(fids::AbstractArray,a::AbstractSparseMatrix) -> AbstractArray

Input:
  - a sparse matrix `a` of size `(M,N)` and a number of nonzero entries `Nnz`
  - an array of indices `sids` with values `∈ {1,...,Nnz}` (sparse indices)
Output:
  - an array of indices `fids` with values `∈ {1,...,MN}` (full indices), whose
    entries are associated to those of `sids`. Zero entries are preserved
"""
function recast_indices(sids::AbstractArray,a::AbstractSparseMatrix)
  fids = similar(sids)
  fill!(fids,zero(eltype(fids)))
  I,J, = findnz(a)
  nrows = size(a,1)
  for (i,nzi) in enumerate(sids)
    if nzi > 0
      fids[i] = I[nzi] + (J[nzi]-1)*nrows
    end
  end
  return fids
end

"""
    recast_split_indices(fids::AbstractArray,a::AbstractSparseMatrix) -> (AbstractArray,AbstractArray)

Input:
  - a sparse matrix `a` of size `(M,N)` and a number of nonzero entries `Nnz`
  - an array of indices `sids` with values `∈ {1,...,Nnz}` (sparse indices)
Output:
  - an array rows of indices `frows` with values `∈ {1,...,M}` (full rows), whose
  entries are associated to those of `sids`. Zero entries are preserved
  - an array rows of indices `fcols` with values `∈ {1,...,N}` (full cols), whose
    entries are associated to those of `sids`. Zero entries are preserved
"""
function recast_split_indices(sids::AbstractArray,a::AbstractSparseMatrix)
  frows = similar(sids)
  fcols = similar(sids)
  fill!(frows,zero(eltype(frows)))
  fill!(fcols,zero(eltype(fcols)))
  I,J, = findnz(a)
  nrows = size(a,1)
  for (i,nzi) in enumerate(sids)
    if nzi > 0
      frows[i] = I[nzi]
      fcols[i] = J[nzi]
    end
  end
  return frows,fcols
end

"""
    sparsify_indices(sids::AbstractArray,a::AbstractSparseMatrix) -> AbstractArray

Input:
  - a sparse matrix `a` of size `(M,N)` and a number of nonzero entries `Nnz`
  - an array of indices `fids` with values `∈ {1,...,MN}` (full indices)
Output:
  - an array of indices `sids` with values `∈ {1,...,Nnz}` (sparse indices), whose
    entries are associated to those of `fids`. Zero entries are preserved
"""
function sparsify_indices(fids::AbstractArray,a::AbstractSparseMatrix)
  sids = similar(fids)
  fill!(sids,zero(eltype(sids)))
  nrows = size(a,1)
  for (j,jrowcol) in enumerate(fids)
    if jrowcol > 0
      jrow = fast_index(jrowcol,nrows)
      jcol = slow_index(jrowcol,nrows)
      sids[j] = nz_index(a,jrow,jcol)
    end
  end
  sids
end

# sparsify_indices in case we don't provide a sparse matrix
function sparsify_indices(fids::AbstractArray)
  sids = similar(fids)
  fill!(sids,zero(eltype(sids)))
  inz = findall(!iszero,fids)
  rowcol = sortperm(fids[inz])
  sids[inz[rowcol]] = LinearIndices(size(rowcol))
  sids
end

function sparsify_split_indices(
  frows::AbstractArray,
  fcols::AbstractArray,
  a::AbstractSparseMatrix
  )

  @assert length(frows) == length(fcols)
  sids = similar(frows)
  fill!(sids,zero(eltype(sids)))
  for j in eachindex(frows)
    jrow = frows[j]
    jcol = fcols[j]
    sids[j] = nz_index(a,jrow,jcol)
  end
  return sids
end

function change_matrix_sparsity(
  A::AbstractSparseMatrix,
  old_rows_to_rows::AbstractVector,
  old_cols_to_cols::AbstractVector,
  )
  
  @abstractmethod
end

function change_matrix_sparsity(
  A::SparseMatrixCSC,
  old_rows_to_rows::AbstractVector,
  old_cols_to_cols::AbstractVector,
  )

  nz_to_old_rows,nz_to_old_cols,nz_to_values = findnz(A)
  nz_to_rows = compose_index(nz_to_old_rows,old_rows_to_rows)
  nz_to_cols = compose_index(nz_to_old_cols,old_cols_to_cols)
  sparse(nz_to_rows,nz_to_cols,nz_to_values,A.m,A.n)
end

"""
    slow_index(i,nfast::Integer) -> Any

Returns the slow index in a tensor product structure. Suppose we have two matrices
`A` and `B` of sizes `Ra × Ca` and `Rb × Rb`. Their kronecker product `AB = A ⊗ B`,
of size `RaRb × CaCb`, can be indexed as

  `AB[i,j] = A[slow_index(i,RbCb)]B[fast_index(i,RbCb)]`,

where `nfast == RbCb`. In other words, this function converts an index belonging
to `AB` to an index belonging to `A`

"""
@inline slow_index(i,nfast::Integer) = cld.(i,nfast)
@inline slow_index(i::Integer,nfast::Integer) = cld(i,nfast)
@inline slow_index(i::Colon,::Integer) = i

"""
    fast_index(i,nfast::Integer) -> Any

Returns the fast index in a tensor product structure. Suppose we have two matrices
`A` and `B` of sizes `Ra × Ca` and `Rb × Rb`. Their kronecker product `AB = A ⊗ B`,
of size `RaRb × CaCb`, can be indexed as

  `AB[i,j] = A[slow_index(i,RbCb)] * B[fast_index(i,RbCb)]`,

where `nfast == RbCb`. In other words, this function converts an index belonging
to `AB` to an index belonging to `B`

"""
@inline fast_index(i,nfast::Integer) = mod.(i .- 1,nfast) .+ 1
@inline fast_index(i::Integer,nfast::Integer) = mod(i - 1,nfast) + 1
@inline fast_index(i::Colon,::Integer) = i

"""
    recast(v::AbstractVector,a::AbstractSparseMatrix) -> AbstractSparseMatrix

Returns a sparse matrix with values equal to `v`, and sparsity pattern equal to
that of `a`
"""
recast(v::AbstractArray,a::AbstractArray) = @abstractmethod
recast(v::AbstractVector,a::SparseMatrixCSC) = SparseMatrixCSC(a.m,a.n,a.colptr,a.rowval,v)
recast(v::AbstractVector,a::SparseMatrixCSR{Bi}) where Bi = SparseMatrixCSR{Bi}(a.m,a.n,a.rowptr,a.colval,v)

sparsify(a::AbstractArray) = nonzeros(a)

"""
    compose_index(i1_to_i2,i2_to_i3) -> AbstractVector

Returns an index map `i1_to_i3` representing the composition `i1_to_i2 ∘ i2_to_i3`
"""
function compose_index(i1_to_i2,i2_to_i3)
  T_i3 = eltype(i2_to_i3)
  n_i1 = length(i1_to_i2)
  i1_to_i3 = zeros(T_i3,n_i1)
  for (i1,i2) in enumerate(i1_to_i2)
    i3 = i2 > 0 ? i2_to_i3[i2] : i2_to_i3[-i2]
    @check sign(i2) == sign(i3)
    i1_to_i3[i1] = i3
  end
  return i1_to_i3
end

"""
    get_group_to_labels(labels::AbstractVector) -> Table

Given a vector of natural numbers `labels`, returns a table of length
`maximum(labels) - minimum(labels)`, whose entries group equal entries of `labels`.
An empty group is assigned whenever a label does not appear in `labels`
Example:
```
julia> labels = [1,3,2,3,1,5,5]
7-element Vector{Int64}:
 1
 3
 2
 3
 1
 5
 5
julia> get_group_to_labels(labels)
5-element Table{Int32, Vector{Int32}, Vector{Int32}}:
 [1, 1]
 [2]
 [3, 3]
 []
 [5, 5]
```
"""
function get_group_to_labels(labels::AbstractVector)
  labmin,labmax = extrema(labels)
  labels = labmin > 0 ? labels : labels .- labmin .+ 1
  perm = sortperm(labels)
  ptrs = zeros(Int32,1+maximum(labels))
  data = zeros(Int32,length(labels))
  count = 0
  for (i,p) in enumerate(perm)
    labi = labels[p]
    ptrs[1+labi] += 1
    data[i] = labi
  end
  length_to_ptrs!(ptrs)
  return Table(data,ptrs)
end

"""
    group_labels(labels::AbstractVector) -> Table

Same as [`get_group_to_labels`](@ref), but removes the empty groups corresponding
to non-existing labels in `labels`
Example:
```
julia> labels = [1,3,2,3,1,5,5]
7-element Vector{Int64}:
 1
 3
 2
 3
 1
 5
 5
julia> group_labels(labels)
4-element Table{Int32, Vector{Int32}, Vector{Int32}}:
 [1, 1]
 [2]
 [3, 3]
 [5, 5]
```
"""
function group_labels(labels::AbstractVector)
  labmin,labmax = extrema(labels)
  labels = labmin > 0 ? labels : labels .- labmin .+ 1
  perm = sortperm(labels)
  ptrs = zeros(Int32,maximum(labels))
  data = zeros(Int32,length(labels))
  count = 0
  for (i,p) in enumerate(perm)
    labi = labels[p]
    ptrs[labi] += 1
    data[i] = labi
  end
  ptrs = ptrs[findall(!iszero,ptrs)]
  pushfirst!(ptrs,zero(eltype(ptrs)))
  length_to_ptrs!(ptrs)
  return Table(data,ptrs)
end

"""
    get_group_to_ilabels(labels::AbstractVector) -> Table

Given a vector of natural numbers `labels`, returns a table of length
`maximum(labels) - minimum(labels)`, whose entries group the indices associated
with equal entries of `labels`. An empty group is assigned whenever a label does
not appear in `labels`
Example:
```
julia> labels = [1,3,2,3,1,5,5]
7-element Vector{Int64}:
 1
 3
 2
 3
 1
 5
 5
julia> get_group_to_ilabels(labels)
5-element Table{Int32, Vector{Int32}, Vector{Int32}}:
 [1, 5]
 [3]
 [2, 4]
 []
 [6, 7]
```
"""
function get_group_to_ilabels(labels::AbstractVector)
  labmin,labmax = extrema(labels)
  labels = labmin > 0 ? labels : labels .- labmin .+ 1
  perm = sortperm(labels)
  ptrs = zeros(Int32,1+maximum(labels))
  data = zeros(Int32,length(labels))
  count = 0
  for (i,p) in enumerate(perm)
    labi = labels[p]
    ptrs[1+labi] += 1
    data[i] = p
  end
  length_to_ptrs!(ptrs)
  return Table(data,ptrs)
end

"""
    group_ilabels(labels::AbstractVector) -> Table

Same as [`get_group_to_ilabels`](@ref), but removes the empty groups corresponding
to non-existing labels in `labels`
Example:
```
julia> labels = [1,3,2,3,1,5,5]
7-element Vector{Int64}:
 1
 3
 2
 3
 1
 5
 5
julia> group_ilabels(labels)
4-element Table{Int32, Vector{Int32}, Vector{Int32}}:
 [1, 5]
 [3]
 [2, 4]
 [6, 7]
```
"""
function group_ilabels(labels::AbstractVector)
  labmin,labmax = extrema(labels)
  labels = labmin > 0 ? labels : labels .- labmin .+ 1
  perm = sortperm(labels)
  ptrs = zeros(Int32,maximum(labels))
  data = zeros(Int32,length(labels))
  count = 0
  for (i,p) in enumerate(perm)
    labi = labels[p]
    ptrs[labi] += 1
    data[i] = p
  end
  ptrs = ptrs[findall(!iszero,ptrs)]
  pushfirst!(ptrs,zero(eltype(ptrs)))
  length_to_ptrs!(ptrs)
  return Table(data,ptrs)
end