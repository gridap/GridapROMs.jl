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
