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

function sparsify_split_indices(frows::AbstractArray,fcols::AbstractArray,a::AbstractSparseMatrix)
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

function inverse_table(cell_dofs::Table)
  ndofs = maximum(cell_dofs.data)
  ptrs = zeros(Int32,ndofs+1)
  for dof in cell_dofs.data
    if dof > 0
      ptrs[dof+1] += 1
    end
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

function invperm_table(a::Table)
  b = copy(a)
  cache = array_cache(a)
  for i in 1:length(a)
    ai = getindex!(cache,a,i)
    bi = invperm(ai)
    pini = a.ptrs[i]
    pend = a.ptrs[i+1]-1
    for (ip,p) in enumerate(pini:pend)
      b.data[p] = bi[ip]
    end
  end
  return b
end

function ptrs_to_length!(ptrs::AbstractVector)
  @inbounds for i in length(ptrs)-1:-1:1
    ptrs[i+1] -= ptrs[i]
  end
  ptrs[1] = zero(eltype(ptrs))
  return
end

for T in (:(AbstractVector{<:Table}),:(Tuple{Vararg{Table}}))
  S = T==:(AbstractVector{<:Table}) ? :(AbstractVector{<:AbstractVector}) : :(Tuple{Vararg{AbstractVector}})
  @eval begin
    # we require all the tables to have the same length
    function common_table(a::$T)
      a1, = a
      ptrs = zeros(eltype(a1.ptrs),length(a1.ptrs))
      for ai in a
        @check length(ai) == length(a1)
        ptrs_to_length!(ai.ptrs)
        ptrs += ai.ptrs
        length_to_ptrs!(ai.ptrs)
      end
      data = zeros(eltype(a1.data),sum(ptrs))
      offset = 0
      for i in 1:length(a1)
        datai = Set{Int}()
        for ai in a
          pinii = ai.ptrs[i]
          pendi = ai.ptrs[i+1]-1
          for p in pinii:pendi
            push!(datai,ai.data[p])
          end
        end
        for (k,datak) in enumerate(datai)
          data[offset+k] = datak
        end
        li = length(datai)
        offset += li
        deltai = ptrs[i+1]-li
        ptrs[i+1] = li
        resize!(data,length(data)-deltai)
      end
      length_to_ptrs!(ptrs)
      Table(data,ptrs)
    end

    # we do not require all the tables to have the same length
    function common_table(a::$S,j_to_bg_j::$T)
      n = 0
      for j_to_bg_ji in j_to_bg_j
        n = max(n,maximum(j_to_bg_ji))
      end
      a1, = a
      ptrs = zeros(eltype(a1.ptrs),n+1)
      for (ai,j_to_bg_ji) in zip(a,j_to_bg_j)
        ptrs_to_length!(ai.ptrs)
        for (j,bg_j) in enumerate(j_to_bg_ji)
          ptrs[bg_j+1] += ai.ptrs[j+1]
        end
        length_to_ptrs!(ai.ptrs)
      end
      data = zeros(eltype(a1.data),sum(ptrs))
      offset = 0
      loc = zeros(Int,length(a))
      for bg_i in 1:n
        touchedi = Set{Int}()
        fill!(loc,zero(Int))
        for (k,j_to_bg_jk) in enumerate(j_to_bg_j)
          j = findfirst(j_to_bg_jk .== bg_i)
          if !isnothing(j)
            push!(touchedi,k)
            loc[k] = j
          end
        end
        datai = Set{Int}()
        for k in touchedi
          ak = a[k]
          j_to_bg_jk = j_to_bg_j[k]
          j = loc[k]
          pinii = ak.ptrs[j]
          pendi = ak.ptrs[j+1]-1
          for p in pinii:pendi
            push!(datai,ak.data[p])
          end
        end
        for (k,datak) in enumerate(datai)
          data[offset+k] = datak
        end
        li = length(datai)
        offset += li
        deltai = ptrs[bg_i+1]-li
        ptrs[bg_i+1] = li
        resize!(data,length(data)-deltai)
      end
      length_to_ptrs!(ptrs)
      Table(data,ptrs)
    end
  end
end
