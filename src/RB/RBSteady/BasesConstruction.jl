"""
    reduction(red::Reduction,A::AbstractArray,args...) -> AbstractArray
    reduction(red::Reduction,A::AbstractArray,X::AbstractSparseMatrix) -> AbstractArray

Given an array (of snapshots) `A`, returns a reduced basis obtained by means of
the reduction strategy `red`
"""
function reduction(red::Reduction,A::AbstractArray,args...)
  @abstractmethod
end

function reduction(red::NoReduction,A::AbstractArray,args...)
  A
end

function reduction(red::PODReduction,A::AbstractArray,args...)
  red_style = ReductionStyle(red)
  U, = tpod(red_style,A,args...)
  return U
end

function reduction(red::TTSVDReduction,A::AbstractArray,args...)
  red_style = ReductionStyle(red)
  cores, = ttsvd(red_style,A,args...)
  return cores
end

function select_rank(red_style::ReductionStyle,args...)
  @abstractmethod
end

function select_rank(red_style::FixedSVDRank,args...)
  red_style.rank
end

function select_rank(red_style::SearchSVDRank,S::AbstractVector)
  tol = red_style.tol
  energies = cumsum(S.^2;dims=1)
  rank = 0
  for outer rank in eachindex(energies)
    energies[rank] >= (1-tol^2)*energies[end] && break
  end
  return rank
end

function truncated_svd(red_style::ReductionStyle,A::AbstractMatrix;issquare=false)
  U,S,V = svd(A)
  issquare && _root!(S)
  rank = select_rank(red_style,S)
  Ur = _truncate_col!(U,rank)
  Sr = _truncate!(S,rank)
  Vr = _truncate_row!(V',rank)
  return Ur,Sr,Vr'
end

function _root!(S)
  for i in eachindex(S)
    S[i] = sqrt(S[i])
  end
  S
end

"""
    tpod(red_style::ReductionStyle,A::AbstractMatrix) -> AbstractMatrix
    tpod(red_style::ReductionStyle,A::AbstractMatrix,X::MatrixOrTensor) -> AbstractMatrix

Truncated proper orthogonal decomposition of `A`. When provided, `X` is a
(symmetric, positive definite) norm matrix with respect to which the output
is made orthogonal. If `X` is not provided, the output is orthogonal with respect
to the euclidean norm
"""
function tpod(red_style::ReductionStyle,A::AbstractMatrix,args...)
  if _is_rectangular(A)
    method_of_snapshots(red_style,A,args...)
  else
    standard_tpod(red_style,A,args...)
  end
end

function tpod(red_style::ReductionStyle,A::AbstractMatrix,X::AbstractRankTensor)
  tpod(red_style,A,kron(X))
end

function standard_tpod(red_style::ReductionStyle,A::AbstractMatrix)
  truncated_svd(red_style,A)
end

function standard_tpod(red_style::ReductionStyle,A::AbstractMatrix,X::AbstractMatrix)
  L,p = _cholesky_decomp(X)
  XA = _forward_cholesky(A,L,p)
  Ũr,Sr,Vr = truncated_svd(red_style,XA)
  Ur = _backward_cholesky(Ũr,L,p)
  return Ur,Sr,Vr
end

function method_of_snapshots(red_style::ReductionStyle,A::AbstractMatrix,args...)
  if size(A,1) > size(A,2)
    method_of_snapshots_row(red_style,A,args...)
  else
    method_of_snapshots_col(red_style,A,args...)
  end
end

function method_of_snapshots_row(red_style::ReductionStyle,A::AbstractMatrix)
  _method_of_snapshots_row(red_style,A,A'*A)
end

function method_of_snapshots_row(red_style::ReductionStyle,A::AbstractMatrix,X::AbstractMatrix)
  _method_of_snapshots_row(red_style,A,A'*(X*A))
end

function _method_of_snapshots_row(red_style::ReductionStyle,A,AA)
  _,Sr,Vr = truncated_svd(red_style,AA;issquare=true)
  Ur = _weighted_mul_row(A,Vr,Sr)
  return Ur,Sr,Vr
end

function method_of_snapshots_col(red_style::ReductionStyle,A::AbstractMatrix)
  _method_of_snapshots_col(red_style,A,A*A')
end

function method_of_snapshots_col(red_style::ReductionStyle,A::AbstractMatrix,X::AbstractMatrix)
  standard_tpod(red_style,A,X)
end

function _method_of_snapshots_col(red_style::ReductionStyle,A,AA)
  Ur,Sr,_ = truncated_svd(red_style,AA;issquare=true)
  Vr = _weighted_mul_col(A,Ur,Sr)
  return Ur,Sr,Vr
end

function _weighted_mul_row(A,V,S)
  S .+= eps()
  U = zeros(eltype(V),size(A,1),length(S))
  D = Diagonal(S)
  mul!(U,A,V)
  rdiv!(U,D)
  U
end

function _weighted_mul_col(A,U,S)
  S .+= eps()
  V = zeros(eltype(U),size(A,2),length(S))
  D = Diagonal(S)
  mul!(V,A',U)
  rdiv!(V,D)
  return V
end

function ttsvd_loop(red_style::ReductionStyle,A::AbstractArray{T,3}) where T
  A′ = reshape(A,size(A,1)*size(A,2),size(A,3))
  Ur,Sr,Vr = tpod(red_style,A′)
  core = reshape(Ur,size(A,1),size(A,2),size(Ur,2))
  remainder = Sr.*Vr'
  return core,remainder
end

function ttsvd_loop(
  red_style::ReductionStyle,
  A::AbstractArray{T,3},
  X::AbstractSparseMatrix
  ) where T

  prev_rank = size(A,1)
  cur_size = size(A,2)
  A′ = reshape(A,prev_rank*cur_size,size(A,3))
  X′ = kron(X,I(prev_rank))
  Ur,Sr,Vr = tpod(red_style,A′,X′)
  core = reshape(Ur,prev_rank,cur_size,size(Ur,2))
  remainder = Sr.*Vr'
  return core,remainder
end

function matching_ttsvd_loop(
  red_style::ReductionStyle,
  A::AbstractArray{T,3},
  X::AbstractSparseMatrix
  ) where T

  prev_rank = size(A,1)
  cur_size = size(A,2)
  A′ = reshape(A,prev_rank*cur_size,:)
  Ur,Sr,Vr = tpod(red_style,A′,X)
  core = reshape(Ur,prev_rank,cur_size,:)
  remainder = Sr.*Vr'
  return core,remainder
end

"""
    ttsvd(red_style::TTSVDRanks,A::AbstractArray) -> AbstractVector{<:AbstractArray}
    ttsvd(red_style::TTSVDRanks,A::AbstractArray,X::AbstractRankTensor) -> AbstractVector{<:AbstractArray}

Tensor train SVD of `A`. When provided, `X` is a norm tensor (representing a
symmetric, positive definite matrix) with respect to which the output is made orthogonal.
Note: if `ndims(A)` = N, the length of the ouptput is `N-1`, since we are not
interested in reducing the axis of the parameters. Check [this](https://arxiv.org/abs/2412.14460)
reference for more details
"""
function ttsvd(
  red_style::TTSVDRanks,
  A::AbstractArray{T,N}
  ) where {T,N}

  cores = Array{T,3}[]
  remainder = first_unfold_3D(A)
  for d in 1:last_dim(A)
    cur_core,cur_remainder = ttsvd_loop(red_style[d],remainder)
    oldrank = size(cur_core,3)
    remainder = reshape(cur_remainder,oldrank,size(A,d+1),size(cur_remainder,2)÷size(A,d+1))
    push!(cores,cur_core)
  end
  return cores,remainder
end

function ttsvd(red_style::TTSVDRanks,A::AbstractArray,X::AbstractSparseMatrix)
  tpod(first(red_style),reshape(A,size(A,1),:),X)
end

function ttsvd(
  red_style::TTSVDRanks,
  A::AbstractArray{T,N},
  X::Rank1Tensor{D}
  ) where {T,N,D}

  @check D ≤ last_dim(A)

  cores = Array{T,3}[]
  remainder = first_unfold_3D(A)
  for d in 1:last_dim(A)
    if d ≤ D
      cur_core,cur_remainder = ttsvd_loop(red_style[d],remainder,X[d])
    else
      cur_core,cur_remainder = ttsvd_loop(red_style[d],remainder)
    end
    remainder = reshape(cur_remainder,size(cur_core,3),size(A,d+1),:)
    push!(cores,cur_core)
  end

  return cores,remainder
end

function ttsvd(
  red_style::TTSVDRanks,
  A::AbstractArray{T,N},
  X::GenericRankTensor{D,K}
  ) where {T,N,D,K}

  @check D ≤ last_dim(A)

  weight = ones(T,1,rank(X),1)
  decomp = get_decomposition(X)
  X′ = get_crossnorm(X)

  cores = Array{T,3}[]
  remainder = first_unfold_3D(A)
  for d in 1:last_dim(A)
    if d ≤ D-1
      cur_core,cur_remainder = ttsvd_loop(red_style[d],remainder,X′[d])
      X_d = getindex.(decomp,d)
      weight = weight_array(weight,cur_core,X_d)
    elseif d == D
      XW = ttnorm_array(X,weight)
      cur_core,cur_remainder = matching_ttsvd_loop(red_style[d],remainder,XW)
    else
      cur_core,cur_remainder = ttsvd_loop(red_style[d],remainder)
    end
    remainder = reshape(cur_remainder,size(cur_core,3),size(A,d+1),:)
    push!(cores,cur_core)
  end

  return cores,remainder
end

last_dim(A::AbstractArray{T,N}) where {T,N} = N-1

first_unfold_3D(A::AbstractArray{T,N}) where {T,N} = reshape(A,1,size(A,1),prod(size(A)[2:N]))

function first_unfold_3D(A::SubArray{T,N}) where {T,N}
  skeep = 1,size(A,1)
  scale = prod(size(A)[2:N-1])
  iview = A.indices[end]
  rview = range_1d(1:scale,iview,scale)
  ncols = prod(size(A.parent)[2:N])
  view(reshape(A.parent,skeep...,ncols),:,:,rview)
end

function first_unfold_3D(A::Snapshots)
  first_unfold_3D(get_all_data(A))
end

function orthogonalise!(cores::AbstractVector,X::AbstractRankTensor{D}) where D
  red_style = SearchSVDRank(1e-10)
  T = promote_type(map(eltype,cores)...)
  weight = ones(T,1,rank(X),1)
  decomp = get_decomposition(X)
  local remainder
  for d in eachindex(cores)
    cur_core = cores[d]
    if d < D
      cur_core′,remainder = reduce_rank(red_style,cur_core)
      X_d = getindex.(decomp,d)
      weight = weight_array(weight,cur_core′,X_d)
    elseif d == D
      XW = ttnorm_array(X,weight)
      cur_core′,remainder = reduce_rank(red_style,cur_core,XW)
    else d > D
      cur_core′,remainder = reduce_rank(red_style,cur_core)
    end
    cores[d] = cur_core′
    if d < length(cores)
      next_core = cores[d+1]
      cores[d+1] = absorb(next_core,remainder)
    end
  end
  return
end

function reduce_rank(red_style::ReductionStyle,core::AbstractArray{T,3},args...) where T
  mat = reshape(core,:,size(core,3))
  Ur,Sr,Vr = tpod(red_style,mat,args...)
  core′ = reshape(Ur,size(core,1),size(core,2),:)
  R = Sr.*Vr'
  return core′,R
end

function absorb(core::AbstractArray{T,3},R::AbstractMatrix) where T
  Rcore = R*reshape(core,size(core,1),:)
  return reshape(Rcore,size(Rcore,1),size(core,2),:)
end

function weight_array(prev_weight,core,X)
  @check length(X) == size(prev_weight,2)
  @check size(core,1) == size(prev_weight,1) == size(prev_weight,3)

  K = length(X)
  rank_prev = size(core,1)
  rank = size(core,3)
  rrprev = rank_prev*rank
  T = eltype(core)
  N = size(core,2)

  cur_weight = zeros(T,rank,K,rank)
  core2D = reshape(permutedims(core,(2,1,3)),N,rrprev)
  cache_right = zeros(T,N,rrprev)
  cache_left = zeros(T,rrprev,rrprev)

  @inbounds for k = 1:K
    Xk = X[k]
    @views Wk_prev = prev_weight[:,k,:]
    mul!(cache_right,Xk,core2D)
    mul!(cache_left,core2D',cache_right)
    resh_weight = reshape(permutedims(reshape(cache_left,rank_prev,rank,rank_prev,rank),(2,4,1,3)),rank^2,:)
    @views cur_weight[:,k,:] = reshape(resh_weight*vec(Wk_prev),rank,rank)
  end
  return cur_weight
end

function ttnorm_array(X::AbstractRankTensor{D,K},WD) where {D,K}
  @check size(WD,1) == size(WD,3)
  @check size(WD,2) == K
  @check all(size(get_factor(X,D,1)) == size(get_factor(X,D,k)) for k = 2:K)

  T = eltype(WD)
  s1 = size(WD,1)*size(get_factor(X,D,1),1)
  s2 = size(WD,3)*size(get_factor(X,D,1),2)
  XW = zeros(T,s1,s2)
  cache = zeros(T,s1,s2)

  for k = 1:rank(X)
    @views WDk = WD[:,k,:]
    kron!(cache,get_factor(X,D,k),WDk)
    @. XW = XW + cache
  end
  symmetrise!(XW) # needed to eliminate roundoff errors

  return sparse(XW)
end

"""
    orth_projection(v::AbstractVector, basis::AbstractMatrix, args...) -> AbstractVector

Orthogonal projection of `v` on the column space of `basis`. When a symmetric,
positive definite matrix `X` is provided as an argument, the output is `X`-orthogonal,
otherwise it is ℓ²-orthogonal
"""
function orth_projection(
  v::AbstractVector,
  basis::AbstractMatrix
  )

  proj = similar(v)
  fill!(proj,zero(eltype(proj)))
  @inbounds for b = eachcol(basis)
    proj += b*dot(v,b)/dot(b,b)
  end
  proj
end

function orth_projection(
  v::AbstractVector,
  basis::AbstractMatrix,
  X::AbstractMatrix
  )

  proj = similar(v)
  fill!(proj,zero(eltype(proj)))
  w = similar(proj)
  @inbounds for b = eachcol(basis)
    mul!(w,X,b)
    proj += b*dot(v,w)/dot(b,w)
  end
  proj
end

"""
    orth_complement!(v::AbstractVector,basis::AbstractMatrix,args...) -> Nothing

In-place orthogonal complement of `v` on the column space of `basis`. When a symmetric,
positive definite matrix `X` is provided as an argument, the output is `X`-orthogonal,
otherwise it is ℓ²-orthogonal
"""
function orth_complement!(
  v::AbstractVector,
  basis::AbstractMatrix,
  args...
  )

  v .-= orth_projection(v,basis,args...)
end

"""
    gram_schmidt(A::AbstractMatrix;kwargs...) -> AbstractMatrix
    gram_schmidt(A::AbstractMatrix,X::Union{AbstractMatrix,Factorization};kwargs...) -> AbstractMatrix

Gram-Schmidt orthogonalization for a matrix `A` under a Euclidean norm. A
(positive definite) sparse matrix `X` representing an inner product on the row space
of `A` can be provided to make the result orthogonal under a different norm
"""
function gram_schmidt(A::AbstractMatrix;tol=1e-10)
  Q,R, = qr!(A,ColumnNorm())
  rank = something(findlast(abs.(diag(R)) .> tol),0)
  Qr = _truncate_col!(Q,rank)
  return Qr
end

function gram_schmidt(A::AbstractMatrix,X::Union{AbstractMatrix,Factorization};tol=1e-10)
  Q,R, = weighted_qr!(A,X)
  rank = something(findlast(abs.(diag(R)) .> tol),0)
  Qr = _truncate_col!(Q,rank)
  return Qr
end

"""
    weighted_qr!(A::AbstractMatrix,X::Union{AbstractMatrix,Factorization}) -> (AbstractMatrix,AbstractMatrix)

Column-pivoted, rank-revealing QR decomposition of `A` with respect to the inner
product induced by the (positive definite) matrix `X`: returns `(Q,R)` such that
`A[:,p] ≈ Q*R` (for the internal pivot vector `p`) and `Q'*X*Q ≈ I`.
"""
function weighted_qr!(A::AbstractMatrix,X::Union{AbstractMatrix,Factorization})
  _mul(a,b) = a*b
  _mul(a::SparseArrays.CHOLMOD.Factor,b) = _forward_cholesky(b,a)
  A = copy(A)
  m,n = size(A)
  T = eltype(A)
  XA = _mul(X,A)
  p = collect(1:n)
  R = zeros(T,n,n)
  colnorms2 = zeros(real(T),n)
  for j in 1:n
    colnorms2[j] = real(dot(view(A,:,j),view(XA,:,j)))
  end
  for j in 1:min(m,n)
    j′ = argmax(view(colnorms2,j:n)) + j - 1
    if j′ != j
      tmp = p[j′]
      p[j′] = p[j]
      p[j] = tmp
      tmp = colnorms2[j′]
      colnorms2[j′] = colnorms2[j]
      colnorms2[j] = tmp
      Base.swapcols!(A,j,j′)
      Base.swapcols!(XA,j,j′)
    end
    normj = sqrt(max(colnorms2[j],zero(real(T))))
    R[j,j] = normj
    iszero(normj) && continue
    @views A[:,j] ./= normj
    @views XA[:,j] ./= normj
    for k in j+1:n
      rjk = dot(view(A,:,j),view(XA,:,k))
      R[j,k] = rjk
      @views A[:,k] .-= rjk .* A[:,j]
      @views XA[:,k] .-= rjk .* XA[:,j]
      colnorms2[k] = real(dot(view(A,:,k),view(XA,:,k)))
    end
  end
  return A,R,p
end

# utils 

function _is_rectangular(A::AbstractMatrix;ratio=10)
  m,n = size(A)
  m > ratio*n || n > ratio*m
end

function symcholesky(X::AbstractSparseMatrix;kwargs...)
  issymmetric(X) && return cholesky(X;check=false)
  @check symmetrise!(X;kwargs...)
  cholesky(X;check=false)
end

symcholesky(X::Rank1Tensor) = symcholesky.(get_factors(X))
symcholesky(X::GenericRankTensor) = symcholesky(get_crossnorm(X))

gram_solver(X;kwargs...) = symcholesky(X;kwargs...)

function symmetrise!(A::AbstractMatrix;atol=1e-12,rtol=1e-8)
  n,m = size(A)
  n == m || return false

  @inbounds for j in 1:n, i in 1:j-1
    !isapprox(A[i,j],A[j,i];atol,rtol) && return false
  end

  @inbounds for j in 1:n, i in 1:j-1
    s = (A[i,j] + A[j,i]) / 2
    A[i,j] = s
    A[j,i] = s
  end

  return true
end

function symmetrise!(A::BlockMatrix;atol=1e-12,rtol=1e-8)
  axes(A,1) == axes(A,2) || return false

  for k in 1:blocksize(A,1), l in 1:k-1
    Akl = blocks(A)[k,l]
    Alk = blocks(A)[l,k]
    size(Akl) == reverse(size(Alk)) || return false
    @inbounds for j in axes(Akl,2), i in axes(Akl,1)
      !isapprox(Akl[i,j],Alk[j,i];atol,rtol) && return false
    end
    @inbounds for j in axes(Akl,2), i in axes(Akl,1)
      s = (Akl[i,j] + Alk[j,i]) / 2
      Akl[i,j] = s
      Alk[j,i] = s
    end
  end

  for i in 1:blocksize(A,1)
    symmetrise!(blocks(A)[i,i];atol,rtol)
  end

  return true
end

function _cholesky_decomp(X::AbstractSparseMatrix)
  C = symcholesky(X)
  L = sparse(C.L)
  p = C.p
  return L,p
end

function _forward_cholesky(A::AbstractMatrix,C)
  _forward_cholesky(A,sparse(C.L),C.p)
end

function _forward_cholesky(A::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector)
  permuterows!(A,p)
  Ã = L'*A
  invpermuterows!(A,p)
  return Ã
end

function _backward_cholesky(Ã::AbstractMatrix,C)
  _backward_cholesky(Ã,sparse(C.L),C.p)
end

function _backward_cholesky(Ã::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector)
  A = L'\Ã
  invpermuterows!(A,p)
  return A
end

function _truncate!(v::AbstractVector,rank)
  rank == length(v) && return v
  Base.deleteat!(v,rank+1:length(v))
  v
end

function _truncate!(A::AbstractMatrix,rank)
  _truncate_col!(A,rank)
end

function _truncate_row!(A::AbstractMatrix,rank)
  rank == size(A,1) && return A
  nrows = size(A,1)
  inds = range_1d(rank+1:nrows,axes(A,2),nrows)
  v = vec(A)
  Base.deleteat!(v,inds)
  reshape(v,rank,:)
end

function _truncate_col!(A::AbstractMatrix,rank)
  rank == size(A,2) && return A
  nrows = size(A,1)
  inds = nrows*rank+1:length(A)
  v = vec(A)
  Base.deleteat!(v,inds)
  reshape(v,nrows,:)
end

for f in (:_truncate,:_truncate_row!,:_truncate_col!)
  @eval $f(A,rank) = $f(Array(A),rank)
end

permutecols!(a::AbstractMatrix,p::AbstractVector{<:Integer}) = _permute!(a,p,Base.swapcols!)
permuterows!(a::AbstractMatrix,p::AbstractVector{<:Integer}) = _permute!(a,p,Base.swaprows!)

@inline function _permute!(a::AbstractMatrix,p::AbstractVector{<:Integer},swapfun!) 
  Base.require_one_based_indexing(a,p)
  p .= .-p
  for i in eachindex(p)
    p[i] > 0 && continue
    j = i
    in = p[j] = -p[j]
    while p[in] < 0
      swapfun!(a,in,j)
      j = in
      in = p[in] = -p[in]
    end
  end
  a
end

invpermutecols!(a::AbstractMatrix,p::AbstractVector{<:Integer}) = _invpermute!(a,p,Base.swapcols!)
invpermuterows!(a::AbstractMatrix,p::AbstractVector{<:Integer}) = _invpermute!(a,p,Base.swaprows!)

@inline function _invpermute!(a::AbstractMatrix,p::AbstractVector{<:Integer},swapfun!) 
  Base.require_one_based_indexing(a,p)
  p .= .-p
  for i in eachindex(p)
    p[i] > 0 && continue
    j = p[i] = -p[i]
    while j != i
      swapfun!(a,j,i)
      j = p[j] = -p[j]
    end
  end
  a
end

