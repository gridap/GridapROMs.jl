"""
    reduction(red::Reduction,A::AbstractArray,args...) -> AbstractArray
    reduction(red::Reduction,A::AbstractArray,X::AbstractSparseMatrix) -> AbstractArray

Given an array (of snapshots) `A`, returns a reduced basis obtained by means of
the reduction strategy `red`
"""
function reduction(red::Reduction,A::AbstractArray,args...)
  iszero(A) ? _zero_reduction(red,A,args...) : _reduction(red,A,args...)
end

function _reduction(red::Reduction,A::AbstractArray,args...)
  @abstractmethod
end

function _reduction(red::PODReduction,A::AbstractArray,args...)
  red_style = ReductionStyle(red)
  U,S,V = tpod(red_style,A,args...)
  return U
end

function _reduction(red::TTSVDReduction,A::AbstractArray,args...)
  red_style = ReductionStyle(red)
  cores,remainder = ttsvd(red_style,A,args...)
  return cores
end

function _zero_reduction(red::Reduction,A::AbstractArray,args...)
  @abstractmethod
end

function _zero_reduction(red::PODReduction,A::AbstractArray)
  U = zeros(size(A,1),1)
  U[1] = 1.0
  return U
end

function _zero_reduction(red::TTSVDReduction,A::AbstractArray{T,N}) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  for d in 1:N-1
    core = zeros(1,size(A,d),1)
    core[1] = 1.0
    cores[d] = core
  end
  return cores
end

# somewhat arbitrary
function _size_cond(A::StridedMatrix)
  length(A) > 1e5 && (size(A,1) > 1e1*size(A,2) || size(A,2) > 1e1*size(A,1))
end

function _size_cond(A::AbstractMatrix)
  length(A) > 1e5 && (size(A,1) > 1e2*size(A,2) || size(A,2) > 1e2*size(A,1))
end

function _cholesky_decomp(X::AbstractSparseMatrix)
  C = try
    cholesky(X)
  catch
    @assert X ≈ X'
    cholesky((X+X')/2)
  end
  L = sparse(C.L)
  p = C.p
  return L,p
end

function _forward_cholesky(A::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector)
  Base.permuterows!(A,p)
  Ã = L'*A
  Base.invpermuterows!(A,p)
  return Ã
end

function _backward_cholesky(Ã::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector)
  A = L'\Ã
  Base.invpermuterows!(A,p)
  return A
end

function _truncate!(A::AbstractMatrix,rank)
  nrows = size(A,1)
  inds = nrows*rank+1:length(A)
  v = vec(A)
  Base.deleteat!(v,inds)
  reshape(v,nrows,:)
end

function _truncate!(v::AbstractVector,rank)
  Base.deleteat!(v,rank+1:length(v))
  v
end

function _truncate_row!(A::AbstractMatrix,rank)
  nrows = size(A,1)
  inds = range_1d(rank+1:nrows,axes(A,2),nrows)
  v = vec(A)
  Base.deleteat!(v,inds)
  reshape(v,rank,:)
end

function select_rank(red_style::ReductionStyle,args...)
  @abstractmethod
end

function select_rank(red_style::SearchSVDRank,S::AbstractVector)
  tol = red_style.tol
  energies = cumsum(S.^2;dims=1)
  rank = findfirst(energies .>= (1-tol^2)*energies[end])
  return rank
end

function truncated_svd(red_style::SearchSVDRank,A::AbstractMatrix;issquare=false)
  U,S,V = svd(A)
  if issquare S = sqrt.(S) end
  rank = select_rank(red_style,S)
  Ur = _truncate!(U,rank)
  Sr = _truncate!(S,rank)
  Vr = _truncate_row!(V',rank)
  return Ur,Sr,Vr'
end

function truncated_svd(red_style::FixedSVDRank,A::AbstractMatrix;issquare=false)
  U,S,V = svd(A)
  if issquare S = sqrt.(S) end
  rank = red_style.rank
  Ur = _truncate!(U,rank)
  Sr = _truncate!(S,rank)
  Vr = _truncate_row!(V',rank)
  return Ur,Sr,Vr'
end

function truncated_svd(red_style::LRApproxRank,A::AbstractMatrix;kwargs...)
  psvd(A,red_style.opts)
end

"""
    tpod(red_style::ReductionStyle,A::AbstractMatrix) -> AbstractMatrix
    tpod(red_style::ReductionStyle,A::AbstractMatrix,X::AbstractSparseMatrix) -> AbstractMatrix

Truncated proper orthogonal decomposition of `A`. When provided, `X` is a
(symmetric, positive definite) norm matrix with respect to which the output
is made orthogonal. If `X` is not provided, the output is orthogonal with respect
to the euclidean norm
"""
function tpod(red_style::ReductionStyle,A::AbstractMatrix,X::AbstractSparseMatrix)
  L,p = _cholesky_decomp(X)
  if _size_cond(A)
    massive_tpod(red_style,A,L,p)
  else
    tpod(red_style,A,L,p)
  end
end

function tpod(red_style::ReductionStyle,A::AbstractMatrix,X::AbstractRankTensor)
  tpod(red_style,A,kron(X))
end

function tpod(red_style::ReductionStyle,A::AbstractMatrix)
  if _size_cond(A)
    massive_tpod(red_style,A)
  else
    truncated_svd(red_style,A)
  end
end

function tpod(red_style::ReductionStyle,A::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector{Int})
  XA = _forward_cholesky(A,L,p)
  Ũr,Sr,Vr = truncated_svd(red_style,XA)
  Ur = _backward_cholesky(Ũr,L,p)
  return Ur,Sr,Vr
end

function massive_tpod(red_style::ReductionStyle,A::AbstractMatrix,args...)
  if size(A,1) > size(A,2)
    massive_rows_tpod(red_style,A,args...)
  else
    massive_cols_tpod(red_style,A,args...)
  end
end

function massive_tpod(red_style::LRApproxRank,A::AbstractMatrix)
  truncated_svd(red_style,A)
end

function massive_tpod(red_style::LRApproxRank,A::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector)
  tpod(red_style,A,L,p)
end

function massive_rows_tpod(red_style::ReductionStyle,A::AbstractMatrix)
  AA = A'*A
  _,Sr,Vr = truncated_svd(red_style,AA;issquare=true)
  Ur = (A*Vr)/Diagonal(Sr)
  return Ur,Sr,Vr
end

function massive_rows_tpod(red_style::ReductionStyle,A::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector{Int})
  XA = _forward_cholesky(A,L,p)
  AXA = XA'*XA
  _,Sr,Vr = truncated_svd(red_style,AXA;issquare=true)
  Ũr = (XA*Vr)/Diagonal(Sr)
  Ur = _backward_cholesky(Ũr,L,p)
  return Ur,Sr,Vr
end

function massive_cols_tpod(red_style::ReductionStyle,A::AbstractMatrix)
  AA = A*A'
  Ur,Sr,_ = truncated_svd(red_style,AA;issquare=true)
  Vr = Diagonal(Sr)\(Ur'A)
  return Ur,Sr,Vr'
end

function massive_cols_tpod(red_style::ReductionStyle,A::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector{Int})
  XA = _forward_cholesky(A,L,p)
  AXA = XA*XA'
  Ũr,Sr,_ = truncated_svd(red_style,AXA;issquare=true)
  Vr = Diagonal(Sr)\(Ũr'XA)
  Ur = _backward_cholesky(Ũr,L,p)
  return Ur,Sr,Vr'
end

function ttsvd_loop(red_style::ReductionStyle,A::AbstractArray{T,3}) where T
  A′ = reshape(A,size(A,1)*size(A,2),:)
  Ur,Sr,Vr = tpod(red_style,A′)
  core = reshape(Ur,size(A,1),size(A,2),:)
  remainder = Sr.*Vr'
  return core,remainder
end

function ttsvd_loop(red_style::ReductionStyle,A::AbstractArray{T,3},X::AbstractSparseMatrix) where T
  prev_rank = size(A,1)
  cur_size = size(A,2)
  A′ = reshape(A,prev_rank*cur_size,:)

  #TODO make this more efficient
  L,p = _cholesky_decomp(kron(X,I(prev_rank)))
  Ur,Sr,Vr = tpod(red_style,A′,L,p)

  core = reshape(Ur,prev_rank,cur_size,:)
  remainder = Sr.*Vr'
  return core,remainder
end

function matching_ttsvd_loop(red_style::ReductionStyle,A::AbstractArray{T,3},X::AbstractSparseMatrix) where T
  prev_rank = size(A,1)
  cur_size = size(A,2)
  A′ = reshape(A,prev_rank*cur_size,:)

  #TODO make this more efficient
  L,p = _cholesky_decomp(X)
  Ur,Sr,Vr = tpod(red_style,A′,L,p)

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
    remainder = reshape(cur_remainder,oldrank,size(A,d+1),:)
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

  weight = ones(1,rank(X),1)
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

first_unfold_3D(A::AbstractArray{T,N}) where {T,N} = reshape(A,1,size(A,1),:)

function first_unfold_3D(A::SubArray{T,N}) where {T,N}
  skeep = 1,size(A,1)
  scale = prod(size(A)[2:N-1])
  iview = A.indices[end]
  rview = range_1d(1:scale,iview,scale)
  view(reshape(A.parent,skeep...,:),:,:,rview)
end

function first_unfold_3D(A::Snapshots)
  first_unfold_3D(get_all_data(A))
end

function orthogonalize!(cores::AbstractVector,X::AbstractRankTensor{D}) where D
  red_style = LRApproxRank(1e-10)
  weight = ones(1,rank(X),1)
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
  N = size(core,2)

  cur_weight = zeros(rank,K,rank)
  core2D = reshape(permutedims(core,(2,1,3)),N,rrprev)
  cache_right = zeros(N,rrprev)
  cache_left = zeros(rrprev,rrprev)

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

  s1 = size(WD,1)*size(get_factor(X,D,1),1)
  s2 = size(WD,3)*size(get_factor(X,D,1),2)
  XW = zeros(s1,s2)
  cache = zeros(s1,s2)

  for k = 1:rank(X)
    @views WDk = WD[:,k,:]
    kron!(cache,get_factor(X,D,k),WDk)
    @. XW = XW + cache
  end
  @. XW = (XW+XW')/2 # needed to eliminate roundoff errors

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
  basis::AbstractMatrix)

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
  X::AbstractMatrix)

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
  args...)

  v .-= orth_projection(v,basis,args...)
end

"""
    gram_schmidt(A::AbstractMatrix,args...) -> AbstractMatrix
    gram_schmidt(A::AbstractMatrix,X::AbstractSparseMatrix,args...) -> AbstractMatrix

Gram-Schmidt orthogonalization for a matrix `A` under a Euclidean norm. A
(positive definite) sparse matrix `X` representing an inner product on the row space
of `A` can be provided to make the result orthogonal under a different norm
"""
function gram_schmidt end

for (f,g) in zip((:gram_schmidt,:gram_schmidt!),(:pivoted_qr,:pivoted_qr!))
  h = g==:pivoted_qr ? :qr : :qr!
  @eval begin
    function $f(A::AbstractMatrix,args...)
      Q, = $g(A,args...)
      return Q
    end

    function $f(A::AbstractMatrix,L::AbstractSparseMatrix,p::AbstractVector{Int},args...)
      XA = _forward_cholesky(A,L,p)
      Q̃, = $g(XA,args...)
      Q = _backward_cholesky(Q̃,L,p)
      return Q
    end

    function $f(A::AbstractMatrix,C::Factorization,args...)
      $f(A,sparse(C.L),C.p,args...)
    end

    function $f(A::AbstractMatrix,X::AbstractSparseMatrix,args...)
      $f(A,cholesky(X),args...)
    end

    function $g(A,tol=1e-10)
      C = $h(A,ColumnNorm())
      r = findlast(abs.(diag(C.R)) .> tol)
      Qr = C.Q[:,1:r]
      Rr = _truncate_row!(C.R,r)
      Base.invpermutecols!(Rr,C.jpvt)
      return Qr,Rr
    end
  end
end

for f in (:gram_schmidt,:gram_schmidt!)
  @eval begin
    function $f(A::AbstractMatrix,basis::AbstractMatrix,args...)
      $f(hcat(basis,A),args...)
    end
  end
end

# for testing purposes

function check_orthogonality(cores::AbstractVector{<:AbstractArray{T,3}},X::AbstractRankTensor) where T
  Xglobal = kron(X)
  basis = cores2basis(cores...)
  isorth = norm(basis'*Xglobal*basis - I) ≤ 1e-10
  return isorth
end
