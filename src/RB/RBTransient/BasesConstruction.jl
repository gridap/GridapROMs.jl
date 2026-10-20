function RBSteady.reduction(red::SteadyReduction,A::TransientSnapshots,args...)
  reduction(red.reduction,A,args...)
end

function RBSteady.reduction(red::TTSVDReduction,A::TransientSnapshots,args...)
  red_style = ReductionStyle(red)
  nparams = num_params(A)
  ntimes = num_times(A)
  cores,remainder = ttsvd(red_style,get_all_data(A),args...)
  remainder = reshape(remainder,size(remainder,1),nparams,ntimes)
  add_temporal_core!(cores,red_style[end],remainder)
  return cores
end

RBSteady.last_dim(A::TransientSnapshots{T,N}) where {T,N} = N-2

function add_temporal_core!(
  cores::Vector{<:AbstractArray{T,3}},
  red_style::ReductionStyle,
  remainder::AbstractArray{T,3}
  ) where T

  remainder = permutedims(remainder,(1,3,2))
  cur_core,cur_remainder = RBSteady.ttsvd_loop(red_style,remainder)
  push!(cores,cur_core)
  return
end

function tucker_loop(red::Reduction,A::AbstractMatrix,args...)
  red_style = ReductionStyle(red)
  Ur,Sr,Vr = tpod(red_style,A,args...)
  remainder = Diagonal(Sr)*Vr'
  return Ur,remainder
end

"""
    tucker(red, A)
    tucker(red, A, X...)

Compute a Tucker decomposition of the multi-dimensional array `A` by sequentially
applying a truncated POD (via `tucker_loop`) along each unfolding mode. The
vector `red` provides a [`Reduction`](@ref) strategy for every mode (its length
must equal `ndims(A) - 1`). When sparse matrices `X` are supplied they are used
as inner-product weights for the corresponding modes.

Returns a `Vector{Matrix{T}}` of orthonormal basis matrices, one per mode.
"""
function tucker(red::AbstractVector{<:Reduction},A::AbstractArray{T,N}) where {T,N}
  @assert length(red) == N-1
  remainder = first_unfold(A)
  bases = ()
  for n in 1:N-1
    Ur,remainder = tucker_loop(red[n],remainder)
    bases = (bases...,Ur)
  end
  return bases
end

function tucker(red::AbstractVector{<:Reduction},A::TransientSnapshots)
  @assert length(red) == 2
  nparams = num_params(A)
  remainder = first_unfold(A)
  bases = ()
  for n in 1:2
    Ur,remainder = tucker_loop(red[n],remainder)
    remainder = n == 1 ? change_mode(remainder,nparams) : remainder
    bases = (bases...,Ur)
  end
  return bases
end

function tucker(
  red::AbstractVector{<:Reduction},
  A::AbstractArray,
  X::AbstractMatrix...
  )

  tucker(red,A,X)
end

function tucker(
  red::AbstractVector{<:Reduction},
  A::AbstractArray{T,N},
  X::NTuple{M,AbstractMatrix}
  ) where {T,N,M}

  @assert length(red) == N-1
  @assert M ≤ N
  remainder = first_unfold(A)
  bases = ()
  for n in 1:N-1
    Ur,remainder = n ≤ M ? tucker_loop(red[n],remainder,X[n]) : tucker_loop(red[n],remainder)
    bases = (bases...,Ur)
  end
  return bases
end

function tucker(
  red::AbstractVector{<:Reduction},
  A::TransientSnapshots,
  X::NTuple{M,AbstractMatrix}
  ) where M

  @assert length(red) == 2
  @assert M ≤ 2
  nparams = num_params(A)
  remainder = first_unfold(A)
  bases = ()
  for n in 1:2
    Ur,remainder = n ≤ M ? tucker_loop(red[n],remainder,X[n]) : tucker_loop(red[n],remainder)
    remainder = n == 1 ? change_mode(remainder,nparams) : remainder
    bases = (bases...,Ur)
  end
  return bases
end

function first_unfold(A::AbstractArray{T,N}) where {T,N}
  reshape(A,size(A,1),prod(size(A)[2:end]))
end

function first_unfold(A::SubArray{T,N}) where {T,N}
  skeep = size(A,1)
  scale = prod(size(A)[2:N-1])
  iview = A.indices[end]
  rview = range_1d(1:scale,iview,scale)
  # reshape(A.parent,skeep,:) can't infer the column count when skeep==0
  # (length(A.parent) is then also 0, so the ':' dimension is ambiguous and
  # silently collapses to 0 instead of the intended column count); compute it
  # directly from the parent's own shape instead, which needs no division.
  ncols = prod(size(A.parent)[2:end])
  view(reshape(A.parent,skeep,ncols),:,rview)
end

function first_unfold(A::Snapshots)
  first_unfold(get_all_data(A))
end