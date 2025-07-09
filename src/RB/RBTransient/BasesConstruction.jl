function RBSteady._zero_reduction(red::TTSVDReduction,A::TransientSnapshots{T,N}) where {T,N}
  cores = Vector{Array{T,3}}(undef,N-1)
  for d in 1:N-1
    s = d == N-1 ? size(A,N) : size(A,d)
    core = zeros(1,s,1)
    core[1] = 1.0
    cores[d] = core
  end
  return cores
end

function RBSteady._reduction(red::TTSVDReduction,A::TransientSnapshots,args...)
  red_style = ReductionStyle(red)
  cores,remainder = ttsvd(red_style,A,args...)
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
  remainder = reshape(cur_remainder,size(cur_core,3),size(remainder,3),:)
  push!(cores,cur_core)
  return
end

function tucker_loop(red::Reduction,A::AbstractMatrix,args...)
  red_style = ReductionStyle(red)
  Ur,Sr,Vr = tpod(red_style,A,args...)
  remainder = Diagonal(Sr)*Vr'
  return Ur,remainder
end

function tucker(red::AbstractVector{<:Reduction},A::AbstractArray{T,N}) where {T,N}
  @assert length(red) == N-1
  bases = Vector{Matrix{T}}(undef,N-1)
  remainder = first_unfold(A)
  for n in 1:N-1
    Ur,remainder = tucker_loop(red[n],remainder)
    bases[n] = Ur
  end
  return bases
end

function tucker(red::AbstractVector{<:Reduction},A::TransientSnapshots{T,N}) where {T,N}
  @assert length(red) == N-1
  nparams = num_params(A)
  bases = Vector{Matrix{T}}(undef,N-1)
  remainder = first_unfold(A)
  for n in 1:N-1
    Ur,remainder = tucker_loop(red[n],remainder)
    remainder = n == N-2 ? change_mode(remainder,nparams) : remainder
    bases[n] = Ur
  end
  return bases
end

function tucker(red::AbstractVector{<:Reduction},A::AbstractArray,X::AbstractSparseMatrix...)
  tucker(red,A,X)
end

function tucker(red::AbstractVector{<:Reduction},A::AbstractArray{T,N},X::NTuple{M}) where {T,N,M}
  @assert length(red) == N-1
  @assert M ≤ N
  bases = Vector{Matrix{T}}(undef,N-1)
  remainder = A
  for n in 1:N-1
    Ur,remainder = reduction(red[n],remainder,X[n])
    bases[n] = Ur
  end
  return bases
end

function tucker(red::AbstractVector{<:Reduction},A::TransientSnapshots{T,N},X::NTuple{M}) where {T,N,M}
  @assert length(red) == N-1
  @assert M ≤ N
  nparams = num_params(A)
  bases = Vector{Matrix{T}}(undef,N-1)
  remainder = A
  for n in 1:N-1
    Ur,remainder = tucker_loop(red[n],remainder)
    remainder = n == N-2 ? change_mode(remainder,nparams) : remainder
    bases[n] = Ur
  end
  return bases
end

first_unfold(A::AbstractArray{T,N}) where {T,N} = reshape(A,size(A,1),:)

function first_unfold(A::SubArray{T,N}) where {T,N}
  skeep = size(A,1)
  scale = prod(size(A)[2:N-1])
  iview = A.indices[end]
  rview = range_1d(1:scale,iview,scale)
  view(reshape(A.parent,skeep,:),:,rview)
end

function first_unfold(A::Snapshots)
  first_unfold(get_all_data(A))
end
