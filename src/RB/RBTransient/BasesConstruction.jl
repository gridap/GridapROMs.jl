# function RBSteady.reduction(red::TransientKroneckerReduction,A::TransientSnapshots{T,N},args...) where {T,N}
#   redvec = [get_reduction_space(red),get_reduction_time(red)]
#   tucker(redvec,swap_param_time(A),args...)
# end

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
