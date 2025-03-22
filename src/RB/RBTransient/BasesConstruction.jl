function RBSteady.reduction(red::TTSVDReduction,A::TransientSnapshots{T,N},args...) where {T,N}
  reduction(red,swap_param_time(A),args...)
end

function swap_param_time(A::TransientSnapshots{T,N}) where {T,N}
  data = get_all_data(A)
  keepdims = ntuple(i -> i,Val{N-2}())
  changedims = (N,N-1)
  permutedims(data,(keepdims...,changedims...))
end

function _permutedims(data::AbstractArray,dims::Dims)
  permutedims(data,dims)
end

function _permutedims(data::SubArray,dims::Dims{N}) where N
  data′ = permutedims(data.parent,dims)
  indices′ = ntuple(i -> data.indices[dims[i]],Val{N}())
  view(data′,indices′...)
end
