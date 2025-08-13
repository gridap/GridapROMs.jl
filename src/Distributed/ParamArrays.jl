function PartitionedArrays.own_values(a::ConsecutiveParamVector,indices)
  ov = view(a.data,own_to_local(indices),:)
  ConsecutiveParamArray(ov)
end

function PartitionedArrays.ghost_values(a::ConsecutiveParamVector,indices)
  gv = view(a.data,ghost_to_local(indices),:)
  ConsecutiveParamArray(gv)
end

function ParamDataStructures.global_parameterize(a::PVector,plength::Integer)
  vector_partition = map(local_views(a)) do a
    global_parameterize(a,plength)
  end
  PVector(vector_partition,a.index_partition)
end

function ParamDataStructures.global_parameterize(a::PSparseMatrix,plength::Integer)
  matrix_partition = map(local_views(a)) do a
    global_parameterize(a,plength)
  end
  PSparseMatrix(matrix_partition,a.row_partition,a.col_partition,a.assembled)
end

function PartitionedArrays.assembly_buffers(
  values::AbstractParamVector,
  local_indices_snd,
  local_indices_rcv
  )

  plength = param_length(values)
  buffer_snd,buffer_rcv = PartitionedArrays.assembly_buffers(
    testitem(values),
    local_indices_snd,
    local_indices_rcv
  )
  pbuffer_snd = global_parameterize(buffer_snd,plength)
  pbuffer_rcv = global_parameterize(buffer_rcv,plength)
  pbuffer_snd,pbuffer_rcv
end

struct ParamJaggedArray{T,Ti,A} <: AbstractVector{A}
  data::Vector{T}
  ptrs::Vector{Ti}
  plength::Int
  array_type::Type{A}

  function ParamJaggedArray(
    a::ConsecutiveParamVector{T},
    ptrs::Vector{Ti}
    ) where {T,Ti}

    A = _get_jagged_type(a)
    data = vec(get_all_data(a))
    plength = param_length(a)
    new{T,Ti,A}(data,ptrs,plength,A)
  end

  function ParamJaggedArray(
    a::AbstractMatrix{T},
    ptrs::Vector{Ti}
    ) where {T,Ti}

    A = _get_jagged_type(a)
    data = vec(a)
    plength = size(a,2)
    new{T,Ti,A}(data,ptrs,plength,A)
  end

  function ParamJaggedArray{T,Ti}(
    a::ConsecutiveParamVector{T},
    ptrs::AbstractVector
    ) where {T,Ti}

    A = _get_jagged_type(a)
    data = vec(get_all_data(a))
    plength = param_length(a)
    new{T,Ti,A}(data,convert(Vector{Ti},ptrs),plength,A)
  end

  function ParamJaggedArray{T,Ti}(
    a::AbstractMatrix{T},
    ptrs::Vector{Ti}
    ) where {T,Ti}

    A = _get_jagged_type(a)
    data = vec(a)
    plength = size(a,2)
    new{T,Ti,A}(data,ptrs,plength,A)
  end
end

for A in (:ConsecutiveParamVector,:AbstractMatrix)
  @eval begin
    function PartitionedArrays.JaggedArray(data::$A,ptrs)
      ParamJaggedArray(data,ptrs)
    end

    function PartitionedArrays.JaggedArray{T,Ti}(data::$A,ptrs) where {T,Ti}
      ParamJaggedArray{T,Ti}(data,ptrs)
    end
  end
end

ParamDataStructures.param_length(a::ParamJaggedArray) = a.plength

function ParamDataStructures.global_parameterize(a::JaggedArray,plength::Int)
  data = global_parameterize(a.data,plength)
  ParamJaggedArray(data,a.ptrs)
end

PartitionedArrays.JaggedArray(data::AbstractParamVector,ptrs) = JaggedArray(data,ptrs)
PartitionedArrays.JaggedArray(a::AbstractParamVector{T}) where T = JaggedArray{T,Int32}(a)
PartitionedArrays.JaggedArray(a::ParamJaggedArray) = a
PartitionedArrays.JaggedArray{T,Ti}(a::ParamJaggedArray{T,Ti}) where {T,Ti} = a

function PartitionedArrays.JaggedArray{T,Ti}(a::AbstractArray{<:AbstractParamVector}) where {T,Ti}
  plength = param_length(first(a))
  @check all(param_length(ai) == plength for ai in a)

  n = length(a)
  ptrs = Vector{Ti}(undef,n+1)
  @inbounds for i in 1:n
    ai = a[i]
    ptrs[i+1] = innerlength(ai)
  end
  length_to_ptrs!(ptrs)

  ndata = ptrs[end]-one(eltype(ptrs))
  data = Matrix{T}(undef,ndata,plength)
  @inbounds for i in 1:n
    ai = a[i]
    for l in 1:plength
      for j in axes(ai.data,1)
        aijl = ai.data[j,l]
        data[(i-1)*n+j,l] = aijl
      end
    end
  end

  ParamJaggedArray(ConsecutiveParamArray(data),ptrs)
end

function PartitionedArrays.JaggedArray{T,Ti}(a::AbstractArray{<:AbstractMatrix}) where {T,Ti}
  plength = size(first(a),2)
  @check all(size(ai,2) == plength for ai in a)

  n = length(a)
  ptrs = Vector{Ti}(undef,n+1)
  @inbounds for i in 1:n
    ai = a[i]
    ptrs[i+1] = size(ai,1)
  end
  length_to_ptrs!(ptrs)

  ndata = ptrs[end]-one(eltype(ptrs))
  data = Matrix{T}(undef,ndata,plength)
  @inbounds for i in 1:n
    ai = a[i]
    for l in 1:plength
      for j in axes(ai,1)
        aijl = ai[j,l]
        data[(i-1)*n+j,l] = aijl
      end
    end
  end

  ParamJaggedArray(data,ptrs)
end

Base.size(a::ParamJaggedArray) = (length(a.ptrs)-1,)

function Base.getindex(a::ParamJaggedArray,i::Int)
  _getindex(a,i)
end

function Base.getindex(a::ParamJaggedArray{T,Ti,<:ConsecutiveParamVector},i::Int) where {T,Ti}
  ConsecutiveParamArray(_getindex(a,i))
end

function Base.setindex!(a::ParamJaggedArray,v,i::Int)
  _setindex!(a,v,i)
end

function Base.setindex!(a::ParamJaggedArray{T,Ti,<:ConsecutiveParamVector},v,i::Int) where {T,Ti}
  _setindex!(get_all_data(a),v,i)
end

function Base.resize!(a::ParamJaggedArray,n::Integer)
  δ = Int(length(a.data)/a.plength) - n
  if δ > 0
    for l in 1:a.plength
      Base._deleteat!(a.data,l*n+1,δ)
    end
  end
  a
end

# utils

_get_jagged_element(a) = @abstractmethod
function _get_jagged_element(a::T) where T<:AbstractVector
  view(zero(T),range_2d(1:1,1:1))
end
_get_jagged_element(a::AbstractMatrix{<:Number}) = _get_jagged_element(vec(a))
_get_jagged_element(a::AbstractParamArray) = ConsecutiveParamArray(_get_jagged_element(get_all_data(a)))

_get_jagged_type(a) = typeof(_get_jagged_element(a))

function _getindex(a::ParamJaggedArray,i::Int)
  axis1 = a.ptrs[i]:a.ptrs[i+1]-1
  axis2 = 1:a.plength
  scale = a.ptrs[end]-1
  ids = range_2d(axis1,axis2,scale)
  view(a.data,ids)
end

function _setindex!(a::ParamJaggedArray,v::AbstractArray,i::Int)
  axis1 = a.ptrs[i]:a.ptrs[i+1]-1
  axis2 = 1:a.plength
  scale = a.ptrs[end]-1
  ids = range_1d(axis1,axis2,scale)
  for (k,ki) in enumerate(ids)
    a.data[ki] = v[k]
  end
end

function _get_delta(a::ParamJaggedArray)
  length(a.ptrs)-1
end
