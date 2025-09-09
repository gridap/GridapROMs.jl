function PartitionedArrays.assembly_buffers(
  values::AbstractParamVector,
  local_indices_snd,
  local_indices_rcv
  )

  T = eltype2(values)
  ptrs = local_indices_snd.ptrs
  data = zeros(T,ptrs[end]-1,param_length(values))
  pdata = ConsecutiveParamArray(data)
  buffer_snd = JaggedArray(data,ptrs)
  ptrs = local_indices_rcv.ptrs
  data = zeros(T,ptrs[end]-1,param_length(values))
  pdata = ConsecutiveParamArray(data)
  buffer_rcv = JaggedArray(data,ptrs)
  buffer_snd,buffer_rcv
end

function PartitionedArrays.assembly_buffers(
  values::ParamSparseMatrix,
  local_indices_snd,
  local_indices_rcv
  )

  PartitionedArrays.assembly_buffers(get_all_data(values),local_indices_snd,local_indices_rcv)
end

function PartitionedArrays.assembly_buffers(
  values::AbstractMatrix,
  local_indices_snd,
  local_indices_rcv
  )

  T = eltype(values)
  ptrs = local_indices_snd.ptrs
  data = zeros(T,ptrs[end]-1,size(values,2))
  buffer_snd = JaggedArray(data,ptrs)
  ptrs = local_indices_rcv.ptrs
  data = zeros(T,ptrs[end]-1,size(values,2))
  buffer_rcv = JaggedArray(data,ptrs)
  buffer_snd,buffer_rcv
end

# in this case, we have to call PartitionedArrays's version
function PartitionedArrays.assembly_buffers(
  values::AbstractSparseMatrix,
  local_indices_snd,
  local_indices_rcv
  )

  T = eltype(values)
  ptrs = local_indices_snd.ptrs
  data = zeros(T,ptrs[end]-1)
  buffer_snd = JaggedArray(data,ptrs)
  ptrs = local_indices_rcv.ptrs
  data = zeros(T,ptrs[end]-1)
  buffer_rcv = JaggedArray(data,ptrs)
  buffer_snd,buffer_rcv
end

struct ParamJaggedArray{T,Ti,A} <: AbstractVector{A}
  data::Matrix{T}
  ptrs::Vector{Ti}
  array_type::Type{A}

  function ParamJaggedArray(a::A,ptrs::Vector{Ti}) where {T,Ti,A<:ConsecutiveParamVector{T}}
    data = get_all_data(a)
    new{T,Ti,A}(data,ptrs,A)
  end

  function ParamJaggedArray(data::A,ptrs::Vector{Ti}) where {T,Ti,A<:AbstractMatrix{T}}
    new{T,Ti,A}(data,ptrs,A)
  end

  function ParamJaggedArray{T,Ti}(a::A,ptrs::AbstractVector) where {T,Ti,A<:ConsecutiveParamVector{T}}
    data = get_all_data(a)
    new{T,Ti,A}(data,convert(Vector{Ti},ptrs),A)
  end

  function ParamJaggedArray{T,Ti}(data::A,ptrs::AbstractVector) where {T,Ti,A<:AbstractMatrix{T}}
    new{T,Ti,A}(data,convert(Vector{Ti},ptrs),A)
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

ParamDataStructures.param_length(a::ParamJaggedArray) = size(a.data,2)

function ParamDataStructures.param_getindex(a::ParamJaggedArray{T},i::Integer) where T
  ids = first(a.ptrs):last(a.ptrs)-1
  data = Vector{T}(undef,length(ids))
  for (ij,j) in enumerate(ids)
    data[ij] = a.data[j,i]
  end
  JaggedArray(data,a.ptrs)
end

function ParamDataStructures.parameterize(a::JaggedArray,plength::Integer)
  data = parameterize(a.data,plength)
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
  view(a.data,a.ptrs[i]:a.ptrs[i+1]-1,:)
end

function Base.getindex(a::ParamJaggedArray{T,Ti,<:ConsecutiveParamVector},i::Int) where {T,Ti}
  data = view(a.data,a.ptrs[i]:a.ptrs[i+1]-1,:)
  ConsecutiveParamArray(data)
end

function Base.setindex!(a::ParamJaggedArray,v,i::Int)
  axis1 = a.ptrs[i]:a.ptrs[i+1]-1
  axis2 = 1:param_length(a)
  scale = a.ptrs[end]-1
  ids = range_1d(axis1,axis2,scale)
  for (k,ki) in enumerate(ids)
    a.data[ki] = v[k]
  end
end
