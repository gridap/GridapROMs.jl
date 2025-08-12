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

struct ParamJaggedArray{T,Ti,A<:ConsecutiveParamVector{T}} <: AbstractVector{SubArray{T,2,Matrix{T},Tuple{UnitRange{Ti},Base.Slice{Base.OneTo{Ti}}},false}}
  data::A
  ptrs::Vector{Ti}

  function ParamJaggedArray(
    data::A,
    ptrs::Vector{Ti}
    ) where {T,Ti,A<:ConsecutiveParamVector{T}}

    new{T,Ti,A}(data,ptrs)
  end

  function ParamJaggedArray{T,Ti}(
    data::A,
    ptrs::AbstractVector
    ) where {T,Ti,A<:ConsecutiveParamVector{T}}

    new{T,Ti,A}(data,convert(Vector{Ti},ptrs))
  end
end

function PartitionedArrays.JaggedArray(data::ConsecutiveParamVector,ptrs)
  ParamJaggedArray(data,ptrs)
end

function PartitionedArrays.JaggedArray{T,Ti}(data::ConsecutiveParamVector,ptrs) where {T,Ti}
  ParamJaggedArray{T,Ti}(data,ptrs)
end

ParamDataStructures.param_length(a::ParamJaggedArray) = param_length(a.data)

function ParamDataStructures.global_parameterize(a::JaggedArray,plength::Int)
  data = global_parameterize(a.data,plength)
  ParamJaggedArray(data,a.ptrs)
end

PartitionedArrays.JaggedArray(data::AbstractParamVector,ptrs) = JaggedArray(data,ptrs)
PartitionedArrays.JaggedArray(a::AbstractParamVector{T}) where T = JaggedArray{T,Int32}(a)
PartitionedArrays.JaggedArray(a::ParamJaggedArray) = a
PartitionedArrays.JaggedArray{T,Ti}(a::ParamJaggedArray{T,Ti}) where {T,Ti} = a

function PartitionedArrays.JaggedArray{T,Ti}(a::AbstractArray{<:AbstractParamVector}) where {T,Ti}
  plength = param_length(testitem(a))
  @check all(param_length(ai) == plength for ai in a)

  n = length(a)
  ptrs = Vector{Ti}(undef,n+1)
  u = one(eltype(ptrs))
  @inbounds for i in 1:n
    ai = a[i]
    ptrs[i+1] = innerlength(ai)
  end
  length_to_ptrs!(ptrs)

  ndata = ptrs[end]-u
  data = Matrix{T}(undef,ndata,plength)
  @inbounds for i in 1:n
    ai = a[i]
    for l in 1:plength
      for j in axis(ai.data,1)
        aijl = ai.data[j,l]
        data[(i-1)*n+j,l] = aijl
      end
    end
  end

  ParamJaggedArray(data,ptrs)
end

Base.size(a::ParamJaggedArray) = (length(a.ptrs)-1,)

function Base.getindex(a::ParamJaggedArray,i::Int)
  u = one(eltype(a.ptrs))
  pini = a.ptrs[i]
  pend = a.ptrs[i+1]-u
  view(a.data.data,pini:pend,:)
end

function Base.setindex!(a::ParamJaggedArray,v,i::Int)
  u = one(eltype(a.ptrs))
  pini = a.ptrs[i]
  pend = a.ptrs[i+1]-u
  a.data.data[pini:pend,:] = v
end
