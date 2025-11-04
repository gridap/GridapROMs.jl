abstract type AbstractParamBroadcast end

Base.size(A::AbstractParamBroadcast) = (param_length(A),)
Base.axes(A::AbstractParamBroadcast) = (Base.OneTo(param_length(A)),)

struct ParamBroadcast{D} <: AbstractParamBroadcast
  data::D
  plength::Int
end

ParamBroadcast(A::AbstractParamArray) = ParamBroadcast(A,param_length(A))

get_all_data(A::ParamBroadcast) = A.data
param_length(A::ParamBroadcast) = A.plength

function Base.broadcasted(f,A::Union{AbstractParamArray,ParamBroadcast}...)
  bc = Base.broadcasted(f,map(get_all_data,A)...)
  plength = find_param_length(A...)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,A::ParamBroadcast,b::Number)
  bc = Base.broadcasted(f,get_all_data(A),b)
  plength = param_length(A)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,a::Number,B::ParamBroadcast)
  bc = Base.broadcasted(f,a,get_all_data(B))
  plength = param_length(B)
  ParamBroadcast(bc,plength)
end

function Base.broadcasted(f,A::AbstractParamArray,b::Number)
  Base.broadcasted(f,A,fill(b,param_length(A)))
end

function Base.broadcasted(f,a::Number,B::AbstractParamArray)
  Base.broadcasted(f,fill(a,param_length(B)),B)
end

function Base.broadcasted(f,A::AbstractParamArray,b::AbstractVector{<:Number})
  @check length(b) == param_length(A)
  C = similar(A)
  for i in param_eachindex(A)
    ai = param_getindex(A,i)
    ci = Base.materialize(Base.broadcasted(f,ai,b[i]))
    param_setindex!(C,ci,i)
  end
  return ParamBroadcast(C)
end

function Base.broadcasted(f,a::AbstractVector{<:Number},B::AbstractParamArray)
  @check length(a) == param_length(B)
  C = similar(B)
  for i in param_eachindex(B)
    bi = param_getindex(B,i)
    ci = Base.materialize(Base.broadcasted(f,a[i],bi))
    param_setindex!(C,ci,i)
  end
  return ParamBroadcast(C)
end

function Base.materialize(B::ParamBroadcast)
  to_param_array(Base.materialize(get_all_data(B)))
end

to_param_array(A::AbstractArray{<:Number}) = ConsecutiveParamArray(A)
to_param_array(A::AbstractArray{<:AbstractArray}) = ParamArray(A)

function Base.materialize!(A::AbstractParamArray,B::ParamBroadcast)
  Base.materialize!(get_all_data(A),get_all_data(B))
  A
end

struct BlockParamBroadcast{A<:AbstractParamBroadcast,N} <: AbstractArray{A,N}
  data::Array{A,N}
end

Base.size(A::BlockParamBroadcast) = size(A.data)
Base.getindex(A::BlockParamBroadcast{B,N},i::Vararg{Integer,N}) where {B,N} = getindex(A.data,i...)

param_length(A::BlockParamBroadcast) = testitem(A.data)
BlockArrays.blocks(A::BlockParamBroadcast) = A.data

function Base.broadcasted(f,A::Union{BlockParamArray,BlockParamBroadcast}...)
  bc = map((x...)->Base.broadcasted(f,x...),map(blocks,A)...)
  BlockParamBroadcast(bc)
end

function Base.broadcasted(f,A::Union{BlockParamArray,BlockParamBroadcast},b::Number)
  bc = map(a->Base.broadcasted(f,a,b),blocks(A))
  BlockParamBroadcast(bc)
end

function Base.broadcasted(f,a::Number,B::Union{BlockParamArray,BlockParamBroadcast})
  bc = map(b->Base.broadcasted(f,a,b),blocks(B))
  BlockParamBroadcast(bc)
end

function Base.broadcasted(f,
  A::Union{BlockParamArray,BlockParamBroadcast},
  b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}})
  Base.broadcasted(f,A,Base.materialize(b))
end

function Base.broadcasted(
  f,
  a::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{0}},
  B::Union{BlockParamArray,BlockParamBroadcast})
  Base.broadcasted(f,Base.materialize(a),B)
end

function Base.materialize(B::BlockParamBroadcast)
  mortar(map(Base.materialize,blocks(B)))
end

function Base.materialize!(A::BlockParamArray,b::Broadcast.Broadcasted)
  map(a -> Base.materialize!(a,b),blocks(A))
  A
end

function Base.materialize!(A::BlockParamArray,B::BlockParamBroadcast)
  map(Base.materialize!,blocks(A),blocks(B))
  A
end
