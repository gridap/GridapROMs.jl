"""
    struct RBParamVector{T,A<:ParamVector{T},B} <: ParamArray{T,1}
      data::A
      fe_data::B
    end

Parametric vector obtained by applying a [`Projection`](@ref) on a high-dimensional
parametric FE vector `fe_data`, which is stored (but mostly unused) for conveniency
"""
struct RBParamVector{T,A<:ParamVector{T},B} <: ParamArray{T,1}
  data::A
  fe_data::B
end

Base.size(a::RBParamVector) = size(a.data)
Base.axes(a::RBParamVector) = axes(a.data)
Base.getindex(a::RBParamVector,i::Integer) = getindex(a.data,i)
Base.setindex!(a::RBParamVector,v,i::Integer) = setindex!(a.data,v,i)
ParamDataStructures.param_length(a::RBParamVector) = param_length(a.data)
ParamDataStructures.get_all_data(a::RBParamVector) = get_all_data(a.data)
ParamDataStructures.param_getindex(a::RBParamVector,i::Integer) = param_getindex(a.data,i)

function ParamDataStructures.param_cat(a::Vector{<:RBParamVector})
  data = param_cat(map(_data,a))
  fe_data = param_cat(map(_fe_data,a))
  RBParamVector(data,fe_data)
end

function Base.copy(a::RBParamVector)
  data′ = copy(a.data)
  fe_data′ = copy(a.fe_data)
  RBParamVector(data′,fe_data′)
end

function Base.similar(a::RBParamVector{R},::Type{S}) where {R,S<:AbstractVector}
  data′ = similar(a.data,S)
  fe_data′ = copy(a.fe_data)
  RBParamVector(data′,fe_data′)
end

function Base.similar(a::RBParamVector{R},::Type{S},dims::Dims{1}) where {R,S<:AbstractVector}
  data′ = similar(a.data,S,dims)
  fe_data′ = similar(a.fe_data,S,dims)
  RBParamVector(data′,fe_data′)
end

function Base.copyto!(a::RBParamVector,b::RBParamVector)
  copyto!(a.data,b.data)
  copyto!(a.fe_data,b.fe_data)
  a
end

function Base.fill!(a::RBParamVector,b::Number)
  fill!(a.data,b)
  fill!(a.fe_data,b)
  return a
end

function LinearAlgebra.rmul!(a::RBParamVector,b::Number)
  rmul!(a.data,b)
  rmul!(a.fe_data,b)
  return a
end

function LinearAlgebra.axpy!(α::Number,a::RBParamVector,b::RBParamVector)
  axpy!(α,a.data,b.data)
  axpy!(α,a.fe_data,b.fe_data)
  return b
end

for T in (:SingleFieldFESpace,:MultiFieldFESpace)
  @eval begin
    function FESpaces.FEFunction(f::$T,fv::RBParamVector)
      FEFunction(f,fv.fe_data)
    end

    function FESpaces.EvaluationFunction(f::$T,fv::RBParamVector)
      EvaluationFunction(f,fv.fe_data)
    end
  end
end

function unfold(a::BlockParamVector{T,<:AbstractVector{<:RBParamVector{T}}}) where T
  data = mortar(map(_data,blocks(a)))
  fe_data = mortar(map(_fe_data,blocks(a)))
  RBParamVector(data,fe_data)
end

# utils

_data(a::RBParamVector) = a.data
_fe_data(a::RBParamVector) = a.fe_data
