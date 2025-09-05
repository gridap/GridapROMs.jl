"""
    eltype2(a) -> Type

Returns the eltype of `eltype(a)`, i.e. it extracts the eltype of a parametric
entry of `a`
"""
eltype2(x) = eltype(eltype(x))

"""
    param_length(a) -> Int

Returns the parametric length of `a`
"""
param_length(a) = @abstractmethod
param_length(a::Union{Nothing,Function,Map,Field,AbstractArray{<:Field}}) = 0
param_length(a::Union{AbstractRealization,AbstractParamFunction}) = length(a)
param_length(a::Union{Number,AbstractArray{<:Number}}) = 0
param_length(a::CellField) = param_length(testitem(get_data(a)))

"""
    get_param_data(a) -> Any

Returns the parametric data of `a`, usually in the form of a `AbstractVector` or
a NTuple
"""
get_param_data(a) = @abstractmethod

"""
    param_getindex(a,i::Integer) -> Any

Returns the parametric entry of `a` at the index `i` ∈ {1,...,`param_length(a)`}
"""
param_getindex(a,i::Integer) = @abstractmethod

"""
    param_setindex!(a,v,i::Integer) -> Any

Sets the parametric entry of `a` to `v` at the index `i` ∈ {1,...,`param_length(a)`}
"""
param_setindex!(a,v,i::Integer) = @abstractmethod

"""
    param_eachindex(a,i::Integer) -> Any

Returns the parametric range of `a` 1:`param_length(a)`
"""
param_eachindex(a) = Base.OneTo(param_length(a))

"""
    parameterize(a,plength::Integer) -> Any

Returns a quantity with parametric length `plength` from `a`. When `a` already
possesses a parametric length, i.e. it is a parametrized quantity, it returns `a`
"""
parameterize(a,plength::Integer) = local_parameterize(a,plength)

function parameterize(a...;plength=find_param_length(a...))
  pa = map(f->parameterize(f,plength),a)
  return pa
end

function parameterize(a::AbstractParamFunction,plength::Integer)
  @check param_length(a) == plength
  return a
end

"""
    find_param_length(a...) -> Int

Returns the parametric length of all parametric quantities. An error is thrown
if there are no parametric quantities or if at least two quantities have different
parametric length
"""
function find_param_length(a...)
  plengths::Tuple{Vararg{Int}} = filter(!iszero,param_length.(a))
  @check all(plengths .== first(plengths))
  return first(plengths)
end

"""
    abstract type AbstractParamData{T,N} <: AbstractArray{T,N} end

Type representing generic parametric quantities.
Subtypes:
- [`ParamNumber`](@ref)
- [`AbstractParamArray`](@ref)
- [`AbstractSnapshots`](@ref)
"""
abstract type AbstractParamData{T,N} <: AbstractArray{T,N} end

get_param_data(a::AbstractParamData) = (param_getindex(a,i) for i in param_eachindex(a))

function parameterize(a::AbstractParamData,plength::Integer)
  @check param_length(a) == plength
  return a
end

"""
    struct ParamNumber{T<:Number,A<:AbstractVector{T}} <: AbstractParamData{T,1}
      data::A
    end

Used as a wrapper for non-array structures, e.g. factorizations or numbers
"""
struct ParamNumber{T<:Number,A<:AbstractVector{T}} <: AbstractParamData{T,1}
  data::A
end

param_length(a::ParamNumber) = length(a.data)
param_getindex(a::ParamNumber,i::Integer) = getindex(a,i)
param_setindex!(a::ParamNumber,v,i::Integer) = setindex!(a,v,i)

Base.size(a::ParamNumber) = (param_length(a),)
Base.getindex(a::ParamNumber,i::Integer) = getindex(a.data,i)
Base.setindex!(a::ParamNumber,v,i::Integer) = setindex!(a.data,v,i)

for op in (:+,:-)
  @eval begin
    function ($op)(A::ParamNumber,b::Number)
      ParamNumber(Base.broadcast($op,A,b))
    end

    function ($op)(a::Number,B::ParamNumber)
      ParamNumber(Base.broadcast($op,a,B))
    end
  end
end
