"""
    struct DofMapArray{T,D,Ti,A<:AbstractArray{T},I<:AbstractArray{Ti,D}} <: AbstractArray{T,D}
      array::A
      dof_map::I
    end

Subtype of `AbstractArray` whose entries stored in `array` are indexed according to the
indexing strategy specified in `dof_map`
"""
struct DofMapArray{T,D,Ti,A<:AbstractArray{T},I<:AbstractArray{Ti,D}} <: AbstractArray{T,D}
  array::A
  dof_map::I
end

Base.size(a::DofMapArray) = size(a.dof_map)
Base.IndexStyle(a::DofMapArray) = IndexLinear()

function Base.getindex(a::DofMapArray,i::Integer)
  i′ = a.dof_map[i]
  i′ == 0 ? zero(eltype(a)) : a.array[i′]
end

function Base.setindex!(a::DofMapArray,v,i::Integer)
  i′ = a.dof_map[i]
  i′ != 0 && setindex!(a.array,v,i′)
end
