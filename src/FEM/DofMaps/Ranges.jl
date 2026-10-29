abstract type RangeOrder end
struct Ordered <: RangeOrder end
struct Shuffled <: RangeOrder end

"""
    struct Range2D{I<:AbstractVector,J<:AbstractVector} <: AbstractMatrix{Int}
      axis1::I
      axis2::J
      scale::Int
    end

Creates the indices of a Kronecker product matrix from the indices of the two factors.
The field `axis1` refers to the first factor, field `axis2` refers to the second
factor. The field `scale` is by default equal to the number of entries of the
first factor.
"""
struct Range2D{O<:RangeOrder,I<:AbstractVector,J<:AbstractVector} <: AbstractMatrix{Int}
  axis1::I
  axis2::J
  scale::Int
  function Range2D{O}(axis1::I,axis2::J,scale::Int) where {O<:RangeOrder,I<:AbstractVector,J<:AbstractVector}
    new{O,I,J}(axis1,axis2,scale)    
  end
end

Base.size(r::Range2D{Ordered}) = (length(r.axis1),length(r.axis2))
Base.size(r::Range2D{Shuffled}) = (length(r.axis2),length(r.axis1))
Base.getindex(r::Range2D{Ordered},i::Integer,j::Integer) = r.axis1[i] + (r.axis2[j]-1)*r.scale
Base.getindex(r::Range2D{Shuffled},i::Integer,j::Integer) = r.axis1[j] + (r.axis2[i]-1)*r.scale
Base.copy(r::Range2D{O}) where O = Range2D{O}(copy(r.axis1),copy(r.axis2),r.scale)

"""
    range_2d(i::AbstractVector,j::AbstractVector,scale=length(i)) -> Range2D

Constructor of a [`Range2D`](@ref) object
"""
range_2d(i::AbstractVector,j::AbstractVector,scale=length(i)) = Range2D{Ordered}(i,j,scale)
"""
    shuffled_range_2d(i::AbstractVector,j::AbstractVector,scale=length(i)) -> Range2D

Same as [`range_2d`](@ref), with the order of the two axes swapped
"""
shuffled_range_2d(i::AbstractVector,j::AbstractVector,scale=length(i)) = Range2D{Shuffled}(i,j,scale)

"""
    struct Range1D{O<:RangeOrder,I<:AbstractVector,J<:AbstractVector} <: AbstractVector{Int}
      range::Range2D{O,I,J}
    end

Represents a [`Range2D`](@ref) reshaped as a vector
"""
struct Range1D{O<:RangeOrder,I<:AbstractVector,J<:AbstractVector} <: AbstractVector{Int}
  parent::Range2D{O,I,J}
end

"""
    range_1d(i::AbstractVector,j::AbstractVector,args...) -> AbstractVector{Int}

Vectorization operation of a [`Range2D`](@ref) object
"""
range_1d(i::AbstractVector,j::AbstractVector,args...) = Range1D(range_2d(i,j,args...))
"""
    shuffled_range_1d(i::AbstractVector,j::AbstractVector,args...) -> AbstractVector{Int}

Vectorization of a [`shuffled_range_2d`](@ref)
"""
shuffled_range_1d(i::AbstractVector,j::AbstractVector,args...) = Range1D(shuffled_range_2d(i,j,args...))

Base.size(r::Range1D) = (length(r.parent),)
Base.getindex(r::Range1D,k::Integer) = r.parent[k]
Base.copy(r::Range1D) = Range1D(copy(r.parent))