"""
    struct OwnAndGhostParamVectors{A,B,T} <: AbstractParamVector{T}
      own_values::A
      ghost_values::A
      permutation::B
    end

Parametric version of [`OwnAndGhostVectors`](@ref)
"""
struct OwnAndGhostParamVectors{A,B,C,T} <: AbstractParamArray{T,1,C}
  own_values::A
  ghost_values::A
  permutation::B
  function OwnAndGhostParamVectors{A,B}(_own_values,_ghost_values,_perm) where {A,B}
    @check param_length(_own_values) == param_length(_ghost_values)
    own_values = convert(A,_own_values)
    ghost_values = convert(A,_ghost_values)
    perm = convert(B,_perm)
    oitem = testitem(own_values)
    gitem = testitem(ghost_values)
    item = OwnAndGhostVectors(oitem,gitem,perm)
    T = eltype2(A)
    C = typeof(item)
    new{A,B,C,T}(own_values,ghost_values,perm)
  end
end

function PartitionedArrays.OwnAndGhostVectors{A,B}(
  own_values,
  ghost_values,
  perm
  ) where {A<:AbstractParamArray,B}

  OwnAndGhostParamVectors{A,B}(own_values,ghost_values,perm)
end

ParamDataStructures.param_length(a::OwnAndGhostParamVectors) = param_length(a.own_values)

function ParamDataStructures.param_getindex(a::OwnAndGhostParamVectors,i::Integer)
  getindex(a,i)
end

function ParamDataStructures.parameterize(a::OwnAndGhostVectors,plength::Integer)
  own_values = parameterize(a.own_values,plength)
  ghost_values = parameterize(a.ghost_values,plength)
  OwnAndGhostVectors(own_values,ghost_values,a.permutation)
end

ParamDataStructures.innersize(a::OwnAndGhostParamVectors) = (size(a.own_values.data,1)+size(a.ghost_values.data,1),)

Base.size(a::OwnAndGhostParamVectors) = (param_length(a),)

function Base.getindex(a::OwnAndGhostParamVectors,i::Int)
  oi = param_getindex(a.own_values,i)
  gi = param_getindex(a.ghost_values,i)
  OwnAndGhostVectors(oi,gi,a.permutation)
end

function Base.setindex!(a::OwnAndGhostParamVectors,v,i::Int)
  n_own = innerlength(a.own_values)
  for (j,vj) in enumerate(v)
    pj = a.permutation[j]
    if pj > n_own
      a.ghost_values.data[pj-n_own,i] = vj
    else
      a.own_values.data[pj,i] = vj
    end
  end
  v
end

function PartitionedArrays.own_values(values::OwnAndGhostParamVectors,indices)
  values.own_values
end

function PartitionedArrays.ghost_values(values::OwnAndGhostParamVectors,indices)
  values.ghost_values
end

function PartitionedArrays.allocate_local_values(values::OwnAndGhostParamVectors,::Type{T},indices) where T
  n_own = own_length(indices)
  n_ghost = ghost_length(indices)
  own_values = similar(values.own_values,T,n_own)
  ghost_values = similar(values.ghost_values,T,n_ghost)
  perm = PartitionedArrays.local_permutation(indices)
  OwnAndGhostVectors(own_values,ghost_values,perm)
end

function GridapDistributed.change_ghost(::Type{<:OwnAndGhostParamVectors},a::PVector,ids::PRange)
  values = map(own_values(a),partition(ids)) do own_vals,ids
    ghost_vals = fill(zero(eltype(a)),ghost_length(ids))
    perm = PartitionedArrays.local_permutation(ids)
    OwnAndGhostVectors(own_vals,ghost_vals,perm)
  end
  return PVector(values,partition(ids))
end

function Base.fill!(a::OwnAndGhostParamVectors,v::Number)
  n_own = innerlength(a.own_values)
  for j in param_eachindex(a)
    for i in 1:innerlength(a)
      k = a.permutation[i]
      if k > n_own
        a.ghost_values.data[k-n_own,j] = v
      else
        a.own_values.data[k,j] = v
      end
    end
  end
end
