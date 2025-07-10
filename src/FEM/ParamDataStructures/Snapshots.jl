"""
    abstract type AbstractSnapshots{T,N} <: AbstractParamData{T,N} end

Type representing N-dimensional arrays of snapshots. Subtypes must contain the
following information:

- data: a (parametric) array
- realization: a subtype of [`AbstractRealization`](@ref), representing the points
  in the parameter space used to compute the array `data`
- dof map: a subtype of [`AbstractDofMap`](@ref), representing a reindexing strategy
  for the array `data`

Subtypes:

- [`Snapshots`](@ref)
- [`BlockSnapshots`](@ref)
"""
abstract type AbstractSnapshots{T,N} <: AbstractParamData{T,N} end

"""
    get_realization(s::AbstractSnapshots) -> AbstractRealization

Returns the realizations associated to the snapshots `s`
"""
get_realization(s::AbstractSnapshots) = @abstractmethod

DofMaps.get_dof_map(s::AbstractSnapshots) = @abstractmethod

num_params(s::AbstractSnapshots) = num_params(get_realization(s))

"""
    abstract type Snapshots{T,N,I,R,A} <: AbstractSnapshots{T,N} end

Type representing a collection of parametric abstract arrays of eltype `T`,
that are associated with a realization of type `R`. Unlike `AbstractParamArray`,
which are arrays of arrays, subtypes of `Snapshots` are arrays of numbers.

Subtypes:

- [`SteadySnapshots`](@ref)
- [`TransientSnapshots`](@ref)
"""
abstract type Snapshots{T,N,I,R,A} <: AbstractSnapshots{T,N} end

function Snapshots(s::AbstractArray,i::AbstractDofMap,r::AbstractRealization)
  @abstractmethod
end

get_all_data(s::Snapshots) = @abstractmethod

get_param_data(s::Snapshots) = ConsecutiveParamArray(get_all_data(s))

num_space_dofs(s::Snapshots) = length(get_dof_map(s))

function Base.reshape(s::Snapshots,dims::Dims)
  reshape(get_all_data(s),dims...)
end

function select_snapshots(s::Snapshots,pindex)
  if num_params(s)==length(pindex)
    s
  else
    _select_snapshots(s,pindex)
  end
end

"""
    const SteadySnapshots{T,N,I,R<:Realization,A} = Snapshots{T,N,I,R,A}
"""
const SteadySnapshots{T,N,I,R<:Realization,A} = Snapshots{T,N,I,R,A}

"""
    space_dofs(s::SteadySnapshots{T,N}) where {T,N} -> NTuple{N-1,Integer}
    space_dofs(s::TransientSnapshots{T,N}) where {T,N} -> NTuple{N-2,Integer}

Returns the spatial size of the snapshots
"""
space_dofs(s::SteadySnapshots{T,N}) where {T,N} = size(get_all_data(s))[1:N-1]

Base.size(s::SteadySnapshots) = (space_dofs(s)...,num_params(s))

function Snapshots(s::AbstractParamVector,i::TrivialDofMap,r::Realization)
  Snapshots(get_all_data(s),i,r)
end

function Snapshots(s::ParamSparseMatrix,i::TrivialDofMap,r::Realization)
  Snapshots(get_all_data(s),i,r)
end

function _select_snapshots(s::SteadySnapshots,pindex)
  prange = _format_index(pindex)
  drange = view(get_all_data(s),:,prange)
  rrange = get_realization(s)[prange]
  Snapshots(drange,get_dof_map(s),rrange)
end

function param_getindex(s::SteadySnapshots{T,N},pindex::Integer) where {T,N}
  view(get_all_data(s),_ncolons(Val{N-1}())...,pindex)
end

_format_index(i) = i
_format_index(i::Number) = i:i

"""
    struct GenericSnapshots{T,N,I,R,A} <: Snapshots{T,N,I,R,A}
      data::A
      dof_map::I
      realization::R
    end

Most standard implementation of a [`Snapshots`](@ref)
"""
struct GenericSnapshots{T,N,I,R,A} <: Snapshots{T,N,I,R,A}
  data::A
  dof_map::I
  realization::R

  function GenericSnapshots(
    data::A,
    dof_map::I,
    realization::R
    ) where {T,N,R,A<:AbstractArray{T,N},I<:AbstractDofMap}

    new{T,N,I,R,A}(data,dof_map,realization)
  end
end

function Snapshots(s::AbstractArray{<:Number},i::TrivialDofMap,r::AbstractRealization)
  GenericSnapshots(s,i,r)
end

get_all_data(s::GenericSnapshots) = s.data
DofMaps.get_dof_map(s::GenericSnapshots) = s.dof_map
get_realization(s::GenericSnapshots) = s.realization

function Base.getindex(s::GenericSnapshots{T,N},i::Vararg{Integer,N}) where {T,N}
  s.data[i...]
end

function Base.setindex!(s::GenericSnapshots{T,N},v,i::Vararg{Integer,N}) where {T,N}
  s.data[i...] = v
end

"""
    struct ReshapedSnapshots{T,N,I,R,A,B} <: Snapshots{T,N,I,R,A}
      data::A
      param_data::B
      dof_map::I
      realization::R
    end

Most standard implementation of a [`Snapshots`](@ref)
"""
struct ReshapedSnapshots{T,N,I,R,A,B} <: Snapshots{T,N,I,R,A}
  data::A
  param_data::B
  dof_map::I
  realization::R

  function ReshapedSnapshots(
    data::A,
    param_data::B,
    dof_map::I,
    realization::R
    ) where {T,N,R,A<:AbstractArray{T,N},B,I<:AbstractDofMap}

    new{T,N,I,R,A,B}(data,param_data,dof_map,realization)
  end
end

function Snapshots(s::AbstractParamVector,i::AbstractDofMap,r::Realization)
  data = get_all_data(s)
  param_data = s
  dims = (size(i)...,num_params(r))
  idata = reshape(data,dims)
  ReshapedSnapshots(idata,param_data,i,r)
end

function Snapshots(s::ParamSparseMatrix,i::SparseMatrixDofMap,r::Realization)
  T = eltype2(s)
  data = get_all_data(s)
  param_data = s
  idata = zeros(T,size(i)...,num_params(r))
  for ip in 1:num_params(r)
    for k in CartesianIndices(i)
      k′ = i[k]
      if k′ > 0
        idata[k.I...,ip] = data[k′,ip]
      end
    end
  end
  ReshapedSnapshots(idata,param_data,i,r)
end

get_all_data(s::ReshapedSnapshots) = s.data
get_param_data(s::ReshapedSnapshots) = s.param_data
DofMaps.get_dof_map(s::ReshapedSnapshots) = s.dof_map
get_realization(s::ReshapedSnapshots) = s.realization

function _select_snapshots(s::ReshapedSnapshots{T,N},pindex) where {T,N}
  prange = _format_index(pindex)
  drange = view(get_all_data(s),_ncolons(Val{N-1}())...,prange)
  pdrange = _get_param_data(s.param_data,prange)
  rrange = get_realization(s)[prange]
  ReshapedSnapshots(drange,pdrange,get_dof_map(s),rrange)
end

function _get_param_data(pdata::ConsecutiveParamVector,prange)
  ConsecutiveParamArray(view(pdata.data,:,prange))
end

# in practice, when dealing with the Jacobian, the param data is never fetched
function _get_param_data(pdata::ConsecutiveParamSparseMatrixCSC,prange)
  datarange = view(pdata.data,:,prange)
  ConsecutiveParamSparseMatrixCSC(pdata.m,pdata.n,pdata.colptr,pdata.rowval,datarange)
end

function Base.getindex(s::ReshapedSnapshots{T,N},i::Vararg{Integer,N}) where {T,N}
  s.data[i...]
end

function Base.setindex!(s::ReshapedSnapshots{T,N},v,i::Vararg{Integer,N}) where {T,N}
  s.data[i...] = v
end

# sparse interface

"""
"""
const SparseSnapshots{T,N,I<:AbstractSparseDofMap,R,A} = Snapshots{T,N,I,R,A}

function DofMaps.recast(a::AbstractArray,s::SparseSnapshots)
  return recast(a,get_dof_map(s))
end

# multi field interface

"""
    struct BlockSnapshots{S<:Snapshots,N} <: AbstractSnapshots{S,N}
      array::Array{S,N}
      touched::Array{Bool,N}
    end

Block container for Snapshots of type `S` in a `MultiField` setting. This
type is conceived similarly to `ArrayBlock` in Gridap
"""
struct BlockSnapshots{S<:Snapshots,N} <: AbstractSnapshots{S,N}
  array::Array{S,N}
  touched::Array{Bool,N}

  function BlockSnapshots(
    array::Array{S,N},
    touched::Array{Bool,N}
    ) where {S<:Snapshots,N}

    @check size(array) == size(touched)
    new{S,N}(array,touched)
  end
end

function Snapshots(
  data::BlockParamArray{T,N},
  i::AbstractArray{<:AbstractDofMap},
  r::AbstractRealization) where {T,N}

  block_values = blocks(data)
  s = size(block_values)
  @check s == size(i)

  array = Array{Snapshots,N}(undef,s)
  touched = Array{Bool,N}(undef,s)
  for (j,dataj) in enumerate(block_values)
    if !iszero(dataj)
      array[j] = Snapshots(dataj,i[j],r)
      touched[j] = true
    else
      touched[j] = false
    end
  end

  BlockSnapshots(array,touched)
end

BlockArrays.blocks(s::BlockSnapshots) = s.array

Base.size(s::BlockSnapshots) = size(s.array)

function Base.getindex(s::BlockSnapshots,i...)
  if !s.touched[i...]
    return nothing
  end
  s.array[i...]
end

function Base.setindex!(s::BlockSnapshots,v,i...)
  @check s.touched[i...] "Only touched entries can be set"
  s.array[i...] = v
end

function Arrays.testitem(s::BlockSnapshots)
  i = findall(s.touched)
  if length(i) != 0
    s.array[i[1]]
  else
    error("This block snapshots structure is empty")
  end
end

DofMaps.get_dof_map(s::BlockSnapshots) = map(get_dof_map,s.array)
get_realization(s::BlockSnapshots) = get_realization(testitem(s))

function get_param_data(s::BlockSnapshots)
  map(get_param_data,s.array) |> mortar
end

function Arrays.return_cache(::typeof(select_snapshots),s::BlockSnapshots,args...)
  S = Snapshots
  N = ndims(s)
  block_cache = Array{S,N}(undef,size(s))
  return block_cache
end

function select_snapshots(s::BlockSnapshots,pindex)
  array = return_cache(select_snapshots,s,pindex)
  touched = s.touched
  for i in eachindex(touched)
    if touched[i]
      array[i] = select_snapshots(s[i],pindex)
    end
  end
  return BlockSnapshots(array,touched)
end

# utils

function Snapshots(a::ArrayContribution,i::ArrayContribution,r::AbstractRealization)
  contribution(a.trians) do trian
    Snapshots(a[trian],i[trian],r)
  end
end

function select_snapshots(a::ArrayContribution,pindex)
  contribution(a.trians) do trian
    select_snapshots(a[trian],pindex)
  end
end

# linear algebra

function Base.:*(A::Snapshots{T,2},B::Snapshots{S,2}) where {T,S}
  consec_mul(get_all_data(A),get_all_data(B))
end

function Base.:*(A::Snapshots{T,2},B::Adjoint{S,<:Snapshots}) where {T,S}
  consec_mul(get_all_data(A),adjoint(get_all_data(B.parent)))
end

function Base.:*(A::Snapshots{T,2},B::AbstractMatrix{S}) where {T,S}
  consec_mul(get_all_data(A),B)
end

function Base.:*(A::Snapshots{T,2},B::Adjoint{T,<:AbstractMatrix{S}}) where {T,S}
  *(get_all_data(A),B)
end

function Base.:*(A::Adjoint{T,<:Snapshots{T,2}},B::Snapshots{S,2}) where {T,S}
  consec_mul(adjoint(get_all_data(A.parent)),get_all_data(B))
end

function Base.:*(A::AbstractMatrix{T},B::Snapshots{S,2}) where {T,S}
  consec_mul(A,get_all_data(B))
end

function Base.:*(A::Adjoint{T,<:AbstractMatrix},B::Snapshots{S,2}) where {T,S}
  consec_mul(A,get_all_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::Snapshots{S,2}) where {T,S}
  consec_mul!(C,get_all_data(A),get_all_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::Adjoint{S,<:Snapshots}) where {T,S}
  consec_mul!(C,get_all_data(A),adjoint(get_all_data(B.parent)))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::AbstractMatrix{S}) where {T,S}
  consec_mul!(C,get_all_data(A),B)
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Snapshots{T,2},B::Adjoint{T,<:AbstractMatrix{S}}) where {T,S}
  consec_mul!(C,get_all_data(A),B)
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Adjoint{T,<:Snapshots{T,2}},B::Snapshots{S,2}) where {T,S}
  consec_mul!(C,adjoint(get_all_data(A.parent)),get_all_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::AbstractMatrix{T},B::Snapshots{S,2}) where {T,S}
  consec_mul!(C,A,get_all_data(B))
end

function LinearAlgebra.mul!(C::AbstractMatrix,A::Adjoint{T,<:AbstractMatrix},B::Snapshots{S,2}) where {T,S}
  consec_mul!(C,A,get_all_data(B))
end

consec_mul(A::AbstractArray,B::AbstractArray) = A*B
consec_mul!(C::AbstractArray,A::AbstractArray,B::AbstractArray) = mul!(C,A,B)

for T in (:ConsecutiveParamArray,:ConsecutiveParamSparseMatrix)
  @eval begin
    consec_mul(A::$T,B::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}}) where S = get_all_data(A)*B
    consec_mul(A::Adjoint{S,<:$T},B::Union{<:AbstractArray,Adjoint{U,<:AbstractArray}}) where {S,U} = adjoint(get_all_data(A.parent))*B
    consec_mul(A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::$T) where S = A*get_all_data(B)
    consec_mul(A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::Adjoint{U,<:$T}) where {S,U} = A*adjoint(get_all_data(B.parent))
    consec_mul!(C::AbstractArray,A::$T,B::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}}) where S = mul!(C,get_all_data(A),B)
    consec_mul!(C::AbstractArray,A::Adjoint{S,<:$T},B::Union{<:AbstractArray,Adjoint{U,<:AbstractArray}}) where {S,U} = mul!(C,adjoint(get_all_data(A.parent)),B)
    consec_mul!(C::AbstractArray,A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::$T) where S = mul!(C,A,get_all_data(B))
    consec_mul!(C::AbstractArray,A::Union{<:AbstractArray,Adjoint{S,<:AbstractArray}},B::Adjoint{U,<:$T}) where {S,U} = mul!(C,A,adjoint(get_all_data(B.parent)))
  end
end

function _adjoint(A::AbstractMatrix)
  adjoint(A)
end

function _adjoint(A::SubArray{T,2}) where T
  view(adjoint(A.parent),A.indices...)
end

function LinearAlgebra.adjoint(s::GenericSnapshots{T,2}) where T
  _adjoint(get_all_data(s))
end
