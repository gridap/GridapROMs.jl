"""
    abstract type AbstractSnapshots{T,N} <: AbstractParamData{T,N} end

Type representing N-dimensional arrays of snapshots. Subtypes must contain the
following information:

- data: a (parametric) array
- realisation: a subtype of [`AbstractRealisation`](@ref), representing the points
  in the parameter space used to compute the array `data`
- dof map: a subtype of [`AbstractDofMap`](@ref), representing a reindexing strategy
  for the array `data`

Subtypes:

- [`Snapshots`](@ref)
- [`BlockSnapshots`](@ref)
"""
abstract type AbstractSnapshots{T,N} <: AbstractParamData{T,N} end

"""
    get_realisation(s::AbstractSnapshots) -> AbstractRealisation

Returns the realisations associated to the snapshots `s`
"""
get_realisation(s::AbstractSnapshots) = @abstractmethod

get_dof_map(s::AbstractSnapshots) = @abstractmethod

num_params(s::AbstractSnapshots) = num_params(get_realisation(s))
param_length(s::AbstractSnapshots) = param_length(get_realisation(s))

"""
    abstract type Snapshots{T,N,I,R} <: AbstractSnapshots{T,N} end

Type representing a collection of parametric abstract arrays of eltype `T`,
that are associated with a realisation of type `R`. Unlike `AbstractParamArray`,
which are arrays of arrays, subtypes of `Snapshots` are arrays of numbers.
"""
abstract type Snapshots{T,N,I,R} <: AbstractSnapshots{T,N} end

function Snapshots(s::AbstractParamArray,r::AbstractRealisation)
  i = VectorDofMap(innerlength(s))
  Snapshots(s,i,r)
end

get_all_data(s::Snapshots) = @abstractmethod

get_param_data(s::Snapshots) = @abstractmethod

"""
    num_space_dofs(s::Snapshots) -> Integer

Total number of spatial DOFs of `s`, i.e. `prod(space_dofs(s))`
"""
num_space_dofs(s::Snapshots) = prod(space_dofs(s))

"""
    space_dofs(s::Snapshots{T,N}) where {T,N} -> NTuple{N-1,Integer}

Returns the spatial size of the snapshots
"""
space_dofs(s::Snapshots{T,N}) where {T,N} = size(get_all_data(s))[1:N-1]

Base.size(s::Snapshots) = (space_dofs(s)...,param_length(s))

function param_getindex(s::Snapshots{T,N},pindex::Integer) where {T,N}
  view(get_all_data(s),_ncolons(Val{N-1}())...,pindex)
end

function Base.reshape(s::Snapshots,dims::Dims)
  reshape(get_all_data(s),dims...)
end

"""
    select_snapshots(s::AbstractSnapshots,pindex) -> AbstractSnapshots

Restriction of `s` to the parameters of index `pindex`
"""
select_snapshots(s::Snapshots,pindex) = @abstractmethod

function flatten(s::Snapshots)
  d = get_all_data(s)
  reshape(d,:,num_params(s))
end

function param_cat(v::AbstractVector{<:Snapshots})
  data = param_cat(map(get_param_data,v))
  i = get_dof_map(first(v))
  r = param_cat(map(get_realisation,v))
  Snapshots(data,i,r)
end

"""
    struct GenericSnapshots{T,N,I,R,A,B} <: Snapshots{T,N,I,R}
      data::A
      param_data::B
      dof_map::I
      realisation::R
    end

Most standard implementation of a [`Snapshots`](@ref)
"""
struct GenericSnapshots{T,N,I,R,A,B} <: Snapshots{T,N,I,R}
  data::A
  param_data::B
  dof_map::I
  realisation::R

  function GenericSnapshots(
    data::A,
    param_data::B,
    dof_map::I,
    realisation::R
    ) where {T,N,R,A<:AbstractArray{T,N},B,I}

    new{T,N,I,R,A,B}(data,param_data,dof_map,realisation)
  end
end

function Snapshots(s::AbstractParamArray,i::TrivialDofMap,r::AbstractRealisation)
  GenericSnapshots(get_all_data(s),s,i,r)
end

function Snapshots(s::AbstractParamArray,i::AbstractDofMap,r::AbstractRealisation)
  data = get_all_data(s)
  plength = param_length(s)
  param_data = s
  if _is_one_to(i)
    dims = (size(i)...,plength)
    idata = reshape(data,dims)
    return GenericSnapshots(idata,param_data,i,r)
  end
  T = eltype2(s)
  idata = zeros(T,size(i)...,plength)
  for ip in 1:plength
    for k in CartesianIndices(i)
      k′ = i[k]
      if k′ > 0
        idata[k.I...,ip] = data[k′,ip]
      end
    end
  end
  GenericSnapshots(idata,param_data,i,r)
end

get_all_data(s::GenericSnapshots) = s.data
get_param_data(s::GenericSnapshots) = s.param_data
get_dof_map(s::GenericSnapshots) = s.dof_map
get_realisation(s::GenericSnapshots) = s.realisation

function select_snapshots(s::GenericSnapshots,pindex)
  prange = _format_index(pindex)
  GenericSnapshots(
    select_all_data(s.data,prange),
    select_param_data(s.param_data,prange),
    get_dof_map(s),
    get_realisation(s)[prange]
  )
end

function Base.getindex(s::GenericSnapshots{T,N},i::Vararg{Integer,N}) where {T,N}
  s.data[i...]
end

function Base.setindex!(s::GenericSnapshots{T,N},v,i::Vararg{Integer,N}) where {T,N}
  s.data[i...] = v
end

# sparse interface

"""
    const SparseSnapshots{T,N,I<:AbstractSparseDofMap,R} = Snapshots{T,N,I,R}

Snapshots of sparse matrices (e.g. Jacobians), storing their nonzero entries
"""
const SparseSnapshots{T,N,I<:AbstractSparseDofMap,R} = Snapshots{T,N,I,R}

function recast(a::AbstractArray,s::SparseSnapshots)
  return recast(a,get_dof_map(s))
end

function recast(s::SparseSnapshots)
  return get_param_data(s)
end

# multi field interface

"""
    struct BlockSnapshots{S<:Snapshots,N} <: AbstractSnapshots{S,N}
      array::Array{S,N}
    end

Block container for Snapshots in a `MultiField` setting. Its param data is
not stored separately; it is always derived from `blocks(s)` (mortared
together), so that it stays consistent with the blocks under any operation
(selecting, clustering, concatenating, ...) instead of needing to be kept in
sync by hand alongside them.
"""
struct BlockSnapshots{S<:Snapshots,N} <: AbstractSnapshots{S,N}
  array::Array{S,N}
end

function Snapshots(
  data::BlockParamArray,
  i::AbstractArray{<:AbstractDofMap},
  r::AbstractRealisation
  )

  block_values = blocks(data)
  array = map(enumerate(block_values)) do (j,dataj)
    Snapshots(dataj,i[j],r)
  end
  BlockSnapshots(array)
end

function Snapshots(
  data::AbstractParamArray,
  i::AbstractArray{<:AbstractDofMap},
  r::AbstractRealisation
  )

  ids = offset_indices(i)
  array = map(enumerate(i)) do (j,ij)
    dataj = get_param_entry(data,ids[j]...)
    Snapshots(dataj,ij,r)
  end
  BlockSnapshots(array)
end

BlockArrays.blocks(s::BlockSnapshots) = s.array
get_param_data(s::BlockSnapshots) = map(get_param_data,blocks(s)) |> mortar

Base.size(s::BlockSnapshots) = size(blocks(s))
Base.getindex(s::BlockSnapshots,i...) = getindex(blocks(s),i...)
Base.setindex!(s::BlockSnapshots,v,i...) = setindex!(blocks(s),v,i...)
Arrays.testitem(s::BlockSnapshots) = first(blocks(s))

get_dof_map(s::BlockSnapshots) = map(get_dof_map,blocks(s))
get_realisation(s::BlockSnapshots) = get_realisation(testitem(s))

function select_snapshots(s::BlockSnapshots{<:Any,N},pindex) where N
  array = map(sj -> select_snapshots(sj,pindex),blocks(s))
  return BlockSnapshots(array)
end

function param_cat(v::AbstractVector{<:BlockSnapshots{<:Any,N}}) where N
  s = first(v)
  @check all(size(si)==size(s) for si in v)
  array = map(CartesianIndices(blocks(s))) do i
    param_cat(map(sv -> getindex(sv,i),v))
  end
  return BlockSnapshots(collect(array))
end

# utils

function Snapshots(a::ArrayContribution,i::ArrayContribution,r::AbstractRealisation)
  contribution(a.trians) do trian
    Snapshots(a[trian],i[trian],r)
  end
end

function select_snapshots(a::ArrayContribution,pindex)
  contribution(a.trians) do trian
    select_snapshots(a[trian],pindex)
  end
end

function select_snapshots(a::Bundle,pindex)
  Bundle(select_snapshots(a.linear,pindex),select_snapshots(a.nonlinear,pindex))
end

"""
    select_all_data(data::AbstractArray,prange) -> AbstractArray

Restriction of the snapshot data `data` to the parameters of index `prange`
"""
function select_all_data(data::AbstractArray{T,N},prange) where {T,N}
  view(data,_ncolons(Val{N-1}())...,prange)
end

"""
    select_param_data(data::AbstractParamArray,prange) -> AbstractParamArray

Restriction of the parameterised snapshot data `data` to the parameters of index `prange`
"""
function select_param_data(pdata::ConsecutiveParamArray{T,N},prange) where {T,N}
  ConsecutiveParamArray(view(pdata.data,_ncolons(Val{N}())...,prange))
end

# in practice, when dealing with the Jacobian, the param data is never fetched
function select_param_data(pdata::ConsecutiveParamSparseMatrixCSC,prange)
  datarange = view(pdata.data,:,prange)
  ConsecutiveParamSparseMatrixCSC(pdata.m,pdata.n,pdata.colptr,pdata.rowval,datarange)
end

function select_param_data(pdata::BlockParamArray,prange) 
  mortar(map(p -> select_param_data(p,prange),blocks(pdata)))
end

function offset_indices(i::AbstractArray{<:AbstractDofMap,N}) where N
  array = Array{Any,N}(undef,size(i))
  offset = 0
  for j in eachindex(i)
    n = length(i[j])
    array[j] = (offset+1:offset+n,)
    offset += n
  end
  return array
end

# linear algebra

for op in (:+,:-)
  @eval function Base.$op(a::Snapshots,b::Snapshots)
    $op(get_all_data(a),get_all_data(b))
  end
end

const SnapshotsMatrix{T} = Snapshots{T,2}

fetch_data(A::AbstractMatrix) = A
fetch_data(A::SnapshotsMatrix) = get_all_data(A)
fetch_data(A::Adjoint{<:Any,<:SnapshotsMatrix}) = adjoint(get_all_data(A.parent))

for S in (:SnapshotsMatrix,:AbstractMatrix), T in (:SnapshotsMatrix,:AbstractMatrix)
  S == :AbstractMatrix && T == :AbstractMatrix && continue
  for AS in (:(Adjoint{<:Any,<:$S}),S), AT in (:(Adjoint{<:Any,<:$T}),T)
    @eval begin
      Base.:*(A::$AS,B::$AT) = *(fetch_data(A),fetch_data(B))

      function LinearAlgebra.mul!(C::AbstractMatrix,A::$AS,B::$AT)
        mul!(C,fetch_data(A),fetch_data(B))
      end
    end
  end
end

# utils

function unfold(A::AbstractArray,dims::Dims)
  @check length(A) == prod(dims)
  reshape(A,dims)
end

@inline function _is_one_to(a::AbstractArray{<:Number})
  @inbounds for i in eachindex(a)
    if a[i] != i
      return false
    end
  end
  return true
end 

_format_index(i) = i
_format_index(i::Number) = i:i
