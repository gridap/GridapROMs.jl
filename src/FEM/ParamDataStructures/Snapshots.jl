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

"""
    abstract type Snapshots{T,N,I,R} <: AbstractSnapshots{T,N} end

Type representing a collection of parametric abstract arrays of eltype `T`,
that are associated with a realisation of type `R`. Unlike `AbstractParamArray`,
which are arrays of arrays, subtypes of `Snapshots` are arrays of numbers.

Subtypes:

- [`SteadySnapshots`](@ref)
- [`TransientSnapshots`](@ref)
"""
abstract type Snapshots{T,N,I,R} <: AbstractSnapshots{T,N} end

function Snapshots(s::AbstractParamArray,r::AbstractRealisation)
  i = VectorDofMap(innerlength(s))
  Snapshots(s,i,r)
end

function Snapshots(s::AbstractParamArray,i::AbstractDofMap,r::AbstractRealisation)
  @abstractmethod
end

get_all_data(s::Snapshots) = @abstractmethod

get_param_data(s::Snapshots) = ConsecutiveParamArray(get_all_data(s))

num_space_dofs(s::Snapshots) = length(get_dof_map(s))

function Base.reshape(s::Snapshots,dims::Dims)
  reshape(get_all_data(s),dims...)
end

select_snapshots(s::Snapshots,pindex) = @abstractmethod

function change_dof_map(s::Snapshots,i)
  pdata = get_param_data(s)
  r = get_realisation(s)
  Snapshots(pdata,i,r)
end

function param_cat(v::AbstractVector{<:Snapshots})
  data = param_cat(map(get_param_data,v))
  i = get_dof_map(first(v))
  r = param_cat(map(get_realisation,v))
  Snapshots(data,i,r)
end

"""
    const SteadySnapshots{T,N,I,R<:Realisation} = Snapshots{T,N,I,R}
"""
const SteadySnapshots{T,N,I,R<:Realisation} = Snapshots{T,N,I,R}

"""
    space_dofs(s::SteadySnapshots{T,N}) where {T,N} -> NTuple{N-1,Integer}
    space_dofs(s::TransientSnapshots{T,N}) where {T,N} -> NTuple{N-2,Integer}

Returns the spatial size of the snapshots
"""
space_dofs(s::SteadySnapshots{T,N}) where {T,N} = size(get_all_data(s))[1:N-1]

Base.size(s::SteadySnapshots) = (space_dofs(s)...,num_params(s))

function param_getindex(s::SteadySnapshots{T,N},pindex::Integer) where {T,N}
  view(get_all_data(s),_ncolons(Val{N-1}())...,pindex)
end

function select_all_data(s::SteadySnapshots{T,N},prange) where {T,N}
  view(get_all_data(s),_ncolons(Val{N-1}())...,prange)
end

function flatten(s::SteadySnapshots)
  d = get_all_data(s)
  reshape(d,:,num_params(s))
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
    ) where {T,N,R,A<:AbstractArray{T,N},B,I<:AbstractDofMap}

    new{T,N,I,R,A,B}(data,param_data,dof_map,realisation)
  end
end

function Snapshots(s::AbstractParamArray,i::TrivialDofMap,r::Realisation)
  GenericSnapshots(get_all_data(s),s,i,r)
end

function Snapshots(s::AbstractParamArray,i::AbstractDofMap,r::Realisation)
  data = get_all_data(s)
  param_data = s
  if _is_one_to(i)
    dims = (size(i)...,num_params(r))
    idata = reshape(data,dims)
    return GenericSnapshots(idata,param_data,i,r)
  end
  T = eltype2(s)
  idata = zeros(T,size(i)...,num_params(r))
  for ip in 1:num_params(r)
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

function select_snapshots(s::GenericSnapshots{T,N},pindex) where {T,N}
  prange = _format_index(pindex)
  GenericSnapshots(
    select_all_data(s,prange),
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
    struct BlockSnapshots{N,B} <: AbstractSnapshots{Snapshots,N}
      array::Array{<:Any,N}
      touched::Array{Bool,N}
      param_data::B
    end

Block container for Snapshots of type `S` in a `MultiField` setting. This
type is conceived similarly to `ArrayBlock` in Gridap
"""
struct BlockSnapshots{N,B} <: AbstractSnapshots{Snapshots,N}
  array::Array{<:Any,N}
  touched::Array{Bool,N}
  param_data::B
  function BlockSnapshots(
    array::Array{<:Any,N},
    touched::Array{Bool,N},
    param_data::B
    ) where {N,B}

    @check size(array) == size(touched)
    new{N,B}(array,touched,param_data)
  end
end

function Snapshots(
  data::BlockParamArray{T,N},
  i::ArrayBlock{<:AbstractDofMap},
  r::AbstractRealisation
  ) where {T,N}

  block_values = blocks(data)
  s = size(block_values)
  @check s == size(i)

  array = Array{Any,N}(undef,s)
  for (j,dataj) in enumerate(block_values)
    if i.touched[j]
      array[j] = Snapshots(dataj,i[j],r)
    end
  end

  BlockSnapshots(array,i.touched,data)
end

function Snapshots(
  data::AbstractParamArray{T,N},
  i::ArrayBlock{<:AbstractDofMap},
  r::AbstractRealisation
  ) where {T,N}

  s = size(i)
  ids = offset_indices(i)
  array = Array{Any,N}(undef,s)
  for j in eachindex(i)
    if i.touched[j]
      dataj = get_param_entry(data,ids[j]...)
      array[j] = Snapshots(dataj,i[j],r)
    end
  end

  BlockSnapshots(array,i.touched,data)
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
  i = findfirst(s.touched)
  if !isnothing(i)
    @inbounds s.array[i[1]]
  else
    error("This block snapshots structure is empty")
  end
end

function get_dof_map(s::BlockSnapshots{N}) where N
  I = eltype(map(get_dof_map,blocks(s)))
  array = Array{I,N}(undef,size(s))
  for i in eachindex(s.touched)
    if s.touched[i]
      array[i] = get_dof_map(s[i])
    end
  end
  return ArrayBlock(array,s.touched)
end

get_realisation(s::BlockSnapshots) = get_realisation(testitem(s))

get_param_data(s::BlockSnapshots) = s.param_data

function select_snapshots(s::BlockSnapshots{N},pindex) where N
  prange = _format_index(pindex)
  array = Array{Any,N}(undef,size(s))
  for i in eachindex(s.touched)
    if s.touched[i]
      array[i] = select_snapshots(s[i],pindex)
    end
  end
  return BlockSnapshots(array,s.touched,select_param_data(s.param_data,prange))
end

function param_cat(v::AbstractVector{<:BlockSnapshots{N}}) where N
  s = first(v)
  touched = s.touched
  @check all(size(si)==size(s) && si.touched==touched for si in v)
  array = Array{Any,N}(undef,size(s))
  for i in eachindex(touched)
    if touched[i]
      array[i] = param_cat(map(s -> getindex(s,i),v))
    end
  end
  param_data = param_cat(map(s -> s.param_data,v))
  return BlockSnapshots(array,touched,param_data)
end

function change_dof_map(s::BlockSnapshots{N},i::ArrayBlock{<:AbstractDofMap,N}) where N
  @check s.touched == i.touched
  array = Array{Any,N}(undef,size(s))
  for j in eachindex(s.touched)
    if s.touched[j]
      array[j] = change_dof_map(s[j],i[j])
    end
  end
  return BlockSnapshots(array,s.touched,s.param_data)
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

function change_dof_map(a::ArrayContribution,i::ArrayContribution)
  a′ = ()
  for j in eachindex(a)
    a′ = (a′...,change_dof_map(a[j],i[j]))
  end
  return Contribution(a′,a.trians)
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

function offset_indices(i::ArrayBlock{A,N}) where {A,N}
  array = Array{Any,N}(undef,size(i))
  offset = 0
  for j in eachindex(i)
    if i.touched[j]
      n = length(i[j])
      array[j] = (offset+1:offset+n,)
      offset += n
    end
  end
  return array
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

function LinearAlgebra.mul!(
  C::AbstractMatrix,
  A::Snapshots{T,2},
  B::Snapshots{S,2}
  ) where {T,S}
  consec_mul!(C,get_all_data(A),get_all_data(B))
end

function LinearAlgebra.mul!(
  C::AbstractMatrix,
  A::Snapshots{T,2},
  B::Adjoint{S,<:Snapshots}
  ) where {T,S}
  consec_mul!(C,get_all_data(A),adjoint(get_all_data(B.parent)))
end

function LinearAlgebra.mul!(
  C::AbstractMatrix,
  A::Snapshots{T,2},
  B::AbstractMatrix{S}
  ) where {T,S}
  consec_mul!(C,get_all_data(A),B)
end

function LinearAlgebra.mul!(
  C::AbstractMatrix,
  A::Snapshots{T,2},
  B::Adjoint{T,<:AbstractMatrix{S}}
  ) where {T,S}
  consec_mul!(C,get_all_data(A),B)
end

function LinearAlgebra.mul!(
  C::AbstractMatrix,
  A::Adjoint{T,<:Snapshots{T,2}},
  B::Snapshots{S,2}
  ) where {T,S}
  consec_mul!(C,adjoint(get_all_data(A.parent)),get_all_data(B))
end

function LinearAlgebra.mul!(
  C::AbstractMatrix,
  A::AbstractMatrix{T},
  B::Snapshots{S,2}
  ) where {T,S}
  consec_mul!(C,A,get_all_data(B))
end

function LinearAlgebra.mul!(
  C::AbstractMatrix,
  A::Adjoint{T,<:AbstractMatrix},
  B::Snapshots{S,2}
  ) where {T,S}
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
