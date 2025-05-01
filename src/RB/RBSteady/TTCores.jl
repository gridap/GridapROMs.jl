function DofMaps.recast(
  a::AbstractVector{<:AbstractArray{T,3}},
  i::TrivialSparseMatrixDofMap
  ) where T

  a′ = Vector{AbstractArray{T,3}}(undef,length(a))
  for n in eachindex(a)
    a′[n] = n == 1 ? SparseCore(a[n],i) : a[n]
  end
  return a′
end

function DofMaps.recast(
  a::AbstractVector{<:AbstractArray{T,3}},
  i::SparseMatrixDofMap
  ) where T

  sparsities_1d = i.sparsity.sparsities_1d
  @check length(sparsities_1d) ≤ length(a)
  a′ = Vector{AbstractArray{T,3}}(undef,length(a))
  for n in eachindex(a)
    a′[n] = n ≤ length(sparsities_1d) ? SparseCore(a[n],sparsities_1d[n]) : a[n]
  end
  return a′
end

"""
    abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

Type for nonstandard representations of tensor train cores.

Subtypes:

- [`DofMapCore`](@ref)
- [`SparseCore`](@ref)
"""
abstract type AbstractTTCore{T,N} <: AbstractArray{T,N} end

"""
    struct DofMapCore{T,A<:AbstractArray{T,3},B<:AbstractArray} <: AbstractTTCore{T,3}
      core::A
      dof_map::B
    end

Represents a tensor train core `core` reindexed by means of an index mapping `dof_map`
"""
struct DofMapCore{T,A<:AbstractArray{T,3},B<:AbstractArray} <: AbstractTTCore{T,3}
  core::A
  dof_map::B
end

Base.size(a::DofMapCore) = (size(a.core,1),length(a.dof_map),size(a.core,3))

function Base.getindex(a::DofMapCore,i::Integer,j::Integer,k::Integer)
  j′ = a.dof_map[j]
  iszero(j′) ? zero(eltype(a)) : a.core[i,j′,k]
end

function Base.setindex!(a::DofMapCore,v,i::Integer,j::Integer,k::Integer)
  j′ = a.dof_map[j]
  !iszero(j′) && (a.core[i,j′,k] = v)
end

"""
    abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

Tensor train cores for sparse matrices.

Subtypes:

- [`SparseCoreCSC`](@ref)
- [`SparseCoreCSC4D`](@ref)
"""
abstract type SparseCore{T,N} <: AbstractTTCore{T,N} end

function SparseCore(array::Array{T,3},sparsity::TrivialSparseMatrixDofMap) where T
  sparsity′ = get_sparsity(sparsity)
  TrivialSparseCore(array,sparsity′)
end

"""
    struct TrivialSparseCore{T,A<:SparsityPattern} <: SparseCore{T,3}
      array::Array{T,3}
      sparsity::A
    end

Trivial tensor train core for sparse matrices format
"""
struct TrivialSparseCore{T,A<:SparsityPattern} <: SparseCore{T,3}
  array::Array{T,3}
  sparsity::A
end

function SparseCore(array::Array{T,3},sparsity::SparsityPattern) where T
  sparsity′ = DofMaps.get_background_sparsity(sparsity)
  SparseCore(array,sparsity′)
end

Base.size(a::TrivialSparseCore) = size(a.array)
Base.getindex(a::TrivialSparseCore,i::Vararg{Integer,3}) = getindex(a.array,i...)

cores2basis(a::TrivialSparseCore) = recast(cores2basis(a.array),a.sparsity)

"""
    struct SparseCoreCSC{T,Ti} <: SparseCore{T,3}
      array::Array{T,3}
      sparsity::SparsityCSC{T,Ti}
    end

Tensor train cores for sparse matrices in CSC format
"""
struct SparseCoreCSC{T,Ti} <: SparseCore{T,3}
  array::Array{T,3}
  sparsity::SparsityCSC{T,Ti}
end

function SparseCore(array::Array{T,3},sparsity::SparsityCSC{T}) where T
  SparseCoreCSC(array,sparsity)
end

Base.size(a::SparseCoreCSC) = size(a.array)
Base.getindex(a::SparseCoreCSC,i::Vararg{Integer,3}) = getindex(a.array,i...)

to_4d_core(a::SparseCoreCSC) = SparseCoreCSC4D(a)

"""
    struct SparseCoreCSC4D{T,Ti} <: SparseCore{T,4}
      core::SparseCoreCSC{T,Ti}
      sparse_indexes::Vector{CartesianIndex{2}}
    end

Tensor train cores for sparse matrices in CSC format, reshaped as 4D arrays
"""
struct SparseCoreCSC4D{T,Ti} <: SparseCore{T,4}
  core::SparseCoreCSC{T,Ti}
  sparse_indexes::Vector{CartesianIndex{2}}
end

function SparseCoreCSC4D(core::SparseCoreCSC)
  irows,icols,_ = findnz(core.sparsity)
  SparseCoreCSC4D(core,CartesianIndex.(irows,icols))
end

Base.size(a::SparseCoreCSC4D) = (size(a.core.array,1),DofMaps.num_rows(a.core.sparsity),
  DofMaps.num_cols(a.core.sparsity),size(a.core.array,3))

function Base.getindex(a::SparseCoreCSC4D,i::Vararg{Integer,4})
  if CartesianIndex(i[2:3]) ∈ a.sparse_indexes
    i2 = findfirst(a.sparse_indexes .== [CartesianIndex(i[2:3])])
    i1,i3 = i[1],i[4]
    getindex(a.core.array,i1,i2,i3)
  else
    zero(eltype(a))
  end
end

# block cores

function first_block(a::AbstractArray{T,3},b::AbstractArray{S,3}) where {T,S}
  @check size(a,1) == size(b,1) == 1
  @check size(a,2) == size(b,2) "Cannot sum the two input cores"
  TS = promote_type(T,S)
  r2 = size(a,2)
  r3 = size(a,3) + size(b,3)
  ab = zeros(TS,1,r2,r3)
  @views ab[:,:,1:size(a,3)] = a
  @views ab[:,:,1+size(a,3):end] = b
  return ab
end

function last_block(a::AbstractArray{T,3},b::AbstractArray{S,3}) where {T,S}
  @check size(a,3) == size(b,3) == 1
  @check size(a,2) == size(b,2) "Cannot sum the two input cores"
  TS = promote_type(T,S)
  r1 = size(a,1) + size(b,1)
  r2 = size(a,2)
  ab = zeros(TS,r1,r2,1)
  @views ab[1:size(a,1),:,:] = a
  @views ab[1+size(a,1):end,:,:] = b
  return ab
end

function block_core(a::AbstractArray{T,3},b::AbstractArray{S,3}) where {T,S}
  @check size(a,2) == size(b,2) "Cannot sum the two input cores"
  TS = promote_type(T,S)
  r1 = size(a,1) + size(b,1)
  r2 = size(a,2)
  r3 = size(a,3) + size(b,3)
  ab = zeros(TS,r1,r2,r3)
  @views ab[1:size(a,1),:,1:size(a,3)] = a
  @views ab[1+size(a,1):end,:,1+size(a,3):end] = b
  return ab
end

for f in (:first_block,:block_core)
  @eval begin
    function $f(a::AbstractVector{<:AbstractArray{T,3}}) where T
      D = length(a)
      @check D ≤ 3
      if D == 1
        a[1]
      elseif D == 2
        $f(a[1],a[2])
      else
        $f($f(a[1],a[2]),a[3])
      end
    end
  end
end

function block_cores(a::AbstractVector{<:AbstractVector{<:AbstractArray{T,3}}}) where T
  D = length(first(a))
  @check all(length(ai)==D for ai in a)
  abfirst = first_block(getindex.(a,1))
  ablasts = map(d -> block_core(getindex.(a,d)),2:D)
  return [abfirst,ablasts...]
end

function block_cat(a::AbstractVector{<:AbstractArray{T,3}};kwargs...) where T
  D = length(a)
  @check D ≤ 3
  if D == 1
    a[1]
  elseif D == 2
    cat(a[1],a[2];kwargs...)
  else
    cat(a[1],a[2],a[3];kwargs...)
  end
end

function _block_cores_add_component(
  a::AbstractVector{<:AbstractVector{<:AbstractArray{T,3}}}
  ) where T

  D = length(first(a))
  ablocks = block_cores(a)
  ND = size(last(ablocks),3)
  N = Int(ND/D)
  In = T.(I(N))
  ablast = zeros(T,ND,D,N)
  for d in 1:D
    @views ablast[(d-1)*N+1:d*N,d,:] = In
  end
  push!(ablocks,ablast)
  return ablocks
end
