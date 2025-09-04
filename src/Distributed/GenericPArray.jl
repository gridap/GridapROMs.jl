"""
    struct GenericPArray{V,A,B,C,D,T,N} <: AbstractArray{T,N}
      array_partition::A
      index_partition::B
      unpartitioned_axes::C
      cache::D
    end

Same as [`PVector`](@ref), but while the latter always stores a vector with entries
partitioned on different cores, this structure stores an array (not necessarily
a vector) partitioned along the first dimension (row-wise)
"""
struct GenericPArray{V,A,B,C,D,T,N} <: AbstractArray{T,N}
  array_partition::A
  index_partition::B
  unpartitioned_axes::C
  cache::D
  @doc """
      GenericPArray(array_partition,index_partition)

  Create an instance of [`GenericPArray`](@ref) from the underlying properties
  `array_partition` and `index_partition`.
  """
  function GenericPArray(
    array_partition::AbstractArray{<:AbstractArray{T,N}},
    index_partition,
    unpartitioned_axes=axes(getany(array_partition))[2:end],
    cache=PartitionedArrays.p_vector_cache(array_partition,index_partition)
    ) where {T,N}

    V = eltype(array_partition)
    A = typeof(array_partition)
    B = typeof(index_partition)
    C = typeof(unpartitioned_axes)
    D = typeof(cache)
    new{V,A,B,C,D,T,N}(array_partition,index_partition,unpartitioned_axes,cache)
  end
end

GridapDistributed.local_views(a::GenericPArray) = partition(a)
PartitionedArrays.partition(a::GenericPArray) = a.array_partition
Base.axes(a::GenericPArray) = (PRange(a.index_partition),a.unpartitioned_axes...)

function PartitionedArrays.local_values(a::GenericPArray)
  partition(a)
end

function PartitionedArrays.own_values(a::GenericPArray)
  map(own_values,partition(a),partition(axes(a,1)))
end

function PartitionedArrays.ghost_values(a::GenericPArray)
  map(ghost_values,partition(a),partition(axes(a,1)))
end

Base.size(a::GenericPArray) = length.(axes(a))
Base.IndexStyle(::Type{<:GenericPArray}) = IndexLinear()

function Base.getindex(a::GenericPArray,gid::Int)
  PartitionedArrays.scalar_indexing_action(a)
end

function Base.setindex!(a::GenericPArray,v,gid::Int)
  PartitionedArrays.scalar_indexing_action(a)
end

function Base.show(io::IO,k::MIME"text/plain",a::GenericPArray)
  T = eltype(partition(a))
  s = size(a)
  np = length(partition(a))
  map_main(partition(a)) do values
    println(io,"GenericPArray{$T} of size $s partitioned into $np parts")
  end
end

function PartitionedArrays.assemble!(a::GenericPArray)
  assemble!(+,a)
end

function PartitionedArrays.assemble!(o,a::GenericPArray)
  t = assemble!(o,partition(a),a.cache)
  @async begin
    wait(t)
    map(ghost_values(a)) do a
      fill!(a,zero(eltype(a)))
    end
    a
  end
end

function consistent!(a::GenericPArray)
  insert(a,b) = b
  cache = map(reverse,a.cache)
  t = assemble!(insert,partition(a),cache)
  @async begin
    wait(t)
    a
  end
end

function Base.similar(a::GenericPArray,::Type{T},inds::Tuple) where T
  rows,uaxes... = inds
  @check isa(rows,PRange)
  values = map(partition(a),partition(rows)) do values,indices
    inds = (local_length(indices),map(length,uaxes)...)
    similar(values,T,inds)
  end
  GenericPArray(values,partition(rows),uaxes)
end

function Base.similar(::Type{<:GenericPArray{V}},inds::Tuple) where V
  rows,uaxes... = inds
  @check isa(rows,PRange)
  values = map(partition(rows)) do indices
    inds = (local_length(indices),map(length,uaxes)...)
    similar(values,T,inds)
  end
  GenericPArray(values,partition(rows),uaxes)
end

function GenericPArray(::UndefInitializer,index_partition,uaxes...)
  GenericPArray{Vector{Float64}}(undef,index_partition,uaxes...)
end

function GenericPArray{A}(::UndefInitializer,index_partition,uaxes...) where A
  array_partition = map(index_partition) do indices
    inds = (local_length(indices),map(length,uaxes)...)
    similar(A,inds)
  end
  GenericPArray(array_partition,index_partition)
end

function Base.copy!(a::GenericPArray,b::GenericPArray)
  @assert size(a) == size(b)
  copyto!(a,b)
end

function Base.copyto!(a::GenericPArray,b::GenericPArray)
  if partition(axes(a,1)) === partition(axes(b,1))
    map(copy!,partition(a),partition(b))
  elseif PartitionedArrays.matching_own_indices(axes(a,1),axes(b,1))
    map(copy!,own_values(a),own_values(b))
  else
    error("Trying to copy a GenericPArray into another one with a different data layout. This case is not implemented yet. It would require communications.")
  end
  a
end

function Base.fill!(a::GenericPArray,v)
  map(partition(a)) do values
    fill!(values,v)
  end
  a
end

function Base.:(==)(a::GenericPArray,b::GenericPArray)
  @boundscheck @assert PartitionedArrays.matching_own_indices(axes(a,1),axes(b,1))
  length(a) == length(b) &&
  reduce(&,map(==,own_values(a),own_values(b)),init=true)
end

function Base.any(f::Function,x::GenericPArray)
  partials = map(own_values(x)) do o
    any(f,o)
  end
  reduce(|,partials,init=false)
end

function Base.all(f::Function,x::GenericPArray)
  partials = map(own_values(x)) do o
    all(f,o)
  end
  reduce(&,partials,init=true)
end

Base.maximum(x::GenericPArray) = maximum(identity,x)
function Base.maximum(f::Function,x::GenericPArray)
  partials = map(own_values(x)) do o
    maximum(f,o,init=typemin(eltype(x)))
  end
  reduce(max,partials,init=typemin(eltype(x)))
end

Base.minimum(x::GenericPArray) = minimum(identity,x)
function Base.minimum(f::Function,x::GenericPArray)
  partials = map(own_values(x)) do o
    minimum(f,o,init=typemax(eltype(x)))
  end
  reduce(min,partials,init=typemax(eltype(x)))
end

function Base.collect(v::GenericPArray)
  own_values_v = own_values(v)
  own_to_global_v = map(own_to_global,partition(axes(v,1)))
  vals = gather(own_values_v,destination=:all)
  ids = gather(own_to_global_v,destination=:all)
  n = length(v)
  T = eltype(v)
  map(vals,ids) do myvals,myids
    u = Vector{T}(undef,n)
    for (a,b) in zip(myvals,myids)
      u[b] = a
    end
    u
  end |> getany
end

function Base.:*(a::Number,b::GenericPArray)
  values = map(partition(b)) do values
    a*values
  end
  GenericPArray(values,partition(axes(b,1)))
end

function Base.:*(b::GenericPArray,a::Number)
  a*b
end

function Base.:/(b::GenericPArray,a::Number)
  (1/a)*b
end

for op in (:+,:-)
  @eval begin
    function Base.$op(a::GenericPArray)
      values = map($op,partition(a))
      GenericPArray(values,partition(axes(a,1)))
    end
    function Base.$op(a::GenericPArray,b::GenericPArray)
      $op.(a,b)
    end
  end
end

function Base.reduce(op,a::GenericPArray;neutral=PartitionedArrays.neutral_element(op,eltype(a)),kwargs...)
  b = map(own_values(a)) do a
    reduce(op,a,init=neutral)
  end
  reduce(op,b;kwargs...)
end

function Base.sum(a::GenericPArray)
  reduce(+,a,init=zero(eltype(a)))
end

function LinearAlgebra.dot(a::GenericPArray,b::GenericPArray)
  c = map(dot,own_values(a),own_values(b))
  sum(c)
end

function LinearAlgebra.rmul!(a::GenericPArray,v::Number)
  map(partition(a)) do l
    rmul!(l,v)
  end
  a
end

function LinearAlgebra.norm(a::GenericPArray,p::Real=2)
  contibs = map(own_values(a)) do oid_to_value
    norm(oid_to_value,p)^p
  end
  reduce(+,contibs;init=zero(eltype(contibs)))^(1/p)
end

function Base.:*(a::PSparseMatrix,b::GenericPArray)
  Ta = eltype(a)
  Tb = eltype(b)
  T = typeof(zero(Ta)*zero(Tb)+zero(Ta)*zero(Tb))
  N = ndims(b)
  paxis,uaxes... = axes(a)
  c = GenericPArray{Array{T,N}}(undef,partition(paxis),uaxes...)
  mul!(c,a,b)
  c
end

function LinearAlgebra.mul!(c::GenericPArray,a::PSparseMatrix,b::GenericPArray,α::Number,β::Number)
  @boundscheck @assert PartitionedArrays.matching_own_indices(axes(c,1),axes(a,1))
  @boundscheck @assert PartitionedArrays.matching_own_indices(axes(a,2),axes(b,1))
  @boundscheck @assert PartitionedArrays.matching_ghost_indices(axes(a,2),axes(b,1))
  # Start the exchange
  t = consistent!(b)
  # Meanwhile, process the owned blocks
  map(own_values(c),own_values(a),own_values(b)) do co,aoo,bo
    if β != 1
      β != 0 ? rmul!(co, β) : fill!(co,zero(eltype(co)))
    end
    mul!(co,aoo,bo,α,1)
  end
  # Wait for the exchange to finish
  wait(t)
  # process the ghost block
  map(own_values(c),own_ghost_values(a),ghost_values(b)) do co,aoh,bh
    mul!(co,aoh,bh,α,1)
  end
  c
end
