abstract type SearchParamLength end

struct AdaptToParamLength <: SearchParamLength end
struct TensProdParamLength <: SearchParamLength end

"""
    struct VariableParamBlock{A} <: ParamBlock{A}
      data::Vector{A}
    end

Same as a [`GenericParamBlock`](@ref), but it does not necessarily store a vector 
of parametric quantities of the correct length. Instead, it stores a vector of parametric quantities of potentially different types, but all of them must be subtypes of the same abstract type `A`. This allows for more flexibility in representing parametric data structures that may have heterogeneous components.
"""
struct VariableParamBlock{A} <: ParamBlock{A}
  data::Vector{A}
end

function Base.getindex(b::VariableParamBlock{A},i::Integer) where A
  b.data[i]
end

function Base.setindex!(b::VariableParamBlock{A},v,i::Integer) where A
  b.data[i] = v 
end

get_param_data(b::VariableParamBlock) = b.data
param_length(b::VariableParamBlock) = length(b.data)
param_getindex(b::VariableParamBlock,i::Integer) = b.data[i]
param_setindex!(b::VariableParamBlock,v,i::Integer) = (b.data[i]=v)

function get_param_entry!(v::AbstractVector,b::VariableParamBlock,i...)
  for k in eachindex(v)
    @inbounds v[k] = b.data[k][i...]
  end
  v
end

Base.copy(a::VariableParamBlock) = VariableParamBlock(copy(a.data))

Base.similar(a::VariableParamBlock) = VariableParamBlock(similar(a.data))

function Base.similar(a::VariableParamBlock{T},::Type{T′}) where {T,T′}
  data′ = map(x -> similar(x,T′),a.data)
  VariableParamBlock(data′)
end

function Base.copyto!(a::VariableParamBlock,b::VariableParamBlock)
  @check size(a) == size(b)
  for i in eachindex(a.data)
    fill!(a.data[i],zero(eltype(a.data[i])))
    copyto!(a.data[i],b.data[i])
  end
  a
end

function Base.:≈(a::VariableParamBlock,b::VariableParamBlock)
  if size(a) != size(b)
    return false
  end
  for i in eachindex(a.data)
    if !(a.data[i] ≈ b.data[i])
      return false
    end
  end
  true
end

function Base.:(==)(a::VariableParamBlock,b::VariableParamBlock)
  if size(a) != size(b)
    return false
  end
  for i in eachindex(a.data)
    if a.data[i] != b.data[i]
      return false
    end
  end
  true
end

function Arrays.testvalue(a::VariableParamBlock{A}) where A
  v = testvalue(A)
  data = Vector{typeof(v)}(undef,param_length(a))
  fill!(data,v)
  VariableParamBlock(data)
end

function Arrays.collect1d(a::VariableParamBlock{A}) where A
  VariableParamBlock(map(collect1d,a.data))
end

# this one misses the param length
function Arrays.testvalue(::Type{VariableParamBlock{A}}) where A
  VariableParamBlock([testvalue(A)])
end

function Arrays.CachedArray(a::VariableParamBlock)
  ai = testitem(a)
  ci = CachedArray(ai)
  data = Vector{typeof(ci)}(undef,param_length(a))
  for i in eachindex(a.data)
    data[i] = CachedArray(a.data[i])
  end
  VariableParamBlock(data)
end

function Arrays.unwrap_cached_array(a::VariableParamBlock)
  cache = return_cache(Arrays.unwrap_cached_array,a)
  evaluate!(cache,Arrays.unwrap_cached_array,a)
end

function Arrays.return_cache(::typeof(Arrays.unwrap_cached_array),a::VariableParamBlock)
  ai = testitem(a)
  ci = return_cache(Arrays.unwrap_cached_array,ai)
  ri = evaluate!(ci,Arrays.unwrap_cached_array,ai)
  c = Vector{typeof(ci)}(undef,length(a.data))
  data = Vector{typeof(ri)}(undef,length(a.data))
  for i in eachindex(a.data)
    c[i] = return_cache(Arrays.unwrap_cached_array,a.data[i])
  end
  VariableParamBlock(data),c
end

function Arrays.evaluate!(cache,::typeof(Arrays.unwrap_cached_array),a::VariableParamBlock)
  r,c = cache
  for i in eachindex(a.data)
    r.data[i] = evaluate!(c[i],Arrays.unwrap_cached_array,a.data[i])
  end
  r
end
