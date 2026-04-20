function unwrap_and_setsize!(a::CachedArray,b::AbstractArray)
  setsize!(a,size(b))
  a.array
end

function unwrap_and_setsize!(a::CachedArray,b::AbstractArray,c::AbstractArray)
  setsize!(a,size(b) .* size(c))
  a.array
end

function unwrap_and_setsize!(a::ArrayBlock,b::ArrayBlock)
  cache = return_cache(unwrap_and_setsize!,a,b)
  evaluate!(cache,unwrap_and_setsize!,a,b)
end

function unwrap_and_setsize!(a::ArrayBlock,b::ArrayBlock,c::ArrayBlock)
  cache = return_cache(unwrap_and_setsize!,a,b,c)
  evaluate!(cache,unwrap_and_setsize!,a,b,c)
end

function Arrays.return_cache(::typeof(unwrap_and_setsize!),a::ArrayBlock,b::ArrayBlock)
  @check a.touched == b.touched
  ai = testitem(a)
  bi = testitem(b)
  ci = return_cache(unwrap_and_setsize!,ai,bi)
  ri = evaluate!(ci,unwrap_and_setsize!,ai,bi)
  c = Array{typeof(ci),ndims(a)}(undef,size(a))
  array = Array{typeof(ri),ndims(a)}(undef,size(a))
  @inbounds for i in eachindex(a.array)
    if a.touched[i]
      c[i] = return_cache(unwrap_and_setsize!,a.array[i],b.array[i])
    end
  end
  ArrayBlock(array,a.touched),c
end

function Arrays.evaluate!(cache,::typeof(unwrap_and_setsize!),a::ArrayBlock,b::ArrayBlock)
  r,c = cache
  @check size(r) == size(a)
  @inbounds for i in eachindex(a.array)
    if a.touched[i]
      r.array[i] = evaluate!(c[i],unwrap_and_setsize!,a.array[i],b.array[i])
    end
  end
  r
end

function Arrays.return_cache(::typeof(unwrap_and_setsize!),a::ArrayBlock,b::ArrayBlock,c::ArrayBlock)
  @check a.touched == b.touched == c.touched
  ai = testitem(a)
  bi = testitem(b)
  ci = testitem(c)
  ki = return_cache(unwrap_and_setsize!,ai,bi,ci)
  ri = evaluate!(ki,unwrap_and_setsize!,ai,bi,ci)
  k = Array{typeof(ki),ndims(a)}(undef,size(a))
  array = Array{typeof(ri),ndims(a)}(undef,size(a))
  @inbounds for i in eachindex(a.array)
    if a.touched[i]
      k[i] = return_cache(unwrap_and_setsize!,a.array[i],b.array[i],c.array[i])
    end
  end
  ArrayBlock(array,a.touched),k
end

function Arrays.evaluate!(cache,::typeof(unwrap_and_setsize!),a::ArrayBlock,b::ArrayBlock,c::ArrayBlock)
  r,k = cache
  @check size(r) == size(a)
  @inbounds for i in eachindex(a.array)
    if a.touched[i]
      r.array[i] = evaluate!(k[i],unwrap_and_setsize!,a.array[i],b.array[i],c.array[i])
    end
  end
  r
end

function unwrap_and_setsize!(a::ArrayBlockView,b::ArrayBlockView)
  cache = return_cache(unwrap_and_setsize!,a,b)
  evaluate!(cache,unwrap_and_setsize!,a,b)
end

function unwrap_and_setsize!(a::ArrayBlockView,b::ArrayBlockView,c::ArrayBlockView)
  cache = return_cache(unwrap_and_setsize!,a,b,c)
  evaluate!(cache,unwrap_and_setsize!,a,b,c)
end

function Arrays.return_cache(::typeof(unwrap_and_setsize!),a::ArrayBlockView,b::ArrayBlockView)
  cache = return_cache(unwrap_and_setsize!,a.array,b.array)
  array = evaluate!(cache,unwrap_and_setsize!,a.array,b.array)
  return ArrayBlockView(array,a.block_map),cache
end

function Arrays.evaluate!(cache,::typeof(unwrap_and_setsize!),a::ArrayBlockView,b::ArrayBlockView)
  r,c = cache
  evaluate!(c,unwrap_and_setsize!,a.array,b.array)
  return r
end

function Arrays.return_cache(::typeof(unwrap_and_setsize!),a::ArrayBlockView,b::ArrayBlockView,c::ArrayBlockView)
  cache = return_cache(unwrap_and_setsize!,a.array,b.array,c.array)
  array = evaluate!(cache,unwrap_and_setsize!,a.array,b.array,c.array)
  return ArrayBlockView(array,a.block_map),cache
end

function Arrays.evaluate!(cache,::typeof(unwrap_and_setsize!),a::ArrayBlockView,b::ArrayBlockView,c::ArrayBlockView)
  r,k = cache
  evaluate!(k,unwrap_and_setsize!,a.array,b.array,c.array)
  return r
end