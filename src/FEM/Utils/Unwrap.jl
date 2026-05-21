function unwrap_and_setsize!(a::CachedVector,b::AbstractVector)
  setsize!(a,size(b))
  a.array
end

function unwrap_and_setsize!(a::VectorBlock,b::VectorBlock)
  cache = return_cache(unwrap_and_setsize!,a,b)
  evaluate!(cache,unwrap_and_setsize!,a,b)
end

function Arrays.return_cache(::typeof(unwrap_and_setsize!),a::VectorBlock,b::VectorBlock)
  @check size(a) == size(b)
  ai = testitem(a)
  bi = testitem(b)
  ki = return_cache(unwrap_and_setsize!,ai,bi)
  ri = evaluate!(ki,unwrap_and_setsize!,ai,bi)
  k = Vector{typeof(ki)}(undef,length(a))
  array = Vector{typeof(ri)}(undef,length(a))
  @inbounds for i in eachindex(a.array)
    if a.touched[i]
      @check b.touched[i]
      k[i] = return_cache(unwrap_and_setsize!,a.array[i],b.array[i])
    end
  end
  ArrayBlock(array,a.touched),k
end

function Arrays.evaluate!(cache,::typeof(unwrap_and_setsize!),a::VectorBlock,b::VectorBlock)
  r,k = cache
  @check r.touched == a.touched 
  @inbounds for i in eachindex(a.array)
    if a.touched[i]
      @check b.touched[i]
      r.array[i] = evaluate!(k[i],unwrap_and_setsize!,a.array[i],b.array[i])
    end
  end
  r
end

function unwrap_and_setsize!(a::CachedVector,b::AbstractVector,c::AbstractVector)
  setsize!(a,(length(b)*length(c),))
  a.array
end

function unwrap_and_setsize!(a::MatrixBlock,b::VectorBlock,c::VectorBlock)
  cache = return_cache(unwrap_and_setsize!,a,b,c)
  evaluate!(cache,unwrap_and_setsize!,a,b,c)
end

function Arrays.return_cache(::typeof(unwrap_and_setsize!),a::MatrixBlock,b::VectorBlock,c::VectorBlock)
  @check size(a,1) == length(b) && size(a,2) == length(c)
  ai = testitem(a)
  bi = testitem(b)
  ci = testitem(c)
  ki = return_cache(unwrap_and_setsize!,ai,bi,ci)
  ri = evaluate!(ki,unwrap_and_setsize!,ai,bi,ci)
  k = Matrix{typeof(ki)}(undef,size(a))
  array = Matrix{typeof(ri)}(undef,size(a))
  @inbounds for j in axes(a,2), i in axes(a,1)
    if a.touched[i,j]
      @check b.touched[i]
      @check c.touched[j]
      k[i,j] = return_cache(unwrap_and_setsize!,a.array[i,j],b.array[i],c.array[j])
    end
  end
  ArrayBlock(array,a.touched),k
end

function Arrays.evaluate!(cache,::typeof(unwrap_and_setsize!),a::MatrixBlock,b::VectorBlock,c::VectorBlock)
  r,k = cache
  @check r.touched == a.touched
  @inbounds for j in axes(a,2), i in axes(a,1)
    if a.touched[i,j]
      @check b.touched[i]
      @check c.touched[j]
      r.array[i,j] = evaluate!(k[i,j],unwrap_and_setsize!,a.array[i,j],b.array[i],c.array[j])
    end
  end
  r
end

function unwrap_and_setsize!(a::VectorBlockView,b::VectorBlockView)
  cache = return_cache(unwrap_and_setsize!,a,b)
  evaluate!(cache,unwrap_and_setsize!,a,b)
end

function Arrays.return_cache(::typeof(unwrap_and_setsize!),a::VectorBlockView,b::VectorBlockView)
  cache = return_cache(unwrap_and_setsize!,a.array,b.array)
  array = evaluate!(cache,unwrap_and_setsize!,a.array,b.array)
  return ArrayBlockView(array,a.block_map),cache
end

function Arrays.evaluate!(cache,::typeof(unwrap_and_setsize!),a::VectorBlockView,b::VectorBlockView)
  r,c = cache
  evaluate!(c,unwrap_and_setsize!,a.array,b.array)
  return r
end

function unwrap_and_setsize!(a::MatrixBlockView,b::VectorBlockView,c::VectorBlockView)
  cache = return_cache(unwrap_and_setsize!,a,b,c)
  evaluate!(cache,unwrap_and_setsize!,a,b,c)
end

function Arrays.return_cache(::typeof(unwrap_and_setsize!),a::MatrixBlockView,b::VectorBlockView,c::VectorBlockView)
  cache = return_cache(unwrap_and_setsize!,a.array,b.array,c.array)
  array = evaluate!(cache,unwrap_and_setsize!,a.array,b.array,c.array)
  return ArrayBlockView(array,a.block_map),cache
end

function Arrays.evaluate!(cache,::typeof(unwrap_and_setsize!),a::MatrixBlockView,b::VectorBlockView,c::VectorBlockView)
  r,k = cache
  evaluate!(k,unwrap_and_setsize!,a.array,b.array,c.array)
  return r
end