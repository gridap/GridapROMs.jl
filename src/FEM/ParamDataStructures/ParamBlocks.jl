"""
    abstract type ParamBlock{A} end

Type representing cell-wise quantities defined during the integration routine.
They are primarily used when lazily evaluating parametric quantities on the mesh.
The implementation of the lazy interface mimics that of `ArrayBlock` in [`Gridap`](@ref).
Subtypes:
-[`GenericParamBlock`](@ref)
-[`TrivialParamBlock`](@ref)
"""
abstract type ParamBlock{A} end

Base.size(b::ParamBlock) = tfill(param_length(b),Val{ndims(b)}())
Base.length(b::ParamBlock) = param_length(b)^ndims(b)
Base.eltype(::Type{<:ParamBlock{A}}) where A = A
Base.eltype(b::ParamBlock{A}) where A = A
Base.ndims(b::ParamBlock{A}) where A = ndims(A)
Base.ndims(::Type{<:ParamBlock{A}}) where A = ndims(A)
Base.ndims(b::ParamBlock{<:Map}) = 0
Base.ndims(::Type{<:ParamBlock{<:Map}}) = 0

Arrays.testitem(b::ParamBlock) = param_getindex(b,1)

function Base.:≈(a::AbstractArray{<:ParamBlock},b::AbstractArray{<:ParamBlock})
  all(z->z[1]≈z[2],zip(a,b))
end

"""
    struct GenericParamBlock{A} <: ParamBlock{A}
      data::Vector{A}
    end

Most standard implementation of a [`ParamBlock`](@ref)
"""
struct GenericParamBlock{A} <: ParamBlock{A}
  data::Vector{A}
end

function Base.getindex(b::GenericParamBlock{A},i::Integer) where A
  b.data[i]
end

function Base.setindex!(b::GenericParamBlock{A},v,i::Integer) where A
  b.data[i] = v 
end

get_param_data(b::GenericParamBlock) = b.data
param_length(b::GenericParamBlock) = length(b.data)
param_getindex(b::GenericParamBlock,i::Integer) = b.data[i]
param_setindex!(b::GenericParamBlock,v,i::Integer) = (b.data[i]=v)

function get_param_entry!(v::AbstractVector,b::GenericParamBlock,i...)
  for k in eachindex(v)
    @inbounds v[k] = b.data[k][i...]
  end
  v
end

Base.copy(a::GenericParamBlock) = GenericParamBlock(copy(a.data))

Base.similar(a::GenericParamBlock) = GenericParamBlock(similar(a.data))

function Base.similar(a::GenericParamBlock{T},::Type{T′}) where {T,T′}
  data′ = map(x -> similar(x,T′),a.data)
  GenericParamBlock(data′)
end

function Base.copyto!(a::GenericParamBlock,b::GenericParamBlock)
  @check size(a) == size(b)
  for i in eachindex(a.data)
    fill!(a.data[i],zero(eltype(a.data[i])))
    copyto!(a.data[i],b.data[i])
  end
  a
end

function Base.:≈(a::GenericParamBlock,b::GenericParamBlock)
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

function Base.:(==)(a::GenericParamBlock,b::GenericParamBlock)
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

function Arrays.testvalue(a::GenericParamBlock{A}) where A
  v = testvalue(A)
  data = Vector{typeof(v)}(undef,param_length(a))
  fill!(data,v)
  GenericParamBlock(data)
end

function Arrays.collect1d(a::GenericParamBlock{A}) where A
  GenericParamBlock(map(collect1d,a.data))
end

# this one misses the param length
function Arrays.testvalue(::Type{GenericParamBlock{A}}) where A
  GenericParamBlock([testvalue(A)])
end

function Arrays.CachedArray(a::GenericParamBlock)
  ai = testitem(a)
  ci = CachedArray(ai)
  data = Vector{typeof(ci)}(undef,param_length(a))
  for i in eachindex(a.data)
    data[i] = CachedArray(a.data[i])
  end
  GenericParamBlock(data)
end

function Arrays.unwrap_cached_array(a::GenericParamBlock)
  cache = return_cache(Arrays.unwrap_cached_array,a)
  evaluate!(cache,Arrays.unwrap_cached_array,a)
end

function Arrays.return_cache(::typeof(Arrays.unwrap_cached_array),a::GenericParamBlock)
  ai = testitem(a)
  ci = return_cache(Arrays.unwrap_cached_array,ai)
  ri = evaluate!(ci,Arrays.unwrap_cached_array,ai)
  c = Vector{typeof(ci)}(undef,length(a.data))
  data = Vector{typeof(ri)}(undef,length(a.data))
  for i in eachindex(a.data)
    c[i] = return_cache(Arrays.unwrap_cached_array,a.data[i])
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,::typeof(Arrays.unwrap_cached_array),a::GenericParamBlock)
  r,c = cache
  for i in eachindex(a.data)
    r.data[i] = evaluate!(c[i],Arrays.unwrap_cached_array,a.data[i])
  end
  r
end

###################### start trivial case ######################

"""
    struct TrivialParamBlock{A} <: ParamBlock{A}
      data::A
      plength::Int
    end

Wrapper for a non-paramentric quantity `data` that we wish assumed a parametric
length `plength`
"""
struct TrivialParamBlock{A} <: ParamBlock{A}
  data::A
  plength::Int
end

function TrivialParamBlock(data::Any)
  plength = 1
  TrivialParamBlock(data,plength)
end

function Base.getindex(b::TrivialParamBlock{A},i::Integer) where A
  @assert 1 <= i <= b.plength
  b.data
end

function Base.setindex!(b::TrivialParamBlock{A},v,i::Integer) where A
  @assert 1 <= i <= b.plength
  copyto!(b.data,v)
end

get_param_data(b::TrivialParamBlock) = Fill(b.data,b.plength)
param_length(b::TrivialParamBlock) = b.plength
param_getindex(b::TrivialParamBlock,i::Integer) = b.data
param_setindex!(b::TrivialParamBlock,v,i::Integer) = copyto!(b.data,v)

function get_param_entry!(v::AbstractVector,b::TrivialParamBlock,i...)
  vk = b.data[i...]
  fill!(v,vk)
end

Base.copy(a::TrivialParamBlock) = TrivialParamBlock(copy(a.data),a.plength)

Base.similar(a::TrivialParamBlock) = TrivialParamBlock(similar(a.data),a.plength)

function Base.similar(a::TrivialParamBlock{T},::Type{T′}) where {T,T′}
  data′ = similar(a.data,T′)
  TrivialParamBlock(data′,a.plength)
end

Base.copyto!(a::TrivialParamBlock,b::TrivialParamBlock) = copyto!(a.data,b.data)

function Base.:≈(a::TrivialParamBlock,b::TrivialParamBlock)
  if size(a) != size(b)
    return false
  end
  a.data ≈ b.data
end

function Base.:(==)(a::TrivialParamBlock,b::TrivialParamBlock)
  if size(a) != size(b)
    return false
  end
  a.data == b.data
end

function Arrays.testvalue(a::TrivialParamBlock{A}) where A
  TrivialParamBlock(testvalue(A),param_length(a))
end

# this one misses the param length
function Arrays.testvalue(::Type{TrivialParamBlock{A}}) where A
  TrivialParamBlock(testvalue(A),1)
end

function Arrays.CachedArray(a::TrivialParamBlock)
  TrivialParamBlock(CachedArray(a.data),a.plength)
end

function Arrays.unwrap_cached_array(a::TrivialParamBlock)
  TrivialParamBlock(Arrays.unwrap_cached_array(a.data),a.plength)
end

###################### end trivial case ######################

function Arrays.return_cache(f::Operation,x::ParamBlock)
  xi = testitem(x)
  li = return_cache(f,xi)
  fix = evaluate!(li,f,xi)
  l = Vector{typeof(li)}(undef,param_length(x))
  g = Vector{typeof(fix)}(undef,param_length(x))
  for i in param_eachindex(x)
    l[i] = return_cache(f,param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,f::Operation,x::ParamBlock)
  g,l = cache
  for i in param_eachindex(x)
    g.data[i] = evaluate!(l[i],f,param_getindex(x,i))
  end
  g
end

function Arrays.return_cache(f::ParamBlock,x)
  fi = testitem(f)
  li = return_cache(fi,x)
  fix = evaluate!(li,fi,x)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(param_getindex(f,i),x)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,f::ParamBlock,x)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],param_getindex(f,i),x)
  end
  g
end

function Arrays.return_cache(f::ParamBlock,x::ParamBlock)
  @check param_length(f) == param_length(x)
  fi = testitem(f)
  xi = testitem(x)
  li = return_cache(fi,xi)
  fix = evaluate!(li,fi,xi)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(param_getindex(f,i),param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,f::ParamBlock,x::ParamBlock)
  @check param_length(f) == param_length(x)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],param_getindex(f,i),param_getindex(x,i))
  end
  g
end

function Fields.linear_combination(u::ParamBlock,f::ParamBlock)
  @check size(u) == size(f)
  fi = testitem(f)
  ui = testitem(u)
  ufi = linear_combination(ui,fi)
  g = Vector{typeof(ufi)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = linear_combination(param_getindex(u,i),param_getindex(f,i))
  end
  GenericParamBlock(g)
end

function Fields.linear_combination(u::ParamBlock,f::AbstractVector{<:Field})
  ufi = linear_combination(testitem(u),f)
  g = Vector{typeof(ufi)}(undef,param_length(u))
  @inbounds for i in param_eachindex(u)
    g[i] = linear_combination(param_getindex(u,i),f)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::LinearCombinationMap,u::ParamBlock,fx::AbstractArray)
  ui = testitem(u)
  li = return_cache(k,ui,fx)
  ufxi = evaluate!(li,k,ui,fx)
  l = Vector{typeof(li)}(undef,param_length(u))
  g = Vector{typeof(ufxi)}(undef,param_length(u))
  for i in param_eachindex(u)
    l[i] = return_cache(k,param_getindex(u,i),fx)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::ParamBlock,fx::AbstractArray)
  g,l = cache
  for i in param_eachindex(u)
    g.data[i] = evaluate!(l[i],k,param_getindex(u,i),fx)
  end
  g
end

function Arrays.return_cache(k::LinearCombinationMap,u::ParamBlock,fx::ParamBlock)
  fxi = testitem(fx)
  ui = testitem(u)
  li = return_cache(k,ui,fxi)
  ufxi = evaluate!(li,k,ui,fxi)
  l = Vector{typeof(li)}(undef,param_length(fx))
  g = Vector{typeof(ufxi)}(undef,param_length(fx))
  for i in param_eachindex(fx)
    l[i] = return_cache(k,param_getindex(u,i),param_getindex(fx,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::ParamBlock,fx::ParamBlock)
  g,l = cache
  for i in param_eachindex(fx)
    g.data[i] = evaluate!(l[i],k,param_getindex(u,i),param_getindex(fx,i))
  end
  g
end

function Base.transpose(f::ParamBlock)
  fi = testitem(f)
  fit = transpose(fi)
  g = Vector{typeof(fit)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = transpose(param_getindex(f,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Fields.TransposeMap,f::ParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(k,param_getindex(f,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Fields.TransposeMap,f::ParamBlock)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],k,param_getindex(f,i))
  end
  g
end

function Fields.integrate(f::ParamBlock,args...)
  fi = testitem(f)
  intfi = integrate(fi,args...)
  g = Vector{typeof(intfi)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = integrate(param_getindex(f,i),args...)
  end
  GenericParamBlock(g)
end

function Arrays.return_value(k::IntegrationMap,fx::ParamBlock,args...)
  fxi = testitem(fx)
  ufxi = return_value(k,fxi,args...)
  g = Vector{typeof(ufxi)}(undef,param_length(fx))
  for i in param_eachindex(fx)
    g[i] = return_value(k,param_getindex(fx,i),args...)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::IntegrationMap,fx::ParamBlock,args...)
  fxi = testitem(fx)
  li = return_cache(k,fxi,args...)
  ufxi = evaluate!(li,k,fxi,args...)
  l = Vector{typeof(li)}(undef,param_length(fx))
  g = Vector{typeof(ufxi)}(undef,param_length(fx))
  for i in param_eachindex(fx)
    l[i] = return_cache(k,param_getindex(fx,i),args...)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx::ParamBlock,args...)
  g,l = cache
  for i in param_eachindex(fx)
    g.data[i] = evaluate!(l[i],k,param_getindex(fx,i),args...)
  end
  g
end

function Arrays.return_value(k::IntegrationMap,fx,w,jx::ParamBlock)
  jxi = testitem(jx)
  ufxi = return_value(k,fx,w,jxi)
  g = Vector{typeof(ufxi)}(undef,param_length(jx))
  for i in param_eachindex(jx)
    g[i] = return_value(k,fx,w,param_getindex(jx,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::IntegrationMap,fx,w,jx::ParamBlock)
  jxi = testitem(jx)
  li = return_cache(k,fx,w,jxi)
  ufxi = evaluate!(li,k,fx,w,jxi)
  l = Vector{typeof(li)}(undef,param_length(jx))
  g = Vector{typeof(ufxi)}(undef,param_length(jx))
  for i in param_eachindex(jx)
    l[i] = return_cache(k,fx,w,param_getindex(jx,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx,w,jx::ParamBlock)
  g,l = cache
  for i in param_eachindex(jx)
    g.data[i] = evaluate!(l[i],k,fx,w,param_getindex(jx,i))
  end
  g
end

function Arrays.return_value(k::IntegrationMap,fx::ParamBlock,w,jx::ParamBlock)
  @check param_length(fx) == param_length(jx)
  fxi = testitem(fx)
  jxi = testitem(jx)
  ufxi = return_value(k,fxi,w,jxi)
  g = Vector{typeof(ufxi)}(undef,param_length(fx))
  for i in param_eachindex(fx)
    g[i] = return_value(k,param_getindex(fx,i),w,param_getindex(jx,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::IntegrationMap,fx::ParamBlock,w,jx::ParamBlock)
  @check param_length(fx) == param_length(jx)
  fxi = testitem(fx)
  jxi = testitem(jx)
  li = return_cache(k,fxi,w,jxi)
  ufxi = evaluate!(li,k,fxi,w,jxi)
  l = Vector{typeof(li)}(undef,param_length(fx))
  g = Vector{typeof(ufxi)}(undef,param_length(fx))
  for i in param_eachindex(fx)
    l[i] = return_cache(k,param_getindex(fx,i),w,param_getindex(jx,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx::ParamBlock,w,jx::ParamBlock)
  @check param_length(fx) == param_length(jx)
  g,l = cache
  for i in param_eachindex(fx)
    g.data[i] = evaluate!(l[i],k,param_getindex(fx,i),w,param_getindex(jx,i))
  end
  g
end

function Arrays.return_value(k::Fields.VoidBasis,f::ParamBlock)
  fi = testitem(f)
  fix = return_value(k,fi)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = return_value(k,param_getindex(f,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Fields.VoidBasis,f::ParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(k,param_getindex(f,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Fields.VoidBasis,f::ParamBlock)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],k,param_getindex(f,i))
  end
  g
end


function Arrays.return_value(k::Broadcasting,f::ParamBlock)
  fi = testitem(f)
  fix = return_value(k,fi)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = return_value(k,param_getindex(f,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting,f::ParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(k,param_getindex(f,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting,f::ParamBlock)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],k,param_getindex(f,i))
  end
  g
end

function Arrays.return_value(k::Broadcasting{typeof(∘)},f::ParamBlock,h::Field)
  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = return_value(k,param_getindex(f,i),h)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::ParamBlock,h::Field)
  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(k,param_getindex(f,i),h)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::ParamBlock,h::Field)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],k,param_getindex(f,i),h)
  end
  g
end

function Arrays.return_value(k::Broadcasting{typeof(∘)},f::ParamBlock,h::ParamBlock)
  @check param_length(h) == param_length(f)
  fi = testitem(f)
  hi = testitem(h)
  fix = return_value(k,fi,hi)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = return_value(k,param_getindex(f,i),param_getindex(h,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::ParamBlock,h::ParamBlock)
  @check param_length(h) == param_length(f)
  fi = testitem(f)
  hi = testitem(h)
  li = return_cache(k,fi,hi)
  fix = evaluate!(li,k,fi,hi)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(k,param_getindex(f,i),param_getindex(h,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::ParamBlock,h::ParamBlock)
  @check param_length(h) == param_length(f)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],k,param_getindex(f,i),param_getindex(h,i))
  end
  g
end

for F in (:Function,:Operation)
  for T in (:Field,:AbstractArray)
    @eval begin
      function Arrays.return_value(k::Broadcasting{<:$F},f::ParamBlock,h::$T)
        fi = testitem(f)
        fix = return_value(k,fi,h)
        g = Vector{typeof(fix)}(undef,param_length(f))
        for i in param_eachindex(f)
          g[i] = return_value(k,param_getindex(f,i),h)
        end
        GenericParamBlock(g)
      end

      function Arrays.return_cache(k::Broadcasting{<:$F},f::ParamBlock,h::$T)
        fi = testitem(f)
        li = return_cache(k,fi,h)
        fix = evaluate!(li,k,fi,h)
        l = Vector{typeof(li)}(undef,param_length(f))
        g = Vector{typeof(fix)}(undef,param_length(f))
        for i in param_eachindex(f)
          l[i] = return_cache(k,param_getindex(f,i),h)
        end
        GenericParamBlock(g),l
      end

      function Arrays.evaluate!(cache,k::Broadcasting{<:$F},f::ParamBlock,h::$T)
        g,l = cache
        for i in param_eachindex(f)
          g.data[i] = evaluate!(l[i],k,param_getindex(f,i),h)
        end
        g
      end

      function Arrays.return_value(k::Broadcasting{<:$F},h::$T,f::ParamBlock)
        fi = testitem(f)
        fix = return_value(k,h,fi)
        g = Vector{typeof(fix)}(undef,param_length(f))
        for i in param_eachindex(f)
          g[i] = return_value(k,h,param_getindex(f,i))
        end
        GenericParamBlock(g)
      end

      function Arrays.return_cache(k::Broadcasting{<:$F},h::$T,f::ParamBlock)
        fi = testitem(f)
        li = return_cache(k,h,fi)
        fix = evaluate!(li,k,h,fi)
        l = Vector{typeof(li)}(undef,param_length(f))
        g = Vector{typeof(fix)}(undef,param_length(f))
        for i in param_eachindex(f)
          l[i] = return_cache(k,h,param_getindex(f,i))
        end
        GenericParamBlock(g),l
      end

      function Arrays.evaluate!(cache,k::Broadcasting{<:$F},h::$T,f::ParamBlock)
        g,l = cache
        for i in param_eachindex(f)
          g.data[i] = evaluate!(l[i],k,h,param_getindex(f,i))
        end
        g
      end
    end
  end
end

function Arrays.return_value(k::Broadcasting{<:Operation},h::ParamBlock,f::ParamBlock)
  @check param_length(h) == param_length(f)
  hi = testitem(h)
  fi = testitem(f)
  fix = return_value(k,hi,fi)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = return_value(k,param_getindex(h,i),param_getindex(f,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},h::ParamBlock,f::ParamBlock)
  @check param_length(h) == param_length(f)
  hi = testitem(h)
  fi = testitem(f)
  li = return_cache(k,hi,fi)
  fix = evaluate!(li,k,hi,fi)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(k,param_getindex(h,i),param_getindex(f,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},h::ParamBlock,f::ParamBlock)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],k,param_getindex(h,i),param_getindex(f,i))
  end
  g
end

const ParamOperation = Operation{<:AbstractParamFunction}

param_length(f::Broadcasting{<:ParamOperation}) = param_length(f.f)
param_getindex(f::Broadcasting{<:ParamOperation},i::Int) = Broadcasting(param_getindex(f.f,i))
Arrays.testitem(f::Broadcasting{<:ParamOperation}) = param_getindex(f,1)

function Arrays.return_value(k::Broadcasting{<:ParamOperation},f::ParamBlock,h::Field)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  fix = return_value(ki,fi,h)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = return_value(param_getindex(k,i),param_getindex(f,i),h)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},f::ParamBlock,h::Field)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  li = return_cache(ki,fi,h)
  fix = evaluate!(li,ki,fi,h)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(param_getindex(k,i),param_getindex(f,i),h)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ParamOperation},f::ParamBlock,h::Field)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],param_getindex(k,i),param_getindex(f,i),h)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ParamOperation},h::Field,f::ParamBlock)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  fix = return_value(ki,h,fi)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = return_value(param_getindex(k,i),h,param_getindex(f,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},h::Field,f::ParamBlock)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  li = return_cache(ki,h,fi)
  fix = evaluate!(li,ki,h,fi)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(param_getindex(k,i),h,param_getindex(f,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ParamOperation},h::Field,f::ParamBlock)
  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],param_getindex(k,i),h,param_getindex(f,i))
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ParamOperation},h::ParamBlock,f::ParamBlock)
  @check param_length(k) == param_length(h) == param_length(f)
  ki = testitem(k)
  hi = testitem(h)
  fi = testitem(f)
  fix = return_value(ki,hi,fi)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    g[i] = return_value(param_getindex(k,i),param_getindex(h,i),param_getindex(f,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ParamOperation},h::ParamBlock,f::ParamBlock)
  @check param_length(k) == param_length(h) == param_length(f)
  ki = testitem(k)
  hi = testitem(h)
  fi = testitem(f)
  li = return_cache(ki,hi,fi)
  fix = evaluate!(li,ki,hi,fi)
  l = Vector{typeof(li)}(undef,param_length(f))
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(param_getindex(k,i),param_getindex(h,i),param_getindex(f,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:ParamOperation},
  h::ParamBlock,
  f::ParamBlock
  )

  g,l = cache
  for i in param_eachindex(f)
    g.data[i] = evaluate!(l[i],param_getindex(k,i),param_getindex(h,i),param_getindex(f,i))
  end
  g
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::ParamBlock)
  fi = testitem(f)
  fix = return_value(k,fi)
  h = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    h[i] = return_value(k,param_getindex(f,i))
  end
  GenericParamBlock(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::ParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = Vector{typeof(li)}(undef,param_length(f))
  h = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(k,param_getindex(f,i))
  end
  GenericParamBlock(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::ParamBlock)
  h,l = cache
  for i in param_eachindex(f)
    v = evaluate!(l[i],k,param_getindex(f,i))
    h.data[i] = v
  end
  h
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::ParamBlock,g::AbstractArray)
  fi = testitem(f)
  fix = return_value(k,fi,g)
  h = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    h[i] = return_value(k,param_getindex(f,i),g)
  end
  GenericParamBlock(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::ParamBlock,g::AbstractArray)
  fi = testitem(f)
  li = return_cache(k,fi,g)
  fix = evaluate!(li,k,fi,g)
  l = Vector{typeof(li)}(undef,param_length(f))
  h = Vector{typeof(fix)}(undef,param_length(f))
  for i in param_eachindex(f)
    l[i] = return_cache(k,param_getindex(f,i),g)
  end
  GenericParamBlock(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::ParamBlock,g::AbstractArray)
  h,l = cache
  for i in param_eachindex(f)
    h.data[i] = evaluate!(l[i],k,param_getindex(f,i),g)
  end
  h
end

for op in (:+,:-,:*)
  @eval begin

    function Arrays.return_value(k::Broadcasting{typeof($op)},f::ParamBlock,g::ParamBlock)
      return_value(BroadcastingFieldOpMap($op),f,g)
    end

    function Arrays.return_cache(k::Broadcasting{typeof($op)},f::ParamBlock,g::ParamBlock)
      return_cache(BroadcastingFieldOpMap($op),f,g)
    end

    function Arrays.evaluate!(cache,k::Broadcasting{typeof($op)},f::ParamBlock,g::ParamBlock)
      evaluate!(cache,BroadcastingFieldOpMap($op),f,g)
    end

  end
end

function Arrays.return_value(k::Broadcasting{typeof(*)},f::Number,g::TrivialParamBlock)
  h = return_value(k,f,g.data)
  TrivialParamBlock(h,g.plength)
end

function Arrays.return_cache(k::Broadcasting{typeof(*)},f::Number,g::TrivialParamBlock)
  c = return_cache(k,f,g.data)
  h = evaluate!(c,k,f,g.data)
  TrivialParamBlock(h,g.plength),c
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(*)},f::Number,g::TrivialParamBlock)
  r,c = cache
  v = evaluate!(c,k,f,g.data)
  copyto!(r.data,v)
  r
end

function Arrays.return_value(k::Broadcasting{typeof(*)},f::Number,g::ParamBlock)
  gi = testitem(g)
  hi = return_value(k,f,gi)
  data = Vector{typeof(hi)}(undef,param_length(g))
  for i in param_eachindex(g)
    data[i] = return_value(k,f,param_getindex(g,i))
  end
  GenericParamBlock(data)
end

function Arrays.return_cache(k::Broadcasting{typeof(*)},f::Number,g::ParamBlock)
  gi = testitem(g)
  ci = return_cache(k,f,gi)
  hi = evaluate!(ci,k,f,gi)
  data = Vector{typeof(hi)}(undef,param_length(g))
  c = Vector{typeof(ci)}(undef,param_length(g))
  for i in param_eachindex(g)
    c[i] = return_cache(k,f,param_getindex(g,i))
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(*)},f::Number,g::ParamBlock)
  r,c = cache
  for i in param_eachindex(g)
    r.data[i] = evaluate!(c[i],k,f,param_getindex(g,i))
  end
  r
end

function Arrays.return_value(k::Broadcasting{typeof(*)},f::ParamBlock,g::Number)
  evaluate(k,f,g)
end

function Arrays.return_cache(k::Broadcasting{typeof(*)},f::ParamBlock,g::Number)
  return_cache(k,g,f)
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(*)},f::ParamBlock,g::Number)
  evaluate!(cache,k,g,f)
end

function Arrays.return_value(k::BroadcastingFieldOpMap,a::ParamBlock...)
  evaluate(k,a...)
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::ParamBlock,g::ParamBlock)
  @check param_length(f) == param_length(g)
  fi = testitem(f)
  gi = testitem(g)
  hi = return_value(k,fi,gi)
  h = Vector{typeof(hi)}(undef,param_length(f))
  for i in param_eachindex(f)
    h[i] = return_value(k,param_getindex(f,i),param_getindex(g,i))
  end
  GenericParamBlock(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::ParamBlock,g::ParamBlock)
  @check param_length(f) == param_length(g)
  fi = testitem(f)
  gi = testitem(g)
  ci = return_cache(k,fi,gi)
  hi = evaluate!(ci,k,fi,gi)
  a = Vector{typeof(hi)}(undef,param_length(f))
  b = Vector{typeof(ci)}(undef,param_length(f))
  for i in param_eachindex(f)
    b[i] = return_cache(k,param_getindex(f,i),param_getindex(g,i))
  end
  GenericParamBlock(a),b
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::ParamBlock,g::ParamBlock)
  @check param_length(f) == param_length(g)
  a,b = cache
  for i in param_eachindex(f)
    a.data[i] = evaluate!(b[i],k,param_getindex(f,i),param_getindex(g,i))
  end
  a
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,a::ParamBlock...)
  a1 = first(a)
  @check all(ai->param_length(ai)==param_length(a1),a)
  ais = map(testitem,a)
  ci = return_cache(k,ais...)
  bi = evaluate!(ci,k,ais...)
  c = Vector{typeof(ci)}(undef,param_length(a1))
  data = Vector{typeof(bi)}(undef,param_length(a1))
  for i in param_eachindex(a1)
    _ais = map(ai->param_getindex(ai,i),a)
    c[i] = return_cache(k,_ais...)
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::ParamBlock...)
  a1 = first(a)
  @check all(ai->param_length(ai)==param_length(a1),a)
  r,c = cache
  for i in param_eachindex(a1)
    ais = map(ai->param_getindex(ai,i),a)
    r.data[i] = evaluate!(c[i],k,ais...)
  end
  r
end

function Arrays.return_value(
  k::BroadcastingFieldOpMap,a::Union{ParamBlock,AbstractArray}...
  )

  return_value(k,lazy_parameterise(a...)...)
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,a::Union{ParamBlock,AbstractArray}...
  )

  return_cache(k,lazy_parameterise(a...)...)
end

function Arrays.evaluate!(
  cache,k::BroadcastingFieldOpMap,a::Union{ParamBlock,AbstractArray}...
  )

  evaluate!(cache,k,lazy_parameterise(a...)...)
end

const ParamBroadcastingFieldOpMap{F<:AbstractParamFunction} = BroadcastingFieldOpMap{F}

param_length(f::ParamBroadcastingFieldOpMap) = param_length(f.op)
param_getindex(f::ParamBroadcastingFieldOpMap,i::Int) = BroadcastingFieldOpMap(param_getindex(f.op,i))
Arrays.testitem(f::ParamBroadcastingFieldOpMap) = param_getindex(f,1)

function Arrays.return_value(k::ParamBroadcastingFieldOpMap,f::ParamBlock)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  hi = return_value(ki,fi)
  h = Vector{typeof(hi)}(undef,param_length(f))
  for i in param_eachindex(f)
    h[i] = return_value(param_getindex(k,i),param_getindex(f,i))
  end
  GenericParamBlock(h)
end

function Arrays.return_cache(k::ParamBroadcastingFieldOpMap,f::ParamBlock)
  @check param_length(k) == param_length(f)
  ki = testitem(k)
  fi = testitem(f)
  ci = return_cache(ki,fi)
  hi = evaluate!(ci,ki,fi)
  a = Vector{typeof(hi)}(undef,param_length(f))
  b = Vector{typeof(ci)}(undef,param_length(f))
  for i in param_eachindex(f)
    b[i] = return_cache(param_getindex(k,i),param_getindex(f,i))
  end
  GenericParamBlock(a),b
end

function Arrays.evaluate!(cache,k::ParamBroadcastingFieldOpMap,f::ParamBlock)
  @check param_length(k) == param_length(f)
  a,b = cache
  for i in param_eachindex(f)
    v = evaluate!(b[i],param_getindex(k,i),param_getindex(f,i))
    a.data[i] = v
  end
  a
end

function Arrays.return_value(k::ParamBroadcastingFieldOpMap,f::ParamBlock,g::ParamBlock)
  @check param_length(k) == param_length(f) == param_length(g)
  ki = testitem(k)
  fi = testitem(f)
  gi = testitem(g)
  hi = return_value(ki,fi,gi)
  h = Vector{typeof(hi)}(undef,param_length(f))
  for i in param_eachindex(f)
    h[i] = return_value(param_getindex(k,i),param_getindex(f,i),param_getindex(g,i))
  end
  GenericParamBlock(h)
end

function Arrays.return_cache(k::ParamBroadcastingFieldOpMap,f::ParamBlock,g::ParamBlock)
  @check param_length(k) == param_length(f) == param_length(g)
  ki = testitem(k)
  fi = testitem(f)
  gi = testitem(g)
  ci = return_cache(ki,fi,gi)
  hi = evaluate!(ci,ki,fi,gi)
  a = Vector{typeof(hi)}(undef,param_length(f))
  b = Vector{typeof(ci)}(undef,param_length(f))
  for i in param_eachindex(f)
    b[i] = return_cache(param_getindex(k,i),param_getindex(f,i),param_getindex(g,i))
  end
  GenericParamBlock(a),b
end

function Arrays.evaluate!(cache,k::ParamBroadcastingFieldOpMap,f::ParamBlock,g::ParamBlock)
  @check param_length(k) == param_length(f) == param_length(g)
  a,b = cache
  for i in param_eachindex(f)
    v = evaluate!(b[i],param_getindex(k,i),param_getindex(f,i),param_getindex(g,i))
    a.data[i] = v
  end
  a
end

function Arrays.return_cache(k::ParamBroadcastingFieldOpMap,a::ParamBlock...)
  @check all(ai->param_length(ai)==param_length(k),a)
  ais = map(testitem,a)
  ci = return_cache(k,ais...)
  bi = evaluate!(ci,k,ais...)
  c = Vector{typeof(ci)}(undef,param_length(k))
  data = Vector{typeof(bi)}(undef,param_length(k))
  for i in param_eachindex(k)
    _ais = map(ai->param_getindex(ai,i),a)
    c[i] = return_cache(param_getindex(k,i),_ais...)
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,k::ParamBroadcastingFieldOpMap,a::ParamBlock...)
  @check all(ai->param_length(ai)==param_length(k),a)
  r,c = cache
  for i in param_eachindex(k)
    ais = map(ai->param_getindex(ai,i),a)
    v = evaluate!(c[i],param_getindex(k,i),ais...)
    r.data[i] = v
  end
  r
end

function Arrays.return_value(
  k::ParamBroadcastingFieldOpMap,a::Union{ParamBlock,AbstractArray}...
  )
  
  return_value(k,lazy_parameterise(a...;plength=param_length(k))...)
end

function Arrays.return_cache(
  k::ParamBroadcastingFieldOpMap,a::Union{ParamBlock,AbstractArray}...
  )

  return_cache(k,lazy_parameterise(a...;plength=param_length(k))...)
end

function Arrays.evaluate!(
  cache,k::ParamBroadcastingFieldOpMap,a::Union{ParamBlock,AbstractArray}...
  )

  evaluate!(cache,k,lazy_parameterise(a...;plength=param_length(k))...)
end

for op in (:+,:-)
  @eval begin
    function $op(a::ParamBlock,b::ParamBlock)
      BroadcastingFieldOpMap($op)(a,b)
    end

    function $op(a::TrivialParamBlock,b::TrivialParamBlock)
      @check size(a) == size(b)
      c = $op(a.data,b.data)
      TrivialParamBlock(c,a.plength)
    end
  end
end

for T in (:Number,:AbstractArray)
  @eval begin
    function Base.:*(a::$T,b::ParamBlock)
      bi = testitem(b)
      ci = a*bi
      data = Vector{typeof(ci)}(undef,param_length(b))
      for i in param_eachindex(b)
        data[i] = a*param_getindex(b,i)
      end
      GenericParamBlock(data)
    end

    function Base.:*(a::ParamBlock,b::$T)
      ai = testitem(a)
      ci = ai*b
      data = Vector{typeof(ci)}(undef,param_length(a))
      for i in param_eachindex(a)
        data[i] = param_getindex(a,i)*b
      end
      GenericParamBlock(data)
    end

    function Base.:*(a::$T,b::TrivialParamBlock)
      TrivialParamBlock(a*b.data,b.plength)
    end

    function Base.:*(a::TrivialParamBlock,b::$T)
      TrivialParamBlock(a.data*b,a.plength)
    end
  end
end

function Base.:*(a::TrivialParamBlock,b::TrivialParamBlock)
  @check param_length(a) == param_length(b)
  c = a.data*b.data
  TrivialParamBlock(c,a.plength)
end

function LinearAlgebra.mul!(c::TrivialParamBlock,a::TrivialParamBlock,b::TrivialParamBlock)
  mul!(c.data,a.data,b.data,1,0)
end

function LinearAlgebra.rmul!(a::TrivialParamBlock,β)
  rmul!(a.data,β)
end

function Base.:*(a::ParamBlock,b::ParamBlock)
  @check param_length(a) == param_length(b)
  ai = testitem(a)
  bi = testitem(b)
  ri = ai*bi
  data = Vector{typeof(ri)}(undef,param_length(a))
  data[1] = ri
  for i in 2:param_length(a)
    data[i] = param_getindex(a,i)*param_getindex(b,i)
  end
  GenericParamBlock(data)
end

function LinearAlgebra.rmul!(a::ParamBlock,β)
  for i in param_eachindex(a)
    rmul!(param_getindex(a,i),β)
  end
end

function LinearAlgebra.mul!(
  c::ParamBlock,
  a::ParamBlock,
  b::AbstractArray,
  α::Number,β::Number
  )

  for i in param_eachindex(c)
    mul!(param_getindex(c,i),param_getindex(a,i),b,α,β)
  end
end

function LinearAlgebra.mul!(
  c::ParamBlock,
  a::AbstractArray,
  b::ParamBlock,
  α::Number,β::Number
  )

  for i in param_eachindex(c)
    mul!(param_getindex(c,i),a,param_getindex(b,i),α,β)
  end
end

function LinearAlgebra.mul!(
  c::ParamBlock,
  a::ParamBlock,
  b::ParamBlock,
  α::Number,β::Number
  )

  for i in param_eachindex(c)
    mul!(param_getindex(c,i),param_getindex(a,i),param_getindex(b,i),α,β)
  end
end

function Arrays.setsize_op!(::typeof(copy),a::AbstractArray,b::ParamBlock)
  for i in param_eachindex(b)
    Arrays.setsize_op!(copy,a,param_getindex(b,i))
  end
end

function Arrays.setsize_op!(::typeof(copy),a::ParamBlock,b::AbstractArray)
  for i in param_eachindex(a)
    Arrays.setsize_op!(copy,param_getindex(a,i),b)
  end
end

function Arrays.setsize_op!(::typeof(copy),a::ParamBlock,b::ParamBlock)
  for i in param_eachindex(a)
    Arrays.setsize_op!(copy,param_getindex(a,i),param_getindex(b,i))
  end
end

function Arrays.setsize_op!(::typeof(*),c::ParamBlock,a::AbstractArray,b::ParamBlock)
  for i in param_eachindex(c)
    Arrays.setsize_op!(*,param_getindex(c,i),a,param_getindex(b,i))
  end
end

function Arrays.setsize_op!(::typeof(*),c::ParamBlock,a::ParamBlock,b::AbstractArray)
  for i in param_eachindex(c)
    Arrays.setsize_op!(*,param_getindex(c,i),param_getindex(a,i),b)
  end
end

function Arrays.setsize_op!(::typeof(*),c::ParamBlock,a::ParamBlock,b::ParamBlock)
  for i in param_eachindex(c)
    Arrays.setsize_op!(*,param_getindex(c,i),param_getindex(a,i),param_getindex(b,i))
  end
end

function Arrays.return_value(::typeof(*),a::AbstractArray,b::ParamBlock)
  ai = testitem(a)
  ri = return_value(*,ai,b)
  data = Vector{typeof(ri)}(undef,param_length(b))
  fill!(data,ri)
  GenericParamBlock(data)
end

function Arrays.return_cache(::typeof(*),a::AbstractArray,b::ParamBlock)
  c1 = CachedArray(a*b)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::AbstractArray,b::ParamBlock)
  c1,c2 = cache
  Arrays.setsize_op!(*,c1,a,b)
  c = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Arrays.return_value(::typeof(*),a::ParamBlock,b::AbstractArray)
  ai = testitem(a)
  bi = testitem(b)
  ri = return_value(*,ai,bi)
  data = Vector{typeof(ri)}(undef,param_length(a))
  fill!(data,ri)
  GenericParamBlock(data)
end

function Arrays.return_cache(::typeof(*),a::ParamBlock,b::AbstractArray)
  c1 = CachedArray(a*b)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::ParamBlock,b::AbstractArray)
  c1,c2 = cache
  Arrays.setsize_op!(*,c1,a,b)
  c = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Arrays.return_value(::typeof(*),a::ParamBlock,b::ParamBlock)
  @check param_length(a) == param_length(b)
  ai = testitem(a)
  bi = testitem(b)
  ri = return_value(*,ai,bi)
  data = Vector{typeof(ri)}(undef,param_length(a))
  fill!(data,ri)
  GenericParamBlock(data)
end

function Arrays.return_cache(::typeof(*),a::ParamBlock,b::ParamBlock)
  c1 = CachedArray(a*b)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::ParamBlock,b::ParamBlock)
  c1,c2 = cache
  Arrays.setsize_op!(*,c1,a,b)
  c = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Arrays.return_value(k::MulAddMap,a,b::ParamBlock,c::ParamBlock)
  x = return_value(*,a,b)
  return_value(+,x,c)
end

function Arrays.return_cache(k::MulAddMap,a,b::ParamBlock,c::ParamBlock)
  c1 = CachedArray(a*b+c)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,k::MulAddMap,a,b::ParamBlock,c::ParamBlock)
  c1,c2 = cache
  Arrays.setsize_op!(copy,c1,c)
  Arrays.setsize_op!(*,c1,a,b)
  d = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  copyto!(d,c)
  iszero(k.α) && isone(k.β) && return d
  mul!(d,a,b,k.α,k.β)
  d
end

# Autodiff related

for f in (:(ForwardDiff.gradient),:(ForwardDiff.jacobian))
  @eval begin
    function Arrays.return_cache(k::Arrays.ConfigMap{typeof($f)},x::ParamBlock)
      xi = testitem(x)
      fi = return_cache(k,xi)
      data = Vector{typeof(fi)}(undef,param_length(x))
      for i in param_eachindex(x)
        data[i] = return_cache(k,param_getindex(x,i))
      end
      GenericParamBlock(data)
    end

    function Arrays.return_cache(k::Arrays.ConfigMap{typeof($f)},x::VectorBlock{<:ParamBlock})
      return BlockParamConfig($f,k.tag,x)
    end
  end
end

for F in (:(ForwardDiff.GradientConfig),:(ForwardDiff.JacobianConfig))
  @eval begin
    function Arrays.testitem(a::LazyArray{A,<:ParamBlock{<:T}} where A) where {Tag,V,N,D,T<:$F{Tag,V,N,D}}
      if length(a) > 0
        first(a)::T
      else
        gi = testitem(a.maps)
        ai = map(testitem,a.args)
        plength = find_param_length(ai...)
        x0 = zeros(V,N)
        x0p = lazy_parameterise(x0,plength)
        return_cache(gi,x0p)
      end::T
    end

    function Arrays.return_value(k::DualizeMap,cfg::ParamBlock{<:$F},x::ParamBlock)
      vi = return_value(k,testitem(cfg),testitem(x))
      v = Vector{typeof(vi)}(undef,param_length(x))
      fill!(v,vi)
      GenericParamBlock(v)
    end
  end
end

function Arrays.evaluate!(cache,k::DualizeMap,cfg::ParamBlock,x::ParamBlock)
  for i in param_eachindex(x)
    evaluate!(nothing,k,param_getindex(cfg,i),param_getindex(x,i))
  end
end

function Arrays.return_cache(k::Arrays.AutoDiffMap,cfg::ParamBlock,ydual::ParamBlock)
  ci = return_cache(k,testitem(cfg),testitem(ydual))
  ri = evaluate!(ci,k,testitem(cfg),testitem(ydual))
  c = Vector{typeof(ci)}(undef,param_length(ydual))
  data = Vector{typeof(ri)}(undef,param_length(ydual))
  for i in param_eachindex(ydual)
    c[i] = return_cache(k,param_getindex(cfg,i),param_getindex(ydual,i))
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(
  cache,
  k::Arrays.AutoDiffMap,
  cfg::ParamBlock,
  ydual::ParamBlock
  )

  r,c = cache
  for i in param_eachindex(ydual)
    r.data[i] = evaluate!(c[i],k,param_getindex(cfg,i),param_getindex(ydual,i))
  end
  r
end

struct BlockParamConfig{C,T,V,N,D,O} <: ForwardDiff.AbstractConfig{N}
  seeds::NTuple{N,ForwardDiff.Partials{N,V}}
  duals::D
  offsets::O
  
  function BlockParamConfig(
    ::C,
    f::F,
    x::VectorBlock{<:ParamBlock{<:AbstractArray{V}}},
    ::T = ForwardDiff.Tag(f,V)
    ) where {C,F,V,T}

    offsets,N = Arrays.block_offsets(x,0)
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N,V})
    duals = similar(x,ForwardDiff.Dual{T,V,N})
    D = typeof(duals)
    O = typeof(offsets)
    new{C,T,V,N,D,O}(seeds,duals,offsets)
  end

  function BlockParamConfig(
    ::C,
    f::F,
    x::VectorBlock{<:VectorBlock{<:ParamBlock{<:AbstractArray{V}}}},
    ::T = ForwardDiff.Tag(f,V)
    ) where {C,F,V,T}

    offsets,N = Arrays.block_offsets(x,0)
    seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N,V})
    duals = similar(x,ForwardDiff.Dual{T,V,N})
    D = typeof(duals)
    O = typeof(offsets)
    new{C,T,V,N,D,O}(seeds,duals,offsets)
  end
end

function Arrays.evaluate!(cache,k::DualizeMap,cfg::BlockParamConfig,x)
  xdual,seeds,offsets = cfg.duals,cfg.seeds,cfg.offsets
  Arrays.seed_block!(xdual,x,seeds,offsets)
  return xdual
end

function Arrays.return_cache(
  ::AutoDiffMap,
  cfg::BlockParamConfig{typeof(ForwardDiff.gradient),T},
  ydual
  ) where T

  ydual isa Real || throw(ForwardDiff.GRAD_ERROR)
  result = CachedArray(similar(cfg.duals,ForwardDiff.valtype(ydual)))
  return result
end

function Arrays.evaluate!(
  result,::AutoDiffMap,
  cfg::BlockParamConfig{typeof(ForwardDiff.gradient),T},
  ydual
  ) where T

  Arrays._setsize!(result,cfg.duals)
  Arrays.extract_gradient_block!(T,result,ydual,cfg.offsets)
  return result
end

function Arrays.return_cache(
  ::AutoDiffMap,
  cfg::BlockParamConfig{typeof(ForwardDiff.jacobian),T},
  ydual
  ) where T

  ydual isa VectorBlock || throw(ForwardDiff.JACOBIAN_ERROR)
  result = Arrays._alloc_jacobian(ydual,cfg.duals)
  return result
end

function Arrays.evaluate!(
  result,
  ::AutoDiffMap,
  cfg::BlockParamConfig{typeof(ForwardDiff.jacobian),T},
  ydual
  ) where T

  Arrays._setsize!(result,ydual)
  Arrays.extract_jacobian_block!(T,result,ydual,cfg.offsets)
  return result
end

function Arrays.return_cache(k::CellData.ZeroVectorMap,a::TrivialParamBlock)
  c = return_cache(k,a.data)
  data = evaluate!(ci,k,a.data)
  TrivialParamBlock(data,v.plength),c
end

function Arrays.evaluate!(cache,k::CellData.ZeroVectorMap,a::TrivialParamBlock)
  r,c = cache
  copyto!(r.data,evaluate!(c[i],k,a.data))
  r
end

function Arrays.return_cache(k::CellData.ZeroVectorMap,a::ParamBlock)
  ai = testitem(a)
  ci = return_cache(k,ai)
  vi = evaluate!(ci,k,ai)
  c = Vector{typeof(ci)}(undef,param_length(a))
  data = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    c[i] = return_cache(k,param_getindex(a,i))
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,k::CellData.ZeroVectorMap,a::ParamBlock)
  r,c = cache
  for i in param_eachindex(a)
    r.data[i] = evaluate!(c[i],k,param_getindex(a,i))
  end
  r
end

# cell datas

function Geometry._cache_compress(a::ParamBlock)
  c1 = CachedArray(a)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  c1,c2
end

function Geometry._setempty_compress!(a::TrivialParamBlock)
  Geometry._setempty_compress!(a.data)
end

function Geometry._setempty_compress!(a::ParamBlock)
  for i in param_eachindex(a)
    Geometry._setempty_compress!(param_getindex(a,i))
  end
end

function Geometry._uncached_compress!(c1::ParamBlock,c2)
  evaluate!(c2,Arrays.unwrap_cached_array,c1)
end

function Geometry._setsize_compress!(a::TrivialParamBlock,b::TrivialParamBlock)
  Geometry._setsize_compress!(a.data,b.data)
end

function Geometry._setsize_compress!(a::ParamBlock,b::ParamBlock)
  @check size(a) == size(b)
  for i in param_eachindex(a)
    Geometry._setsize_compress!(param_getindex(a,i),param_getindex(b,i))
  end
end

function Geometry._copyto_compress!(a::TrivialParamBlock,b::TrivialParamBlock)
  Geometry._copyto_compress!(a.data,b.data)
end

function Geometry._copyto_compress!(a::ParamBlock,b::ParamBlock)
  @check size(a) == size(b)
  for i in param_eachindex(a)
    Geometry._copyto_compress!(param_getindex(a,i),param_getindex(b,i))
  end
end

function Geometry._addto_compress!(a::TrivialParamBlock,b::TrivialParamBlock)
  Geometry._addto_compress!(a.data,b.data)
end

function Geometry._addto_compress!(a::ParamBlock,b::ParamBlock)
  @check size(a) == size(b)
  for i in param_eachindex(a)
    Geometry._addto_compress!(param_getindex(a,i),param_getindex(b,i))
  end
end

function Geometry._similar_empty(val::TrivialParamBlock)
  TrivialParamBlock(Geometry._similar_empty(val.data),val.plength)
end

function Geometry._similar_empty(val::ParamBlock)
  a = deepcopy(val)
  for i in param_eachindex(a)
    a.data[i] = Geometry._similar_empty(param_getindex(val,i))
  end
  a
end

function Geometry.pos_neg_data(
  ipos_to_val::AbstractArray{<:ParamBlock},i_to_iposneg::PosNegPartition
  )

  nineg = length(i_to_iposneg.ineg_to_i)
  val = testitem(ipos_to_val)
  void = Geometry._similar_empty(val)
  ineg_to_val = Fill(void,nineg)
  ipos_to_val,ineg_to_val
end

function Geometry.pos_neg_data(
  ipos_to_val::AbstractArray{<:ParamBlock{<:Field}},i_to_iposneg::PosNegPartition
  )

  nineg = length(i_to_iposneg.ineg_to_i)
  ipos_to_v = lazy_map(VoidFieldMap(false),ipos_to_val)
  ineg_to_v = Fill(VoidField(testitem(ipos_to_val),true),nineg)
  ipos_to_v,ineg_to_v
end

# reference FEs

function Arrays.return_cache(b::LagrangianDofBasis,f::ParamBlock)
  fi = testitem(f)
  ci = return_cache(b,fi)
  ri = evaluate!(ci,b,fi)
  c = Vector{typeof(ci)}(undef,param_length(f))
  data = Vector{typeof(ri)}(undef,param_length(f))
  for i in param_eachindex(f)
    c[i] = return_cache(b,param_getindex(f,i))
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(cache,b::LagrangianDofBasis,f::ParamBlock)
  r,c = cache
  for i in param_eachindex(f)
    r.data[i] = evaluate!(c[i],b,param_getindex(f,i))
  end
  r
end

# array block interface

function Arrays.return_value(
  k::BroadcastingFieldOpMap,
  f::ArrayBlock{A,N},
  h::ParamBlock
  ) where {A,N}

  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = Array{typeof(fix),N}(undef,size(f.array))
  for i in eachindex(f.array)
    if f.touched[i]
      g[i] = return_value(k,f.array[i],h)
    end
  end
  ArrayBlock(g,f.touched)
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,
  f::ArrayBlock{A,N},
  h::ParamBlock
  ) where {A,N}

  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = Array{typeof(li),N}(undef,size(f.array))
  g = Array{typeof(fix),N}(undef,size(f.array))
  for i in eachindex(f.array)
    if f.touched[i]
      l[i] = return_cache(k,f.array[i],h)
    end
  end
  ArrayBlock(g,f.touched),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::ArrayBlock,h::ParamBlock)
  g,l = cache
  @check g.touched == f.touched
  for i in eachindex(f.array)
    if f.touched[i]
      g.array[i] = evaluate!(l[i],k,f.array[i],h)
    end
  end
  g
end

function Arrays.return_value(
  k::BroadcastingFieldOpMap,
  h::ParamBlock,
  f::ArrayBlock{A,N}
  ) where {A,N}

  fi = testitem(f)
  fix = return_value(k,h,fi)
  g = Array{typeof(fix),N}(undef,size(f.array))
  for i in eachindex(f.array)
    if f.touched[i]
      g[i] = return_value(k,h,f.array[i])
    end
  end
  ArrayBlock(g,f.touched)
end

function Arrays.return_cache(
  k::BroadcastingFieldOpMap,
  h::ParamBlock,
  f::ArrayBlock{A,N}
  ) where {A,N}

  fi = testitem(f)
  li = return_cache(k,h,fi)
  fix = evaluate!(li,k,h,fi)
  l = Array{typeof(li),N}(undef,size(f.array))
  g = Array{typeof(fix),N}(undef,size(f.array))
  for i in eachindex(f.array)
    if f.touched[i]
      l[i] = return_cache(k,h,f.array[i])
    end
  end
  ArrayBlock(g,f.touched),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,h::ParamBlock,f::ArrayBlock)
  g,l = cache
  @check g.touched == f.touched
  for i in eachindex(f.array)
    if f.touched[i]
      g.array[i] = evaluate!(l[i],k,h,f.array[i])
    end
  end
  g
end

#TODO this fix should go in Gridap 
for S in (:ParamBlock,:AbstractArray,:ArrayBlock)
  for T in (:ParamBlock,:AbstractArray,:ArrayBlock)
    @eval begin
      function Arrays.return_value(
        k::BroadcastingFieldOpMap,
        a::ArrayBlock{<:$S,N},
        b::ArrayBlock{<:$T,N}
        ) where N
        
        evaluate(k,a,b)
      end
    end
  end
end

for S in (:ParamBlock,:AbstractArray)
  for T in (:ParamBlock,:AbstractArray)
    (S == :AbstractArray && T == :AbstractArray) && continue
    for U in (S,:(ArrayBlock{<:$S})), V in (T,:(ArrayBlock{<:$T}))
      @eval begin
        function Arrays.return_cache(
          k::BroadcastingFieldOpMap,
          f::ArrayBlock{<:$U,N},
          g::ArrayBlock{<:$V,N}
          ) where N

          @notimplementedif size(f) != size(g)
          fi,gi = _test_item_values(f,g)
          ci = return_cache(k,fi,gi)
          hi = evaluate!(ci,k,fi,gi)
          m = Fields.ZeroBlockMap()
          a = Array{typeof(hi),N}(undef,size(f.array))
          b = Array{typeof(ci),N}(undef,size(f.array))
          zf = Array{typeof(return_cache(m,fi,gi))}(undef,size(f.array))
          zg = Array{typeof(return_cache(m,gi,fi))}(undef,size(f.array))
          t = map(|,f.touched,g.touched)
          for i in eachindex(f.array)
            if f.touched[i] && g.touched[i]
              b[i] = return_cache(k,f.array[i],g.array[i])
            elseif f.touched[i]
              _fi = f.array[i]
              zg[i] = return_cache(m,gi,_fi)
              _gi = evaluate!(zg[i],m,gi,_fi)
              b[i] = return_cache(k,_fi,_gi)
            elseif g.touched[i]
              _gi = g.array[i]
              zf[i] = return_cache(m,fi,_gi)
              _fi = evaluate!(zf[i],m,fi,_gi)
              b[i] = return_cache(k,_fi,_gi)
            end
          end
          ArrayBlock(a,t),b,zf,zg
        end
      end
    end
  end
end

for A in (:ArrayBlock,:ParamBlock)
  for B in (:ArrayBlock,:ParamBlock)
    for C in (:ArrayBlock,:ParamBlock)
      if !(A == B == C)
        @eval begin
          function Arrays.return_value(k::BroadcastingFieldOpMap,a::$A,b::$B,c::$C)
            evaluate(k,a,b,c)
          end

          function Arrays.return_cache(k::BroadcastingFieldOpMap,a::$A,b::$B,c::$C)
            tup = (a,b,c)
            i = findfirst(Base.Fix2(isa,ArrayBlock),tup)
            @notimplementedif isnothing(i)
            m = Arrays.MatchingBlockMap(tup[i])
            ca = return_cache(m,a)
            cb = return_cache(m,b)
            cc = return_cache(m,c)
            ea = evaluate!(ca,m,a)
            eb = evaluate!(cb,m,b)
            ec = evaluate!(cc,m,c)
            ctup = return_cache(k,ea,eb,ec)
            return ctup,m,(ca,cb,cc)
          end

          function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,a::$A,b::$B,c::$C)
            ctup,m,(ca,cb,cc) = cache
            ea = evaluate!(ca,m,a)
            eb = evaluate!(cb,m,b)
            ec = evaluate!(cc,m,c)
            evaluate!(ctup,k,ea,eb,ec)
          end
        end
      end
      for D in (:ArrayBlock,:ParamBlock)
        if !(A == B == C == D)
          @eval begin
            function Arrays.return_value(k::BroadcastingFieldOpMap,a::$A,b::$B,c::$C,d::$D)
              evaluate(k,a,b,c,d)
            end

            function Arrays.return_cache(k::BroadcastingFieldOpMap,a::$A,b::$B,c::$C,d::$D)
              tup = (a,b,c,d)
              i = findfirst(Base.Fix2(isa,ArrayBlock),tup)
              @notimplementedif isnothing(i)
              m = Arrays.MatchingBlockMap(tup[i])
              ca = return_cache(m,a)
              cb = return_cache(m,b)
              cc = return_cache(m,c)
              cd = return_cache(m,d)
              ea = evaluate!(ca,m,a)
              eb = evaluate!(cb,m,b)
              ec = evaluate!(cc,m,c)
              ed = evaluate!(cd,m,d)
              ctup = return_cache(k,ea,eb,ec,ed)
              return ctup,m,(ca,cb,cc,cd)
            end

            function Arrays.evaluate!(
              cache,
              k::BroadcastingFieldOpMap,
              a::$A,
              b::$B,
              c::$C,
              d::$D
              )

              ctup,m,(ca,cb,cc,cd) = cache
              ea = evaluate!(ca,m,a)
              eb = evaluate!(cb,m,b)
              ec = evaluate!(cc,m,c)
              ed = evaluate!(cd,m,d)
              evaluate!(ctup,k,ea,eb,ec,ed)
            end
          end
        end
      end
    end
  end
end

function Arrays.return_cache(k::Fields.ZeroBlockMap,h::ParamBlock,f::ParamBlock)
  @check param_length(h) == param_length(f)
  c = return_cache(k,testitem(h),testitem(f))
  lazy_parameterise(c,param_length(h))
end

function Arrays.evaluate!(c,k::Fields.ZeroBlockMap,h::ParamBlock,f::ParamBlock)
  for i in param_eachindex(h)
    evaluate!(param_getindex(c,i),k,param_getindex(h,i),param_getindex(f,i))
  end
  c
end

function Arrays.return_cache(k::Fields.ZeroBlockMap,h::ArrayBlock{<:ParamBlock},f::ArrayBlock) 
  N = ndims(f)
  hi = testitem(h)
  fi = testitem(f)
  ci = return_cache(k,hi,fi)
  vi = evaluate!(ci,k,hi,fi)
  array = Array{typeof(vi),N}(undef,size(f))
  for i in eachindex(array)
    if f.touched[i]
      array[i] = evaluate!(ci,k,hi,f.array[i])
    end
  end
  ArrayBlock(array,f.touched)
end

for T in (:AbstractArray,:Nothing)
  @eval begin
    function Arrays.return_cache(k::Fields.ZeroBlockMap,h::ParamBlock,f::$T)
      return_cache(k,h,lazy_parameterise(f,param_length(h)))
    end
    function Arrays.return_cache(k::Fields.ZeroBlockMap,h::$T,f::ParamBlock)
      return_cache(k,lazy_parameterise(h,param_length(f)),f)
    end
    function Arrays.evaluate!(cache,k::Fields.ZeroBlockMap,h::ParamBlock,f::$T)
      evaluate!(cache,k,h,lazy_parameterise(f,param_length(h)))
    end
    function Arrays.evaluate!(cache,k::Fields.ZeroBlockMap,h::$T,f::ParamBlock)
      evaluate!(cache,k,lazy_parameterise(h,param_length(f)),f)
    end
    function Arrays.evaluate!(cache::ParamBlock,k::Fields.ZeroBlockMap,h::$T,f::AbstractArray)
      plength = param_length(cache)
      evaluate!(cache,k,lazy_parameterise(h,plength),lazy_parameterise(f,plength))
    end
  end
end

# utils

function Fields.AffineField(gradients::ParamBlock,origins::ParamBlock)
  data = map(AffineField,get_param_data(gradients),get_param_data(origins))
  GenericParamBlock(data)
end

function Fields.VoidField(field::ParamBlock,isvoid::Bool)
  data = map(a -> VoidField(a,isvoid),get_param_data(field))
  GenericParamBlock(data)
end

function Arrays.return_value(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x
  )

  @check param_length(gradients) == param_length(origins)
  gi = testitem(gradients)
  oi = testitem(origins)
  vi = return_value(k,gi,oi,x)
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    g[i] = return_value(k,param_getindex(gradients,i),param_getindex(origins,i),x)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x
  )

  @check param_length(gradients) == param_length(origins)
  gi = testitem(gradients)
  oi = testitem(origins)
  li = return_cache(k,gi,oi,x)
  vi = evaluate!(li,k,gi,oi,x)
  l = Vector{typeof(li)}(undef,param_length(gradients))
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    l[i] = return_cache(k,param_getindex(gradients,i),param_getindex(origins,i),x)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(
  cache,k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x
  )

  @check param_length(gradients) == param_length(origins)
  g,l = cache
  for i in param_eachindex(gradients)
    g.data[i] = evaluate!(l[i],k,param_getindex(gradients,i),param_getindex(origins,i),x)
  end
  g
end

function Arrays.return_value(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x::ParamBlock
  )

  @check param_length(gradients) == param_length(origins) == param_length(x)
  gi = testitem(gradients)
  oi = testitem(origins)
  xi = testitem(x)
  vi = return_value(k,gi,oi,xi)
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    g[i] = return_value(k,param_getindex(gradients,i),param_getindex(origins,i),param_getindex(x,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x::ParamBlock
  )

  @check param_length(gradients) == param_length(origins) == param_length(x)
  gi = testitem(gradients)
  oi = testitem(origins)
  xi = testitem(x)
  li = return_cache(k,gi,oi,xi)
  vi = evaluate!(li,k,gi,oi,xi)
  l = Vector{typeof(li)}(undef,param_length(gradients))
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    l[i] = return_cache(k,param_getindex(gradients,i),param_getindex(origins,i),param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(
  cache,k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x::ParamBlock
  )

  @check param_length(gradients) == param_length(origins) == param_length(x)
  g,l = cache
  for i in param_eachindex(gradients)
    g.data[i] = evaluate!(l[i],k,param_getindex(gradients,i),param_getindex(origins,i),param_getindex(x,i))
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ConstantMap},a::ParamBlock,x)
  ai = testitem(a)
  vi = return_value(k,ai,x)
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    g[i] = return_value(k,param_getindex(a,i),x)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ConstantMap},a::ParamBlock,x)
  ai = testitem(a)
  li = return_cache(k,ai,x)
  vi = evaluate!(li,k,ai,x)
  l = Vector{typeof(li)}(undef,param_length(a))
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    l[i] = return_cache(k,param_getindex(a,i),x)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ConstantMap},a::ParamBlock,x)
  g,l = cache
  for i in param_eachindex(a)
    g.data[i] = evaluate!(l[i],k,param_getindex(a,i),x)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ConstantMap},a::ParamBlock,x::ParamBlock)
  @check param_length(a) == param_length(x)
  ai = testitem(a)
  xi = testitem(x)
  vi = return_value(k,ai,xi)
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    g[i] = return_value(k,param_getindex(a,i),param_getindex(x,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ConstantMap},a::ParamBlock,x::ParamBlock)
  @check param_length(a) == param_length(x)
  ai = testitem(a)
  xi = testitem(x)
  li = return_cache(k,ai,xi)
  vi = evaluate!(li,k,ai,xi)
  l = Vector{typeof(li)}(undef,param_length(a))
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    l[i] = return_cache(k,param_getindex(a,i),param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ConstantMap},a::ParamBlock,x::ParamBlock)
  @check param_length(a) == param_length(x)
  g,l = cache
  for i in param_eachindex(a)
    g.data[i] = evaluate!(l[i],k,param_getindex(a,i),param_getindex(x,i))
  end
  g
end

function Arrays.return_value(f::Field,x::ParamBlock)
  xi = testitem(x)
  vi = return_value(f,xi)
  g = Vector{typeof(vi)}(undef,param_length(x))
  for i in param_eachindex(x)
    g[i] = return_value(f,param_getindex(x,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(f::Field,x::ParamBlock)
  xi = testitem(x)
  li = return_cache(f,xi)
  vi = evaluate!(li,f,xi)
  l = Vector{typeof(li)}(undef,param_length(x))
  g = Vector{typeof(vi)}(undef,param_length(x))
  for i in param_eachindex(x)
    l[i] = return_cache(f,param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,f::Field,x::ParamBlock)
  g,l = cache
  for i in param_eachindex(x)
    g.data[i] = evaluate!(l[i],f,param_getindex(x,i))
  end
  g
end

for f in (:(Fields.GenericField),:(Fields.ZeroField),:(Fields.ConstantField),:(Fields.inverse_map))
  @eval begin
    $f(a::ParamBlock) = GenericParamBlock(map($f,get_param_data(a)))
  end
end

for op in (:(Fields.gradient),:(Fields.symmetric_gradient),:(Fields.divergence),
  :(Fields.curl),:(Fields.laplacian))
  @eval begin
    ($op)(a::ParamBlock) = GenericParamBlock(map($op,get_param_data(a)))
  end
end

# constructors

lazy_parameterise(a,plength::Integer) = parameterise(a,plength)

function lazy_parameterise(a...;plength=find_param_length(a...))
  pa = map(f->lazy_parameterise(f,plength),a)
  return pa
end

function lazy_parameterise(a::ParamBlock,plength::Integer=param_length(a))
  @check param_length(a) == plength
  a
end

for T in (:Nothing,:Number,:Field)
  @eval begin 
    function lazy_parameterise(a::Union{$T,AbstractArray{<:$T}},plength::Integer)
      TrivialParamBlock(a,plength)
    end

    function local_parameterise(a::Union{$T,AbstractArray{<:$T}},plength::Integer)
      data = Vector{typeof(a)}(undef,plength)
      @inbounds for i in 1:plength
        data[i] = copy(a)
      end
      GenericParamBlock(data)
    end
  end
end

function local_parameterise(a::AbstractArray{<:AbstractArray},plength::Integer)
  @check length(a) == plength
  GenericParamBlock(a)
end

function local_parameterise(a::ParamBlock,plength::Integer)
  @check param_length(a) == plength
  a
end

function Fields.GenericField(f::AbstractParamFunction)
  GenericParamBlock(map(i -> GenericField(f[i]),1:length(f)))
end

#TODO this fix should go in Gridap

for T in (:ParamBlock,:(ArrayBlock{<:ParamBlock}))
  @eval begin
    function CellData.add_contribution!(
      a::DomainContribution,
      trian::Triangulation,
      b::AbstractArray{<:$T},
      op=+
      )

      if haskey(a.dict,trian)
        isempty(b) && return a
        a.dict[trian] = lazy_map(Broadcasting(op),a.dict[trian],b)
      else
        if (op == +) || isempty(b)
         a.dict[trian] = b
        else
         a.dict[trian] = lazy_map(Broadcasting(op),b)
        end
      end
      a
    end
  end
end

# utils

_test_values(_h,_f) = @abstractmethod

function _test_values(h::ParamBlock,f::ParamBlock)
  @check param_length(h) == param_length(f)
  hi = testvalue(h)
  fi = testvalue(f)
  return hi,fi
end

function _test_values(h::ParamBlock,_f)
  f = lazy_parameterise(_f,param_length(h))
  _test_values(h,f)
end

function _test_values(_h,f::ParamBlock)
  h = lazy_parameterise(_h,param_length(f))
  _test_values(h,f)
end

function _test_values(h::ArrayBlock{A,N},f::ArrayBlock{B,N}) where {A,B,N}
  hi = testitem(h)
  fi = testitem(f)
  plength = find_param_length(hi,fi)
  _hi = _param_zero_like(hi,plength)
  _fi = _param_zero_like(fi,plength)
  _htv,_ftv = _test_values(_hi,_fi)
  bh = Array{typeof(_htv),N}(undef,size(h.array))
  bf = Array{typeof(_ftv),N}(undef,size(f.array))
  for i in eachindex(h.array)
    if h.touched[i] && f.touched[i]
      bh[i],bf[i] = _test_values(h.array[i],f.array[i])
    elseif h.touched[i]
      bh[i],bf[i] = _test_values(h.array[i],_fi)
    elseif f.touched[i]
      bh[i],bf[i] = _test_values(_hi,f.array[i])
    end
  end
  ArrayBlock(bh,h.touched),ArrayBlock(bf,f.touched)
end

function _test_values(h::ArrayBlock,f)
  @notimplemented
end

function _test_values(h,f::ArrayBlock)
  @notimplemented
end

function _test_item_values(h::ArrayBlock,f::ArrayBlock)
  _test_values(testitem(h),testitem(f))
end

function _param_zero_like(a,plength::Int)
  testvalue(a)
end

function _param_zero_like(a::ParamBlock,plength::Int)
  @check param_length(a) == plength
  a
end

function _param_zero_like(a::ArrayBlock{A,N},plength::Int) where {A,N}
  inner_zero = _param_zero_like(testitem(a),plength)
  array = Array{typeof(inner_zero),N}(undef,size(a.array))
  touched = fill(false,size(a.array))
  ArrayBlock(array,touched)
end

@inline function Arrays.block_offsets(x::ParamBlock,offset) 
  Arrays.block_offsets(testitem(x),offset)
end

function Arrays.seed_block!(
  duals::TrivialParamBlock,
  x::TrivialParamBlock, 
  seeds::NTuple,
  offset
  ) 

  @check param_length(duals) == param_length(x)
  Arrays.seed_block!(duals.data,x.data,seeds,offset)
  return duals
end

function Arrays.seed_block!(
  duals::ParamBlock,
  x::ParamBlock, 
  seeds::NTuple,
  offset
  ) 

  @check param_length(duals) == param_length(x)
  for i in param_eachindex(duals)
    Arrays.seed_block!(param_getindex(duals,i),param_getindex(x,i),seeds,offset)
  end
  return duals
end

for f in (:(Arrays.extract_gradient_block!),:(Arrays.extract_jacobian_block!))
  @eval begin
    function $f(
      ::Type{T}, 
      result::TrivialParamBlock, 
      dual::TrivialParamBlock, 
      offset
      ) where T

      @check param_length(dual) == param_length(result)
      $f(T,result.data,dual.data,offset)
      return result
    end

    function $f(
      ::Type{T}, 
      result::ParamBlock, 
      dual::ParamBlock, 
      offset
      ) where T

      @check param_length(dual) == param_length(result)
      for i in param_eachindex(dual)
        $f(T,param_getindex(result,i),param_getindex(dual,i),offset)
      end
      return result
    end
  end
end

function Arrays._alloc_jacobian(ydual::ParamBlock,xdual::ParamBlock)
  @check param_length(ydual) == param_length(xdual)
  ci = Arrays._alloc_jacobian(testitem(ydual),testitem(xdual))
  c = Vector{typeof(ci)}(undef,param_length(ydual))
  for i in param_eachindex(ydual)
    c[i] = Arrays._alloc_jacobian(param_getindex(ydual,i),param_getindex(xdual,i))
  end
  GenericParamBlock(c)
end

function Arrays._setsize!(result::VectorBlock{<:ParamBlock},duals::VectorBlock{<:ParamBlock})
  ni = size(result.array,1)
  for i in 1:ni
    if result.touched[i]
      for k in param_eachindex(duals[i])
        setsize!(param_getindex(result[i],k),(length(param_getindex(duals[i],k)),))
      end
    end
  end
end

function Arrays._setsize!(result::MatrixBlock{<:ParamBlock},ydual::VectorBlock{<:ParamBlock})
  ni,nj = size(result)
  for i in 1:ni
    for j in 1:nj
      if result.touched[i,j]
        for k in param_eachindex(ydual[i])
          setsize!(
            param_getindex(result[i,j],k),
            (length(param_getindex(ydual[i],k)),
            length(param_getindex(ydual[j],k)))
          )
        end
      end
    end
  end
end