"""
    struct VariableParamBlock{A,N} <: ParamBlock{A}
      data::Array{A,N}
    end

Like a [`GenericParamBlock`](@ref) but with variable parametric shape.
`N` encodes the interaction history:
- N=1 (vector): standalone or result of two `VariableParamBlock`s interacting
- N=2 (matrix): result of a `VariableParamBlock` interacting with a [`ParamBlock`](@ref)
  (tensor product: `data[i,j] = op(self[i], other[j])`)

When a `VariableParamBlock` is combined with a `ParamBlock` in any map, the
output length is `param_length(self) × param_length(other)` and the result is
stored as an `N=2` `VariableParamBlock`. When two `VariableParamBlock`s are
combined, the result is `N=1` (element-wise, equal lengths required).
"""
struct VariableParamBlock{A,N} <: ParamBlock{A}
  data::Array{A,N}
end

# ─── basic interface ──────────────────────────────────────────────────────────

Base.getindex(b::VariableParamBlock,i::Integer) = b.data[i]
Base.setindex!(b::VariableParamBlock,v,i::Integer) = (b.data[i] = v)

get_param_data(b::VariableParamBlock) = b.data
param_length(b::VariableParamBlock) = length(b.data)
param_getindex(b::VariableParamBlock,i::Integer) = b.data[i]
param_setindex!(b::VariableParamBlock,v,i::Integer) = (b.data[i] = v)

function get_param_entry!(v::AbstractVector,b::VariableParamBlock,i...)
  for k in eachindex(v)
    @inbounds v[k] = b.data[k][i...]
  end
  v
end

Base.copy(a::VariableParamBlock) = VariableParamBlock(copy(a.data))
Base.similar(a::VariableParamBlock) = VariableParamBlock(similar(a.data))

function Base.similar(a::VariableParamBlock,::Type{T′}) where T′
  VariableParamBlock(similar(a.data,T′))
end

function Base.copyto!(a::VariableParamBlock,b::VariableParamBlock)
  @check size(a.data) == size(b.data)
  for i in eachindex(a.data)
    fill!(a.data[i],zero(eltype(a.data[i])))
    copyto!(a.data[i],b.data[i])
  end
  a
end

function Base.:≈(a::VariableParamBlock,b::VariableParamBlock)
  size(a.data) != size(b.data) && return false
  for i in eachindex(a.data)
    !(a.data[i] ≈ b.data[i]) && return false
  end
  true
end

function Base.:(==)(a::VariableParamBlock,b::VariableParamBlock)
  size(a.data) != size(b.data) && return false
  for i in eachindex(a.data)
    a.data[i] != b.data[i] && return false
  end
  true
end

function Arrays.testvalue(a::VariableParamBlock{A}) where A
  v = testvalue(A)
  data = similar(a.data,typeof(v))
  fill!(data,v)
  VariableParamBlock(data)
end

function Arrays.testvalue(::Type{VariableParamBlock{A,N}}) where {A,N}
  VariableParamBlock(fill(testvalue(A),ntuple(i->1,Val{N}())))
end

function Arrays.collect1d(a::VariableParamBlock)
  VariableParamBlock(map(collect1d,a.data))
end

function Arrays.CachedArray(a::VariableParamBlock)
  ai = testitem(a)
  ci = CachedArray(ai)
  data = similar(a.data,typeof(ci))
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
  c = similar(a.data,typeof(ci))
  data = similar(a.data,typeof(ri))
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

# ─── Operation ────────────────────────────────────────────────────────────────

function Arrays.return_cache(f::Operation,x::VariableParamBlock)
  xi = testitem(x)
  li = return_cache(f,xi)
  fix = evaluate!(li,f,xi)
  l = similar(x.data,typeof(li))
  g = similar(x.data,typeof(fix))
  for i in eachindex(x.data)
    l[i] = return_cache(f,x.data[i])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,f::Operation,x::VariableParamBlock)
  g,l = cache
  for i in eachindex(x.data)
    g.data[i] = evaluate!(l[i],f,x.data[i])
  end
  g
end

# ─── VariableParamBlock as function ──────────────────────────────────────────

function Arrays.return_cache(f::VariableParamBlock,x)
  fi = testitem(f)
  li = return_cache(fi,x)
  fix = evaluate!(li,fi,x)
  l = similar(f.data,typeof(li))
  g = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    l[i] = return_cache(f.data[i],x)
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,f::VariableParamBlock,x)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],f.data[i],x)
  end
  g
end

function Arrays.return_cache(f::VariableParamBlock,x::ParamBlock)
  fi = testitem(f)
  xi = testitem(x)
  ci = return_cache(fi,xi)
  ri = evaluate!(ci,fi,xi)
  La = param_length(f)
  Lb = param_length(x)
  c = Matrix{typeof(ci)}(undef,La,Lb)
  g = Matrix{typeof(ri)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    c[i,j] = return_cache(f.data[i],param_getindex(x,j))
  end
  VariableParamBlock(g),c
end

function Arrays.evaluate!(cache,f::VariableParamBlock,x::ParamBlock)
  g,c = cache
  La,Lb = size(g.data)
  for j in 1:Lb, i in 1:La
    g.data[i,j] = evaluate!(c[i,j],f.data[i],param_getindex(x,j))
  end
  g
end

function Arrays.return_cache(f::VariableParamBlock,x::VariableParamBlock)
  fi = testitem(f)
  xi = testitem(x)
  ci = return_cache(fi,xi)
  ri = evaluate!(ci,fi,xi)
  L = param_length(f)
  c = Vector{typeof(ci)}(undef,L)
  g = Vector{typeof(ri)}(undef,L)
  for i in 1:L
    c[i] = return_cache(f.data[i],x.data[i])
  end
  VariableParamBlock(g),c
end

function Arrays.evaluate!(cache,f::VariableParamBlock,x::VariableParamBlock)
  g,c = cache
  for i in eachindex(g.data)
    g.data[i] = evaluate!(c[i],f.data[i],x.data[i])
  end
  g
end

# ─── linear_combination ───────────────────────────────────────────────────────

function Fields.linear_combination(u::VariableParamBlock,f::AbstractVector{<:Field})
  ufi = linear_combination(testitem(u),f)
  g = similar(u.data,typeof(ufi))
  @inbounds for i in eachindex(u.data)
    g[i] = linear_combination(u.data[i],f)
  end
  VariableParamBlock(g)
end

function Fields.linear_combination(u::VariableParamBlock,f::ParamBlock)
  La = param_length(u)
  Lb = param_length(f)
  ufi = linear_combination(testitem(u),testitem(f))
  g = Matrix{typeof(ufi)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    g[i,j] = linear_combination(u.data[i],param_getindex(f,j))
  end
  VariableParamBlock(g)
end

function Fields.linear_combination(u::VariableParamBlock,f::VariableParamBlock)
  @check param_length(u) == param_length(f)
  ufi = linear_combination(testitem(u),testitem(f))
  g = Vector{typeof(ufi)}(undef,param_length(u))
  for i in eachindex(u.data)
    g[i] = linear_combination(u.data[i],f.data[i])
  end
  VariableParamBlock(g)
end

# ─── LinearCombinationMap ─────────────────────────────────────────────────────

function Arrays.return_cache(k::LinearCombinationMap,u::VariableParamBlock,fx::AbstractArray)
  ui = testitem(u)
  ci = return_cache(k,ui,fx)
  ufxi = evaluate!(ci,k,ui,fx)
  l = similar(u.data,typeof(ci))
  g = similar(u.data,typeof(ufxi))
  for i in eachindex(u.data)
    l[i] = return_cache(k,u.data[i],fx)
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::VariableParamBlock,fx::AbstractArray)
  g,l = cache
  for i in eachindex(u.data)
    g.data[i] = evaluate!(l[i],k,u.data[i],fx)
  end
  g
end

function Arrays.return_cache(k::LinearCombinationMap,u::VariableParamBlock,fx::ParamBlock)
  La = param_length(u)
  Lb = param_length(fx)
  ui = testitem(u)
  fxi = testitem(fx)
  ci = return_cache(k,ui,fxi)
  ufxi = evaluate!(ci,k,ui,fxi)
  c = Matrix{typeof(ci)}(undef,La,Lb)
  g = Matrix{typeof(ufxi)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    c[i,j] = return_cache(k,u.data[i],param_getindex(fx,j))
  end
  VariableParamBlock(g),c
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::VariableParamBlock,fx::ParamBlock)
  g,c = cache
  La,Lb = size(g.data)
  for j in 1:Lb, i in 1:La
    g.data[i,j] = evaluate!(c[i,j],k,u.data[i],param_getindex(fx,j))
  end
  g
end

function Arrays.return_cache(k::LinearCombinationMap,u::VariableParamBlock,fx::VariableParamBlock)
  @check param_length(u) == param_length(fx)
  ui = testitem(u)
  fxi = testitem(fx)
  ci = return_cache(k,ui,fxi)
  ufxi = evaluate!(ci,k,ui,fxi)
  L = param_length(u)
  c = Vector{typeof(ci)}(undef,L)
  g = Vector{typeof(ufxi)}(undef,L)
  for i in 1:L
    c[i] = return_cache(k,u.data[i],fx.data[i])
  end
  VariableParamBlock(g),c
end

function Arrays.evaluate!(cache,k::LinearCombinationMap,u::VariableParamBlock,fx::VariableParamBlock)
  g,c = cache
  for i in eachindex(g.data)
    g.data[i] = evaluate!(c[i],k,u.data[i],fx.data[i])
  end
  g
end

# ─── TransposeMap ─────────────────────────────────────────────────────────────

function Base.transpose(f::VariableParamBlock)
  fi = testitem(f)
  g = similar(f.data,typeof(transpose(fi)))
  for i in eachindex(f.data)
    g[i] = transpose(f.data[i])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::Fields.TransposeMap,f::VariableParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = similar(f.data,typeof(li))
  g = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Fields.TransposeMap,f::VariableParamBlock)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i])
  end
  g
end

# ─── Fields.integrate ─────────────────────────────────────────────────────────

function Fields.integrate(f::VariableParamBlock,args...)
  fi = testitem(f)
  intfi = integrate(fi,args...)
  g = similar(f.data,typeof(intfi))
  for i in eachindex(f.data)
    g[i] = integrate(f.data[i],args...)
  end
  VariableParamBlock(g)
end

# ─── IntegrationMap ───────────────────────────────────────────────────────────

function Arrays.return_value(k::IntegrationMap,fx::VariableParamBlock,args...)
  fxi = testitem(fx)
  ufxi = return_value(k,fxi,args...)
  g = similar(fx.data,typeof(ufxi))
  for i in eachindex(fx.data)
    g[i] = return_value(k,fx.data[i],args...)
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::IntegrationMap,fx::VariableParamBlock,args...)
  fxi = testitem(fx)
  li = return_cache(k,fxi,args...)
  ufxi = evaluate!(li,k,fxi,args...)
  l = similar(fx.data,typeof(li))
  g = similar(fx.data,typeof(ufxi))
  for i in eachindex(fx.data)
    l[i] = return_cache(k,fx.data[i],args...)
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx::VariableParamBlock,args...)
  g,l = cache
  for i in eachindex(fx.data)
    g.data[i] = evaluate!(l[i],k,fx.data[i],args...)
  end
  g
end

function Arrays.return_value(k::IntegrationMap,fx,w,jx::VariableParamBlock)
  jxi = testitem(jx)
  ufxi = return_value(k,fx,w,jxi)
  g = similar(jx.data,typeof(ufxi))
  for i in eachindex(jx.data)
    g[i] = return_value(k,fx,w,jx.data[i])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::IntegrationMap,fx,w,jx::VariableParamBlock)
  jxi = testitem(jx)
  li = return_cache(k,fx,w,jxi)
  ufxi = evaluate!(li,k,fx,w,jxi)
  l = similar(jx.data,typeof(li))
  g = similar(jx.data,typeof(ufxi))
  for i in eachindex(jx.data)
    l[i] = return_cache(k,fx,w,jx.data[i])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx,w,jx::VariableParamBlock)
  g,l = cache
  for i in eachindex(jx.data)
    g.data[i] = evaluate!(l[i],k,fx,w,jx.data[i])
  end
  g
end

function Arrays.return_value(k::IntegrationMap,fx::VariableParamBlock,w,jx::ParamBlock)
  fxi = testitem(fx)
  jxi = testitem(jx)
  ufxi = return_value(k,fxi,w,jxi)
  La = param_length(fx)
  Lb = param_length(jx)
  g = Matrix{typeof(ufxi)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    g[i,j] = return_value(k,fx.data[i],w,param_getindex(jx,j))
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::IntegrationMap,fx::VariableParamBlock,w,jx::ParamBlock)
  fxi = testitem(fx)
  jxi = testitem(jx)
  li = return_cache(k,fxi,w,jxi)
  ufxi = evaluate!(li,k,fxi,w,jxi)
  La = param_length(fx)
  Lb = param_length(jx)
  l = Matrix{typeof(li)}(undef,La,Lb)
  g = Matrix{typeof(ufxi)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    l[i,j] = return_cache(k,fx.data[i],w,param_getindex(jx,j))
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx::VariableParamBlock,w,jx::ParamBlock)
  g,l = cache
  La,Lb = size(g.data)
  for j in 1:Lb, i in 1:La
    g.data[i,j] = evaluate!(l[i,j],k,fx.data[i],w,param_getindex(jx,j))
  end
  g
end

function Arrays.return_value(k::IntegrationMap,fx::ParamBlock,w,jx::VariableParamBlock)
  fxi = testitem(fx)
  jxi = testitem(jx)
  ufxi = return_value(k,fxi,w,jxi)
  La = param_length(fx)
  Lb = param_length(jx)
  g = Matrix{typeof(ufxi)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    g[i,j] = return_value(k,param_getindex(fx,i),w,jx.data[j])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::IntegrationMap,fx::ParamBlock,w,jx::VariableParamBlock)
  fxi = testitem(fx)
  jxi = testitem(jx)
  li = return_cache(k,fxi,w,jxi)
  ufxi = evaluate!(li,k,fxi,w,jxi)
  La = param_length(fx)
  Lb = param_length(jx)
  l = Matrix{typeof(li)}(undef,La,Lb)
  g = Matrix{typeof(ufxi)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    l[i,j] = return_cache(k,param_getindex(fx,i),w,jx.data[j])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx::ParamBlock,w,jx::VariableParamBlock)
  g,l = cache
  La,Lb = size(g.data)
  for j in 1:Lb, i in 1:La
    g.data[i,j] = evaluate!(l[i,j],k,param_getindex(fx,i),w,jx.data[j])
  end
  g
end

function Arrays.return_value(k::IntegrationMap,fx::VariableParamBlock,w,jx::VariableParamBlock)
  @check param_length(fx) == param_length(jx)
  fxi = testitem(fx)
  jxi = testitem(jx)
  ufxi = return_value(k,fxi,w,jxi)
  g = Vector{typeof(ufxi)}(undef,param_length(fx))
  for i in eachindex(g)
    g[i] = return_value(k,fx.data[i],w,jx.data[i])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::IntegrationMap,fx::VariableParamBlock,w,jx::VariableParamBlock)
  @check param_length(fx) == param_length(jx)
  fxi = testitem(fx)
  jxi = testitem(jx)
  li = return_cache(k,fxi,w,jxi)
  ufxi = evaluate!(li,k,fxi,w,jxi)
  L = param_length(fx)
  l = Vector{typeof(li)}(undef,L)
  g = Vector{typeof(ufxi)}(undef,L)
  for i in 1:L
    l[i] = return_cache(k,fx.data[i],w,jx.data[i])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::IntegrationMap,fx::VariableParamBlock,w,jx::VariableParamBlock)
  g,l = cache
  for i in eachindex(g.data)
    g.data[i] = evaluate!(l[i],k,fx.data[i],w,jx.data[i])
  end
  g
end

# ─── VoidBasis ────────────────────────────────────────────────────────────────

function Arrays.return_value(k::Fields.VoidBasis,f::VariableParamBlock)
  fi = testitem(f)
  fix = return_value(k,fi)
  g = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    g[i] = return_value(k,f.data[i])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::Fields.VoidBasis,f::VariableParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = similar(f.data,typeof(li))
  g = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Fields.VoidBasis,f::VariableParamBlock)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i])
  end
  g
end

# ─── Broadcasting (unary) ─────────────────────────────────────────────────────

function Arrays.return_value(k::Broadcasting,f::VariableParamBlock)
  fi = testitem(f)
  fix = return_value(k,fi)
  g = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    g[i] = return_value(k,f.data[i])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting,f::VariableParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = similar(f.data,typeof(li))
  g = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting,f::VariableParamBlock)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i])
  end
  g
end

# Broadcasting{typeof(∘)}: (VPB,Field), (VPB,PB)→N=2, (VPB,VPB)→N=1

function Arrays.return_value(k::Broadcasting{typeof(∘)},f::VariableParamBlock,h::Field)
  fi = testitem(f)
  fix = return_value(k,fi,h)
  g = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    g[i] = return_value(k,f.data[i],h)
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::VariableParamBlock,h::Field)
  fi = testitem(f)
  li = return_cache(k,fi,h)
  fix = evaluate!(li,k,fi,h)
  l = similar(f.data,typeof(li))
  g = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i],h)
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::VariableParamBlock,h::Field)
  g,l = cache
  for i in eachindex(f.data)
    g.data[i] = evaluate!(l[i],k,f.data[i],h)
  end
  g
end

function Arrays.return_value(k::Broadcasting{typeof(∘)},f::VariableParamBlock,h::ParamBlock)
  fi = testitem(f)
  hi = testitem(h)
  fix = return_value(k,fi,hi)
  La = param_length(f)
  Lb = param_length(h)
  g = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    g[i,j] = return_value(k,f.data[i],param_getindex(h,j))
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::VariableParamBlock,h::ParamBlock)
  fi = testitem(f)
  hi = testitem(h)
  li = return_cache(k,fi,hi)
  fix = evaluate!(li,k,fi,hi)
  La = param_length(f)
  Lb = param_length(h)
  l = Matrix{typeof(li)}(undef,La,Lb)
  g = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    l[i,j] = return_cache(k,f.data[i],param_getindex(h,j))
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::VariableParamBlock,h::ParamBlock)
  g,l = cache
  La,Lb = size(g.data)
  for j in 1:Lb, i in 1:La
    g.data[i,j] = evaluate!(l[i,j],k,f.data[i],param_getindex(h,j))
  end
  g
end

function Arrays.return_value(k::Broadcasting{typeof(∘)},f::VariableParamBlock,h::VariableParamBlock)
  @check param_length(f) == param_length(h)
  fi = testitem(f)
  hi = testitem(h)
  fix = return_value(k,fi,hi)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in eachindex(g)
    g[i] = return_value(k,f.data[i],h.data[i])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{typeof(∘)},f::VariableParamBlock,h::VariableParamBlock)
  @check param_length(f) == param_length(h)
  fi = testitem(f)
  hi = testitem(h)
  li = return_cache(k,fi,hi)
  fix = evaluate!(li,k,fi,hi)
  L = param_length(f)
  l = Vector{typeof(li)}(undef,L)
  g = Vector{typeof(fix)}(undef,L)
  for i in 1:L
    l[i] = return_cache(k,f.data[i],h.data[i])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(∘)},f::VariableParamBlock,h::VariableParamBlock)
  g,l = cache
  for i in eachindex(g.data)
    g.data[i] = evaluate!(l[i],k,f.data[i],h.data[i])
  end
  g
end

# Broadcasting{<:Function/Operation} with (VPB,Field/AbstractArray) and (Field/AbstractArray,VPB)

for F in (:Function,:Operation)
  for T in (:Field,:AbstractArray)
    @eval begin
      function Arrays.return_value(k::Broadcasting{<:$F},f::VariableParamBlock,h::$T)
        fi = testitem(f)
        fix = return_value(k,fi,h)
        g = similar(f.data,typeof(fix))
        for i in eachindex(f.data)
          g[i] = return_value(k,f.data[i],h)
        end
        VariableParamBlock(g)
      end

      function Arrays.return_cache(k::Broadcasting{<:$F},f::VariableParamBlock,h::$T)
        fi = testitem(f)
        li = return_cache(k,fi,h)
        fix = evaluate!(li,k,fi,h)
        l = similar(f.data,typeof(li))
        g = similar(f.data,typeof(fix))
        for i in eachindex(f.data)
          l[i] = return_cache(k,f.data[i],h)
        end
        VariableParamBlock(g),l
      end

      function Arrays.evaluate!(cache,k::Broadcasting{<:$F},f::VariableParamBlock,h::$T)
        g,l = cache
        for i in eachindex(f.data)
          g.data[i] = evaluate!(l[i],k,f.data[i],h)
        end
        g
      end

      function Arrays.return_value(k::Broadcasting{<:$F},h::$T,f::VariableParamBlock)
        fi = testitem(f)
        fix = return_value(k,h,fi)
        g = similar(f.data,typeof(fix))
        for i in eachindex(f.data)
          g[i] = return_value(k,h,f.data[i])
        end
        VariableParamBlock(g)
      end

      function Arrays.return_cache(k::Broadcasting{<:$F},h::$T,f::VariableParamBlock)
        fi = testitem(f)
        li = return_cache(k,h,fi)
        fix = evaluate!(li,k,h,fi)
        l = similar(f.data,typeof(li))
        g = similar(f.data,typeof(fix))
        for i in eachindex(f.data)
          l[i] = return_cache(k,h,f.data[i])
        end
        VariableParamBlock(g),l
      end

      function Arrays.evaluate!(cache,k::Broadcasting{<:$F},h::$T,f::VariableParamBlock)
        g,l = cache
        for i in eachindex(f.data)
          g.data[i] = evaluate!(l[i],k,h,f.data[i])
        end
        g
      end
    end
  end
end

# Broadcasting{<:Operation}: (VPB,PB)→N=2, (PB,VPB)→N=2, (VPB,VPB)→N=1

function Arrays.return_value(k::Broadcasting{<:Operation},h::VariableParamBlock,f::ParamBlock)
  hi = testitem(h)
  fi = testitem(f)
  fix = return_value(k,hi,fi)
  La = param_length(h)
  Lb = param_length(f)
  g = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    g[i,j] = return_value(k,h.data[i],param_getindex(f,j))
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},h::VariableParamBlock,f::ParamBlock)
  hi = testitem(h)
  fi = testitem(f)
  li = return_cache(k,hi,fi)
  fix = evaluate!(li,k,hi,fi)
  La = param_length(h)
  Lb = param_length(f)
  l = Matrix{typeof(li)}(undef,La,Lb)
  g = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    l[i,j] = return_cache(k,h.data[i],param_getindex(f,j))
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},h::VariableParamBlock,f::ParamBlock)
  g,l = cache
  La,Lb = size(g.data)
  for j in 1:Lb, i in 1:La
    g.data[i,j] = evaluate!(l[i,j],k,h.data[i],param_getindex(f,j))
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:Operation},h::ParamBlock,f::VariableParamBlock)
  hi = testitem(h)
  fi = testitem(f)
  fix = return_value(k,hi,fi)
  La = param_length(h)
  Lb = param_length(f)
  g = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    g[i,j] = return_value(k,param_getindex(h,i),f.data[j])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},h::ParamBlock,f::VariableParamBlock)
  hi = testitem(h)
  fi = testitem(f)
  li = return_cache(k,hi,fi)
  fix = evaluate!(li,k,hi,fi)
  La = param_length(h)
  Lb = param_length(f)
  l = Matrix{typeof(li)}(undef,La,Lb)
  g = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    l[i,j] = return_cache(k,param_getindex(h,i),f.data[j])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},h::ParamBlock,f::VariableParamBlock)
  g,l = cache
  La,Lb = size(g.data)
  for j in 1:Lb, i in 1:La
    g.data[i,j] = evaluate!(l[i,j],k,param_getindex(h,i),f.data[j])
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:Operation},h::VariableParamBlock,f::VariableParamBlock)
  @check param_length(h) == param_length(f)
  hi = testitem(h)
  fi = testitem(f)
  fix = return_value(k,hi,fi)
  g = Vector{typeof(fix)}(undef,param_length(f))
  for i in eachindex(g)
    g[i] = return_value(k,h.data[i],f.data[i])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:Operation},h::VariableParamBlock,f::VariableParamBlock)
  @check param_length(h) == param_length(f)
  hi = testitem(h)
  fi = testitem(f)
  li = return_cache(k,hi,fi)
  fix = evaluate!(li,k,hi,fi)
  L = param_length(f)
  l = Vector{typeof(li)}(undef,L)
  g = Vector{typeof(fix)}(undef,L)
  for i in 1:L
    l[i] = return_cache(k,h.data[i],f.data[i])
  end
  VariableParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:Operation},h::VariableParamBlock,f::VariableParamBlock)
  g,l = cache
  for i in eachindex(g.data)
    g.data[i] = evaluate!(l[i],k,h.data[i],f.data[i])
  end
  g
end

# ─── BroadcastingFieldOpMap ───────────────────────────────────────────────────

function Arrays.return_value(k::BroadcastingFieldOpMap,f::VariableParamBlock)
  fi = testitem(f)
  fix = return_value(k,fi)
  g = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    g[i] = return_value(k,f.data[i])
  end
  VariableParamBlock(g)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::VariableParamBlock)
  fi = testitem(f)
  li = return_cache(k,fi)
  fix = evaluate!(li,k,fi)
  l = similar(f.data,typeof(li))
  h = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i])
  end
  VariableParamBlock(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::VariableParamBlock)
  h,l = cache
  for i in eachindex(f.data)
    h.data[i] = evaluate!(l[i],k,f.data[i])
  end
  h
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::VariableParamBlock,g::AbstractArray)
  fi = testitem(f)
  fix = return_value(k,fi,g)
  h = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    h[i] = return_value(k,f.data[i],g)
  end
  VariableParamBlock(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::VariableParamBlock,g::AbstractArray)
  fi = testitem(f)
  li = return_cache(k,fi,g)
  fix = evaluate!(li,k,fi,g)
  l = similar(f.data,typeof(li))
  h = similar(f.data,typeof(fix))
  for i in eachindex(f.data)
    l[i] = return_cache(k,f.data[i],g)
  end
  VariableParamBlock(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::VariableParamBlock,g::AbstractArray)
  h,l = cache
  for i in eachindex(f.data)
    h.data[i] = evaluate!(l[i],k,f.data[i],g)
  end
  h
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::VariableParamBlock,g::ParamBlock)
  fi = testitem(f)
  gi = testitem(g)
  fix = return_value(k,fi,gi)
  La = param_length(f)
  Lb = param_length(g)
  h = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    h[i,j] = return_value(k,f.data[i],param_getindex(g,j))
  end
  VariableParamBlock(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::VariableParamBlock,g::ParamBlock)
  fi = testitem(f)
  gi = testitem(g)
  li = return_cache(k,fi,gi)
  fix = evaluate!(li,k,fi,gi)
  La = param_length(f)
  Lb = param_length(g)
  l = Matrix{typeof(li)}(undef,La,Lb)
  h = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    l[i,j] = return_cache(k,f.data[i],param_getindex(g,j))
  end
  VariableParamBlock(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::VariableParamBlock,g::ParamBlock)
  h,l = cache
  La,Lb = size(h.data)
  for j in 1:Lb, i in 1:La
    h.data[i,j] = evaluate!(l[i,j],k,f.data[i],param_getindex(g,j))
  end
  h
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::ParamBlock,g::VariableParamBlock)
  fi = testitem(f)
  gi = testitem(g)
  fix = return_value(k,fi,gi)
  La = param_length(f)
  Lb = param_length(g)
  h = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    h[i,j] = return_value(k,param_getindex(f,i),g.data[j])
  end
  VariableParamBlock(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::ParamBlock,g::VariableParamBlock)
  fi = testitem(f)
  gi = testitem(g)
  li = return_cache(k,fi,gi)
  fix = evaluate!(li,k,fi,gi)
  La = param_length(f)
  Lb = param_length(g)
  l = Matrix{typeof(li)}(undef,La,Lb)
  h = Matrix{typeof(fix)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    l[i,j] = return_cache(k,param_getindex(f,i),g.data[j])
  end
  VariableParamBlock(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::ParamBlock,g::VariableParamBlock)
  h,l = cache
  La,Lb = size(h.data)
  for j in 1:Lb, i in 1:La
    h.data[i,j] = evaluate!(l[i,j],k,param_getindex(f,i),g.data[j])
  end
  h
end

function Arrays.return_value(k::BroadcastingFieldOpMap,f::VariableParamBlock,g::VariableParamBlock)
  @check param_length(f) == param_length(g)
  fi = testitem(f)
  gi = testitem(g)
  fix = return_value(k,fi,gi)
  h = Vector{typeof(fix)}(undef,param_length(f))
  for i in eachindex(h)
    h[i] = return_value(k,f.data[i],g.data[i])
  end
  VariableParamBlock(h)
end

function Arrays.return_cache(k::BroadcastingFieldOpMap,f::VariableParamBlock,g::VariableParamBlock)
  @check param_length(f) == param_length(g)
  fi = testitem(f)
  gi = testitem(g)
  li = return_cache(k,fi,gi)
  fix = evaluate!(li,k,fi,gi)
  L = param_length(f)
  l = Vector{typeof(li)}(undef,L)
  h = Vector{typeof(fix)}(undef,L)
  for i in 1:L
    l[i] = return_cache(k,f.data[i],g.data[i])
  end
  VariableParamBlock(h),l
end

function Arrays.evaluate!(cache,k::BroadcastingFieldOpMap,f::VariableParamBlock,g::VariableParamBlock)
  h,l = cache
  for i in eachindex(h.data)
    h.data[i] = evaluate!(l[i],k,f.data[i],g.data[i])
  end
  h
end

# ─── arithmetic ───────────────────────────────────────────────────────────────

for T in (:Number,:AbstractArray)
  @eval begin
    function Base.:*(a::$T,b::VariableParamBlock)
      bi = testitem(b)
      ci = a*bi
      data = similar(b.data,typeof(ci))
      for i in eachindex(b.data)
        data[i] = a*b.data[i]
      end
      VariableParamBlock(data)
    end

    function Base.:*(a::VariableParamBlock,b::$T)
      ai = testitem(a)
      ci = ai*b
      data = similar(a.data,typeof(ci))
      for i in eachindex(a.data)
        data[i] = a.data[i]*b
      end
      VariableParamBlock(data)
    end
  end
end

function Base.:*(a::VariableParamBlock,b::VariableParamBlock)
  @check param_length(a) == param_length(b)
  ai = testitem(a)
  bi = testitem(b)
  ri = ai*bi
  data = Vector{typeof(ri)}(undef,param_length(a))
  data[1] = ri
  for i in 2:param_length(a)
    data[i] = a.data[i]*b.data[i]
  end
  VariableParamBlock(data)
end

function Base.:*(a::VariableParamBlock,b::ParamBlock)
  ai = testitem(a)
  bi = testitem(b)
  ri = ai*bi
  La = param_length(a)
  Lb = param_length(b)
  data = Matrix{typeof(ri)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    data[i,j] = a.data[i]*param_getindex(b,j)
  end
  VariableParamBlock(data)
end

function Base.:*(a::ParamBlock,b::VariableParamBlock)
  ai = testitem(a)
  bi = testitem(b)
  ri = ai*bi
  La = param_length(a)
  Lb = param_length(b)
  data = Matrix{typeof(ri)}(undef,La,Lb)
  for j in 1:Lb, i in 1:La
    data[i,j] = param_getindex(a,i)*b.data[j]
  end
  VariableParamBlock(data)
end

function LinearAlgebra.rmul!(a::VariableParamBlock,β)
  for i in eachindex(a.data)
    rmul!(a.data[i],β)
  end
end

function LinearAlgebra.mul!(c::VariableParamBlock,a::AbstractArray,b::VariableParamBlock,α::Number,β::Number)
  for i in eachindex(c.data)
    mul!(c.data[i],a,b.data[i],α,β)
  end
end

function LinearAlgebra.mul!(c::VariableParamBlock,a::VariableParamBlock,b::AbstractArray,α::Number,β::Number)
  for i in eachindex(c.data)
    mul!(c.data[i],a.data[i],b,α,β)
  end
end

function LinearAlgebra.mul!(c::VariableParamBlock,a::VariableParamBlock,b::VariableParamBlock,α::Number,β::Number)
  for i in eachindex(c.data)
    mul!(c.data[i],a.data[i],b.data[i],α,β)
  end
end

function LinearAlgebra.mul!(c::VariableParamBlock,a::VariableParamBlock,b::ParamBlock,α::Number,β::Number)
  La,Lb = size(c.data)
  for j in 1:Lb, i in 1:La
    mul!(c.data[i,j],a.data[i],param_getindex(b,j),α,β)
  end
end

function LinearAlgebra.mul!(c::VariableParamBlock,a::ParamBlock,b::VariableParamBlock,α::Number,β::Number)
  La,Lb = size(c.data)
  for j in 1:Lb, i in 1:La
    mul!(c.data[i,j],param_getindex(a,i),b.data[j],α,β)
  end
end

# ─── setsize_op! ──────────────────────────────────────────────────────────────

function Arrays.setsize_op!(::typeof(copy),a::AbstractArray,b::VariableParamBlock)
  for i in eachindex(b.data)
    Arrays.setsize_op!(copy,a,b.data[i])
  end
end

function Arrays.setsize_op!(::typeof(copy),a::VariableParamBlock,b::AbstractArray)
  for i in eachindex(a.data)
    Arrays.setsize_op!(copy,a.data[i],b)
  end
end

function Arrays.setsize_op!(::typeof(copy),a::VariableParamBlock,b::VariableParamBlock)
  for i in eachindex(a.data)
    Arrays.setsize_op!(copy,a.data[i],b.data[i])
  end
end

function Arrays.setsize_op!(::typeof(*),c::VariableParamBlock,a::AbstractArray,b::VariableParamBlock)
  for i in eachindex(c.data)
    Arrays.setsize_op!(*,c.data[i],a,b.data[i])
  end
end

function Arrays.setsize_op!(::typeof(*),c::VariableParamBlock,a::VariableParamBlock,b::AbstractArray)
  for i in eachindex(c.data)
    Arrays.setsize_op!(*,c.data[i],a.data[i],b)
  end
end

function Arrays.setsize_op!(::typeof(*),c::VariableParamBlock,a::VariableParamBlock,b::VariableParamBlock)
  for i in eachindex(c.data)
    Arrays.setsize_op!(*,c.data[i],a.data[i],b.data[i])
  end
end

function Arrays.setsize_op!(::typeof(*),c::VariableParamBlock,a::VariableParamBlock,b::ParamBlock)
  La,Lb = size(c.data)
  for j in 1:Lb, i in 1:La
    Arrays.setsize_op!(*,c.data[i,j],a.data[i],param_getindex(b,j))
  end
end

function Arrays.setsize_op!(::typeof(*),c::VariableParamBlock,a::ParamBlock,b::VariableParamBlock)
  La,Lb = size(c.data)
  for j in 1:Lb, i in 1:La
    Arrays.setsize_op!(*,c.data[i,j],param_getindex(a,i),b.data[j])
  end
end

# lazy * dispatch

function Arrays.return_value(::typeof(*),a::AbstractArray,b::VariableParamBlock)
  bi = testitem(b)
  ri = return_value(*,a,bi)
  g = similar(b.data,typeof(ri))
  fill!(g,ri)
  VariableParamBlock(g)
end

function Arrays.return_cache(::typeof(*),a::AbstractArray,b::VariableParamBlock)
  c1 = CachedArray(a*b)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::AbstractArray,b::VariableParamBlock)
  c1,c2 = cache
  Arrays.setsize_op!(*,c1,a,b)
  c = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Arrays.return_value(::typeof(*),a::VariableParamBlock,b::AbstractArray)
  ai = testitem(a)
  bi = testitem(b)
  ri = return_value(*,ai,bi)
  g = similar(a.data,typeof(ri))
  fill!(g,ri)
  VariableParamBlock(g)
end

function Arrays.return_cache(::typeof(*),a::VariableParamBlock,b::AbstractArray)
  c1 = CachedArray(a*b)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::VariableParamBlock,b::AbstractArray)
  c1,c2 = cache
  Arrays.setsize_op!(*,c1,a,b)
  c = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Arrays.return_value(::typeof(*),a::VariableParamBlock,b::VariableParamBlock)
  @check param_length(a) == param_length(b)
  ai = testitem(a)
  bi = testitem(b)
  ri = return_value(*,ai,bi)
  g = Vector{typeof(ri)}(undef,param_length(a))
  fill!(g,ri)
  VariableParamBlock(g)
end

function Arrays.return_cache(::typeof(*),a::VariableParamBlock,b::VariableParamBlock)
  c1 = CachedArray(a*b)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::VariableParamBlock,b::VariableParamBlock)
  c1,c2 = cache
  Arrays.setsize_op!(*,c1,a,b)
  c = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Arrays.return_value(::typeof(*),a::VariableParamBlock,b::ParamBlock)
  ai = testitem(a)
  bi = testitem(b)
  ri = return_value(*,ai,bi)
  g = Matrix{typeof(ri)}(undef,param_length(a),param_length(b))
  fill!(g,ri)
  VariableParamBlock(g)
end

function Arrays.return_cache(::typeof(*),a::VariableParamBlock,b::ParamBlock)
  c1 = CachedArray(a*b)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::VariableParamBlock,b::ParamBlock)
  c1,c2 = cache
  Arrays.setsize_op!(*,c1,a,b)
  c = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Arrays.return_value(::typeof(*),a::ParamBlock,b::VariableParamBlock)
  ai = testitem(a)
  bi = testitem(b)
  ri = return_value(*,ai,bi)
  g = Matrix{typeof(ri)}(undef,param_length(a),param_length(b))
  fill!(g,ri)
  VariableParamBlock(g)
end

function Arrays.return_cache(::typeof(*),a::ParamBlock,b::VariableParamBlock)
  c1 = CachedArray(a*b)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,::typeof(*),a::ParamBlock,b::VariableParamBlock)
  c1,c2 = cache
  Arrays.setsize_op!(*,c1,a,b)
  c = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  mul!(c,a,b)
  c
end

function Arrays.return_value(k::MulAddMap,a,b::VariableParamBlock,c::VariableParamBlock)
  x = return_value(*,a,b)
  return_value(+,x,c)
end

function Arrays.return_cache(k::MulAddMap,a,b::VariableParamBlock,c::VariableParamBlock)
  c1 = CachedArray(a*b+c)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  (c1,c2)
end

function Arrays.evaluate!(cache,k::MulAddMap,a,b::VariableParamBlock,c::VariableParamBlock)
  c1,c2 = cache
  Arrays.setsize_op!(copy,c1,c)
  Arrays.setsize_op!(*,c1,a,b)
  d = evaluate!(c2,Arrays.unwrap_cached_array,c1)
  copyto!(d,c)
  iszero(k.α) && isone(k.β) && return d
  mul!(d,a,b,k.α,k.β)
  d
end

# ─── autodiff ─────────────────────────────────────────────────────────────────

for f in (:gradient,:jacobian)
  @eval begin
    function Arrays.return_cache(k::Arrays.ConfigMap{typeof(ForwardDiff.$f)},x::VariableParamBlock)
      xi = testitem(x)
      fi = return_cache(k,xi)
      data = similar(x.data,typeof(fi))
      for i in eachindex(x.data)
        data[i] = return_cache(k,x.data[i])
      end
      VariableParamBlock(data)
    end
  end
end

for F in (:GradientConfig,:JacobianConfig)
  @eval begin
    function Arrays.return_value(k::DualizeMap,cfg::VariableParamBlock{<:ForwardDiff.$F},x::VariableParamBlock)
      vi = return_value(k,testitem(cfg),testitem(x))
      v = similar(x.data,typeof(vi))
      fill!(v,vi)
      VariableParamBlock(v)
    end
  end
end

function Arrays.evaluate!(cache,k::DualizeMap,cfg::VariableParamBlock,x::VariableParamBlock)
  for i in eachindex(x.data)
    evaluate!(nothing,k,cfg.data[i],x.data[i])
  end
end

function Arrays.return_cache(k::Arrays.AutoDiffMap,cfg::VariableParamBlock,ydual::VariableParamBlock)
  ci = return_cache(k,testitem(cfg),testitem(ydual))
  ri = evaluate!(ci,k,testitem(cfg),testitem(ydual))
  c = similar(cfg.data,typeof(ci))
  data = similar(cfg.data,typeof(ri))
  for i in eachindex(ydual.data)
    c[i] = return_cache(k,cfg.data[i],ydual.data[i])
  end
  VariableParamBlock(data),c
end

function Arrays.evaluate!(cache,k::Arrays.AutoDiffMap,cfg::VariableParamBlock,ydual::VariableParamBlock)
  r,c = cache
  for i in eachindex(ydual.data)
    r.data[i] = evaluate!(c[i],k,cfg.data[i],ydual.data[i])
  end
  r
end

# ─── ZeroVectorMap ────────────────────────────────────────────────────────────

function Arrays.return_cache(k::CellData.ZeroVectorMap,a::VariableParamBlock)
  ai = testitem(a)
  ci = return_cache(k,ai)
  vi = evaluate!(ci,k,ai)
  l = similar(a.data,typeof(ci))
  data = similar(a.data,typeof(vi))
  for i in eachindex(a.data)
    l[i] = return_cache(k,a.data[i])
  end
  VariableParamBlock(data),l
end

function Arrays.evaluate!(cache,k::CellData.ZeroVectorMap,a::VariableParamBlock)
  r,c = cache
  for i in eachindex(a.data)
    r.data[i] = evaluate!(c[i],k,a.data[i])
  end
  r
end

# ─── LagrangianDofBasis ───────────────────────────────────────────────────────

function Arrays.return_cache(b::LagrangianDofBasis,f::VariableParamBlock)
  fi = testitem(f)
  ci = return_cache(b,fi)
  ri = evaluate!(ci,b,fi)
  l = similar(f.data,typeof(ci))
  data = similar(f.data,typeof(ri))
  for i in eachindex(f.data)
    l[i] = return_cache(b,f.data[i])
  end
  VariableParamBlock(data),l
end

function Arrays.evaluate!(cache,b::LagrangianDofBasis,f::VariableParamBlock)
  r,c = cache
  for i in eachindex(f.data)
    r.data[i] = evaluate!(c[i],b,f.data[i])
  end
  r
end

# ─── Geometry ─────────────────────────────────────────────────────────────────

function Geometry._cache_compress(a::VariableParamBlock)
  c1 = CachedArray(a)
  c2 = return_cache(Arrays.unwrap_cached_array,c1)
  c1,c2
end

function Geometry._uncached_compress!(c1::VariableParamBlock,c2)
  evaluate!(c2,Arrays.unwrap_cached_array,c1)
end

function Geometry._setempty_compress!(a::VariableParamBlock)
  for i in eachindex(a.data)
    Geometry._setempty_compress!(a.data[i])
  end
end

function Geometry._setsize_compress!(a::VariableParamBlock,b::VariableParamBlock)
  @check size(a.data) == size(b.data)
  for i in eachindex(a.data)
    Geometry._setsize_compress!(a.data[i],b.data[i])
  end
end

function Geometry._copyto_compress!(a::VariableParamBlock,b::VariableParamBlock)
  @check size(a.data) == size(b.data)
  for i in eachindex(a.data)
    Geometry._copyto_compress!(a.data[i],b.data[i])
  end
end

function Geometry._addto_compress!(a::VariableParamBlock,b::VariableParamBlock)
  @check size(a.data) == size(b.data)
  for i in eachindex(a.data)
    Geometry._addto_compress!(a.data[i],b.data[i])
  end
end

function Geometry._similar_empty(val::VariableParamBlock)
  a = deepcopy(val)
  for i in eachindex(a.data)
    a.data[i] = Geometry._similar_empty(a.data[i])
  end
  a
end
