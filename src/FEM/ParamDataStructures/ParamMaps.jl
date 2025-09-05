const ParamReindex = Union{Reindex{<:ParamBlock},Reindex{<:AbstractParamArray}}

param_length(k::ParamReindex) = param_length(k.values)
param_getindex(k::ParamReindex,i::Integer) = Reindex(param_getindex(k.values,i))

Arrays.testitem(k::ParamReindex) = param_getindex(k,1)
Arrays.testargs(k::ParamReindex,i...) = testargs(testitem(k),i...)

const PosNegParamReindex = Union{PosNegReindex{<:ParamBlock,<:ParamBlock},PosNegReindex{<:AbstractParamArray,<:AbstractParamArray}}

function param_length(k::PosNegParamReindex)
  @check param_length(k.values_pos) == param_length(k.values_neg)
  param_length(k.values_pos)
end

param_getindex(k::PosNegParamReindex,i::Integer) = PosNegReindex(param_getindex(k.values_pos,i),param_getindex(k.values_neg,i))

Arrays.testitem(k::PosNegParamReindex) = param_getindex(k,1)
Arrays.testargs(k::PosNegParamReindex,i...) = testargs(testitem(k),i...)

for T in (:ParamReindex,:PosNegParamReindex)
  @eval begin
    function Arrays.return_value(
      f::Broadcasting{<:$T},
      x::Union{Number,AbstractArray{<:Number}}...)

      vi = return_value(Broadcasting(testitem(f.f)),x...)
      local_parameterize(vi,param_length(f.f))
    end

    function Arrays.return_cache(
      f::Broadcasting{<:$T},
      x::Union{Number,AbstractArray{<:Number}}...)

      c = return_cache(Broadcasting(testitem(f.f)),x...)
      a = evaluate!(c,Broadcasting(testitem(f.f)),x...)
      cache = Vector{typeof(c)}(undef,param_length(f.f))
      data = local_parameterize(a,param_length(f.f))
      @inbounds for i = param_eachindex(f.f)
        cache[i] = return_cache(Broadcasting(param_getindex(f.f,i)),x...)
      end
      cache,data
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{<:$T},
      x::Union{Number,AbstractArray{<:Number}}...)

      c,data = cache
      @inbounds for i = param_eachindex(f.f)
        vi = evaluate!(c[i],Broadcasting(param_getindex(f.f,i)),x...)
        param_setindex!(data,vi,i)
      end
      data
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{<:$T},
      x::AbstractArray{<:Number})

      c,data = cache
      @inbounds for i = param_eachindex(f.f)
        vi = evaluate!(c[i],Broadcasting(param_getindex(f.f,i)),x)
        param_setindex!(data,vi,i)
      end
      data
    end

    function Arrays.evaluate!(
      cache,
      f::Broadcasting{<:$T},
      x::Number...)

      c,data = cache
      @inbounds for i = param_eachindex(f.f)
        vi = evaluate!(c[i],Broadcasting(param_getindex(f.f,i)),x...)
        param_setindex!(data,vi,i)
      end
      data
    end

    function Arrays.return_value(k::$T,j::Integer)
      vi = return_value(testitem(k),j)
      local_parameterize(vi,param_length(k))
    end

    function Arrays.return_cache(k::$T,j::Integer)
      c = return_cache(testitem(k),j)
      a = evaluate!(c,testitem(k),j)
      cache = Vector{typeof(c)}(undef,param_length(k))
      data = local_parameterize(a,param_length(k))
      for i = param_eachindex(k)
        cache[i] = return_cache(param_getindex(k,i),j)
      end
      cache,data
    end

    function Arrays.evaluate!(cache,k::$T,j::Integer)
      c,data = cache
      @inbounds for i = param_eachindex(k)
        vi = evaluate!(c[i],param_getindex(k,i),j)
        param_setindex!(data,vi,i)
      end
      data
    end
  end
end

function Arrays.return_cache(k::OReindex,indices::AbstractVector,values::Union{ParamBlock,AbstractParamVector})
  v = testitem(values)
  c = return_cache(k,indices,v)
  a = evaluate!(c,k,indices,v)
  data = local_parameterize(a,param_length(values))
  cache = Vector{typeof(c)}(undef,param_length(values))
  for i = param_eachindex(values)
    cache[i] = return_cache(k,indices,param_getindex(values,i))
  end
  cache,data
end

function Arrays.evaluate!(cache,k::OReindex,indices::AbstractVector,values::Union{ParamBlock,AbstractParamVector})
  c,data = cache
  @inbounds for i = param_eachindex(values)
    vi = evaluate!(c[i],k,indices,param_getindex(values,i))
    param_setindex!(data,vi,i)
  end
  data
end

struct FetchParam <: Map
  index::Int
end

function Arrays.lazy_map(k::FetchParam,a::LazyArray{<:Fill{FetchParam}})
  lazy_map(k,a.args...)
end

function lazy_param_getindex(a,index::Int)
  lazy_map(FetchParam(index),a)
end

function lazy_testitem(a)
  lazy_param_getindex(a,1)
end

function Arrays.evaluate!(cache,k::FetchParam,a::ParamBlock)
  a.data[k.index]
end

function Arrays.return_cache(k::FetchParam,a::ArrayBlock{A,N}) where {A,N}
  ai = testitem(a)
  li = return_cache(k,ai)
  fix = evaluate!(li,k,ai)
  l = Array{typeof(li),N}(undef,size(a.array))
  g = Array{typeof(fix),N}(undef,size(a.array))
  for i in eachindex(a.array)
    if a.touched[i]
      a[i] = return_cache(k,a.array[i])
    end
  end
  ArrayBlock(g,a.touched),l
end

function Arrays.evaluate!(cache,k::FetchParam,a::ArrayBlock{A,N}) where {A,N}
  g,l = cache
  @check g.touched == a.touched
  for i in eachindex(a.array)
    if a.touched[i]
      g.array[i] = evaluate!(l[i],k,a.array[i])
    end
  end
  g
end
