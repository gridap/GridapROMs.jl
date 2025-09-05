struct PosZeroNegReindex{A,B} <: Map
  values_pos::A
  values_neg::B
end

function Arrays.testargs(k::PosZeroNegReindex,i::Integer)
  @check length(k.values_pos) !=0 || length(k.values_neg) != 0 "This map has empty domain"
  @check eltype(k.values_pos) == eltype(k.values_neg) "This map is type-unstable"
  length(k.values_pos) !=0 ? (one(i),) : (-one(i))
end

function Arrays.return_value(k::PosZeroNegReindex,i::Integer)
  if length(k.values_pos)==0 && length(k.values_neg)==0
    @check eltype(k.values_pos) == eltype(k.values_neg) "This map is type-unstable"
    testitem(k.values_pos)
  else
    evaluate(k,testargs(k,i)...)
  end
end

function Arrays.return_cache(k::PosZeroNegReindex,i::Integer)
  c_p = array_cache(k.values_pos)
  c_n = array_cache(k.values_neg)
  z = zero(eltype(k.values_pos))
  c_p,c_n,z
end

function Arrays.evaluate!(cache,k::PosZeroNegReindex,i::Integer)
  c_p,c_n,z = cache
  i>0 ? getindex!(c_p,k.values_pos,i) : (i<0 ? getindex!(c_n,k.values_neg,-i) : z)
end

function Arrays.evaluate(k::PosZeroNegReindex,i::Integer)
  i>0 ? k.values_pos[i] : (i<0 ? k.values_neg[-i] : zero(eltype(k.values_pos)))
end

const PosZeroNegParamReindex = Union{
  PosZeroNegReindex{<:ParamBlock,<:ParamBlock},
  PosZeroNegReindex{<:AbstractParamArray,<:AbstractParamArray}
}

function ParamDataStructures.param_length(k::PosZeroNegParamReindex)
  @check param_length(k.values_pos) == param_length(k.values_neg)
  param_length(k.values_pos)
end

function ParamDataStructures.param_getindex(k::PosZeroNegParamReindex,i::Integer)
  PosZeroNegReindex(param_getindex(k.values_pos,i),param_getindex(k.values_neg,i))
end

Arrays.testitem(k::PosZeroNegParamReindex) = param_getindex(k,1)

function Arrays.return_value(
  k::Broadcasting{<:PosZeroNegParamReindex},
  x::Union{Number,AbstractArray{<:Number}}...)

  vi = return_value(Broadcasting(testitem(k.f)),x...)
  local_parameterize(vi,param_length(k.f))
end

function Arrays.return_cache(
  k::Broadcasting{<:PosZeroNegParamReindex},
  x::Union{Number,AbstractArray{<:Number}}...)

  fi = testitem(k.f)
  c = return_cache(Broadcasting(fi),x...)
  a = evaluate!(c,Broadcasting(fi),x...)
  cache = Vector{typeof(c)}(undef,param_length(k.f))
  data = local_parameterize(a,param_length(k.f))
  @inbounds for i = param_eachindex(k.f)
    cache[i] = return_cache(Broadcasting(param_getindex(k.f,i)),x...)
  end
  cache,data
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:PosZeroNegParamReindex},
  x::Union{Number,AbstractArray{<:Number}}...)

  c,data = cache
  @inbounds for i = param_eachindex(k.f)
    vi = evaluate!(c[i],Broadcasting(param_getindex(k.f,i)),x...)
    param_setindex!(data,vi,i)
  end
  data
end

function Arrays.evaluate!(
  cache,
  k::Broadcasting{<:PosZeroNegParamReindex},
  x::AbstractArray{<:Number})

  c,data = cache
  @inbounds for i = param_eachindex(k.f)
    vi = evaluate!(c[i],Broadcasting(param_getindex(k.f,i)),x)
    param_setindex!(data,vi,i)
  end
  data
end

function Arrays.return_value(k::PosZeroNegParamReindex,i::Integer)
  vi = return_value(testitem(k),i)
  local_parameterize(vi,param_length(k))
end

function Arrays.return_cache(k::PosZeroNegParamReindex,i::Integer)
  ki = testitem(k)
  c = return_cache(ki,i)
  a = evaluate!(c,ki,i)
  cache = Vector{typeof(c)}(undef,param_length(k))
  data = local_parameterize(a,param_length(k))
  @inbounds for i = param_eachindex(k)
    cache[i] = return_cache(param_getindex(k,i),i)
  end
  cache,data
end

function Arrays.evaluate!(cache,k::PosZeroNegParamReindex,i::Integer)
  c,data = cache
  @inbounds for i = param_eachindex(k)
    vi = evaluate!(c[i],param_getindex(k,i),x...)
    param_setindex!(data,vi,i)
  end
  data
end

function Arrays.evaluate(k::PosZeroNegParamReindex,i::Integer)
  c = return_cache(k,i)
  evaluate!(c,k,i)
end
