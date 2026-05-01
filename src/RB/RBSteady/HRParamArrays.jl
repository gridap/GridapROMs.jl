struct HRParamArray{T,N,A,B,C<:ParamArray{T,N}} <: ParamArray{T,N}
  fecache::A
  coeff::B
  hypred::C
end

function hr_array(fecache,coeff,hypred::ParamArray)
  HRParamArray(fecache,coeff,hypred)
end

Base.size(a::HRParamArray) = size(a.hypred)
Base.getindex(a::HRParamArray{T,N},i::Vararg{Integer,N}) where {T,N} = getindex(a.hypred,i...)
Base.setindex!(a::HRParamArray{T,N},v,i::Vararg{Integer,N}) where {T,N} = setindex!(a.hypred,v,i...)
ParamDataStructures.param_length(a::HRParamArray) = param_length(a.hypred)
ParamDataStructures.get_all_data(a::HRParamArray) = get_all_data(a.hypred)
ParamDataStructures.param_getindex(a::HRParamArray,i::Integer) = param_getindex(a.hypred,i)

struct NoHRParamArray{T,N,A,B,C,D<:ParamArray{T,N}} <: ParamArray{T,N}
  fecache::A
  rbfecache::B
  coeff::C
  hypred::D
end

function nohr_array(fecache,rbfecache,coeff,hypred::ParamArray)
  NoHRParamArray(fecache,rbfecache,coeff,hypred)
end

function nohr_array(fecache,â::HRParamArray)
  NoHRParamArray(fecache,â.fecache,â.coeff,â.hypred)
end

Base.size(a::NoHRParamArray) = size(a.hypred)
Base.getindex(a::NoHRParamArray{T,N},i::Vararg{Integer,N}) where {T,N} = getindex(a.hypred,i...)
Base.setindex!(a::NoHRParamArray{T,N},v,i::Vararg{Integer,N}) where {T,N} = setindex!(a.hypred,v,i...)
ParamDataStructures.param_length(a::NoHRParamArray) = param_length(a.hypred)
ParamDataStructures.get_all_data(a::NoHRParamArray) = get_all_data(a.hypred)
ParamDataStructures.param_getindex(a::NoHRParamArray,i::Integer) = param_getindex(a.hypred,i)

const AbstractHRParamArray{T,N} = Union{HRParamArray{T,N},NoHRParamArray{T,N}}

for f in (:(Base.copy),:(Base.similar))
  @eval begin
    function $f(a::NoHRParamArray)
      fe_quantity′ = $f(a.fecache)
      rb_quantity′ = $f(a.rbfecache)
      coeff′ = $f(a.coeff)
      hypred′ = $f(a.hypred)
      nohr_array(fe_quantity′,rb_quantity′,coeff′,hypred′)
    end
    function $f(a::HRParamArray)
      fe_quantity′ = $f(a.fecache)
      coeff′ = $f(a.coeff)
      hypred′ = $f(a.hypred)
      hr_array(fe_quantity′,coeff′,hypred′)
    end
  end
end

function Base.copyto!(a::NoHRParamArray,b::NoHRParamArray)
  copyto!(a.fecache,b.fecache)
  copyto!(a.rbfecache,b.rbfecache)
  copyto!(a.coeff,b.coeff)
  copyto!(a.hypred,b.hypred)
  a
end

function Base.fill!(a::NoHRParamArray,b::Number)
  fill!(a.fecache,b)
  fill!(a.rbfecache,b)
  fill!(a.coeff,b)
  fill!(a.hypred,b)
end

function Base.copyto!(a::HRParamArray,b::HRParamArray)
  copyto!(a.fecache,b.fecache)
  copyto!(a.coeff,b.coeff)
  copyto!(a.hypred,b.hypred)
  a
end

function Base.fill!(a::HRParamArray,b::Number)
  fill!(a.fecache,b)
  fill!(a.coeff,b)
  fill!(a.hypred,b)
end

function LinearAlgebra.fillstored!(a::AbstractHRParamArray,b::Number)
  fill!(a,b)
end

function Base.fill!(a::ArrayBlock,b::Number)
  for i in eachindex(a)
    if a.touched[i]
      fill!(a.array[i],b)
    end
  end
end

function LinearAlgebra.rmul!(a::AbstractHRParamArray,b::Number)
  rmul!(a.hypred,b)
end

function LinearAlgebra.axpy!(α::Number,a::AbstractHRParamArray,b::AbstractHRParamArray)
  axpy!(α,a.hypred,b.hypred)
end

function LinearAlgebra.axpy!(α::Number,a::AbstractHRParamArray,b::ParamArray)
  axpy!(α,a.hypred,b)
end

function LinearAlgebra.norm(a::AbstractHRParamArray)
  norm(a.hypred)
end

function Utils.change_domains(a::AbstractHRParamArray,trians)
  fecache = change_domains(a.fecache,trians)
  coeff = change_domains(a.coeff,trians)
  hypred = a.hypred
  hr_array(fecache,coeff,hypred)
end

function ParamAlgebra.compatible_cache(a::AbstractHRParamArray,b::AbstractHRParamArray)
  hypred′ = compatible_cache(a.hypred,b.hypred)
  hr_array(a.fecache,a.coeff,hypred′)
end

function Utils.change_domains(a::NoHRParamArray,trians)
  fecache = change_domains(a.fecache,trians)
  rbfecache = change_domains(a.rbfecache,trians)
  coeff = change_domains(a.coeff,trians)
  hypred = a.hypred
  nohr_array(fecache,rbfecache,coeff,hypred)
end

function ParamAlgebra.compatible_cache(a::NoHRParamArray,b::NoHRParamArray)
  hypred′ = compatible_cache(a.hypred,b.hypred)
  nohr_array(a.fecache,a.rbfecache,a.coeff,hypred′)
end