struct HRParamArray{T,N,A,B,C<:ParamArray{T,N}} <: ParamArray{T,N}
  fecache::A
  coeff::B
  hypred::C
end

Base.size(a::HRParamArray) = size(a.hypred)
Base.getindex(a::HRParamArray{T,N},i::Vararg{Integer,N}) where {T,N} = getindex(a.hypred,i...)
Base.setindex!(a::HRParamArray{T,N},v,i::Vararg{Integer,N}) where {T,N} = setindex!(a.hypred,v,i...)
ParamDataStructures.param_length(a::HRParamArray) = param_length(a.hypred)
ParamDataStructures.get_all_data(a::HRParamArray) = get_all_data(a.hypred)
ParamDataStructures.param_getindex(a::HRParamArray,i::Integer) = param_getindex(a.hypred,i)

for f in (:copy,:similar)
  @eval begin
    function Base.$f(a::HRParamArray)
      fe_quantity′ = Base.$f(a.fecache)
      coeff′ = Base.$f(a.coeff)
      hypred′ = Base.$f(a.hypred)
      HRParamArray(fe_quantity′,coeff′,hypred′)
    end
  end
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

function LinearAlgebra.fillstored!(a::HRParamArray,b::Number)
  fill!(a,b)
end

function LinearAlgebra.rmul!(a::HRParamArray,b::Number)
  rmul!(a.hypred,b)
end

function LinearAlgebra.axpy!(α::Number,a::HRParamArray,b::HRParamArray)
  axpy!(α,a.hypred,b.hypred)
end

function LinearAlgebra.axpy!(α::Number,a::HRParamArray,b::ParamArray)
  axpy!(α,a.hypred,b)
end

function LinearAlgebra.norm(a::HRParamArray,p::Real=2)
  norm(a.hypred,p)
end

function Utils.change_domains(a::HRParamArray,trians)
  fecache = change_domains(a.fecache,trians)
  coeff = change_domains(a.coeff,trians)
  hypred = a.hypred
  HRParamArray(fecache,coeff,hypred)
end

function ParamAlgebra.compatible_cache(a::HRParamArray,b::HRParamArray)
  hypred′ = compatible_cache(a.hypred,b.hypred)
  HRParamArray(a.fecache,a.coeff,hypred′)
end
