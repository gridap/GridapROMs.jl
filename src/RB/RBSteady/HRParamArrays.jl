"""
    struct HRParamArray{T,N,A,B,C<:ParamArray{T,N}} <: ParamArray{T,N}
      fecache::A
      coeff::B
      hypred::C
    end

Parametric vector returned after the online phase of a hyper-reduction strategy.
Fields:

- `fecache`: represents a parametric residual/Jacobian computed via integration
  on an [`IntegrationDomain`](@ref)
- `coeff`: parameter-dependent coefficient computed during the online phase
  according to the formula

  `coeff = Φi⁻¹ fecache[i,:]`

  where `(Φi,i)` are the interpolation and the reduced integration domain stored
  in a `HyperReduction` object.
- `hypred`: the ouptut of the online phase of a hyper-reduction strategy, acoording
  to the formula

  `hypred = Φrb * coeff`

  where `Φrb` is the basis stored in a `HyperReduction` object
"""
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

for f in (:(Base.copy),:(Base.similar))
  @eval begin
    function $f(a::HRParamArray)
      fe_quantity′ = $f(a.fecache)
      coeff′ = $f(a.coeff)
      hypred′ = $f(a.hypred)
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

# this correction is needed

function Base.fill!(a::ArrayBlock,b::Number)
  for i in eachindex(a)
    if a.touched[i]
      fill!(a.array[i],b)
    end
  end
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

function LinearAlgebra.norm(a::HRParamArray)
  norm(a.hypred)
end

# utils

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
