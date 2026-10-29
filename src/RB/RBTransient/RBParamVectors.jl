"""
    struct ODERBParamVector{T,A<:RBParamVector{T},B<:Tuple{Vararg{RBParamVector{T}}}} <: ParamArray{T,1}
      data::A
      data0::B
    end

Solution of a transient reduced problem, `data`, together with its initial
conditions `data0`
"""
struct ODERBParamVector{T,A<:RBParamVector{T},B<:Tuple{Vararg{RBParamVector{T}}}} <: ParamArray{T,1}
  data::A
  data0::B
end

Base.size(a::ODERBParamVector) = size(a.data)
Base.axes(a::ODERBParamVector) = axes(a.data)
Base.getindex(a::ODERBParamVector,i::Integer) = getindex(a.data,i)
Base.setindex!(a::ODERBParamVector,v,i::Integer) = setindex!(a.data,v,i)
ParamDataStructures.param_length(a::ODERBParamVector) = param_length(a.data)
ParamDataStructures.get_all_data(a::ODERBParamVector) = get_all_data(a.data)
ParamDataStructures.param_getindex(a::ODERBParamVector,i::Integer) = param_getindex(a.data,i)

Base.collect(a::ODERBParamVector) = a.data
ParamODEs.initial_conditions(a::ODERBParamVector) = a.data0

function ParamDataStructures.param_cat(a::Vector{<:ODERBParamVector})
  data = param_cat(map(x -> x.data,a))
  n = length(first(a).data0)
  data0 = ntuple(i -> param_cat(map(x -> x.data0[i],a)),n)
  ODERBParamVector(data,data0)
end

function _permutelastdims(a::ODERBParamVector;kwargs...)
  data′ = _permutelastdims(a.data;kwargs...)
  ODERBParamVector(data′,a.data0)
end

function ParamDataStructures.Snapshots(a::RBParamVector,a0::Tuple{Vararg{RBParamVector}},r::TransientRealisation)
  Snapshots(a.fe_data,map(x -> x.fe_data,a0),r)
end

for I in (:AbstractDofMap,:TrivialDofMap,:(AbstractArray{<:AbstractDofMap}))
  @eval begin
    function ParamDataStructures.Snapshots(a::RBParamVector,a0::Tuple{Vararg{RBParamVector}},i::$I,r::TransientRealisation)
      Snapshots(a.fe_data,map(x -> x.fe_data,a0),i,r)
    end
  end
end