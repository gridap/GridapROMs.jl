"""
    TrivialParamFESpace{S} <: SingleFieldParamFESpace{S}

Wrapper for non-parametric FE spaces that we wish assumed a parametric length
"""
struct TrivialParamFESpace{S} <: SingleFieldParamFESpace{S}
  space::S
  plength::Int
end

FESpaces.get_fe_space(f::TrivialParamFESpace) = f.space

ParamDataStructures.param_length(f::TrivialParamFESpace) = f.plength
ParamDataStructures.param_getindex(f::TrivialParamFESpace,index::Integer) = f.space

function ParamDataStructures.parameterize(f::SingleFieldFESpace,plength::Int)
  TrivialParamFESpace(f,plength)
end

function ParamDataStructures.parameterize(f::SingleFieldParamFESpace,plength::Int)
  @check param_length(f) == plength
  f
end

function FESpaces.TrialFESpace(tf::TrivialParamFESpace)
  f = tf.space
  U = TrialFESpace(f)
  TrivialParamFESpace(U,param_length(tf))
end

# utils

remove_layer(f::TrivialParamFESpace) = TrivialParamFESpace(f.space.space,param_length(f))
remove_layer(f::TrivialParamFESpace{<:OrderedFESpace}) = TrialParamFESpace(remove_layer(f),param_length(f))
