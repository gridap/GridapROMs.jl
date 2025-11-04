"""
    TrialParamFESpace{S} <: SingleFieldParamFESpace{S}

Trial FE space equipped with parametric dirichlet values
"""
struct TrialParamFESpace{S} <: SingleFieldParamFESpace{S}
  dirichlet_values::AbstractParamVector
  space::S
  function TrialParamFESpace(dirichlet_values::AbstractParamVector,space::SingleFieldFESpace)
    new{typeof(space)}(dirichlet_values,space)
  end
end

function TrialParamFESpace(U::SingleFieldFESpace)
  U
end

function TrialParamFESpace(dirichlet_values::AbstractVector,space::SingleFieldFESpace)
  TrialFESpace(dirichlet_values,space)
end

function TrialParamFESpace(space::SingleFieldFESpace,objects)
  dirichlet_values = [compute_dirichlet_values_for_tags(space,objects[i]) for i = param_eachindex(objects)]
  TrialParamFESpace(ConsecutiveParamArray(dirichlet_values),space)
end

"""
    TrialParamFESpace!(dir_values::AbstractParamVector,space::SingleFieldFESpace,objects
      ) -> TrialParamFESpace

Allows do-block syntax for the construction of a [`TrialParamFESpace`](@ref)
"""
function TrialParamFESpace!(dir_values::AbstractParamVector,space::SingleFieldFESpace,objects)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  TrialParamFESpace!(dir_values,space)
end

function TrialParamFESpace!(space::TrialParamFESpace,objects)
  dir_values = get_dirichlet_dof_values(space)
  dir_values_scratch = zero_dirichlet_values(space)
  dir_values = compute_dirichlet_values_for_tags!(dir_values,dir_values_scratch,space,objects)
  space
end

# Allow do-block syntax

function TrialParamFESpace(f::Function,space::SingleFieldFESpace)
  TrialParamFESpace(space,f)
end

function TrialParamFESpace!(f::Function,dir_values::AbstractParamVector,space::SingleFieldFESpace)
  TrialParamFESpace!(dir_values,space,f)
end

function TrialParamFESpace!(f::Function,space::TrialParamFESpace)
  TrialParamFESpace!(space,f)
end

"""
    HomogeneousTrialParamFESpace(U::SingleFieldFESpace,plength::Int) -> TrialParamFESpace

Returns a [`TrialParamFESpace`](@ref) equipped with homogeneous parametric
dirichlet values
"""
function HomogeneousTrialParamFESpace(U::SingleFieldFESpace,plength::Int)
  dv = zero_dirichlet_values(U)
  dirichlet_values = parameterize(dv,plength)
  TrialParamFESpace(dirichlet_values,U)
end

function HomogeneousTrialParamFESpace!(dirichlet_values::AbstractParamVector,U::SingleFieldFESpace,args...)
  fill!(dirichlet_values,zero(eltype(dirichlet_values)))
  TrialParamFESpace(dirichlet_values,U)
end

FESpaces.get_fe_space(f::TrialParamFESpace) = f.space

FESpaces.get_dirichlet_dof_values(f::TrialParamFESpace) = f.dirichlet_values

ParamDataStructures.param_length(f::TrialParamFESpace) = param_length(f.dirichlet_values)

function ParamDataStructures.param_getindex(f::TrialParamFESpace,index::Integer)
  dv = param_getindex(f.dirichlet_values,index)
  TrialFESpace(dv,f.space)
end

# utils

remove_layer(f::TrialParamFESpace) = TrialParamFESpace(f.dirichlet_values,f.space.space)
remove_layer(f::TrialParamFESpace{<:OrderedFESpace}) = TrialParamFESpace(f.dirichlet_values,remove_layer(f))
