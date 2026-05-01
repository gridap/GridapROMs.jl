"""
    struct RBParamVector{T,A<:ParamVector{T},B} <: ParamArray{T,1}
      data::A
      fe_data::B
    end

Parametric vector obtained by applying a [`Projection`](@ref) on a high-dimensional
parametric FE vector `fe_data`, which is stored (but mostly unused) for conveniency
"""
struct RBParamVector{T,A<:ParamVector{T},B} <: ParamArray{T,1}
  data::A
  fe_data::B
end

function reduced_vector(data::AbstractParamVector,fe_data::AbstractParamVector)
  RBParamVector(data,fe_data)
end

Base.size(a::RBParamVector) = size(a.data)
Base.axes(a::RBParamVector) = axes(a.data)
Base.getindex(a::RBParamVector,i::Integer) = getindex(a.data,i)
Base.setindex!(a::RBParamVector,v,i::Integer) = setindex!(a.data,v,i)
ParamDataStructures.param_length(a::RBParamVector) = param_length(a.data)
ParamDataStructures.get_all_data(a::RBParamVector) = get_all_data(a.data)
ParamDataStructures.param_getindex(a::RBParamVector,i::Integer) = param_getindex(a.data,i)

function ParamDataStructures.param_cat(a::Vector{<:RBParamVector})
  data = param_cat(map(_data,a))
  fe_data = param_cat(map(_fe_data,a))
  RBParamVector(data,fe_data)
end

function Base.copy(a::RBParamVector)
  data′ = copy(a.data)
  fe_data′ = copy(a.fe_data)
  RBParamVector(data′,fe_data′)
end

function Base.similar(a::RBParamVector{R},::Type{S}) where {R,S<:AbstractVector}
  data′ = similar(a.data,S)
  fe_data′ = copy(a.fe_data)
  RBParamVector(data′,fe_data′)
end

function Base.similar(a::RBParamVector{R},::Type{S},dims::Dims{1}) where {R,S<:AbstractVector}
  data′ = similar(a.data,S,dims)
  fe_data′ = similar(a.fe_data,S,dims)
  RBParamVector(data′,fe_data′)
end

function Base.copyto!(a::RBParamVector,b::RBParamVector)
  copyto!(a.data,b.data)
  copyto!(a.fe_data,b.fe_data)
  a
end

function Base.fill!(a::RBParamVector,b::Number)
  fill!(a.data,b)
  return a
end

# multi field

function MultiField.restrict_to_field(f::MultiFieldFESpace,fv::RBParamVector,i::Integer)
  data_i = blocks(fv.data)[i]
  fe_data_i = MultiField.restrict_to_field(f,fv.fe_data,i)
  RBParamVector(data_i,fe_data_i)
end

for F in (:SingleFieldParamFESpace,:SingleFieldFESpace)
  @eval begin
    function FESpaces.scatter_free_and_dirichlet_values(f::$F,fv::RBParamVector,dv::AbstractParamVector)
      scatter_free_and_dirichlet_values(f,fv.fe_data,dv)
    end

    function FESpaces.gather_free_and_dirichlet_values!(fv::RBParamVector,dv::AbstractParamVector,f::$F,cv)
      gather_free_and_dirichlet_values!(fv.fe_data,dv,f,cv)
    end
  end
end

for T in (
  :FESpaceWithLinearConstraints,
  :(FESpaceWithConstantFixed{FESpaces.FixConstant}),
  :(FESpaceWithConstantFixed{FESpaces.DoNotFixConstant})
  )
  @eval begin
    function FESpaces.scatter_free_and_dirichlet_values(
      f::SingleFieldParamFESpace{<:$T},
      fv::RBParamVector,
      dv::AbstractParamVector
      )

      scatter_free_and_dirichlet_values(f,fv.fe_data,dv)
    end
    function FESpaces.scatter_free_and_dirichlet_values(
      f::SingleFieldParamFESpace{<:OrderedFESpace{<:$T}},
      fv::RBParamVector,
      dv::AbstractParamVector
      )

      scatter_free_and_dirichlet_values(f,fv.fe_data,dv)
    end
    function FESpaces.gather_free_and_dirichlet_values!(
      fv::RBParamVector,
      dv::AbstractParamVector,
      f::SingleFieldParamFESpace{<:$T},
      cv
      )

      gather_free_and_dirichlet_values!(fv.fe_data,dv,f,cv)
    end
    function FESpaces.gather_free_and_dirichlet_values!(
      fv::RBParamVector,
      dv::AbstractParamVector,
      f::SingleFieldParamFESpace{<:OrderedFESpace{<:$T}},
      cv
      )

      gather_free_and_dirichlet_values!(fv.fe_data,dv,f,cv)
    end
  end
end

function unfold(a::BlockParamVector{T,<:AbstractVector{<:RBParamVector{T}}}) where T
  data = mortar(map(_data,blocks(a)))
  fe_data = mortar(map(_fe_data,blocks(a)))
  RBParamVector(data,fe_data)
end

# utils

_data(a::RBParamVector) = a.data
_fe_data(a::RBParamVector) = a.fe_data
