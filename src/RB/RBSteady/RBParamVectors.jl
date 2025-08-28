"""
    struct RBVector{T,A<:AbstractVector{T},B} <: AbstractVector{T}
      data::A
      fe_data::B
    end

Vector obtained by applying a [`Projection`](@ref) on a high-dimensional FE vector
`fe_data`, which is stored (but mostly unused) for conveniency
"""
struct RBVector{T,A<:AbstractVector{T},B} <: AbstractVector{T}
  data::A
  fe_data::B
end

function reduced_vector(data::AbstractVector,fe_data::AbstractVector)
  RBVector(data,fe_data)
end

Base.size(a::RBVector) = size(a.data)
Base.getindex(a::RBVector,i::Integer) = getindex(a.data,i)
Base.setindex!(a::RBVector,v,i::Integer) = setindex!(a.data,v,i)

function Algebra.allocate_vector(::Type{V},n::Int) where V<:RBVector
  allocate_vector(Vector{eltype(V)},n)
end

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

const AbstractRBVector{T} = Union{<:RBVector{T},<:RBParamVector{T}}

Base.size(a::RBParamVector) = size(a.data)
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

for T in (:RBParamVector,:RBVector)
  @eval begin
    function Base.copy(a::$T)
      data′ = copy(a.data)
      fe_data′ = copy(a.fe_data)
      RBParamVector(data′,fe_data′)
    end

    function Base.similar(a::$T{R},::Type{S}) where {R,S<:AbstractVector}
      data′ = similar(a.data,S)
      fe_data′ = copy(a.fe_data)
      $T(data′,fe_data′)
    end

    function Base.similar(a::$T{R},::Type{S},dims::Dims{1}) where {R,S<:AbstractVector}
      data′ = similar(a.data,S,dims)
      fe_data′ = similar(a.fe_data,S,dims)
      $T(data′,fe_data′)
    end

    function Base.copyto!(a::$T,b::$T)
      copyto!(a.data,b.data)
      copyto!(a.fe_data,b.fe_data)
      a
    end

    function Base.fill!(a::$T,b::Number)
      fill!(a.data,b)
      return a
    end

    # multi field

    function MultiField.restrict_to_field(f::MultiFieldFESpace,fv::$T,i::Integer)
      data_i = blocks(fv.data)[i]
      fe_data_i = MultiField.restrict_to_field(f,fv.fe_data,i)
      $T(data_i,fe_data_i)
    end
  end
end

for (F,S,T) in zip(
  (:SingleFieldParamFESpace,:SingleFieldFESpace),
  (:RBParamVector,:RBVector),
  (:AbstractParamVector,:AbstractVector))
  @eval begin
    function FESpaces.scatter_free_and_dirichlet_values(f::$F,fv::$S,dv::$T)
      scatter_free_and_dirichlet_values(f,fv.fe_data,dv)
    end

    function FESpaces.gather_free_and_dirichlet_values!(fv::$S,dv::$T,f::$F,cv)
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
      dv::AbstractParamVector)

      scatter_free_and_dirichlet_values(f,fv.fe_data,dv)
    end
    function FESpaces.scatter_free_and_dirichlet_values(
      f::SingleFieldParamFESpace{<:OrderedFESpace{<:$T}},
      fv::RBParamVector,
      dv::AbstractParamVector)

      scatter_free_and_dirichlet_values(f,fv.fe_data,dv)
    end
    function FESpaces.gather_free_and_dirichlet_values!(
      fv::RBParamVector,
      dv::AbstractParamVector,
      f::SingleFieldParamFESpace{<:$T},
      cv)

      gather_free_and_dirichlet_values!(fv.fe_data,dv,f,cv)
    end
    function FESpaces.gather_free_and_dirichlet_values!(
      fv::RBParamVector,
      dv::AbstractParamVector,
      f::SingleFieldParamFESpace{<:OrderedFESpace{<:$T}},
      cv)

      gather_free_and_dirichlet_values!(fv.fe_data,dv,f,cv)
    end
  end
end

function ParamDataStructures.parameterize(a::RBVector,plength::Int)
  data = global_parameterize(a.data,plength)
  fe_data = global_parameterize(a.fe_data,plength)
  RBParamVector(data,fe_data)
end

function unfold(a::BlockVector{T,<:AbstractVector{<:RBVector{T}}}) where T

  ax = axes(a)
  data = mortar(map(_data,blocks(a)))
  fe_data = mortar(map(_fe_data,blocks(a)))
  RBVector(data,fe_data)
end

function unfold(a::BlockParamVector{T,<:AbstractVector{<:RBParamVector{T}}}) where T

  data = mortar(map(_data,blocks(a)))
  fe_data = mortar(map(_fe_data,blocks(a)))
  RBParamVector(data,fe_data)
end

# utils

_data(a::RBVector) = a.data
_fe_data(a::RBVector) = a.fe_data

_data(a::RBParamVector) = a.data
_fe_data(a::RBParamVector) = a.fe_data
