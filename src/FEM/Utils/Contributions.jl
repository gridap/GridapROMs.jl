"""
    struct Contribution{V,T}
      values::Tuple{Vararg{V}}
      trians::Tuple{Vararg{T}}
    end

Collection of values corresponding to a set of triangulations. Similarly to `DomainContribution`,
the values can be accessed by indexing the corresponding triangulation.
"""
struct Contribution{V,T}
  values::Tuple{Vararg{V}}
  trians::Tuple{Vararg{T}}
  function Contribution(values::Tuple,trians::Tuple)
    @check length(values) == length(trians)
    V = eltype(values)
    T = eltype(trians)
    new{V,T}(values,trians)
  end
end

Contribution(v,t) = Contribution((v,),(t,))

CellData.get_domains(a::Contribution) = a.trians

get_contributions(a::Contribution) = a.values

Base.length(a::Contribution) = length(a.values)
Base.size(a::Contribution,i...) = size(a.values,i...)
Base.getindex(a::Contribution,i...) = a.values[i...]
Base.setindex!(a::Contribution,v,i...) = a.values[i...] = v
Base.eachindex(a::Contribution) = eachindex(a.values)

"""
    contribution(f,trians) -> Contribution

Constructor of a [`Contribution`](@ref) that allows do-block syntax. `f` is a
function such that


`values[i] = f(trians[i]) for i...`


This constructor first builds the tuple of values, then builds the `Contribution`
object from `values` and `trians`
"""
@inline function contribution(f,trians)
  values = map(f,trians)
  Contribution(values,trians)
end

function contribution!(a,values)
  a.values .= values
end

function contribution!(a,f,trians)
  contribution!(a,map(f,trians))
end

function Base.getindex(a::Contribution{V,T},trian::T...) where {V,T}
  perm = find_trian_permutation(trian,a.trians)
  getindex(a,perm...)
end

function change_domains(a::Contribution,trians::Tuple)
  values = ()
  for i in eachindex(trians)
    if i > length(a)
      valuei = similar(last(get_contributions(a)))
    else
      valuei = get_contributions(a)[i]
    end
    values = (values...,valuei)
  end
  Contribution(values,trians)
end

function set_domains(a::Contribution,trians::Tuple)
  Contribution(get_contributions(a),trians)
end

"""
    const ArrayContribution{T,N} = Contribution{<:Union{AbstractArray{T,N},ArrayBlock{T,N}}}

[`Contribution`](@ref) whose field `values` are `AbstractArray`s
"""
const ArrayContribution{T,N} = Contribution{<:Union{AbstractArray{T,N},ArrayBlock{T,N}}}

"""
    const VectorContribution{T} = ArrayContribution{T,1}
"""
const VectorContribution{T} = ArrayContribution{T,1}

"""
    const MatrixContribution{T} = ArrayContribution{T,2}
"""
const MatrixContribution{T} = ArrayContribution{T,2}

Base.eltype(::Type{<:ArrayContribution{T}}) where T = T
Base.ndims(::ArrayContribution{<:Any,N}) where N = N
Base.ndims(::Type{<:ArrayContribution{<:Any,N}}) where N = N
Base.copy(a::ArrayContribution) = Contribution(copy.(a.values),a.trians)
Base.similar(a::ArrayContribution) = Contribution(similar.(a.values),a.trians)
Base.copyto!(a::ArrayContribution,b::ArrayContribution) = map(copyto!,a.values,b.values)

Base.sum(a::ArrayContribution) = sum(a.values)

function Base.fill!(a::ArrayContribution,v)
  for vals in a.values
    fill!(vals,v)
  end
  a
end

function LinearAlgebra.fillstored!(a::ArrayContribution,v)
  for vals in a.values
    LinearAlgebra.fillstored!(vals,v)
  end
  a
end

function LinearAlgebra.mul!(
  c::VectorContribution,
  a::MatrixContribution,
  b::AbstractVector,
  α::Number,β::Number
  )

  @check length(c) == length(a)
  for (ci,ai) in zip(c.values,a.values)
    mul!(ci,ai,b,α,β)
  end
  c
end

function LinearAlgebra.mul!(
  c::VectorContribution,
  a::MatrixContribution,
  b::VectorContribution,
  α::Number,β::Number
  )

  @check length(c) == length(a) == length(b)
  for (ci,ai,bi) in zip(c.values,a.values,b.values)
    mul!(ci,ai,bi,α,β)
  end
  c
end

function LinearAlgebra.axpy!(α::Number,a::ArrayContribution,b::ArrayContribution)
  @check length(a) == length(b)
  for (ai,bi) in zip(a.values,b.values)
    axpy!(α,ai,bi)
  end
  b
end

function Algebra.copy_entries!(a::ArrayContribution,b::ArrayContribution)
  @check length(a) == length(b)
  for (ai,bi) in zip(a.values,b.values)
    copy_entries!(ai,bi)
  end
  a
end

"""
    struct ContributionTuple{N,C}
      array::NTuple{N,C}
    end

Concrete wrapper around a tuple of [`Contribution`](@ref)s (e.g. one per time
derivative order in unsteady settings, as in [`ArrayContributionTuple`](@ref)).
"""
struct ContributionTuple{N,C}
  array::NTuple{N,C}
end

ContributionTuple(cs::Contribution...) = ContributionTuple(cs)

Base.length(a::ContributionTuple) = length(a.array)
Base.size(a::ContributionTuple) = size(a.array)
Base.iterate(a::ContributionTuple,state...) = iterate(a.array,state...)
Base.getindex(a::ContributionTuple,i::Integer) = a.array[i]
Base.eachindex(a::ContributionTuple) = eachindex(a.array)
Base.firstindex(a::ContributionTuple) = firstindex(a.array)
Base.map(f,a::ContributionTuple) = ContributionTuple(map(f,a.array))
Base.lastindex(a::ContributionTuple) = lastindex(a.array)

"""
    const ArrayContributionTuple{T} = ContributionTuple{N,<:ArrayContribution{T}} where N

Specifically allows to deal with tuples of Jacobians in unsteady settings
"""
const ArrayContributionTuple{T} = ContributionTuple{N,<:ArrayContribution{T}} where N

Base.eltype(::ArrayContributionTuple{T}) where T = T
Base.eltype(::Type{<:ArrayContributionTuple{T}}) where T = T

function CellData.get_domains(a::ArrayContributionTuple)
  trians = ()
  for ai in a
    trians = (trians...,CellData.get_domains(ai))
  end
  trians
end

function get_contributions(a::ArrayContributionTuple)
  values = ()
  for ai in a
    values = (values...,get_contributions(ai))
  end
  values
end

for f in (:copy,:similar)
  @eval begin
    function Base.$f(a::ArrayContributionTuple)
      b = ()
      for ai in a
        b = (b...,Base.$f(ai))
      end
      ContributionTuple(b)
    end
  end
end

function Base.fill!(a::ArrayContributionTuple,v)
  for ai in a
    LinearAlgebra.fill!(ai,v)
  end
  a
end

function LinearAlgebra.fillstored!(a::ArrayContributionTuple,v)
  for ai in a
    LinearAlgebra.fillstored!(ai,v)
  end
  a
end

function Algebra.copy_entries!(a::ArrayContributionTuple,b::ArrayContributionTuple)
  @check length(a) == length(b)
  for (ai,bi) in zip(a,b)
    copy_entries!(ai,bi)
  end
  a
end

for f in (:change_domains,:set_domains)
  @eval begin
    function $f(a::ArrayContributionTuple,trians::Tuple)
      @check length(a) == length(trians)
      b = ()
      for (ai,ti) in zip(a,trians)
        b = (b...,$f(ai,ti))
      end
      ContributionTuple(b)
    end
  end
end
