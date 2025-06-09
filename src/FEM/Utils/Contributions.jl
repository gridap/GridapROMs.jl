"""
    struct Contribution{V}
      values::Tuple{Vararg{V}}
      trians::Tuple{Vararg{Triangulation}}
    end

Collection of values corresponding to a set of triangulations. Similarly to `DomainContribution`,
the values can be accessed by indexing the corresponding triangulation.
"""
struct Contribution{V}
  values::Tuple{Vararg{V}}
  trians::Tuple{Vararg{Triangulation}}
  function Contribution(values::Tuple{Vararg{V}},trians::Tuple{Vararg{Triangulation}}) where V
    @check length(values) == length(trians)
    @check !any([t === first(trians) for t = trians[2:end]])
    new{V}(values,trians)
  end
end

Contribution(v::V,t::Triangulation) where V = Contribution((v,),(t,))

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

function Base.getindex(a::Contribution,trian::Triangulation...)
  perm = find_trian_permutation(trian,a.trians)
  getindex(a,perm...)
end

function change_domains(a::Contribution,trians::Tuple{Vararg{Triangulation}})
  values = ()
  for (i,trian) in enumerate(trians)
    if i > length(a)
      valuei = similar(last(get_contributions(a)))
    else
      valuei = get_contributions(a)[i]
    end
    values = (values...,valuei)
  end
  Contribution(values,trians)
end

function set_domains(a::Contribution,trians::Tuple{Vararg{Triangulation}})
  get_contributions(a)
end

"""
    const ArrayContribution{T,N} = Contribution{<:AbstractArray{T,N}}

[`Contribution`](@ref) whose field `values` are `AbstractArray`s
"""
const ArrayContribution{T,N} = Contribution{<:AbstractArray{T,N}}

"""
    const VectorContribution{T} = ArrayContribution{T,1}
"""
const VectorContribution{T} = ArrayContribution{T,1}

"""
    const MatrixContribution{T} = ArrayContribution{T,2}
"""
const MatrixContribution{T} = ArrayContribution{T,2}

Base.eltype(::ArrayContribution{T}) where T = T
Base.eltype(::Type{<:ArrayContribution{T}}) where T = T
Base.ndims(::ArrayContribution{T,N}) where {T,N} = N
Base.ndims(::Type{<:ArrayContribution{T,N}}) where {T,N} = N
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
  α::Number,β::Number)

  for c in c.values, a in a.values
    mul!(c,a,b,α,β)
  end
  a
end

function LinearAlgebra.mul!(
  c::VectorContribution,
  a::MatrixContribution,
  b::VectorContribution,
  α::Number,β::Number)

  @check length(c) == length(b)
  for (c,b) in zip(c.values,b.values), a in a.values
    mul!(c,a,b,α,β)
  end
  a
end

function LinearAlgebra.axpy!(α::Number,a::ArrayContribution,b::ArrayContribution)
  @check length(a) == length(b)
  for (a,b) in (a.values,b.values)
    axpy!(α,a,b)
  end
  b
end

function Algebra.copy_entries!(a::ArrayContribution,b::ArrayContribution)
  @check length(a) == length(b)
  for (a,b) in zip(a.values,b.values)
    copy_entries!(a,b)
  end
  a
end

struct ContributionBroadcast{D,T}
  contrib::D
  trians::T
end

function Base.broadcasted(f,a::ArrayContribution,b::Number)
  ContributionBroadcast(map(values -> Base.broadcasted(f,values,b),a.values),a.trians)
end

function Base.materialize(c::ContributionBroadcast)
  Contribution(map(Base.materialize,c.contrib),c.trians)
end

function Base.materialize!(a::ArrayContribution,c::ContributionBroadcast)
  @check a.trians === c.trians
  map(Base.materialize!,a.values,c.contrib)
  a
end

"""
    const TupOfArrayContribution{T} = Tuple{Vararg{ArrayContribution{T}}}

Specifically allows to deal with tuples of Jacobians in unsteady settings
"""
const TupOfArrayContribution{T} = Tuple{Vararg{ArrayContribution{T}}}

Base.eltype(::TupOfArrayContribution{T}) where T = T
Base.eltype(::Type{<:TupOfArrayContribution{T}}) where T = T

function CellData.get_domains(a::TupOfArrayContribution)
  trians = ()
  for ai in a
    trians = (trians...,CellData.get_domains(ai))
  end
  trians
end

function get_contributions(a::TupOfArrayContribution)
  values = ()
  for ai in a
    values = (values...,get_contributions(ai))
  end
  values
end

for f in (:(Base.copy),:(Base.similar))
  @eval begin
    function $f(a::TupOfArrayContribution)
      b = ()
      for ai in a
        b = (b...,$f(ai))
      end
      b
    end
  end
end

function Base.fill!(a::TupOfArrayContribution,v)
  for ai in a
    LinearAlgebra.fill!(ai,v)
  end
  a
end

function LinearAlgebra.fillstored!(a::TupOfArrayContribution,v)
  for ai in a
    LinearAlgebra.fillstored!(ai,v)
  end
  a
end

function Algebra.copy_entries!(a::TupOfArrayContribution,b::TupOfArrayContribution)
  @check length(a) == length(b)
  for (a,b) in zip(a,b)
    copy_entries!(a,b)
  end
  a
end

for f in (:change_domains,:set_domains)
  @eval begin
    function $f(
      a::TupOfArrayContribution,
      trians::Tuple{Vararg{Tuple{Vararg{Triangulation}}}})

      @check length(a) == length(trians)
      b = ()
      for (ai,ti) in zip(a,trians)
        b = (b...,$f(ai,ti))
      end
      b
    end
  end
end
