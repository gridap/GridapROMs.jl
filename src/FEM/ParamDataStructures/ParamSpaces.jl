"""
    abstract type AbstractRealisation end

Type representing parametric realisations, i.e. samples extracted from a given
parameter space.
Subtypes:
- [`Realisation`](@ref)
- [`TransientRealisation`](@ref)
"""
abstract type AbstractRealisation end

"""
    get_params(r::AbstractRealisation) -> Realisation
"""
get_params(r::AbstractRealisation) = @abstractmethod

# this function should stay local
_get_params(r::AbstractRealisation) = @abstractmethod

"""
    num_params(r::AbstractRealisation) -> Int
"""
num_params(r::AbstractRealisation) = length(_get_params(r))

param_length(r::AbstractRealisation) = length(r)

param_getindex(r::AbstractRealisation,i::Integer) = getindex(r,i)

"""
    struct Realisation{P<:AbstractVector} <: AbstractRealisation
      params::P
    end

Represents standard parametric realisations, i.e. samples extracted from
a given parameter space. The field `params` is most commonly a vector of vectors.
When the parameters are scalars, they still need to be defined as vectors of
vectors of unit length. In other words, we treat the case in which `params` is a
vector of numbers as the case in which `params` is a vector of one vector.
"""
struct Realisation{P<:AbstractVector} <: AbstractRealisation
  params::P
end

const TrivialRealisation = Realisation{<:AbstractVector{<:Real}}

get_params(r::Realisation) = r # we only want to deal with a Realisation type
_get_params(r::Realisation) = r.params # this function should stay local
_get_params(r::TrivialRealisation) = [r.params] # this function should stay local

Base.length(r::Realisation) = num_params(r)

Base.getindex(r::Realisation,i) = Realisation(getindex(_get_params(r),i))

# when iterating over a Realisation{P}, we return eltype(P) ∀ index i
function Base.iterate(r::Realisation,state=1)
  if state > length(r)
    return nothing
  end
  rstate = _get_params(r)[state]
  return rstate, state+1
end

function Base.zero(r::Realisation)
  μ1 = first(_get_params(r))
  Realisation(zeros(eltype(μ1),length(μ1)) .+ eps())
end

function param_cat(r::Vector{Realisation{P}}) where P
  n = sum(param_length(ri) for ri in r)
  params = P(undef,n)
  count = 0
  for ri in r
    for μ in ri
      count += 1
      params[count] = μ
    end
  end
  Realisation(params)
end

"""
    abstract type TransientRealisation{P<:Realisation,T<:Real} <: AbstractRealisation end

Represents temporal parametric realisations, i.e. samples extracted from
a given parameter space for every time step in a temporal range. The most obvious
application of this type are transient PDEs, where an initial condition is given.
Following this convention, the initial time instant is kept separate from the
other time steps.
"""
abstract type TransientRealisation{P<:Realisation,T<:Real} <: AbstractRealisation end

"""
    get_times(r::TransientRealisation) -> Any
"""
get_times(r::TransientRealisation) = @abstractmethod

"""
    get_times(r::TransientRealisation) -> Int
"""
num_times(r::TransientRealisation) = length(get_times(r))

Base.length(r::TransientRealisation) = num_params(r)*num_times(r)

Base.getindex(r::TransientRealisation,i) = getindex(r,i,:)

"""
    struct GenericTransientRealisation{P,T,A} <: TransientRealisation{P,T}
      params::P
      times::A
      t0::T
    end

Most standard implementation of a `TransientRealisation`.
"""
struct GenericTransientRealisation{P,T,A} <: TransientRealisation{P,T}
  params::P
  times::A
  t0::T
end

get_params(r::GenericTransientRealisation) = get_params(r.params)
_get_params(r::GenericTransientRealisation) = _get_params(r.params)
get_times(r::GenericTransientRealisation) = r.times

function TransientRealisation(params::Realisation,times::AbstractVector{<:Real},t0::Real)
  GenericTransientRealisation(params,times,t0)
end

function TransientRealisation(params::Realisation,time::Real,args...)
  TransientRealisation(params,[time],args...)
end

function TransientRealisation(params::Realisation,times::AbstractVector{<:Real})
  t0,inner_times... = times
  TransientRealisation(params,inner_times,t0)
end

function Base.getindex(r::GenericTransientRealisation,i,j)
  TransientRealisation(
    getindex(get_params(r),i),
    getindex(get_times(r),j),
    r.t0)
end

function Base.iterate(r::GenericTransientRealisation,state...)
  iterator = Iterators.product(_get_params(r),get_times(r))
  iterate(iterator,state...)
end

function Base.zero(r::GenericTransientRealisation)
  GenericTransientRealisation(zero(get_params(r)),get_times(r),get_initial_time(r))
end

function param_cat(r::Vector{<:GenericTransientRealisation})
  r1 = first(r)
  times = get_times(r1)
  t0 = r1.t0
  @check all(get_times(ri)==times && ri.t0==t0 for ri in r)
  params = param_cat(map(get_params,r))
  GenericTransientRealisation(params,times,t0)
end

"""
    get_initial_time(r::GenericTransientRealisation) -> Real
"""
get_initial_time(r::GenericTransientRealisation) = r.t0

"""
    get_final_time(r::GenericTransientRealisation) -> Real
"""
get_final_time(r::GenericTransientRealisation) = last(get_times(r))

get_delta(r::GenericTransientRealisation) = (get_final_time(r) - get_initial_time(r)) / num_times(r)

"""
    shift!(r::TransientRealisation,δ::Real) -> Nothing

In-place uniform shifting by a constant `δ` of the temporal domain in the
realisation `r`
"""
function shift!(r::GenericTransientRealisation,δ::Real)
  r.times .+= δ
end

"""
    get_at_time(r::GenericTransientRealisation,time) -> TransientRealisationAt

Returns a [`TransientRealisationAt`](@ref) from a [`GenericTransientRealisation`](@ref)
at a time instant specified by `time`
"""
function get_at_time(r::GenericTransientRealisation,time=:initial)
  if time == :initial
    get_at_time(r,get_initial_time(r))
  elseif time == :final
    get_at_time(r,get_final_time(r))
  else
    @notimplemented
  end
end

function get_at_time(r::GenericTransientRealisation{P,T} where P,time::T) where T
  TransientRealisationAt(get_params(r),Ref(time))
end

function get_at_timestep(r::GenericTransientRealisation,timestep::Int)
  @check 0 <= timestep <= num_times(r)
  timestep == 0 ? get_at_time(r,:initial) : get_at_time(r,r.times[timestep])
end

"""
    struct TransientRealisationAt{P,T} <: TransientRealisation{P,T}
      params::P
      t::Base.RefValue{T}
    end

Represents a GenericTransientRealisation{P,T} at a certain time instant `t`.
To avoid making it a mutable struct, the time instant `t` is stored as a Base.RefValue{T}.
"""
struct TransientRealisationAt{P,T} <: TransientRealisation{P,T}
  params::P
  t::Base.RefValue{T}
end

get_params(r::TransientRealisationAt) = get_params(r.params)

_get_params(r::TransientRealisationAt) = _get_params(r.params)

get_times(r::TransientRealisationAt) = r.t[]

function Base.getindex(r::TransientRealisationAt,i,j)
  @assert j ∈ (1,Colon())
  new_param = getindex(get_params(r),i)
  TransientRealisationAt(new_param,r.t)
end

Base.iterate(r::TransientRealisationAt,i...) = iterate(r.params,i...)

function shift!(r::TransientRealisationAt,δ::Real)
  r.t[] += δ
end

"""
    struct ParamSpace{P<:AbstractVector{<:AbstractVector},S<:SamplingStyle} <: AbstractSet{Realisation}
      param_domain::P
      sampling_style::S
    end

Fields:
- `param_domain`: domain of definition of the parameters
- `sampling_style`: distribution on `param_domain` according to which we can
  sample the parameters (by default it is set to `HaltonSampling`)
"""
struct ParamSpace{P<:AbstractVector{<:AbstractVector},S<:SamplingStyle} <: AbstractSet{Realisation}
  param_domain::P
  sampling_style::S
end

get_sampling_style(p::ParamSpace) = p.sampling_style

function ParamSpace(domain_tuple::NTuple{N,T},style=HaltonSampling()) where {N,T}
  @notimplementedif !isconcretetype(T)
  @notimplementedif isodd(N)
  param_domain = Vector{Vector{T}}(undef,Int(N/2))
  for (i,n) in enumerate(1:2:N)
    param_domain[i] = [domain_tuple[n],domain_tuple[n+1]]
  end
  ParamSpace(param_domain,style)
end

function ParamSpace(param_domain;kwargs...)
  style = _sampling_to_style(;kwargs...)
  ParamSpace(param_domain,style)
end

function Base.show(io::IO,::MIME"text/plain",p::ParamSpace)
  msg = "Set of parameters in $(p.param_domain), sampled with $(p.sampling_style)"
  println(io,msg)
end

"""
    realisation(p::ParamSpace;nparams=1,sampling=get_sampling_style(p),kwargs...) -> Realisation
    realisation(p::TransientParamSpace;nparams=1,sampling=get_sampling_style(p),kwargs...) -> TransientRealisation

Extraction of a set of `nparams` parameters from a given parametric space, by
default according to the sampling strategy specified in `p`.
"""
function realisation(p::ParamSpace;nparams=1,sampling=get_sampling_style(p),kwargs...)
  params = _generate_params(sampling,p.param_domain,nparams;kwargs...)
  Realisation(params)
end

"""
    struct TransientParamSpace{P<:ParamSpace,T} <: AbstractSet{TransientRealisation}
      parametric_space::P
      temporal_domain::T
    end

Fields:
- `parametric_space`: underlying parameter space
- `temporal_domain`: underlying temporal space

It represents, in essence, the set of tuples (p,t) in `parametric_space` × `temporal_domain`
"""
struct TransientParamSpace{P<:ParamSpace,T} <: AbstractSet{TransientRealisation}
  parametric_space::P
  temporal_domain::T
end

function TransientParamSpace(
  param_domain::Union{Tuple,AbstractVector},
  temporal_domain::AbstractVector{<:Real},
  args...;
  kwargs...
  )

  parametric_space = ParamSpace(param_domain,args...;kwargs...)
  TransientParamSpace(parametric_space,temporal_domain)
end

function Base.show(io::IO,::MIME"text/plain",p::TransientParamSpace)
  msg = "Set of tuples (p,t) in $(p.parametric_space.param_domain) × $(p.temporal_domain)"
  println(io,msg)
end

function realisation(
  p::TransientParamSpace;
  time_locations=eachindex(p.temporal_domain),
  kwargs...
  )

  params = realisation(p.parametric_space;kwargs...)
  times = p.temporal_domain[time_locations]
  TransientRealisation(params,times)
end

function shift!(p::TransientParamSpace,δ::Real)
  p.temporal_domain .+= δ
end

"""
    abstract type AbstractParamFunction{P<:Realisation} <: Function end

Representation of parametric functions with domain a parametric space.
Subtypes:
- [`ParamFunction`](@ref)
- [`TransientParamFunction`](@ref)
"""
abstract type AbstractParamFunction{P<:Realisation} <: Function end

param_length(f::AbstractParamFunction) = length(f)
param_getindex(f::AbstractParamFunction,i::Integer) = getindex(f,i)
Arrays.testitem(f::AbstractParamFunction) = param_getindex(f,1)
Arrays.evaluate!(cache,f::AbstractParamFunction,x) = pteval(f,x)
Arrays.evaluate!(cache,f::AbstractParamFunction,x::CellPoint) = CellField(f,get_triangulation(x))(x)
(f::AbstractParamFunction)(x) = evaluate(f,x)

function Arrays.return_cache(f::AbstractParamFunction,x)
  v = return_value(f,x)
  V = eltype(v)
  plength = param_length(f)
  cache = Vector{V}(undef,plength)
  return cache
end

"""
    struct ParamFunction{F,P} <: AbstractParamFunction{P}
      fun::F
      params::P
    end

Representation of parametric functions with domain a parametric space. Given a
function `f` : Ω₁ × ... × Ωₙ × D, where D is a `ParamSpace`,
the evaluation of `f` in `μ ∈ D` returns the restriction of `f` to Ω₁ × ... × Ωₙ
"""
struct ParamFunction{F,P} <: AbstractParamFunction{P}
  fun::F
  params::P
end

function ParamFunction(f::Function,p::Union{AbstractArray,TransientRealisation})
  @notimplemented "Use a Realisation as a parameter input"
end

function ParamFunction(f::Function,pt::TransientRealisationAt)
  ParamFunction(f,get_params(pt))
end

function parameterise(a::Any,p::AbstractRealisation)
  parameterise(a,param_length(p))
end

"""
    parameterise(f::Function,r::Realisation) -> ParamFunction
    parameterise(f::Function,r::TransientRealisation) -> TransientParamFunction

Method that parameterises an input quantity by a realisation `r`
"""
function parameterise(f::Function,p::Realisation)
  ParamFunction(f,p)
end

get_params(f::ParamFunction) = get_params(f.params)
_get_params(f::ParamFunction) = _get_params(f.params)
num_params(f::ParamFunction) = length(_get_params(f))
Base.length(f::ParamFunction) = num_params(f)
Base.getindex(f::ParamFunction,i::Integer) = f.fun(_get_params(f)[i])

function Base.:*(f::ParamFunction,α::Number)
  _fun(μ) = x -> α*f.fun(μ)(x)
  ParamFunction(_fun,f.params)
end

Base.:*(α::Number,f::ParamFunction) = f*α

for op in (:(Fields.gradient),:(Fields.symmetric_gradient),:(Fields.divergence),
  :(Fields.curl),:(Fields.laplacian))
  @eval begin
    function ($op)(f::ParamFunction)
      _op(μ) = x -> $op(f.fun(μ))(x)
      ParamFunction(_op,f.params)
    end
  end
end

# when iterating over a ParamFunction{P}, we return return f(eltype(P)) ∀ index i
function pteval(f::ParamFunction,x)
  T = return_type(f,x)
  v = Vector{T}(undef,length(f))
  @inbounds for (i,μ) in enumerate(get_params(f))
    v[i] = f.fun(μ)(x)
  end
  return v
end

Arrays.return_value(f::ParamFunction,x) = f.fun(testitem(_get_params(f)))(x)

"""
    struct TransientParamFunction{F,P,T} <: AbstractParamFunction{P}
      fun::F
      params::P
      times::T
    end

Representation of parametric functions with domain a transient parametric space.
Given a function `f : Ω₁ × ... × Ωₙ × D × [t₁,t₂]`, where `[t₁,t₂]` is a
temporal domain and `D` is a `ParamSpace`, or equivalently
`f : Ω₁ × ... × Ωₙ × D × [t₁,t₂]`, where `D` is a
`TransientParamSpace`, the evaluation of `f` in `(μ,t) ∈ D × [t₁,t₂]`
returns the restriction of `f` to `Ω₁ × ... × Ωₙ`
"""
struct TransientParamFunction{F,P,T} <: AbstractParamFunction{P}
  fun::F
  params::P
  times::T
end

function TransientParamFunction(f::Function,p::AbstractArray,t)
  @notimplemented "Use a TransientRealisation as a parameter input"
end

function TransientParamFunction(f::Function,r::Realisation)
  @notimplemented "Use a TransientRealisation as a parameter input"
end

function TransientParamFunction(f::Function,r::TransientRealisation)
  TransientParamFunction(f,get_params(r),get_times(r))
end

function parameterise(f::Function,p::Realisation,t)
  TransientParamFunction(f,p,t)
end

function parameterise(f::Function,r::TransientRealisation)
  TransientParamFunction(f,r)
end

get_params(f::TransientParamFunction) = get_params(f.params)
_get_params(f::TransientParamFunction) = _get_params(f.params)
num_params(f::TransientParamFunction) = length(_get_params(f))
get_times(f::TransientParamFunction) = f.times
num_times(f::TransientParamFunction) = length(get_times(f))
Base.length(f::TransientParamFunction) = num_params(f)*num_times(f)
function Base.getindex(f::TransientParamFunction,i::Integer)
  np = num_params(f)
  p = _get_params(f)[fast_index(i,np)]
  t = get_times(f)[slow_index(i,np)]
  f.fun(p,t)
end

function Base.:*(f::TransientParamFunction,α::Number)
  _fun(μ,t) = x -> α*f.fun(μ,t)(x)
  TransientParamFunction(_fun,f.params,f.times)
end

Base.:*(α::Number,f::TransientParamFunction) = f*α

for op in (:(Fields.gradient),:(Fields.symmetric_gradient),:(Fields.divergence),
  :(Fields.curl),:(Fields.laplacian))
  @eval begin
    function ($op)(f::TransientParamFunction)
      _op(μ,t) = x -> $op(f.fun(μ,t))(x)
      TransientParamFunction(_op,f.params,f.times)
    end
  end
end

function pteval(f::TransientParamFunction,x)
  iterator = Iterators.product(_get_params(f),get_times(f))
  T = return_type(f,x)
  v = Vector{T}(undef,length(f))
  @inbounds for (i,(μ,t)) in enumerate(iterator)
    v[i] = f.fun(μ,t)(x)
  end
  return v
end

Arrays.return_value(f::TransientParamFunction,x) = f.fun(testitem(_get_params(f)),testitem(get_times(f)))(x)
