@doc raw"""
    abstract type TimeCombination end

Encodes how an ODE time-marching scheme combines contributions from different
time levels when building a reduced space-time system.

For every ODE solver there is exactly one `TimeCombination` that stores the
scheme parameters (time step, weights, …).  Use the constructor

    TimeCombination(odesolver::ODESolver)

to obtain the appropriate concrete subtype:

| Solver               | `TimeCombination` subtype    |
|:---------------------|:-----------------------------|
| `ThetaMethod`        | [`ThetaMethodCombination`](@ref) |
| `GeneralizedAlpha1`  | [`GenAlpha1Combination`](@ref)   |
| `GeneralizedAlpha2`  | [`GenAlpha2Combination`](@ref)   |

The per-order time-level weights are accessed through
[`get_coefficients`](@ref), while [`get_time_combination`](@ref) applies the
full combination to a parametric solution vector.

See also: [`CombinationOrder`](@ref), [`HighDimHyperReduction`](@ref).
"""
abstract type TimeCombination end

TimeCombination(odesolver::ODESolver) = @abstractmethod

"""
    get_coefficients(c::TimeCombination, args...) -> NTuple{N,Real}

Return the tuple of coefficients that the combination `c` uses to weight
snapshots at successive time levels. The length of the tuple equals the
number of time levels involved (the *stencil width* of the scheme for the
corresponding [`CombinationOrder`](@ref)).
"""
get_coefficients(c::TimeCombination,args...) = @abstractmethod

"""
    get_time_combination(c::TimeCombination, u, us0) -> NTuple{N,AbstractParamVector}

Apply the time combination `c` to the parametric solution vector `u` and the
`N` initial-condition vectors `us0`.  Returns an `N`-tuple of combined
vectors, one per derivative order of the ODE (e.g. ``u_{\\theta}`` and
``\\dot{u}_{\\theta}`` for a first-order scheme).
"""
function get_time_combination(
  c::TimeCombination,
  u::AbstractParamVector,
  us0::NTuple{N,AbstractParamVector}
  ) where N

  usx = ntuple(_ -> similar(u),Val{N}())
  for i in eachindex(us0)
    tcomb_i = CombinationOrder{i}(c)
    _combination!(usx[i],tcomb_i,u,us0)
  end
  return usx
end

@doc raw"""
    struct ThetaMethodCombination <: TimeCombination
      dt::Real
      θ::Real
    end

[`TimeCombination`](@ref) for the θ-method applied to a first-order ODE.
Stores the time step `dt` and the implicitness parameter `θ`.

Two [`CombinationOrder`](@ref) levels are defined:
- Order 1 (stiffness ``A``):  coefficients ``(\theta,\, 1-\theta)``.
- Order 2 (mass ``M``):       coefficients ``(1/\Delta t,\, -1/\Delta t)``.
"""
struct ThetaMethodCombination <: TimeCombination
  dt::Real
  θ::Real
end

function TimeCombination(odesolver::ThetaMethod)
  ThetaMethodCombination(odesolver.dt,odesolver.θ)
end

"""
    struct GenAlpha1Combination <: TimeCombination
      dt::Real
      αf::Real
      αm::Real
      γ::Real
    end

[`TimeCombination`](@ref) for the Generalized-α method for first-order ODEs.
Stores `dt`, `αf`, `αm`, and `γ`.

Two [`CombinationOrder`](@ref) levels are defined (stiffness and mass/damping).
"""
struct GenAlpha1Combination <: TimeCombination
  dt::Real
  αf::Real
  αm::Real
  γ::Real
end

function TimeCombination(odesolver::GeneralizedAlpha1)
  GenAlpha1Combination(
    odesolver.dt,
    odesolver.αf,
    odesolver.αm,
    odesolver.γ
  )
end

"""
    struct GenAlpha2Combination <: TimeCombination
      dt::Real
      αf::Real
      αm::Real
      γ::Real
      β::Real
    end

[`TimeCombination`](@ref) for the Generalized-α method for second-order ODEs
(Newmark family).  Stores `dt`, `αf`, `αm`, `γ`, and `β`.

Three [`CombinationOrder`](@ref) levels are defined (stiffness, damping, mass).
"""
struct GenAlpha2Combination <: TimeCombination
  dt::Real
  αf::Real
  αm::Real
  γ::Real
  β::Real
end

function TimeCombination(odesolver::GeneralizedAlpha2)
  GenAlpha2Combination(
    odesolver.dt,
    odesolver.αf,
    odesolver.αm,
    odesolver.γ,
    odesolver.β
  )
end

@doc raw"""
    struct CombinationOrder{A<:TimeCombination, N} <: TimeCombination
      combination::A
    end

Wraps a [`TimeCombination`](@ref) and selects the *N*-th derivative order
for coefficient retrieval. In a ``p``-th order ODE, the time-marching scheme
produces ``p+1`` operators (e.g. stiffness, damping, mass for ``p=2``);
`CombinationOrder{A,N}` isolates the coefficients for the ``N``-th one.

The type parameter `N` is a positive integer:
- `N = 1` — zeroth-derivative operator (stiffness ``A``).
- `N = 2` — first-derivative operator (mass ``M`` for first-order; damping for second-order).
- `N = 3` — second-derivative operator (mass ``M`` for second-order).

Convenience aliases:
- [`ThetaMethodStrategy{N}`](@ref)  = `CombinationOrder{ThetaMethodCombination, N}`
- [`GenAlpha1Strategy{N}`](@ref)    = `CombinationOrder{GenAlpha1Combination, N}`
- [`GenAlpha2Strategy{N}`](@ref)    = `CombinationOrder{GenAlpha2Combination, N}`
"""
struct CombinationOrder{A,N} <: TimeCombination
  combination::A 
  CombinationOrder{N}(c::A) where {A,N} = new{A,N}(c)
  CombinationOrder{N}(c::CombinationOrder) where N = CombinationOrder{N}(c.combination)
end

@doc raw"""
    const ThetaMethodStrategy{N} = CombinationOrder{ThetaMethodCombination, N}

Specialised alias for `CombinationOrder` with a [`ThetaMethodCombination`](@ref).
`N = 1` yields stiffness coefficients ``(\theta,\, 1-\theta)``;
`N = 2` yields mass coefficients ``(1/\Delta t,\, -1/\Delta t)``.
"""
const ThetaMethodStrategy{N} = CombinationOrder{ThetaMethodCombination,N}

function get_coefficients(c::ThetaMethodStrategy{1},args...)
  (c.combination.θ,1-c.combination.θ)
end

function get_coefficients(c::ThetaMethodStrategy{2},args...)
  (1/c.combination.dt,-1/c.combination.dt)
end

function _combination!(
  uθ::AbstractParamVector,
  c::ThetaMethodStrategy,
  u::AbstractParamVector,
  us0::NTuple{2,AbstractParamVector}
  )
  
  u0, = us0 

  np = param_length(u0)
  nt = round(Int,param_length(u) / np)
  η = get_coefficients(c,nt)

  dataθ = get_all_data(uθ)
  data = get_all_data(u)
  data0 = get_all_data(u0)
  
  for ipt = param_eachindex(u)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        dataθ[is,ipt] = η[1]*data[is,ipt] + η[2]*data0[is,ipt]
      end
    else
      for is in axes(data,1)
        dataθ[is,ipt] = η[1]*data[is,ipt] + η[2]*data[is,ipt-np]
      end
    end
  end

  return dataθ
end

"""
    const GenAlpha1Strategy{N} = CombinationOrder{GenAlpha1Combination, N}

Specialised alias for `CombinationOrder` with a [`GenAlpha1Combination`](@ref).
Orders 1 and 2 correspond to the stiffness and mass/damping operators of a
first-order Generalized-α scheme.
"""
const GenAlpha1Strategy{N} = CombinationOrder{GenAlpha1Combination,N}

function get_coefficients(c::GenAlpha1Strategy{1},args...)
  (c.combination.αf,1-c.combination.αf)
end

function get_coefficients(c::GenAlpha1Strategy{2},N::Int)
  @unpack dt,αf,αm,γ = c.combination 
  a = 1 / (γ*dt)
  b = 1 - 1/γ
  c = a * (1 - αm + b*αm)
  η = (a*αm,c - a*αm)
  for j in 3:N
    η = (η...,c*(b^(j-2) - b^(j-3)))
  end
  η
end

function _combination!(
  uα::AbstractParamVector,
  c::GenAlpha1Strategy{1},
  u::AbstractParamVector,
  us0::NTuple{2,AbstractParamVector}
  )
  
  u0, = us0 

  np = param_length(u0)
  nt = round(Int,param_length(u) / np)
  η = get_coefficients(c,nt)

  dataα = get_all_data(uα)
  data = get_all_data(u)
  data0 = get_all_data(u0)
  
  for ipt = param_eachindex(u)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        dataα[is,ipt] = η[1]*data[is,ipt] + η[2]*data0[is,ipt]
      end
    else
      for is in axes(data,1)
        dataα[is,ipt] = η[1]*data[is,ipt] + η[2]*data[is,ipt-np]
      end
    end
  end

  return dataα
end

function _combination!(
  vα::AbstractParamVector,
  c::GenAlpha1Strategy{2},
  u::AbstractParamVector,
  us0::NTuple{2,AbstractParamVector}
  )
  
  u0,v0 = us0 

  np = param_length(u0)
  nt = round(Int,param_length(u) / np)
  η = get_coefficients(c,nt)

  @unpack dt,αf,αm,γ = c.combination 
  a = 1 / (γ*dt)
  b = 1 - 1/γ
  c = a * (1 - αm + b*αm)

  np = param_length(u0)

  ddataα = get_all_data(vα)
  data = get_all_data(u)
  data0 = get_all_data(u0)
  ddata0 = get_all_data(v0)
  for ipt = param_eachindex(u)
    ip = fast_index(ipt,np)
    it = slow_index(ipt,np)
    n = it - 1
    if it == 1
      for is in axes(data,1)
        ddataα[is,ipt] = η[1]*data[is,ipt] - a*αm*data0[is,ip] + c / a * ddata0[is,ip]
      end
    else
      for is in axes(data,1)
        ddataα[is,ipt] = η[1]*data[is,ipt] + η[2]*data[is,ipt-np] + (c / a * b^n)*ddata0[is,ip] - c * b^(n-1)*data0[is,ip]
        for (j,ipt_back) in enumerate(ipt-2*np : -np : 1)
          ddataα[is,ipt] += η[2+j]*data[is,ipt_back]
        end
      end
    end
  end

  return ddataα
end

"""
    const GenAlpha2Strategy{N} = CombinationOrder{GenAlpha2Combination, N}

Specialised alias for `CombinationOrder` with a [`GenAlpha2Combination`](@ref).
Orders 1, 2, and 3 correspond to the stiffness, damping, and mass operators of a
second-order Generalized-α (Newmark) scheme.
"""
const GenAlpha2Strategy{N} = CombinationOrder{GenAlpha2Combination,N}

function get_coefficients(c::GenAlpha2Strategy{1},args...)
  (1-c.combination.αf,c.combination.αf)
end

function get_coefficients(c::GenAlpha2Strategy{2},N::Int)
  @unpack dt,αf,αm,γ,β = c.combination 
  
  a = γ / (dt * β)
  b = -a 
  c = 1 - γ / β 
  d = dt * (1 - γ / (2*β))

  e = 1 / (dt^2 * β)
  f = -e 
  g = - 1 / (dt * β)
  h = 1 - 1 / (2*β)

  P = [c d 
      g h]

  αnj(n,j) = ([1 0] * P^(n-j) * [a,e])[1]
  βnj(n,j) = ([1 0] * P^(n-j) * [b,f])[1]
  κnj(n,j) = αnj(n,j-1) + βnj(n,j)
  aαn(n) = (1-αf) * αnj(n,n)
  bαn(n) = (1-αf) * κnj(n,n) + αf * αnj(n-1,n-1)
  fαnj(n,j) = (1-αf) * κnj(n,j) + αf * κnj(n-1,j)

  η = (aαn(0),bαn(0))
  for j in 1:N-2
    η = (η...,fαnj(j+1,1))
  end
  
  return η
end

function get_coefficients(c::GenAlpha2Strategy{3},N::Int)
  @unpack dt,αf,αm,γ,β = c.combination 
  
  a = γ / (dt * β)
  b = -a 
  c = 1 - γ / β 
  d = dt * (1 - γ / (2*β))

  e = 1 / (dt^2 * β)
  f = -e 
  g = - 1 / (dt * β)
  h = 1 - 1 / (2*β)

  P = [c d 
      g h]

  γnj(n,j) = ([0 1] * P^(n-j) * [a,e])[1]
  δnj(n,j) = ([0 1] * P^(n-j) * [b,f])[1]
  ηnj(n,j) = γnj(n,j-1) + δnj(n,j)
  gαn(n) = (1-αm) * γnj(n,n)
  hαn(n) = (1-αm) * ηnj(n,n) + αm * γnj(n-1,n-1)
  lαnj(n,j) = (1-αm) * ηnj(n,j) + αm * ηnj(n-1,j)

  η = (gαn(0),hαn(0))
  for j in 1:N-2
    η = (η...,lαnj(j+1,1))
  end

  return η
end

function _combination!(
  uα::AbstractParamVector,
  c::GenAlpha2Strategy{1},
  u::AbstractParamVector,
  us0::NTuple{3,AbstractParamVector}
  )
  
  u0, = us0 

  np = param_length(u0)
  nt = round(Int,param_length(u) / np)
  η = get_coefficients(c,nt)

  dataα = get_all_data(uα)
  data = get_all_data(u)
  data0 = get_all_data(u0)
  
  for ipt = param_eachindex(u)
    it = slow_index(ipt,np)
    if it == 1
      for is in axes(data,1)
        dataα[is,ipt] = η[1]*data[is,ipt] + η[2]*data0[is,ipt]
      end
    else
      for is in axes(data,1)
        dataα[is,ipt] = η[1]*data[is,ipt] + η[2]*data[is,ipt-np]
      end
    end
  end

  return dataα
end

function _combination!(
  vα::AbstractParamVector,
  c::GenAlpha2Strategy{2},
  u::AbstractParamVector,
  us0::NTuple{3,AbstractParamVector}
  )
  
  u0,v0,a0 = us0 

  np = param_length(u0)
  nt = round(Int,param_length(u) / np)
  η = get_coefficients(c,nt)

  @unpack dt,αf,αm,γ,β = c.combination 
  a = γ / (dt * β)
  b = -a 
  c = 1 - γ / β 
  d = dt * (1 - γ / (2*β))

  e = 1 / (dt^2 * β)
  f = -e 
  g = - 1 / (dt * β)
  h = 1 - 1 / (2*β)

  P = [c d 
      g h]

  an(n) = ([1 0] * P^(n) * [1,0])[1]
  bn(n) = ([1 0] * P^(n) * [0,1])[1]
  αnj(n,j) = ([1 0] * P^(n-j) * [a,e])[1]
  βnj(n,j) = ([1 0] * P^(n-j) * [b,f])[1]
  cαn(n) = (1-αf) * βnj(n,0) + αf * βnj(n-1,0)
  dαn(n) = (1-αf) * an(n+1) + αf * an(n)
  eαn(n) = (1-αf) * bn(n+1) + αf * bn(n)

  βnj00 = βnj(0,0)
  an1 = an(1)
  bn1 = bn(1)
  cvec = zeros(nt-1)
  dvec = zeros(nt-1)
  evec = zeros(nt-1)
  for it in 1:nt-1
    cvec[it] = cαn(it)
    dvec[it] = dαn(it)
    evec[it] = eαn(it)
  end

  ddataα = get_all_data(vα)
  data = get_all_data(u)
  data0 = get_all_data(u0)
  ddata0 = get_all_data(v0)
  dddata0 = get_all_data(a0)
  for ipt = param_eachindex(u)
    ip = fast_index(ipt,np)
    it = slow_index(ipt,np)
    n = it - 1
    if it == 1
      for is in axes(data,1)
        ddataα[is,ipt] = η[1]*data[is,ipt] + (1-αf)*βnj00*data0[is,ip] + ((1-αf)*an1 + αf)*ddata0[is,ip] + (1-αf)*bn1*dddata0[is,ip]
      end
    else
      for is in axes(data,1)
        ddataα[is,ipt] = η[1]*data[is,ipt] + η[2]*data[is,ipt-np] + cvec[n]*data0[is,ip] + dvec[n]*ddata0[is,ip] + evec[n]*dddata0[is,ip]
        for (j,ipt_back) in enumerate(ipt-2*np : -np : 1)
          ddataα[is,ipt] += η[2+j]*data[is,ipt_back]
        end
      end
    end
  end

  return ddataα
end

function _combination!(
  aα::AbstractParamVector,
  c::GenAlpha2Strategy{3},
  u::AbstractParamVector,
  us0::NTuple{3,AbstractParamVector}
  )
  
  u0,v0,a0 = us0 

  np = param_length(u0)
  nt = round(Int,param_length(u) / np)
  η = get_coefficients(c,nt)

  @unpack dt,αf,αm,γ,β = c.combination 
  a = γ / (dt * β)
  b = -a 
  c = 1 - γ / β 
  d = dt * (1 - γ / (2*β))

  e = 1 / (dt^2 * β)
  f = -e 
  g = - 1 / (dt * β)
  h = 1 - 1 / (2*β)

  P = [c d 
      g h]

  cn(n) = ([0 1] * P^(n) * [1,0])[1]
  dn(n) = ([0 1] * P^(n) * [0,1])[1]
  γnj(n,j) = ([0 1] * P^(n-j) * [a,e])[1]
  δnj(n,j) = ([0 1] * P^(n-j) * [b,f])[1]
  iαn(n) = (1-αm) * δnj(n,0) + αm * δnj(n-1,0)
  jαn(n) = (1-αm) * cn(n+1) + αm * cn(n)
  kαn(n) = (1-αm) * dn(n+1) + αm * dn(n)

  ivec = zeros(nt-1)
  jvec = zeros(nt-1)
  kvec = zeros(nt-1)
  for it in 1:nt-1
    ivec[it] = iαn(it)
    jvec[it] = jαn(it)
    kvec[it] = kαn(it)
  end
  δnj00 = δnj(0,0)
  cn1 = cn(1)
  dn1 = dn(1)

  dddataα = get_all_data(aα)
  data = get_all_data(u)
  data0 = get_all_data(u0)
  ddata0 = get_all_data(v0)
  dddata0 = get_all_data(a0)
  for ipt = param_eachindex(u)
    ip = fast_index(ipt,np)
    it = slow_index(ipt,np)
    n = it - 1
    if it == 1
      for is in axes(data,1)
        dddataα[is,ipt] = η[1]*data[is,ipt] + (1-αm)*δnj00*data0[is,ip] + (1-αm)*cn1*ddata0[is,ip] + ((1-αm)*dn1 + αm)*dddata0[is,ip]
      end
    else
      for is in axes(data,1)
        dddataα[is,ipt] = η[1]*data[is,ipt] + η[2]*data[is,ipt-np] + ivec[n]*data0[is,ip] + jvec[n]*ddata0[is,ip] + kvec[n]*dddata0[is,ip]
        for (j,ipt_back) in enumerate(ipt-2*np : -np : 1)
          dddataα[is,ipt] += η[2+j]*data[is,ipt_back]
        end
      end
    end
  end

  return dddataα
end