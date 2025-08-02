"""
    abstract type SamplingStyle end

Subtypes:
- [`UniformSampling`](@ref)
- [`NormalSampling`](@ref)
- [`HaltonSampling`](@ref)
- [`LatinHypercubeSampling`](@ref)
- [`UniformTensorialSampling`](@ref)
"""
abstract type SamplingStyle end

function _generate_params(sampling::SamplingStyle,param_domain,nparams;kwargs...)
  [generate_param(sampling,param_domain;kwargs...) for i = 1:nparams]
end

function _generate_params(sampling::Symbol,args...;kwargs...)
  style = _sampling_to_style(;sampling)
  _generate_params(style,args...;kwargs...)
end

function _sampling_to_style(;sampling::Symbol=:halton,start=1)
  if sampling == :uniform
    UniformSampling()
  elseif sampling == :normal
    NormalSampling()
  elseif sampling == :halton
    HaltonSampling(start)
  elseif sampling == :latin_hypercube
    LatinHypercubeSampling()
  elseif sampling == :tensorial_uniform
    TensorialUniformSampling()
  else
    @notimplemented "Need to implement more sampling strategies"
  end
end

"""
    struct UniformSampling <: SamplingStyle end

Sampling according to a uniform distribution
"""
struct UniformSampling <: SamplingStyle end

function generate_param(::UniformSampling,param_domain)
  [rand(Uniform(first(d),last(d))) for d = param_domain]
end

"""
    struct NormalSampling <: SamplingStyle end

Sampling according to a normal distribution
"""
struct NormalSampling <: SamplingStyle end

function generate_param(::NormalSampling,param_domain)
  [rand(Uniform(first(d),last(d))) for d = param_domain]
end

"""
    struct HaltonSampling <: SamplingStyle
      start::Int
    end

Sampling according to a Halton sequence

!!! note
  Halton is a sequence, not a distribution, hence this sampling strategy repeats
  realizations since the draws are not randomized; to draw different parameters,
  one needs to provide a starting point in the sequence (start = 1 by default)
"""
struct HaltonSampling <: SamplingStyle
  start::Int
end

HaltonSampling() = HaltonSampling(1)

function _generate_params(s::HaltonSampling,param_domain,nparams)
  d = length(param_domain)
  hs = HaltonPoint(d;length=nparams,start=s.start)
  hs′ = collect(hs)
  for x in hs′
    for (di,xdi) in enumerate(x)
      a,b = param_domain[di]
      x[di] = a+(b-a)*xdi
    end
  end
  return hs′
end

"""
    struct LatinHypercubeSampling <: SamplingStyle end

Sampling according to a Latin HyperCube distribution
"""
struct LatinHypercubeSampling <: SamplingStyle end

function _generate_params(::LatinHypercubeSampling,param_domain,nparams;rng=GLOBAL_RNG)
  D = length(param_domain)
  lhc = zeros(nparams,D)
  for d in 1:D
    a,b = param_domain[d]
    perm = randperm(rng,nparams)
    for np in 1:nparams
      u = (perm[np]-rand(rng))/nparams
      lhc[np,d] = a+u*(b-a)
    end
  end
  params = collect.(eachrow(lhc))
  return params
end

"""
    struct TensorialUniformSampling <: SamplingStyle end

Sampling according to a tensorial uniform distribution
"""
struct TensorialUniformSampling <: SamplingStyle end

function _generate_params(::TensorialUniformSampling,param_domain,nparams)
  mesh = _get_tensor_mesh(param_domain,nparams)
  D = length(param_domain)
  npoints = length(mesh)
  tus = zeros(D,npoints)
  for (i,p) in enumerate(mesh)
    @views tus[:,i] .= p
  end
  params = collect.(eachcol(tus))
  return params
end

function _get_tensor_mesh(param_domain,nparams)
  D = length(param_domain)
  n_per_dim = ceil(Int,nparams^(1/D))
  grids = [range(a,stop=b,length=n_per_dim) for (a,b) in param_domain]
  return Iterators.product(grids...)
end
