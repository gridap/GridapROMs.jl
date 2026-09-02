"""
    abstract type HighDimReduction{A<:ReductionStyle,B<:AssembleOperator} <: Reduction{A,B} end

Abstract supertype for reduction methods in high-order (e.g. transient)
parametric problems.

Concrete subtypes:
- [`SteadyReduction`](@ref) — wraps a steady `Reduction`; no temporal
  compression is applied (the ROM still time-steps).
- [`KroneckerReduction`](@ref) — builds a Kronecker (Tucker) product space
  from independent spatial and temporal reductions.
- [`SequentialReduction`](@ref) — uses TT-SVD (tensor-train) decomposition
  for the snapshot tensor.

Use the generic constructor `HighDimReduction(args...; kwargs...)` to dispatch
to the appropriate subtype based on the arguments.
"""
abstract type HighDimReduction{A<:ReductionStyle,B<:AssembleOperator} <: Reduction{A,B} end

"""
    struct SteadyReduction{A,B} <: HighDimReduction{A,B}
      reduction::Reduction{A,B}
    end

Wrapper for steady reduction methods in high order problems, such as transient ones. 
In practice, the resulting ROM will still need to run the time marching scheme, since 
no temporal reduction occurs.
"""
struct SteadyReduction{A,B} <: HighDimReduction{A,B}
  reduction::Reduction{A,B}
end

function SteadyReduction(args...;kwargs...)
  reduction = Reduction(args...;kwargs...)
  SteadyReduction(reduction)
end

function SteadyReduction(coupling::AssembleOperator,args...;supr_tol=1e-2,kwargs...)
  reduction = SteadyReduction(args...;kwargs...)
  SupremizerReduction(reduction,coupling,supr_tol)
end

RBSteady.ReductionStyle(r::SteadyReduction) = ReductionStyle(r.reduction)
RBSteady.NormStyle(r::SteadyReduction) = NormStyle(r.reduction)
ParamDataStructures.num_params(r::SteadyReduction) = num_params(r.reduction)

"""
    struct KroneckerReduction{A,B} <: HighDimReduction{A,B}
      reductions::AbstractVector{<:Reduction}
    end

Wrapper for reduction methods in high order problems, such as transient ones. The
reduced subspaces are constructed as Kronecker product spaces
"""
struct KroneckerReduction{A,B} <: HighDimReduction{A,B}
  reductions::AbstractVector{<:Reduction}
  function KroneckerReduction(reductions::AbstractVector{<:Reduction})
    A = typeof(ReductionStyle(first(reductions)))
    B = typeof(NormStyle(first(reductions)))
    new{A,B}(reductions)
  end
end

function KroneckerReduction(r::AbstractVector{<:LocalReduction})
  r′ = KroneckerReduction(get_reduction.(r))
  nc = num_centroids(first(r))
  LocalReduction(r′,nc)
end

RBSteady.ReductionStyle(r::KroneckerReduction) = ReductionStyle(first(r.reductions))
RBSteady.NormStyle(r::KroneckerReduction) = NormStyle(first(r.reductions))
ParamDataStructures.num_params(r::KroneckerReduction) = num_params(first(r.reductions))

get_reduction_space(r::KroneckerReduction) = first(r.reductions)
get_reduction_time(r::KroneckerReduction) = last(r.reductions)

# generic constructor

function HighDimReduction end

const TransientReduction = HighDimReduction

function HighDimReduction(reduction::HighDimReduction,args...;kwargs...)
  reduction
end

function HighDimReduction(styles::AbstractVector{<:ReductionStyle},args...;kwargs...)
  reductions = map(s -> Reduction(s,args...;kwargs...),styles)
  KroneckerReduction(reductions)
end

function HighDimReduction(tolranks::AbstractVector{<:Union{Int,Float64}},args...;kwargs...)
  reductions = map(t -> Reduction(t,args...;kwargs...),tolranks)
  KroneckerReduction(reductions)
end

function HighDimReduction(tolrank::Union{Int,Float64},args...;dim=2,kwargs...)
  HighDimReduction(Fill(tolrank,dim),args...;kwargs...)
end

function HighDimReduction(red_style::ReductionStyle,args...;dim=2,kwargs...)
  HighDimReduction(Fill(red_style,dim),args...;kwargs...)
end

"""
    struct SequentialReduction{A,B} <: HighDimReduction{A,B}
      reduction::Reduction{A,B}
    end

Wrapper for sequential reduction methods in high-order problems, e.g. TT-SVD in
transient applications
"""
struct SequentialReduction{A,B} <: HighDimReduction{A,B}
  reduction::Reduction{A,B}
end

RBSteady.get_reduction(r::SequentialReduction) = r.reduction
RBSteady.ReductionStyle(r::SequentialReduction) = ReductionStyle(r.reduction)
RBSteady.NormStyle(r::SequentialReduction) = NormStyle(r.reduction)
ParamDataStructures.num_params(r::SequentialReduction) = num_params(r.reduction)

function SequentialReduction(r::LocalReduction)
  r′ = SequentialReduction(get_reduction(r))
  nc = num_centroids(r)
  LocalReduction(r′,nc)
end

function HighDimReduction(red_style::TTSVDRanks,args...;kwargs...)
  reduction = Reduction(red_style,args...;kwargs...)
  SequentialReduction(reduction)
end

function HighDimReduction(tolrank::Union{Vector{Int},Vector{Float64}},args...;kwargs...)
  reduction = Reduction(tolrank,args...;kwargs...)
  SequentialReduction(reduction)
end

function HighDimReduction(coupling::AssembleOperator,args...;supr_tol=1e-2,kwargs...)
  reduction = HighDimReduction(args...;kwargs...)
  SupremizerReduction(reduction,coupling,supr_tol)
end

@doc raw"""
    abstract type HighDimHyperReduction{A} <: HyperReduction{A} end

Hyper-reduction strategies employed in high-order (e.g. transient) problems.

Every concrete subtype stores a [`TimeCombination`](@ref), which encodes the
way an ODE time-marching scheme combines contributions from different time
levels. See [`TimeCombination`](@ref) and [`CombinationOrder`](@ref) for
the full treatment; here we summarise the key idea for the simplest case.

### Theta method (first-order ODE)

Consider

```math
M \dot{u}(t) + A\, u(t) = f(t).
```

The ``\theta``-method reads

```math
M \frac{u_{n+1} - u_n}{\Delta t}
+ \theta\, A\, u_{n+1} + (1-\theta)\, A\, u_n
= f_{n+\theta},
```

which can be rewritten as

```math
\left( \frac{1}{\Delta t} M + \theta\, A \right) u_{n+1}
=
\left( \frac{1}{\Delta t} M - (1-\theta)\, A \right) u_n
+ f_{n+\theta}.
```

The scheme is therefore purely one-step:

```math
\boxed{
\left( \frac{1}{\Delta t} M + \theta\, A \right) u_{n+1}
=
\left( \frac{1}{\Delta t} M - (1-\theta)\, A \right) u_n
+ f_{n+\theta}.
}
```

Within the ROM framework the two operators ``A`` and ``M`` are associated
with distinct [`CombinationOrder`](@ref) indices (1 and 2 for a first-order
problem).  The [`TimeCombination`](@ref) object stores the scheme parameters
(``\theta``, ``\Delta t``, …) and the function [`get_coefficients`](@ref)
returns the per-order weights that combine snapshots from successive time
levels.  Higher-order schemes (Newmark / Generalized-α) follow the same
pattern with additional combination orders.
"""
abstract type HighDimHyperReduction{A} <: HyperReduction{A} end

function HighDimHyperReduction end

const TransientHyperReduction = HighDimHyperReduction

"""
    HighDimHyperReduction(combination, args...; compression=:global, hypred_strategy=:deim, kwargs...)

Factory for transient high-dimensional hyper-reduction strategies.

Supported `hypred_strategy` values:

- `:deim` (existing)
- `:sopt` (existing)
- `:rbf` (existing)
- `:none` (new, aliases: `:no`, `:nohr`) -> [`HighDimNoHyperReduction`](@ref)
- `:affine` (new) -> [`HighDimAffineHyperReduction`](@ref)

When `compression=:local`, this dispatches to
[`HighDimLocalHyperReduction`](@ref) with the selected strategy.
"""
function HighDimHyperReduction(
  combination::TimeCombination,
  args...;compression=:global,
  hypred_strategy=:deim,
  kwargs...
  )

  if hypred_strategy in (:no,:none,:nohr)
    return HighDimNoHyperReduction(combination)
  elseif hypred_strategy == :affine
    return HighDimAffineHyperReduction(combination)
  elseif compression==:global
    reduction = HighDimReduction(args...;kwargs...)
    if hypred_strategy==:deim
      return HighDimDEIMHyperReduction(combination,reduction)
    elseif hypred_strategy==:sopt
      return HighDimSOPTHyperReduction(combination,reduction)
    elseif hypred_strategy==:rbf
      return HighDimRBFHyperReduction(combination,reduction)
    else
      error("Unknown high-dimensional hyper-reduction strategy: $hypred_strategy")
    end
  else
    HighDimLocalHyperReduction(combination,args...;hypred_strategy,kwargs...)
  end
end

function HighDimHyperReduction(
  combination::TimeCombination,
  reduction::HighDimReduction,
  args...;kwargs...
  )

  red_style = ReductionStyle(reduction)
  HighDimHyperReduction(combination,red_style;kwargs...)
end

function HighDimHyperReduction(
  reduction::HighDimReduction,
  combination::TimeCombination;kwargs...
  )

  red_style = ReductionStyle(reduction)
  HighDimDEIMHyperReduction(combination,red_style;kwargs...)
end

function HighDimHyperReduction(
  combination::TimeCombination,
  reduction::SupremizerReduction,
  args...;kwargs...
  )

  HighDimHyperReduction(combination,get_reduction(reduction),args...;kwargs...)
end

function HighDimHyperReduction(reduction::SupremizerReduction,args...;kwargs...)
  HighDimHyperReduction(get_reduction(reduction),args...;kwargs...)
end

function HighDimHyperReduction(combination::TimeCombination,r::LocalReduction,args...;ncentroids=num_centroids(r),kwargs...)
  HighDimLocalHyperReduction(combination,get_reduction(r),args...;ncentroids,kwargs...)
end

function HighDimHyperReduction(r::LocalReduction,combination::TimeCombination;ncentroids=num_centroids(r),kwargs...)
  HighDimLocalHyperReduction(combination,get_reduction(r);ncentroids,kwargs...)
end

get_time_combination(r::HighDimHyperReduction) = @abstractmethod

function HighDimHyperReduction(
  combination::TimeCombination,
  reduction::SteadyReduction,
  args...;kwargs...
  )

  hr = HyperReduction(reduction,args...;kwargs...)
  _replace_reduction(hr)
end

function HighDimHyperReduction(
  reduction::SteadyReduction,
  combination::TimeCombination;kwargs...
  )

  hr = SteadyHyperReduction(reduction;kwargs...)
  _replace_reduction(hr)
end

abstract type HighDimTrivialHyperReduction <: HighDimHyperReduction{NoReductionStyle} end

RBSteady.get_reduction(r::HighDimTrivialHyperReduction) = NoReduction()
RBSteady.ReductionStyle(r::HighDimTrivialHyperReduction) = NoReductionStyle()
RBSteady.NormStyle(r::HighDimTrivialHyperReduction) = EuclideanNorm()
ParamDataStructures.num_params(r::HighDimTrivialHyperReduction) = 1

"""
    struct HighDimNoHyperReduction <: HighDimTrivialHyperReduction 
      combination::TimeCombination
    end

Reduction employed when the input data is independent with respect to the
considered realisation. Therefore, simply considering a number of parameters
equal to 1 suffices for this type of reduction
"""
struct HighDimNoHyperReduction <: HighDimTrivialHyperReduction 
  combination::TimeCombination
end

get_time_combination(r::HighDimNoHyperReduction) = r.combination

"""
    struct HighDimAffineHyperReduction <: HighDimTrivialHyperReduction 
      combination::TimeCombination
    end

Reduction employed when transient reduced contributions are affine with
parameter-independent (μ-independent) structure. As in the no-hyper-reduction
case, this uses a single effective parameter sample (`num_params = 1`) for the
hyper-reduction stage.
"""
struct HighDimAffineHyperReduction <: HighDimTrivialHyperReduction 
  combination::TimeCombination
end

get_time_combination(r::HighDimAffineHyperReduction) = r.combination

"""
    struct HighDimDEIMHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: HighDimHyperReduction{A}

Transient hyper-reduction based on the Matrix Discrete Empirical Interpolation
Method (DEIM). Combines a spatial [`HighDimReduction`](@ref) with a
[`TimeCombination`](@ref) encoding the ODE time-marching coefficients.

# Fields
- `reduction::R`: the underlying spatial reduction.
- `combination::TimeCombination`: time-marching combination.
"""
struct HighDimDEIMHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: HighDimHyperReduction{A}
  reduction::R
  combination::TimeCombination
end

function HighDimDEIMHyperReduction(combination::TimeCombination,args...;kwargs...)
  reduction = HighDimReduction(args...;kwargs...)
  HighDimDEIMHyperReduction(reduction,combination)
end

RBSteady.get_reduction(r::HighDimDEIMHyperReduction) = r.reduction
get_time_combination(r::HighDimDEIMHyperReduction) = r.combination

"""
    struct HighDimSOPTHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: HighDimHyperReduction{A}

Transient hyper-reduction based on the SOPT (Second-Order Proper
Transformation) strategy. Stores a spatial [`HighDimReduction`](@ref) and a
[`TimeCombination`](@ref).

# Fields
- `reduction::R`: the underlying spatial reduction.
- `combination::TimeCombination`: time-marching combination.
"""
struct HighDimSOPTHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: HighDimHyperReduction{A}
  reduction::R
  combination::TimeCombination
end

function HighDimSOPTHyperReduction(combination::TimeCombination,args...;kwargs...)
  reduction = HighDimReduction(args...;kwargs...)
  HighDimSOPTHyperReduction(reduction,combination)
end

RBSteady.get_reduction(r::HighDimSOPTHyperReduction) = r.reduction
get_time_combination(r::HighDimSOPTHyperReduction) = r.combination

"""
    struct HighDimRBFHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: HighDimHyperReduction{A}

Transient hyper-reduction based on radial basis function (RBF) interpolation.
In addition to the spatial [`HighDimReduction`](@ref) and
[`TimeCombination`](@ref), it stores an `AbstractRadialBasis` strategy that
governs the RBF kernel.

# Fields
- `reduction::R`: the underlying spatial reduction.
- `combination::TimeCombination`: time-marching combination.
- `strategy::AbstractRadialBasis`: radial basis function kernel (default `PHS()`).
"""
struct HighDimRBFHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: HighDimHyperReduction{A}
  reduction::R
  combination::TimeCombination
  strategy::AbstractRadialBasis
end

function HighDimRBFHyperReduction(combination::TimeCombination,args...;strategy=PHS(),kwargs...)
  reduction = HighDimReduction(args...;kwargs...)
  HighDimRBFHyperReduction(reduction,combination,strategy)
end

RBSteady.get_reduction(r::HighDimRBFHyperReduction) = r.reduction
RBSteady.interp_strategy(r::HighDimRBFHyperReduction) = r.strategy
get_time_combination(r::HighDimRBFHyperReduction) = r.combination

# local

function HighDimLocalHyperReduction(args...;ncentroids=10,kwargs...)
  reduction = HighDimHyperReduction(args...;kwargs...)
  LocalReduction(reduction,ncentroids)
end

# utils

_steady_reduction(r::HyperReduction) = SteadyReduction(get_reduction(r))
_replace_reduction(r::HighDimNoHyperReduction) = r
_replace_reduction(r::HighDimAffineHyperReduction) = r
_replace_reduction(r::DEIMHyperReduction) = DEIMHyperReduction(_steady_reduction(r))
_replace_reduction(r::SOPTHyperReduction) = SOPTHyperReduction(_steady_reduction(r))
_replace_reduction(r::RBFHyperReduction) = RBFHyperReduction(_steady_reduction(r),r.strategy)