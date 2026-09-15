"""
    abstract type TransientReduction{A<:ReductionStyle,B<:AssembleOperator} <: Reduction{A,B} end

Abstract supertype for reduction methods in high-order (e.g. transient)
parametric problems.

Concrete subtypes:
- [`SteadyReduction`](@ref) — wraps a steady `Reduction`; no temporal
  compression is applied (the ROM still time-steps).
- [`KroneckerReduction`](@ref) — builds a Kronecker (Tucker) product space
  from independent spatial and temporal reductions.
- [`SequentialReduction`](@ref) — uses TT-SVD (tensor-train) decomposition
  for the snapshot tensor.

Use the generic constructor `TransientReduction(args...; kwargs...)` to dispatch
to the appropriate subtype based on the arguments.
"""
abstract type TransientReduction{A<:ReductionStyle,B<:AssembleOperator} <: Reduction{A,B} end

"""
    struct SteadyReduction{A,B} <: TransientReduction{A,B}
      reduction::Reduction{A,B}
    end

Wrapper for steady reduction methods in high order problems, such as transient ones. 
In practice, the resulting ROM will still need to run the time marching scheme, since 
no temporal reduction occurs.
"""
struct SteadyReduction{A,B} <: TransientReduction{A,B}
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
    struct KroneckerReduction{A,B} <: TransientReduction{A,B}
      reductions::AbstractVector{<:Reduction}
    end

Wrapper for reduction methods in high order problems, such as transient ones. The
reduced subspaces are constructed as Kronecker product spaces
"""
struct KroneckerReduction{A,B} <: TransientReduction{A,B}
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

function TransientReduction(reduction::TransientReduction,args...;kwargs...)
  reduction
end

function TransientReduction(styles::AbstractVector{<:ReductionStyle},args...;kwargs...)
  reductions = map(s -> Reduction(s,args...;kwargs...),styles)
  KroneckerReduction(reductions)
end

function TransientReduction(tolranks::AbstractVector{<:Union{Int,Float64}},args...;kwargs...)
  reductions = map(t -> Reduction(t,args...;kwargs...),tolranks)
  KroneckerReduction(reductions)
end

function TransientReduction(tolrank::Union{Int,Float64},args...;dim=2,kwargs...)
  TransientReduction(Fill(tolrank,dim),args...;kwargs...)
end

function TransientReduction(red_style::ReductionStyle,args...;dim=2,kwargs...)
  TransientReduction(Fill(red_style,dim),args...;kwargs...)
end

"""
    struct SequentialReduction{A,B} <: TransientReduction{A,B}
      reduction::Reduction{A,B}
    end

Wrapper for sequential reduction methods in high-order problems, e.g. TT-SVD in
transient applications
"""
struct SequentialReduction{A,B} <: TransientReduction{A,B}
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

function TransientReduction(red_style::TTSVDRanks,args...;kwargs...)
  reduction = Reduction(red_style,args...;kwargs...)
  SequentialReduction(reduction)
end

function TransientReduction(tolrank::Union{Vector{Int},Vector{Float64}},args...;kwargs...)
  reduction = Reduction(tolrank,args...;kwargs...)
  SequentialReduction(reduction)
end

function TransientReduction(coupling::AssembleOperator,args...;supr_tol=1e-2,kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  SupremizerReduction(reduction,coupling,supr_tol)
end

@doc raw"""
    abstract type TransientHyperReduction{A} <: HyperReduction{A} end

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
abstract type TransientHyperReduction{A} <: HyperReduction{A} end

"""
    TransientHyperReduction(combination, args...; compression=:global, hypred_strategy=:deim, kwargs...)

Factory for transient high-dimensional hyper-reduction strategies.

Supported `hypred_strategy` values:

- `:deim` (existing)
- `:sopt` (existing)
- `:rbf` (existing)
- `:none` (new, aliases: `:no`, `:nohr`) -> [`TransientNoHyperReduction`](@ref)
- `:affine` (new) -> [`TransientAffineHyperReduction`](@ref)

When `compression=:local`, this dispatches to
[`TransientLocalHyperReduction`](@ref) with the selected strategy.
"""
function TransientHyperReduction(
  combination::TimeCombination,
  args...;compression=:global,
  hypred_strategy=:deim,
  kwargs...
  )

  if hypred_strategy in (:no,:none,:nohr)
    return TransientNoHyperReduction(combination)
  elseif hypred_strategy == :affine
    return TransientAffineHyperReduction(combination)
  elseif compression==:global
    reduction = TransientReduction(args...;kwargs...)
    if hypred_strategy==:deim
      return TransientDEIMHyperReduction(combination,reduction)
    elseif hypred_strategy==:sopt
      return TransientSOPTHyperReduction(combination,reduction)
    elseif hypred_strategy==:rbf
      return TransientRBFHyperReduction(combination,reduction)
    else
      error("Unknown high-dimensional hyper-reduction strategy: $hypred_strategy")
    end
  else
    TransientLocalHyperReduction(combination,args...;hypred_strategy,kwargs...)
  end
end

function TransientHyperReduction(
  combination::TimeCombination,
  reduction::TransientReduction,
  args...;kwargs...
  )

  red_style = ReductionStyle(reduction)
  TransientHyperReduction(combination,red_style;kwargs...)
end

function TransientHyperReduction(
  reduction::TransientReduction,
  combination::TimeCombination;kwargs...
  )

  red_style = ReductionStyle(reduction)
  TransientDEIMHyperReduction(combination,red_style;kwargs...)
end

function TransientHyperReduction(
  combination::TimeCombination,
  reduction::SupremizerReduction,
  args...;kwargs...
  )

  TransientHyperReduction(combination,get_reduction(reduction),args...;kwargs...)
end

function TransientHyperReduction(reduction::SupremizerReduction,args...;kwargs...)
  TransientHyperReduction(get_reduction(reduction),args...;kwargs...)
end

function TransientHyperReduction(combination::TimeCombination,r::LocalReduction,args...;ncentroids=num_centroids(r),kwargs...)
  TransientLocalHyperReduction(combination,get_reduction(r),args...;ncentroids,kwargs...)
end

function TransientHyperReduction(r::LocalReduction,combination::TimeCombination;ncentroids=num_centroids(r),kwargs...)
  TransientLocalHyperReduction(combination,get_reduction(r);ncentroids,kwargs...)
end

get_time_combination(r::TransientHyperReduction) = @abstractmethod

function TransientHyperReduction(
  combination::TimeCombination,
  reduction::SteadyReduction,
  args...;kwargs...
  )

  hr = HyperReduction(reduction,args...;kwargs...)
  _replace_reduction(hr)
end

function TransientHyperReduction(
  reduction::SteadyReduction,
  combination::TimeCombination;kwargs...
  )

  hr = SteadyHyperReduction(reduction;kwargs...)
  _replace_reduction(hr)
end

abstract type TransientTrivialHyperReduction <: TransientHyperReduction{NoReductionStyle} end

RBSteady.get_reduction(r::TransientTrivialHyperReduction) = NoReduction()
RBSteady.ReductionStyle(r::TransientTrivialHyperReduction) = NoReductionStyle()
RBSteady.NormStyle(r::TransientTrivialHyperReduction) = EuclideanNorm()
ParamDataStructures.num_params(r::TransientTrivialHyperReduction) = 1

"""
    struct TransientNoHyperReduction <: TransientTrivialHyperReduction 
      combination::TimeCombination
    end

Reduction employed when the input data is independent with respect to the
considered realisation. Therefore, simply considering a number of parameters
equal to 1 suffices for this type of reduction
"""
struct TransientNoHyperReduction <: TransientTrivialHyperReduction 
  combination::TimeCombination
end

get_time_combination(r::TransientNoHyperReduction) = r.combination

"""
    struct TransientAffineHyperReduction <: TransientTrivialHyperReduction 
      combination::TimeCombination
    end

Reduction employed when transient reduced contributions are affine with
parameter-independent (μ-independent) structure. As in the no-hyper-reduction
case, this uses a single effective parameter sample (`num_params = 1`) for the
hyper-reduction stage.
"""
struct TransientAffineHyperReduction <: TransientTrivialHyperReduction 
  combination::TimeCombination
end

get_time_combination(r::TransientAffineHyperReduction) = r.combination

"""
    struct TransientDEIMHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: TransientHyperReduction{A}

Transient hyper-reduction based on the Matrix Discrete Empirical Interpolation
Method (DEIM). Combines a spatial [`TransientReduction`](@ref) with a
[`TimeCombination`](@ref) encoding the ODE time-marching coefficients.

# Fields
- `reduction::R`: the underlying spatial reduction.
- `combination::TimeCombination`: time-marching combination.
"""
struct TransientDEIMHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: TransientHyperReduction{A}
  reduction::R
  combination::TimeCombination
end

function TransientDEIMHyperReduction(combination::TimeCombination,args...;kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  TransientDEIMHyperReduction(reduction,combination)
end

RBSteady.get_reduction(r::TransientDEIMHyperReduction) = r.reduction
get_time_combination(r::TransientDEIMHyperReduction) = r.combination

"""
    struct TransientSOPTHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: TransientHyperReduction{A}

Transient hyper-reduction based on the SOPT (Second-Order Proper
Transformation) strategy. Stores a spatial [`TransientReduction`](@ref) and a
[`TimeCombination`](@ref).

# Fields
- `reduction::R`: the underlying spatial reduction.
- `combination::TimeCombination`: time-marching combination.
"""
struct TransientSOPTHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: TransientHyperReduction{A}
  reduction::R
  combination::TimeCombination
end

function TransientSOPTHyperReduction(combination::TimeCombination,args...;kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  TransientSOPTHyperReduction(reduction,combination)
end

RBSteady.get_reduction(r::TransientSOPTHyperReduction) = r.reduction
get_time_combination(r::TransientSOPTHyperReduction) = r.combination

"""
    struct TransientRBFHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: TransientHyperReduction{A}

Transient hyper-reduction based on radial basis function (RBF) interpolation.
In addition to the spatial [`TransientReduction`](@ref) and
[`TimeCombination`](@ref), it stores an `AbstractRadialBasis` strategy that
governs the RBF kernel.

# Fields
- `reduction::R`: the underlying spatial reduction.
- `combination::TimeCombination`: time-marching combination.
- `strategy::AbstractRadialBasis`: radial basis function kernel (default `PHS()`).
"""
struct TransientRBFHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: TransientHyperReduction{A}
  reduction::R
  combination::TimeCombination
  strategy::AbstractRadialBasis
end

function TransientRBFHyperReduction(combination::TimeCombination,args...;strategy=PHS(),kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  TransientRBFHyperReduction(reduction,combination,strategy)
end

RBSteady.get_reduction(r::TransientRBFHyperReduction) = r.reduction
RBSteady.interp_strategy(r::TransientRBFHyperReduction) = r.strategy
get_time_combination(r::TransientRBFHyperReduction) = r.combination

# local

function TransientLocalHyperReduction(args...;ncentroids=10,kwargs...)
  reduction = TransientHyperReduction(args...;kwargs...)
  LocalReduction(reduction,ncentroids)
end

# utils

_steady_reduction(r::HyperReduction) = SteadyReduction(get_reduction(r))
_replace_reduction(r::TransientNoHyperReduction) = r
_replace_reduction(r::TransientAffineHyperReduction) = r
_replace_reduction(r::DEIMHyperReduction) = DEIMHyperReduction(_steady_reduction(r))
_replace_reduction(r::SOPTHyperReduction) = SOPTHyperReduction(_steady_reduction(r))
_replace_reduction(r::RBFHyperReduction) = RBFHyperReduction(_steady_reduction(r),r.strategy)