"""
    abstract type ReductionStyle end

Type indicating the reduction strategy to employ.

Subtypes:

- [`SearchSVDRank`](@ref)
- [`FixedSVDRank`](@ref)
- [`LRApproxRank`](@ref)
- [`TTSVDRanks`](@ref)
"""
abstract type ReductionStyle end

ReductionStyle(args...;kwargs...) = @abstractmethod

"""
    struct SearchSVDRank <: ReductionStyle
      tol::Float64
    end

Struct employed when the chosen reduction algorithm is a truncated POD at a
tolerance `tol`. Check [this](https://doi.org/10.1007/978-3-319-15431-2) reference
for more details on the truncated POD algorithm
"""
struct SearchSVDRank <: ReductionStyle
  tol::Float64
end

"""
    struct FixedSVDRank <: ReductionStyle
      rank::Int
    end

Struct employed when the chosen reduction algorithm is a truncated POD at a
rank `rank`. Check [this](https://doi.org/10.1007/978-3-319-15431-2) reference
for more details on the truncated POD algorithm
"""
struct FixedSVDRank <: ReductionStyle
  rank::Int
end

"""
    struct LRApproxRank <: ReductionStyle
      opts::LRAOptions
    end

Struct employed when the chosen reduction algorithm is a randomized POD that
leverages the package `LowRankApprox`. The field `opts` specifies the
options needed to run the randomized POD
"""
struct LRApproxRank <: ReductionStyle
  opts::LRAOptions
end

function LRApproxRank(tol::Float64;maxdet_tol=0.,sketch_randn_niter=1,sketch=:sprn,kwargs...)
  opts = LRAOptions(;rtol=tol,maxdet_tol,sketch_randn_niter,sketch)
  return LRApproxRank(opts)
end

function LRApproxRank(rank::Int;maxdet_tol=0.,sketch_randn_niter=1,sketch=:sprn,kwargs...)
  opts = LRAOptions(;rank=rank,maxdet_tol,sketch_randn_niter,sketch)
  return LRApproxRank(opts)
end

function ReductionStyle(tol::Float64;sketch=nothing,kwargs...)
  isa(sketch,Symbol) ? LRApproxRank(tol;sketch,kwargs...) : SearchSVDRank(tol)
end

function ReductionStyle(rank::Int;sketch=nothing,kwargs...)
  isa(sketch,Symbol) ? LRApproxRank(rank;sketch,kwargs...) : FixedSVDRank(rank)
end

"""
    struct TTSVDRanks <: ReductionStyle
      style::Vector{<:ReductionStyle}
    end

Struct employed when the chosen reduction algorithm is a TTSVD, with reduction
algorithm at each step specified in the vector of reduction styles `style`. Check
[this](https://doi.org/10.1137/090752286) reference for more details on the TTSVD algorithm
"""
struct TTSVDRanks <: ReductionStyle
  style::Vector{<:ReductionStyle}
end

function TTSVDRanks(tolranks::Vector{<:Union{Float64,Int}},args...;kwargs...)
  style = map(tolrank -> ReductionStyle(tolrank,args...;kwargs...),tolranks)
  TTSVDRanks(style)
end

function TTSVDRanks(tolrank::Union{Float64,Int},D=3,args...;kwargs...)
  TTSVDRanks(fill(tolrank,D),args...;kwargs...)
end

ReductionStyle(tolrank::Union{Float64,Int};kwargs...) = TTSVDRanks(tolrank;kwargs...)
ReductionStyle(tolrank::Vector{<:Union{Float64,Int}};kwargs...) = TTSVDRanks(tolrank;kwargs...)

Base.size(r::TTSVDRanks) = (length(r.style),)
Base.getindex(r::TTSVDRanks,i::Integer) = r.style[i]
Base.iterate(r::TTSVDRanks,state...) = iterate(r.style,state...)
Base.lastindex(r::TTSVDRanks) = lastindex(r.style)
Base.repeat(r::ReductionStyle,D::Int) = TTSVDRanks(fill(r,D))

"""
    abstract type NormStyle end

Subtypes:

- [`EuclideanNorm`](@ref)
- [`EnergyNorm`](@ref)
"""
abstract type NormStyle end

get_norm(n::NormStyle) = @abstractmethod

"""
    struct EuclideanNorm <: NormStyle end

Trait indicating that the reduction algorithm will produce a basis orthogonal in
the euclidean norm
"""
struct EuclideanNorm <: NormStyle end

"""
    struct EnergyNorm <: NormStyle
      norm_op::Function
    end

Trait indicating that the reduction algorithm will produce a basis orthogonal in
the norm specified by `norm_op`. Note: `norm_op` should represent a symmetric,
positive definite bilinear form (matrix)
"""
struct EnergyNorm <: NormStyle
  norm_op::Function
end

get_norm(n::EnergyNorm) = n.norm_op

"""
    abstract type Reduction{A<:ReductionStyle,B<:NormStyle} end

Type indicating the reduction strategy to employ, and the information regarding
the norm with respect to which the reduction should occur.

Subtypes:

- [`DirectReduction`](@ref)
- [`GreedyReduction`](@ref)
- [`SupremizerReduction`](@ref)
- [`HyperReduction`](@ref)
- [`TransientReduction`](@ref)
"""
abstract type Reduction{A<:ReductionStyle,B<:NormStyle} end

"""
    abstract type DirectReduction{A,B} <: Reduction{A,B} end

Type representing direct reduction methods, e.g. truncated POD, TTSVD, etc.

Subtypes:

- [`AffineReduction`](@ref)
- [`PODReduction`](@ref)
- [`TTSVDReduction`](@ref)
"""
abstract type DirectReduction{A,B} <: Reduction{A,B} end
abstract type GreedyReduction{A,B} <: Reduction{A,B} end

get_reduction(r::Reduction) = r
ReductionStyle(r::Reduction) = @abstractmethod
NormStyle(r::Reduction) = @abstractmethod
ParamDataStructures.num_params(r::Reduction) = @abstractmethod
get_norm(r::Reduction) = get_norm(NormStyle(r))

"""
    struct AffineReduction{A,B} <: DirectReduction{A,B}
      red_style::A
      norm_style::B
    end

Reduction employed when the input data is independent with respect to the
considered realization. Therefore, simply considering a number of parameters
equal to 1 suffices for this type of reduction
"""
struct AffineReduction{A,B} <: DirectReduction{A,B}
  red_style::A
  norm_style::B
end

function AffineReduction(red_style::ReductionStyle,norm_op::Function)
  norm_style = EnergyNorm(norm_op)
  AffineReduction(red_style,norm_style)
end

function AffineReduction(tol::Float64,norm_style=EuclideanNorm())
  red_style = SearchSVDRank(tol)
  AffineReduction(red_style,norm_style)
end

function AffineReduction(rank::Int,norm_style=EuclideanNorm())
  red_style = FixedSVDRank(rank)
  AffineReduction(red_style,norm_style)
end

ReductionStyle(r::AffineReduction) = r.red_style
NormStyle(r::AffineReduction) = r.norm_style
ParamDataStructures.num_params(r::AffineReduction) = 1

"""
    struct PODReduction{A,B} <: DirectReduction{A,B}
      red_style::A
      norm_style::B
      nparams::Int
    end

Reduction by means of a truncated POD. The field `nparams` indicates the number
of parameters selected for the computation of the snapshots
"""
struct PODReduction{A,B} <: DirectReduction{A,B}
  red_style::A
  norm_style::B
  nparams::Int
end

function PODReduction(red_style::ReductionStyle,norm_style::NormStyle=EuclideanNorm();nparams=50)
  iszero(nparams) && return AffineReduction(red_style,norm_style)
  PODReduction(red_style,norm_style,nparams)
end

function PODReduction(red_style::ReductionStyle,norm_op::Function;kwargs...)
  norm_style = EnergyNorm(norm_op)
  PODReduction(red_style,norm_style;kwargs...)
end

function PODReduction(tolrank::Union{Float64,Int},args...;nparams=50,kwargs...)
  red_style = ReductionStyle(tolrank;kwargs...)
  PODReduction(red_style,args...;nparams)
end

ReductionStyle(r::PODReduction) = r.red_style
NormStyle(r::PODReduction) = r.norm_style
ParamDataStructures.num_params(r::PODReduction) = r.nparams

"""
    struct TTSVDReduction{B} <: DirectReduction{TTSVDRanks,B}
      red_style::TTSVDRanks
      norm_style::B
      nparams::Int
    end

Reduction by means of a TTSVD. The field `nparams` indicates the number
of parameters selected for the computation of the snapshots
"""
struct TTSVDReduction{B} <: DirectReduction{TTSVDRanks,B}
  red_style::TTSVDRanks
  norm_style::B
  nparams::Int
end

function TTSVDReduction(red_style::ReductionStyle,norm_style::NormStyle=EuclideanNorm();nparams=50)
  TTSVDReduction(red_style,norm_style,nparams)
end

function TTSVDReduction(red_style::ReductionStyle,norm_op::Function;nparams=50,kwargs...)
  norm_style = EnergyNorm(norm_op)
  TTSVDReduction(red_style,norm_style;nparams)
end

function TTSVDReduction(tolrank,args...;nparams=50,kwargs...)
  red_style = TTSVDRanks(tolrank;kwargs...)
  TTSVDReduction(red_style,args...;nparams)
end

Base.size(r::TTSVDReduction) = (length(r.tols),)
Base.getindex(r::TTSVDReduction,i::Integer) = Reduction(r.red_style[i],r.norm_style;nparams=r.nparams)

ReductionStyle(r::TTSVDReduction) = r.red_style
NormStyle(r::TTSVDReduction) = r.norm_style
ParamDataStructures.num_params(r::TTSVDReduction) = r.nparams

function PODReduction(r::TTSVDReduction)
  pod_style = first(ReductionStyle(r))
  norm_style = NormStyle(r)
  nparams = num_params(r)
  PODReduction(pod_style,norm_style,nparams)
end

function TTSVDReduction(r::PODReduction,D=3)
  ttsvd_style = repeat(ReductionStyle(r),D)
  norm_style = NormStyle(r)
  nparams = num_params(r)
  TTSVDReduction(ttsvd_style,norm_style,nparams)
end

"""
    struct LocalReduction{A,B,R<:Reduction{A,B}} <: Reduction{A,B}
      reduction::R
      ncentroids::Int
    end
"""
struct LocalReduction{A,B,R<:Reduction{A,B}} <: Reduction{A,B}
  reduction::R
  ncentroids::Int
end

function LocalReduction(args...;ncentroids=10,kwargs...)
  reduction = Reduction(args...;kwargs...)
  LocalReduction(reduction,ncentroids)
end

get_ncentroids(r::LocalReduction) = r.ncentroids
get_reduction(r::LocalReduction) = r.reduction
ReductionStyle(r::LocalReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::LocalReduction) = NormStyle(get_reduction(r))
ParamDataStructures.num_params(r::LocalReduction) = num_params(get_reduction(r))

"""
    struct SupremizerReduction{A,R<:Reduction{A,EnergyNorm}} <: Reduction{A,EnergyNorm}
      reduction::R
      supr_op::Function
      supr_tol::Float64
    end

Wrapper for reduction methods `reduction` that require an additional step of
stabilization, by means of a supremizer enrichment. Check [this](https://doi.org/10.1002/nme.4772)
for more details in a steady setting, and [this](https://doi.org/10.1137/22M1509114) for
more details in a transient setting. The fields `supr_op` and `supr_tol` (which
is only needed only in transient applications) are respectively the supremizing
operator and the tolerance involved in the enrichment. For a saddle point problem
with a Jacobian of the form

[ A   Bᵀ
  B   0 ]

this operator is given by the bilinear form representing the matrix Bᵀ.
"""
struct SupremizerReduction{A,R<:Reduction{A,EnergyNorm}} <: Reduction{A,EnergyNorm}
  reduction::R
  supr_op::Function
  supr_tol::Float64
end

function SupremizerReduction(supr_op::Function,args...;supr_tol=1e-2,kwargs...)
  reduction = Reduction(args...;kwargs...)
  SupremizerReduction(reduction,supr_op,supr_tol)
end

get_supr(r::SupremizerReduction) = r.supr_op
get_supr_tol(r::SupremizerReduction) = r.supr_tol

get_reduction(r::SupremizerReduction) = r.reduction
ReductionStyle(r::SupremizerReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::SupremizerReduction) = NormStyle(get_reduction(r))
ParamDataStructures.num_params(r::SupremizerReduction) = num_params(get_reduction(r))

# generic constructor

function Reduction(red_style::ReductionStyle,args...;kwargs...)
  PODReduction(red_style,args...;kwargs...)
end

function Reduction(red_style::TTSVDRanks,args...;kwargs...)
  TTSVDReduction(red_style,args...;kwargs...)
end

function Reduction(tolrank::Union{Float64,Int,AbstractVector},args...;nparams=50,kwargs...)
  red_style = ReductionStyle(tolrank;kwargs...)
  Reduction(red_style,args...;nparams)
end

function Reduction(red::Reduction,args...;kwargs...)
  red
end

function Reduction(supr_op::Function,args...;kwargs...)
  SupremizerReduction(supr_op,args...;kwargs...)
end

"""
    abstract type HyperReduction{A} <: Reduction{A,EuclideanNorm} end

Type representing a hyper-reduction approximation by means of a MDEIMProjection algorithm.
Check [this](https://doi.org/10.1016/j.jcp.2015.09.046) for more details on MDEIMProjection. This
reduction strategy is usually employed only for the approximation of a residual
and/or Jacobian of a differential problem. Note that orthogonality with respect
to a norm other than the euclidean is not required for this reduction type.

Subtypes:

- [`MDEIMReduction`](@ref)
- [`TransientMDEIMReduction`](@ref)
"""
abstract type HyperReduction{A<:ReductionStyle} <: Reduction{A,EuclideanNorm} end

function HyperReduction(args...;kwargs...)
  reduction = Reduction(args...;kwargs...)
  MDEIMReduction(reduction)
end

function HyperReduction(reduction::Reduction;kwargs...)
  red_style = ReductionStyle(reduction)
  HyperReduction(red_style;kwargs...)
end

"""
    struct MDEIMReduction{A} <: HyperReduction{A}
      reduction::Reduction{A,EuclideanNorm}
    end

MDEIMProjection struct employed in steady problems
"""
struct MDEIMReduction{A} <: HyperReduction{A}
  reduction::Reduction{A,EuclideanNorm}
end

get_reduction(r::MDEIMReduction) = r.reduction
ReductionStyle(r::MDEIMReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::MDEIMReduction) = NormStyle(get_reduction(r))
ParamDataStructures.num_params(r::MDEIMReduction) = num_params(get_reduction(r))

"""
    struct InterpHyperReduction{A} <: HyperReduction{A}
      reduction::Reduction{A,EuclideanNorm}
      strategy::AbstractRadialBasis
    end
"""
struct InterpHyperReduction{A} <: HyperReduction{A}
  reduction::Reduction{A,EuclideanNorm}
  strategy::AbstractRadialBasis
end

function InterpHyperReduction(args...;kwargs...)
  reduction = Reduction(args...;kwargs...)
  strategy = PHS()
  InterpHyperReduction(reduction,strategy)
end

function InterpHyperReduction(reduction::Reduction;kwargs...)
  red_style = ReductionStyle(reduction)
  InterpHyperReduction(red_style;kwargs...)
end

get_reduction(r::InterpHyperReduction) = r.reduction
ReductionStyle(r::InterpHyperReduction) = ReductionStyle(get_reduction(r))
NormStyle(r::InterpHyperReduction) = NormStyle(get_reduction(r))
interp_strategy(r::InterpHyperReduction) = r.strategy
ParamDataStructures.num_params(r::InterpHyperReduction) = num_params(get_reduction(r))

const LocalHyperReduction{A} = LocalReduction{A,EuclideanNorm,<:HyperReduction{A}}

function LocalHyperReduction(args...;interp=false,ncentroids=10,kwargs...)
  red = interp ? InterpHyperReduction(args...;kwargs...) : HyperReduction(args...;kwargs...)
  LocalReduction(red;ncentroids)
end

"""
    struct AdaptiveReduction{A,B,R<:DirectReduction{A,B}} <: GreedyReduction{A,B}
      reduction::R
      adaptive_nparams::Int
      adaptive_tol::Float64
      adaptive_maxiter::Int
    end

Not implemented yet. Will serve as a parameter-adaptivity greedy reduction algorithm
"""
struct AdaptiveReduction{A,B,R<:DirectReduction{A,B}} <: GreedyReduction{A,B}
  reduction::R
  adaptive_nparams::Int
  adaptive_tol::Float64
  adaptive_maxiter::Int
end

function AdaptiveReduction(
  red_style::ReductionStyle,
  args...;
  adaptive_nparams=10,
  adaptive_tol=1e-2,
  adaptive_maxiter=10)

  reduction = Reduction(red_style,args...;kwargs...)
  AdaptiveReduction(reduction,adaptive_nparams,adaptive_tol,adaptive_maxiter)
end
