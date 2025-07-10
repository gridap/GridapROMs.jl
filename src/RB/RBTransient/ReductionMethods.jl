abstract type HighOrderReduction{A<:ReductionStyle,B<:NormStyle} <: Reduction{A,B} end

RBSteady.get_reduction(r::HighOrderReduction) = @abstractmethod

"""
    struct KroneckerReduction{A,B} <: HighOrderReduction{A,B}
      reductions::AbstractVector{<:Reduction}
    end

Wrapper for reduction methods in high order problems, such as transient ones. The
reduced subspaces are constructed as Kronecker product spaces
"""
struct KroneckerReduction{A,B} <: HighOrderReduction{A,B}
  reductions::AbstractVector{<:Reduction}
  function KroneckerReduction(reductions::AbstractVector{<:Reduction})
    A = ReductionStyle(first(reductions))
    B = NormStyle(first(reductions))
    new{A,B}(reductions)
  end
end

ReductionStyle(r::KroneckerReduction) = ReductionStyle(first(r.reductions))
ParamDataStructures.num_params(r::KroneckerReduction) = num_params(first(r.reductions))

# generic constructor

function HighOrderReduction end

const TransientReduction = HighOrderReduction

function HighOrderReduction(reduction::HighOrderReduction,args...;kwargs...)
  reduction
end

function HighOrderReduction(styles::AbstractVector{<:ReductionStyle},args...;kwargs...)
  reductions = map(s -> Reduction(s,args...;kwargs...),styles)
  KroneckerReduction(reductions)
end

function HighOrderReduction(tolranks::AbstractVector{<:Union{Int,Float64}},args...;kwargs...)
  reductions = map(t -> Reduction(t,args...;kwargs...),tolranks)
  KroneckerReduction(reductions)
end

function HighOrderReduction(tolrank::Union{Int,Float64},args...;dim=2,kwargs...)
  HighOrderReduction(Fill(tolrank,dim),args...;kwargs...)
end

function HighOrderReduction(red_style::ReductionStyle,args...;dim=2,kwargs...)
  HighOrderReduction(Fill(red_style,dim),args...;kwargs...)
end

"""
    struct SequentialReduction{A,B} <: HighOrderReduction{A,B}
      reduction::Reduction{A,B}
    end

Wrapper for sequential reduction methods in high-order problems, e.g. TT-SVD in
transient applications
"""
struct SequentialReduction{A,B} <: HighOrderReduction{A,B}
  reduction::Reduction{A,B}
end

RBSteady.get_reduction(r::SequentialReduction) = r.reduction

function HighOrderReduction(red_style::TTSVDRanks,args...;kwargs...)
  reduction = TTSVDReduction(red_style,args...;kwargs...)
  SequentialReduction(reduction)
end

function HighOrderReduction(tolrank::Union{Vector{Int},Vector{Float64}},args...;kwargs...)
  reduction = TTSVDReduction(tolrank,args...;kwargs...)
  SequentialReduction(reduction)
end

function HighOrderReduction(supr_op::Function,args...;supr_tol=1e-2,kwargs...)
  reduction = HighOrderReduction(args...;kwargs...)
  SupremizerReduction(reduction,supr_op,supr_tol)
end

@doc raw"""
    abstract type HighOrderHyperReduction{A} <: HyperReduction{A} end

Hyper reduction strategies employed in high-order (e.g. transient) problems.
They feature a field `combine`, a function used to group the reductions relative
to the various Jacobians(in general, more than one in transient problems) in a
smart way. We consider, for example, the ODE

``\tfrac{du}{dt} - \nu \Delta u = f \ \ \text{in} \ \ Ω \times [0,T]``

subject to initial/boundary conditions. Upon applying a FE discretization in space,
and a `θ`-method in time, one gets the space-time system

``A_{\theta} u_{\theta} = f_{\theta}``

where

```math
A_{\theta} = \begin{bmatrix}
A_1 + M / (\theta \Delta t) & & & & & \\
- M / (\theta \Delta t) & A_2 + M / (\theta \Delta t) & & & & \\
& - M / (\theta \Delta t) & A_3 + M / (\theta \Delta t) & & & \\
& & \ddots & \ddots & & \\
& & & & - M / (\theta \Delta t) & A_n + M / (\theta \Delta t)
\end{bmatrix};
```

```math
u_{\theta} = \begin{bmatrix}
& (1-\theta)u_0 + \theta u_1 & \hdots & (1-\theta)u_{n-1} + \theta u_n
\end{bmatrix}^T;
```

```math
f_{\theta} = \begin{bmatrix}
& f_1 & \hdots & f_n
\end{bmatrix}^T;
```

```math
A_k = A(t_{k-1} + \theta \Delta t);
```

```math
f_k = f(t_{k-1} + \theta \Delta t).
```

Note: instead of multiplying ``A_{\theta}`` by ``u_{\theta}``, we multiply ``\tilde{A}_{\theta}`` by ``u``, where

```math
\tilde{A}_{\theta} = tridiag((1-\theta)A_{k-1} - M / \Delta t, \theta A_k + M / \Delta t, 0).
```

We now denote with ``\Phi`` and ``\Psi`` the spatial and temporal basis obtained by reducing the
snapshots associated to the state variable ``u``. The Galerkin projection of the
space-time system is equal to ``\hat{A}_{\theta}\hat{u} = \hat{f}_{\theta}``, where ``\hat{u}`` is the unknown, and

```math
\begin{align*}
\hat{A}_{\theta} &= \sum\limits_{k=1}^{n-1} ( (1-θ) \Phi^T A_k \Phi - \Phi^T M \Phi / \Delta t) \otimes \Psi[k-1,:]^T \Psi[k,:]
  + \sum\limits_{k=1}^n (\theta \Phi^T A_k \Phi + \Phi^T M \Phi / \Delta t) \otimes \Psi[k,:]^T \Psi[k,:] \\
  &= \theta A_{backwards} + (1-\theta)A_{forwards} + (M_{backwards} + M_{forwards}) / \Delta t \\
\hat{f}_{\theta} &= \sum\limits_{k=1}^n \Phi^T f_k \otimes \Psi[k,:]
\end{align*}
```

We notice that the expression of ``\hat{A}_{\theta}`` can be written in a more general form as

```math
\hat{A}_{\theta} = combine_A(A_{backwards},A_{forwards}) + combine_M(M_{backwards},M_{forwards}),
```

where combine_A and combine_M are two function specific to A and M:

```math
\begin{align*}
combine_A(x,y) &= \theta y + (1-\theta)y \\
combine_M(x,y) &= (x - y) / \Delta t
\end{align*}
```

The same can be said of any time marching scheme. This is the meaning of the
function `combine`. Note that for a time marching with ``p`` interpolation points (e.g.
for ``\theta`` method, ``p = 2``) the `combine` functions will have to accept ``p`` arguments.
"""
abstract type HighOrderHyperReduction{A} <: HyperReduction{A} end

function HighOrderHyperReduction end

const TransientHyperReduction = HighOrderHyperReduction

function HighOrderHyperReduction(combine::Function,args...;hypred_strategy=:mdeim,kwargs...)
  reduction = HighOrderReduction(args...;kwargs...)
  hypred_strategy==:mdeim ? HighOrderMDEIMHyperReduction(reduction,combine) : HighOrderRBFHyperReduction(reduction,combine)
end

function HighOrderHyperReduction(reduction::HighOrderReduction,combine::Function;kwargs...)
  red_style = ReductionStyle(reduction)
  HighOrderMDEIMHyperReduction(combine,red_style;kwargs...)
end

get_combine(r::HighOrderHyperReduction) = @abstractmethod

struct HighOrderMDEIMHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: HighOrderHyperReduction{A}
  reduction::R
  combine::Function
end

function HighOrderMDEIMHyperReduction(combine::Function,args...;kwargs...)
  reduction = HighOrderReduction(args...;kwargs...)
  HighOrderMDEIMHyperReduction(reduction,combine)
end

RBSteady.get_reduction(r::HighOrderMDEIMHyperReduction) = r.reduction
get_combine(r::HighOrderMDEIMHyperReduction) = r.combine

struct HighOrderRBFHyperReduction{A,R<:Reduction{A,EuclideanNorm}} <: HighOrderHyperReduction{A}
  reduction::R
  combine::Function
  strategy::AbstractRadialBasis
end

function HighOrderRBFHyperReduction(combine::Function,args...;strategy=PHS(),kwargs...)
  reduction = HighOrderReduction(args...;kwargs...)
  HighOrderRBFHyperReduction(reduction,combine,strategy)
end

RBSteady.get_reduction(r::HighOrderRBFHyperReduction) = r.reduction
RBSteady.interp_strategy(r::HighOrderRBFHyperReduction) = r.strategy
get_combine(r::HighOrderRBFHyperReduction) = r.combine

const LocalHyperReduction{A} = LocalReduction{A,EuclideanNorm,<:HighOrderHyperReduction{A}}

function LocalHighOrderHyperReduction(args...;ncentroids=10,kwargs...)
  reduction = HighOrderHyperReduction(args...;kwargs...)
  LocalReduction(reduction;ncentroids)
end
