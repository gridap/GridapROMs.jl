TransientReductionStyle(args...;kwargs...) = @abstractmethod

function TransientReductionStyle(tolrank_space,tolrank_time;kwargs...)
  style_space = ReductionStyle(tol_space;kwargs...)
  style_time = ReductionStyle(tol_time;kwargs...)
  return style_space,style_time
end

function TransientReductionStyle(tolrank;kwargs...)
  TransientReductionStyle(tolrank,tolrank;kwargs...)
end

"""
    struct TransientReduction{A,B,RS<:Reduction{A,B},RT<:Reduction{A,EuclideanNorm}} <: Reduction{A,B}
      reduction_space::RS
      reduction_time::RT
    end

Wrapper for reduction methods in transient problems. The fields `reduction_space`
and `reduction_time` respectively represent the spatial reduction method, and
the temporal one
"""
struct TransientReduction{A,B,RS<:Reduction{A,B},RT<:Reduction{A,EuclideanNorm}} <: Reduction{A,B}
  reduction_space::RS
  reduction_time::RT
end

const TransientAffineReduction{A,B} = TransientReduction{A,B,AffineReduction{A,B},AffineReduction{A,EuclideanNorm}}
const TransientPODReduction{A,B} = TransientReduction{A,B,PODReduction{A,B},PODReduction{A,EuclideanNorm}}

# generic constructor

function TransientReduction(style_space::ReductionStyle,style_time::ReductionStyle,args...;kwargs...)
  reduction_space = Reduction(style_space,args...;kwargs...)
  reduction_time = Reduction(style_time;kwargs...)
  TransientReduction(reduction_space,reduction_time)
end

function TransientReduction(red_style::ReductionStyle,args...;kwargs...)
  TransientReduction(red_style,red_style,args...;kwargs...)
end

function TransientReduction(red_style::TTSVDRanks,args...;kwargs...)
  TTSVDReduction(red_style,args...;kwargs...)
end

function TransientReduction(
  tolrank_space::Union{Int,Float64},
  tolrank_time::Union{Int,Float64},
  args...;kwargs...)

  reduction_space = Reduction(tolrank_space,args...;kwargs...)
  reduction_time = Reduction(tolrank_time;kwargs...)
  TransientReduction(reduction_space,reduction_time)
end

function TransientReduction(tolrank::Union{Int,Float64},args...;kwargs...)
  TransientReduction(tolrank,tolrank,args...;kwargs...)
end

function TransientReduction(tolrank::Union{Vector{Int},Vector{Float64}},args...;kwargs...)
  TTSVDReduction(tolrank,args...;kwargs...)
end

function TransientReduction(supr_op::Function,args...;supr_tol=1e-2,kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  SupremizerReduction(reduction,supr_op,supr_tol)
end

get_reduction_space(r::TransientReduction) = get_reduction(r.reduction_space)
get_reduction_time(r::TransientReduction) = get_reduction(r.reduction_time)
RBSteady.ReductionStyle(r::TransientReduction) = ReductionStyle(get_reduction_space(r))
RBSteady.NormStyle(r::TransientReduction) = NormStyle(get_reduction_space(r))
ParamDataStructures.num_params(r::TransientReduction) = num_params(get_reduction_space(r))

@doc raw"""
    struct TransientMDEIMReduction{A,R<:Reduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
      reduction::R
      combine::Function
    end

MDEIM struct employed in transient problems. The field `combine` is a function
used to group the reductions relative to the various Jacobians(in general, more
than one in transient problems) in a smart way. We consider, for example, the ODE

``\tfrac{du}{dt} - \nu \Delta u = f \ \ \text{in} \ \ Ω \times [0,T]``

subject to initial/boundary conditions. Upon applying a FE discretization in space,
and a `θ`-method in time, one gets the space-time system

``A_{\theta} u_{\theta} = f_{\theta}``

where

```math
\begin{equation}
A_{\theta} =
\begin{bmatrix}
A_1 + M / (\theta \Delta t) & 0 & 0 & \hdots & 0 & 0 \\
- M / (\theta \Delta t) & A_2 + M / (\theta \Delta t) & 0 & \hdots & 0 & 0 \\
0 & - M / (\theta \Delta t) & A_3 + M / (\theta \Delta t) & & 0 & 0 \\
& \ddots & \ddots & \ddots & & \\
0 & 0 & 0 & \hdots & - M / (\theta \Delta t) & A_n + M / (\theta \Delta t)
\end{bmatrix};
\end{equation}
```

```math
\begin{equation}
u_{\theta} = [(1-\theta)u_0 + \theta u_1, \hdots, (1-\theta)u_{n-1} + \theta u_n]^T;
\end{equation}
```

```math
\begin{equation}
f_{\theta} = [f_1, \hdots, f_n]^T;
\end{equation}
```

```math
\begin{equation}
A_k = A(t_{k-1} + \theta \Delta t);
\end{equation}
```

```math
\begin{equation}
f_k = f(t_{k-1} + \theta \Delta t).
\end{equation}
```

Note: instead of multiplying ``A_{\theta}`` by ``u_{\theta}``, we multiply ``\tilde{A}_{\theta}`` by ``u``, where

```math
\begin{equation}
\tilde{A}_{\theta} = tridiag((1-\theta)A_{k-1} - M / \Delta t, \theta A_k + M / \Delta t, 0).
\end{equation}
```

We now denote with ``\Phi`` and ``\Psi`` the spatial and temporal basis obtained by reducing the
snapshots associated to the state variable ``u``. The Galerkin projection of the
space-time system is equal to ``\hat{A}_{\theta}\hat{u} = \hat{f}_{\theta}``, where ``\hat{u}`` is the unknown, and

```math
\begin{equation}
\begin{split}
\hat{A}_{\theta} &= \sum\limits_{k=1}^{n-1} ( (1-θ) \Phi^T A_k \Phi - \Phi^T M \Phi / \Delta t) \otimes \Psi[k-1,:]^T \Psi[k,:]
  + \sum\limits_{k=1}^n (θ \Phi^T A_k \Phi + \Phi^T M \Phi / \Delta t) \otimes \Psi[k,:]^T \Psi[k,:] \\
  &= \theta A_{backwards} + (1-\theta)A_{forwards} + (M_{backwards} + M_{forwards}) / \Delta t \\
\hat{f}_{\theta} &= \sum\limits_{k=1}^n \Phi^T f_k \otimes \Psi[k,:]
\end{split}
\end{equation}
```

We notice that the expression of ``\hat{A}_{\theta}`` can be written in a more general form as

```math
\begin{equation}
\hat{A}_{\theta} = combine_A(A_{backwards},A_{forwards}) + combine_M(M_{backwards},M_{forwards}),
\end{equation}
```

where combine_A and combine_M are two function specific to A and M:

```math
\begin{equation}
\begin{split}
combine_A(x,y) &= \theta y + (1-\theta)y \\
combine_M(x,y) &= (x - y) / \Delta t
\end{split}
\end{equation}
```

The same can be said of any time marching scheme. This is the meaning of the
function `combine`. Note that for a time marching with ``p`` interpolation points (e.g.
for ``\theta`` method, ``p = 2``) the `combine` functions will have to accept ``p`` arguments.
"""
struct TransientMDEIMReduction{A,R<:Reduction{A,EuclideanNorm}} <: AbstractMDEIMReduction{A}
  reduction::R
  combine::Function
end

function TransientMDEIMReduction(combine::Function,args...;kwargs...)
  reduction = TransientReduction(args...;kwargs...)
  TransientMDEIMReduction(reduction,combine)
end

RBSteady.get_reduction(r::TransientMDEIMReduction) = get_reduction(r.reduction)
RBSteady.ReductionStyle(r::TransientMDEIMReduction) = ReductionStyle(get_reduction(r))
RBSteady.NormStyle(r::TransientMDEIMReduction) = NormStyle(get_reduction(r))
ParamDataStructures.num_params(r::TransientMDEIMReduction) = num_params(get_reduction(r))
get_combine(r::TransientMDEIMReduction) = r.combine
