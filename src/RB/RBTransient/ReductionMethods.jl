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

``\tfrac{du}{dt} - \nu \Delta u = f in Ω \times [0,T]``

subject to initial/boundary conditions. Upon applying a FE discretization in space,
and a `θ`-method in time, one gets the space-time system

``A\_{\theta} u\_{\theta} = f\_{\theta}``

where

```math
A\_{\theta} =
\begin{bmatrix}
A\_1 + M / (\theta \Delta t) & 0 & 0 & \hdots & 0 & 0 \\
- M / (\theta \Delta t) & A\_2 + M / (\theta \Delta t) & 0 & \hdots & 0 & 0 \\
0 & - M / (\theta \Delta t) & A\_3 + M / (\theta \Delta t) & & 0 & 0 \\
& \ddots & \ddots & \ddots & & \\
0 & 0 & 0 & \hdots & - M / (\theta \Delta t) & A\_n + M / (\theta \Delta t)
\end{bmatrix}
```

```
u\_{\theta} = [(1-\theta)u\_0 + \theta u\_1, \hdots, (1-\theta)u\_{n-1} + \theta u\_n]\^T
```

```
f\_{\theta} = [f\_1, \hdots, f\_n]\^T
```

where ``A\_k = A(t\_{k-1} + \theta \Delta t)`` and ``f\_k = f(t\_{k-1} + \theta \Delta t)``.

Note: instead of multiplying `A\_{\theta}` by `u\_{\theta}`, we multiply `A\^{\tilde}\_{\theta}` by `u`, where

  ```A\^{\tilde}\_{\theta} = tridiag((1-\theta)A\_{k-1} - M / \Delta t, \theta A\_k + M / \Delta t, 0) ```

We now denote with ``\Phi`` and ``\Psi`` the spatial and temporal basis obtained by reducing the
snapshots associated to the state variable `u`. The Galerkin projection of the
space-time system is equal to `Âθ * û = f̂θ`, where `û` is the unknown, and

```math
\begin{equation}
\begin{split}
\hat{A}\_{\theta} &= \sum\limits\_{k=1}\^{n-1} ( (1-θ) \Phi\^T A\_k \Phi - \Phi\^T M \Phi / \Delta t) \otimes \Psi[k-1,:]\^T \Psi[k,:]
  + \sum\limits\_{k=1}\^n (θ \Phi\^T A\_k \Phi + \Phi\^T M \Phi / \Delta t) \otimes \Psi[k,:]\^T \Psi[k,:] \\
  &= \theta A_{backwards} + (1-\theta)A_{forwards} + (M_{backwards} + M_{forwards}) / \Delta t \\
\hat{f}\_{\theta} &= \sum\limits\_{k=1}\^n \Phi\^T f\_k \otimes \Psi[k,:]
\end{split}
\end{equation}
```

We notice that the expression of ``\hat{A}\_{\theta}`` can be written in a more general form as

```\hat{A}\_{\theta} = combine_A(A_{backwards},A_{forwards}) + combine_M(M_{backwards},M_{forwards})```

where combine_A and combine_M are two function specific to A and M:

  ```combine_A(x,y) = \theta y + (1-\theta)y
     combine_M(x,y) = (x - y) / \Delta t```

The same can be said of any time marching scheme. This is the meaning of the
function `combine`. Note that for a time marching with `p` interpolation points (e.g.
for `θ` method, `p = 2`) the `combine` functions will have to accept `p` arguments.
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
