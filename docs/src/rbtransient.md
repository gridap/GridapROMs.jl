```@meta
CurrentModule = GridapROMs.RBTransient
```

# GridapROMs.RBTransient 

Reduced-basis infrastructure for transient parametric PDEs, extending
[`RBSteady`](@ref) to the space–time setting.

## Reduction methods

High-dimensional (transient) reduction methods extend the steady
[`Reduction`](@ref) to handle the time dimension. The abstract supertype
[`TransientReduction`](@ref) dispatches to one of three strategies depending
on the arguments: [`SteadyReduction`](@ref), [`KroneckerReduction`](@ref),
and [`SequentialReduction`](@ref).

## Hyper-reduction methods

Transient hyper-reduction wraps a [`TransientReduction`](@ref) together with a
[`TimeCombination`](@ref) that encodes the ODE time-marching coefficients.
See [`TransientHyperReduction`](@ref) for details on how the θ-method (and
higher-order schemes) lead to per-operator combination orders. Concrete
subtypes include [`TransientDEIMHyperReduction`](@ref),
[`TransientSOPTHyperReduction`](@ref), and [`TransientRBFHyperReduction`](@ref).

## Bases construction

The [`tucker`](@ref) function computes a Tucker decomposition of
multi-dimensional snapshot arrays.

## Transient projections

[`TransientProjection`](@ref) wraps spatial and temporal projections for
space–time reduced-basis problems, while
[`TransientIntegrationDomain`](@ref) provides the corresponding reduced
integration domain.

## Full API

```@autodocs
Modules = [RBTransient,]
```