```@meta
CurrentModule = GridapROMs.RBTransient
```

# GridapROMs.RBTransient 

Reduced-basis infrastructure for transient parametric PDEs, extending
[`RBSteady`](@ref) to the space–time setting.

## Reduction methods

High-dimensional (transient) reduction methods extend the steady
[`Reduction`](@ref) to handle the time dimension. The abstract supertype
[`HighDimReduction`](@ref) dispatches to one of three strategies depending
on the arguments:

```@docs
HighDimReduction
SteadyReduction
KroneckerReduction
SequentialReduction
```

## Hyper-reduction methods

Transient hyper-reduction wraps a [`HighDimReduction`](@ref) together with a
[`TimeCombination`](@ref) that encodes the ODE time-marching coefficients.
See [`HighDimHyperReduction`](@ref) for details on how the θ-method (and
higher-order schemes) lead to per-operator combination orders.

```@docs
HighDimHyperReduction
HighDimMDEIMHyperReduction
HighDimSOPTHyperReduction
HighDimRBFHyperReduction
```

## Bases construction

```@docs
tucker
```

## Transient projections

```@docs
TransientProjection
TransientIntegrationDomain
```

## Full API

```@autodocs
Modules = [RBTransient,]
```