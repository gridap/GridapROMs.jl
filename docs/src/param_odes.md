```@meta
CurrentModule = GridapROMs.ParamODEs
```

# GridapROMs.ParamODEs 

This module provides the parametric ODE infrastructure, including time-stepping
schemes (θ-method, Generalized-α) and the [`TimeCombination`](@ref) framework
that encodes how each scheme combines contributions from different time levels.

## Time combinations

A [`TimeCombination`](@ref) stores the parameters of an ODE time-marching
scheme and provides the coefficients needed to combine snapshots from
successive time levels in the reduced-basis context. Each operator in the
semi-discrete ODE (stiffness, damping, mass) is assigned a
[`CombinationOrder`](@ref), which selects the appropriate set of
coefficients.

```@docs
TimeCombination
CombinationOrder
ThetaMethodCombination
GenAlpha1Combination
GenAlpha2Combination
ThetaMethodStrategy
GenAlpha1Strategy
GenAlpha2Strategy
get_coefficients
time_combination
```

## Full API

```@autodocs
Modules = [ParamODEs,]
```