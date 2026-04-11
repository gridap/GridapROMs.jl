# GridapROMs.jl Release Notes

## Release: v0.1.2

### New features

#### Space-time ROMs for Generalized-α methods

Space-time reduced-order models are now supported for the **Generalized-α
method for first-order ODEs** (`GeneralizedAlpha1`) and the **Generalized-α
method for second-order ODEs** (`GeneralizedAlpha2`, Newmark family).
Previously, space-time Galerkin projection was limited to the θ-method
(`ThetaMethod`).

The generalized-α schemes require tracking multiple derivative levels
simultaneously (velocity and acceleration for second-order problems), which
demanded a more general treatment of how time-level contributions are combined
before the reduced projection is applied. This is handled by the new
`TimeCombination` abstraction described below.

#### `TimeCombination` abstraction for space-time Galerkin projections

Space-time reduced systems are assembled by combining FOM snapshots at
successive time levels before projecting onto the reduced basis. Different
ODE schemes weight these time levels differently, and can involve several
derivative orders (stiffness, damping, mass). The new `TimeCombination`
abstract type encodes this scheme-specific logic in a single, composable
object.

A `TimeCombination` stores the time-marching parameters (time step, implicitness
weights, etc.) and provides two key operations:

- `get_coefficients(c, N)` — returns the tuple of weights applied to snapshots
  at successive time levels for a given derivative order.
- `time_combination(c, u, us0)` — applies the full combination to a
  parametric solution vector `u` and initial-condition vectors `us0`, returning
  one combined vector per derivative order of the ODE.

The concrete subtypes and the solvers they correspond to are:

| Solver              | `TimeCombination` subtype   |
|:--------------------|:----------------------------|
| `ThetaMethod`       | `ThetaMethodCombination`    |
| `GeneralizedAlpha1` | `GenAlpha1Combination`      |
| `GeneralizedAlpha2` | `GenAlpha2Combination`      |

Each subtype is dispatched on via `CombinationOrder{A,N}` (aliased as
`ThetaMethodStrategy{N}`, `GenAlpha1Strategy{N}`, `GenAlpha2Strategy{N}`),
where `N` selects the derivative order (1 = stiffness, 2 = damping/mass for
first-order, 3 = mass for second-order). This makes the space-time assembly
path fully generic: the `residual` and `jacobian` methods in `SpaceTime.jl`
call `TimeCombination(solver)` and `time_combination` without any
solver-specific branching.
