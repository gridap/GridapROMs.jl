# GridapROMs.jl Release Notes

## Release: v0.1.3

### New features

#### Improved handling of local quantities

The local (clustered) ROM execution path has been strengthened across
parametric blocks, local operator changes, and local solve workflows. In
practice this improves robustness when evaluating and assembling local
contributions under local compression settings and local hyper-reduction.

At API level, local workflows are now first-class in both steady and
transient stacks, including dedicated local projection/hyper-reduction
components (for example `LocalProjection`, `LocalHyperReduction`,
`HighDimLocalHyperReduction`, and their related local interpolation and
contribution paths).

#### Diagnostics API for steady and transient workflows

Diagnostics are now supported and exercised in both steady and transient
test pipelines:

- Steady diagnostics test: [test/RBSteady/diagnostics.jl](test/RBSteady/diagnostics.jl)
- Transient diagnostics test: [test/RBTransient/diagnostics.jl](test/RBTransient/diagnostics.jl)

The diagnostics entry-point signatures used in tests are:

- Steady: `rom_diagnostics(dir,rbsolver,feop)`
- Transient: `rom_diagnostics(dir,rbsolver,feop,xh0μ)`

Both tests run end-to-end diagnostics after snapshot generation and test
execution (`run_test`), then print the diagnostics object returned by
`rom_diagnostics`.

The main diagnostics implementation is provided by new dedicated modules:

- `src/RB/RBSteady/Diagnostics.jl`
- `src/RB/RBTransient/Diagnostics.jl`

and is integrated in the standard test matrix via:

- `test/runtests.jl` testset `steady diagnostics`
- `test/runtests.jl` testset `transient diagnostics`

#### Hyper-reduction strategy keywords, including affine and none

The hyper-reduction strategy keyword interface now explicitly supports:

- `:mdeim`
- `:rbf`
- `:sopt`
- `:none`
- `:affine`

The strategies `:mdeim`, `:rbf`, and `:sopt` were already available in
previous releases. This release extends the interface with `:none` and
`:affine`.

In addition, `:none` accepts aliases `:no` and `:nohr`.

The steady and transient diagnostics tests validate these options by looping
over all strategies when constructing the RB solver:

- `RBSolver(...; hypred_strategy=:mdeim, ...)`
- `RBSolver(...; hypred_strategy=:rbf, ...)`
- `RBSolver(...; hypred_strategy=:sopt, ...)`
- `RBSolver(...; hypred_strategy=:none, ...)`
- `RBSolver(...; hypred_strategy=:affine, ...)`

The `:affine` option targets parameter-independent (μ-independent)
structures, while `:none` disables hyper-reduction entirely for validation,
debugging, or accuracy-oriented runs.

### Additional technical changes since v0.1.2

#### Transient Galerkin and projection workflow updates

Transient Galerkin/projection paths have been substantially updated and are
now covered by a dedicated test module:

- `test/RBTransient/galerkin.jl`

This complements the broader transient algorithm suite and improves
confidence in projection consistency for transient reduced operators.

#### Test matrix expansion and refresh

The default test matrix (`test/runtests.jl`) now includes dedicated steady
and transient diagnostics testsets, a transient Galerkin testset, and an
updated moving-geometry suite focused on moving Poisson, elasticity, and
Stokes workflows.

### Maintenance

#### Dependency and compat updates

Compatibility bounds were revised for the current supported ecosystem. In
particular:

- `Gridap`: `0.19` -> `0.20`
- `GridapSolvers`: `0.6` -> `0.7`
- Added dependency: `MiniQhull = 0.4.0`

#### CI and release engineering updates

Project automation has been expanded with additional workflows for docs,
downgrade testing, and benchmarks, plus updates to CI/CompatHelper/TagBot
configuration. This improves regression detection and release reliability.

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
