# GridapROMs.jl Release Notes

## Release: v0.2.0

### Breaking changes

#### Weak forms no longer take measures as arguments

Residuals, Jacobians and right-hand sides are written as in Gridap, closing over
their measures: `a(μ,u,v) = ∫(...)dΩ` instead of `a(μ,u,v,dΩ) = ∫(...)dΩ`. The
triangulations on which each form is integrated are still passed to the operator
through `FEDomains`, which is what the per-triangulation hyper-reduction relies on.

#### Linear/nonlinear splitting with `Bundle`

`LinearNonlinearParamOperator`, `LinearNonlinearParamFEOperator` and their transient
counterparts have been removed. A problem split into a linear and a nonlinear part is
now built by passing `Bundle`s of residuals, Jacobians and domains to `ParamOperator`
(or `TransientParamOperator`):

```julia
res = Bundle(rhs_lin,res_nlin)       # rhs_lin(μ,v), res_nlin(μ,u,v)
jac = Bundle(jac_lin,jac_nlin)       # jac_lin(μ,du,v), jac_nlin(μ,u,du,v)
domains = Bundle(domains_lin,domains_nlin)
feop = ParamOperator(res,jac,pspace,trial,test,domains)
```

The linear part follows the conventions of `LinearParamOperator` (a right-hand side
and a bilinear form), the nonlinear part those of `ParamOperator`.

#### Snapshots, online solves and performance

- `solution_snapshots` returns the snapshots only, without the cost statistics.
- The online phase uses the same function: `solve(rbsolver,rbop,μ)` is replaced by
  `solution_snapshots(rbsolver,rbop,μ)`.
- Costs are recorded in the `RBPerformanceTracker` stored in the `RBSolver` (printed
  as they are recorded unless `verbose=false`), and `eval_performance(...)` is replaced
  by `rom_performance(rbsolver,rbop,x,x̂)`.
- `residual_snapshots`, `jacobian_snapshots`, `save_residuals`/`load_residuals`,
  `save_jacobians`/`load_jacobians` and `create_dir` have been removed.

#### Renames

| v0.1.x                             | v0.2.0                             |
|:-----------------------------------|:-----------------------------------|
| `MDEIMHyperReduction`, `:mdeim`    | `DEIMHyperReduction`, `:deim`      |
| `empirical_interpolation`, `s_opt` | `DEIM`, `SOPT`                     |
| `HighDimReduction`                 | `TransientReduction`               |
| `HighDimHyperReduction`            | `TransientHyperReduction`          |
| `ROMPerformance`                   | `RBPerformanceTracker`             |
| `RBOperator`, `TransientRBOperator` (exports) | `ROMOperator`, `TransientROMOperator` |

The `sketch` keyword (randomized compression) has been removed.

### New features

#### Distributed-memory ROMs

The whole offline and online pipeline now runs on distributed models
(`GridapDistributed`), with the same code as in serial:

- distributed snapshots (`DistributedSnapshots`) and row-partitioned bases
  (`GenericPArray`);
- POD by the method of snapshots, norm-weighted and Householder QR decompositions,
  distributed Gram solves and supremizer enrichment;
- DEIM and S-OPT with per-process reduced integration domains, distributed Galerkin
  projections and hyper-reduced assembly;
- space-time (Kronecker) ROMs, local ROMs and multi-field problems;
- per-process I/O: `save(dir,rbop)` and `load_operator(dir,feop[,ranks])`.

Random samplings (`:uniform`, `:normal`, `:latin_hypercube`) of `realisation(feop;...)`
now draw the same parameters on every process.

#### Other additions

- `EnergyNorm`: norms induced by user-defined bilinear forms.
- `ParamSpace` accepts bounds of mixed types, e.g. `(1,10,0.5,1.5)`.
- The Plots extension (`run_test`, `try_loading_*`) respects the verbosity of the
  `RBSolver`.

### Bug fixes

- Space-only reductions of linear transient problems (`SteadyReduction`): the
  residual snapshots are now computed at the states of the time-marching scheme,
  restoring the accuracy of the ROM.
- Distributed space-time projections with fewer temporal modes than time steps.
- Distributed save and load of local, block (multi-field) and transient reduced
  operators, which now also preserve the block structure of Jacobians.

### Documentation

- New Literate tutorials: parameterised PDEs, reduced order models (POD, TT-SVD,
  hyper-reduction strategies, save/load, multi-field and nonlinear problems, transient
  problems), local reduced operators, distributed-memory ROMs.
- Docstrings for every exported name, new README with an example gallery, and a new logo.

### Known issues

- `TTSVDReduction` cannot be combined with an `EnergyNorm`: use `L2()`, `H1()` (or
  another norm assembled as a rank tensor) instead.
- For `SteadyReduction`, only global DEIM and S-OPT hyper-reductions are currently
  supported.

## Release: v0.1.5

### Breaking changes

#### Energy norms rewritten as `AssembleOperator`s

The old norm-function/`norm_op`-based API (`get_norm()`) has been replaced by
a new `src/RB/RBSteady/EnergyNorms.jl` module built around a proper type
hierarchy:

- `abstract type AssembleOperator end`, with `NormStyle` (`ℓ2`/`EuclideanNorm`,
  `L2`, `H1`, `NitscheH1`) and `CouplingStyle` (`DivCoupling`) subtypes —
  all empty (or near-empty) structs, no longer function-valued.
- `BlockOperator`, aliased as `BlockNorm`/`BlockCoupling` for readability, for
  per-field composition in multi-field/saddle-point problems (e.g. Stokes,
  Navier-Stokes supremizer enrichment).
- `assemble_operator(a::AssembleOperator, feop)` is the single public
  entry point; `l2_norm`/`h1_norm`/`div_coupling` remain as direct helpers.

Reduction constructors (`PODReduction`, `TTSVDReduction`, ...) now take an
`AssembleOperator` (`norm`/`coupling` keywords) instead of a `norm_op`
function.

#### `TProduct` simplification

`TProductAssembly.jl`, `TProductCellData.jl`, `TProductGeometry.jl`, and the
`OrderedFESpace`/`OrderingMaps` machinery have been removed in favor of a
leaner `TProductFESpace`/`RankTensor` design (`Rank1Tensor`,
`GenericRankTensor`). `TProductFESpace` now accepts a
concrete, polytope-supplied `reffe` directly — the same one you would pass to
a plain `TestFESpace` — in addition to the existing tuple form, so no ad-hoc
reffe construction is required for `:ttsvd` runs.

### New features

#### Distributed (MPI) test suite

`test/Distributed/` now provides shared driver modules
(`PoissonDistributed.jl`, `StokesDistributed.jl`, each exposing a
`main(distribute, parts)` function) with two thin runners apiece:

- `test/Distributed/debug/` — runs under `PartitionedArrays.with_debug`
  (single-process simulated partitioning, no real MPI required) and is
  included by default from `test/runtests.jl`.
- `test/Distributed/mpi/` — runs under `PartitionedArrays.with_mpi` and must
  be launched through `mpiexecjl`, mirroring the pattern used by
  `GridapSolvers.jl`. The entry point is `test/runtests_mpi.jl`:

  ```
  mpiexecjl -n 4 julia --project=. test/runtests_mpi.jl
  ```

  which spawns each `Distributed/mpi/*.jl` file as its own `MPI.mpiexec()`
  subprocess.

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
`TransientLocalHyperReduction`, and their related local interpolation and
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

- `:deim`
- `:rbf`
- `:sopt`
- `:none`
- `:affine`

The strategies `:deim`, `:rbf`, and `:sopt` were already available in
previous releases. This release extends the interface with `:none` and
`:affine`.

In addition, `:none` accepts aliases `:no` and `:nohr`.

The steady and transient diagnostics tests validate these options by looping
over all strategies when constructing the RB solver:

- `RBSolver(...; hypred_strategy=:deim, ...)`
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
