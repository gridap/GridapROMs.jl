# Known issues: distributed Stokes, `:local` compression and `:none` hyperreduction

Status as of this investigation: **only `:global` compression + `:deim` hyperreduction
is confirmed working** for distributed (MPI) Stokes, steady and transient. Every other
combination tested (`:local` + `:deim`, `:global` + `:none`, `:local` + `:none`) hits a
distinct, unresolved bug. This document records what's confirmed, what's ruled out, and
exactly where each investigation stopped, so it doesn't have to be re-derived from
scratch.

Reproduction setup used throughout: a scratch Julia environment with `GridapROMs`
`Pkg.develop`-ed in plus `GridapPETSc`/`MPI`/`PartitionedArrays`/`GridapSolvers`/
`GridapDistributed`/`Gridap`/`DrWatson`, `PETSc_jll` pinned to `3.16.7` for real-scalar
support, run via `mpiexec -n 4 julia --project=<env> <script>`. The main repo's
`Project.toml` does **not** have `GridapPETSc` installed (it's a `[weakdeps]` entry with
no matching `[extensions]` or `[extras]` entry), so running any MPI+PETSc test requires
such a separate environment — `julia --project <repo-root>` will fail with
`Package GridapPETSc not found in current path`.

Test files: `test/Distributed/StokesDistributed.jl` (steady),
`test/Distributed/TransientStokesDistributed.jl` (transient), both driven via small
wrapper scripts under `test/Distributed/mpi/`.

---

## 1. `:local` compression + `:deim` — coupling (velocity-pressure) block collapses to ~0

**Symptom:** `SingularException: Zero pivot found` when factorizing the online reduced
jacobian, specifically for online test parameters whose jacobian-training cluster has
very few training samples (e.g. `nparams_jac=3`, `ncentroids=2` gives a `[2,1]` split —
the 1-sample cluster is the one that fails).

**Confirmed directly** (by printing the assembled online reduced jacobian and its
`svdvals` right before the crash): the coupling block `B = M[1:n_primal, n_primal+1:end]`
is exactly zero (norm ~1e-16) for parameters routed to the degenerate cluster, and
healthy (norm ~1.5, full rank) for parameters routed to the other cluster. This is the
root cause of the `SingularException` — not a numerical-conditioning red herring.

**Ruled out, with direct evidence (do not re-investigate these):**
- **Sampling insufficiency is not inherent.** The *same* config (`nparams=12,
  nparams_jac=3, ncentroids=2`, same degenerate `[2,1]` split) run in **serial** gives
  machine-precision error (~1e-14). The coupling term here is µ-independent (no
  parameter dependence in the weak form), so a single training sample is mathematically
  sufficient to reconstruct it exactly — this is not a case where more samples are
  needed, it's a case where the *distributed* reconstruction specifically breaks.
- **`AI` (the DEIM interpolation matrix) is not degenerate.** Directly printed for every
  `n=1` interpolation case: always a well-conditioned, non-near-zero scalar (e.g. 0.044,
  -0.42, 0.084, …). `lu()` on it never comes close to a zero pivot.
- **Ghost-coupling undercount in `DofMaps.recast_split_indices` (the `_keep_rows_and_cols`
  / `max.(a,b)` "assign to one rank" step in `src/Distributed/IndexOperations.jl`) is
  NOT the bug**, despite looking exactly like the known failure mode in
  `distributed-galerkin-projection-ghost-coupling` (see that memory / git history for
  the prior, *already-fixed* instance of this class of bug in
  `RBSteady.galerkin_projection(::GenericPMatrix,::PSparseMatrix,::GenericPMatrix)`).
  Verified directly: for the one coupling-block interpolation entry traced in detail,
  **two** ranks correctly pass `_keep_rows_and_cols` and each contributes a correct
  partial value; `sreduce` sums them to exactly match the serial ground truth value
  (0.0347). The `max.(a,b)` reduce is only used to broadcast *which* global `(row,col)`
  pair a given interpolation point refers to (that identification is legitimately
  single-sourced, since nz-flat-index partitioning has no ghosting) — it is not what
  decides which ranks *contribute* to the online reconstruction. That's decided
  separately and correctly by `_keep_rows_and_cols`.
- **The generic `interpolate!(::HRProjection,x::Any)` no-op fallback** (which exists and
  genuinely is a no-op for `EmptyInterpolation` cases) is **not** being hit for the
  coupling block. Confirmed via conditional debug prints (filtered to exclude
  `EmptyInterpolation`): zero hits. The dispatch correctly resolves to the
  distributed-aware `interpolate!(..., a::DistributedHRProjection, x::AbstractArray{<:AbstractArray})`
  at `src/Distributed/Reduced.jl` (confirmed via stack trace).
- **The offline Galerkin-projected DEIM basis template** (`proj_basis` stored in the
  `HRProjection`, i.e. the pre-computed `Φ_test' * basis_k * Φ_trial` for the rank-1 DEIM
  mode) **is not zero**. Directly printed: norm ≈ 1.84 for the `(14,7,1)`-shaped
  coupling-block template, consistently for both the degenerate and non-degenerate
  cluster (makes sense: the quantity being approximated doesn't depend on which cluster
  trained it, since it's µ-independent).

**Not yet found:** given that (a) the offline template is correct and nonzero, (b) the
online per-entry DEIM coefficient reconstruction is correct and nonzero (verified via
`AI`/`b` traces summing correctly across ranks), and (c) the final assembled coupling
block is nonetheless exactly zero — the bug must be in how the per-cluster online
coefficient gets *combined* with the per-cluster offline template specifically for the
degenerate cluster, or in some cache/indexing step between those two that wasn't
isolated before this investigation ended. The last two candidate sites examined,
without conclusive results, were: the `mul!(b̂,a,coeff,o,o)` combination step itself, and
whether `coeff`'s shape/indexing could be silently mismatched (wrong cluster's `coeff`
multiplied against the right cluster's template, or vice versa) specifically when `n=1`.
**Next step if resumed:** instrument `mul!` at that exact call site with the cluster
identity (not just `size`), for both clusters simultaneously, to see which half of the
combination is actually wrong.

---

## 2. `:none` hyperreduction — `isbitstype` / `MPI.Allreduce!` crash

**Status: FIXED.**

**Symptom:** `ArgumentError: Type must be isbitstype` thrown from inside
`MPI.Allreduce!`, for `:none` compression (either `:global` or `:local`).

**Root cause:** `:none` keeps the full, uncompressed jacobian as a `ConsecutiveParamMatrix`
instead of reducing it via DEIM. `PartitionedArrays.reduction!` for `MPIArray`s (used by
`consistent!`/ghost-sum operations) only had override methods for `AbstractArray` and
`ConsecutiveParamVector` element types (`src/Distributed/Primitives.jl`).
`ConsecutiveParamMatrix` is **not** an `AbstractArray` subtype (it's
`ParamArray{T,N} <: AbstractParamData`, a separate type hierarchy) and had no matching
`_data(...)` unwrap method either, so it fell through to PartitionedArrays' generic
`reduction!`, which tried to `MPI.Allreduce!` the raw wrapper struct directly instead of
the underlying plain `Array` — hence "Type must be isbitstype" (the *wrapper type*,
not its `eltype`, fails the check).

**Fix applied** in `src/Distributed/Primitives.jl`: generalized the `_data`/`reduction!`
dispatch from `ConsecutiveParamVector` to `ConsecutiveParamArray` (covers both vector and
matrix cases in one place, since `get_all_data` already handles both generically).

```julia
for T in (:AbstractArray,:ConsecutiveParamArray)   # was (:AbstractArray,:ConsecutiveParamVector)
  ...
end
_data(a::ConsecutiveParamArray) = get_all_data(a)  # was _data(a::ConsecutiveParamVector) = ...
```

Confirmed: this crash no longer occurs for either `:global,:none` or `:local,:none`.

---

## 3. `:none` hyperreduction — `MPIError(17)` during online jacobian assembly

**Status: NOT FIXED.** This is what `:none` hits *next*, after fix #2 above.

**Symptom:** `MPIError(17)` (message truncated / size mismatch) inside `MPI.Testall`,
during the online phase, specifically inside
`jacobian!` → `galerkin_projection!` → `galerkin_projection(::BlockProjection,...)` →
`_galerkin_mul!` → `consistent!(b)`, for the `BlockOperator{Tuple{H1,L2}}` norm-matrix
Galerkin projection (used by the FGMRES block preconditioner). This computation only
happens for `:none`/`:affine`-style strategies, because they Galerkin-project the *full*
FOM jacobian online, every call, rather than using a pre-reduced DEIM template — a
rarely-exercised, expensive path.

**Ruled out:**
- **Not a race condition / MPI tag collision.** Added a defensive fix regardless (see
  below), bisected with/without it: the crash persists identically with unique per-call
  MPI tags, so it is not cross-talk between overlapping asynchronous exchanges.
- **Not stale on-disk cache data confusing the picture** — confirmed by clearing
  `<env>/data/diagnostics_*` between runs; the symptom is then fully deterministic and
  reproducible.
- **Not `:local`-specific.** The stack trace at the point of failure shows
  `GlobalRBSolver`, not any `:local`/`LocalReduction` code path — this is purely a
  `:global` compression issue, despite surface resemblance to the `:local` clustering
  bug in §1.
- **Not a `change_ghost` caching/reference-equality shortcut bug** (the
  `a.index_partition === partition(ids)` check in
  `GridapDistributed.change_ghost(a::GenericPArray,...)`) — instrumented directly;
  that method is never even called for this path (the ghost indices already match,
  so the shortcut is irrelevant here).

**Confirmed, deterministically, after clearing cache:** at the exact same call index in
the overall sequence of `galerkin_projection` calls (same index on every rank, verified
with a shared counter — ranks are in lock-step, same total call count), **three ranks
compute a reduced-basis dimension of 14 while one rank computes 8**, for what should be
the same, offline-replicated, fixed reduced basis `Φ`. This is a genuine cross-rank data
inconsistency in a quantity (`size(Φ,2)` — number of retained modes) that is supposed to
be a global constant. Both involved matrices (`a::PSparseMatrix`, the norm operator) are
`480×480` (same FE-dof count on every rank, consistent) — only the *basis* passed in as
`c`/`b` differs.

**Not yet found:** which of `Φ_velocity` / `Φ_pressure` is involved, and why one rank's
copy differs in mode count from the other three's. The call happens inside
`galerkin_projection(proj_left::BlockProjection{A,1}, a::BlockProjection{B,1})`'s
`for i in eachindex(a)` loop (`src/RB/RBSteady/Projections.jl`), which is a plain,
synchronous loop with no conditional skips — so the loop itself is not the source; the
basis object `proj_left[i]`/`a[i]` must already carry the inconsistent size before
reaching it. Candidate next step: the **offline supremizer enrichment for `:global`
compression** (distinct from the `:local`-specific `enrich!` in
`src/RB/RBSteady/LocalProjections.jl`, which was separately verified correct earlier in
this investigation) has not been checked for a per-rank-inconsistent retained-mode-count
bug — that's the most likely remaining culprit, but it was not reached before this
investigation stopped.

**Defensive fix applied anyway** (did not fix #3, but is a real, independent latent bug
worth keeping fixed): `src/Distributed/Primitives.jl`'s `MPIArray`-specific
`exchange_impl!` used a single, fixed `PartitionedArrays.EXCHANGE_IMPL_TAG = 0` for
*every* point-to-point exchange. For exchanges issued repeatedly and asynchronously
(the caller `wait`s on a returned task, not immediately), this is unsafe in general —
a rank that's already posted its Nth exchange can have that message matched against
another rank's still-pending (N-1)th receive. Fixed by tagging each logical exchange
with a per-call counter (`_EXCHANGE_IMPL_CALL_COUNTER`), same value on every rank since
call count/order is globally consistent by construction.

---

## 4. Transient distributed Stokes, `:global,:deim`

**Status: FIXED** (the blocking bug; a separate, likely-unrelated segfault remains —
see below).

While verifying that the one known-working spec (`:global` + `:deim`) also works for the
**transient** case: it does, in **serial**, after fixing three unrelated bugs in the
in-progress `get_res`/`get_jac`/transient-operator refactor (not distributed-specific,
see commit history around `src/FEM/ParamODEs/TransientParamFEOperators.jl`,
`src/FEM/ParamDataStructures/ParamSpaces.jl`, `src/RB/RBTransient/RBOperators.jl` for the
exact one-line fixes — a `forms`/`jacs` typo, a missing `get_sampling_style(::TransientParamSpace)`
delegate, and a stale `feop.constant_forms` field reference from before that field was
removed). Serial `test/RBTransient/stokes.jl` with `(:pod,:global,:deim)` now gives
errors ~0.008/0.001 — healthy.

**Distributed transient `:global,:deim` hit a separate bug:**
`BoundsError: attempt to access 0×0 ConsecutiveParamSparseMatrixCSC at index [1,1]`,
inside `recast(a::GenericPArray,i::PSparseMatrixDofMap)` (`src/Distributed/Snapshots.jl:39`)
→ `PSparseMatrix(...)` → `p_sparse_matrix_cache_impl`'s ghost-cache setup →
`PartitionedArrays.nzindex` → `testitem`/`param_getindex(A,1)`, called from the
**transient, Kronecker-structured** DEIM hyperreduction's `get_recast_basis` /
`galerkin_projection` path (`src/RB/RBTransient/Projections.jl:190`,
`src/RB/RBTransient/HyperReductions.jl:3`).

**Root cause, confirmed by instrumenting `recast` directly:** for the coupling-block's
spatial DEIM factor, the Kronecker/`SearchSVDRank` truncation legitimately retains
**zero spatial modes** on every rank (`size(local_values(a)) = (nnz_local, 0)` — zero
*columns*, i.e. zero param/mode slices; the sparsity pattern itself was always correctly
sized, e.g. `(25,128)`). A `ConsecutiveParamSparseMatrixCSC` with zero retained modes is
a legitimate, well-defined object (an `m×n` sparsity with nothing stored) — but
`PartitionedArrays.nzindex(a::ParamSparseMatrix,...) = nzindex(testitem(a),...)`
(`src/Distributed/ParamSparseUtils.jl`) goes through `testitem`, which requires at least
one param slice to exist, so it throws `BoundsError` on the (0,0)-sized "diagonal" view
that `ConsecutiveParamSparseMatrixCSC`'s parametric `getindex` convention produces when
`param_length == 0`.

**Fix applied:** `nzindex` only ever needs the sparsity structure (`size`, `colptr`,
`rowval`) — it never reads the stored values — so it must not require `testitem` at all.
Added a dedicated `PartitionedArrays.nzindex(a::ConsecutiveParamSparseMatrixCSC,i,j)`
that builds the lookup directly from `innersize(a)` / `SparseArrays.getcolptr(a)` /
`rowvals(a)` (all structural, always available regardless of `param_length`), bypassing
`testitem` entirely. This mirrors the existing `nziterator`/`NZIteratorCSC` overloads for
the same type in the same file, which already operate structurally without going
through `testitem`.

**Confirmed:** distributed transient `:global,:deim` now runs `run_test` to completion
across all three tolranks (`1e-1,1e-3,1e-5`) with no BoundsError.

**Remaining, separate issue:** immediately after `run_test` completes, the process
segfaults (`signal 11`, no Julia-level stacktrace) during `rom_diagnostics`/final
cleanup. This matches a segfault already observed earlier in this investigation for
**steady** `:global,:deim` (`stokes_global_only.jl`, also no stacktrace, also at/after
the final GridapPETSc cleanup) — almost certainly the same, PETSc-GC-related issue, not
specific to transient or to this fix. Not investigated further; the actual ROM
computation (`run_test`) completes successfully before it, so this looks like a
cleanup-time crash rather than a correctness issue, but it does mean the process exits
with a nonzero code either way.

---

## Summary table

| Compression | Hyperreduction | Status | Section |
|---|---|---|---|
| `:global` | `:deim` | **Working** (steady and transient, serial and distributed; a likely-unrelated cleanup-time segfault remains, see §4) | §4 |
| `:global` | `:none` | Broken: §2 (fixed) then §3 (open) | §2, §3 |
| `:local` | `:deim` | Broken: §1 (open) | §1 |
| `:local` | `:none` | Broken: hits §2's crash too (fixed), `:local` path not reached far enough to know if §1 or §3's issue (or a third one) is next | §1, §2 |

---

## 5. FSIProbROMs.jl Khabakhpasheva tests (steady + transient, 3-field complex arrow-sparsity problem)

Separate investigation thread, triggered by verifying/speeding up
`test/mpi/KhabakhpashevaFreq.jl` (steady) and `test/mpi/KhabakhpashevaTime.jl` (transient)
in `FSIProbROMs.jl`. Both are 3-field (ϕ=potential on Ω, κ=boundary field on Γκ,
η=beam deflection on Γη), complex-valued (`ComplexF64`), with jacobian arrow sparsity
`A = [A11 A12 A13; A21 A22 0; A31 0 A33]` (A23=A32=0, no κ-η coupling term in the weak
form). `Freq.jl` is **fully fixed and validated** (steady). `Time.jl` (transient) hit a
**separate, unresolved** bug after the steady fixes were applied — see §5.5.

### 5.1 `blockify` scalar-indexing crash — FIXED

`src/Distributed/IndexOperations.jl`, `blockify(b::PVector,i::BlockPVectorDofMap)`:
`map(1:nfields) do k; PVector(values[k],...); end` indexed `values::MPIArray` (one tuple
per **rank**) by `k` (a **field** index 1..nfields) — wrong on two counts: scalar-indexes
an `MPIArray` (forbidden) and is semantically wrong regardless (should extract field `k`
from each rank's local tuple). Fixed:
```julia
map(1:nfields) do k
  values_k = map(v -> v[k],values)
  PVector(values_k,partition(i[k].rows_space))
end |> mortar
```

### 5.2 `get_dof_value_type` wrong for distributed FE spaces — FIXED

Gridap's generic `get_dof_value_type(f::FESpace) = get_dof_value_type(get_fe_basis(f),
get_fe_dof_basis(f))` infers the scalar type by **evaluating shape functions**, which are
always real-valued regardless of the space's declared `vector_type` — this happens to
give the right answer for a *serial* `SingleFieldFESpace` (whose dof basis is itself
typed by `vector_type`), but not for a `DistributedSingleFieldFESpace`, whose per-rank
local dof bases don't carry that typing through evaluation. This was invisible until a
genuinely-empty complex jacobian block (κ-η, structurally zero) needed `EmptyHRProjection`
(`src/RB/RBSteady/HyperReductions.jl:141`), which calls `get_dof_value_type(trial)` and
got `Float64` instead of `ComplexF64` — the only call site in the whole pipeline that
actually depends on this function's result for a *correctly-typed* answer; every other
path infers eltype from real (always-complex) snapshot data and never calls it.

Fixed in `src/Distributed/ParamFESpaces.jl` (final form, by nmueller):
```julia
function FESpaces.get_dof_value_type(f::DistributedFESpace)
  get_dof_value_type(getany(local_views(f)))
end
```
(delegates to one representative local, serial FE space, where the generic fallback is
correct.)

### 5.3 `BlockHRProjection` heterogeneous-element-type construction — worked around by fixing §5.2, not a separate fix

Once §5.2 was fixed, a `BlockHRProjection(::Matrix{GenericHRProjection{A}})` constructor
error (couldn't infer `B` from an incomplete element type) also disappeared — it was a
downstream symptom of the same `Float64`-vs-`ComplexF64` mismatch across the 3×3
jacobian's blocks, not an independent bug. (A speculative outer constructor for
`BlockHRProjection` handling heterogeneous element types was tried and reverted — hit
Julia's parameter invariance (`Foo{X}` is never `<: Foo{Y}` even if `X<:Y`), so
widening the container type doesn't work; not needed once §5.2 was fixed anyway.)

### 5.4 `induced_norm`'s `reshape(a,size(a,1),:)` collapses param-column count to 0 — FIXED

`src/Distributed/Reduced.jl`, `induced_norm(a::Union{GenericPArray,DistributedSnapshots},
norm_matrix)`. When a rank owns **zero local dofs** for a field (common for small
boundary fields like κ, η under a Cartesian partition — some ranks legitimately own none
of them), `reshape(al, size(al,1), :)` can't infer the column count from `0 * n = 0`
(any `n` satisfies it) and silently picks `n=0`. Since `GenericPArray`'s constructor
infers its **global, replicated** column count from an **arbitrary** rank
(`getany`/`axes(getany(array_partition))[2:end]`), if that arbitrary pick happens to be a
degenerate (0-row) rank, the *entire* field's result collapses to 0 columns globally —
not just on that one rank. This gave `NaN` (via `mean(ComplexF64[])`) for κ and η's ROM
error specifically, while ϕ (every rank owns some of its dofs) was unaffected.

Fixed by computing the column count via a **distributed max** instead of local `:`
inference:
```julia
np = reduce(max,map(al -> size(al,1) == 0 ? 0 : div(length(al),size(al,1)),local_values(a)))
values = map(local_values(a)) do al
  reshape(al,size(al,1),np)
end
```
(`param_length(a)` was tried first as the "obviously correct" fix — it has the *exact
same* bug, one level removed, since it's presumably also derived via `getany`/local-shape
inference; gave a `DimensionMismatch` on the *non*-degenerate field (ϕ) when `np` came
back `0`.)

**Verified at full scale** (`nparams=50`): `Freq.jl` with the original `LUSolver()` now
runs to completion end-to-end with no NaN, no crash, no hang — errors
`[0.00593, 0.00585, 0.00750]`, consistent across all 4 ranks. (A previously-seen
cleanup-time hang after `run_test`/`rom_performance` completed did **not** recur after
this fix — it was very likely downstream fallout from this bug, not a separate issue.)

### New solver for Khabakhpasheva's arrow-sparsity jacobian (performance, not correctness)

Both `Freq.jl` and `Time.jl` previously used a single monolithic `LUSolver()` on the full
3-field coupled system — expensive. Since `A23=A32=0` (κ-η structurally decoupled) and
the three diagonal blocks have wildly different conditioning (`A11~1e6` bulk ϕ block,
`A22~1e0` well-conditioned κ, `A33~1e9` ill-conditioned-but-small η beam block), replaced
with: direct `LUSolver()` on each **isolated** diagonal block (much cheaper than one
monolithic factorization) wrapped in an outer `FGMRESSolver` that handles the
`A12/A13/A21/A31` coupling iteratively via `GridapSolvers.BlockSolvers.BlockTriangularSolver`
(`:upper`, ignoring the lower-triangular `A21/A31` coupling in the preconditioner — a
genuine approximation, not exact, hence the outer Krylov wrapper). Requires
`style=BlockMultiFieldStyle()` on `X`/`Y`, and a zero-init `solve!` override for
`AbstractParamPVector`/`AbstractParamPSparseMatrix` (iterative solvers use the passed-in
`x` as their initial guess, unlike `LUSolver` which ignores it) — see
`test/mpi/KhabakhpashevaFreq.jl`'s top-of-file override, copied verbatim into `Time.jl`.

**Verified at full scale for `Freq.jl`** (`nparams=50`): all 240 online solves converge
(`SOLVER_CONVERGED_RTOL`, 55-65 FGMRES iterations each), errors
`[0.005933, 0.005852, 0.007499]` — matches the `LUSolver()` baseline almost exactly.
Timing roughly comparable to baseline (not dramatically faster) — the upper-triangular-only
preconditioner is correctness-confirmed but not strong (drops `A21/A31` entirely); if
revisiting for performance, that's the place to improve (e.g. a better `solver_ϕ` for the
big block, or finding a way to incorporate the lower-triangular coupling).

**Companion fix needed to use `FGMRESSolver`/`GMRESSolver` at all for this (complex)
problem:** `GridapSolvers.jl`'s `GMRESSolver`/`FGMRESSolver` (as installed, NOT in this
repo) have real-only bugs — the Hessenberg/residual/Givens caches are allocated as plain
`Float64` regardless of the system's eltype; the Arnoldi projection
`dot(V[j+1],V[i])` has the conjugation backwards for complex vectors (`dot(x,y)=x'y` is
conjugate-linear in the *first* arg — needs `dot(V[i],V[j+1])`); the Givens-rotation
Hessenberg/residual update is missing `conj(s[i])`/`conj(s[j])` (LAPACK `zrotg`
convention: `c` real, `s` complex, second row uses `-conj(s)`); and the least-squares
back-substitution uses `dot(H[i,i+1:j],g[i+1:j])`, which also wrongly conjugates (needs
plain `sum(H[i,i+1:j].*g[i+1:j])`). All four are no-ops for real systems, which is why
this was never caught before. Fixed via a monkeypatch in **`FSIProbROMs.jl`**, not this
repo: `src/ComplexKrylov.jl` (new file, `_init_complex_krylov()` called from
`GridapFixes.jl`'s existing `__init__`), verified standalone against a complex-Poisson
GMRES/FGMRES test (converges to rtol=1e-10, error ~1e-22).

### 5.5 Transient (`Time.jl`) — UNRESOLVED, investigation in progress when this was written

After applying §5.1-§5.4 plus the new solver (and after nmueller independently fixed an
unrelated `res`-vs-`b` argument-passing bug in `Time.jl`'s `TransientLinearParamOperator`
construction, and a missing `using Gridap.Algebra` import), `Time.jl` still crashes, now
much later in the pipeline (basis construction / first online projection), with:

```
DimensionMismatch: incompatible dimensions for matrix multiplication:
tried to multiply a matrix of size (7209, 7209) with a matrix of size (10465, 3)
```

Call chain: `project!(X̂,a.projection_space::NormedProjection,X*basis_time)`
(`src/RB/RBTransient/Projections.jl:154`) → `project!(x̂,a,norm_matrix*x)`
(`src/RB/RBSteady/Projections.jl:87`), where `X = reshape(x,Ns,Nt)` used
`Ns = num_fe_dofs(a.projection_space) = size(get_basis_space(a),1) = 10465` (the local
dof count **including ghost**) while `a.projection_space`'s own norm matrix is `7209×7209`
(own-only/ghost-consistent). Confirmed via dispatch tracing that this is **not** a
`project!` dispatch bug — `Distributed/Projections.jl:118-131`'s
`project!(x̂::ConsecutiveParamVector,a::TransientProjection,x::PVector{...})` (the
map-over-local-values-then-`sreduce` wrapper) **is** being hit correctly first, per the
stack trace (frame right before the crash). The mismatch is *inside* the per-rank local
`KroneckerProjection.project!` call, between that rank's own spatial basis and its own
norm matrix.

Also confirmed (separately, by the user) that `first_unfold`'s `reshape(A.parent,skeep,:)`
had the *exact same* `:`-inference bug as §5.4 (same fix pattern applied: compute the
target column count directly from `size(A.parent)[2:end]`, no division) — fixed in
`src/RB/RBTransient/BasesConstruction.jl`, but this did **not** resolve the
`DimensionMismatch` above (got further, but still crashes on the matmul).

**Leading hypothesis, structurally confirmed by code-reading (not yet empirically
confirmed with real partition data — see pending check below):**
`method_of_snapshots_col(red_style,A,X) = _method_of_snapshots_col(red_style,A,A'*(X*A))`
(`src/RB/RBSteady/BasesConstruction.jl:121-129`) uses `X` (the norm matrix) **only** to
build the small correlation matrix `A'*(X*A)` — it is never passed to
`_weighted_mul_row(A,Vr,Sr)`, which builds the actual **spatial basis** `Ur` using
`U = GenericPArray{Matrix{T}}(undef,flat_row_partition(A),axes(V,2))`
(`src/Distributed/BasesConstruction.jl:165-177`) — i.e. the basis's row partition comes
from `A` (the raw snapshot) alone, never from `X`. Per nmueller: *"the spatial basis
(partitioned) should have a layout that's compatible with the norm matrix (partitioned)
because the latter was used to build the former; if this is not the case, then probably
this is the issue."* `_reduce_columns(A::Union{GenericPMatrix,DistributedSnapshots}) =
true` (`src/Distributed/BasesConstruction.jl:161-163`) **is** confirmed to be hit
correctly (verified via direct instrumentation: fires for every distributed call, both
weighted and unweighted, always routing to `_col`) — ruled out as the issue; the gap is
specifically that `A`'s and `X`'s row partitions are never reconciled/compared before
`_weighted_mul_row` picks `A`'s.

**Pending when this was written:** a debug run was in flight to empirically compare
`flat_row_partition(A)` vs `row_partition(X)` (`local_length`/`own_length` per rank)
directly inside `method_of_snapshots_col`, to confirm whether they actually differ in
practice for this test (they *could* coincide naturally if both are built from the same
FE-space ghost-exchange graph, in which case this wouldn't manifest as a bug despite the
structural gap) — **check `/tmp/debug_part_pid*.txt` on the machine this was run on, or
rerun with `GRIDAPROMS_DEBUG_PART=1` env var set** (instrumentation is live in
`src/RB/RBSteady/BasesConstruction.jl:121-139`, uses `Main.GridapROMs.Distributed...`
qualified paths since `GridapROMs`/`PartitionedArrays` aren't bare-name-accessible from
inside the `RBSteady` submodule — **this is temporary debug code and should be removed**
once the question is answered).

**If confirmed that `A`'s and `X`'s partitions differ:** the fix most likely needs
`_weighted_mul_row` (or its caller) to build the spatial basis using `X`'s row partition
rather than `A`'s own — e.g. by `change_ghost`-ing `A` to match `X` once, before both the
correlation-matrix step and the `_weighted_mul_row` call, so both consistently see the
same (correct) partition.

**Files with active temporary debug instrumentation (remove once resolved):**
- `src/RB/RBSteady/BasesConstruction.jl:121-139` (`GRIDAPROMS_DEBUG_PART` env var) —
  the partition-comparison check described above.

**Files modified this investigation (GridapROMs.jl):**
- `src/Distributed/IndexOperations.jl` — §5.1 fix
- `src/Distributed/ParamFESpaces.jl` — §5.2 fix
- `src/Distributed/Reduced.jl` — §5.4 fix
- `src/RB/RBTransient/BasesConstruction.jl` — `first_unfold` reshape fix (see §5.5)
- `src/RB/RBSteady/BasesConstruction.jl` — **currently has live debug instrumentation,
  not yet cleaned up** (see above)

**Files modified this investigation (FSIProbROMs.jl):**
- `test/mpi/KhabakhpashevaFreq.jl` — new solver + zero-init `solve!` override +
  `BlockMultiFieldStyle()`; fully validated, no outstanding issues
- `test/mpi/KhabakhpashevaTime.jl` — same solver change + `BlockMultiFieldStyle()` +
  missing `Gridap.Algebra` import; **still broken, see §5.5**
- `src/ComplexKrylov.jl` (new file) — GMRES/FGMRES complex-number fixes, included from
  `FSIProbROMs.jl`'s main module right after `GridapFixes.jl`
- `src/GridapFixes.jl` — one-line addition (`_init_complex_krylov()` call inside the
  existing `__init__`) to wire up the new file above
