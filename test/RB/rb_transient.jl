module RBTransientTests

using Test
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.Arrays: Table
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.RBTransient

# Pull in non-exported symbols used throughout the tests
const time_combinations         = RBTransient.time_combinations
const KroneckerProjection       = RBTransient.KroneckerProjection
const SequentialProjection      = RBTransient.SequentialProjection
const KroneckerDomain           = RBTransient.KroneckerDomain
const SequentialDomain          = RBTransient.SequentialDomain
const time_enrichment           = RBTransient.time_enrichment
const get_combine               = RBTransient.get_combine
const get_domain_style          = RBTransient.get_domain_style
const get_indices_time          = RBTransient.get_indices_time
const get_integration_domain_space = RBTransient.get_integration_domain_space
const _get_shift                = RBTransient._get_shift
const _to_realisation           = RBTransient._to_realisation

# ─── Helpers ──────────────────────────────────────────────────────────────────

# Build a small TransientRealisation: np parameter samples, nt time steps
# TransientRealisation(params, times, t0) stores t0 separately and times = [t1,...,tN]
function _make_realisation(np=3, nt=4)
  params = Realisation([[i*0.1, i*0.2] for i in 1:np])
  times  = range(0.25, 1.0; length=nt) |> collect
  TransientRealisation(params, times, 0.0)
end

# ─── HighDimReduction constructors ────────────────────────────────────────────

@testset "KroneckerReduction via HighDimReduction — ReductionStyle vector" begin
  # AbstractVector{<:ReductionStyle} dispatches to KroneckerReduction
  red = HighDimReduction([SearchSVDRank(1e-2), SearchSVDRank(1e-2)])
  @test red isa KroneckerReduction
  @test length(red.reductions) == 2
end

@testset "KroneckerReduction from scalar tol + dim" begin
  # Scalar tolerance broadcasts to `dim` independent reductions → KroneckerReduction
  red = HighDimReduction(1e-2; dim=2)
  @test red isa KroneckerReduction
  @test num_params(red) == num_params(first(red.reductions))
end

@testset "SequentialReduction via HighDimReduction — Float64 vector" begin
  # Vector{Float64} dispatches to SequentialReduction (TT-SVD with tolerance-per-mode)
  seq = HighDimReduction([1e-2, 1e-2])
  @test seq isa SequentialReduction
end

@testset "SequentialReduction via HighDimReduction — TTSVDRanks" begin
  # TTSVDRanks reduction style also selects the TT-SVD path
  seq = HighDimReduction(TTSVDRanks([SearchSVDRank(1e-2), SearchSVDRank(1e-2)]))
  @test seq isa SequentialReduction
end

@testset "SteadyReduction wraps a steady reduction" begin
  red = SteadyReduction(SearchSVDRank(1e-3))
  @test red isa SteadyReduction
  @test ReductionStyle(red) isa SearchSVDRank
end

# ─── time_combinations ────────────────────────────────────────────────────────

@testset "time_combinations ThetaMethod θ=1 (implicit Euler)" begin
  # With θ=1, combine_jac(x,y) = 1·x + 0·y = x
  dt = 0.1
  fesolver = ThetaMethod(LUSolver(), dt, 1.0)
  combine_res, combine_jac, combine_djac = time_combinations(fesolver)

  @test combine_res isa Function
  @test combine_jac isa Function
  @test combine_djac isa Function

  x = rand(4, 3); y = rand(4, 3)
  @test combine_jac(x, y) ≈ x
  @test combine_djac(x, y) ≈ (x .- y) / dt
  @test combine_res(x) === x        # identity — no copy
end

@testset "time_combinations ThetaMethod θ=0.5 (Crank-Nicolson)" begin
  dt = 0.2
  fesolver = ThetaMethod(LUSolver(), dt, 0.5)
  _, combine_jac, combine_djac = time_combinations(fesolver)

  x = rand(3); y = rand(3)
  @test combine_jac(x, y)  ≈ 0.5 .* x .+ 0.5 .* y
  @test combine_djac(x, y) ≈ (x .- y) / dt
end

# ─── HighDimHyperReduction constructors ───────────────────────────────────────

@testset "HighDimMDEIMHyperReduction construction (explicit)" begin
  _, cjac, _ = time_combinations(ThetaMethod(LUSolver(), 0.1, 0.5))
  red = HighDimReduction(1e-2; dim=2)
  hr  = HighDimMDEIMHyperReduction(red, cjac)
  @test hr isa HighDimMDEIMHyperReduction
  @test get_combine(hr) === cjac
end

@testset "HighDimHyperReduction dispatch :mdeim" begin
  _, cjac, _ = time_combinations(ThetaMethod(LUSolver(), 0.1, 0.5))
  hr = HighDimHyperReduction(cjac, 1e-2; dim=2, hypred_strategy=:mdeim)
  @test hr isa HighDimMDEIMHyperReduction
  @test get_combine(hr) === cjac
end

@testset "HighDimHyperReduction dispatch :sopt" begin
  _, cjac, _ = time_combinations(ThetaMethod(LUSolver(), 0.1, 0.5))
  hr = HighDimHyperReduction(cjac, 1e-2; dim=2, hypred_strategy=:sopt)
  @test hr isa HighDimSOPTHyperReduction
  @test get_combine(hr) === cjac
end

# ─── tucker ───────────────────────────────────────────────────────────────────

@testset "tucker on 3D array — SearchSVDRank" begin
  # Array shape: Ns × Nt × Np.  The plain-array `tucker` performs a hierarchical
  # two-mode unfolding (no change_mode step):
  #   Mode 1 unfold: (Ns, Nt*Np) → Φs is (Ns, k1), remainder is (k1, Nt*Np)
  #   Mode 2 unfold: (k1, Nt*Np) → Φt is (k1, k2), remainder is (k2, Nt*Np)
  Ns, Nt, Np = 20, 8, 15
  A = rand(Ns, Nt, Np)
  reds = [Reduction(SearchSVDRank(1e-2)), Reduction(SearchSVDRank(1e-2))]
  bases = tucker(reds, A)

  @test length(bases) == 2                         # N-1 = 2 bases
  Φs, Φt = bases
  @test size(Φs, 1) == Ns                          # spatial rows = Ns
  @test size(Φs, 2) <= Ns                          # compressed rank
  @test norm(Φs' * Φs - I(size(Φs,2))) < 1e-10    # orthonormal columns
  # Second basis: rows = k1 (first basis rank), not Nt
  @test size(Φt, 1) == size(Φs, 2)
  @test norm(Φt' * Φt - I(size(Φt,2))) < 1e-10
end

@testset "tucker on 3D array — FixedSVDRank" begin
  # With FixedSVDRank: bases[1] is (Ns, ks), bases[2] is (ks, kt)
  Ns, Nt, Np = 30, 10, 12
  A = rand(Ns, Nt, Np)
  ks, kt = 4, 3
  reds = [Reduction(FixedSVDRank(ks)), Reduction(FixedSVDRank(kt))]
  bases = tucker(reds, A)
  @test size(bases[1], 1) == Ns
  @test size(bases[1], 2) == ks
  @test size(bases[2], 1) == ks    # second basis rows = k1
  @test size(bases[2], 2) == kt
end

# ─── time_enrichment ──────────────────────────────────────────────────────────

@testset "time_enrichment enriches when dual is orthogonal to primal" begin
  # Enrichment is triggered when the dual vector has small projection onto the
  # current primal span.  Using canonical basis vectors guarantees exact
  # orthogonality: primal = e1..e5, dual = e6..e8 → basis_primal'*basis_dual = 0
  # → dist = 0 ≤ tol → all 3 dual directions are added, giving 5+3=8 columns.
  nt = 10
  Φ_primal = Matrix(I(nt))[:, 1:5]   # nt×5 canonical
  Φ_dual   = Matrix(I(nt))[:, 6:8]   # nt×3 canonical, orthogonal to primal
  result = time_enrichment(PODProjection(Φ_primal), Φ_dual; tol=1e-4)
  @test get_basis(result) isa AbstractMatrix
  @test size(get_basis(result), 1) == nt
  @test size(get_basis(result), 2) == 8    # 5 primal + 3 dual
end

@testset "time_enrichment no-op when dual already in span" begin
  # Dual is a subset of the primal columns → basis_primal'*basis_dual[:,i] is a
  # unit vector → dist >> tol → no enrichment
  nt = 8
  Φ = Matrix(qr(rand(nt, 4)).Q)
  result = time_enrichment(PODProjection(Φ), Φ[:, 1:2]; tol=1e-10)
  @test size(get_basis(result), 2) == 4   # column count unchanged
end

# ─── KroneckerProjection ──────────────────────────────────────────────────────

@testset "KroneckerProjection — field accessors" begin
  Ns, Nt, ns, nt = 20, 8, 4, 3
  proj_s = PODProjection(Matrix(qr(rand(Ns, ns)).Q))
  proj_t = PODProjection(Matrix(qr(rand(Nt, nt)).Q))
  kp = KroneckerProjection(proj_s, proj_t)

  @test num_reduced_dofs(kp) == ns * nt
  @test num_space_dofs(kp)   == Ns
  @test num_times(kp)        == Nt
end

@testset "KroneckerProjection — project!/inv_project! idempotency" begin
  # After projecting then reconstructing, a second projection must give the same x̂.
  # (P is a projection operator, so P·(P^{-1}·(P·x)) = P·x)
  Ns, Nt, ns, nt = 20, 8, 4, 3
  Φs = Matrix(qr(rand(Ns, ns)).Q)
  Φt = Matrix(qr(rand(Nt, nt)).Q)
  kp = KroneckerProjection(PODProjection(Φs), PODProjection(Φt))

  x  = rand(Ns * Nt)
  x̂  = zeros(ns * nt)
  project!(x̂, kp, x)

  x_rec = zeros(Ns * Nt)
  inv_project!(x_rec, kp, x̂)

  x̂2 = zeros(ns * nt)
  project!(x̂2, kp, x_rec)
  @test x̂2 ≈ x̂ atol=1e-10
end

# ─── galerkin_projection for transient ────────────────────────────────────────

@testset "galerkin_projection — matrix×matrix×matrix with combine" begin
  # All three matrices are Nt × k_i; the inner loop contracts over Nt time points.
  # Result shape: (nleft, n_basis, nright).
  Nt = 12; nleft = 4; n = 6; nright = 5
  Φl = Matrix(qr(rand(Nt, nleft)).Q)
  A  = rand(Nt, n)
  Φr = Matrix(qr(rand(Nt, nright)).Q)

  θ = 0.5
  combine(x, y) = θ .* x .+ (1 - θ) .* y

  proj = galerkin_projection(Φl, A, Φr, combine)
  @test size(proj) == (nleft, n, nright)
end

@testset "galerkin_projection — combine respects θ=1 (implicit)" begin
  # With θ=1, combine(x,y)=x, so result equals the standard Galerkin projection
  # Φl' * (A*Φr_scaled) contracted over time — checked via the identity limit.
  Nt = 10; n = 3; nl = 2; nr = 4
  Φl = Matrix(qr(rand(Nt, nl)).Q)
  A  = rand(Nt, n)
  Φr = Matrix(qr(rand(Nt, nr)).Q)

  proj1 = galerkin_projection(Φl, A, Φr, (x, y) -> x)  # θ=1 → take x
  proj0 = galerkin_projection(Φl, A, Φr, (x, y) -> y)  # θ=0 → take y (shifted)

  # The two results should generally differ (shifted vs current time step)
  @test size(proj1) == (nl, n, nr)
  @test size(proj0) == (nl, n, nr)
  # With a generic random A, the shifted and unshifted projections are different
  @test !(proj1 ≈ proj0)
end

# ─── TransientIntegrationDomain ───────────────────────────────────────────────

@testset "TransientIntegrationDomain — struct construction and accessors" begin
  # Build a minimal GenericDomain as the space-only sub-domain.
  # GenericDomain requires cells::Vector{Int32}, cell_idofs::Table, metadata.
  # Table(data, ptrs): ptrs[i]:ptrs[i+1]-1 gives row i entries.
  cells        = Int32[1, 2, 3]
  # 3 cells, 1 dof each: data=[1,2,3], ptrs=[1,2,3,4]
  cell_idofs   = Table(Int32[1, 2, 3], Int32[1, 2, 3, 4])
  rows         = Int32[1, 2, 3]
  domain_space = GenericDomain(cells, cell_idofs, rows)

  indices_time = Int32[2, 4, 6]
  tid = TransientIntegrationDomain(KroneckerDomain(), domain_space, indices_time)

  @test get_domain_style(tid) isa KroneckerDomain
  @test get_indices_time(tid) == indices_time
  @test get_integration_domain_space(tid) === domain_space
  @test get_integration_cells(tid) === cells
end

@testset "TransientIntegrationDomain — SequentialDomain variant" begin
  cells        = Int32[5, 7]
  cell_idofs   = Table(Int32[10, 11, 12], Int32[1, 3, 4])
  rows         = Int32[10, 11, 12]
  domain_space = GenericDomain(cells, cell_idofs, rows)

  indices_time = Int32[1, 3]
  tid = TransientIntegrationDomain(SequentialDomain(), domain_space, indices_time)

  @test get_domain_style(tid) isa SequentialDomain
  @test get_indices_time(tid) == indices_time
end

# ─── TransientRBSolver construction ───────────────────────────────────────────

@testset "RBSolver(ODESolver, Reduction) builds a TransientRBSolver" begin
  # RBSolver dispatches on ODESolver to build hyper-reduction for each Jacobian.
  fesolver = ThetaMethod(LUSolver(), 0.1, 0.5)
  red      = HighDimReduction(1e-2; dim=2)
  solver   = RBSolver(fesolver, red; nparams_res=5, nparams_jac=5, nparams_djac=5)

  @test solver isa RBSolver    # TransientRBSolver is a type alias for RBSolver{<:ODESolver,...}
  @test get_fe_solver(solver) === fesolver
  # residual and both Jacobian hyper-reductions are MDEIM by default
  @test solver.residual_reduction isa HighDimMDEIMHyperReduction
  @test length(solver.jacobian_reduction) == 2
  @test all(r -> r isa HighDimMDEIMHyperReduction, solver.jacobian_reduction)
end

# ─── _get_shift ───────────────────────────────────────────────────────────────

@testset "_get_shift returns :spacetime for HighDimReduction" begin
  # HighDimReduction (Kronecker or Sequential) requires full space-time assemby
  red = HighDimReduction(1e-2; dim=2)
  @test _get_shift(red) == :spacetime
end

@testset "_get_shift returns :spaceonly for SteadyReduction" begin
  # SteadyReduction keeps the time loop intact → only space-shift needed
  red = SteadyReduction(SearchSVDRank(1e-3))
  @test _get_shift(red) == :spaceonly
end

# ─── _to_realisation ──────────────────────────────────────────────────────────

@testset "_to_realisation creates a single-param TransientRealisation" begin
  # _to_realisation prepends the initial time so the result is compatible with
  # TransientRealisation(params, [t0, t1, ..., tN]) which stores t0 separately.
  r  = _make_realisation(3, 4)          # 3 params, 4 time steps
  # Iterating get_params(r) yields the raw parameter vectors (AbstractVector),
  # unlike get_params(r)[i] which wraps in a Realisation
  μ  = collect(get_params(r))[2]        # second parameter sample as Vector
  rμ = _to_realisation(r, μ)

  @test rμ isa TransientRealisation
  @test num_params(rμ) == 1
  # The inner time vector [t1,...,tN] has the same length as the original
  @test num_times(rμ) == num_times(r)
end

end # module
