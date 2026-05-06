module RBTransientTests

using Test
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.Arrays: Table
using Gridap.ODEs: Newmark
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.ParamODEs
using GridapROMs.RBSteady
using GridapROMs.RBTransient

import GridapROMs.RBTransient: get_time_combination, time_enrichment, get_time_order,
  get_domain_style, get_indices_time, get_integration_domain_space, _to_realisation

function _make_realisation(np=3,nt=4)
  params = Realisation([[i*0.1,i*0.2] for i in 1:np])
  times  = range(0.25,1.0;length=nt) |> collect
  TransientRealisation(params,times,0.0)
end

# ─── HighDimReduction constructors ────────────────────────────────────────────

@testset "KroneckerReduction via HighDimReduction — ReductionStyle vector" begin
  red = HighDimReduction([SearchSVDRank(1e-2),SearchSVDRank(1e-2)])
  @test red isa KroneckerReduction
  @test length(red.reductions) == 2
end

@testset "KroneckerReduction from scalar tol + dim" begin
  red = HighDimReduction(1e-2;dim=2)
  @test red isa KroneckerReduction
  @test num_params(red) == num_params(first(red.reductions))
end

@testset "SequentialReduction via HighDimReduction — Float64 vector" begin
  seq = HighDimReduction([1e-2,1e-2])
  @test seq isa SequentialReduction
end

@testset "SequentialReduction via HighDimReduction — TTSVDRanks" begin
  seq = HighDimReduction(TTSVDRanks([SearchSVDRank(1e-2),SearchSVDRank(1e-2)]))
  @test seq isa SequentialReduction
end

@testset "SteadyReduction wraps a steady reduction" begin
  red = SteadyReduction(SearchSVDRank(1e-3))
  @test red isa SteadyReduction
  @test ReductionStyle(red) isa SearchSVDRank
end

# ─── HighDimHyperReduction constructors ───────────────────────────────────────

@testset "HighDimHyperReduction construction" begin
  red = HighDimReduction(1e-2;dim=2)

  solver = ThetaMethod(LUSolver(),0.1,0.5)
  tcomb  = TimeCombination(solver)

  hrs = ntuple(i -> HighDimHyperReduction(CombinationOrder{i}(tcomb),red),Val(get_time_order(solver)+1))
  @test length(hrs) == 2
  @test hrs[1] isa HighDimMDEIMHyperReduction
  @test hrs[2] isa HighDimMDEIMHyperReduction
  @test isa(get_time_combination(hrs[1]),ThetaMethodStrategy{1})
  @test isa(get_time_combination(hrs[2]),ThetaMethodStrategy{2})

  hr_no  = HighDimHyperReduction(CombinationOrder{1}(tcomb),red;hypred_strategy=:no)
  hr_aff = HighDimHyperReduction(CombinationOrder{1}(tcomb),red;hypred_strategy=:affine)
  @test hr_no isa RBSteady.NoHyperReduction
  @test hr_aff isa RBSteady.AffineHyperReduction

  solver = Newmark(LUSolver(),0.1,0.5,0.25)
  tcomb  = TimeCombination(solver)

  hrs = ntuple(i -> HighDimHyperReduction(CombinationOrder{i}(tcomb),red;hypred_strategy=:sopt),
    Val(get_time_order(solver)+1))
  @test length(hrs) == 3
  @test hrs[1] isa HighDimSOPTHyperReduction
  @test hrs[2] isa HighDimSOPTHyperReduction
  @test hrs[3] isa HighDimSOPTHyperReduction
  @test isa(get_time_combination(hrs[1]),GenAlpha2Strategy{1})
  @test isa(get_time_combination(hrs[2]),GenAlpha2Strategy{2})
  @test isa(get_time_combination(hrs[3]),GenAlpha2Strategy{3})
end

# ─── tucker ───────────────────────────────────────────────────────────────────

@testset "tucker on 3D array — SearchSVDRank" begin
  Ns,Nt,Np = 20,8,15
  A = rand(Ns,Nt,Np)
  reds = [Reduction(SearchSVDRank(1e-2)),Reduction(SearchSVDRank(1e-2))]
  bases = tucker(reds,A)

  @test length(bases) == 2
  Φs,Φt = bases
  @test size(Φs,1) == Ns
  @test size(Φs,2) <= Ns
  @test norm(Φs'*Φs - I(size(Φs,2))) < 1e-10
  @test size(Φt,1) == size(Φs,2)
  @test norm(Φt'*Φt - I(size(Φt,2))) < 1e-10
end

@testset "tucker on 3D array — FixedSVDRank" begin
  Ns,Nt,Np = 30,10,12
  A = rand(Ns,Nt,Np)
  ks,kt = 4,3
  reds = [Reduction(FixedSVDRank(ks)),Reduction(FixedSVDRank(kt))]
  bases = tucker(reds,A)
  @test size(bases[1],1) == Ns
  @test size(bases[1],2) == ks
  @test size(bases[2],1) == ks
  @test size(bases[2],2) == kt
end

# ─── time_enrichment ──────────────────────────────────────────────────────────

@testset "time_enrichment enriches when dual is orthogonal to primal" begin
  nt = 10
  Φ_primal = Matrix(I(nt))[:,1:5]
  Φ_dual   = Matrix(I(nt))[:,6:8]
  result = time_enrichment(PODProjection(Φ_primal),Φ_dual;tol=1e-4)
  @test get_basis(result) isa AbstractMatrix
  @test size(get_basis(result),1) == nt
  @test size(get_basis(result),2) == 8
end

@testset "time_enrichment no-op when dual already in span" begin
  nt = 8
  Φ = Matrix(qr(rand(nt,4)).Q)
  result = time_enrichment(PODProjection(Φ),Φ[:,1:2];tol=1e-10)
  @test size(get_basis(result),2) == 4
end

# ─── KroneckerProjection ──────────────────────────────────────────────────────

@testset "KroneckerProjection — field accessors" begin
  Ns,Nt,ns,nt = 20,8,4,3
  proj_s = PODProjection(Matrix(qr(rand(Ns,ns)).Q))
  proj_t = PODProjection(Matrix(qr(rand(Nt,nt)).Q))
  kp = KroneckerProjection(proj_s,proj_t)

  @test num_reduced_dofs(kp) == ns*nt
  @test num_space_dofs(kp)   == Ns
  @test num_times(kp)        == Nt
end

@testset "KroneckerProjection — project!/inv_project! idempotency" begin
  Ns,Nt,ns,nt = 20,8,4,3
  Φs = Matrix(qr(rand(Ns,ns)).Q)
  Φt = Matrix(qr(rand(Nt,nt)).Q)
  kp = KroneckerProjection(PODProjection(Φs),PODProjection(Φt))

  x  = rand(Ns*Nt)
  x̂  = zeros(ns*nt)
  project!(x̂,kp,x)

  x_rec = zeros(Ns*Nt)
  inv_project!(x_rec,kp,x̂)

  x̂2 = zeros(ns*nt)
  project!(x̂2,kp,x_rec)
  @test x̂2 ≈ x̂ atol=1e-10
end

# ─── TransientIntegrationDomain ───────────────────────────────────────────────

@testset "TransientIntegrationDomain — struct construction and accessors" begin
  cells        = Int32[1,2,3]
  cell_irows   = Table(Int32[1,2,3],Int32[1,2,3,4])
  rows         = Int32[1,2,3]
  domain_space = VectorDomain(cells,cell_irows,rows)

  indices_time = Int32[2,4,6]
  tid = TransientIntegrationDomain(KroneckerDomain(),domain_space,indices_time)

  @test get_domain_style(tid) isa KroneckerDomain
  @test get_indices_time(tid) == indices_time
  @test get_integration_domain_space(tid) === domain_space
  @test get_integration_cells(tid) === cells
end

@testset "TransientIntegrationDomain — SequentialDomain variant" begin
  cells        = Int32[5,7]
  cell_irows   = Table(Int32[10,11,12],Int32[1,3,4])
  rows         = Int32[10,11,12]
  domain_space = VectorDomain(cells,cell_irows,rows)

  indices_time = Int32[1,3]
  tid = TransientIntegrationDomain(SequentialDomain(),domain_space,indices_time)

  @test get_domain_style(tid) isa SequentialDomain
  @test get_indices_time(tid) == indices_time
end

# ─── TransientRBSolver construction ───────────────────────────────────────────

@testset "RBSolver(ODESolver, Reduction) builds a TransientRBSolver" begin
  fesolver = ThetaMethod(LUSolver(),0.1,0.5)
  red      = HighDimReduction(1e-2;dim=2)
  solver   = RBSolver(fesolver,red;nparams_res=5,nparams_jacs=(5,5))

  @test solver isa RBSolver
  @test get_fe_solver(solver) === fesolver
  @test solver.residual_reduction isa HighDimMDEIMHyperReduction
  @test length(solver.jacobian_reduction) == 2
  @test all(r -> r isa HighDimMDEIMHyperReduction,solver.jacobian_reduction)

  solver_no = RBSolver(fesolver,red;nparams_res=5,nparams_jacs=(5,5),hypred_strategy=:no)
  @test solver_no.residual_reduction isa RBSteady.NoHyperReduction
  @test all(r -> r isa RBSteady.NoHyperReduction,solver_no.jacobian_reduction)

  solver_aff = RBSolver(fesolver,red;nparams_res=5,nparams_jacs=(5,5),hypred_strategy=:affine)
  @test solver_aff.residual_reduction isa RBSteady.AffineHyperReduction
  @test all(r -> r isa RBSteady.AffineHyperReduction,solver_aff.jacobian_reduction)
end

# ─── _to_realisation ──────────────────────────────────────────────────────────

@testset "_to_realisation creates a single-param TransientRealisation" begin
  r  = _make_realisation(3,4)
  μ  = collect(get_params(r))[2]
  rμ = _to_realisation(r,μ)

  @test rμ isa TransientRealisation
  @test num_params(rμ) == 1
  @test num_times(rμ) == num_times(r)
end

end # module
