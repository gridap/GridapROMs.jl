module RBTransientTests

using Test
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.Arrays: Table
using Gridap.ODEs: Newmark
using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures
using GridapROMs.ParamODEs
using GridapROMs.RBSteady
using GridapROMs.RBTransient

import GridapROMs.RBTransient: get_time_combination,time_enrichment,get_time_order,
  get_interpolation_style,get_indices_time

function _make_realisation(np=3,nt=4)
  params = Realisation([[i*0.1,i*0.2] for i in 1:np])
  times  = range(0.25,1.0;length=nt) |> collect
  TransientRealisation(params,times,0.0)
end

# ─── TransientReduction constructors ────────────────────────────────────────────

@testset "KroneckerReduction via TransientReduction — ReductionStyle vector" begin
  red = TransientReduction([PSVDRank(1e-2),PSVDRank(1e-2)])
  @test red isa KroneckerReduction
  @test length(red.reductions) == 2
end

@testset "KroneckerReduction from scalar tol + dim" begin
  red = TransientReduction(1e-2;dim=2)
  @test red isa KroneckerReduction
  @test num_params(red) == num_params(first(red.reductions))
end

@testset "SequentialReduction via TransientReduction — Float64 vector" begin
  seq = TransientReduction([1e-2,1e-2])
  @test seq isa SequentialReduction
end

@testset "SequentialReduction via TransientReduction — TTSVDRanks" begin
  seq = TransientReduction(TTSVDRanks([PSVDRank(1e-2),PSVDRank(1e-2)]))
  @test seq isa SequentialReduction
end

@testset "TimeMarchingReduction wraps a steady reduction" begin
  red = TimeMarchingReduction(PSVDRank(1e-3))
  @test red isa TimeMarchingReduction
  @test ReductionStyle(red) isa PSVDRank
end

# ─── TransientHyperReduction constructors ───────────────────────────────────────

@testset "TransientHyperReduction construction" begin
  red = TransientReduction(1e-2;dim=2)

  solver = ThetaMethod(LUSolver(),0.1,0.5)
  tcomb  = TimeCombination(solver)

  hrs = ntuple(i -> TransientHyperReduction(CombinationOrder{i}(tcomb),red),Val(get_time_order(solver)+1))
  @test length(hrs) == 2
  @test hrs[1] isa TransientDEIMHyperReduction
  @test hrs[2] isa TransientDEIMHyperReduction
  @test isa(get_time_combination(hrs[1]),ThetaMethodStrategy{1})
  @test isa(get_time_combination(hrs[2]),ThetaMethodStrategy{2})

  hr_no  = TransientHyperReduction(CombinationOrder{1}(tcomb),red;hypred_strategy=:no)
  hr_aff = TransientHyperReduction(CombinationOrder{1}(tcomb),red;hypred_strategy=:affine)
  @test hr_no isa TransientNoHyperReduction
  @test hr_aff isa TransientAffineHyperReduction

  solver = Newmark(LUSolver(),0.1,0.5,0.25)
  tcomb  = TimeCombination(solver)

  hrs = ntuple(i -> TransientHyperReduction(CombinationOrder{i}(tcomb),red;hypred_strategy=:sopt),
    Val(get_time_order(solver)+1))
  @test length(hrs) == 3
  @test hrs[1] isa TransientSOPTHyperReduction
  @test hrs[2] isa TransientSOPTHyperReduction
  @test hrs[3] isa TransientSOPTHyperReduction
  @test isa(get_time_combination(hrs[1]),GenAlpha2Strategy{1})
  @test isa(get_time_combination(hrs[2]),GenAlpha2Strategy{2})
  @test isa(get_time_combination(hrs[3]),GenAlpha2Strategy{3})
end

# ─── tucker ───────────────────────────────────────────────────────────────────

@testset "tucker on 3D array — PSVDRank" begin
  Ns,Nt,Np = 20,8,15
  A = rand(Ns,Nt,Np)
  reds = [Reduction(PSVDRank(1e-2)),Reduction(PSVDRank(1e-2))]
  bases = tucker(reds,A)

  @test length(bases) == 2
  Φs,Φt = bases
  @test size(Φs,1) == Ns
  @test size(Φs,2) <= Ns
  @test norm(Φs'*Φs - I(size(Φs,2))) < 1e-10
  @test size(Φt,1) == size(Φs,2)
  @test norm(Φt'*Φt - I(size(Φt,2))) < 1e-10
end

# ─── time_enrichment ──────────────────────────────────────────────────────────

@testset "time_enrichment enriches when dual is orthogonal to primal" begin
  nt = 10
  Φ_primal = Matrix(I(nt))[:,1:5]
  Φ_dual   = Matrix(I(nt))[:,6:8]
  result = time_enrichment(Φ_primal,Φ_dual;tol=1e-4)
  @test result isa AbstractMatrix
  @test size(result,1) == nt
  @test size(result,2) == 8
end

@testset "time_enrichment no-op when dual already in span" begin
  nt = 8
  Φ = Matrix(qr(rand(nt,4)).Q)
  result = time_enrichment(Φ,Φ[:,1:2];tol=1e-10)
  @test size(result,2) == 4
end

# ─── KroneckerProjection ──────────────────────────────────────────────────────

@testset "KroneckerProjection — field accessors" begin
  Ns,Nt,ns,nt = 20,8,4,3
  proj_s = PODProjection(Matrix(qr(rand(Ns,ns)).Q))
  basis_t = Matrix(qr(rand(Nt,nt)).Q)
  kp = KroneckerProjection(proj_s,basis_t)

  @test kp isa KroneckerProjection
  @test num_reduced_dofs(kp) == ns*nt
  @test num_space_dofs(kp)   == Ns
  @test num_times(kp)        == Nt
  @test get_basis_time(kp)   === basis_t
end

@testset "KroneckerProjection — project!/inv_project! idempotency" begin
  Ns,Nt,ns,nt = 20,8,4,3
  Φs = Matrix(qr(rand(Ns,ns)).Q)
  Φt = Matrix(qr(rand(Nt,nt)).Q)
  kp = KroneckerProjection(PODProjection(Φs),Φt)

  x  = rand(Ns*Nt)
  x̂  = zeros(ns*nt)
  project!(x̂,kp,x)

  x_rec = zeros(Ns*Nt)
  inv_project!(x_rec,kp,x̂)

  x̂2 = zeros(ns*nt)
  project!(x̂2,kp,x_rec)
  @test x̂2 ≈ x̂ atol=1e-10
end

# ─── SequentialProjection ─────────────────────────────────────────────────────

@testset "SequentialProjection — project!/inv_project! with a $name spatial projection" for name in (:pod,:tt)
  Ns,Nt,rs,n = 20,8,4,3
  Φs = Matrix(qr(rand(Ns,rs)).Q)
  ct = reshape(Matrix(qr(rand(rs*Nt,n)).Q),rs,Nt,n)
  ps = name == :pod ? PODProjection(Φs) : TTSVDProjection([reshape(Φs,1,Ns,rs)],VectorDofMap(Ns))
  sp = SequentialProjection(ps,ct)

  @test sp isa SequentialProjection
  @test num_reduced_dofs(sp) == n
  @test num_space_dofs(sp)   == Ns
  @test num_times(sp)        == Nt
  @test get_basis_space(sp)  ≈ Φs

  Φ = reshape(Φs*reshape(ct,rs,Nt*n),Ns*Nt,n)
  x̂ = rand(n)
  x = zeros(Ns*Nt)
  inv_project!(x,sp,x̂)
  @test x ≈ Φ*x̂

  x̂2 = zeros(n)
  project!(x̂2,sp,x)
  @test x̂2 ≈ x̂
end

# ─── TransientIntegrationDomain ───────────────────────────────────────────────

@testset "TransientInterpolation — struct construction and accessors" begin
  cells        = Int32[1,2,3]
  cell_row_ids   = Table(Int32[1,2,3],Int32[1,2,3,4])
  rows         = Int32[1,2,3]
  domain_space = GenericDomain(cells,cell_row_ids,rows)

  indices_time = Int32[2,4,6]
  tid = TransientInterpolation(KroneckerStyle(),domain_space,indices_time)

  @test get_interpolation_style(tid) isa KroneckerStyle
  @test get_indices_time(tid) == indices_time
  @test tid.interp_space === domain_space
  @test get_integration_cells(tid) === cells
end

@testset "TransientInterpolation — SequentialStyle variant" begin
  cells        = Int32[5,7]
  cell_row_ids   = Table(Int32[10,11,12],Int32[1,3,4])
  rows         = Int32[10,11,12]
  domain_space = GenericDomain(cells,cell_row_ids,rows)

  indices_time = Int32[1,3]
  tid = TransientInterpolation(SequentialStyle(),domain_space,indices_time)

  @test get_interpolation_style(tid) isa SequentialStyle
  @test get_indices_time(tid) == indices_time
end

# ─── TransientRBSolver construction ───────────────────────────────────────────

@testset "RBSolver(ODESolver,Reduction) builds a TransientRBSolver" begin
  fesolver = ThetaMethod(LUSolver(),0.1,0.5)
  red      = TransientReduction(1e-2;dim=2)
  solver   = RBSolver(fesolver,red;nparams_res=5,nparams_jacs=(5,5))

  @test solver isa RBSolver
  @test get_fe_solver(solver) === fesolver
  @test solver.residual_reduction isa TransientDEIMHyperReduction
  @test length(solver.jacobian_reduction) == 2
  @test all(r -> r isa TransientDEIMHyperReduction,solver.jacobian_reduction)

  solver_no = RBSolver(fesolver,red;nparams_res=5,nparams_jacs=(5,5),hypred_strategy=:no)
  @test solver_no.residual_reduction isa TransientNoHyperReduction
  @test all(r -> r isa TransientNoHyperReduction,solver_no.jacobian_reduction)

  solver_aff = RBSolver(fesolver,red;nparams_res=5,nparams_jacs=(5,5),hypred_strategy=:affine)
  @test solver_aff.residual_reduction isa TransientAffineHyperReduction
  @test all(r -> r isa TransientAffineHyperReduction,solver_aff.jacobian_reduction)
end

# ─── to_realisation ───────────────────────────────────────────────────────────

@testset "to_realisation creates a single-param TransientRealisation" begin
  r  = _make_realisation(3,4)
  μ  = collect(get_params(r))[2]
  rμ = RBSteady.to_realisation(r,μ)

  @test rμ isa TransientRealisation
  @test num_params(rμ) == 1
  @test num_times(rμ) == num_times(r)
end

end
