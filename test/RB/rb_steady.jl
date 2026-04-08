module RBSteadyTests

using Test
using LinearAlgebra
using SparseArrays
using GridapROMs
using GridapROMs.RBSteady

# ─── Reduction style constructors ─────────────────────────────────────────────

@testset "ReductionStyle constructors" begin
  @test SearchSVDRank(1e-4) isa SearchSVDRank
  @test FixedSVDRank(5) isa FixedSVDRank
  @test TTSVDRanks([1e-3,1e-3]) isa TTSVDRanks
  @test EuclideanNorm() isa EuclideanNorm
  @test EnergyNorm(x -> x) isa EnergyNorm
end

# ─── truncated POD ────────────────────────────────────────────────────────────

@testset "tpod SearchSVDRank" begin
  # low-rank data: rank-3 matrix + noise
  A = rand(100,50) * rand(50,30) + 1e-10*rand(100,30)
  red = SearchSVDRank(1e-3)
  U,S,V = tpod(red,A)
  @test size(U,1) == 100
  @test size(U,2) <= 30          # compressed rank ≤ full rank
  @test norm(U'*U - I(size(U,2))) < 1e-10  # orthonormal columns
end

@testset "tpod FixedSVDRank" begin
  A = rand(50,40)
  k = 5
  red = FixedSVDRank(k)
  U,S,V = tpod(red,A)
  @test size(U,2) == k
  @test norm(U'*U - I(k)) < 1e-10
end

# ─── gram-schmidt ─────────────────────────────────────────────────────────────

@testset "gram_schmidt produces orthonormal columns" begin
  A = rand(20,5)
  Q = gram_schmidt(A)
  @test size(Q) == size(A)
  @test norm(Q'*Q - I(5)) < 1e-10
end

# ─── orth_projection and orth_complement! ─────────────────────────────────────

@testset "orth_projection" begin
  # project onto the first coordinate axis
  e1 = [1.0,0.0,0.0]
  basis = reshape(e1,:,1)   # 3×1
  v = [3.0,2.0,1.0]
  proj = orth_projection(v,basis)
  @test proj ≈ [3.0,0.0,0.0]
end

@testset "orth_complement!" begin
  e1 = [1.0,0.0,0.0]
  basis = reshape(e1,:,1)
  v = [3.0,2.0,1.0]
  orth_complement!(v,basis)
  @test abs(dot(v,[1.0,0.0,0.0])) < 1e-12
end

# ─── ttsvd ────────────────────────────────────────────────────────────────────

@testset "ttsvd on 3D tensor" begin
  # random 3D tensor: space1 × space2 × params
  A = rand(10,8,20)
  red = TTSVDRanks([SearchSVDRank(1e-2),SearchSVDRank(1e-2)])
  cores,remainder = ttsvd(red,A)
  @test length(cores) == 2
  # each core must be a 3D array (rank_in × size × rank_out)
  for c in cores
    @test ndims(c) == 3
  end
end

# ─── galerkin_projection ──────────────────────────────────────────────────────

@testset "galerkin_projection" begin
  # galerkin_projection(Φₗ,A) = Φₗ'*A
  n,r1,r2 = 20,4,6
  Phi_l = Matrix(qr(rand(n,r1)).Q)
  Phi_r = Matrix(qr(rand(n,r2)).Q)
  proj = galerkin_projection(Phi_l,Phi_r)
  @test size(proj) == (r1,r2)
  @test proj ≈ Phi_l'*Phi_r
end

end # module
