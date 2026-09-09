module TProductTests

using Test
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.CellData
using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.TProduct

# ─── helpers ──────────────────────────────────────────────────────────────────

"""Build a standard 2D 10×10 Cartesian model on [0,1]²."""
make_model_2d() = CartesianDiscreteModel((0,1,0,1),(10,10))

"""Build a standard 3D 4×4×4 Cartesian model on [0,1]³."""
make_model_3d() = CartesianDiscreteModel((0,1,0,1,0,1),(4,4,4))

# ─── FE spaces ────────────────────────────────────────────────────────────────

@testset "TProductFESpace scalar" begin
  model = make_model_2d()
  trian = Triangulation(model)
  order = 1

  V = TProductFESpace(trian,ReferenceFE(lagrangian,Float64,order);conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7,8])
  @test V isa TProductFESpace
  @test length(V.spaces_1d) == 2

  # The 1D spaces must each have the right size
  n1d = 11   # 10 intervals + 1 node per direction for Q1
  @test num_free_dofs(V.spaces_1d[1]) + num_dirichlet_dofs(V.spaces_1d[1]) == n1d
  @test num_free_dofs(V.spaces_1d[2]) + num_dirichlet_dofs(V.spaces_1d[2]) == n1d
end

@testset "TProductFESpace homogeneous Dirichlet" begin
  model = make_model_2d()
  trian = Triangulation(model)
  V = TProductFESpace(trian,ReferenceFE(lagrangian,Float64,1);conformity=:H1,dirichlet_tags="boundary")
  @test num_free_dofs(V) == 9*9   # interior nodes for 10×10 mesh, Q1
end

@testset "TProductFESpace no Dirichlet" begin
  model = make_model_2d()
  trian = Triangulation(model)
  V = TProductFESpace(trian,ReferenceFE(lagrangian,Float64,1);conformity=:H1)
  @test num_free_dofs(V) == 11*11
end

@testset "TProductFESpace quadratic" begin
  model = make_model_2d()
  trian = Triangulation(model)
  V = TProductFESpace(trian,ReferenceFE(lagrangian,Float64,2);conformity=:H1,dirichlet_tags="boundary")
  @test V isa TProductFESpace
  # Q2 on 10×10: 21 nodes per direction, interior = 19×19
  @test num_free_dofs(V) == 19*19
end

@testset "TProductFESpace 3D" begin
  model = make_model_3d()
  trian = Triangulation(model)
  V = TProductFESpace(trian,ReferenceFE(lagrangian,Float64,1);conformity=:H1,dirichlet_tags="boundary")
  @test V isa TProductFESpace
  @test length(V.spaces_1d) == 3
  # Q1 on 4×4×4 with full Dirichlet: 3×3×3 interior
  @test num_free_dofs(V) == 3*3*3
end

@testset "TProductFESpace vector-valued" begin
  model = make_model_2d()
  trian = Triangulation(model)
  V = TProductFESpace(trian,ReferenceFE(lagrangian,VectorValue{2,Float64},2);conformity=:H1,dirichlet_tags="boundary")
  @test V isa TProductFESpace
  # scalar 1D factors are shared across all components
  @test length(V.spaces_1d) == 2
  @test num_free_dofs(V) == 2*19*19
end

# ─── rank tensors ─────────────────────────────────────────────────────────────

@testset "Rank1Tensor" begin
  A1 = rand(4,4)
  A2 = rand(5,5)
  a = Rank1Tensor([A1,A2])

  @test rank(a) == 1
  @test GridapROMs.TProduct.dimension(a) == 2
  @test get_factors(a) === a.factors
  @test a[1] === A1
  @test a[2] === A2

  # Kronecker product: A2 ⊗ A1 (reversed per kron convention)
  K = kron(a)
  @test K ≈ kron(A2,A1)

  # Matrix-tensor multiply: a * b = A1*b*A2'
  b = rand(4,5)
  @test a * b ≈ A1 * b * A2'
  @test b * a ≈ A1 * b * A2'   # symmetric in this case when A1=A2 but checking dispatch
end

@testset "GenericRankTensor" begin
  A1,A2 = rand(4,4),rand(5,5)
  dA1,dA2 = rand(4,4),rand(5,5)

  # Build like tproduct_array(gradient,...) does
  d1 = Rank1Tensor([dA1,A2])
  d2 = Rank1Tensor([A1,dA2])
  g = GenericRankTensor([d1,d2])

  @test rank(g) == 2
  @test GridapROMs.TProduct.dimension(g) == 2

  # get_factor: g[d,k] = d-th factor of k-th decomposition
  @test TProduct.get_factor(g,1,1) === dA1
  @test TProduct.get_factor(g,2,1) === A2
  @test TProduct.get_factor(g,1,2) === A1
  @test TProduct.get_factor(g,2,2) === dA2

  # kron: sum over ranks
  K = kron(g)
  @test K ≈ kron(A2,dA1) + kron(dA2,A1)
end

@testset "Rank1Tensor cholesky" begin
  A1 = rand(4,4); A1 = A1'*A1 + 4I
  A2 = rand(5,5); A2 = A2'*A2 + 4I
  a = Rank1Tensor([A1,A2])
  c = cholesky(a)
  @test length(c) == 2
  @test c[1].U ≈ cholesky(A1).U
  @test c[2].U ≈ cholesky(A2).U
end

# ─── tensor-product assembly (mass/stiffness on spaces_1d) ────────────────────

"""1D mass matrix assembled directly on a `spaces_1d` factor."""
function _mass_1d(V1d)
  Ω1d = get_triangulation(V1d)
  dΩ1d = Measure(Ω1d,2)
  assemble_matrix((u,v) -> ∫(u*v)dΩ1d,V1d,V1d)
end

"""1D stiffness matrix assembled directly on a `spaces_1d` factor."""
function _stiffness_1d(V1d)
  Ω1d = get_triangulation(V1d)
  dΩ1d = Measure(Ω1d,2)
  assemble_matrix((u,v) -> ∫(∇(u)⋅∇(v))dΩ1d,V1d,V1d)
end

@testset "tensor-product mass matrix via Rank1Tensor" begin
  model = make_model_2d()
  trian = Triangulation(model)
  V = TProductFESpace(trian,ReferenceFE(lagrangian,Float64,1);conformity=:H1)

  mass_1d = map(_mass_1d,V.spaces_1d)
  M = Rank1Tensor(mass_1d)

  @test M isa Rank1Tensor
  @test rank(M) == 1
  @test length(get_factors(M)) == 2
  @test get_factors(M)[1] isa AbstractSparseMatrix
  @test get_factors(M)[2] isa AbstractSparseMatrix

  Mk = kron(M)
  @test issymmetric(Mk)
  @test isposdef(Matrix(Mk))

  # standard (dense) assembly on the same (lexicographically reindexed) space
  # must produce exactly the Kronecker product of the 1D mass matrices
  dΩ = Measure(trian,2)
  M_std = assemble_matrix((u,v) -> ∫(u*v)dΩ,V,V)
  @test kron(M) ≈ M_std
end

@testset "tensor-product stiffness matrix via GenericRankTensor" begin
  model = make_model_2d()
  trian = Triangulation(model)
  V = TProductFESpace(trian,ReferenceFE(lagrangian,Float64,1);conformity=:H1,dirichlet_tags="boundary")

  mass_1d = map(_mass_1d,V.spaces_1d)
  stiff_1d = map(_stiffness_1d,V.spaces_1d)
  # swap in the derivative factor in dimension i, keeping mass elsewhere
  inds = LinearIndices(mass_1d)
  decompositions = map(inds) do i
    di = copy(mass_1d)
    di[i] = stiff_1d[i]
    Rank1Tensor(di)
  end
  K = GenericRankTensor(decompositions)

  @test K isa GenericRankTensor
  @test rank(K) == 2

  dΩ = Measure(trian,2)
  K_std = assemble_matrix((u,v) -> ∫(∇(u)⋅∇(v))dΩ,V,V)
  @test kron(K) ≈ K_std
end

# ─── dof maps ─────────────────────────────────────────────────────────────────

@testset "get_dof_map TProductFESpace" begin
  model = make_model_2d()
  trian = Triangulation(model)
  V = TProductFESpace(trian,ReferenceFE(lagrangian,Float64,1);conformity=:H1)
  dmap = get_dof_map(V)
  # For no-Dirichlet Q1 on 10×10: 11×11 dof array
  @test ndims(dmap) == 2
  @test size(dmap) == (11,11)
end

@testset "get_sparse_dof_map TProductFESpace" begin
  model = make_model_2d()
  trian = Triangulation(model)
  V = TProductFESpace(trian,ReferenceFE(lagrangian,Float64,2);conformity=:H1,dirichlet_tags=[1,3,7])
  sdm = get_sparse_dof_map(V,V)
  @test !(sdm isa TrivialSparseMatrixDofMap)
end

end
