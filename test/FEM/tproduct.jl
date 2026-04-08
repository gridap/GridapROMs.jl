module TProductTests

using Test
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.CellData
using GridapROMs
using GridapROMs.TProduct

# ─── helpers ──────────────────────────────────────────────────────────────────

"""Build a standard 2D 10×10 tensor product model on [0,1]²."""
function make_model_2d()
  TProductDiscreteModel((0,1,0,1),(10,10))
end

"""Build a standard 3D 4×4×4 tensor product model on [0,1]³."""
function make_model_3d()
  TProductDiscreteModel((0,1,0,1,0,1),(4,4,4))
end

# ─── geometry ─────────────────────────────────────────────────────────────────

@testset "TProductDiscreteModel" begin
  model = make_model_2d()

  # D-dimensional model must be CartesianDiscreteModel{2}
  @test model.model isa CartesianDiscreteModel{2}
  # Must have exactly D=2 1D factor models
  @test length(model.models_1d) == 2
  @test all(m -> m isa CartesianDiscreteModel{1},model.models_1d)

  # Gridap DiscreteModel interface must be satisfied via the wrapped model
  @test get_grid(model) === get_grid(model.model)
  @test get_face_labeling(model) === get_face_labeling(model.model)

  # 3D case
  model3 = make_model_3d()
  @test length(model3.models_1d) == 3
  @test all(m -> m isa CartesianDiscreteModel{1},model3.models_1d)
end

@testset "TProductTriangulation and TProductMeasure" begin
  model = make_model_2d()
  Ω = Triangulation(model)

  @test Ω isa TProductTriangulation
  @test length(Ω.trians_1d) == 2
  @test get_background_model(Ω) === model

  dΩ = Measure(Ω,2)
  @test dΩ isa TProductMeasure
  @test length(dΩ.measures_1d) == 2

  # Equality is based on the underlying triangulation
  Ω2 = Triangulation(model)
  @test Ω == Ω2
end

# ─── reference FE ─────────────────────────────────────────────────────────────

@testset "TensorProductReferenceFE" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)

  @test tp_reffe isa TProduct.TensorProductReferenceFE
  # Must have D=2 1D reffes
  @test length(tp_reffe.reffes_1d) == 2
  # The wrapped D-dim reffe must be a 2D reference FE
  @test get_polytope(tp_reffe) isa Gridap.ReferenceFEs.Polytope{2}
  # num_dofs delegates to the wrapped reffe
  @test num_dofs(tp_reffe) == num_dofs(tp_reffe.reffe)
  # polynomial order round-trip
  @test get_order(tp_reffe) == 1

  # Order-2 reffe
  tp_reffe2 = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,2)
  @test get_order(tp_reffe2) == 2
end

# ─── FE spaces ────────────────────────────────────────────────────────────────

@testset "TProductFESpace scalar" begin
  model = make_model_2d()
  order = 1
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  # Construction via TensorProductReferenceFE
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,order)
  V = FESpace(model,tp_reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7,8])
  @test V isa TProductFESpace
  @test length(V.spaces_1d) == 2

  # Construction via the tuple reffe on TProductTriangulation
  V2 = FESpace(Ω,(lagrangian,Float64,order);conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7,8])
  @test V2 isa TProductFESpace

  # Free and Dirichlet DOF counts must agree between both paths
  @test num_free_dofs(V) == num_free_dofs(V2)
  @test num_dirichlet_dofs(V) == num_dirichlet_dofs(V2)

  # The 1D spaces must each have the right size
  n1d = 11   # 10 intervals + 1 node per direction for Q1
  @test num_free_dofs(V.spaces_1d[1]) + num_dirichlet_dofs(V.spaces_1d[1]) == n1d
  @test num_free_dofs(V.spaces_1d[2]) + num_dirichlet_dofs(V.spaces_1d[2]) == n1d
end

@testset "TProductFESpace homogeneous Dirichlet" begin
  model = make_model_2d()
  order = 1
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,order)
  # Full Dirichlet boundary
  V = FESpace(model,tp_reffe;conformity=:H1,dirichlet_tags="boundary")
  @test num_free_dofs(V) == 9*9   # interior nodes for 10×10 mesh, Q1
end

@testset "TProductFESpace no Dirichlet" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V = FESpace(model,tp_reffe;conformity=:H1)
  @test num_free_dofs(V) == 11*11
end

@testset "TProductFESpace quadratic" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,2)
  V = FESpace(model,tp_reffe;conformity=:H1,dirichlet_tags="boundary")
  @test V isa TProductFESpace
  # Q2 on 10×10: 21 nodes per direction, interior = 19×19
  @test num_free_dofs(V) == 19*19
end

@testset "TProductFESpace 3D" begin
  model = make_model_3d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V = FESpace(model,tp_reffe;conformity=:H1,dirichlet_tags="boundary")
  @test V isa TProductFESpace
  @test length(V.spaces_1d) == 3
  # Q1 on 4×4×4 with full Dirichlet: 3×3×3 interior
  @test num_free_dofs(V) == 3*3*3
end

# ─── tensor product basis ──────────────────────────────────────────────────────

@testset "TProductFEBasis" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V = FESpace(model,tp_reffe;conformity=:H1)
  U = TrialFESpace(V)

  bv = get_tp_fe_basis(V)
  bu = get_tp_trial_fe_basis(U)

  @test bv isa GridapROMs.TProduct.TProductFEBasis
  @test bu isa GridapROMs.TProduct.TProductFEBasis
  @test length(bv.basis) == 2
  @test length(bu.basis) == 2
  # Test and trial basis styles must differ
  @test FESpaces.BasisStyle(bv) isa FESpaces.TestBasis
  @test FESpaces.BasisStyle(bu) isa FESpaces.TrialBasis
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

  # get_arrays_1d / get_gradients_1d
  @test GridapROMs.TProduct.get_arrays_1d(g)[1] === A1
  @test GridapROMs.TProduct.get_arrays_1d(g)[2] === A2
  @test GridapROMs.TProduct.get_gradients_1d(g)[1] === dA1
  @test GridapROMs.TProduct.get_gradients_1d(g)[2] === dA2

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

# ─── assembly ─────────────────────────────────────────────────────────────────

@testset "TProductSparseMatrixAssembler mass matrix" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V = FESpace(model,tp_reffe;conformity=:H1)

  Ω = TProduct.get_tp_triangulation(V)
  dΩ = Measure(Ω,2)

  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(V)

  a_tp = ∫(v*u)dΩ
  assem = TProductSparseMatrixAssembler(V,V)
  matdata = collect_cell_matrix(V,V,a_tp)
  M = assemble_matrix(assem,matdata)

  @test M isa Rank1Tensor
  @test rank(M) == 1
  @test length(get_factors(M)) == 2
  # Each 1D factor must be a sparse matrix
  @test get_factors(M)[1] isa AbstractSparseMatrix
  @test get_factors(M)[2] isa AbstractSparseMatrix
  # Full Kronecker product must be symmetric positive definite
  Mk = kron(M)
  @test issymmetric(Mk)
  @test isposdef(Matrix(Mk))
end

@testset "TProductSparseMatrixAssembler stiffness matrix" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V = FESpace(model,tp_reffe;conformity=:H1,dirichlet_tags="boundary")

  Ω = TProduct.get_tp_triangulation(V)
  dΩ = Measure(Ω,2)

  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(V)

  a_tp = ∫(∇(v)⋅∇(u))dΩ
  assem = TProductSparseMatrixAssembler(V,V)
  matdata = collect_cell_matrix(V,V,a_tp)
  K = assemble_matrix(assem,matdata)

  @test K isa GenericRankTensor
  @test rank(K) == 2
  # Full matrix must match standard assembly
  Ω_std = get_triangulation(V)
  dΩ_std = Measure(Ω_std,2)
  v_std = get_fe_basis(V)
  u_std = get_trial_fe_basis(V)
  K_std = assemble_matrix(SparseMatrixAssembler(V,V),collect_cell_matrix(V,V,∫(∇(v_std)⋅∇(u_std))dΩ_std))
  @test kron(K) ≈ K_std
end

@testset "TProductSparseMatrixAssembler in-place assembly Rank1Tensor" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V = FESpace(model,tp_reffe;conformity=:H1)

  Ω = TProduct.get_tp_triangulation(V)
  dΩ = Measure(Ω,2)

  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(V)
  a_tp = ∫(v*u)dΩ

  assem = TProductSparseMatrixAssembler(V,V)
  matdata = collect_cell_matrix(V,V,a_tp)

  M_alloc = allocate_matrix(assem,matdata)
  assemble_matrix!(M_alloc,assem,matdata)

  M_direct = assemble_matrix(assem,matdata)

  @test kron(M_alloc) ≈ kron(M_direct)
end

@testset "TProductSparseMatrixAssembler in-place assembly GenericRankTensor" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V = FESpace(model,tp_reffe;conformity=:H1,dirichlet_tags="boundary")

  Ω = TProduct.get_tp_triangulation(V)
  dΩ = Measure(Ω,2)

  v = get_tp_fe_basis(V)
  u = get_tp_trial_fe_basis(V)
  a_tp = ∫(∇(v)⋅∇(u))dΩ

  assem = TProductSparseMatrixAssembler(V,V)
  matdata = collect_cell_matrix(V,V,a_tp)

  K_alloc = allocate_matrix(assem,matdata)
  assemble_matrix!(K_alloc,assem,matdata)

  K_direct = assemble_matrix(assem,matdata)

  @test kron(K_alloc) ≈ kron(K_direct)
end

# ─── dof maps ─────────────────────────────────────────────────────────────────

@testset "get_dof_map TProductFESpace" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V = FESpace(model,tp_reffe;conformity=:H1)
  dmap = get_dof_map(V)
  # For no-Dirichlet Q1 on 10×10: 11×11 dof array
  @test ndims(dmap) == 2
  @test size(dmap) == (11,11)
end

@testset "get_dof_map_with_diri TProductFESpace" begin
  model = make_model_2d()
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V = FESpace(model,tp_reffe;conformity=:H1,dirichlet_tags="boundary")
  dmap = get_dof_map_with_diri(V)
  @test ndims(dmap) == 2
  @test size(dmap) == (11,11)
end

# ─── consistency check: TensorProductReferenceFE vs tuple path ────────────────

@testset "TensorProductReferenceFE matches tuple construction" begin
  model = make_model_2d()
  Ω = Triangulation(model)

  # New path
  tp_reffe = TProduct.TensorProductReferenceFE(model,lagrangian,Float64,1)
  V1 = FESpace(model,tp_reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7,8])

  # Legacy tuple path
  V2 = FESpace(Ω,(lagrangian,Float64,1);conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7,8])

  @test num_free_dofs(V1) == num_free_dofs(V2)
  @test num_dirichlet_dofs(V1) == num_dirichlet_dofs(V2)
  @test length(V1.spaces_1d) == length(V2.spaces_1d)
  for d in 1:2
    @test num_free_dofs(V1.spaces_1d[d]) == num_free_dofs(V2.spaces_1d[d])
    @test num_dirichlet_dofs(V1.spaces_1d[d]) == num_dirichlet_dofs(V2.spaces_1d[d])
  end
end

end # module
