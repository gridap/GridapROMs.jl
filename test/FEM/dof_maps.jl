module DofMapsTests

using Test
using LinearAlgebra
using SparseArrays
using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using GridapROMs
using GridapROMs.DofMaps

# ─── helpers ──────────────────────────────────────────────────────────────────

function make_space_2d(;dirichlet=false)
  model = CartesianDiscreteModel((0,1,0,1),(8,8))
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  if dirichlet
    FESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
  else
    FESpace(model,reffe;conformity=:H1)
  end
end

# ─── IndexOperations ──────────────────────────────────────────────────────────

@testset "recast_indices / sparsify_indices round-trip" begin
  A = sprand(10,10,0.4) + I(10)
  nnzA = nnz(A)
  sids = collect(1:nnzA)
  fids = GridapROMs.DofMaps.recast_indices(sids,A)
  sids2 = GridapROMs.DofMaps.sparsify_indices(fids,A)
  @test sids2 == sids
end

@testset "recast_split_indices" begin
  A = sprand(8,8,0.4) + I(8)   # 8×8 with guaranteed non-zero diagonal
  nnzA = nnz(A)
  sids = collect(1:nnzA)
  frows,fcols = GridapROMs.DofMaps.recast_split_indices(sids,A)
  I_nz,J_nz, = findnz(A)
  @test frows == I_nz
  @test fcols == J_nz
end

@testset "slow_index and fast_index" begin
  # slow_index(i,n) = ceil(i/n), fast_index(i,n) = mod1(i,n)
  n = 3
  for i in 1:9
    @test GridapROMs.DofMaps.slow_index(i,n) == cld(i,n)
    @test GridapROMs.DofMaps.fast_index(i,n) == mod1(i,n)
  end
end

# ─── VectorDofMap ─────────────────────────────────────────────────────────────

@testset "VectorDofMap identity" begin
  dm = VectorDofMap((9,))
  @test size(dm) == (9,)
  # identity map: each index maps to itself
  for j in 1:9
    @test dm[j] == j
  end
  invert(dm)   # must not throw
end

@testset "VectorDofMap 2D" begin
  dm = VectorDofMap((3,4))
  @test ndims(dm) == 2
  @test size(dm) == (3,4)
  @test length(dm) == 12
end

@testset "VectorDofMap flatten" begin
  dm = VectorDofMap((3,4))
  f = flatten(dm)
  @test ndims(f) == 1
  @test size(f) == (12,)
end

@testset "InverseDofMap" begin
  dm = VectorDofMap((6,))
  inv_dm = invert(dm)
  @test inv_dm isa InverseDofMap
  # for identity dof map, inverse is identity
  for j in 1:6
    @test inv_dm[j] == j
  end
end

# ─── SparsityPattern ──────────────────────────────────────────────────────────

@testset "SparsityCSC from FE spaces" begin
  V = make_space_2d()
  U = TrialFESpace(V)
  sp = get_sparsity(U,V)
  @test sp isa SparsityCSC
  @test nnz(sp) > 0
  @test size(sp.matrix,1) == num_free_dofs(V)
  @test size(sp.matrix,2) == num_free_dofs(U)
end

# ─── get_dof_map / get_dof_map_with_diri ─────────────────────────────────────

@testset "get_dof_map for SingleFieldFESpace" begin
  # For a plain FESpace, get_dof_map returns a 1D VectorDofMap over free DOFs
  V = make_space_2d(;dirichlet=false)
  dm = get_dof_map(V)
  @test dm isa AbstractDofMap
  @test ndims(dm) == 1
  # Q1 on 8×8 mesh: 9×9 = 81 free DOFs
  @test length(dm) == 81
end

@testset "get_dof_map with Dirichlet" begin
  V = make_space_2d(;dirichlet=true)
  dm = get_dof_map(V)
  @test dm isa AbstractDofMap
  # free DOFs only: 7×7 = 49
  @test num_free_dofs(V) == 49
  @test length(dm) == 49
end

@testset "get_sparse_dof_map" begin
  V = make_space_2d(;dirichlet=false)
  U = TrialFESpace(V)
  dm = get_sparse_dof_map(U,V)
  @test dm isa AbstractSparseDofMap
  # nnz of the dof map must match the sparsity nnz
  sp = get_sparsity(U,V)
  @test nnz(dm.sparsity) == nnz(sp)
end

# ─── DofMapArray ──────────────────────────────────────────────────────────────

@testset "DofMapArray wraps data and map" begin
  dm = VectorDofMap((9,))
  data = rand(Float64,9)
  dma = DofMapArray(data,dm)
  @test dma isa DofMapArray
  @test size(dma) == (9,)
  # indexing through the map should return the right element
  for j in 1:9
    @test dma[j] == data[dm[j]]
  end
end

# ─── OrderedFESpace ───────────────────────────────────────────────────────────

@testset "OrderedFESpace wraps an FESpace" begin
  V = make_space_2d(;dirichlet=false)
  Vord = OrderedFESpace(V)
  @test Vord isa OrderedFESpace
  # free DOF count must be preserved
  @test num_free_dofs(Vord) == num_free_dofs(V)
end

end # module
