module ParamDataStructuresTests

using Test
using LinearAlgebra
using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures

# ─── ParamSpace and Realisation ───────────────────────────────────────────────

@testset "ParamSpace construction" begin
  p = ParamSpace((0.0,1.0,2.0,3.0))   # two parameters in [0,1] and [2,3]
  @test p isa ParamSpace
  @test length(p.param_domain) == 2
  @test p.param_domain[1] == [0.0,1.0]
  @test p.param_domain[2] == [2.0,3.0]
end

@testset "realisation sampling" begin
  p = ParamSpace((0.0,1.0))            # one parameter in [0,1]
  r = realisation(p;nparams=5)
  @test r isa Realisation
  @test num_params(r) == 5
  @test length(r) == 5
  # each sample must be within [0,1]
  for μ in r
    @test 0.0 <= μ[1] <= 1.0
  end
end

@testset "Realisation indexing and iteration" begin
  p = ParamSpace((0.0,1.0))
  r = realisation(p;nparams=3)
  r1 = r[1:2]
  @test num_params(r1) == 2

  # iterate produces raw parameter vectors
  all_μ = [μ for μ in r]
  @test length(all_μ) == 3
end

@testset "TransientParamSpace and TransientRealisation" begin
  p = TransientParamSpace((0.0,1.0),[0.0,0.1,0.2,0.3])
  r = realisation(p;nparams=2)
  @test r isa TransientRealisation
  @test num_params(r) == 2
  @test num_times(r) == 3
  @test length(r) == 6   # nparams × ntimes

  @test get_initial_time(r) == 0.0
  @test get_final_time(r) == 0.3
end

@testset "TransientRealisation shift!" begin
  p = TransientParamSpace((0.0,1.0),[0.0,0.1,0.2,0.3])
  r = realisation(p;nparams=1)
  shift!(r,1.0)
  @test get_initial_time(r) ≈ 0.0   # t0 is fixed
  @test get_final_time(r) ≈ 1.3
end

@testset "parameterise steady" begin
  p = ParamSpace((0.0,1.0))
  r = realisation(p;nparams=3)
  f = parameterise(μ -> x -> μ[1]*x[1],r)
  @test f isa AbstractParamFunction
  @test length(f) == 3
  # each index returns a function
  g = f[1]
  @test g(0.5) isa Number
end

@testset "parameterise transient" begin
  p = TransientParamSpace((0.0,1.0),[0.0,0.1,0.2])
  r = realisation(p;nparams=2)
  f = parameterise((μ,t) -> x -> μ[1] + t,r)
  @test f isa AbstractParamFunction
  @test length(f) == 4   # 2 params × 2 times
end

# ─── ConsecutiveParamArray ────────────────────────────────────────────────────

@testset "ConsecutiveParamVector construction and access" begin
  l = 4
  n = 6
  data = rand(Float64,n,l)
  A = ConsecutiveParamArray(data)
  @test A isa ConsecutiveParamArray
  @test param_length(A) == l
  @test innersize(A) == (n,)

  # param_getindex returns a view into the data
  v = param_getindex(A,2)
  @test v == data[:,2]

  # setindex/getindex round-trip
  A[1] # diagonal access returns the first block
  @test A[1] == data[:,1]
end

@testset "ConsecutiveParamMatrix construction and access" begin
  l = 3
  m,n = 4,5
  data = rand(Float64,m,n,l)
  A = ConsecutiveParamArray(data)
  @test param_length(A) == l
  @test innersize(A) == (m,n)
  # diagonal access
  @test A[2,2] == data[:,:,2]
  # off-diagonal must be zero
  @test iszero(A[1,2])
end

@testset "ConsecutiveParamArray arithmetic" begin
  l = 5
  n = 8
  A = ConsecutiveParamArray(rand(Float64,n,l))
  B = ConsecutiveParamArray(rand(Float64,n,l))

  C = A + B
  @test get_all_data(C) ≈ get_all_data(A) + get_all_data(B)

  D = A * 2.0
  @test get_all_data(D) ≈ 2.0 .* get_all_data(A)

  E = copy(A)
  @test get_all_data(E) == get_all_data(A)
  @test E !== A
end

@testset "TrivialParamArray" begin
  v = [1.0,2.0,3.0]
  A = TrivialParamArray(v,5)
  @test param_length(A) == 5
  @test innersize(A) == (3,)
  # diagonal access (i,i) returns the underlying data
  @test A[2] == v   # A[i] where all indices equal → returns data
end

@testset "ParamArray constructor from vector-of-vectors" begin
  vecs = [rand(4) for _ in 1:6]
  A = ParamArray(vecs)
  @test param_length(A) == 6
  @test innersize(A) == (4,)
  for i in 1:6
    @test param_getindex(A,i) ≈ vecs[i]
  end
end

# ─── Snapshots ────────────────────────────────────────────────────────────────

@testset "SteadySnapshots (GenericSnapshots) construction and access" begin
  p = ParamSpace((0.0,1.0))
  r = realisation(p;nparams=4)
  ndofs = 10
  data = rand(Float64,ndofs,4)   # ndofs × nparams
  dm = VectorDofMap(ndofs)
  s = Snapshots(data,dm,r)
  @test s isa AbstractSnapshots
  @test num_params(s) == 4
  @test num_space_dofs(s) == ndofs

  # select a subset
  s2 = select_snapshots(s,1:2)
  @test num_params(s2) == 2
end

end # module
