module ParamDataStructuresTests

using Test
using LinearAlgebra
using Gridap
using Gridap.Arrays
using Gridap.Fields
import Gridap.Fields: BroadcastingFieldOpMap
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

@testset "Snapshots construction and access" begin
  nparams = 4
  p = ParamSpace((0.0,1.0))
  r = realisation(p;nparams)
  ndofs = 10
  pdata = ParamArray([rand(ndofs) for _ in 1:nparams])
  dm = VectorDofMap(ndofs)
  s = Snapshots(pdata,dm,r)
  @test s isa AbstractSnapshots
  @test num_params(s) == nparams
  @test num_space_dofs(s) == ndofs

  # select a subset
  s2 = select_snapshots(s,1:2)
  @test num_params(s2) == 2
end


# ─── GenericParamBlock ────────────────────────────────────────────────────────

@testset "GenericParamBlock basic interface" begin
  n = 4
  L = 3
  vecs = [rand(n) for _ in 1:L]
  b = GenericParamBlock(copy(vecs))

  @test b isa GenericParamBlock
  @test param_length(b) == L
  for i in 1:L
    @test param_getindex(b,i) == vecs[i]
  end
  @test testitem(b) == vecs[1]

  b2 = copy(b)
  @test b2 ≈ b
  @test b2 == b
  @test b2 !== b

  # setindex! mutates in place
  new_v = rand(n)
  param_setindex!(b2,new_v,1)
  @test param_getindex(b2,1) == new_v
  @test b2 != b   # b is unchanged

  # copyto!
  b3 = copy(b)
  src = GenericParamBlock([rand(n) for _ in 1:L])
  copyto!(b3,src)
  @test b3 ≈ src
end

@testset "GenericParamBlock BroadcastingFieldOpMap" begin
  n = 5
  L = 3
  va = [rand(n) for _ in 1:L]
  vb = [rand(n) for _ in 1:L]
  a = GenericParamBlock(copy(va))
  b = GenericParamBlock(copy(vb))

  # (PB, PB) element-wise
  k = BroadcastingFieldOpMap(+)
  cache = return_cache(k,a,b)
  c = evaluate!(cache,k,a,b)
  @test c isa GenericParamBlock
  @test param_length(c) == L
  for i in 1:L
    @test param_getindex(c,i) ≈ va[i] .+ vb[i]
  end

  # (PB, AbstractArray) broadcast with fixed array
  arr = rand(n)
  cache2 = return_cache(k,a,arr)
  d = evaluate!(cache2,k,a,arr)
  @test d isa GenericParamBlock
  for i in 1:L
    @test param_getindex(d,i) ≈ va[i] .+ arr
  end
end

@testset "GenericParamBlock arithmetic" begin
  n = 4
  L = 3
  vecs = [rand(n) for _ in 1:L]
  a = GenericParamBlock(copy(vecs))

  # scalar multiplication
  c = 3.0 * a
  @test c isa GenericParamBlock
  for i in 1:L
    @test param_getindex(c,i) ≈ 3.0 .* vecs[i]
  end

  # matrix-vector product: mul!(c, A, b, α, β)
  m = 6
  A = rand(m,n)
  c_out = GenericParamBlock([zeros(m) for _ in 1:L])
  mul!(c_out,A,a,1.0,0.0)
  for i in 1:L
    @test param_getindex(c_out,i) ≈ A * vecs[i]
  end

  # rmul!
  expected = [2.0 .* v for v in vecs]
  b2 = GenericParamBlock([copy(v) for v in vecs])
  rmul!(b2,2.0)
  for i in 1:L
    @test param_getindex(b2,i) ≈ expected[i]
  end
end

# ─── TrivialParamBlock ────────────────────────────────────────────────────────

@testset "TrivialParamBlock basic interface" begin
  v = rand(5)
  L = 4
  b = TrivialParamBlock(copy(v),L)

  @test b isa TrivialParamBlock
  @test param_length(b) == L
  for i in 1:L
    @test param_getindex(b,i) == v   # same value every time
  end
  @test testitem(b) == v

  b2 = copy(b)
  @test b2 ≈ b
  @test b2 == b
end

end
