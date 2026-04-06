module ExtensionsTests

using Test
using LinearAlgebra
using Gridap
using Gridap.Arrays
using GridapROMs
using GridapROMs.Extensions

# ─── PosZeroNegReindex ────────────────────────────────────────────────────────

@testset "PosZeroNegReindex basic evaluation" begin
  pos = [10.0,20.0,30.0]
  neg = [100.0,200.0]
  k = PosZeroNegReindex(pos,neg)

  # positive index → from pos
  @test evaluate(k,1) == 10.0
  @test evaluate(k,3) == 30.0
  # negative index → from neg
  @test evaluate(k,-1) == 100.0
  @test evaluate(k,-2) == 200.0
  # zero → zero value
  @test evaluate(k,0) == 0.0
end

@testset "PosZeroNegReindex cache-based evaluation" begin
  pos = [1.0,2.0,4.0]
  neg = [8.0,16.0]
  k = PosZeroNegReindex(pos,neg)
  cache = return_cache(k,1)
  @test evaluate!(cache,k,2) == 2.0
  @test evaluate!(cache,k,-1) == 8.0
  @test evaluate!(cache,k,0) == 0.0
end

@testset "PosZeroNegReindex type consistency" begin
  pos = Float64[1.0,2.0]
  neg = Float64[3.0,4.0]
  k = PosZeroNegReindex(pos,neg)
  @test eltype(pos) == eltype(neg)
  rv = return_value(k,1)
  @test rv isa Float64
end

end # module
