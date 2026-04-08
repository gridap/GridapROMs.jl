module UtilsTests

using Test
using LinearAlgebra
using Gridap
using Gridap.CellData
using GridapROMs
using GridapROMs.Utils

# ─── PerformanceTrackers ──────────────────────────────────────────────────────

@testset "CostTracker construction" begin
  t = CostTracker()
  @test t isa CostTracker
  @test Utils.get_time(t) == 0.0
  @test Utils.get_nallocs(t) == 0.0
  @test Utils.get_nruns(t) == 1

  t2 = CostTracker(;name="foo",time=1.5,nallocs=10.0,nruns=3)
  @test Utils.get_name(t2) == "foo"
  @test Utils.get_time(t2) == 1.5
  @test Utils.get_nruns(t2) == 3
end

@testset "CostTracker from NamedTuple" begin
  stats = (time=2.0,bytes=Int(1e6))
  t = CostTracker(stats;nruns=2,name="bar")
  @test Utils.get_time(t) == 2.0
  @test Utils.get_nallocs(t) ≈ 1.0   # 1e6 bytes → 1.0 Mb
  @test Utils.get_nruns(t) == 2
  @test Utils.get_name(t) == "bar"
end

@testset "reset_tracker! and update_tracker!" begin
  t = CostTracker(;name="x",time=5.0,nallocs=2.0,nruns=1)
  reset_tracker!(t)
  @test Utils.get_time(t) == 0.0
  @test Utils.get_nallocs(t) == 0.0
  @test Utils.get_nruns(t) == 0

  stats = (time=3.0,bytes=Int(2e6))
  update_tracker!(t,stats)
  @test Utils.get_time(t) == 3.0
  @test Utils.get_nallocs(t) ≈ 2.0
end

@testset "compute_speedup" begin
  t1 = CostTracker(;name="hi-fi",time=10.0,nallocs=100.0,nruns=1)
  t2 = CostTracker(;name="rom",time=1.0,nallocs=10.0,nruns=1)
  su = compute_speedup(t1,t2)
  @test su isa Speedup
  @test Utils.get_speedup_time(su) ≈ 10.0
  @test Utils.get_speedup_memory(su) ≈ 10.0
end

@testset "compute_error" begin
  # exact solution → zero error
  sol = rand(Float64,20,5)
  err = compute_error(sol,sol)
  @test err ≈ 0.0 atol=1e-12

  # known relative error: all columns scaled by factor 2
  sol_approx = 2 .* sol
  err2 = compute_error(sol,sol_approx)
  @test err2 > 0.0
end

@testset "compute_relative_error" begin
  v = [3.0,4.0]                 # norm = 5
  v_approx = [3.0,3.0]          # diff = [0,1], norm = 1
  rel = compute_relative_error(v,v_approx)
  @test rel ≈ 1.0/5.0
end

@testset "induced_norm with and without norm matrix" begin
  v = [1.0,0.0,0.0]
  @test induced_norm(v) ≈ 1.0

  M = 4.0*I(3)                  # energy norm squares the norm by 4
  @test induced_norm(v,M) ≈ 2.0 # sqrt(v'*4I*v) = sqrt(4) = 2
end

# ─── Contributions ────────────────────────────────────────────────────────────

@testset "Contribution construction and access" begin
  domain = (0,1,0,1)
  model = CartesianDiscreteModel(domain,(4,4))
  Ω = Triangulation(model)

  val = [1.0,2.0,3.0]
  c = Contribution((val,),(Ω,))
  @test length(c) == 1
  @test c[1] === val

  # contribution do-block
  c2 = contribution((Ω,)) do trian
    [4.0,5.0]
  end
  @test c2[1] == [4.0,5.0]
end

@testset "ArrayContribution sum" begin
  domain = (0,1,0,1)
  model = CartesianDiscreteModel(domain,(4,4))
  Ω = Triangulation(model)
  Γ = BoundaryTriangulation(model)

  c = Contribution(([1.0,2.0],[3.0,4.0]),(Ω,Γ))
  @test sum(c) == [1.0,2.0] + [3.0,4.0]
end

end # module
