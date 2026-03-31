module ParamFESpacesTests

using Test
using LinearAlgebra
using Gridap
using Gridap.FESpaces
using Gridap.ReferenceFEs
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces

# ─── helpers ──────────────────────────────────────────────────────────────────

function make_model()
  CartesianDiscreteModel((0,1,0,1),(8,8))
end

function make_test_space(model)
  reffe = ReferenceFE(lagrangian,Float64,1)
  FESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
end

function make_realisation(n=4)
  p = ParamSpace((1.0,2.0,0.5,1.5))
  realisation(p;nparams=n)
end

# ─── TrivialParamFESpace ──────────────────────────────────────────────────────

@testset "TrivialParamFESpace wraps a plain FESpace" begin
  model = make_model()
  V = make_test_space(model)
  l = 5
  Vp = TrivialParamFESpace(V,l)
  @test Vp isa TrivialParamFESpace
  @test num_free_dofs(Vp) == num_free_dofs(V)
end

# ─── TrialParamFESpace ────────────────────────────────────────────────────────

@testset "TrialParamFESpace with parametric Dirichlet data" begin
  model = make_model()
  V = make_test_space(model)
  r = make_realisation(4)
  g = parameterise(μ -> x -> μ[1]*x[1] + μ[2]*x[2],r)
  U = TrialParamFESpace(V,g)
  @test U isa TrialParamFESpace
  @test num_free_dofs(U) == num_free_dofs(V)
end

@testset "HomogeneousTrialParamFESpace" begin
  model = make_model()
  V = make_test_space(model)
  l = 6
  U = HomogeneousTrialParamFESpace(V,l)
  @test U isa HomogeneousTrialParamFESpace
  @test num_free_dofs(U) == num_free_dofs(V)
end

# ─── SingleFieldParamFESpace ──────────────────────────────────────────────────

@testset "SingleFieldParamFESpace param_zero_free_values" begin
  model = make_model()
  V = make_test_space(model)
  r = make_realisation(4)
  U = TrialParamFESpace(V,r)
  z = param_zero_free_values(U)
  @test z isa AbstractParamArray
  @test param_length(z) == 4
  @test all(iszero,get_all_data(z))
end

# ─── MultiFieldParamFESpace ───────────────────────────────────────────────────

@testset "MultiFieldParamFESpace" begin
  model = make_model()
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},2)
  reffe_p = ReferenceFE(lagrangian,Float64,1)
  V_u = FESpace(model,reffe_u;conformity=:H1,dirichlet_tags="boundary")
  V_p = FESpace(model,reffe_p;conformity=:H1)
  l = 3
  U_u = HomogeneousTrialParamFESpace(V_u,l)
  U_p = HomogeneousTrialParamFESpace(V_p,l)
  VY = MultiFieldParamFESpace([U_u,U_p])
  @test VY isa MultiFieldParamFESpace
  @test num_fields(VY) == 2
end

end # module
