using Test
using GridapROMs
using GridapROMs.NeuralOperatorROMs
using Gridap
using Lux
using Zygote
using Random
using LinearAlgebra
using Statistics

@testset "NeuralOperatorROMs" begin

  @testset "snapshot collection" begin
    # Minimal Gridap problem: Poisson on [0,1]²
    model = CartesianDiscreteModel((0,1,0,1),(8,8))
    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
    U = TrialFESpace(V,0.0)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2)

    function solve_poisson(μ)
      κ(x) = μ[1]
      a(u,v) = ∫(κ⊙(∇(u)⋅∇(v)))dΩ
      l(v) = ∫(1.0*v)dΩ
      op = AffineFEOperator(a,l,U,V)
      Gridap.solve(op)
    end

    params = [[1.0],[2.0],[3.0]]
    data = collect_snapshots(solve_poisson,params;trial=U)

    @test size(data.parameters) == (1,3)
    @test size(data.solutions,2) == 3
    @test size(data.solutions,1) == num_free_dofs(V)
    @test size(data.coordinates,1) == 2  # 2D

    # Different κ should give different solutions
    @test data.solutions[:,1] ≉ data.solutions[:,2]
    # Higher κ → smaller solution magnitude (for -∇·(κ∇u)=1)
    @test norm(data.solutions[:,3]) < norm(data.solutions[:,1])
  end

  @testset "coordinate extraction" begin
    model = CartesianDiscreteModel((0,1,0,1),(4,4))
    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
    U = TrialFESpace(V,0.0)

    coords = extract_coordinates(U)
    @test size(coords,1) == 2  # 2D
    @test all(0.0 .<= coords[1,:] .<= 1.0)
    @test all(0.0 .<= coords[2,:] .<= 1.0)
  end

  @testset "DeepONet construction and forward pass" begin
    rng = Random.MersenneTwister(123)

    net = build_deeponet(;
      param_dim=2,n_dofs=49,spatial_dim=2,
      latent_dim=16,branch_width=32,trunk_width=32,
      n_branch_layers=2,n_trunk_layers=2,
    )

    ps,st = Lux.setup(rng,net)

    # Mock data
    coords = Float32.(rand(rng,2,49))
    trunk_matrix = precompute_trunk_matrix(net,coords,ps,st)
    @test size(trunk_matrix) == (49,16)

    mu = Float32.(rand(rng,2,5))  # batch of 5
    u_hat,_ = Lux.apply(net,(mu,trunk_matrix),ps,st)
    @test size(u_hat) == (49,5)

    # Gradients should flow
    (loss,_),grads = Zygote.withgradient(ps) do p
      u,_ = Lux.apply(net,(mu,trunk_matrix),p,st)
      return sum(u.^2),nothing
    end
    @test loss > 0
    @test !isnothing(grads[1])
  end

  @testset "training smoke test" begin
    rng = Random.MersenneTwister(456)

    # Synthetic data: simple linear map μ → u = μ₁ * ones(10)
    M = 30
    params = randn(rng,1,M)
    sols = params[1,:]' .* ones(10)  # 10 × M

    data = SnapshotData(
      Float64.(params),
      Float64.(sols),
      Float64.(rand(rng,2,10))  # dummy coordinates
    )

    net = build_deeponet(;
      param_dim=1,n_dofs=10,spatial_dim=2,
      latent_dim=8,branch_width=16,trunk_width=16,
    )

    config = TrainingConfig(;epochs=100,lr=1e-3,batch_size=8,
                             patience=100,verbose=false)
    result = train_operator(data,net;config,rng)

    # Loss should decrease
    @test result.train_losses[end] < result.train_losses[1]
    @test length(result.val_losses) > 0

    # Prediction should be in the right ballpark
    μ_test = [2.0]
    pred = evaluate_rom(result,μ_test)
    @test length(pred) == 10
    expected = 2.0 * ones(10)
    @test norm(pred .- expected) / norm(expected) < 0.5  # within 50% for smoke test
  end

  @testset "FEFunction reconstruction" begin
    model = CartesianDiscreteModel((0,1,0,1),(4,4))
    reffe = ReferenceFE(lagrangian,Float64,1)
    V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
    U = TrialFESpace(V,0.0)

    N = num_free_dofs(V)
    fake_dofs = rand(N)

    # Verify we can construct FEFunction from DOF vector
    uh = FEFunction(U,fake_dofs)
    recovered = get_free_dof_values(uh)
    @test recovered ≈ fake_dofs
  end

  @testset "parameter sampling" begin
    bounds = [(0.0,1.0),(10.0,20.0),(-5.0,5.0)]
    samples = sample_parameters(bounds,100)
    @test length(samples) == 100
    @test all(length(s) == 3 for s in samples)
    for s in samples
      @test 0.0 <= s[1] <= 1.0
      @test 10.0 <= s[2] <= 20.0
      @test -5.0 <= s[3] <= 5.0
    end
  end

end # top-level testset
