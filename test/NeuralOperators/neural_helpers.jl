module NeuralHelpersTests

using Test
using Gridap
using GridapROMs
using GridapROMs.RBSteady

@testset "Formatting and Utils" begin
  # format_eta
  @test RBSteady.format_eta(45) == "00:45"
  @test RBSteady.format_eta(125) == "02:05"
  @test RBSteady.format_eta(3665) == "01:01:05"
  
  # batch_size resolution
  @test RBSteady.resolve_batch_size(0, 100) == 100
  @test RBSteady.resolve_batch_size(-5, 100) == 100
  @test RBSteady.resolve_batch_size(32, 100) == 32
end

@testset "Z-Score stats computation" begin
  data = Float32[1 2 3; 4 5 6] # 2 features, 3 samples
  stats = RBSteady.compute_zscore_stats(data)
  
  @test size(stats.μ) == (2, 1)
  @test size(stats.σ) == (2, 1)
  @test stats.μ[1] ≈ 2.0f0
  
  # Array of ones => dev = 0 converted to 1
  data_const = ones(Float32, 2, 10)
  stats_const = RBSteady.compute_zscore_stats(data_const)
  @test all(stats_const.μ .≈ 1.0f0)
  @test all(stats_const.σ .== 1.0f0) # Forced to 1.0
end

@testset "Learning Rate Schedulers" begin
  # CosineAnnealing
  ca = CosineAnnealing(lr_max=1.0f0, lr_min=0.0f0)
  @test get_initial_lr(ca) == 1.0f0
  
  opt_state = Optimisers.setup(Adam(1.0f0), [1.0f0])
  # Half training (50/100), cos(pi/2) = 0, lr = 0.5
  RBSteady.step_scheduler!(ca, opt_state, 50, 100, 1.0f0)
  @test opt_state.rule.eta ≈ 0.5f0
  
  # ReduceLROnPlateau
  plat = ReduceLROnPlateau(patience=2, factor=0.5f0, min_lr=0.1f0, start_lr=1.0f0)
  @test get_initial_lr(plat) == 1.0f0
  
  opt_state_plat = Optimisers.setup(Adam(1.0f0), [1.0f0])
  
  # Epoch 1: improvement
  RBSteady.step_scheduler!(plat, opt_state_plat, 1, 100, 0.5f0)
  @test plat.wait == 0
  
  # Epoch 2: No improvement
  RBSteady.step_scheduler!(plat, opt_state_plat, 2, 100, 0.6f0)
  @test plat.wait == 1
  @test opt_state_plat.rule.eta == 1.0f0 # No drop yet
  
  # Epoch 3: Patience limit reached, drop lr by half
  RBSteady.step_scheduler!(plat, opt_state_plat, 3, 100, 0.6f0)
  @test opt_state_plat.rule.eta ≈ 0.5f0
  @test plat.wait == 0 # Patience resetted 
end

@testset "Model Resolution and Arch Building" begin
  # AutoDeepONet
  config_don = AutoDeepONet(width=32, depth=2)
  model_don = RBSteady.resolve_model(config_don, 5, 2)
  @test model_don.branch_layers == (5, 32, 32, 32)
  @test model_don.trunk_layers == (2, 32, 32, 32)
  
  # AutoNOMAD
  config_nomad = AutoNOMAD(width=64, depth=3)
  model_nomad = RBSteady.resolve_model(config_nomad, 10, 3)
  @test model_nomad.layers == (13, 64, 64, 64, 1)

  # build_lux_chain
  chain = RBSteady.build_lux_chain((2, 10, 10, 1), tanh)
  @test chain isa Lux.Chain
  @test length(chain.layers) == 3
  
  @test chain.layers[1].out_dims == 10
  @test chain.layers[2].out_dims == 10
  @test chain.layers[3].out_dims == 1
  @test chain.layers[1].activation === tanh
  @test chain.layers[2].activation === tanh
  @test chain.layers[3].activation === identity
  
  # build_model (NeuralOperators.jl wrappers)
  lux_don = RBSteady.build_model(model_don)
  @test lux_don isa NeuralOperators.DeepONet
  
  lux_nomad = RBSteady.build_model(model_nomad)
  @test lux_nomad isa NeuralOperators.NOMAD
end

@testset "Coordinate Extraction with OrderedFESpace" begin
  # Small 1D mesh: domain (0, 1) with 2 elements -> Nodes: 0.0, 0.5, 1.0
  model = CartesianDiscreteModel((0.0, 1.0), (2,))
  reffe = ReferenceFE(lagrangian, Float64, 1)
  
  # Space with no boundary conditions -> 3 Free DoFs
  V_std = FESpace(model, reffe)
  V = OrderedFESpace(V_std)
  
  coords = RBSteady.get_coords_with_order(V)
  
  @test size(coords) == (1, 3) # (D_phys, N_dofs)
  
  expected_coords = Float32[0.0 0.5 1.0]
  
  @test sort(vec(coords)) ≈ sort(vec(expected_coords))
end

end # module