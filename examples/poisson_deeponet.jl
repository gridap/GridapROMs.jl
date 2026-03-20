# End-to-end PoC: Neural Operator ROM for Parametric Poisson Equation
#
# Problem: -∇·(κ(μ) ∇u) = f  on [0,1]²,  u = 0 on ∂Ω
#
# The diffusivity coefficient κ depends on parameters:
#   κ(x; μ) = μ₁ + μ₂ · sin(π·x₁) · sin(π·x₂)
#
# We train a DeepONet to learn the map μ → u_h(μ) (the DOF vector),
# then predict solutions for new parameters without running FEM.

module PoissonDeepONet

using Gridap
using GridapNeuralROMs
using LinearAlgebra
using Random
using Statistics

function main(;n_train=60,n_test=10,epochs=300,seed=42)
  rng = Random.MersenneTwister(seed)

  println("="^60)
  println("Neural Operator ROM for Parametric Poisson")
  println("="^60)

  # ── 1. Gridap FEM setup ──────────────────────────────────────────

  domain = (0,1,0,1)
  partition = (16,16)
  model = CartesianDiscreteModel(domain,partition)

  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
  U = TrialFESpace(V,0.0)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)

  N_dofs = num_free_dofs(V)
  println("\nMesh: $(partition[1])×$(partition[2]),  Free DOFs: $N_dofs")

  # ── 2. Parametric solver ─────────────────────────────────────────
  # Black-box function: μ → FEFunction
  # This is what makes neural operator ROMs non-intrusive —
  # we only need input/output pairs, not the PDE operators.

  function solve_poisson(μ)
    κ(x) = μ[1] + μ[2] * sin(π*x[1]) * sin(π*x[2])
    f(x) = 1.0
    a(u,v) = ∫( κ ⊙ (∇(u)⋅∇(v)) )dΩ
    l(v)   = ∫( f*v )dΩ
    op = AffineFEOperator(a,l,U,V)
    return Gridap.solve(op)
  end

  # ── 3. Collect snapshots ─────────────────────────────────────────

  param_bounds = [(0.1,5.0),(0.0,4.0)]  # μ₁ ∈ [0.1,5], μ₂ ∈ [0,4]

  println("\nGenerating $n_train training snapshots...")
  train_params = sample_parameters(param_bounds,n_train)
  t_snap = @elapsed train_data = collect_snapshots(
    solve_poisson,train_params;trial=U
  )
  println("  Snapshot generation: $(round(t_snap,digits=2))s")
  println("  Solution matrix: $(size(train_data.solutions))")
  println("  Coordinate matrix: $(size(train_data.coordinates))")

  # ── 4. Build and train DeepONet ──────────────────────────────────

  d_param = length(first(train_params))
  spatial_dim = size(train_data.coordinates,1)

  deeponet = build_deeponet(;
    param_dim=d_param,
    n_dofs=N_dofs,
    spatial_dim=spatial_dim,
    latent_dim=32,
    branch_width=64,
    trunk_width=64,
    n_branch_layers=3,
    n_trunk_layers=3,
  )

  println("\nTraining DeepONet (latent_dim=32, width=64, 3 layers)...")
  config = TrainingConfig(;epochs=epochs,lr=1e-3,batch_size=16,
                           patience=80,verbose=true)
  t_train = @elapsed result = train_operator(train_data,deeponet;config,rng)
  println("  Training time: $(round(t_train,digits=2))s")
  println("  Best epoch: $(result.best_epoch)")
  println("  Final val loss: $(round(result.val_losses[result.best_epoch],digits=6))")

  # ── 5. Evaluate on test parameters ──────────────────────────────

  println("\nEvaluating on $n_test test parameters...")
  test_params = sample_parameters(param_bounds,n_test)

  relative_errors = Float64[]
  speedups = Float64[]

  for (i,μ) in enumerate(test_params)
    # Ground truth via FEM
    t_fem = @elapsed uh_true = solve_poisson(μ)
    dofs_true = get_free_dof_values(uh_true)

    # Neural operator prediction
    t_rom = @elapsed uh_pred = reconstruct_fe_function(result,μ,U)
    dofs_pred = get_free_dof_values(uh_pred)

    # Relative L2 error on DOF vector
    err = norm(dofs_pred .- dofs_true) / norm(dofs_true)
    push!(relative_errors,err)
    push!(speedups,t_fem / max(t_rom,1e-10))
  end

  # ── 6. Report ───────────────────────────────────────────────────

  println("\n" * "="^60)
  println("RESULTS")
  println("="^60)
  println("  Mean relative L2 error: $(round(mean(relative_errors),digits=6))")
  println("  Max  relative L2 error: $(round(maximum(relative_errors),digits=6))")
  println("  Min  relative L2 error: $(round(minimum(relative_errors),digits=6))")
  println("  Mean speedup (FEM/ROM): $(round(mean(speedups),digits=1))×")
  println("  Training samples:       $n_train")
  println("  Test samples:           $n_test")
  println("  FEM DOFs:               $N_dofs")
  println("="^60)

  return (;relative_errors,speedups,result)
end

end # module

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
  PoissonDeepONet.main()
end
