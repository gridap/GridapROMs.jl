using Gridap
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using LinearAlgebra
using Lux

println("=== DEEPONET ACCURACY TEST (CLASSIC 1D POISSON) ===")

# ===================================================================
# 1. PHYSICS & FEM SETUP
# ===================================================================
ν(μ) = x -> 1.0 + μ[1]^2
f(μ) = x -> 10.0

νₚ(μ) = parameterise(ν, μ)
fₚ(μ) = parameterise(f, μ)

domain = (0.0, 1.0)
partition = (200,)
model = CartesianDiscreteModel(domain, partition)

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V_std = TestFESpace(model, reffe; dirichlet_tags=["boundary"])

V = OrderedFESpace(V_std)

u_bc(μ) = x -> 0.0
u_bcₚ(μ) = parameterise(u_bc, μ)
U = ParamTrialFESpace(V, u_bcₚ)

# Parameter Space: μ ∈ [0.1, 5.0]
D = ParamSpace((0.1, 5.0))

degree = 2
trian = Triangulation(model)
dΩ = Measure(trian, degree)

a(μ, u, v, dΩ) = ∫(νₚ(μ) * ∇(v) ⋅ ∇(u))dΩ
r(μ, u, v, dΩ) = a(μ, u, v, dΩ) - ∫(fₚ(μ) * v)dΩ

domains = FEDomains((trian,), (trian,))
feop = LinearParamOperator(r, a, D, U, V, domains)

fe_solver = LUSolver()

# ===================================================================
# 2. DATA GENERATION (TRAIN & TEST SPLIT)
# ===================================================================
println("\nGenerating Training and Testing Datasets...")
μ_train = realisation(D; nparams=200, sampling=:halton)
μ_test  = realisation(D; nparams=20,  sampling=:halton)

s_train, _ = solution_snapshots(fe_solver, feop, μ_train)
s_test, _  = solution_snapshots(fe_solver, feop, μ_test)

println("Train Snapshots: ", size(get_all_data(s_train)))
println("Test  Snapshots: ", size(get_all_data(s_test)))

# ===================================================================
# 3. DEEPONET SETUP
# ===================================================================
model = DeepONet(
  branch_layers = (1, 64, 64, 32),
  trunk_layers  = (1, 64, 64, 32),
  activation    = Lux.gelu
)

strategy = DeepONetStrategy(
  model      = model,
  epochs     = 20000,
  batch_size = 32,
  lr         = 1e-4,
)

reduction = DeepONetReduction(strategy)
solver = NeuralOpSolver(fe_solver, reduction)

# ===================================================================
# 4. TRAINING (OFFLINE)
# ===================================================================
println("\nTraining DeepONet (Offline Phase)...")
neural_rb_op = reduced_operator(solver, feop, s_train)
println("Training completed!")

# ===================================================================
# 5. INFERENCE & EVALUATION (ONLINE)
# ===================================================================
println("\nEvaluating Model on Unseen Test Data...")

x_approx, stats = solve(solver, neural_rb_op, μ_test)

# Estrazione dei dati
U_true = get_all_data(s_test)
U_pred = get_all_data(x_approx.fe_data)

# Calcolo dell'Errore Relativo L2 per ogni campione
errors = zeros(size(U_true, 2))
for i in 1:size(U_true, 2)
    err_norm  = norm(U_true[:, i] - U_pred[:, i])
    true_norm = norm(U_true[:, i])
    errors[i] = err_norm / true_norm
end

mean_rel_error = sum(errors) / length(errors) * 100
max_rel_error  = maximum(errors) * 100

println("\n=== PERFORMANCE RESULTS ===")
println("Inference Time        : $(round(stats.time, digits=5)) s (for $(size(U_true, 2)) samples)")
println("Mean Relative L2 Error: $(round(mean_rel_error, digits=3)) %")
println("Max  Relative L2 Error: $(round(max_rel_error, digits=3)) %")
println("===========================")