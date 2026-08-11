using Gridap
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using LinearAlgebra
using Lux

println("=== NOMAD ACCURACY TEST (CLASSIC 1D POISSON) ===")

# ===================================================================
# 1. PHYSICS & FEM SETUP
# ===================================================================
ν(μ) = x -> 1.0 + μ[1]^2
f(μ) = x -> 10.0

νₚ(μ) = parameterise(ν, μ)
fₚ(μ) = parameterise(f, μ)

domain = (0.0, 1.0)
partition = (100,) # Medium grid
model = CartesianDiscreteModel(domain, partition)

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V_std = TestFESpace(model, reffe; dirichlet_tags=["boundary"])

# Crucial step for ROMs: wrap the space in OrderedFESpace
V = OrderedFESpace(V_std)

u_bc(μ) = x -> 0.0
u_bcₚ(μ) = parameterise(u_bc, μ)
U = ParamTrialFESpace(V, u_bcₚ)

# Parameter Space
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
μ_train = realisation(D; nparams=50, sampling=:halton)
μ_test  = realisation(D; nparams=10, sampling=:halton)

s_train, _ = solution_snapshots(fe_solver, feop, μ_train)
s_test, _  = solution_snapshots(fe_solver, feop, μ_test)

println("Train Snapshots: ", size(get_all_data(s_train)))
println("Test  Snapshots: ", size(get_all_data(s_test)))

# ===================================================================
# 3. NOMAD SETUP (MANUAL CONFIGURATION)
# ===================================================================
# Manual definition of the network layers.
# Input dimension: 1 parameter + 1 spatial coordinate = 2
# Output dimension: 1 (scalar PDE solution)
model = NOMAD(
  layers = (2, 64, 64, 32, 1),
  activation = tanh
)

strategy = NOMADStrategy(
  model        = model,
  epochs       = 1500, # Enough epochs to see convergence in 1-2 minutes
  batch_size   = 1024, 
  lr_scheduler = CosineAnnealing(lr_max=2f-3, lr_min=1f-5)
)

reduction = NOMADReduction(strategy)
solver = NeuralOpSolver(fe_solver, reduction)

# ===================================================================
# 4. TRAINING (OFFLINE PHASE)
# ===================================================================
println("\nTraining Steady NOMAD (Offline Phase)...")
neural_rb_op = reduced_operator(solver, feop, s_train)
println("Training completed!")

# ===================================================================
# 5. INFERENCE & EVALUATION (ONLINE PHASE)
# ===================================================================
println("\nEvaluating Model on Unseen Test Data...")
x_approx, stats = solve(solver, neural_rb_op, μ_test)

U_true = get_all_data(s_test)
U_pred = get_all_data(x_approx.fe_data)

errors = zeros(size(U_true, 2))
for i in 1:size(U_true, 2)
    err_norm  = norm(U_true[:, i] - U_pred[:, i])
    true_norm = norm(U_true[:, i])
    errors[i] = err_norm / true_norm
end

mean_rel_error = sum(errors) / length(errors) * 100

println("\n=== PERFORMANCE RESULTS ===")
println("Inference Time        : $(round(stats.time, digits=5)) s (for $(size(U_true, 2)) samples)")
println("Mean Relative L2 Error: $(round(mean_rel_error, digits=3)) %")
println("===========================")