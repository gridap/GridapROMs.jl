using Gridap
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamODEs
using LinearAlgebra
using Lux

println("=== DEEPONET TRANSIENT TEST (1D GAUSSIAN PULSE) ===")

# ===================================================================
# 1. PHYSICS & FEM SETUP
# ===================================================================
L, nx = 3.0, 600
t0, tf, dt = 0.0, 1.0, 0.02
c = 1.0
θ = 0.5

domain = (-L, L)
partition = (nx,)
model = CartesianDiscreteModel(domain, partition)
tdomain = t0:dt:tf

# Spazi Parametrici
pdomain = (0.01, 0.1)
D_p = ParamSpace(pdomain)                 # Spazio puramente parametrico per campionare
D   = TransientParamSpace(pdomain, tdomain) # Spazio misto richiesto dall'operatore

c_vec = VectorValue(c)
u₀(σ) = x -> (1 / √(2 * π * σ[1])) * exp(-x[1]^2 / (2 * σ[1]))
u(σ, t) = x -> (1 / √(2 * π * σ[1])) * exp(-(x[1] - c * t)^2 / (2 * σ[1]))

u₀ₚ(σ) = parameterise(u₀, σ)
uₚₜ(σ, t) = parameterise(u, σ, t)

order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V_std = TestFESpace(model, reffe)
V = OrderedFESpace(V_std)
U = TransientTrialParamFESpace(V, uₚₜ)

degree = 2 * order
τₕ = Triangulation(model)
dΩ = Measure(τₕ, degree)

m(σ, t, du, v) = ∫(v * du)dΩ
a(σ, t, u, v) = ∫(v * (c_vec ⋅ ∇(u)))dΩ
r(σ, t, u, v) = m(σ, t, ∂t(u), v) + a(σ, t, u, v)

feop = TransientLinearParamOperator(r, (a, m), D, U, V)
fe_solver = ThetaMethod(LUSolver(), dt, θ)

# ===================================================================
# 2. DATA GENERATION (TRAIN & TEST SPLIT)
# ===================================================================
println("\nGenerating Training and Testing Datasets...")

# Campioniamo i parametri scalari
p_train = realisation(D_p; nparams=100, sampling=:halton)
p_test  = realisation(D_p; nparams=10, sampling=:halton)

# Uniamo parametri e tempo nelle realizzazioni Transient
σ_train = TransientRealisation(p_train, tdomain)
σ_test  = TransientRealisation(p_test, tdomain)

# Funzioni per la condizione iniziale
uh₀ₚ_train(σ) = interpolate_everywhere(u₀ₚ(σ), U(σ, t0))
uh₀ₚ_test(σ)  = interpolate_everywhere(u₀ₚ(σ), U(σ, t0))

s_train, _ = solution_snapshots(fe_solver, feop, σ_train, uh₀ₚ_train)
s_test, _  = solution_snapshots(fe_solver, feop, σ_test, uh₀ₚ_test)

println("Train Snapshots: ", size(get_all_data(s_train)))
println("Test  Snapshots: ", size(get_all_data(s_test)))

# ===================================================================
# 3. DEEPONET SETUP (OPERATOR LEARNING)
# ===================================================================
m_sensors = 100
x_sensors = range(-L, L, length=m_sensors)

# IL SAMPLER: Mappa lo scalare σ nella campionatura della curva Gaussiana iniziale
branch_sampler_func = (σ) -> [(1 / √(2 * π * σ[1])) * exp(-x^2 / (2 * σ[1])) for x in x_sensors]

model = DeepONet(
  branch_layers = (m_sensors, 64, 64, 32), # 100 Sensori (Input del Sampler)
  trunk_layers  = (2, 64, 64, 32),         # 2 Input (Coordinata 'x' + Tempo 't')
  activation    = tanh
)

strategy = DeepONetStrategy(
  model          = model,
  epochs         = 10000,
  batch_size     = 32,
  step_x         = 2,  # Sub-campionamento spaziale
  step_t         = 2,  # Sub-campionamento temporale
  branch_sampler = branch_sampler_func,
  lr_scheduler   = ReduceLROnPlateau(patience=500, factor=0.5f0, start_lr=1e-3)
)

reduction = DeepONetReduction(strategy)
solver = NeuralOpSolver(fe_solver, reduction)

# ===================================================================
# 4. TRAINING (OFFLINE)
# ===================================================================
println("\nTraining Transient DeepONet (Offline Phase)...")
neural_rb_op = reduced_operator(solver, feop, s_train)
println("Training completed!")

# ===================================================================
# 5. INFERENCE (ONLINE)
# ===================================================================
println("\nEvaluating Model on Unseen Transient Test Data...")
x_approx, stats = solve(solver, neural_rb_op, σ_test)

# Estrazione dei dati
# U_true e U_pred avranno dimensione (N_dofs, N_samples, N_time)
U_true = get_all_data(s_test)
U_pred = get_all_data(x_approx.fe_data)

n_samples = size(U_true, 2)
errors = zeros(n_samples)

# Calcolo dell'Errore Relativo L2 per ogni campione (su tutto lo spazio-tempo)
for i in 1:n_samples
    # Estraiamo la fetta (N_dofs, N_time) per il campione i-esimo
    true_slice = U_true[:, i, :]
    pred_slice = U_pred[:, i, :]
    
    # norm() di default su matrici fa la norma di Frobenius (equivalente L2 "appiattita")
    err_norm  = norm(true_slice - pred_slice)
    true_norm = norm(true_slice)
    errors[i] = err_norm / true_norm
end

mean_rel_error = sum(errors) / length(errors) * 100
max_rel_error  = maximum(errors) * 100

println("\n=== PERFORMANCE RESULTS ===")
println("Inference Time        : $(round(stats.time, digits=5)) s (for $n_samples samples)")
println("Mean Relative L2 Error: $(round(mean_rel_error, digits=3)) %")
println("Max  Relative L2 Error: $(round(max_rel_error, digits=3)) %")
println("===========================")