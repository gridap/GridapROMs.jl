using Gridap
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamODEs
using LinearAlgebra
using Lux

println("=== NOMAD TRANSIENT TEST (1D GAUSSIAN PULSE) ===")

# ===================================================================
# 1. PHYSICS & FEM SETUP
# ===================================================================
L, nx = 3.0, 100 # Medium grid
t0, tf, dt = 0.0, 1.0, 0.05
c = 1.0
θ = 0.5

domain = (-L, L)
partition = (nx,)
model = CartesianDiscreteModel(domain, partition)
tdomain = t0:dt:tf

pdomain = (0.01, 0.1)
D_p = ParamSpace(pdomain)                 
D   = TransientParamSpace(pdomain, tdomain) 

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

p_train = realisation(D_p; nparams=20, sampling=:halton)
p_test  = realisation(D_p; nparams=5,  sampling=:halton)

σ_train = TransientRealisation(p_train, tdomain)
σ_test  = TransientRealisation(p_test, tdomain)

uh₀ₚ_train(σ) = interpolate_everywhere(u₀ₚ(σ), U(σ, t0))
uh₀ₚ_test(σ)  = interpolate_everywhere(u₀ₚ(σ), U(σ, t0))

s_train, _ = solution_snapshots(fe_solver, feop, σ_train, uh₀ₚ_train)
s_test, _  = solution_snapshots(fe_solver, feop, σ_test,  uh₀ₚ_test)

println("Train Snapshots: ", size(get_all_data(s_train)))

# ===================================================================
# 3. NOMAD SETUP (MANUAL CONFIGURATION)
# ===================================================================
m_sensors = 50
x_sensors = range(-L, L, length=m_sensors)

# Sampler mapping the scalar σ to the initial Gaussian curve sampled at the sensors
branch_sampler_func = (σ) -> [(1 / √(2 * π * σ[1])) * exp(-x^2 / (2 * σ[1])) for x in x_sensors]

# Manual definition of the network layers.
# Input dimension: 50 sensors + 1 spatial coordinate + 1 time coordinate = 52
model = NOMAD(
  layers     = (52, 64, 64, 32, 1),
  activation = tanh
)

strategy = NOMADStrategy(
  model          = model,
  epochs         = 1500, # Approx 1-2 minutes
  batch_size     = 4096, 
  step_x         = 2,    
  step_t         = 1,    
  branch_sampler = branch_sampler_func,
  lr_scheduler   = CosineAnnealing(lr_max=2f-3, lr_min=1f-5)
)

reduction = NOMADReduction(strategy)
solver = NeuralOpSolver(fe_solver, reduction)

# ===================================================================
# 4. TRAINING (OFFLINE PHASE)
# ===================================================================
println("\nTraining Transient NOMAD (Offline Phase)...")
neural_rb_op = reduced_operator(solver, feop, s_train)
println("Training completed!")

# ===================================================================
# 5. INFERENCE (ONLINE PHASE)
# ===================================================================
println("\nEvaluating Model on Unseen Transient Test Data...")
x_approx, stats = solve(solver, neural_rb_op, σ_test)

U_true = get_all_data(s_test)
U_pred = get_all_data(x_approx.fe_data)

n_samples = size(U_true, 2)
errors = zeros(n_samples)

for i in 1:n_samples
    true_slice = U_true[:, i, :]
    pred_slice = U_pred[:, i, :]
    
    err_norm  = norm(true_slice - pred_slice)
    true_norm = norm(true_slice)
    errors[i] = err_norm / true_norm
end

mean_rel_error = sum(errors) / length(errors) * 100

println("\n=== PERFORMANCE RESULTS ===")
println("Inference Time        : $(round(stats.time, digits=5)) s (for $n_samples samples)")
println("Mean Relative L2 Error: $(round(mean_rel_error, digits=3)) %")
println("===========================")